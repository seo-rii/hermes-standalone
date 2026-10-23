"""Fetch and inspect the explicitly pinned pure Python build inputs.

This module never imports downloaded code. Fetching is a separate command from
building, so an offline build either uses verified inputs or fails.
"""

from __future__ import annotations

import argparse
import base64
import csv
import email.parser
import email.policy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import tarfile
import tempfile
import unicodedata
import urllib.parse
import urllib.request
import zipfile


DEFAULT_MANIFEST = Path(__file__).with_name("dependencies.json")
DEFAULT_CACHE = Path(__file__).parent / ".cache" / "downloads"
NATIVE_SUFFIXES = {
    ".so", ".pyd", ".dll", ".dylib", ".exe", ".node", ".wasm", ".a", ".lib",
    ".o", ".obj", ".pyc", ".pyo", ".class", ".jar",
}
NATIVE_MAGIC = (
    b"\x7fELF", b"MZ", b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca",
    b"\x00asm", b"!<arch>\n",
)
MAX_WHEEL_SIZE = 32 * 1024 * 1024
MAX_MEMBER_SIZE = 32 * 1024 * 1024
MAX_EXPANDED_WHEEL_SIZE = 128 * 1024 * 1024
RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


class VendorError(ValueError):
    """A dependency input failed validation."""


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def validate_member_path(name: str) -> str:
    """Return an unchanged, portable relative POSIX file path."""
    if (
        not isinstance(name, str)
        or not name
        or name.startswith("/")
        or "\\" in name
        or ":" in name
        or unicodedata.normalize("NFC", name) != name
    ):
        raise VendorError(f"Unsafe archive path: {name!r}")
    for part in name.split("/"):
        if (
            part in ("", ".", "..")
            or part[-1:] in (".", " ")
            or any(ord(character) < 32 or ord(character) == 127 for character in part)
            or part.split(".", 1)[0].casefold() in RESERVED_NAMES
        ):
            raise VendorError(f"Unsafe archive path: {name!r}")
    return name


class ArchivePaths:
    """Track portable file/directory identities across one combined archive."""

    def __init__(self):
        self.members: set[str] = set()
        self.files: set[str] = set()
        self.prefixes: dict[str, str] = {}

    def add(self, path: str, *, directory: bool = False) -> None:
        validate_member_path(path)
        if path in self.members:
            raise VendorError(f"Duplicate archive path: {path}")
        parts = path.split("/")
        prefixes = ["/".join(parts[:end]) for end in range(1, len(parts) + 1)]
        for prefix in prefixes:
            previous = self.prefixes.get(prefix.casefold())
            if previous is not None and previous != prefix:
                raise VendorError(f"Case-colliding archive paths: {previous}, {prefix}")
        if any(prefix in self.files for prefix in prefixes[:-1]):
            raise VendorError(f"Archive file is also a parent directory: {path}")
        if not directory and path.casefold() in self.prefixes:
            raise VendorError(f"Archive file is also a parent directory: {path}")
        self.members.add(path)
        if not directory:
            self.files.add(path)
        self.prefixes.update((prefix.casefold(), prefix) for prefix in prefixes)


def reject_native(path: str, content: bytes) -> None:
    """Reject compiled executables by both filename and recognizable magic."""
    lower = path.lower()
    if (
        Path(lower).suffix in NATIVE_SUFFIXES
        or ".so." in Path(lower).name
        or content.startswith(NATIVE_MAGIC)
    ):
        raise VendorError(f"Native or compiled payload is forbidden: {path}")


def _validate_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "files.pythonhosted.org"
        or parsed.port not in (None, 443)
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise VendorError("Dependency archive URLs must use HTTPS on files.pythonhosted.org")


def _validate_package(package: dict) -> None:
    if not isinstance(package, dict):
        raise VendorError("Each dependency entry must be an object")
    for key in ("name", "version", "filename", "url", "sha256"):
        if not isinstance(package.get(key), str) or not package[key]:
            raise VendorError(f"Dependency entry needs a non-empty {key}")
    name = package["name"]
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise VendorError(f"Dependency name must be normalized: {name!r}")
    if package.get("pin_source", "upstream-lock") not in {"upstream-lock", "supplemental-lock"}:
        raise VendorError(f"Invalid dependency pin source: {name}")
    filename = package["filename"]
    validate_member_path(filename)
    kind = package.get("kind", "wheel")
    if kind not in {"wheel", "sdist"}:
        raise VendorError(f"Unsupported dependency archive kind: {kind!r}")
    suffix = "-none-any.whl" if kind == "wheel" else ".tar.gz"
    if "/" in filename or not filename.endswith(suffix):
        raise VendorError(f"Unsupported {kind} filename: {filename}")
    if kind == "sdist":
        _validate_source_layout(package.get("source_layout"))
    _validate_url(package["url"])
    if urllib.parse.unquote(urllib.parse.urlsplit(package["url"]).path.rsplit("/", 1)[-1]) != filename:
        raise VendorError(f"URL does not match the pinned archive filename: {name}")
    if not re.fullmatch(r"[0-9a-f]{64}", package["sha256"]):
        raise VendorError(f"Invalid pinned SHA-256: {name}")
    if type(package.get("size")) is not int or not 0 < package["size"] <= MAX_WHEEL_SIZE:
        raise VendorError(f"Invalid pinned archive size: {name}")
    dependencies = package.get("dependencies", [])
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) or item != canonical_name(item)
        for item in dependencies
    ):
        raise VendorError(f"Invalid dependency edges: {name}")
    notices = package.get("additional_licenses", [])
    if not isinstance(notices, list):
        raise VendorError(f"Additional license notices must be a list: {name}")
    for notice in notices:
        if not isinstance(notice, dict) or set(notice) != {"filename", "text", "sha256", "source_url"}:
            raise VendorError(f"Additional license notice is incomplete: {name}")
        filename = validate_member_path(notice["filename"])
        if "/" in filename or not filename.lower().startswith(("license", "licence", "copying", "notice")):
            raise VendorError(f"Invalid additional license filename: {filename}")
        if not isinstance(notice["text"], str) or not notice["text"].strip():
            raise VendorError(f"Additional license notice is empty: {name}")
        if hashlib.sha256(notice["text"].encode("utf-8")).hexdigest() != notice["sha256"]:
            raise VendorError(f"Additional license notice SHA-256 differs: {name}")
        if not isinstance(notice["source_url"], str) or re.fullmatch(
            r"https://raw\.githubusercontent\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[0-9a-f]{40}/[^?#]+",
            notice["source_url"],
        ) is None:
            raise VendorError(f"Additional license notice needs immutable upstream provenance: {name}")
    _validate_patches(package)


def _validate_patches(package: dict) -> None:
    patches = package.get("source_patches", [])
    if not isinstance(patches, list):
        raise VendorError("Reviewed source patches must be a list")
    paths = ArchivePaths()
    for patch in patches:
        if not isinstance(patch, dict) or set(patch) != {"path", "original_sha256", "patched_sha256", "replacements"}:
            raise VendorError("Reviewed source patch is incomplete")
        paths.add(patch["path"])
        if not patch["path"].endswith(".py") or any(part.endswith(".dist-info") for part in patch["path"].split("/")):
            raise VendorError("Reviewed source patches must target Python source")
        for key in ("original_sha256", "patched_sha256"):
            if not isinstance(patch[key], str) or re.fullmatch(r"[0-9a-f]{64}", patch[key]) is None:
                raise VendorError("Reviewed source patch requires both source hashes")
        replacements = patch["replacements"]
        if not isinstance(replacements, list) or not replacements:
            raise VendorError("Reviewed source patch requires exact replacement blocks")
        for replacement in replacements:
            if (not isinstance(replacement, dict) or set(replacement) != {"old", "new"}
                    or not isinstance(replacement["old"], str) or not replacement["old"]
                    or not isinstance(replacement["new"], str)):
                raise VendorError("Reviewed source patch contains an invalid replacement block")
    replacements = package.get("replaced_dependencies", {})
    if not isinstance(replacements, dict):
        raise VendorError("Dependency replacements must be a mapping")
    for dependency, replacement in replacements.items():
        if dependency not in package.get("dependencies", []) or not patches:
            raise VendorError("Dependency replacement needs its upstream edge and reviewed source patch")
        if (not isinstance(replacement, dict) or set(replacement) != {"module", "reason"}
                or not isinstance(replacement["module"], str)
                or re.fullmatch(r"standalone(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+", replacement["module"]) is None
                or not isinstance(replacement["reason"], str) or not replacement["reason"].strip()):
            raise VendorError("Dependency replacement requires an explicit standalone module and reason")


def _validate_source_layout(layout: object) -> None:
    """Source archives are data inputs; only explicitly reviewed files are copied."""
    required = {"root", "files", "metadata", "licenses"}
    if (
        not isinstance(layout, dict) or not required.issubset(layout)
        or set(layout) - required - {"excluded_build_bytecode", "excluded_native_payloads"}
    ):
        raise VendorError("A source distribution requires an explicit reviewed source_layout")
    root = validate_member_path(layout["root"])
    if "/" in root:
        raise VendorError("Source distribution root must be one directory")
    validate_member_path(layout["metadata"])
    files = layout["files"]
    if not isinstance(files, dict) or not files:
        raise VendorError("Source distribution requires explicit source-to-runtime file mappings")
    paths = ArchivePaths()
    for destination, source in files.items():
        paths.add(destination)
        validate_member_path(source)
        if destination.endswith(".pth") or any(part.endswith(".dist-info") for part in destination.split("/")):
            raise VendorError(f"Source mapping cannot install startup hooks or metadata: {destination}")
    licenses = layout["licenses"]
    if (
        not isinstance(licenses, list) or not licenses
        or any(not isinstance(path, str) for path in licenses)
        or len(set(licenses)) != len(licenses)
    ):
        raise VendorError("Source distribution requires distinct license notice paths")
    for path in licenses:
        validate_member_path(path)
    excluded = layout.get("excluded_build_bytecode", {})
    if not isinstance(excluded, dict):
        raise VendorError("Reviewed build-bytecode exclusions must map paths to SHA-256 hashes")
    for path, digest in excluded.items():
        validate_member_path(path)
        if (
            "__pycache__" not in path.split("/") or not path.endswith(".pyc")
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or path in files.values() or path == layout["metadata"] or path in licenses
        ):
            raise VendorError(f"Invalid reviewed build-bytecode exclusion: {path}")
    exclusions = layout.get("excluded_native_payloads", {})
    if not isinstance(exclusions, dict):
        raise VendorError("Native exclusions must map paths to reviewed hashes and reasons")
    for path, review in exclusions.items():
        validate_member_path(path)
        if (not isinstance(review, dict) or set(review) != {"sha256", "reason"}
                or Path(path.lower()).suffix not in NATIVE_SUFFIXES
                or not isinstance(review["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", review["sha256"]) is None
                or not isinstance(review["reason"], str) or not review["reason"].strip()
                or path in files.values() or path == layout["metadata"] or path in licenses or path in excluded):
            raise VendorError(f"Invalid reviewed native exclusion: {path}")


def load_dependencies(path: Path = DEFAULT_MANIFEST) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise VendorError("Unsupported dependency manifest format")
    packages = manifest.get("packages")
    if not isinstance(packages, list):
        raise VendorError("Dependency manifest must contain a packages list")
    names: set[str] = set()
    filenames: set[str] = set()
    for package in packages:
        _validate_package(package)
        name = package["name"]
        filename = package["filename"].casefold()
        if name in names or filename in filenames:
            raise VendorError(f"Duplicate dependency: {name}")
        names.add(name)
        filenames.add(filename)
    for package in packages:
        missing = set(package.get("dependencies", [])) - names - set(package.get("replaced_dependencies", {}))
        if missing:
            raise VendorError(
                f"Incomplete reviewed dependency closure for {package['name']}: "
                + ", ".join(sorted(missing))
            )
    return manifest


def _verified_bytes(package: dict, cache_dir: Path) -> bytes:
    _validate_package(package)
    path = Path(cache_dir) / package["filename"]
    if path.is_symlink():
        raise VendorError(f"Cached archive must not be a symlink: {path.name}")
    if not path.is_file():
        raise VendorError(
            f"Missing cached archive: {path.name}. "
            "Run python3 -I -S standalone/vendor.py fetch first; builds never fetch."
        )
    if path.stat().st_size != package["size"]:
        raise VendorError(f"Archive size mismatch: {path.name}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != package["sha256"]:
        raise VendorError(f"Archive SHA-256 mismatch: {path.name}")
    return content


def _wheel_files(content: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    paths = ArchivePaths()
    total_size = 0
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if len(archive.infolist()) > 20_000:
            raise VendorError("Wheel contains too many members")
        for member in archive.infolist():
            if member.orig_filename != member.filename:
                raise VendorError(f"Unsafe archive path: {member.orig_filename!r}")
            path = validate_member_path(member.filename[:-1] if member.is_dir() else member.filename)
            paths.add(path, directory=member.is_dir())
            file_type = stat.S_IFMT(member.external_attr >> 16)
            expected = stat.S_IFDIR if member.is_dir() else stat.S_IFREG
            if file_type not in (0, expected):
                raise VendorError(f"Wheel contains a symlink or special file: {path}")
            if member.flag_bits & 1:
                raise VendorError(f"Encrypted wheel member is forbidden: {path}")
            if member.is_dir():
                continue
            total_size += member.file_size
            if member.file_size > MAX_MEMBER_SIZE or total_size > MAX_EXPANDED_WHEEL_SIZE:
                raise VendorError("Wheel exceeds the expanded-size limit")
            if "/" in path and path.split("/", 1)[0].endswith(".data"):
                raise VendorError(f"Wheel install-scheme relocation requires review: {path}")
            if Path(path).suffix == ".pth":
                raise VendorError(f"Wheel startup hooks are forbidden: {path}")
            data = archive.read(member)
            reject_native(path, data)
            files[path] = data
    return files


def _source_archive_files(content: bytes, layout: dict) -> dict[str, bytes]:
    """Read bounded regular tar members without extracting or running build code."""
    files: dict[str, bytes] = {}
    paths = ArchivePaths()
    total_size = 0
    root = layout["root"]
    excluded = layout.get("excluded_build_bytecode", {})
    excluded_native = layout.get("excluded_native_payloads", {})
    omitted: set[str] = set()
    omitted_native: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        for index, member in enumerate(archive):
            if index >= 20_000:
                raise VendorError("Source distribution contains too many members")
            path = validate_member_path(
                member.name[:-1] if member.isdir() and member.name.endswith("/") else member.name
            )
            paths.add(path, directory=member.isdir())
            if path != root and not path.startswith(root + "/"):
                raise VendorError(f"Source member is outside the reviewed root: {path}")
            if member.isdir():
                continue
            if not member.isreg():
                raise VendorError(f"Source distribution contains a symlink or special file: {path}")
            total_size += member.size
            if member.size < 0 or member.size > MAX_MEMBER_SIZE or total_size > MAX_EXPANDED_WHEEL_SIZE:
                raise VendorError("Source distribution exceeds the expanded-size limit")
            stream = archive.extractfile(member)
            if stream is None:
                raise VendorError(f"Source distribution member is unreadable: {path}")
            data = stream.read(MAX_MEMBER_SIZE + 1)
            if len(data) != member.size:
                raise VendorError(f"Source distribution member size differs: {path}")
            relative = path.removeprefix(root + "/")
            if relative in excluded:
                if hashlib.sha256(data).hexdigest() != excluded[relative]:
                    raise VendorError(f"Reviewed build-bytecode hash differs: {relative}")
                omitted.add(relative)
                continue
            if relative in excluded_native:
                if hashlib.sha256(data).hexdigest() != excluded_native[relative]["sha256"]:
                    raise VendorError(f"Reviewed native payload hash differs: {relative}")
                omitted_native.add(relative)
                continue
            reject_native(path, data)
            files[relative] = data
    if omitted != set(excluded):
        raise VendorError("Reviewed build-bytecode exclusions are absent from the source archive")
    if omitted_native != set(excluded_native):
        raise VendorError("Reviewed native exclusions are absent from the source archive")
    return files


def _record_bytes(files: dict[str, bytes], metadata_dir: str) -> bytes:
    record_path = metadata_dir + "/RECORD"
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for path, data in sorted(files.items()):
        if path == record_path:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        writer.writerow((path, "sha256=" + digest, len(data)))
    writer.writerow((record_path, "", ""))
    return record.getvalue().encode("utf-8")


def _sdist_files(package: dict, content: bytes) -> dict[str, bytes]:
    """Copy reviewed upstream Python fallbacks and synthesize install metadata."""
    layout = package["source_layout"]
    original = _source_archive_files(content, layout)
    files: dict[str, bytes] = {}
    paths = ArchivePaths()

    def copy(source: str, destination: str) -> None:
        if source not in original:
            raise VendorError(f"Reviewed source file is missing: {source}")
        paths.add(destination)
        files[destination] = original[source]

    for destination, source in sorted(layout["files"].items()):
        copy(source, destination)
    metadata_dir = package["name"].replace("-", "_") + "-" + package["version"] + ".dist-info"
    copy(layout["metadata"], metadata_dir + "/METADATA")
    for path in layout["licenses"]:
        copy(path, metadata_dir + "/licenses/" + path)
    files[metadata_dir + "/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: hermes-standalone-sdist/1\n"
        b"Root-Is-Purelib: true\nTag: py3-none-any\n"
    )
    files[metadata_dir + "/INSTALLER"] = b"hermes-standalone\n"
    files[metadata_dir + "/RECORD"] = _record_bytes(files, metadata_dir)
    return files


def _patch_sources(files: dict[str, bytes], package: dict, metadata_dir: str, paths: ArchivePaths) -> None:
    for patch in package.get("source_patches", []):
        path = patch["path"]
        original = files.get(path)
        if original is None or hashlib.sha256(original).hexdigest() != patch["original_sha256"]:
            raise VendorError(f"Reviewed source patch input differs: {path}")
        text = original.decode("utf-8")
        for replacement in patch["replacements"]:
            if text.count(replacement["old"]) != 1:
                raise VendorError(f"Reviewed source patch needs exactly one matching block: {path}")
            text = text.replace(replacement["old"], replacement["new"], 1)
        patched = text.encode("utf-8")
        if hashlib.sha256(patched).hexdigest() != patch["patched_sha256"]:
            raise VendorError(f"Reviewed source patch output differs: {path}")
        reject_native(path, patched)
        saved = metadata_dir + "/upstream_sources/" + path
        paths.add(saved)
        files[saved] = original
        files[path] = patched


def read_wheel(package: dict, cache_dir: Path) -> tuple[dict[str, bytes], dict]:
    """Verify a pinned wheel or reviewed sdist without executing its build system.

    The existing build/probe API retains its name. Source distributions only add
    explicit file selection and generated metadata; selected source bytes remain
    identical to the pinned archive.
    """
    content = _verified_bytes(package, cache_dir)
    kind = package.get("kind", "wheel")
    files = _sdist_files(package, content) if kind == "sdist" else _wheel_files(content)
    metadata_paths = [path for path in files if re.fullmatch(r"[^/]+\.dist-info/METADATA", path)]
    wheel_paths = [path for path in files if re.fullmatch(r"[^/]+\.dist-info/WHEEL", path)]
    if len(metadata_paths) != 1 or len(wheel_paths) != 1:
        raise VendorError(f"Wheel needs exactly one METADATA and WHEEL: {package['name']}")
    metadata_path = metadata_paths[0]
    metadata_dir = metadata_path.rsplit("/", 1)[0]
    if wheel_paths[0].rsplit("/", 1)[0] != metadata_dir:
        raise VendorError(f"Wheel metadata directories disagree: {package['name']}")
    parser = email.parser.BytesParser(policy=email.policy.default)
    metadata = parser.parsebytes(files[metadata_path])
    wheel = parser.parsebytes(files[wheel_paths[0]])
    if canonical_name(str(metadata.get("Name", ""))) != package["name"] or metadata.get("Version") != package["version"]:
        raise VendorError(f"Wheel name/version differs from the pin: {package['name']}")
    tags = wheel.get_all("Tag", [])
    if wheel.get("Root-Is-Purelib", "").lower() != "true" or not tags or any(
        not str(tag).endswith("-none-any") for tag in tags
    ):
        raise VendorError(f"Wheel metadata declares a non-pure distribution: {package['name']}")
    if package.get("additional_licenses") or package.get("source_patches"):
        paths = ArchivePaths()
        for path in files:
            paths.add(path)
        record_path = metadata_dir + "/RECORD"
        if record_path not in files:
            raise VendorError(f"Reviewed metadata/source transformation needs the original RECORD: {package['name']}")
        original_record = record_path + ".upstream"
        paths.add(original_record)
        files[original_record] = files[record_path]
        _patch_sources(files, package, metadata_dir, paths)
        for notice in package.get("additional_licenses", []):
            path = metadata_dir + "/licenses/" + notice["filename"]
            paths.add(path)
            data = notice["text"].encode("utf-8")
            reject_native(path, data)
            files[path] = data
        files[record_path] = _record_bytes(files, metadata_dir)
    license_paths = [
        path for path in files
        if path.startswith(metadata_dir + "/")
        and (
            "/licenses/" in path.lower()
            or Path(path).name.lower().startswith(("license", "licence", "copying", "notice"))
        )
    ]
    if not license_paths:
        raise VendorError(f"Wheel does not contain a license notice: {package['name']}")
    license_text = metadata.get("License-Expression") or metadata.get("License")
    if not license_text:
        license_text = "; ".join(
            item.removeprefix("License :: ")
            for item in metadata.get_all("Classifier", [])
            if item.startswith("License :: ")
        ) or "See included license notices"
    provenance = {key: package[key] for key in ("name", "version", "filename", "url", "sha256", "size")}
    provenance.update({
        "kind": kind,
        "pin_source": package.get("pin_source", "upstream-lock"),
        "license": str(license_text),
        "metadata_path": "vendor/" + metadata_path,
        "license_files": ["vendor/" + path for path in sorted(license_paths)],
        "requires_python": metadata.get("Requires-Python"),
        "requires_dist": metadata.get_all("Requires-Dist", []),
        "dependencies": package.get("dependencies", []),
    })
    if package.get("additional_licenses"):
        provenance["additional_licenses"] = [
            {key: notice[key] for key in ("filename", "sha256", "source_url")}
            for notice in package["additional_licenses"]
        ]
    if package.get("additional_licenses") or package.get("source_patches"):
        provenance["original_record_path"] = "vendor/" + metadata_dir + "/RECORD.upstream"
    if package.get("source_patches"):
        provenance["source_patches"] = [
            {key: patch[key] for key in ("path", "original_sha256", "patched_sha256")}
            for patch in package["source_patches"]
        ]
        provenance["replaced_dependencies"] = package.get("replaced_dependencies", {})
    if kind == "sdist":
        layout_bytes = json.dumps(
            package["source_layout"], sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        provenance["source_transformation"] = {
            "method": "reviewed-sdist-files-v1",
            "selected_source_bytes_unchanged": not bool(package.get("source_patches")),
            "source_layout_sha256": hashlib.sha256(layout_bytes).hexdigest(),
            "source_files": len(package["source_layout"]["files"]),
            "metadata_source": package["source_layout"]["metadata"],
            "generated_metadata": ["INSTALLER", "RECORD", "WHEEL"],
            "omitted_build_bytecode": package["source_layout"].get("excluded_build_bytecode", {}),
            "omitted_native_payloads": package["source_layout"].get("excluded_native_payloads", {}),
        }
    return files, provenance


def fetch_dependencies(manifest: dict, cache_dir: Path) -> list[dict]:
    """Explicitly download pinned archives, validating before cache publication."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    results = []
    for package in manifest["packages"]:
        _validate_package(package)
        destination = cache_dir / package["filename"]
        if destination.exists() or destination.is_symlink():
            read_wheel(package, cache_dir)
            results.append({"name": package["name"], "status": "verified"})
            continue
        request = urllib.request.Request(
            package["url"], headers={"User-Agent": "hermes-standalone-vendor/1"}
        )
        descriptor, temporary = tempfile.mkstemp(prefix=".download-", dir=cache_dir)
        temporary_path = Path(temporary)
        try:
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(descriptor, "wb") as output:
                with urllib.request.urlopen(request, timeout=45) as response:
                    _validate_url(response.geturl())
                    while chunk := response.read(128 * 1024):
                        size += len(chunk)
                        if size > package["size"]:
                            raise VendorError(f"Download exceeds pinned size: {package['name']}")
                        digest.update(chunk)
                        output.write(chunk)
            if size != package["size"] or digest.hexdigest() != package["sha256"]:
                raise VendorError(f"Downloaded archive size/SHA-256 mismatch: {package['name']}")
            # Validate through the same path as an offline build before making it available.
            with tempfile.TemporaryDirectory(prefix=".inspect-", dir=cache_dir) as inspection:
                candidate = Path(inspection) / package["filename"]
                candidate.write_bytes(temporary_path.read_bytes())
                read_wheel(package, Path(inspection))
            temporary_path.replace(destination)
            results.append({"name": package["name"], "status": "downloaded"})
        finally:
            temporary_path.unlink(missing_ok=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "verify"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args(argv)
    try:
        manifest = load_dependencies(args.manifest)
        if args.command == "fetch":
            results = fetch_dependencies(manifest, args.cache_dir)
        else:
            results = []
            for package in manifest["packages"]:
                read_wheel(package, args.cache_dir)
                results.append({"name": package["name"], "status": "verified"})
        print(json.dumps({"packages": results, "cache_dir": str(args.cache_dir.resolve())}, indent=2))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError) as error:
        print(f"vendor: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
