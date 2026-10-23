#!/usr/bin/env python3
"""Build template for the single-file, standard-library-only launcher."""

import sys

# A ZIP application is temporarily first on sys.path even under -I. Remove
# its loader path before importing stdlib modules; outer members are data,
# never a place from which bootstrap dependencies may be imported.
_BUNDLE_CONTAINER = getattr(globals().get("__loader__"), "archive", None)
if _BUNDLE_CONTAINER is not None:
    sys.path[:] = [entry for entry in sys.path if entry != _BUNDLE_CONTAINER]

import os

try:
    _BUNDLE_INFO = __HERMES_BUNDLE_INFO_LITERAL__
    _BUNDLE_PAYLOAD = __HERMES_BUNDLE_PAYLOAD_LITERAL__
    _WINDOWS_ACL_SOURCE = __HERMES_WINDOWS_ACL_SOURCE__
except NameError:
    raise SystemExit(
        "This is a build template. Run standalone/build.py, then run the generated artifact."
    ) from None

# Prepare bundled code only in an isolated interpreter. Python's initial startup
# is controlled by the caller; explicit `python -I -S -B -X utf8 artifact.py` also
# isolates that first startup, including the Windows waiting launcher.
if __name__ == "__main__" and not (
    sys.flags.isolated
    and sys.flags.no_site
    and sys.flags.dont_write_bytecode
    and sys.flags.utf8_mode
):
    if not sys.executable:
        raise SystemExit("hermes-standalone: the Python executable cannot be located")
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
        os.path.abspath(_BUNDLE_CONTAINER or __file__),
        *sys.argv[1:],
    ]
    try:
        if os.name == "nt":
            # Windows has no POSIX process replacement. Keep the original
            # process alive for its caller and preserve the child's exit code.
            import subprocess

            raise SystemExit(subprocess.run(command, check=False).returncode)
        os.execv(sys.executable, command)
    except OSError as error:
        raise SystemExit(f"hermes-standalone: isolated startup failed: {error}") from None

import base64
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import stat
import tempfile
import unicodedata
import zipfile

_MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
_MAX_LZMA_MEMORY = 64 * 1024 * 1024
_MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_FILES = 50000
_LITERAL_CHUNK_BYTES = 256 * 1024
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MODULE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*\Z")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_NATIVE_SUFFIXES = {
    ".so",
    ".pyd",
    ".dll",
    ".dylib",
    ".exe",
    ".node",
    ".wasm",
    ".a",
    ".lib",
    ".o",
    ".obj",
    ".pyc",
    ".pyo",
    ".class",
    ".jar",
}
_NATIVE_MAGIC = (
    b"\x7fELF",
    b"MZ",
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
    b"\x00asm",
    b"!<arch>\n",
)
_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


class BundleError(Exception):
    """A malformed bundle or unsafe cache; no bundled code has run."""


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise BundleError(f"invalid JSON constant: {value}")


def _archive_path(name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or unicodedata.normalize("NFC", name) != name
    ):
        raise BundleError(f"unsafe archive path: {name!r}")
    parts = name.split("/")
    for part in parts:
        if (
            part in ("", ".", "..")
            or part[-1:] in (".", " ")
            or any(ord(character) < 32 or ord(character) == 127 for character in part)
            or part.split(".", 1)[0].casefold() in _RESERVED_NAMES
        ):
            raise BundleError(f"unsafe archive path: {name!r}")
    if name != "bundle-manifest.json" and (
        len(parts) < 2 or parts[0] not in ("app", "vendor")
    ):
        raise BundleError(f"archive path outside app/vendor: {name!r}")
    filename = parts[-1].lower()
    if PurePosixPath(filename).suffix in _NATIVE_SUFFIXES or ".so." in filename:
        raise BundleError(f"native or executable payload is forbidden: {name}")
    return parts


def _module_is_bundled(module, files):
    path = module.replace(".", "/")
    return any(
        candidate in files
        for prefix in ("app", "vendor")
        for candidate in (f"{prefix}/{path}.py", f"{prefix}/{path}/__init__.py")
    )


def _archive_spec():
    if (not isinstance(_BUNDLE_INFO, dict)
            or type(_BUNDLE_INFO.get("format_version")) is not int
            or _BUNDLE_INFO["format_version"] not in (1, 2, 3, 4, 5)):
        raise BundleError("unsupported bundle format")
    size, digest = _BUNDLE_INFO.get("size"), _BUNDLE_INFO.get("sha256")
    if (type(size) is not int or not 0 < size <= _MAX_ARCHIVE_BYTES
            or not isinstance(digest, str) or not _HASH.fullmatch(digest)
            or not isinstance(_BUNDLE_INFO.get("version"), str)
            or not _valid_payload_container(_BUNDLE_INFO["format_version"])):
        raise BundleError("invalid bundle metadata or size")
    return size, digest, _BUNDLE_INFO["format_version"]


def _valid_payload_container(version):
    if version == 5:
        return (type(_BUNDLE_PAYLOAD) is tuple and not _BUNDLE_PAYLOAD and isinstance(_BUNDLE_CONTAINER, str)
                and bool(_BUNDLE_CONTAINER) and _BUNDLE_INFO.get("payload_encoding") == "zip-member")
    if version == 4:
        return isinstance(_BUNDLE_PAYLOAD, tuple) and 0 < len(_BUNDLE_PAYLOAD) <= 4096
    return (isinstance(_BUNDLE_PAYLOAD, str)
            and len(_BUNDLE_PAYLOAD) <= ((_MAX_ARCHIVE_BYTES + 3) // 4) * 5)


def _payload_spec(version, size, digest):
    if version == 1:
        return size, digest, "zip"
    payload_size = _BUNDLE_INFO.get("payload_size")
    payload_hash = _BUNDLE_INFO.get("payload_sha256")
    compression = _BUNDLE_INFO.get("compression")
    allowed = ("lzma", "zip") if version >= 3 else ("lzma",)
    if (compression not in allowed or type(payload_size) is not int
            or not 0 < payload_size <= _MAX_ARCHIVE_BYTES
            or not isinstance(payload_hash, str) or not _HASH.fullmatch(payload_hash)):
        raise BundleError("invalid compressed bundle metadata")
    return payload_size, payload_hash, compression


def _chunked_payload_blocks(size):
    count = (size + _LITERAL_CHUNK_BYTES - 1) // _LITERAL_CHUNK_BYTES
    if (_BUNDLE_INFO.get("payload_encoding") != "latin1-chunks"
            or len(_BUNDLE_PAYLOAD) != count):
        raise BundleError("bundle integrity check failed (literal payload chunks or encoding)")
    for index, chunk in enumerate(_BUNDLE_PAYLOAD):
        expected = min(_LITERAL_CHUNK_BYTES, size - index * _LITERAL_CHUNK_BYTES)
        if (not isinstance(chunk, str) or len(chunk) != expected + 2
                or not chunk.startswith("\n") or not chunk.endswith("\n")):
            raise BundleError("bundle integrity check failed (literal payload chunk size or delimiters)")
        yield chunk[1:-1].encode("latin-1")


def _payload_blocks(version, size):
    block_size = 1024 * 1024
    try:
        if version == 5:
            yield from _zip_payload_blocks(size)
            return
        if version == 4:
            yield from _chunked_payload_blocks(size)
            return
        if version == 3:
            if (_BUNDLE_INFO.get("payload_encoding") != "latin1"
                    or len(_BUNDLE_PAYLOAD) != size + 2
                    or not _BUNDLE_PAYLOAD.startswith("\n") or not _BUNDLE_PAYLOAD.endswith("\n")):
                raise BundleError("bundle integrity check failed (literal payload size or encoding)")
            for start in range(1, len(_BUNDLE_PAYLOAD) - 1, block_size):
                yield _BUNDLE_PAYLOAD[start:min(start + block_size, len(_BUNDLE_PAYLOAD) - 1)].encode("latin-1")
        else:
            if len(_BUNDLE_PAYLOAD) != (size * 5 + 3) // 4:
                raise BundleError("bundle integrity check failed (encoded payload size)")
            encoded_block = (block_size // 5) * 5
            for start in range(0, len(_BUNDLE_PAYLOAD), encoded_block):
                yield base64.b85decode(_BUNDLE_PAYLOAD[start:start + encoded_block])
    except (ValueError, UnicodeError) as error:
        raise BundleError(f"invalid bundle encoding: {error}") from None


def _zip_payload_blocks(size):
    with zipfile.ZipFile(_BUNDLE_CONTAINER, "r") as outer:
        members = outer.infolist()
        if (len(members) != 2 or {item.filename for item in members} != {"__main__.py", "payload.bin"}
                or outer.comment):
            raise BundleError("invalid ZIP application members")
        for item in members:
            mode = (item.external_attr >> 16) & 0xFFFF
            limit = size if item.filename == "payload.bin" else 1024 * 1024
            if (item.compress_type != zipfile.ZIP_STORED or item.flag_bits & 1 or item.extra
                    or item.compress_size != item.file_size or not 0 < item.file_size <= limit
                    or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))):
                raise BundleError("invalid ZIP application member metadata")
        member = outer.getinfo("payload.bin")
        if member.file_size != size:
            raise BundleError("ZIP application payload size differs")
        with outer.open(member) as source:
            while block := source.read(1024 * 1024):
                yield block


def _verify_payload(version, size, digest):
    actual_size = 0
    actual_hash = hashlib.sha256()
    for block in _payload_blocks(version, size):
        actual_size += len(block)
        actual_hash.update(block)
    if actual_size != size or actual_hash.hexdigest() != digest:
        raise BundleError("bundle integrity check failed (size or SHA-256 mismatch)")


def _decoder(compression):
    if compression != "lzma":
        return None
    try:
        import lzma
    except ImportError:
        raise BundleError(
            "this artifact requires Python's standard-library lzma module; "
            "use an artifact built with --compression deflate on this interpreter"
        ) from None
    try:
        return lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=_MAX_LZMA_MEMORY)
    except lzma.LZMAError as error:
        raise BundleError(f"cannot decompress bundle within memory limit: {error}") from None


def _decompress(decoder, block, limit):
    import lzma
    try:
        return decoder.decompress(block, max_length=limit)
    except lzma.LZMAError as error:
        raise BundleError(f"cannot decompress bundle within memory limit: {error}") from None


def _write_decoded(stream, block, digest, written, limit):
    written += len(block)
    if written > limit:
        raise BundleError("compressed bundle exceeds declared archive size")
    stream.write(block)
    digest.update(block)
    return written


def _decode_to_file(stream, version, payload_size, size, digest, compression):
    decoder = _decoder(compression)
    written = 0
    actual_hash = hashlib.sha256()
    for block in _payload_blocks(version, payload_size):
        if decoder is None:
            written = _write_decoded(stream, block, actual_hash, written, size)
            continue
        if decoder.eof:
            raise BundleError("compressed bundle has trailing data or concatenated streams")
        while True:
            output = _decompress(decoder, block, min(1024 * 1024, size + 1 - written))
            block = b""
            written = _write_decoded(stream, output, actual_hash, written, size)
            if decoder.eof:
                if decoder.unused_data:
                    raise BundleError("compressed bundle has trailing data or concatenated streams")
                break
            if decoder.needs_input:
                break
    if decoder is not None and not decoder.eof:
        raise BundleError("compressed bundle stream is truncated")
    if written != size or actual_hash.hexdigest() != digest:
        raise BundleError("decoded archive integrity check failed (size or SHA-256 mismatch)")
    stream.seek(0)


def _decode_archive():
    size, digest, version = _archive_spec()
    payload_size, payload_hash, compression = _payload_spec(version, size, digest)
    # Verify compressed input before giving any bytes to the decoder, as in
    # the original loader. Re-reading a bounded string avoids retaining a
    # second complete payload or the complete decoded ZIP in memory.
    _verify_payload(version, payload_size, payload_hash)
    stream = tempfile.TemporaryFile(mode="w+b")
    try:
        _decode_to_file(stream, version, payload_size, size, digest, compression)
        return stream
    except BaseException:
        stream.close()
        raise


class _DecodedArchive(zipfile.ZipFile):
    def __init__(self, source):
        self._source = source
        try:
            super().__init__(source, "r")
        except BaseException:
            source.close()
            self._source = None
            raise

    def close(self):
        try:
            super().close()
        finally:
            source = getattr(self, "_source", None)
            if source is not None:
                source.close()
                self._source = None


def _close_archive_on_error(reader):
    def read():
        archive = _DecodedArchive(_decode_archive())
        try:
            return reader(archive)
        except BaseException:
            archive.close()
            raise
    return read


def _approved_file_encoding(name, encoding):
    if encoding == 'identity':
        return True
    if encoding == 'byteplanes4':
        return name in (
            'app/standalone/assets/piper/hebrew/nakdimon.onnx',
            'app/standalone/assets/piper/tashkeel/model.onnx',
        )
    if encoding == 'sudachi-v1-blocks':
        return name == 'app/standalone/local_piper_japanese/data/sudachi/system.dic'
    return False


@_close_archive_on_error
def _read_archive(archive):
    infos = archive.infolist()
    if not infos or len(infos) > _MAX_FILES:
        raise BundleError("invalid archive member count")
    members = {}
    folded_paths = {}
    total_size = 0
    for info in infos:
        if info.orig_filename != info.filename:
            raise BundleError(f"unsafe archive path: {info.orig_filename!r}")
        parts = _archive_path(info.filename)
        if info.filename in members:
            raise BundleError(f"duplicate archive member: {info.filename}")
        for end in range(1, len(parts) + 1):
            prefix = "/".join(parts[:end])
            previous = folded_paths.setdefault(prefix.casefold(), prefix)
            if previous != prefix:
                raise BundleError(f"case-colliding archive paths: {previous}, {prefix}")
        mode = info.external_attr >> 16
        if (
            info.is_dir()
            or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
            or mode & 0o111
            or info.external_attr & 0x10
            or info.flag_bits & 1
            or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        ):
            raise BundleError(f"archive member is not a plain data file: {info.filename}")
        if not 0 <= info.file_size <= _MAX_FILE_BYTES:
            raise BundleError(f"archive member is too large: {info.filename}")
        total_size += info.file_size
        if total_size > _MAX_TOTAL_BYTES:
            raise BundleError("uncompressed archive is too large")
        members[info.filename] = info
    for name in members:
        parts = name.split("/")
        if any("/".join(parts[:end]) in members for end in range(1, len(parts))):
            raise BundleError(f"archive file is also a parent directory: {name}")
    manifest_info = members.get("bundle-manifest.json")
    if manifest_info is None or manifest_info.file_size > _MAX_MANIFEST_BYTES:
        raise BundleError("missing or oversized bundle manifest")
    manifest_bytes = archive.read(manifest_info)
    manifest = json.loads(
        manifest_bytes.decode("utf-8"),
        object_pairs_hook=_json_object,
        parse_constant=_invalid_json_constant,
    )
    required = {
        "format_version",
        "project",
        "version",
        "profile",
        "upstream",
        "runtime",
        "entrypoint",
        "files",
        "distributions",
        "capabilities",
    }
    if (
        not isinstance(manifest, dict)
        or not required.issubset(manifest)
        or manifest["project"] != "hermes-standalone"
        or manifest["format_version"] != 1
        or manifest["version"] != _BUNDLE_INFO["version"]
        or not isinstance(manifest["files"], list)
    ):
        raise BundleError("invalid or incompatible bundle manifest")
    transformation = manifest.get("source_transform", {"mode": "none"})
    if not isinstance(transformation, dict) or transformation.get("mode") not in {"none", "ast"}:
        raise BundleError("invalid source transformation metadata")
    if transformation["mode"] == "ast":
        toolchain = transformation.get("toolchain")
        current_version = ".".join(map(str, sys.version_info[:3]))
        if (not isinstance(toolchain, dict)
                or toolchain.get("python_version") != current_version
                or toolchain.get("python_implementation") != sys.implementation.name):
            raise BundleError("AST artifact requires its exact verified Python toolchain; "
                              "rebuild it with this interpreter or use the source-preserving artifact")
    files = {}
    for item in manifest["files"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or not _HASH.fullmatch(item["sha256"])
            or type(item.get("size")) is not int
            or not 0 <= item["size"] <= _MAX_FILE_BYTES
        ):
            raise BundleError("invalid manifest file entry")
        name = item["path"]
        _archive_path(name)
        encoding = item.get('encoding', 'identity')
        if not _approved_file_encoding(name, encoding):
            raise BundleError('invalid or unreviewed file encoding: ' + name)
        if name == "bundle-manifest.json" or name in files:
            raise BundleError(f"duplicate or recursive manifest entry: {name}")
        if name not in members or members[name].file_size != item["size"]:
            raise BundleError(f"manifest/archive mismatch: {name}")
        files[name] = item
    if set(files) != set(members) - {"bundle-manifest.json"}:
        raise BundleError("manifest does not describe every archive file")
    entrypoint = manifest["entrypoint"]
    if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
        raise BundleError("invalid bundle entrypoint")
    module, function = entrypoint.split(":")
    if (
        not _MODULE.fullmatch(module)
        or not _IDENTIFIER.fullmatch(function)
        or not _module_is_bundled(module, files)
    ):
        raise BundleError("entrypoint is not a bundled Python module")
    internal_modules = manifest.get("internal_modules", [])
    if (
        not isinstance(internal_modules, list)
        or any(
            not isinstance(module, str)
            or not _MODULE.fullmatch(module)
            or not _module_is_bundled(module, files)
            for module in internal_modules
        )
        or len(set(internal_modules)) != len(internal_modules)
    ):
        raise BundleError("invalid bundled internal-module allowlist")
    return archive, manifest, manifest_bytes, files


def _is_link(metadata):
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & 0x400
    )


_windows_cache_acl = None


def _windows_permissions():
    global _windows_cache_acl
    if _windows_cache_acl is None:
        namespace = {"__name__": "_hermes_bootstrap_windows_acl"}
        exec(compile(_WINDOWS_ACL_SOURCE, "<hermes-bootstrap-windows-acl>", "exec"), namespace)
        _windows_cache_acl = namespace["WindowsCacheACL"]()
    return _windows_cache_acl


def _private_metadata(path, metadata, directory, descriptor=None):
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if _is_link(metadata) or not expected(metadata.st_mode):
        kind = "directory" if directory else "file"
        raise BundleError(f"cache entry is not a regular {kind}: {path}")
    if os.name == "posix":
        if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise BundleError(f"cache entry must be owned by this user and private: {path}")
    elif os.name == "nt":
        _windows_permissions().check(path, descriptor)
    if not directory and metadata.st_nlink != 1:
        raise BundleError(f"hard-linked cache file is forbidden: {path}")


def _cache_directory(argument):
    if argument is None:
        if os.name == "nt":
            parent = os.environ.get("LOCALAPPDATA")
            base = (
                Path(parent)
                if parent and os.path.isabs(parent)
                else Path.home() / "AppData" / "Local"
            )
            path = base / "hermes-standalone" / "Cache"
        elif sys.platform == "darwin":
            path = Path.home() / "Library" / "Caches" / "hermes-standalone"
        else:
            parent = os.environ.get("XDG_CACHE_HOME")
            base = (
                Path(parent)
                if parent and os.path.isabs(parent)
                else Path.home() / ".cache"
            )
            path = base / "hermes-standalone"
    else:
        if not argument:
            raise BundleError("--cache-dir requires a nonempty directory path")
        path = Path(argument).expanduser()
    path = Path(os.path.abspath(path))
    # Canonicalize ancestors, including macOS's /var and /tmp system links.
    # Keep the selected cache root itself unresolved so a root symlink or
    # junction is still rejected, as are all links within the cache.
    path = path.parent.resolve(strict=False) / path.name
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                if os.name == "nt":
                    _windows_permissions().mkdir(current)
                else:
                    current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            metadata = current.lstat()
        if _is_link(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise BundleError(f"cache path contains a link or non-directory: {current}")
    _private_metadata(path, path.lstat(), directory=True)
    return path


def _expected_directories(files):
    directories = set()
    for name in files:
        parts = name.split("/")
        directories.update("/".join(parts[:end]) for end in range(1, len(parts)))
    return directories


def _check_cached_file(path, expected_size, expected_hash):
    metadata = path.lstat()
    _private_metadata(path, metadata, directory=False)
    if metadata.st_size != expected_size:
        raise BundleError(f"cache integrity check failed (size): {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        _private_metadata(path, os.fstat(stream.fileno()), directory=False, descriptor=stream.fileno())
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_hash:
        raise BundleError(f"cache integrity check failed (SHA-256): {path}")


def _validate_cache(root, files, manifest_bytes):
    _private_metadata(root, root.lstat(), directory=True)
    expected_files = set(files) | {"bundle-manifest.json"}
    expected_directories = _expected_directories(files)
    found_files = set()
    found_directories = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode) and not _is_link(metadata):
                    _private_metadata(path, metadata, directory=True)
                    if relative not in expected_directories:
                        raise BundleError(f"unexpected cache directory: {path}")
                    found_directories.add(relative)
                    pending.append(path)
                else:
                    _private_metadata(path, metadata, directory=False)
                    if relative not in expected_files:
                        raise BundleError(f"unexpected cache file: {path}")
                    found_files.add(relative)
    if found_files != expected_files or found_directories != expected_directories:
        raise BundleError(f"cache is incomplete: {root}")
    for name, item in files.items():
        _check_cached_file(root / name, item["size"], item["sha256"])
    _check_cached_file(
        root / "bundle-manifest.json",
        len(manifest_bytes),
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _write_private_file(path, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
        stream.write(data)


def _restore_model_planes(content):
    restored, offset = bytearray(len(content)), 0
    for index in range(4):
        count = (len(content) + 3 - index) // 4
        restored[index::4] = content[offset:offset + count]
        offset += count
    return bytes(restored)


def _restore_encoded_file(source, item, name):
    content = source.read(item['size'] + 1)
    if len(content) != item['size']:
        raise BundleError('encoded model size differs: ' + name)
    decoded = (_restore_model_planes(content) if item['encoding'] == 'byteplanes4'
               else transform_dictionary(content, inverse=True))
    if len(decoded) != item['size']:
        raise BundleError('restored resource size differs: ' + name)
    if any(decoded.startswith(magic) for magic in _NATIVE_MAGIC):
        raise BundleError('native or executable payload is forbidden: ' + name)
    if hashlib.sha256(decoded).hexdigest() != item['sha256']:
        raise BundleError('restored model integrity check failed: ' + name)
    return decoded


def _extract(archive, files, manifest_bytes, cache):
    root = cache / _BUNDLE_INFO["sha256"]
    if os.path.lexists(root):
        _validate_cache(root, files, manifest_bytes)
        return root
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=cache))
    published = False
    try:
        for name in sorted(
            _expected_directories(files), key=lambda item: (item.count("/"), item)
        ):
            (staging / name).mkdir(mode=0o700)
        for name, item in files.items():
            target = staging / name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            digest = hashlib.sha256()
            size = 0
            with archive.open(name) as source, os.fdopen(
                os.open(target, flags, 0o600), "wb"
            ) as destination:
                if item.get('encoding') in ('byteplanes4', 'sudachi-v1-blocks'):
                    destination.write(_restore_encoded_file(source, item, name))
                    continue
                prefix = source.read(8)
                if any(prefix.startswith(magic) for magic in _NATIVE_MAGIC):
                    raise BundleError(f"native or executable payload is forbidden: {name}")
                block = prefix
                while block:
                    size += len(block)
                    if size > item["size"]:
                        raise BundleError(f"archive member exceeds declared size: {name}")
                    digest.update(block)
                    destination.write(block)
                    block = source.read(1024 * 1024)
            if size != item["size"] or digest.hexdigest() != item["sha256"]:
                raise BundleError(f"archive file integrity check failed: {name}")
        _write_private_file(staging / "bundle-manifest.json", manifest_bytes)
        # Other starters publish a complete nonempty directory, so rename
        # either publishes this one atomically or leaves the winner untouched.
        try:
            if os.path.lexists(root):
                _validate_cache(root, files, manifest_bytes)
            else:
                staging.rename(root)
                published = True
        except OSError:
            if not os.path.lexists(root):
                raise
            _validate_cache(root, files, manifest_bytes)
        if not published:
            _validate_cache(root, files, manifest_bytes)
        return root
    finally:
        # Only remove the unique staging directory created by this process.
        # An existing cache, even a corrupt one, is never deleted or repaired.
        if not published:
            shutil.rmtree(staging)


__HERMES_DICTIONARY_PACK_SOURCE__


def _main(argv):
    if sys.version_info < (3, 11):
        raise BundleError("bootstrap commands require Python 3.11 or newer")
    argv = list(argv)
    cache_argument = None
    if argv and argv[0] == "--cache-dir":
        if len(argv) < 2:
            raise BundleError("--cache-dir requires a directory path")
        cache_argument = argv[1]
        del argv[:2]
    elif argv and argv[0].startswith("--cache-dir="):
        cache_argument = argv.pop(0).split("=", 1)[1]
    if argv and (argv[0] == "--cache-dir" or argv[0].startswith("--cache-dir=")):
        raise BundleError("--cache-dir may be specified only once, before the command")
    archive = None
    try:
        archive, manifest, manifest_bytes, files = _read_archive()
        cache = _cache_directory(cache_argument)
        root = _extract(archive, files, manifest_bytes, cache)
    except (OSError, ValueError, UnicodeError, zipfile.BadZipFile) as error:
        raise BundleError(f"cannot prepare bundle: {error}") from None
    finally:
        if archive is not None:
            archive.close()
    context = {
        "artifact": str(Path(_BUNDLE_CONTAINER or __file__).resolve(strict=True)),
        "root": str(root),
        "cache_dir": str(cache),
        "manifest": manifest,
        "bundle_sha256": _BUNDLE_INFO["sha256"],
    }
    sys.path[:0] = [str(root / "app"), str(root / "vendor")]
    from standalone.runtime import initialize_bundle

    initialize_bundle(context)
    if argv and argv[0] == "--internal-module":
        if len(argv) < 2 or argv[1] not in manifest.get("internal_modules", []):
            raise BundleError("internal module is not in the bundled allowlist")
        module = argv[1]
        sys.argv = [module, *argv[2:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0
    module_name, function_name = manifest["entrypoint"].split(":")
    entrypoint = getattr(importlib.import_module(module_name), function_name)
    return entrypoint(argv, context)


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except BundleError as error:
        print(f"hermes-standalone: {error}", file=sys.stderr)
        raise SystemExit(2) from None
