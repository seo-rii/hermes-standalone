"""Stage an already built original dashboard, retaining provenance and notices.

This standard-library tool does not install Node or invoke a compiler. Prepare
the original locked web workspace with scripts/build/node-deps.mjs and build
with scripts/build/web.mjs, then pass those two directories here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


def _digest(data):
    return hashlib.sha256(data).hexdigest()


_GENERATED = {"node_modules", "dist", "build", "release", ".cache", ".git", "coverage",
              "test-results", "playwright-report"}
_WEB_INPUTS = ("web", "apps/shared", "package.json", "package-lock.json", ".npmrc", "pm/lock.json",
               "scripts/build", "scripts/generate-icons.mjs", "scripts/generate_icons.py",
               "assets", "pyproject.toml", "uv.lock")
_NOTICE_PREFIXES = ("license", "licence", "notice", "copying", "copyright")


def _tree_hash(root, inputs, *, skip=lambda _name: False, output=False):
    """Match the producer's treeHash, including UTF-16 ordering and missing inputs."""
    digest = hashlib.sha256()

    def visit(name):
        base = name.rsplit("/", 1)[-1]
        if (base in {".DS_Store", ".localized", "Thumbs.db", "Desktop.ini"}
                or base.startswith("._") or skip(name)):
            return
        path = root / name
        if output and path.is_symlink():
            raise ValueError("Dashboard assets must not contain symlinks")
        digest.update(name.encode("utf-8") + b"\0")
        if not path.exists():
            digest.update(b"missing\0")
            return
        if path.is_dir():
            digest.update(b"directory\0")
            for child in sorted(path.iterdir(), key=lambda p: p.name.encode("utf-16-be", "surrogatepass")):
                visit(name + "/" + child.name)
        elif path.is_file():
            digest.update(b"file" if output and ("node_modules" in name.split("/")
                          or name.startswith("native/")) else path.read_bytes())
            digest.update(b"\0")
        else:
            raise ValueError("Dashboard inputs and assets must be regular files and directories")

    for name in inputs:
        visit(name)
    return digest.hexdigest()


def _output_hash(product):
    """Match scripts/build/freshness.mjs outputHash, without invoking Node."""
    names = sorted((path.name for path in product.iterdir()), key=lambda name: name.encode("utf-16-be", "surrogatepass"))
    return _tree_hash(product, names, output=True, skip=lambda name: name in {"hermes-build.json", ".hermes-product"})


def _source_hash(source):
    def generated(name):
        parts = name.split("/")
        return (not name.startswith("scripts/") and any(part in _GENERATED for part in parts)
                or any(part.startswith((".dist-", ".staging-")) or part == "__pycache__" for part in parts)
                or name.endswith((".tsbuildinfo", ".pyc")))

    return _tree_hash(source, _WEB_INPUTS, skip=generated)


def _verify_inputs(source, receipt):
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("sourceHash") != _source_hash(source):
        raise ValueError("Dashboard source differs from its original build receipt")
    prepared = inputs.get("prepared")
    if not isinstance(prepared, list) or len(prepared) != 1 or not isinstance(prepared[0], dict):
        raise ValueError("Dashboard receipt requires its prepared icons inventory")
    icons = prepared[0]
    path = icons.get("path")
    if (icons.get("name") != "icons" or not isinstance(path, str) or not Path(path).is_absolute()
            or not Path(path).is_dir() or icons.get("hash") != _tree_hash(Path(path), (".",))):
        raise ValueError("Dashboard prepared icons differ from the original build receipt")
    return [{"name": "icons", "hash": icons["hash"]}]


def _runtime_packages(source, pins):
    """Resolve the web/shared runtime graph as Node does, retaining optional omissions."""
    from standalone.vendor import validate_member_path

    pending, visited, required = ["web", "apps/shared"], set(), set()
    while pending:
        owner = pending.pop()
        if owner in visited:
            continue
        visited.add(owner)
        pin = pins.get(owner)
        if not isinstance(pin, dict) or not (source / owner / "package.json").is_file():
            raise ValueError("Missing locked dashboard package: " + owner)
        if "node_modules" in owner.split("/"):
            required.add(owner)
        dependencies = {**pin.get("peerDependencies", {}), **pin.get("dependencies", {}),
                        **pin.get("optionalDependencies", {})}
        optional = set(pin.get("optionalDependencies", {})) | {
            name for name, metadata in pin.get("peerDependenciesMeta", {}).items() if metadata.get("optional")
        }
        for name in dependencies:
            validate_member_path(name)
            if re.fullmatch(r"(?:@[^/]+/)?[^/]+", name) is None:
                raise ValueError("Invalid dashboard dependency name")
            directory = PurePosixPath(owner)
            candidates = [(parent / "node_modules" / name).as_posix()
                          for parent in (directory, *directory.parents) if parent.name != "node_modules"]
            target = next((candidate for candidate in candidates if (source / candidate / "package.json").is_file()), None)
            if target is None:
                if name in optional:
                    continue
                raise ValueError("Missing locked dashboard dependency: " + name)
            if target not in pins:
                raise ValueError("Dashboard dependency is absent from its source package lock: " + name)
            if pins[target].get("link"):
                target = pins[target].get("resolved", "")
                validate_member_path(target)
            pending.append(target)
    return required


def _package_notices(directory):
    """Include nested upstream notices without recopying installed child packages."""
    for current, directories, filenames in os.walk(directory):
        directories[:] = sorted(name for name in directories if name not in {"node_modules", ".git"})
        for name in sorted(filenames):
            path = Path(current) / name
            if name.lower().startswith(_NOTICE_PREFIXES) and path.is_file():
                yield path


def _inventory_packages(source, lock, add):
    from standalone.vendor import validate_member_path

    pins = lock.get("packages")
    if not isinstance(pins, dict):
        raise ValueError("Dashboard requires a complete source package lock")
    required = _runtime_packages(source, pins)
    packages = []
    for name, pin in sorted(pins.items()):
        if not name:
            continue
        validate_member_path(name)
        directory = source / name
        manifest = directory / "package.json"
        if "node_modules" not in name.split("/"):
            if name not in {"web", "apps/shared"}:
                continue
        elif not manifest.is_file():
            if directory.exists() or directory.is_symlink() or name in required:
                raise ValueError("Missing dashboard package manifest: " + name)
            continue
        if pin.get("link"):
            target = pin.get("resolved", "")
            validate_member_path(target)
            if directory.resolve() != (source / target).resolve() or target not in pins:
                raise ValueError("Dashboard workspace package identity differs: " + name)
            pin = pins[target]
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        expected_name = pin.get("name") or name.rsplit("node_modules/", 1)[-1]
        if metadata.get("name") != expected_name or metadata.get("version") != pin.get("version"):
            raise ValueError("Dashboard installed package identity differs from its lock: " + name)
        if "node_modules" not in name.split("/") or pins[name].get("link"):
            continue
        notices = []
        for path in _package_notices(directory):
            destination = "third-party/" + name.replace("node_modules", "packages") + "/" + path.relative_to(directory).as_posix()
            add(destination, path.read_bytes())
            notices.append(destination)
        if name in required and not notices:
            raise ValueError("Missing upstream license notice for dashboard runtime package: " + expected_name)
        packages.append({"name": metadata["name"], "version": metadata["version"],
                         "license": metadata.get("license"), "integrity": pin.get("integrity"),
                         "resolved": pin.get("resolved"), "notices": notices})
    return packages


def stage(source: Path, product: Path, output: Path):
    from standalone.vendor import ArchivePaths, reject_native

    receipt = json.loads((product / "hermes-build.json").read_text())
    if receipt.get("schema") != 1 or receipt.get("product") != "web" or not (product / "index.html").is_file():
        raise ValueError("Expected a complete original Hermes web product")
    if receipt.get("outputHash") != _output_hash(product):
        raise ValueError("Dashboard product differs from its original build receipt")
    prepared = _verify_inputs(source, receipt)
    if output.exists():
        raise FileExistsError("Choose a new output directory; an existing asset tree is never overwritten")
    files, paths = {}, ArchivePaths()

    def add(name, data):
        paths.add(name)
        reject_native(name, data)
        files[name] = data

    for path in sorted(product.rglob("*")):
        if path.is_symlink():
            raise ValueError("Dashboard assets must not contain symlinks")
        if (path.is_file() and path.name not in (".hermes-product", "hermes-build.json", ".DS_Store", ".localized", "Thumbs.db", "Desktop.ini")
                and not path.name.startswith("._")):
            add(path.relative_to(product).as_posix(), path.read_bytes())
    lock_bytes = (source / "package-lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    packages = _inventory_packages(source, lock, add)
    manifest = {"schema": 1, "builder": "scripts/build/web.mjs", "node": receipt["node"],
                "source_hash": receipt["inputs"]["sourceHash"], "package_lock_sha256": _digest(lock_bytes),
                "upstream_output_hash": receipt["outputHash"], "compiler_platform": receipt["platform"],
                "compiler_arch": receipt["arch"], "prepared_inputs": prepared, "packages": packages,
                "files": {name: {"size": len(data), "sha256": _digest(data)} for name, data in sorted(files.items())}}
    add("standalone-assets.json", (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hermes-assets-", dir=output.parent) as temporary:
        staged = Path(temporary) / "assets"
        staged.mkdir()
        for name, data in files.items():
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        staged.rename(output)
    return {"files": len(files), "bytes": sum(map(len, files.values())), "packages": len(packages)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    print(json.dumps(stage(args.source.resolve(), args.product.resolve(), args.output.resolve())))


if __name__ == "__main__":
    main()
