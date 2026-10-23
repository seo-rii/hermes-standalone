#!/usr/bin/env python3
"""Restore pinned standalone source assets from checked-in, bounded XZ chunks.

Run with ``python3 -I -S scripts/standalone_assets.py restore`` before building.
The manifest authenticates local resources; this tool never fetches dependencies.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import secrets
import stat
import sys


MIB = 1024 * 1024
MAX_MANIFEST_BYTES = MIB
MAX_ASSET_BYTES = 512 * MIB
MAX_CHUNK_BYTES = 16 * MIB
MAX_COMPRESSED_BYTES = 512 * MIB
MAX_ASSETS = 32
MAX_CHUNKS = 64
READ_BYTES = MIB
DECODER_MEMORY_BYTES = 64 * MIB
MANIFEST_PATH = "build-resources/manifest.json"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,254}\Z")
_CHUNK = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.xz\.[0-9]{3}\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    prefix + str(number) for prefix in ("COM", "LPT") for number in range(1, 10)
}


class AssetError(ValueError):
    """An invalid manifest, archive, or occupied destination prevented restoration."""


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise AssetError(f"Invalid {label} fields")


def _integer(value, maximum, label):
    if type(value) is not int or not 0 < value <= maximum:
        raise AssetError(f"Invalid {label} bound")


def _digest(value):
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise AssetError("Invalid SHA-256 digest")


def _relative_path(value, prefix):
    if type(value) is not str or len(value) > 512:
        raise AssetError("Invalid resource path")
    parts = value.split("/")
    if len(parts) < 2 or parts[0] != prefix:
        raise AssetError(f"Resource path must be confined to {prefix}/")
    for part in parts:
        if (not _COMPONENT.fullmatch(part) or part.endswith(".")
                or part.partition(".")[0].upper() in _RESERVED):
            raise AssetError("Resource path is not portable and canonical")
    if prefix == "build-resources" and (len(parts) != 2 or not _CHUNK.fullmatch(parts[1])):
        raise AssetError("Invalid XZ chunk path")
    return value


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssetError("Duplicate manifest field")
        result[key] = value
    return result


def _validate_asset(asset, used_chunks):
    _keys(asset, ("path", "size", "sha256", "compressed_size", "compressed_sha256", "chunks"), "asset")
    _relative_path(asset["path"], "standalone")
    _integer(asset["size"], MAX_ASSET_BYTES, "asset size")
    _integer(asset["compressed_size"], MAX_COMPRESSED_BYTES, "compressed size")
    _digest(asset["sha256"])
    _digest(asset["compressed_sha256"])
    chunks = asset["chunks"]
    if type(chunks) is not list or not 0 < len(chunks) <= MAX_CHUNKS:
        raise AssetError("Invalid chunk count")
    chunk_bytes, chunk_stem = 0, None
    for number, chunk in enumerate(chunks):
        _keys(chunk, ("path", "size", "sha256"), "chunk")
        path = _relative_path(chunk["path"], "build-resources")
        _integer(chunk["size"], MAX_CHUNK_BYTES, "chunk size")
        _digest(chunk["sha256"])
        stem, _, suffix = path.rpartition(".")
        chunk_stem = stem if chunk_stem is None else chunk_stem
        if stem != chunk_stem or suffix != f"{number:03d}" or path.casefold() in used_chunks:
            raise AssetError("Duplicate or nonsequential XZ chunk path")
        used_chunks.add(path.casefold())
        chunk_bytes += chunk["size"]
    if chunk_bytes != asset["compressed_size"]:
        raise AssetError("Chunk sizes do not match the compressed size")


def _validate_manifest(manifest):
    _keys(manifest, ("format_version", "compression", "assets"), "manifest")
    if type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
        raise AssetError("Unsupported source asset manifest version")
    if manifest["compression"] != "xz":
        raise AssetError("Unsupported source asset compression")
    assets = manifest["assets"]
    if type(assets) is not list or not 0 < len(assets) <= MAX_ASSETS:
        raise AssetError("Invalid asset count")
    used_chunks = set()
    for asset in assets:
        _validate_asset(asset, used_chunks)
    paths = {asset["path"].casefold() for asset in assets}
    if len(paths) != len(assets):
        raise AssetError("Duplicate or overlapping source asset paths")
    for path in paths:
        parts = path.split("/")
        if any("/".join(parts[:end]) in paths for end in range(1, len(parts))):
            raise AssetError("Duplicate or overlapping source asset paths")
    if len(used_chunks) > MAX_CHUNKS:
        raise AssetError("Too many source asset chunks")
    if sum(asset["compressed_size"] for asset in assets) > MAX_COMPRESSED_BYTES:
        raise AssetError("Total compressed assets exceed the size bound")
    return manifest


def _identity(info):
    # Creating or removing the staging hard link changes ctime, not file content.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _is_reparse_point(info):
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


class _Repository:
    """Keep directory operations anchored to descriptors where the OS supports it."""

    def __init__(self, root):
        self.root = Path(os.path.abspath(root))
        info = self.root.lstat()
        if _is_reparse_point(info) or not stat.S_ISDIR(info.st_mode):
            raise AssetError("The source root must be a directory, never a symlink or reparse point")
        self.fd = None
        self.directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        required = {os.open, os.mkdir, os.stat, os.unlink, os.link}
        if hasattr(os, "O_NOFOLLOW") and required <= os.supports_dir_fd:
            self.fd = os.open(self.root, self.directory_flags)
            opened = os.fstat(self.fd)
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                os.close(self.fd)
                self.fd = None
                raise AssetError("Source root changed while opening it")

    def close(self):
        if self.fd is not None:
            os.close(self.fd)

    @contextmanager
    def parent(self, relative, create=False):
        parts = relative.split("/")[:-1]
        path = self.root
        fd = os.dup(self.fd) if self.fd is not None else None
        try:
            for part in parts:
                name = part if fd is not None else path / part
                if create:
                    try:
                        os.mkdir(name, mode=0o755, dir_fd=fd)
                    except FileExistsError:
                        pass
                if fd is not None:
                    next_fd = os.open(name, self.directory_flags, dir_fd=fd)
                    os.close(fd)
                    fd = next_fd
                else:
                    info = (path / part).lstat()
                    if _is_reparse_point(info) or not stat.S_ISDIR(info.st_mode):
                        raise AssetError("Symlink, reparse point, or nondirectory resource parent")
                path /= part
            yield fd, path
        finally:
            if fd is not None:
                os.close(fd)

    @contextmanager
    def file(self, relative):
        with self.parent(relative) as parent:
            with _regular_file(parent, relative.rsplit("/", 1)[-1]) as stream:
                yield stream


def _name(parent, name):
    fd, path = parent
    return name if fd is not None else path / name


@contextmanager
def _regular_file(parent, name):
    fd, _ = parent
    path = _name(parent, name)
    before = os.stat(path, dir_fd=fd, follow_symlinks=False)
    if _is_reparse_point(before) or not stat.S_ISREG(before.st_mode):
        raise AssetError("Resource must be a regular file, never a symlink or reparse point")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags, dir_fd=fd)
    with os.fdopen(descriptor, "rb") as stream:
        initial = os.fstat(stream.fileno())
        if (_is_reparse_point(initial) or not stat.S_ISREG(initial.st_mode)
                or _identity(initial) != _identity(before)):
            raise AssetError("Resource changed while opening it")
        yield stream
        current = os.stat(path, dir_fd=fd, follow_symlinks=False)
        if (_is_reparse_point(current) or not stat.S_ISREG(current.st_mode)
                or _identity(initial) != _identity(os.fstat(stream.fileno()))
                or _identity(initial) != _identity(current)):
            raise AssetError("Resource changed while reading it")


@contextmanager
def _repository(root):
    repo = None
    try:
        repo = _Repository(root)
        yield repo
    except OSError as error:
        raise AssetError(f"Cannot access source assets: {error.strerror or type(error).__name__}") from error
    finally:
        if repo is not None:
            repo.close()


def _load_manifest(repo):
    with repo.file(MANIFEST_PATH) as stream:
        data = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise AssetError("Source asset manifest exceeds the size bound")
    try:
        manifest = json.loads(data, object_pairs_hook=_object_pairs)
    except (ValueError, RecursionError, UnicodeError) as error:
        raise AssetError("Invalid source asset manifest JSON") from error
    return _validate_manifest(manifest)


def load_manifest(root):
    """Return the strictly validated version-1 manifest for a source checkout."""
    with _repository(root) as repo:
        return _load_manifest(repo)


def _asset_blocks(repo, asset):
    combined = hashlib.sha256()
    for chunk in asset["chunks"]:
        digest, count = hashlib.sha256(), 0
        with repo.file(chunk["path"]) as stream:
            if os.fstat(stream.fileno()).st_size != chunk["size"]:
                raise AssetError(f"Chunk size mismatch: {chunk['path']}")
            while block := stream.read(min(READ_BYTES, chunk["size"] - count + 1)):
                count += len(block)
                if count > chunk["size"]:
                    raise AssetError(f"Chunk size mismatch: {chunk['path']}")
                digest.update(block)
                combined.update(block)
                yield block
        if count != chunk["size"] or digest.hexdigest() != chunk["sha256"]:
            raise AssetError(f"Chunk SHA-256 mismatch: {chunk['path']}")
    if combined.hexdigest() != asset["compressed_sha256"]:
        raise AssetError(f"Compressed asset SHA-256 mismatch: {asset['path']}")


def _verify_archives(repo, assets):
    for asset in assets:
        for _ in _asset_blocks(repo, asset):
            pass


def _matches_file(parent, name, asset):
    digest, count = hashlib.sha256(), 0
    with _regular_file(parent, name) as stream:
        if os.fstat(stream.fileno()).st_size != asset["size"]:
            raise AssetError(f"Existing asset size mismatch: {asset['path']}")
        while block := stream.read(min(READ_BYTES, asset["size"] - count + 1)):
            count += len(block)
            if count > asset["size"]:
                raise AssetError(f"Existing asset size mismatch: {asset['path']}")
            digest.update(block)
    if count != asset["size"] or digest.hexdigest() != asset["sha256"]:
        raise AssetError(f"Existing asset SHA-256 mismatch: {asset['path']}")


def _present(repo, asset):
    try:
        with repo.parent(asset["path"]) as parent:
            _matches_file(parent, asset["path"].rsplit("/", 1)[-1], asset)
    except FileNotFoundError:
        return False
    return True


def _decode(repo, asset, output):
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=DECODER_MEMORY_BYTES)
    digest, count = hashlib.sha256(), 0
    try:
        for compressed in _asset_blocks(repo, asset):
            if decoder.eof:
                raise AssetError(f"Trailing or concatenated XZ data: {asset['path']}")
            pending = compressed
            while pending or not decoder.needs_input:
                block = decoder.decompress(pending, max_length=READ_BYTES)
                pending = b""
                count += len(block)
                if count > asset["size"]:
                    raise AssetError(f"Decoded asset exceeds its size bound: {asset['path']}")
                output.write(block)
                digest.update(block)
                if decoder.eof:
                    if decoder.unused_data:
                        raise AssetError(f"Trailing or concatenated XZ data: {asset['path']}")
                    break
    except lzma.LZMAError as error:
        raise AssetError(f"Invalid XZ data or decoder memory limit: {asset['path']}") from error
    if not decoder.eof:
        raise AssetError(f"Truncated XZ data: {asset['path']}")
    if count != asset["size"] or digest.hexdigest() != asset["sha256"]:
        raise AssetError(f"Decoded asset size or SHA-256 mismatch: {asset['path']}")


def _publish(parent, temporary, asset):
    name = asset["path"].rsplit("/", 1)[-1]
    fd, _ = parent
    try:
        os.link(_name(parent, temporary), _name(parent, name),
                src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
    except FileExistsError:
        _matches_file(parent, name, asset)
        return "present"
    return "restored"


def _restore_one(repo, asset):
    with repo.parent(asset["path"], create=True) as parent:
        temporary = ".source-asset-" + secrets.token_hex(12)
        fd, _ = parent
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(_name(parent, temporary), flags, 0o600, dir_fd=fd)
        try:
            with os.fdopen(descriptor, "wb") as output:
                _decode(repo, asset, output)
                output.flush()
                os.fsync(output.fileno())
            return _publish(parent, temporary, asset)
        finally:
            os.unlink(_name(parent, temporary), dir_fd=fd)


def restore_assets(root):
    """Restore absent originals; return each path's ``restored``/``present`` status.

    Every compressed hash is checked before decoding. A second pass rechecks
    compressed bytes during bounded decoding, before atomic publication.
    Mismatched existing files are always preserved and cause AssetError.
    """
    with _repository(root) as repo:
        assets = _load_manifest(repo)["assets"]
        present = {asset["path"]: _present(repo, asset) for asset in assets}
        _verify_archives(repo, assets)
        return {asset["path"]: "present" if present[asset["path"]] else _restore_one(repo, asset)
                for asset in assets}


def verify_assets(root):
    """Verify all compressed resources and restored originals; return path statuses."""
    with _repository(root) as repo:
        assets = _load_manifest(repo)["assets"]
        _verify_archives(repo, assets)
        for asset in assets:
            if not _present(repo, asset):
                raise AssetError(f"Missing restored source asset: {asset['path']}")
        return {asset["path"]: "verified" for asset in assets}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("restore", "verify"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Source checkout containing build-resources/manifest.json")
    args = parser.parse_args(argv)
    try:
        result = {"restore": restore_assets, "verify": verify_assets}[args.command](args.root)
    except AssetError as error:
        print(f"Source assets: {error}", file=sys.stderr)
        return 1
    for path, status in result.items():
        print(f"{status}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
