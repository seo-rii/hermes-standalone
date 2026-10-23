"""Measure lossless payload encodings without importing or running bundled code."""

from __future__ import annotations

import argparse
import ast
import base64
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import stat
import time
import tokenize
import zipfile
import zlib


def _triple_literal(source, terminator):
    body = ["'''\n"]
    for raw in source:
        if raw.rstrip("\r\n") == terminator:
            body.append("'''\n")
            return ast.literal_eval("".join(body))
        body.append(raw)
    raise ValueError("Unterminated literal bundle payload")


def _literal_chunks(source, first):
    if first != "(":
        yield _triple_literal(source, "'''") if first == "'''" else ast.literal_eval(first)
        return
    for raw in source:
        line = raw.strip()
        if line == ")":
            return
        yield _triple_literal(source, "''',") if line == "'''" else ast.literal_eval(line)
    raise ValueError("Unterminated bundle payload container")


def _read_literals(path):
    info, chunks = None, []
    with tokenize.open(path) as source:
        for raw in source:
            line = raw.strip()
            if line.startswith("_BUNDLE_INFO = "):
                info = ast.literal_eval(line.partition(" = ")[2])
            elif line.startswith("_BUNDLE_PAYLOAD = "):
                chunks = list(_literal_chunks(source, line.partition(" = ")[2]))
                break
    if not isinstance(info, dict) or not chunks or any(not isinstance(item, str) for item in chunks):
        raise ValueError("Cannot locate literal bundle metadata and payload")
    return info, chunks


def _chunked_bytes(info, chunks):
    size = info.get("payload_size")
    width = 256 * 1024
    if (type(size) is not int or not 0 < size <= 1024 * 1024 * 1024
            or info.get("payload_encoding") != "latin1-chunks"
            or len(chunks) != (size + width - 1) // width):
        raise ValueError("Invalid literal payload chunks or encoding")
    output = []
    for index, chunk in enumerate(chunks):
        expected = min(width, size - index * width)
        if len(chunk) != expected + 2 or not chunk.startswith("\n") or not chunk.endswith("\n"):
            raise ValueError("Invalid literal payload chunk size or delimiters")
        output.append(chunk[1:-1].encode("latin-1"))
    return b"".join(output)


def _payload_bytes(info, chunks):
    if sum(len(item) for item in chunks) > (512 * 1024 * 1024 * 5 + 3) // 4:
        raise ValueError("Encoded payload exceeds the measurement limit")
    if info.get("format_version") == 4:
        return _chunked_bytes(info, chunks)
    encoded = "".join(chunks)
    if info.get("format_version") == 3:
        if (info.get("payload_encoding") != "latin1" or len(encoded) != info.get("payload_size", -2) + 2
                or not encoded.startswith("\n") or not encoded.endswith("\n")):
            raise ValueError("Invalid literal payload encoding or size")
        return encoded[1:-1].encode("latin-1")
    return base64.b85decode(encoded)


def _inflate_payload(info, payload):
    if info.get("format_version") in (2, 3, 4, 5):
        if len(payload) != info["payload_size"] or hashlib.sha256(payload).hexdigest() != info["payload_sha256"]:
            raise ValueError("Compressed payload integrity check failed")
        if info.get("compression") not in (("lzma", "zip") if info["format_version"] >= 3 else ("lzma",)):
            raise ValueError("Unsupported payload encoding")
        if info.get("compression") == "lzma":
            import lzma

            decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64 * 1024 * 1024)
            payload = decoder.decompress(payload, max_length=info["size"] + 1)
            if not decoder.eof or decoder.unused_data:
                raise ValueError("Incomplete or trailing compressed data")
    elif info.get("format_version") != 1:
        raise ValueError("Unsupported bundle version")
    return payload


def _validate_zipapp_member(item):
    mode = (item.external_attr >> 16) & 0xFFFF
    limit = 1024 * 1024 if item.filename == "__main__.py" else 1024 * 1024 * 1024
    if (item.compress_type != zipfile.ZIP_STORED or item.flag_bits & 1 or item.extra
            or item.compress_size != item.file_size or not 0 < item.file_size <= limit
            or stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
        raise ValueError("Invalid ZIP application member metadata")


def _zipapp_payload(path):
    with zipfile.ZipFile(path, "r") as outer:
        members = outer.infolist()
        if (len(members) != 2 or {item.filename for item in members} != {"__main__.py", "payload.bin"}
                or outer.comment):
            raise ValueError("Invalid ZIP application members")
        for item in members:
            _validate_zipapp_member(item)
        lines = outer.read("__main__.py").decode("utf-8").splitlines()
        values = [ast.literal_eval(line.strip().partition(" = ")[2])
                  for line in lines if line.strip().startswith("_BUNDLE_INFO = ")]
        if (len(values) != 1 or not isinstance(values[0], dict)
                or values[0].get("format_version") != 5
                or values[0].get("payload_encoding") != "zip-member"):
            raise ValueError("Invalid ZIP application metadata")
        info = values[0]
        if (type(info.get("payload_size")) is not int
                or outer.getinfo("payload.bin").file_size != info["payload_size"]):
            raise ValueError("ZIP application payload size differs")
        return info, outer.read("payload.bin")


def read_payload(path: Path) -> tuple[dict, bytes]:
    path = Path(path)
    with path.open("rb") as source:
        zipapp = source.read(4) == b"PK\x03\x04"
    if zipapp:
        info, encoded = _zipapp_payload(path)
    else:
        info, chunks = _read_literals(path)
        encoded = _payload_bytes(info, chunks)
    if type(info.get("format_version")) is not int:
        raise ValueError("Unsupported bundle version")
    if type(info.get("size")) is not int or not 0 < info["size"] <= 1024 * 1024 * 1024:
        raise ValueError("Invalid declared ZIP size")
    payload = _inflate_payload(info, encoded)
    if len(payload) != info["size"] or hashlib.sha256(payload).hexdigest() != info["sha256"]:
        raise ValueError("ZIP integrity check failed")
    return info, payload


def _category(path: str) -> str:
    parts = path.split("/")
    if parts[0] == "vendor":
        return "vendor"
    if len(parts) > 2 and parts[1] in {"skills", "optional-skills", "locales"}:
        return "app/" + parts[1]
    if parts[0] == "app":
        return "app/runtime-and-root"
    return "bundle-manifest"


def compare_archive(archive_bytes: bytes) -> dict:
    import lzma

    categories = defaultdict(lambda: {"files": 0, "bytes": 0, "zip_compressed_bytes": 0})
    entries = []
    stored_buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as original, zipfile.ZipFile(
        stored_buffer, "w", compression=zipfile.ZIP_STORED
    ) as stored:
        for item in sorted(original.infolist(), key=lambda item: item.filename):
            data = original.read(item)
            category = categories[_category(item.filename)]
            category["files"] += 1
            category["bytes"] += len(data)
            category["zip_compressed_bytes"] += item.compress_size
            entries.append({"path": item.filename, "bytes": len(data), "zip_compressed_bytes": item.compress_size})
            member = zipfile.ZipInfo(item.filename, date_time=item.date_time)
            member.create_system = item.create_system
            member.external_attr = item.external_attr
            stored.writestr(member, data, compress_type=zipfile.ZIP_STORED)
    raw = stored_buffer.getvalue()
    options = {}
    for name, compress, decompress in (
        ("solid-zlib-9", lambda data: zlib.compress(data, level=9), zlib.decompress),
        ("solid-lzma-6", lambda data: lzma.compress(data, format=lzma.FORMAT_XZ, preset=6), lzma.decompress),
        ("solid-lzma-8", lambda data: lzma.compress(data, format=lzma.FORMAT_XZ, preset=8), lzma.decompress),
        ("solid-lzma-8-48m-pb0", lambda data: lzma.compress(
            data, format=lzma.FORMAT_XZ,
            filters=[{"id": lzma.FILTER_LZMA2, "preset": 8, "dict_size": 48 * 1024 * 1024, "pb": 0}],
        ), lambda data: lzma.decompress(data, memlimit=64 * 1024 * 1024)),
    ):
        start = time.perf_counter()
        compressed = compress(raw)
        elapsed = time.perf_counter() - start
        start = time.perf_counter()
        restored = decompress(compressed)
        restored_seconds = time.perf_counter() - start
        if restored != raw:
            raise ValueError(f"Lossless round-trip failed: {name}")
        encoded = base64.b85encode(compressed).decode("ascii")
        layouts = {}
        for width in (100, 8192):
            literal = "_PAYLOAD = (\n" + "\n".join(
                repr(encoded[index:index + width]) for index in range(0, len(encoded), width)
            ) + "\n)\n"
            start = time.perf_counter()
            compile(literal, "<payload-size-measurement>", "exec")
            layouts[str(width)] = {
                "source_bytes": len(literal.encode("utf-8")),
                "compile_seconds": round(time.perf_counter() - start, 3),
            }
        options[name] = {
            "bytes": len(compressed),
            "base85_bytes": len(encoded),
            "compression_seconds": round(elapsed, 3),
            "decompression_seconds": round(restored_seconds, 3),
            "sha256": hashlib.sha256(compressed).hexdigest(),
            "roundtrip_exact": True,
            "literal_layouts": layouts,
        }
    return {
        "archive_bytes": len(archive_bytes),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "stored_zip_bytes": len(raw),
        "stored_zip_sha256": hashlib.sha256(raw).hexdigest(),
        "file_count": len(entries),
        "categories": dict(sorted(categories.items())),
        "largest_files": sorted(entries, key=lambda item: (-item["bytes"], item["path"]))[:20],
        "encodings": options,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(__file__).parents[1] / "dist" / "hermes.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _, archive_bytes = read_payload(args.artifact)
    report = compare_archive(archive_bytes)
    report["artifact_bytes"] = args.artifact.stat().st_size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output.resolve()), "encodings": report["encodings"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
