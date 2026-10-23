"""Bounded source literals preserve every byte and real launcher behavior."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from standalone import build, size_report
from standalone.tests.test_bootstrap import TEMPLATE, fixture_archive


class ChunkedPayloadTests(unittest.TestCase):
    def test_all_byte_pairs_and_cross_chunk_escapes_survive_crlf(self):
        width = 256 * 1024
        data = bytearray(random.Random(23).randbytes(3 * width + 17))
        for boundary in (width, 2 * width, 3 * width):
            data[boundary - 4:boundary + 5] = b"\\\x000'''7\r\n"
        pairs = b"".join(bytes((a, b)) for a in range(256) for b in range(256))
        original = bytes(data) + pairs
        template = ("#!/usr/bin/env python3\n"
                    "_BUNDLE_INFO = __HERMES_BUNDLE_INFO_LITERAL__\n"
                    "_BUNDLE_PAYLOAD = __HERMES_BUNDLE_PAYLOAD_LITERAL__\n")
        info = {"format_version": 1, "version": "한글", "size": len(original),
                "sha256": hashlib.sha256(original).hexdigest()}
        source = build._render_encoded(template, info, original, payload_encoding="latin1-chunks")
        for crlf in (False, True):
            with self.subTest(crlf=crlf):
                raw = source.encode("latin-1")
                if crlf:
                    raw = raw.replace(b"\n", b"\r\n")
                self.assertNotIn(b"\0", raw)
                namespace = {}
                exec(compile(raw, "<chunked-byte-pairs>", "exec"), namespace)
                chunks = namespace["_BUNDLE_PAYLOAD"]
                self.assertIsInstance(chunks, tuple)
                self.assertTrue(all(2 < len(chunk) <= width + 2 for chunk in chunks))
                self.assertEqual(b"".join(chunk[1:-1].encode("latin-1") for chunk in chunks), original)
                self.assertEqual(namespace["_BUNDLE_INFO"]["format_version"], 4)
                self.assertEqual(namespace["_BUNDLE_INFO"]["version"], "한글")

    def test_relocated_normal_startup_and_managed_child_restore_exact_bytes(self):
        data = random.Random(24).randbytes(800003)
        archive = fixture_archive({"vendor/fixturepkg/all-bytes.bin": data})
        for compression in ("deflate", "lzma"):
            with self.subTest(compression=compression), tempfile.TemporaryDirectory(prefix="chunked ") as directory:
                root = Path(directory)
                artifact, cache = root / "one relocated file.py", root / "private cache"
                source = build.render_artifact(TEMPLATE, archive, "fixture", compression=compression,
                                               payload_encoding="latin1-chunks")
                artifact.write_bytes(source.encode("latin-1"))
                info, decoded = size_report.read_payload(artifact)
                self.assertEqual(decoded, archive)
                self.assertEqual(info["format_version"], 4)
                environment = {"HOME": str(root), "USERPROFILE": str(root), "HERMES_HOME": str(root / "hermes")}
                for name in ("SYSTEMROOT", "WINDIR"):
                    if name in os.environ:
                        environment[name] = os.environ[name]
                result = subprocess.run([sys.executable, str(artifact), "--cache-dir", str(cache),
                                         "child", "한글", "quoted path"], cwd=root, env=environment,
                                        capture_output=True, text=True, encoding="utf-8", timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertTrue(report["isolated"] and report["no_site"] and report["utf8"])
                self.assertEqual(report["argv"], ["한글", "quoted path"])
                files = list(cache.rglob("all-bytes.bin"))
                self.assertEqual(len(files), 1)
                self.assertEqual(files[0].read_bytes(), data)

    def test_invalid_chunk_shapes_and_checksums_precede_any_decode_file(self):
        archive = fixture_archive({"vendor/fixturepkg/data.bin": random.Random(25).randbytes(300003)})
        source = build.render_artifact(TEMPLATE, archive, "fixture", payload_encoding="latin1-chunks")
        changes = {
            "list": lambda chunks: list(chunks),
            "string": lambda chunks: "".join(chunks),
            "missing": lambda chunks: chunks[:-1],
            "reordered": lambda chunks: tuple(reversed(chunks)),
            "oversized": lambda chunks: (chunks[0] + "x", *chunks[1:]),
            "delimiter": lambda chunks: ("x" + chunks[0][1:], *chunks[1:]),
            "unicode": lambda chunks: (chunks[0][:2] + "\u1000" + chunks[0][3:], *chunks[1:]),
            "checksum": lambda chunks: (chunks[0][:2] + chr(ord(chunks[0][2]) ^ 1) + chunks[0][3:], *chunks[1:]),
            "many": lambda chunks: ("\nx\n",) * 4097,
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                namespace = {"__name__": "owned_chunk_validation", "__file__": "fixture.py"}
                exec(compile(source.encode("latin-1"), "<invalid-chunks>", "exec"), namespace)
                namespace["_BUNDLE_PAYLOAD"] = change(namespace["_BUNDLE_PAYLOAD"])
                with mock.patch.object(namespace["tempfile"], "TemporaryFile", side_effect=AssertionError("unverified input")):
                    with self.assertRaises(namespace["BundleError"]):
                        namespace["_decode_archive"]()

    def test_truncated_and_concatenated_xz_fail_before_cache_creation(self):
        archive = fixture_archive()
        info, payload = build.encode_payload(archive, "fixture", "lzma")
        for malformed in (payload[:-1], payload + payload):
            with self.subTest(size=len(malformed)), tempfile.TemporaryDirectory(prefix="chunk-refusal-") as directory:
                root = Path(directory)
                artifact, cache = root / "hermes.py", root / "must-not-exist"
                source = build._render_encoded(TEMPLATE, info, malformed, payload_encoding="latin1-chunks")
                artifact.write_bytes(source.encode("latin-1"))
                result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", str(artifact),
                                         "--cache-dir", str(cache)], cwd=root, capture_output=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"compressed bundle", result.stderr)
                self.assertFalse(cache.exists())
