"""All-byte Python literals and real isolated bootstrap parity, including CRLF."""

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from standalone import build, size_report
from standalone.tests.test_bootstrap import TEMPLATE, fixture_archive


class PayloadLiteralTests(unittest.TestCase):
    def test_all_bytes_escape_boundaries_and_template_tokens_round_trip(self):
        template = ("#!/usr/bin/env python3\n"
                    "_BUNDLE_INFO = __HERMES_BUNDLE_INFO_LITERAL__\n"
                    "_BUNDLE_PAYLOAD = __HERMES_BUNDLE_PAYLOAD_LITERAL__\n"
                    "_WINDOWS_ACL_SOURCE = __HERMES_WINDOWS_ACL_SOURCE__\n")
        pairs = b"".join(bytes((first, second)) for first in range(256) for second in range(256))
        quotes = b"".join(b"\\" * slashes + b"'" * count + b"\0" + b"0123456789\n"
                           for slashes in range(9) for count in range(33))
        data = (pairs + quotes + bytes(range(256)) * 3 + b"\x000123456789\r\n\\\n'''\\'\""
                + b"__HERMES_WINDOWS_ACL_SOURCE__\n__HERMES_BUNDLE_INFO_LITERAL__"
                + random.Random(55).randbytes(32768))
        info = {"format_version": 1, "version": "한글", "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data)}
        for ending in (b"", b"'", b"''", b"'''", b"\\", b"\0", b"\r", b"\n"):
            with self.subTest(ending=ending):
                for crlf in (False, True):
                    rendered = build._render_encoded(template, info, data + ending, payload_encoding="latin1")
                    source = rendered.encode("latin-1")
                    if crlf:
                        source = source.replace(b"\n", b"\r\n")
                    self.assertNotIn(b"\0", source)
                    scope = {}
                    exec(compile(source, "<all-byte-literal>", "exec"), scope)
                    self.assertEqual(scope["_BUNDLE_PAYLOAD"][1:-1].encode("latin-1"), data + ending)
                    self.assertEqual(scope["_BUNDLE_INFO"]["version"], "한글")
                    self.assertEqual(scope["_BUNDLE_INFO"]["format_version"], 3)
                    self.assertEqual(scope["_BUNDLE_INFO"]["payload_size"], len(data + ending))

    def test_real_literal_launcher_restores_exact_resources_and_reenters(self):
        data = bytes(range(256)) * 256
        original = fixture_archive({"vendor/fixturepkg/all-bytes.bin": data})
        for compression in ("deflate", "lzma"):
            with self.subTest(compression=compression), tempfile.TemporaryDirectory(prefix="literal ") as temporary:
                root = Path(temporary)
                artifact, cache = root / "hermes literal.py", root / "private cache"
                source = build.render_artifact(TEMPLATE, original, "fixture", compression=compression,
                                               payload_encoding="latin1")
                artifact.write_bytes(source.encode("latin-1"))
                info, decoded = size_report.read_payload(artifact)
                self.assertEqual(decoded, original)
                self.assertEqual(info["format_version"], 3)
                environment = {"HOME": str(root), "USERPROFILE": str(root), "HERMES_HOME": str(root / "hermes")}
                for name in ("SYSTEMROOT", "WINDIR"):
                    if name in os.environ:
                        environment[name] = os.environ[name]
                command = [sys.executable, "-I", "-S", "-B", "-X", "utf8", str(artifact), "--cache-dir", str(cache)]
                for arguments in ([], ["child", "한글", "quoted path"]):
                    result = subprocess.run(command + arguments, env=environment, capture_output=True,
                                            text=True, encoding="utf-8", timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertTrue(report["isolated"] and report["no_site"] and report["utf8"])
                    self.assertEqual(report["argv"], arguments[1:] if arguments else [])
                resources = list(cache.rglob("all-bytes.bin"))
                self.assertEqual(len(resources), 1)
                self.assertEqual(resources[0].read_bytes(), data)

    def test_literal_metadata_corruption_is_rejected_before_cache_creation(self):
        original = fixture_archive()
        source = build.render_artifact(TEMPLATE, original, "fixture", payload_encoding="latin1")
        mutations = [source.replace("'payload_encoding': 'latin1'", "'payload_encoding': 'unknown'"),
                     source.replace("'payload_size': ", "'payload_size': 1 + ")]
        for corrupted in mutations:
            with tempfile.TemporaryDirectory(prefix="literal-refusal-") as temporary:
                root = Path(temporary)
                artifact, cache = root / "hermes.py", root / "must-not-exist"
                artifact.write_bytes(corrupted.encode("latin-1"))
                result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", str(artifact),
                                         "--cache-dir", str(cache)], capture_output=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"literal payload", result.stderr)
                self.assertFalse(cache.exists())
