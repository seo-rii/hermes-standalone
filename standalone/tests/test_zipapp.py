"""One-file ZIP applications retain the original isolation and integrity checks."""
import io
import json
import os
from pathlib import Path
import random
import runpy
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile

from standalone import build, size_report
from standalone.tests.test_bootstrap import TEMPLATE, fixture_archive


def repack(raw, *, entries=None, change=None, comment=b""):
    result = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as old, zipfile.ZipFile(result, "w") as new:
        values = [(item.filename, old.read(item)) for item in old.infolist()]
        values = values if entries is None else entries(values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name, content in values:
                item = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                item.create_system = 3
                item.external_attr = 0o100644 << 16
                if change:
                    change(item)
                new.writestr(item, content)
        new.comment = comment
    return result.getvalue()


class ZipappTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="zip application 한글 ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifact = self.root / "relocated hermes.pyz"
        self.cache = self.root / "private cache"
        self.environment = {"HOME": str(self.root), "USERPROFILE": str(self.root),
                            "HERMES_HOME": str(self.root / "untouched profile")}
        for name in ("SYSTEMROOT", "WINDIR"):
            if name in os.environ:
                self.environment[name] = os.environ[name]

    def launch(self, *args, isolated=True):
        flags = ["-I", "-S", "-B", "-X", "utf8"] if isolated else []
        return subprocess.run([sys.executable, *flags, str(self.artifact), "--cache-dir", str(self.cache), *args],
                              cwd=self.root, env=self.environment, capture_output=True, text=True,
                              encoding="utf-8", timeout=20)

    def test_deterministic_bytes_relocation_normal_launch_and_managed_child(self):
        data = random.Random(61).randbytes(300003)
        original = fixture_archive({"vendor/fixturepkg/data.bin": data})
        for compression in ("lzma", "deflate"):
            with self.subTest(compression=compression):
                raw = build.render_zipapp(TEMPLATE, original, "fixture", compression=compression)
                self.assertEqual(raw, build.render_zipapp(TEMPLATE, original, "fixture", compression=compression))
                self.artifact.write_bytes(raw)
                info, decoded = size_report.read_payload(self.artifact)
                self.assertEqual(info["format_version"], 5)
                self.assertEqual(decoded, original)
                result = self.launch("one", "한글", isolated=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["artifact"], str(self.artifact.resolve()))
                self.assertTrue(report["isolated"] and report["no_site"] and report["utf8"])
                self.assertNotIn(str(self.artifact), report["sys_path"])
                child = self.launch("child", "한글", "quoted path")
                self.assertEqual(child.returncode, 0, child.stderr)
                worker = json.loads(child.stdout)
                self.assertEqual(worker["argv"], ["한글", "quoted path"])
                self.assertTrue(worker["isolated"] and worker["no_site"] and worker["no_bytecode"])
                self.assertNotIn(str(self.artifact), worker["sys_path"])
                self.assertEqual(next(self.cache.rglob("data.bin")).read_bytes(), data)
                self.assertFalse(Path(self.environment["HERMES_HOME"]).exists())

    def test_runpy_reads_real_archive_and_closes_owned_zip(self):
        original = fixture_archive()
        self.artifact.write_bytes(build.render_zipapp(TEMPLATE, original, "fixture"))
        namespace = runpy.run_path(str(self.artifact), run_name="owned_zipapp_observation")
        archive, manifest, raw_manifest, files = namespace["_read_archive"]()
        source = archive._source
        self.assertEqual(archive.read("vendor/fixturepkg/data.txt"), b"embedded resource")
        self.assertTrue(raw_manifest and manifest["files"] and files)
        archive.close()
        self.assertTrue(source.closed)

    def test_actual_child_command_retains_exact_module_identity(self):
        worker = b'''import json,sys
from standalone.runtime import bundle_module_argv,current_bundle
print(json.dumps({'identity':bundle_module_argv(sys.orig_argv),'artifact':current_bundle()['artifact']}))
'''
        original = fixture_archive({"app/fixture_worker.py": worker})
        self.artifact.write_bytes(build.render_zipapp(TEMPLATE, original, "fixture"))
        result = self.launch("child", "한글", "space argument")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["identity"], ["fixture_worker", ["한글", "space argument"]])
        self.assertEqual(report["artifact"], str(self.artifact.resolve()))

    def test_relative_path_normal_launch_removes_outer_import_path(self):
        raw = build.render_zipapp(TEMPLATE, fixture_archive(), "fixture")
        self.artifact.write_bytes(raw)
        command = [sys.executable, "./" + self.artifact.name, "--cache-dir", str(self.cache)]
        result = subprocess.run(command, cwd=self.root, env=self.environment, capture_output=True,
                                text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["isolated"] and report["no_site"] and report["utf8"])
        self.assertEqual(report["artifact"], str(self.artifact.resolve()))
        self.assertNotIn(str(self.artifact), report["sys_path"])
        marker = self.root / "relative-shadow-must-not-run"
        malicious = ("open(" + repr(str(marker)) + ", 'w').write('unverified')\n").encode()
        self.artifact.write_bytes(repack(raw, entries=lambda values: [*values, ("json.py", malicious)]))
        result = subprocess.run(command, cwd=self.root, env=self.environment, capture_output=True,
                                text=True, encoding="utf-8", timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid ZIP application members", result.stderr)
        self.assertFalse(marker.exists())

    def test_outer_stdlib_shadow_modules_never_execute(self):
        raw = build.render_zipapp(TEMPLATE, fixture_archive(), "fixture")
        marker = self.root / "must-not-run"
        malicious = ("open(" + repr(str(marker)) + ", 'w').write('unverified module ran')\n").encode("utf-8")
        for module in ("json.py", "pathlib.py", "hashlib.py", "runpy.py"):
            with self.subTest(module=module):
                self.artifact.write_bytes(repack(raw, entries=lambda values: [*values, (module, malicious)]))
                result = self.launch()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid ZIP application members", result.stderr)
                self.assertFalse(marker.exists())
                self.assertFalse(self.cache.exists())
                with self.assertRaises(ValueError):
                    size_report.read_payload(self.artifact)

    def test_malformed_outer_members_are_rejected_before_cache_creation(self):
        raw = build.render_zipapp(TEMPLATE, fixture_archive(), "fixture")
        changes = {
            "duplicate": lambda: repack(raw, entries=lambda values: [*values, values[1]]),
            "missing": lambda: repack(raw, entries=lambda values: values[:1]),
            "compressed": lambda: repack(raw, change=lambda item: setattr(item, "compress_type", zipfile.ZIP_DEFLATED)),
            "link": lambda: repack(raw, change=lambda item: setattr(item, "external_attr", (stat.S_IFLNK | 0o600) << 16)),
            "extra": lambda: repack(raw, change=lambda item: setattr(item, "extra", b"\x01\x00\x00\x00")),
            "comment": lambda: repack(raw, comment=b"unexpected"),
        }
        for name, changed in changes.items():
            with self.subTest(case=name):
                self.artifact.write_bytes(changed())
                result = self.launch()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.cache.exists())
                with self.assertRaises((ValueError, zipfile.BadZipFile)):
                    size_report.read_payload(self.artifact)

    def test_payload_mutation_and_invalid_xz_keep_cache_absent(self):
        original = fixture_archive()
        raw = build.render_zipapp(TEMPLATE, original, "fixture")
        def changed(values):
            return [(name, content if name == "__main__.py" else content[:-1] + bytes([content[-1] ^ 1]))
                    for name, content in values]
        invalid = [repack(raw, entries=changed)]
        invalid.append(repack(raw, entries=lambda values: [(name, content if name == "__main__.py" else content + b"x")
                                                         for name, content in values]))
        info, payload = build.encode_payload(original, "fixture", "lzma")
        invalid.extend(build._render_zipapp(TEMPLATE, info, data) for data in (payload[:-1], payload + payload))
        for value in invalid:
            with self.subTest(bytes=len(value)):
                self.artifact.write_bytes(value)
                result = self.launch()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.cache.exists())
                with self.assertRaises(ValueError):
                    size_report.read_payload(self.artifact)

    def test_warm_cache_extra_and_modified_source_are_still_rejected(self):
        self.artifact.write_bytes(build.render_zipapp(TEMPLATE, fixture_archive(), "fixture"))
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        root = Path(json.loads(result.stdout)["root"])
        source = root / "app/fixture_worker.py"
        original = source.read_bytes()
        source.write_bytes(original + b"\n# changed\n")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cache integrity", result.stderr)
        source.write_bytes(original)
        extra = root / "app/extra.py"
        extra.write_bytes(b"raise AssertionError('unverified')\n")
        extra.chmod(0o600)
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected cache file", result.stderr)
