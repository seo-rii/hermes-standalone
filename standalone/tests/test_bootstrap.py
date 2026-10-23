"""Exercise the real generated launcher in fresh, isolated subprocesses."""

import base64
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile


STANDALONE = Path(__file__).resolve().parents[1]
TEMPLATE = (STANDALONE / "bootstrap.py").read_text(encoding="utf-8")
CLI = b"""
import importlib.metadata
import importlib.resources
import json
from pathlib import Path
import subprocess
import sys
from standalone.runtime import child_command

def main(argv, context):
    if argv and argv[0] == "exit-code":
        print("completed before launcher exit")
        return int(argv[1])
    if argv and argv[0] == "unicode":
        text = chr(0x00a9) + " caf" + chr(0x00e9) + " " + chr(0xd55c) + chr(0xae00)
        print(text)
        print(text, file=sys.stderr)
        return 0
    if argv and argv[0] == "child":
        result = subprocess.run(
            child_command(context, "fixture_worker", argv[1:]),
            text=True, capture_output=True, check=True
        )
        print(result.stdout, end="")
        return 0
    print(json.dumps({
        "argv": argv,
        "artifact": context["artifact"],
        "root": context["root"],
        "cache_dir": context["cache_dir"],
        "isolated": bool(sys.flags.isolated),
        "no_site": bool(sys.flags.no_site),
        "no_bytecode": bool(sys.flags.dont_write_bytecode),
        "utf8": bool(sys.flags.utf8_mode),
        "sys_path": sys.path,
        "resource": importlib.resources.files("fixturepkg").joinpath("data.txt").read_text(),
        "version": importlib.metadata.version("fixturepkg"),
    }))
    return 0
"""
WORKER = b"""
import importlib.resources
import json
import sys
print(json.dumps({
    "argv": sys.argv[1:],
    "isolated": bool(sys.flags.isolated),
    "no_site": bool(sys.flags.no_site),
    "no_bytecode": bool(sys.flags.dont_write_bytecode),
    "utf8": bool(sys.flags.utf8_mode),
    "sys_path": sys.path,
    "resource": importlib.resources.files("fixturepkg").joinpath("data.txt").read_text(),
    "unicode": chr(0xd55c) + chr(0xae00) + " " + chr(0x00a9),
}, ensure_ascii=False))
"""


def fixture_archive(extra=None, modes=None, manifest_edit=None):
    files = {
        "app/standalone/__init__.py": b"",
        "app/standalone/cli.py": CLI,
        "app/standalone/runtime.py": (STANDALONE / "runtime.py").read_bytes(),
        "app/fixture_worker.py": WORKER,
        "vendor/fixturepkg/__init__.py": b"",
        "vendor/fixturepkg/data.txt": b"embedded resource",
        "vendor/fixturepkg-1.2.3.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: fixturepkg\nVersion: 1.2.3\n"
        ),
    }
    files.update(extra or {})
    manifest = {
        "format_version": 1,
        "project": "hermes-standalone",
        "version": "fixture",
        "profile": "fixture",
        "upstream": {},
        "runtime": {"bootstrap_python_min": "3.11", "hermes_python": "3.14"},
        "entrypoint": "standalone.cli:main",
        "internal_modules": ["fixture_worker"],
        "files": [
            {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            for name, data in sorted(files.items())
        ],
        "distributions": [],
        "capabilities": {},
    }
    if manifest_edit:
        manifest_edit(manifest)
    files["bundle-manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (modes or {}).get(name, stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return stream.getvalue()


def write_artifact(
    path, archive, expected_archive=None, *, compression="deflate", payload_override=None, info_edit=None
):
    expected_archive = archive if expected_archive is None else expected_archive
    info = {
        "format_version": 1,
        "version": "fixture",
        "sha256": hashlib.sha256(expected_archive).hexdigest(),
        "size": len(expected_archive),
    }
    payload = archive
    if compression == "lzma":
        payload = lzma.compress(archive, format=lzma.FORMAT_XZ, preset=6)
        if payload_override is not None:
            payload = payload_override
        info.update({
            "format_version": 2,
            "compression": "lzma",
            "payload_size": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        })
    if info_edit:
        info_edit(info)
    path.write_text(
        TEMPLATE.replace("__HERMES_BUNDLE_INFO_LITERAL__", repr(info)).replace(
            "__HERMES_BUNDLE_PAYLOAD_LITERAL__", repr(base64.b85encode(payload).decode())
        ).replace("__HERMES_WINDOWS_ACL_SOURCE__", repr((STANDALONE / "windows_acl.py").read_text(encoding="utf-8")))
         .replace('__HERMES_DICTIONARY_PACK_SOURCE__', (STANDALONE / 'dictionary_pack.py').read_text(encoding='utf8')),
        encoding="utf-8",
    )


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hermes bootstrap test ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.artifact = self.directory / "hermes fixture.py"
        self.cache = self.directory / "private cache"
        write_artifact(self.artifact, fixture_archive())

    def command(self, *args, isolated=True):
        return [
            sys.executable,
            *(["-I", "-S"] if isolated else []),
            str(self.artifact),
            "--cache-dir",
            str(self.cache),
            *args,
        ]

    def run_artifact(self, *args, isolated=True):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(self.directory / "untrusted import path")
        return subprocess.run(
            self.command(*args, isolated=isolated),
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=self.directory,
            env=environment,
            timeout=30,
        )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_one_file_runs_after_relocation_with_resources_and_metadata(self):
        moved = self.directory / "relocated directory"
        moved.mkdir()
        relocated = moved / "one file.py"
        shutil.move(self.artifact, relocated)
        self.artifact = relocated
        report = self.assert_success(self.run_artifact("diagnostic"))
        self.assertEqual(report["artifact"], str(relocated))
        self.assertEqual(report["resource"], "embedded resource")
        self.assertEqual(report["version"], "1.2.3")
        self.assertEqual(report["argv"], ["diagnostic"])
        self.assertTrue(report["isolated"] and report["no_site"] and report["no_bytecode"])
        self.assertTrue(report["utf8"])
        self.assertFalse(any("site-packages" in item for item in report["sys_path"]))
        self.assertFalse(any(self.cache.rglob("__pycache__")))

    def test_plain_launch_reenters_isolated_interpreter(self):
        report = self.assert_success(self.run_artifact(isolated=False))
        self.assertTrue(report["isolated"] and report["no_site"] and report["no_bytecode"])
        self.assertNotIn(str(self.directory / "untrusted import path"), report["sys_path"])

    def test_model_byteplanes_restore_original_bytes_and_verify_decoded_cache(self):
        name = 'app/standalone/assets/piper/hebrew/nakdimon.onnx'
        original = bytes(range(256)) * 131 + b'odd'
        encoded = b''.join(original[index::4] for index in range(4))
        def describe(manifest):
            row = next(item for item in manifest['files'] if item['path'] == name)
            row.update(sha256=hashlib.sha256(original).hexdigest(), encoding='byteplanes4')
        write_artifact(self.artifact, fixture_archive({name: encoded}, manifest_edit=describe))
        report = self.assert_success(self.run_artifact())
        path = Path(report['root']) / name
        self.assertEqual(path.read_bytes(), original)
        self.assert_success(self.run_artifact())
        path.write_bytes(encoded)
        result = self.run_artifact()
        self.assertEqual(result.returncode, 2)
        self.assertIn('cache integrity', result.stderr)
        self.assertEqual(path.read_bytes(), encoded)

    def test_model_encoding_refuses_unknown_targets_corruption_and_decoded_executables(self):
        approved = 'app/standalone/assets/piper/tashkeel/model.onnx'
        for name, original, encoding, digest, message in (
            (approved, b'valid-data', 'unknown-codec', None, 'file encoding'),
            ('app/standalone/encoded.py', b'code', 'byteplanes4', None, 'file encoding'),
            (approved, b'valid-data', 'sudachi-v1-blocks', None, 'file encoding'),
            (approved, b'valid-data', 'byteplanes4', '0' * 64, 'restored model integrity'),
            (approved, b'\x7fELF' + bytes(range(24)), 'byteplanes4', None, 'native or executable'),
        ):
            with self.subTest(name=name, encoding=encoding, message=message):
                encoded = b''.join(original[index::4] for index in range(4))
                def describe(manifest):
                    row = next(item for item in manifest['files'] if item['path'] == name)
                    row.update(sha256=digest or hashlib.sha256(original).hexdigest(), encoding=encoding)
                write_artifact(self.artifact, fixture_archive({name: encoded}, manifest_edit=describe))
                result = self.run_artifact()
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
                if self.cache.exists():
                    self.assertFalse(list(self.cache.iterdir()))

    def test_dictionary_encoding_restores_original_bytes_before_cache_publication(self):
        from standalone.dictionary_pack import transform_dictionary
        from standalone.tests.test_dictionary_pack import fixture_dictionary
        name = 'app/standalone/local_piper_japanese/data/sudachi/system.dic'
        original = fixture_dictionary()
        encoded = transform_dictionary(original)
        def describe(manifest):
            item = next(row for row in manifest['files'] if row['path'] == name)
            item.update(sha256=hashlib.sha256(original).hexdigest(), encoding='sudachi-v1-blocks')
        write_artifact(self.artifact, fixture_archive({name: encoded}, manifest_edit=describe))
        report = self.assert_success(self.run_artifact())
        path = Path(report['root']) / name
        self.assertEqual(path.read_bytes(), original)
        self.assert_success(self.run_artifact())
        path.write_bytes(encoded)
        result = self.run_artifact()
        self.assertEqual(result.returncode, 2)
        self.assertIn('cache integrity', result.stderr)
        self.assertEqual(path.read_bytes(), encoded)

    def test_solid_payload_restores_exact_sources_and_preserves_child_reentry(self):
        archive = fixture_archive()
        write_artifact(self.artifact, archive, compression="lzma")
        report = self.assert_success(self.run_artifact())
        self.assertEqual(Path(report["root"]).name, hashlib.sha256(archive).hexdigest())
        self.assertEqual((Path(report["root"]) / "app/standalone/cli.py").read_bytes(), CLI)
        self.assertEqual(report["resource"], "embedded resource")
        child = self.assert_success(self.run_artifact("child", "two words"))
        self.assertEqual(child["argv"], ["two words"])
        self.assertTrue(child["isolated"] and child["no_site"])

    def test_cache_identity_uses_decoded_zip_not_transport_compression(self):
        archive = fixture_archive()
        write_artifact(self.artifact, archive)
        first = self.assert_success(self.run_artifact())
        manifest = Path(first["root"]) / "bundle-manifest.json"
        original_time = manifest.stat().st_mtime_ns
        write_artifact(self.artifact, archive, compression="lzma")
        second = self.assert_success(self.run_artifact())
        self.assertEqual(first["root"], second["root"])
        self.assertEqual(manifest.stat().st_mtime_ns, original_time)
        self.assertEqual(len(list(self.cache.iterdir())), 1)

    def test_compressed_payload_tampering_and_decoded_hash_mismatch_fail_before_cache_creation(self):
        for field, expected in (
            ("payload_sha256", "bundle integrity"),
            ("sha256", "decoded archive integrity"),
        ):
            with self.subTest(field=field):
                write_artifact(
                    self.artifact, fixture_archive(), compression="lzma",
                    info_edit=lambda info: info.update({field: "0" * 64}),
                )
                result = self.run_artifact()
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(expected, result.stderr)
                self.assertFalse(self.cache.exists())

    def test_compressed_stream_must_end_once_without_truncation_or_trailing_bytes(self):
        archive = fixture_archive()
        payload = lzma.compress(archive)
        for changed, expected in (
            (payload[:-8], "truncated"),
            (payload + b"trailing bytes", "trailing data"),
            (payload + payload, "concatenated streams"),
        ):
            with self.subTest(expected=expected):
                write_artifact(
                    self.artifact, archive, compression="lzma", payload_override=changed
                )
                result = self.run_artifact()
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(expected, result.stderr)
                self.assertFalse(self.cache.exists())

    def test_compressed_output_is_bounded_by_declared_and_absolute_size(self):
        archive = fixture_archive()
        for size, expected in (
            (len(archive) - 1, "exceeds declared archive size"),
            (512 * 1024 * 1024 + 1, "decoded archive integrity check failed"),
            (1024 * 1024 * 1024 + 1, "invalid bundle metadata"),
        ):
            with self.subTest(size=size):
                write_artifact(
                    self.artifact, archive, compression="lzma",
                    info_edit=lambda info: info.update({"size": size}),
                )
                result = self.run_artifact()
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(expected, result.stderr)
                self.assertFalse(self.cache.exists())

    def test_decoder_enforces_memory_budget(self):
        write_artifact(self.artifact, fixture_archive(), compression="lzma")
        namespace = {"__name__": "bootstrap_memory_fixture", "__file__": str(self.artifact)}
        exec(compile(self.artifact.read_text(), str(self.artifact), "exec"), namespace)
        namespace["_MAX_LZMA_MEMORY"] = 1024
        with self.assertRaisesRegex(namespace["BundleError"], "memory limit"):
            namespace["_decode_archive"]()
        self.assertFalse(self.cache.exists())

    def test_waiting_reentry_preserves_output_arguments_and_exit_status(self):
        # Exercise the Windows launcher strategy with real child processes on
        # this host; this is not a Windows OS/ACL compatibility test.
        source = self.artifact.read_text(encoding="utf-8")
        self.artifact.write_text(
            source.replace('if os.name == "nt":', "if True:", 1),
            encoding="utf-8",
        )
        report = self.assert_success(self.run_artifact("two words", isolated=False))
        self.assertEqual(report["argv"], ["two words"])
        self.assertTrue(report["isolated"] and report["no_site"] and report["no_bytecode"])
        result = self.run_artifact("exit-code", "17", isolated=False)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(result.stdout, "completed before launcher exit\n")

    def test_module_child_reenters_artifact_and_preserves_arguments(self):
        report = self.assert_success(self.run_artifact("child", "two words", "--cache-dir", "literal"))
        self.assertEqual(report["argv"], ["two words", "--cache-dir", "literal"])
        self.assertEqual(report["resource"], "embedded resource")
        self.assertEqual(report["unicode"], "한글 ©")
        self.assertTrue(report["isolated"] and report["no_site"] and report["no_bytecode"])
        self.assertTrue(report["utf8"])
        self.assertFalse(any("site-packages" in item for item in report["sys_path"]))

    def test_already_isolated_ascii_locale_reenters_utf8_for_stdout_and_stderr(self):
        environment = dict(os.environ)
        environment["LC_ALL"] = "C"
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            "utf8=0",
            str(self.artifact),
            "--cache-dir",
            str(self.cache),
        ]
        result = subprocess.run(
            [*command, "unicode"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "© café 한글\n")
        self.assertEqual(result.stderr, "© café 한글\n")
        child_result = subprocess.run(
            [*command, "child"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
            timeout=30,
        )
        report = self.assert_success(child_result)
        self.assertEqual(report["unicode"], "한글 ©")
        self.assertTrue(report["utf8"])

    def test_arbitrary_internal_module_is_rejected(self):
        result = self.run_artifact("--internal-module", "os")
        self.assertEqual(result.returncode, 2)
        self.assertIn("allowlist", result.stderr)

    def test_corrupt_archive_bytes_fail_before_cache_creation(self):
        original = fixture_archive()
        changed = bytearray(original)
        changed[len(changed) // 2] ^= 1
        write_artifact(self.artifact, bytes(changed), expected_archive=original)
        result = self.run_artifact()
        self.assertEqual(result.returncode, 2)
        self.assertIn("bundle integrity", result.stderr)
        self.assertFalse(self.cache.exists())

    def test_traversal_and_native_payloads_are_rejected(self):
        cases = [
            ({"../escape.py": b"raise RuntimeError()"}, {}, "unsafe archive path"),
            ({"app/escape.py": b"target"}, {"app/escape.py": stat.S_IFLNK | 0o777}, "plain data file"),
            ({"vendor/native.so": b"native"}, {}, "native or executable"),
            ({"vendor/innocent.data": b"\x7fELF" + b"\x00" * 16}, {}, "native or executable"),
            ({"vendor/innocent.data": b"MZ" + b"\x00" * 16}, {}, "native or executable"),
        ]
        for extra, modes, message in cases:
            with self.subTest(extra=tuple(extra), modes=modes):
                write_artifact(self.artifact, fixture_archive(extra, modes))
                result = self.run_artifact()
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse((self.directory / "escape.py").exists())
                if self.cache.exists():
                    self.assertEqual(list(self.cache.iterdir()), [])

    def test_case_colliding_paths_are_rejected(self):
        write_artifact(
            self.artifact,
            fixture_archive({"vendor/FixturePkg/other.txt": b"case collision"}),
        )
        result = self.run_artifact()
        self.assertEqual(result.returncode, 2)
        self.assertIn("case-colliding", result.stderr)

    def test_invalid_manifest_file_hash_is_rejected(self):
        def break_hash(manifest):
            manifest["files"][0]["sha256"] = "0" * 64

        write_artifact(self.artifact, fixture_archive(manifest_edit=break_hash))
        result = self.run_artifact()
        self.assertEqual(result.returncode, 2)
        self.assertIn("archive file integrity", result.stderr)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_foreign_project_is_rejected_before_cache_creation(self):
        for compression in ("deflate", "lzma"):
            with self.subTest(compression=compression):
                write_artifact(
                    self.artifact,
                    fixture_archive(manifest_edit=lambda manifest: manifest.update(project="other-project")),
                    compression=compression,
                )
                result = self.run_artifact()
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("invalid or incompatible bundle manifest", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(self.cache.exists())

    def test_cache_is_reused_and_tampering_fails_without_deleting_data(self):
        first = self.assert_success(self.run_artifact())
        root = Path(first["root"])
        manifest_path = root / "bundle-manifest.json"
        original_mtime = manifest_path.stat().st_mtime_ns
        second = self.assert_success(self.run_artifact())
        self.assertEqual(second["root"], first["root"])
        self.assertEqual(manifest_path.stat().st_mtime_ns, original_mtime)
        target = root / "vendor/fixturepkg/data.txt"
        target.write_text("tampered resource", encoding="utf-8")
        result = self.run_artifact()
        self.assertEqual(result.returncode, 2)
        self.assertIn("cache integrity", result.stderr)
        self.assertEqual(target.read_text(encoding="utf-8"), "tampered resource")

    def test_unexpected_cache_file_is_rejected(self):
        report = self.assert_success(self.run_artifact())
        path = Path(report["root"]) / "app/unlisted.py"
        path.write_text("pass", encoding="utf-8")
        path.chmod(0o600)
        result = self.run_artifact()
        self.assertEqual(result.returncode, 2)
        self.assertIn("unexpected cache file", result.stderr)

    def test_ast_proof_requires_its_recorded_interpreter_before_cache_creation(self):
        current = {"python_implementation": sys.implementation.name,
                   "python_version": ".".join(map(str, sys.version_info[:3]))}
        for toolchain in (None, {**current, "python_version": "0.0.0"},
                          {**current, "python_implementation": "different-interpreter"}):
            with self.subTest(toolchain=toolchain):
                write_artifact(self.artifact, fixture_archive(manifest_edit=lambda manifest:
                    manifest.update(source_transform={"mode": "ast", "toolchain": toolchain})))
                result = self.run_artifact()
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("exact verified Python toolchain", result.stderr)
                self.assertFalse(self.cache.exists())
        write_artifact(self.artifact, fixture_archive(manifest_edit=lambda manifest:
            manifest.update(source_transform={"mode": "ast", "toolchain": current})))
        self.assert_success(self.run_artifact())

    @unittest.skipUnless(os.name == "posix", "requires POSIX symlinks and permission bits")
    def test_symlink_and_nonprivate_cache_roots_are_rejected(self):
        destination = self.directory / "unrelated"
        destination.mkdir(mode=0o700)
        self.cache.symlink_to(destination, target_is_directory=True)
        result = self.run_artifact()
        self.assertEqual(result.returncode, 2)
        self.assertIn("link or non-directory", result.stderr)
        self.assertEqual(list(destination.iterdir()), [])
        self.cache.unlink()
        self.cache.mkdir(mode=0o755)
        self.cache.chmod(0o755)
        result = self.run_artifact()
        self.assertEqual(result.returncode, 2)
        self.assertIn("private", result.stderr)

    @unittest.skipUnless(os.name == "posix", "requires POSIX symlinks")
    def test_symlink_ancestor_is_canonicalized_without_following_cache_root_links(self):
        actual_parent = self.directory / "real parent"
        actual_parent.mkdir(mode=0o700)
        linked_parent = self.directory / "linked parent"
        linked_parent.symlink_to(actual_parent, target_is_directory=True)
        self.cache = linked_parent / "nested parent" / "cache"
        report = self.assert_success(self.run_artifact())
        canonical_cache = actual_parent / "nested parent" / "cache"
        self.assertEqual(report["cache_dir"], str(canonical_cache))
        self.assertEqual(Path(report["root"]).parent, canonical_cache)
        self.assertEqual(canonical_cache.stat().st_mode & 0o777, 0o700)
        # A different selected root which is itself a link must still fail.
        self.cache = linked_parent / "cache link"
        self.cache.symlink_to(canonical_cache, target_is_directory=True)
        result = self.run_artifact()
        self.assertEqual(result.returncode, 2)
        self.assertIn("link or non-directory", result.stderr)

    @unittest.skipUnless(os.name == "posix", "requires POSIX symlinks")
    def test_symlink_inside_existing_cache_is_rejected(self):
        report = self.assert_success(self.run_artifact())
        target = Path(report["root"]) / "vendor/fixturepkg/data.txt"
        target.unlink()
        outside = self.directory / "unrelated.txt"
        outside.write_text("embedded resource", encoding="utf-8")
        target.symlink_to(outside)
        result = self.run_artifact()
        self.assertEqual(result.returncode, 2)
        self.assertIn("not a regular file", result.stderr)
        self.assertEqual(outside.read_text(encoding="utf-8"), "embedded resource")

    def test_concurrent_first_starts_publish_one_complete_cache(self):
        children = [
            subprocess.Popen(
                self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            for _ in range(4)
        ]
        reports = []
        try:
            for child in children:
                stdout, stderr = child.communicate(timeout=30)
                self.assertEqual(child.returncode, 0, stderr)
                reports.append(json.loads(stdout))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()
        self.assertEqual(len({report["root"] for report in reports}), 1)
        self.assertEqual([path.name for path in self.cache.iterdir()], [Path(reports[0]["root"]).name])

    def test_cache_flag_is_consumed_only_before_the_command(self):
        report = self.assert_success(self.run_artifact("command", "--cache-dir", "argument"))
        self.assertEqual(report["argv"], ["command", "--cache-dir", "argument"])
        result = self.run_artifact("--cache-dir", "duplicate")
        self.assertEqual(result.returncode, 2)
        self.assertIn("only once", result.stderr)


if __name__ == "__main__":
    unittest.main()
