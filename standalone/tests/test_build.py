from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

from standalone import build
from standalone.tests.test_vendor import make_sdist, make_wheel


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cache = self.root / "wheel-cache"
        self.cache.mkdir()
        for name in build.RUNTIME_DIRECTORIES + build.RESOURCE_DIRECTORIES + ("standalone",):
            (self.root / name).mkdir()
        for name in build.REQUIRED_ROOT_RESOURCES:
            (self.root / name).write_text("resource\n")
        (self.root / "uv.lock").write_text("version = 1\npackage = []\n")
        # Source packaging must not require the upstream runtime's interpreter.
        (self.root / "agent" / "source.py").write_text("type NewSyntax[T] = T\n")
        (self.root / "standalone" / "__init__.py").write_text("")
        (self.root / "standalone" / "cli.py").write_text("def main(argv, context): return 0\n")
        (self.root / "standalone" / "selftest_worker.py").write_text("VALUE = 'selftest fixture'\n")
        (self.root / "standalone" / "agent_probe_worker.py").write_text("VALUE = 'agent probe fixture'\n")
        for module in build.INTERNAL_MODULES:
            path = self.root / (module.replace(".", "/") + ".py")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("VALUE = 'internal module fixture'\n")
        contract = {
            "format_version": 1, "version": "0.1.0.dev0", "profile": "foundation",
            "upstream": {"repository": "https://example.invalid/upstream", "commit": "a" * 40},
            "runtime": {"bootstrap_python_min": "3.11", "hermes_python": f"{sys.version_info.major}.{sys.version_info.minor}"},
            "entrypoint": "standalone.cli:main",
        }
        self.dependencies = {
            "format_version": 1,
            "source": {"upstream_commit": "a" * 40, "lockfile": "uv.lock", "target_python": f"{sys.version_info.major}.{sys.version_info.minor}"},
            "packages": [],
        }
        for name, data in (
            ("CONTRACT.json", contract),
            ("dependencies.json", self.dependencies),
            ("capabilities.json", {"agent": {"status": "blocked"}}),
        ):
            (self.root / "standalone" / name).write_text(json.dumps(data))
        (self.root / "standalone" / "bootstrap.py").write_text(
            "BUNDLE_INFO = __HERMES_BUNDLE_INFO_LITERAL__\n"
            "BUNDLE_PAYLOAD = __HERMES_BUNDLE_PAYLOAD_LITERAL__\n"
        )

    def add_dependency(self, *, name="example", additional=None):
        workers = [
            ("browser_harness/__init__.py", b""),
            ("browser_harness/run.py", b"VALUE = 'browser worker fixture'\n"),
            ("browser_harness/daemon.py", b"VALUE = 'browser daemon fixture'\n"),
            ("browser_harness/mcp_cli.py", b"VALUE = 'browser MCP fixture'\n"),
        ]
        additional = [*(workers if not self.dependencies["packages"] else []), *(additional or [])]
        package = make_wheel(self.cache, additional, name=name)
        self.dependencies["packages"].append(package)
        (self.root / "standalone" / "dependencies.json").write_text(json.dumps(self.dependencies))
        (self.root / "uv.lock").write_text(
            "version = 1\n" + "".join(
                f'[[package]]\nname = "{item["name"]}"\nversion = "1.0"\n'
                f'wheels = [{{ url = "{item["url"]}", hash = "sha256:{item["sha256"]}", size = {item["size"]} }}]\n'
                for item in self.dependencies["packages"]
            )
        )
        return package

    def test_ast_verification_refuses_interpreter_older_than_target(self):
        with mock.patch.object(build.sys, "version_info", (3, 8, 0)):
            with self.assertRaisesRegex(build.BuildError, "AST verification requires the target"):
                build.create_archive(self.root, self.cache, minify="ast")

    def test_build_is_identical_after_source_mtime_changes(self):
        self.add_dependency()
        first = self.root / "dist" / "first.py"
        second = self.root / "dist" / "second.py"
        first_result = build.build_bundle(self.root, first, self.cache)
        for path in self.root.rglob("*"):
            if path.is_file():
                os.utime(path, (1_700_000_000, 1_700_000_000))
        second_result = build.build_bundle(self.root, second, self.cache)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(first_result["sha256"], second_result["sha256"])
        self.assertNotIn("minify_report", first_result)
        self.assertFalse(list(first.parent.glob("*.minify.json")))
        scope = {}
        exec(compile(first.read_bytes(), str(first), "exec"), scope)
        payload = b"".join(chunk[1:-1].encode("latin-1") for chunk in scope["BUNDLE_PAYLOAD"])
        info = scope["BUNDLE_INFO"]
        self.assertEqual(info["format_version"], 4)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), info["payload_sha256"])
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64 * 1024 * 1024)
        decoded = decoder.decompress(payload)
        self.assertTrue(decoder.eof)
        self.assertFalse(decoder.unused_data)

        self.assertEqual(hashlib.sha256(decoded).hexdigest(), info["sha256"])
        with zipfile.ZipFile(io.BytesIO(decoded)) as archive:
            self.assertEqual(archive.namelist(), sorted(archive.namelist()))
            self.assertTrue(all(item.date_time == build.ZIP_TIMESTAMP for item in archive.infolist()))
            manifest = json.loads(archive.read("bundle-manifest.json"))
            self.assertEqual(
                manifest["internal_modules"],
                [
                    "standalone.selftest_worker", "standalone.agent_probe_worker", "standalone.entry_worker",
                    "hermes_cli.main", "gateway.run", "cli", "gateway.restart_watcher",
                    "hermes_cli.stderr_timestamp", "gateway.cgroup_cleanup", "gateway.systemd_stop_mark",
                "agent.managed_worker", "hermes_cli.kanban_worker_client", "tools.mcp_death_supervisor",
                "hermes_cli.plugin_host_child", "standalone.code_kernel_worker",
                "standalone.profile_worker", "standalone.local_piper_download_voices",
                "browser_harness.run", "browser_harness.daemon", "browser_harness.mcp_cli",
                "plugins.web.ddgs._search_worker",
                "plugins.google_meet.meet_bot",
                "acp_adapter.entry", "acp_adapter",
                ],
            )
            for entry in manifest["files"]:
                content = archive.read(entry["path"])
                self.assertEqual(entry["size"], len(content))
                self.assertEqual(entry["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(archive.read("vendor/example-1.0.dist-info/LICENSE"), b"Fixture license notice\n")
            self.assertEqual(archive.read("vendor/example/data.txt"), b"resource contents\n")

    def test_zipapp_build_preserves_every_inner_byte_and_is_deterministic(self):
        self.add_dependency()
        first, second = self.root / "dist/first.pyz", self.root / "dist/second.pyz"
        result = build.build_bundle(self.root, first, self.cache, container="zipapp")
        repeated = build.build_bundle(self.root, second, self.cache, container="zipapp")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(result["sha256"], repeated["sha256"])
        self.assertEqual(result["container"], "zipapp")
        self.assertEqual(result["payload_encoding"], "zip-member")
        with zipfile.ZipFile(first) as outer:
            self.assertEqual(outer.namelist(), ["__main__.py", "payload.bin"])
            namespace = {}
            exec(compile(outer.read("__main__.py"), "<fixture-zipapp>", "exec"), namespace)
            raw = lzma.decompress(outer.read("payload.bin"))
        self.assertEqual(namespace["BUNDLE_INFO"]["format_version"], 5)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), result["bundle_sha256"])
        self.assertEqual(len(raw), result["bundle_size"])
        with zipfile.ZipFile(io.BytesIO(raw)) as inner:
            self.assertEqual(inner.read("app/agent/source.py"), b"type NewSyntax[T] = T\n")

    def test_reviewed_source_distribution_is_bound_to_upstream_lock_and_inventory(self):
        package = make_sdist(self.cache)
        self.dependencies["packages"] = [package]
        (self.root / "standalone" / "dependencies.json").write_text(json.dumps(self.dependencies))
        (self.root / "uv.lock").write_text(
            f'version = 1\n[[package]]\nname = "example"\nversion = "1.0"\n'
            f'sdist = {{ url = "{package["url"]}", hash = "sha256:{package["sha256"]}", size = {package["size"]} }}\n'
        )
        archive_bytes, manifest = build.create_archive(self.root, self.cache)
        expected, _ = build.vendor.read_wheel(package, self.cache)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for path, content in expected.items():
                self.assertEqual(archive.read("vendor/" + path), content)
        self.assertEqual(manifest["distributions"][0]["kind"], "sdist")
        self.assertTrue(manifest["distributions"][0]["source_transformation"]["selected_source_bytes_unchanged"])
        (self.root / "uv.lock").write_text(
            'version = 1\n[[package]]\nname = "example"\nversion = "1.0"\n'
        )
        with self.assertRaisesRegex(build.BuildError, "pin differs from upstream"):
            build.create_archive(self.root, self.cache)

    def test_supplemental_pin_has_distinct_included_provenance_and_rejects_drift(self):
        package = make_wheel(self.cache, name="extra")
        package["pin_source"] = "supplemental-lock"
        self.dependencies["packages"] = [package]
        self.dependencies["source"]["supplemental_lockfile"] = build.SUPPLEMENTAL_LOCKFILE
        (self.root / "standalone" / "dependencies.json").write_text(json.dumps(self.dependencies))
        pin = {
            **package, "release_url": "https://pypi.org/project/extra/1.0/",
            "metadata_url": "https://pypi.org/pypi/extra/1.0/json", "metadata_sha256": "b" * 64,
            "review_reason": "Fixture for a separately reviewed dependency",
        }
        supplemental = {"format_version": 1, "authority": "pypi-release-metadata", "packages": [pin]}
        path = self.root / build.SUPPLEMENTAL_LOCKFILE
        path.write_text(json.dumps(supplemental))
        archive_bytes, manifest = build.create_archive(self.root, self.cache)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            self.assertEqual(archive.read("app/" + build.SUPPLEMENTAL_LOCKFILE), path.read_bytes())
        self.assertEqual(manifest["distributions"][0]["pin_source"], "supplemental-lock")
        pin["size"] += 1
        path.write_text(json.dumps(supplemental))
        with self.assertRaisesRegex(build.BuildError, "Supplemental dependency pin differs"):
            build.create_archive(self.root, self.cache)
        self.dependencies["packages"][0].pop("pin_source")
        (self.root / "standalone" / "dependencies.json").write_text(json.dumps(self.dependencies))
        with self.assertRaisesRegex(build.BuildError, "absent from upstream"):
            build.create_archive(self.root, self.cache)

    def test_default_compression_preserves_every_source_resource_and_vendor_byte(self):
        package = self.add_dependency()
        sources = build.collect_sources(self.root)
        dependencies, _ = build.vendor.read_wheel(package, self.cache)
        archive_bytes, manifest = build.create_archive(self.root, self.cache)
        info, encoded = build.encode_payload(archive_bytes, manifest["version"])
        self.assertEqual(lzma.decompress(encoded), archive_bytes)
        self.assertEqual(info["size"], len(archive_bytes))
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for path, data in sources.items():
                self.assertEqual(archive.read(path), data)
            for path, data in dependencies.items():
                self.assertEqual(archive.read("vendor/" + path), data)
            self.assertTrue(all(member.compress_type == zipfile.ZIP_STORED for member in archive.infolist()))
        self.assertEqual(manifest["source_transform"]["mode"], "none")
        self.assertTrue(manifest["source_transform"]["byte_exact"])

    def test_deflate_output_remains_format_one_and_needs_no_lzma_import(self):
        output = self.root / "dist" / "legacy.py"
        with mock.patch.dict(build.sys.modules, {"lzma": None}):
            build.build_bundle(self.root, output, self.cache, compression="deflate", payload_encoding="base85")
            with self.assertRaisesRegex(build.BuildError, "--compression deflate"):
                build.encode_payload(b"archive", "fixture")
        scope = {}
        exec(compile(output.read_text(), str(output), "exec"), scope)
        info = scope["BUNDLE_INFO"]
        payload = base64.b85decode(scope["BUNDLE_PAYLOAD"])
        self.assertEqual(info["format_version"], 1)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), info["sha256"])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertTrue(all(member.compress_type == zipfile.ZIP_DEFLATED for member in archive.infolist()))

    def test_solid_compression_uses_repetition_between_files(self):
        repeated = b"asset\n" + random.Random(42).randbytes(48 * 1024)
        for index in range(4):
            (self.root / "skills" / f"resource-{index}.dat").write_bytes(repeated)
        solid_archive, _ = build.create_archive(self.root, self.cache)
        legacy_archive, _ = build.create_archive(self.root, self.cache, compression="deflate")
        _, solid_payload = build.encode_payload(solid_archive, "fixture")
        self.assertLess(len(solid_payload), len(legacy_archive) // 2)
        self.assertEqual(lzma.decompress(solid_payload), solid_archive)

    def test_optional_ast_rewrite_reports_provenance_and_preserves_vendor_records(self):
        package = self.add_dependency()
        source = b'"""Preserved module documentation."""\n' + b"\n" * 250 + b"answer = 40 + 2\n"
        (self.root / "agent" / "rewrite.py").write_bytes(source)
        vendor_files, _ = build.vendor.read_wheel(package, self.cache)
        proof = {}
        archive_bytes, manifest = build.create_archive(
            self.root, self.cache, minify="ast", proof_collector=proof
        )
        again, _ = build.create_archive(self.root, self.cache, minify="ast")
        self.assertEqual(archive_bytes, again)
        reports = {item["path"]: item for item in proof["files"]}
        transform = manifest["source_transform"]
        self.assertNotIn("files", transform)
        self.assertEqual(transform["report_sha256"], hashlib.sha256(build.canonical_json(proof)).hexdigest())
        self.assertEqual(sum(transform["reason_counts"].values()), transform["summary"]["files"])
        self.assertEqual(transform["toolchain"], proof["toolchain"])
        report = reports["app/agent/rewrite.py"]
        self.assertEqual(report["status"], "minified")
        self.assertEqual(report["original_sha256"], hashlib.sha256(source).hexdigest())
        self.assertFalse(any(path.startswith("vendor/") for path in reports))
        self.assertGreater(manifest["source_transform"]["summary"]["changed"], 0)
        self.assertGreater(manifest["source_transform"]["summary"]["unchanged"], 0)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            transformed = archive.read("app/agent/rewrite.py")
            self.assertEqual(ast.dump(ast.parse(transformed)), ast.dump(ast.parse(source)))
            self.assertEqual(report["output_sha256"], hashlib.sha256(transformed).hexdigest())
            for path, content in vendor_files.items():
                self.assertEqual(archive.read("vendor/" + path), content)

    def test_ast_sidecar_is_deterministic_hash_bound_and_optional_at_runtime(self):
        self.add_dependency()
        standalone = Path(build.__file__).parent
        (self.root / "standalone" / "bootstrap.py").write_bytes((standalone / "bootstrap.py").read_bytes())
        (self.root / "standalone" / "runtime.py").write_bytes((standalone / "runtime.py").read_bytes())
        (self.root / "standalone" / "variant.py").write_bytes((standalone / "variant.py").read_bytes())
        (self.root / "standalone" / "cli.py").write_text(
            "def main(argv, context):\n    print('artifact runs without a sidecar')\n    return 0\n"
        )
        first = self.root / "dist" / "first.py"
        second = self.root / "elsewhere" / "second.py"
        first_result = build.build_bundle(self.root, first, self.cache, minify="ast")
        first_report = Path(first_result["minify_report"])
        report_bytes = first_report.read_bytes()
        for path in self.root.rglob("*"):
            if path.is_file():
                os.utime(path, (1_700_000_000, 1_700_000_000))
        second_result = build.build_bundle(self.root, second, self.cache, minify="ast")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(report_bytes, Path(second_result["minify_report"]).read_bytes())
        self.assertEqual(report_bytes, build.canonical_json(json.loads(report_bytes)))
        self.assertEqual(first_report.name, "first.py.minify.json")
        digest = hashlib.sha256(report_bytes).hexdigest()
        self.assertEqual(first_result["minify_report_sha256"], digest)
        self.assertEqual(first_result["minify_report_size"], len(report_bytes))
        self.assertNotIn(str(self.root).encode(), report_bytes)
        namespace = {"__name__": "inspect_built_artifact", "__file__": str(first)}
        exec(compile(first.read_bytes(), str(first), "exec"), namespace)
        archive, manifest, _, _ = namespace["_read_archive"]()
        archive.close()
        self.assertEqual(manifest["source_transform"]["report_sha256"], digest)
        self.assertNotIn("files", manifest["source_transform"])
        first_report.unlink()
        result = subprocess.run(
            [sys.executable, "-I", "-S", str(first), "--cache-dir", str(self.root / "run-cache")],
            text=True, capture_output=True, timeout=30, cwd=self.root,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "artifact runs without a sidecar\n")

    def test_ast_sidecar_symlink_cannot_overwrite_an_unrelated_file(self):
        output = self.root / "dist" / "existing.py"
        output.parent.mkdir()
        output.write_text("previous artifact\n")
        unrelated = self.root / "unrelated.txt"
        unrelated.write_text("must be preserved\n")
        output.with_name(output.name + ".minify.json").symlink_to(unrelated)
        with self.assertRaisesRegex(build.BuildError, "output symlink"):
            build.build_bundle(self.root, output, self.cache, minify="ast")
        self.assertEqual(output.read_text(), "previous artifact\n")
        self.assertEqual(unrelated.read_text(), "must be preserved\n")
        self.assertFalse(list(output.parent.glob(".hermes-build-*")))

    def test_missing_dependency_fails_offline_and_preserves_existing_output(self):
        package = self.add_dependency()
        (self.cache / package["filename"]).unlink()
        (self.root / "dist").mkdir()
        output = self.root / "dist" / "existing.py"
        output.write_text("original artifact\n")
        with mock.patch.object(build.vendor.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(build.vendor.VendorError, "builds never fetch"):
                build.build_bundle(self.root, output, self.cache)
            request.assert_not_called()
        self.assertEqual(output.read_text(), "original artifact\n")

    def test_rejects_native_data_and_source_symlinks(self):
        unexpected = self.root / "agent" / "unexpected.txt"
        unexpected.write_bytes(b"\x7fELF" + b"\0" * 24)
        with self.assertRaisesRegex(build.vendor.VendorError, "compiled payload"):
            build.create_archive(self.root, self.cache)
        unexpected.unlink()
        unexpected.symlink_to(self.root / "LICENSE")
        with self.assertRaisesRegex(build.BuildError, "regular file"):
            build.create_archive(self.root, self.cache)

    def test_original_piper_resource_requires_reviewed_bytes_and_excludes_other_models(self):
        name = "assets/piper/tashkeel/input_id_map.json"
        original = Path(build.__file__).parent / name
        target = self.root / "standalone" / name
        target.parent.mkdir(parents=True)
        target.write_bytes(original.read_bytes())
        unreviewed = target.with_name("unreviewed.onnx")
        unreviewed.write_bytes(b"unreviewed model data")
        sources = build.collect_sources(self.root)
        self.assertEqual(sources["app/standalone/" + name], original.read_bytes())
        self.assertNotIn("app/standalone/assets/piper/tashkeel/unreviewed.onnx", sources)
        target.write_bytes(original.read_bytes() + b" ")
        with self.assertRaisesRegex(build.BuildError, "Reviewed Piper resource hash differs"):
            build.collect_sources(self.root)

    def test_model_packing_preserves_all_byte_pairs_and_short_remainders(self):
        inputs = [bytes(range(length)) for length in range(13)]
        inputs.append(b"".join(bytes((left, right)) for left in range(256) for right in range(256)))
        for original in inputs:
            with self.subTest(size=len(original)):
                packed = build._pack_model_planes(original)
                self.assertEqual(len(packed), len(original))
                self.assertEqual(packed, build._pack_model_planes(original))
                output, offset = bytearray(len(original)), 0
                for index in range(4):
                    count = len(range(index, len(original), 4))
                    output[index::4] = packed[offset:offset + count]
                    offset += count
                self.assertEqual(bytes(output), original)

    def test_reviewed_python_model_configuration_stays_byte_exact_in_ast_bundle(self):
        self.add_dependency()
        root = self.root / 'standalone'
        resource = 'assets/piper/fixture/config.py'
        original = b'# Original configuration is also consumed as source data.\n' + b'\n' * 300 + b'config = {"value": 42}\n'
        path = root / resource
        path.parent.mkdir(parents=True)
        path.write_bytes(original)
        manifest_name = 'piper-fixture-resources-provenance.json'
        manifest = build.canonical_json({'resources': {resource: {'sha256': hashlib.sha256(original).hexdigest()}}})
        (root / manifest_name).write_bytes(manifest)
        (root / 'local_piper_fixture.py').write_bytes(b'\n' * 300 + b'VALUE = 42\n')
        unreviewed = path.with_name('unreviewed.onnx')
        unreviewed.write_bytes(b'not a reviewed resource')
        entries = (('local_piper_fixture.py', manifest_name, hashlib.sha256(manifest).hexdigest()),)
        proof = {}
        with mock.patch.object(build, 'PIPER_REVIEWED_RESOURCE_MANIFESTS', entries):
            data, metadata = build.create_archive(self.root, self.cache, minify='ast', proof_collector=proof)
            path.write_bytes(original + b' ')
            with self.assertRaisesRegex(build.BuildError, 'Reviewed Piper resource hash differs'):
                build.collect_sources(self.root)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            bundled = 'app/standalone/' + resource
            self.assertEqual(archive.read(bundled), original)
            scope = {}
            exec(compile(archive.read(bundled), bundled, 'exec'), scope)
            self.assertEqual(scope['config'], {'value': 42})
            self.assertNotIn('app/standalone/assets/piper/fixture/unreviewed.onnx', archive.namelist())
        reports = {item['path']: item for item in proof['files']}
        report = reports[bundled]
        self.assertEqual((report['status'], report['reason'], report['scope']),
                         ('unchanged', 'reviewed_resource', 'byte-identical-reviewed-resource'))
        self.assertEqual(report['original_sha256'], report['output_sha256'])
        self.assertEqual(reports['app/standalone/local_piper_fixture.py']['status'], 'minified')
        self.assertEqual(metadata['source_transform']['reason_counts']['reviewed_resource'], 1)

    def test_original_espeak_resources_require_complete_pinned_data_and_manifest(self):
        original = Path(build.__file__).parent
        manifest = original / build.PIPER_ESPEAK_MANIFEST
        target = self.root / 'standalone'
        (target / 'local_piper_voices.py').write_bytes((original / 'local_piper_voices.py').read_bytes())
        with self.assertRaisesRegex(build.BuildError, 'requires its reviewed eSpeak resource manifest'):
            build.collect_sources(self.root)
        (target / manifest.name).write_bytes(manifest.read_bytes())
        records = json.loads(manifest.read_bytes())['resources']
        for name, record in records.items():
            data = (original / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record['sha256'])
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        unexpected = target / 'assets/piper/espeak-ng-data/unreviewed_dict'
        unexpected.write_bytes(b'unreviewed dictionary')
        sources = build.collect_sources(self.root)
        for name in records:
            self.assertEqual(sources['app/standalone/' + name], (original / name).read_bytes())
        self.assertNotIn('app/standalone/assets/piper/espeak-ng-data/unreviewed_dict', sources)
        changed = target / 'assets/piper/espeak-ng-data/phontab'
        original_bytes = changed.read_bytes()
        changed.write_bytes(original_bytes + b'changed')
        with self.assertRaisesRegex(build.BuildError, 'Reviewed Piper resource hash differs'):
            build.collect_sources(self.root)
        changed.write_bytes(original_bytes)
        changed.unlink()
        with self.assertRaises(FileNotFoundError):
            build.collect_sources(self.root)
        (target / manifest.name).write_bytes(manifest.read_bytes() + b' ')
        with self.assertRaisesRegex(build.BuildError, 'resource manifest hash differs'):
            build.collect_sources(self.root)

    def test_rejects_nonportable_source_names(self):
        for name in ("con.txt", "e\u0301.txt", "trailing.", "trailing ", "control\x7f.txt"):
            path = self.root / "agent" / name
            with self.subTest(name=name):
                path.write_text("resource\n")
                try:
                    with self.assertRaisesRegex(build.vendor.VendorError, "Unsafe archive path"):
                        build.create_archive(self.root, self.cache)
                finally:
                    path.unlink()

    def test_rejects_path_conflicts_across_individually_valid_distributions(self):
        for first, second in (
            ("shared/Branch/one.txt", "shared/branch/two.txt"),
            ("shared/parent", "shared/parent/child.txt"),
            ("shared/parent/child.txt", "shared/parent"),
        ):
            with self.subTest(first=first, second=second):
                self.dependencies["packages"] = []
                self.add_dependency(name="first", additional=[(first, b"one")])
                self.add_dependency(name="second", additional=[(second, b"two")])
                # Each source wheel is valid alone; their combined namespace is not.
                for package in self.dependencies["packages"]:
                    build.vendor.read_wheel(package, self.cache)
                with self.assertRaisesRegex(build.vendor.VendorError, "Case-colliding|parent directory"):
                    build.create_archive(self.root, self.cache)

    def test_excludes_build_inputs_and_test_trees_from_runtime(self):
        standalone = self.root / "standalone"
        (standalone / "build.py").write_text("build_only = True\n")
        (standalone / "minify.py").write_text("build_only = True\n")
        (standalone / "size_report.py").write_text("build_only = True\n")
        (standalone / "offline_agent_probe.py").write_text("build_only = True\n")
        (standalone / "upstream_check.py").write_text("build_only = True\n")
        (standalone / "web_assets.py").write_text("build_only = True\n")
        (standalone / ".cache").mkdir()
        (standalone / ".cache" / "secret.py").write_text("not_a_runtime_module = True\n")
        (self.root / "agent" / "tests").mkdir()
        (self.root / "agent" / "tests" / "test_fake.py").write_text("test_only = True\n")
        payload, _ = build.create_archive(self.root, self.cache)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            self.assertIn("app/agent/source.py", names)
            self.assertIn("app/standalone/cli.py", names)
            self.assertNotIn("app/standalone/build.py", names)
            self.assertNotIn("app/standalone/minify.py", names)
            self.assertNotIn("app/standalone/size_report.py", names)
            self.assertNotIn("app/standalone/offline_agent_probe.py", names)
            self.assertNotIn("app/standalone/upstream_check.py", names)
            self.assertNotIn("app/standalone/web_assets.py", names)
            self.assertIn("app/standalone/agent_probe_worker.py", names)
            self.assertFalse(any("secret.py" in name or "test_fake.py" in name for name in names))

    def test_rejects_dependency_pin_not_present_in_lock(self):
        self.add_dependency()
        (self.root / "uv.lock").write_text("version = 1\npackage = []\n")
        with self.assertRaisesRegex(build.BuildError, "absent from upstream"):
            build.create_archive(self.root, self.cache)

    def test_rejects_ambiguous_bootstrap_template(self):
        template = "__HERMES_BUNDLE_INFO_LITERAL__\n__HERMES_BUNDLE_INFO_LITERAL__\n__HERMES_BUNDLE_PAYLOAD_LITERAL__"
        with self.assertRaisesRegex(build.BuildError, "exactly one"):
            build.render_artifact(template, b"payload", "0.1")

    def test_output_cannot_overwrite_or_reenter_source_inputs(self):
        original = (self.root / "agent" / "source.py").read_bytes()
        for relative in ("hermes.py", "agent/source.py", "standalone/generated.py"):
            with self.subTest(relative=relative):
                with self.assertRaisesRegex(build.BuildError, "source include set"):
                    build.build_bundle(self.root, self.root / relative, self.cache)
        self.assertEqual((self.root / "agent" / "source.py").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
