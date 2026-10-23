from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import unittest
from unittest import mock
import zipfile

from standalone import build
from standalone.tests import test_build as build_fixtures


class BuildVariantTests(unittest.TestCase):
    def setUp(self):
        # Reuse only the synthetic input setup, without inheriting its tests.
        build_fixtures.BuildTests.setUp(self)
        self.optional = {
            "assets/piper/hebrew/fixture.dat": b"Hebrew model fixture\x00\x01",
            "local_piper_japanese/data/fixture-lexicon.dat": b"Japanese dictionary fixture\n",
            "local_piper_japanese/data/metadata.py": b"VALUE = 'optional data metadata'\n",
            "piper_chinese_data/model/fixture-model.dat": b"Chinese pronunciation fixture\x01\x02",
        }
        for name, data in self.optional.items():
            path = self.root / "standalone" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (self.root / "standalone/local_piper_japanese/__init__.py").write_bytes(
            b"VALUE = 'retained Japanese implementation'\n"
        )
        manifest_name = "piper-test-resources-provenance.json"
        manifest = build.canonical_json({
            "resources": {
                name: {"sha256": hashlib.sha256(data).hexdigest()}
                for name, data in self.optional.items()
            }
        })
        self.resource_manifest = self.root / "standalone" / manifest_name
        self.resource_manifest.write_bytes(manifest)
        for attribute, value in (
            ("PIPER_RESOURCE_SHA256", {
                name: hashlib.sha256(data).hexdigest()
                for name, data in self.optional.items()
                if name.startswith("assets/piper/")
            }),
            ("PIPER_REVIEWED_RESOURCE_MANIFESTS", ((
                "local_piper_japanese/__init__.py", manifest_name,
                hashlib.sha256(manifest).hexdigest(),
            ),)),
        ):
            patch = mock.patch.object(build, attribute, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.capabilities = {
            "features": [
                {"id": "agent-and-gateway", "status": "implemented", "description": "Core fixture"},
                {"id": "local-piper-model", "status": "partial", "description": "Optional speech fixture"},
            ],
            "full_parity_verified": False,
        }
        (self.root / "standalone/capabilities.json").write_bytes(build.canonical_json(self.capabilities))
        self.vendor_optional = {
            "googleapiclient/discovery_cache/documents/example.v1.json": b'{"discoveryVersion":"v1"}\n',
            "hermes_tltk/sylseg.3g": b"\x80\x03]q\x00.",
            "hermes_tltk/syllables.txt": b"Thai language resource fixture\n",
        }
        self.vendor_retained = {
            "googleapiclient/discovery_cache/__init__.py": b"VALUE = 'discovery implementation'\n",
            "googleapiclient/discovery_cache/documents/chat.v1.json": b'{"id":"chat:v1"}\n',
            "googleapiclient/discovery_cache/documents-index.json": b'{"retained":true}\n',
            "hermes_tltk/__init__.py": b"VALUE = 'Thai implementation'\n",
            "hermes_tltk/nlp.py": b"VALUE = 'Thai Python processing'\n",
        }
        self.package = build_fixtures.BuildTests.add_dependency(
            self, additional=[*self.vendor_optional.items(), *self.vendor_retained.items()]
        )

    def test_default_full_preserves_optional_data_and_capabilities(self):
        default_bytes, default = build.create_archive(self.root, self.cache)
        explicit_bytes, explicit = build.create_archive(self.root, self.cache, variant="full")
        self.assertEqual(default_bytes, explicit_bytes)
        self.assertEqual(default, explicit)
        self.assertEqual(default["build_variant"]["name"], "full")
        self.assertEqual(default["build_variant"]["excluded_features"], [])
        self.assertEqual(default["capabilities"], self.capabilities)
        with zipfile.ZipFile(io.BytesIO(default_bytes)) as archive:
            for name, original in self.optional.items():
                self.assertEqual(archive.read("app/standalone/" + name), original)
            for name, original in self.vendor_optional.items():
                self.assertEqual(archive.read("vendor/" + name), original)
            self.assertEqual(json.loads(archive.read("app/standalone/capabilities.json")), self.capabilities)

    def test_lite_omits_optional_app_resources_before_reading_them(self):
        original_source_file = build._source_file
        forbidden = {self.root / "standalone" / name for name in self.optional}

        def only_retained(path):
            self.assertNotIn(path, forbidden, "Lite attempted to read an excluded resource")
            return original_source_file(path)

        with mock.patch.object(build, "_source_file", side_effect=only_retained):
            sources = build.collect_sources(self.root, variant="lite")
        for name in self.optional:
            self.assertNotIn("app/standalone/" + name, sources)
        self.assertIn("app/standalone/local_piper_japanese/__init__.py", sources)
        self.assertEqual(
            sources["app/standalone/" + self.resource_manifest.name], self.resource_manifest.read_bytes()
        )

    def test_lite_builds_without_restored_resources_but_keeps_provenance_pins(self):
        for name in self.optional:
            (self.root / "standalone" / name).unlink()
        archive, manifest = build.create_archive(self.root, self.cache, variant="lite")
        self.assertTrue(archive)
        self.assertEqual(manifest["build_variant"]["name"], "lite")
        with self.assertRaises((OSError, build.BuildError)):
            build.create_archive(self.root, self.cache, variant="full")
        self.resource_manifest.write_bytes(self.resource_manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(build.BuildError, "resource manifest hash differs"):
            build.create_archive(self.root, self.cache, variant="lite")

    def test_lite_preserves_retained_bytes_metadata_licenses_and_source_tree(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        full_bytes, full = build.create_archive(self.root, self.cache, variant="full")
        lite_bytes, lite = build.create_archive(self.root, self.cache, variant="lite")
        after = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        omitted = {
            *("app/standalone/" + name for name in self.optional),
            *("vendor/" + name for name in self.vendor_optional),
        }
        generated = {"app/standalone/capabilities.json"}
        full_rows = {row["path"]: row for row in full["files"]}
        lite_rows = {row["path"]: row for row in lite["files"]}
        self.assertEqual(set(lite_rows), set(full_rows) - omitted)
        self.assertEqual(set(lite["build_variant"]["generated_files"]), generated)
        with zipfile.ZipFile(io.BytesIO(full_bytes)) as full_archive, \
                zipfile.ZipFile(io.BytesIO(lite_bytes)) as lite_archive:
            for name in sorted(set(lite_rows) - generated):
                self.assertEqual(lite_rows[name], full_rows[name], name)
                self.assertEqual(lite_archive.read(name), full_archive.read(name), name)
            for name, original in self.vendor_retained.items():
                self.assertEqual(lite_archive.read("vendor/" + name), original)
            for distribution in lite["distributions"]:
                self.assertTrue(lite_archive.read(distribution["metadata_path"]))
                for name in distribution["license_files"]:
                    self.assertEqual(lite_archive.read(name), full_archive.read(name))
            self.assertEqual(json.loads(lite_archive.read("app/standalone/capabilities.json")), lite["capabilities"])
        self.assertEqual(lite["profile"], full["profile"])
        self.assertEqual(lite["build_variant"]["excluded_features"], ["local-piper-model", "google-static-discovery"])
        features = {item["id"]: item for item in lite["capabilities"]["features"]}
        for name, expected_status in (("local-piper-model", "excluded"), ("google-static-discovery", "partial")):
            self.assertEqual(features[name]["status"], expected_status)
            self.assertIn("lite", features[name]["description"].lower())
        self.assertEqual(features["agent-and-gateway"], self.capabilities["features"][0])

    def test_lite_still_rejects_vendor_hash_mismatch_without_network(self):
        path = self.cache / self.package["filename"]
        damaged = bytearray(path.read_bytes())
        damaged[-1] ^= 1
        path.write_bytes(damaged)
        with mock.patch.object(build.vendor.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(build.vendor.VendorError, "SHA-256 mismatch"):
                build.create_archive(self.root, self.cache, variant="lite")
            request.assert_not_called()

    def test_lite_is_deterministic_and_has_a_distinct_cache_identity(self):
        first, first_manifest = build.create_archive(self.root, self.cache, variant="lite")
        for path in self.root.rglob("*"):
            if path.is_file():
                os.utime(path, (1_700_000_000, 1_700_000_000))
        second, second_manifest = build.create_archive(self.root, self.cache, variant="lite")
        full, _ = build.create_archive(self.root, self.cache, variant="full")
        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        lite_info, _ = build.encode_payload(first, first_manifest["version"], compression="deflate")
        full_info, _ = build.encode_payload(full, first_manifest["version"], compression="deflate")
        self.assertNotEqual(lite_info["sha256"], full_info["sha256"])

    def test_ast_variants_preserve_proof_contract_and_retained_vendor_bytes(self):
        source = b'"""Variant AST proof fixture."""\n' + b"\n" * 200 + b"answer = 40 + 2\n"
        (self.root / "agent/rewrite.py").write_bytes(source)
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        vendor_files, _ = build.vendor.read_wheel(self.package, self.cache)
        reports_by_variant = {}
        for variant in ("full", "lite"):
            with self.subTest(variant=variant):
                proof = {}
                archive_bytes, manifest = build.create_archive(
                    self.root, self.cache, variant=variant, minify="ast", proof_collector=proof,
                )
                repeated_proof = {}
                repeated_bytes, _ = build.create_archive(
                    self.root, self.cache, variant=variant, minify="ast", proof_collector=repeated_proof,
                )
                self.assertEqual(archive_bytes, repeated_bytes)
                self.assertEqual(proof, repeated_proof)
                transform = manifest["source_transform"]
                self.assertEqual(proof["scope"], "app-python-sources")
                self.assertEqual(proof["toolchain"]["proof_scope"], "python-ast-and-bytecode-excluding-source-locations")
                self.assertEqual(transform["scope"], proof["scope"])
                self.assertEqual(transform["toolchain"], proof["toolchain"])
                self.assertEqual(transform["report_sha256"], hashlib.sha256(build.canonical_json(proof)).hexdigest())
                self.assertNotIn("files", transform)
                reports = {item["path"]: item for item in proof["files"]}
                reports_by_variant[variant] = reports
                self.assertFalse(any(path.startswith("vendor/") for path in reports))
                rewrite = reports["app/agent/rewrite.py"]
                self.assertEqual(rewrite["status"], "minified")
                self.assertEqual(rewrite["original_sha256"], hashlib.sha256(source).hexdigest())
                self.assertEqual(rewrite["scope"], proof["toolchain"]["proof_scope"])
                self.assertRegex(rewrite["code_sha256"], r"^[0-9a-f]{64}$")
                with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                    rewritten = archive.read("app/agent/rewrite.py")
                    self.assertEqual(ast.dump(ast.parse(rewritten)), ast.dump(ast.parse(source)))
                    self.assertEqual(rewrite["output_sha256"], hashlib.sha256(rewritten).hexdigest())
                    for path, original in vendor_files.items():
                        if variant == "lite" and path in self.vendor_optional:
                            self.assertNotIn("vendor/" + path, archive.namelist())
                        else:
                            self.assertEqual(archive.read("vendor/" + path), original, path)
        excluded_python = "app/standalone/local_piper_japanese/data/metadata.py"
        self.assertEqual(set(reports_by_variant["full"]) - set(reports_by_variant["lite"]), {excluded_python})
        for path, report in reports_by_variant["lite"].items():
            self.assertEqual(report, reports_by_variant["full"][path], path)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_cli_both_creates_named_distinct_source_and_zipapp_outputs(self):
        for container, extension in (("source", ".py"), ("zipapp", ".pyz")):
            with self.subTest(container=container):
                output = self.root / "dist" / container
                stdout = io.StringIO()
                with mock.patch.object(build, "DEFAULT_ROOT", self.root), contextlib.redirect_stdout(stdout):
                    status = build.main([
                        "--variant", "both", "--container", container, "--compression", "deflate",
                        "--cache-dir", str(self.cache), "--output-dir", str(output),
                    ])
                self.assertEqual(status, 0, stdout.getvalue())
                paths = {name: output / ("hermes-" + name + extension) for name in ("full", "lite")}
                self.assertEqual({path.name for path in output.iterdir()}, {path.name for path in paths.values()})
                identities = {}
                for name, path in paths.items():
                    if container == "source":
                        namespace = {}
                        exec(compile(path.read_bytes(), str(path), "exec"), namespace)
                        info = namespace["BUNDLE_INFO"]
                        payload = b"".join(chunk[1:-1].encode("latin-1") for chunk in namespace["BUNDLE_PAYLOAD"])
                    else:
                        with zipfile.ZipFile(path) as outer:
                            namespace = {}
                            exec(compile(outer.read("__main__.py"), str(path), "exec"), namespace)
                            info = namespace["BUNDLE_INFO"]
                            payload = outer.read("payload.bin")
                    if info.get("compression") == "lzma":
                        payload = lzma.decompress(payload)
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), info["sha256"])
                    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                        manifest = json.loads(archive.read("bundle-manifest.json"))
                    self.assertEqual(manifest["build_variant"]["name"], name)
                    identities[name] = info["sha256"]
                self.assertNotEqual(identities["full"], identities["lite"])

    def test_cli_rejects_ambiguous_output_before_writing_anything(self):
        destination = self.root / "dist" / "custom.py"
        for arguments in (
            ["--variant", "both", "--output", str(destination)],
            ["--output", str(destination), "--output-dir", str(destination.parent)],
        ):
            with self.subTest(arguments=arguments), mock.patch.object(build, "DEFAULT_ROOT", self.root), \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    status = build.main([*arguments, "--cache-dir", str(self.cache)])
                except SystemExit as error:
                    status = error.code
                self.assertNotEqual(status, 0)
                self.assertFalse(destination.parent.exists())

    def test_cli_default_and_lite_have_stable_distinct_output_names(self):
        full = self.root / "dist/hermes.py"
        lite = self.root / "dist/hermes-lite.py"
        default_bytes = None
        for arguments, expected in (([], full), (["--variant", "lite"], lite), (["--variant", "full"], full)):
            with self.subTest(arguments=arguments), mock.patch.object(build, "DEFAULT_ROOT", self.root), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = build.main([
                    *arguments, "--cache-dir", str(self.cache), "--compression", "deflate",
                ])
                self.assertEqual(status, 0)
                self.assertTrue(expected.is_file())
                if not arguments:
                    default_bytes = expected.read_bytes()
        self.assertEqual(full.read_bytes(), default_bytes)
        self.assertNotEqual(full.read_bytes(), lite.read_bytes())
        self.assertEqual({path.name for path in full.parent.iterdir()}, {full.name, lite.name})

    def test_nonexistent_variant_is_rejected_by_packaging_apis(self):
        for variant in ("both", "smallest", "", None):
            with self.subTest(variant=variant):
                with self.assertRaises(build.BuildError):
                    build.collect_sources(self.root, variant=variant)
                with self.assertRaises(build.BuildError):
                    build.create_archive(self.root, self.cache, variant=variant)
                destination = self.root / "dist" / "invalid.py"
                with self.assertRaises(build.BuildError):
                    build.build_bundle(self.root, destination, self.cache, variant=variant)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
