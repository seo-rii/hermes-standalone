"""Validate frontend producer receipts and their complete notice inventory."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from standalone.web_assets import _output_hash, _source_hash, stage


# Computed by scripts/build/freshness.mjs buildInputs for this fixed source
# fixture. Absolute prepared paths are recorded separately from their tree hash.
SOURCE_HASH = "1099b95c3ea5c77bc547b7eeac91703e0bd962a2d02709c723047b5249a198b3"
ICONS_HASH = "2d8a4129f5940625be29beed82ac539cf05214adc58587686d84e9b045e0cdc7"


class WebAssetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hermes-assets-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.product = self.root / "source", self.root / "product"
        self.source.mkdir()
        (self.product / "assets").mkdir(parents=True)
        (self.source / "web/src").mkdir(parents=True)
        (self.source / "web/public").mkdir()
        (self.source / "apps/shared").mkdir(parents=True)
        self.package = self.source / "node_modules/fixture-lib"
        self.package.mkdir(parents=True)
        self.root_metadata = {"name": "fixture", "version": "1.0.0"}
        self.web_metadata = {"name": "web", "version": "0.0.0", "dependencies": {"fixture-lib": "1.0.0"},
                             "optionalDependencies": {"fixture-optional": "1.0.0"}}
        self.shared_metadata = {"name": "@hermes/shared", "version": "0.0.0"}
        self.lock = {"lockfileVersion": 3, "packages": {
            "": self.root_metadata, "web": self.web_metadata, "apps/shared": self.shared_metadata,
            "node_modules/fixture-lib": {"version": "1.0.0", "resolved": "https://example.invalid/lib.tgz",
                                         "integrity": "sha512-fixture", "license": "MIT"},
            "node_modules/fixture-optional": {"version": "1.0.0", "optional": True, "license": "MIT"},
            "node_modules/unused-workspace-dependency": {"version": "1.0.0", "license": "MIT"},
        }}
        for name, value in (("package.json", self.root_metadata), ("web/package.json", self.web_metadata),
                            ("apps/shared/package.json", self.shared_metadata), ("package-lock.json", self.lock)):
            (self.source / name).write_text(json.dumps(value), encoding="utf-8")
        self.write_package({"name": "fixture-lib", "version": "1.0.0", "license": "MIT"})
        (self.package / "LICENSE").write_text("Fixture upstream copyright and license notice.\n", encoding="utf-8")
        (self.source / "web/src/view.js").write_bytes(b"export const value = 1;\n")
        (self.source / "web/src/\U00010000.js").write_bytes(b"astral\n")
        (self.source / "web/src/\ue000.js").write_bytes(b"bmp\n")
        (self.source / "web/public/favicon.ico").write_bytes(b"fixture icon")
        (self.product / "assets/app.js").write_bytes(b"console.log(1)")
        (self.product / "index.html").write_bytes(b"<html>fixture</html>")
        self.receipt = {"schema": 1, "product": "web", "outputHash": _output_hash(self.product), "node": "fixture",
                        "platform": "fixture", "arch": "fixture", "inputs": {
                            "sourceHash": SOURCE_HASH, "prepared": [{"name": "icons", "hash": ICONS_HASH,
                                "path": str(self.source / "web/public")}],
                        }}
        self.write_receipt()

    def write_package(self, metadata):
        (self.package / "package.json").write_text(json.dumps(metadata), encoding="utf-8")

    def write_receipt(self):
        (self.product / "hermes-build.json").write_text(json.dumps(self.receipt), encoding="utf-8")

    def test_original_receipt_layout_and_modified_output_refusal(self):
        expected = hashlib.sha256(b"assets\0directory\0assets/app.js\0console.log(1)\0index.html\0<html>fixture</html>\0").hexdigest()
        (self.product / "assets/.DS_Store").write_bytes(b"ignored OS metadata")
        self.assertEqual(_output_hash(self.product), expected)
        output = self.root / "output"
        self.assertEqual(stage(self.source, self.product, output)["files"], 4)
        self.assertFalse((output / "assets/.DS_Store").exists())
        manifest = json.loads((output / "standalone-assets.json").read_text(encoding="utf-8"))
        self.assertEqual([item["name"] for item in manifest["packages"]], ["fixture-lib"])
        notice = manifest["packages"][0]["notices"][0]
        self.assertEqual((output / notice).read_bytes(), (self.package / "LICENSE").read_bytes())
        (self.product / "assets/app.js").write_bytes(b"changed after the build")
        with self.assertRaisesRegex(ValueError, "differs from its original build receipt"):
            stage(self.source, self.product, self.root / "tampered")
        self.assertFalse((self.root / "tampered").exists())

    def test_changed_source_or_lock_is_rejected_before_staging(self):
        for index, name in enumerate(("web/src/view.js", "package-lock.json")):
            with self.subTest(name=name):
                path = self.source / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                output = self.root / f"rejected-{index}"
                try:
                    with self.assertRaisesRegex(ValueError, "source"):
                        stage(self.source, self.product, output)
                    self.assertFalse(output.exists())
                finally:
                    path.write_bytes(original)

    def test_changed_prepared_icons_are_rejected(self):
        external = self.root / "icons"
        external.mkdir()
        (external / "favicon.ico").write_bytes(b"different prepared icons")
        self.receipt["inputs"]["prepared"][0]["path"] = str(external)
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "prepared|icons"):
            stage(self.source, self.product, self.root / "rejected")

    def test_installed_package_name_and_version_must_match_the_lock(self):
        for index, metadata in enumerate(({"name": "other-library", "version": "1.0.0", "license": "MIT"},
                                        {"name": "fixture-lib", "version": "9.0.0", "license": "MIT"})):
            with self.subTest(metadata=metadata):
                self.write_package(metadata)
                output = self.root / f"rejected-{index}"
                with self.assertRaisesRegex(ValueError, "package|identity"):
                    stage(self.source, self.product, output)
                self.assertFalse(output.exists())

    def test_missing_runtime_dependency_or_notice_is_rejected(self):
        (self.package / "LICENSE").unlink()
        with self.assertRaisesRegex(ValueError, "notice"):
            stage(self.source, self.product, self.root / "no-notice")
        self.assertFalse((self.root / "no-notice").exists())
        shutil.rmtree(self.package)
        with self.assertRaisesRegex(ValueError, "dependency|package"):
            stage(self.source, self.product, self.root / "no-package")
        self.assertFalse((self.root / "no-package").exists())

    def test_recursive_notices_and_producer_supported_package_symlinks(self):
        (self.package / "licenses").mkdir()
        (self.package / "LICENSE").rename(self.package / "licenses/COPYING")
        relocated = self.root / "prepared-package"
        self.package.rename(relocated)
        try:
            self.package.symlink_to(relocated, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Package symlinks are unavailable: {exc}")
        output = self.root / "output"
        stage(self.source, self.product, output)
        manifest = json.loads((output / "standalone-assets.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["packages"]), 1)
        notice = manifest["packages"][0]["notices"][0]
        self.assertEqual((output / notice).read_bytes(), (relocated / "licenses/COPYING").read_bytes())

    def test_runtime_resolution_skips_an_uninstalled_nearer_lock_entry(self):
        # npm locks all workspace layouts even when the selected workspace
        # install does not materialize every closer duplicate.
        self.lock["packages"]["web/node_modules/fixture-lib"] = {"version": "1.0.0", "license": "MIT"}
        (self.source / "package-lock.json").write_text(json.dumps(self.lock), encoding="utf-8")
        self.receipt["inputs"]["sourceHash"] = _source_hash(self.source)
        self.write_receipt()
        output = self.root / "output"
        self.assertEqual(stage(self.source, self.product, output)["packages"], 1)
