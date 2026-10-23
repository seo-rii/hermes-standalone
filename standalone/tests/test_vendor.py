from __future__ import annotations

import hashlib
import csv
import importlib.metadata
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

from standalone import vendor


def make_wheel(
    cache: Path,
    additional: list[tuple[str | zipfile.ZipInfo, bytes]] | None = None,
    *,
    name: str = "example",
) -> dict:
    """A tiny self-contained distribution used without network or site-packages."""
    output = io.BytesIO()
    files = [
        (f"{name}/__init__.py", b'VALUE = "fixture"\n'),
        (f"{name}/data.txt", b"resource contents\n"),
        (f"{name}-1.0.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\nLicense: MIT\n\n".encode()),
        (f"{name}-1.0.dist-info/WHEEL", b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
        (f"{name}-1.0.dist-info/LICENSE", b"Fixture license notice\n"),
    ]
    record = f"{name}-1.0.dist-info/RECORD"
    files.append((record, "".join(f"{path},,\n" for path in [item[0] for item in files] + [record]).encode()))
    with zipfile.ZipFile(output, "w") as archive:
        for member_name, data in files + (additional or []):
            archive.writestr(member_name, data)
    content = output.getvalue()
    filename = f"{name}-1.0-py3-none-any.whl"
    (cache / filename).write_bytes(content)
    return {
        "name": name,
        "version": "1.0",
        "filename": filename,
        "url": "https://files.pythonhosted.org/packages/example/" + filename,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
        "dependencies": [],
    }


def make_sdist(cache: Path, additional=None) -> dict:
    output = io.BytesIO()
    root = "example-1.0"
    content = {
        "src/example/__init__.py": b'VALUE = "upstream Python implementation"\n',
        "src/example/data.json": b'{"preserved": true}\n',
        "src/example/extension.c": b"/* optional accelerator source */\n",
        "setup.py": b'raise AssertionError("Build hooks must never run")\n',
        "PKG-INFO": b"Metadata-Version: 2.4\nName: example\nVersion: 1.0\nLicense-Expression: MIT\nLicense-File: LICENSE.txt\n\nOriginal metadata\n",
        "LICENSE.txt": b"Original source distribution license\n",
    }
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for path, data in content.items():
            member = tarfile.TarInfo(root + "/" + path)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        for member, data in additional or []:
            if isinstance(member, str):
                member = tarfile.TarInfo(member)
                member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    data = output.getvalue()
    filename = root + ".tar.gz"
    (cache / filename).write_bytes(data)
    return {
        "name": "example", "version": "1.0", "kind": "sdist", "filename": filename,
        "url": "https://files.pythonhosted.org/packages/example/" + filename,
        "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "dependencies": [],
        "source_layout": {
            "root": root,
            "files": {
                "example/__init__.py": "src/example/__init__.py",
                "example/data.json": "src/example/data.json",
            },
            "metadata": "PKG-INFO", "licenses": ["LICENSE.txt"],
        },
    }


class VendorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name)

    def test_preserves_distribution_metadata_license_and_resources(self):
        package = make_wheel(self.cache)
        files, provenance = vendor.read_wheel(package, self.cache)
        extracted = self.cache / "extracted"
        for name, content in files.items():
            path = extracted / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        distributions = list(importlib.metadata.distributions(path=[str(extracted)]))
        self.assertEqual(len(distributions), 1)
        self.assertEqual(distributions[0].version, "1.0")
        self.assertEqual(distributions[0].read_text("LICENSE"), "Fixture license notice\n")
        self.assertEqual(files["example/data.txt"], b"resource contents\n")
        self.assertEqual(provenance["license_files"], ["vendor/example-1.0.dist-info/LICENSE"])
        self.assertEqual(provenance["sha256"], package["sha256"])

    def test_rejects_same_size_hash_mismatch_without_network(self):
        package = make_wheel(self.cache)
        path = self.cache / package["filename"]
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
        with mock.patch.object(vendor.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(vendor.VendorError, "SHA-256 mismatch"):
                vendor.read_wheel(package, self.cache)
            request.assert_not_called()

    def test_additional_upstream_notice_preserves_sources_and_original_record(self):
        package = make_wheel(self.cache)
        original, _ = vendor.read_wheel(package, self.cache)
        notice = {
            "filename": "NOTICE.txt", "text": "Immutable upstream license notice\n",
            "source_url": "https://raw.githubusercontent.com/owner/project/" + "a" * 40 + "/NOTICE",
        }
        notice["sha256"] = hashlib.sha256(notice["text"].encode()).hexdigest()
        package["additional_licenses"] = [notice]
        files, provenance = vendor.read_wheel(package, self.cache)
        record = "example-1.0.dist-info/RECORD"
        self.assertEqual(files[record + ".upstream"], original[record])
        for path, data in original.items():
            if path != record:
                self.assertEqual(files[path], data)
        license_path = "example-1.0.dist-info/licenses/NOTICE.txt"
        self.assertEqual(files[license_path], notice["text"].encode())
        rows = {row[0]: row[1:] for row in csv.reader(io.StringIO(files[record].decode()))}
        self.assertEqual(set(rows), set(files))
        self.assertEqual(rows[license_path][1], str(len(files[license_path])))
        self.assertIn("vendor/" + license_path, provenance["license_files"])
        self.assertEqual(vendor.read_wheel(package, self.cache), (files, provenance))
        notice["text"] += "changed"
        with self.assertRaisesRegex(vendor.VendorError, "notice SHA-256 differs"):
            vendor.read_wheel(package, self.cache)
        notice["text"] = notice["text"].removesuffix("changed")
        notice["source_url"] = "https://raw.githubusercontent.com/owner/project/main/NOTICE"
        with self.assertRaisesRegex(vendor.VendorError, "immutable upstream provenance"):
            vendor.read_wheel(package, self.cache)

    def test_reviewed_native_exclusion_requires_exact_hash_and_cannot_install_binary(self):
        content = b"MZ fixture optional library"
        package = make_sdist(self.cache, [("example-1.0/bin/optional.dll", content)])
        review = {"sha256": hashlib.sha256(content).hexdigest(), "reason": "Optional system codec excluded from source selection"}
        package["source_layout"]["excluded_native_payloads"] = {"bin/optional.dll": review}
        files, provenance = vendor.read_wheel(package, self.cache)
        self.assertFalse(any(path.endswith(".dll") for path in files))
        self.assertEqual(provenance["source_transformation"]["omitted_native_payloads"]["bin/optional.dll"], review)
        review["sha256"] = "0" * 64
        with self.assertRaisesRegex(vendor.VendorError, "native payload hash differs"):
            vendor.read_wheel(package, self.cache)
        review["sha256"] = hashlib.sha256(content).hexdigest()
        package["source_layout"]["files"]["example/optional.dll"] = "bin/optional.dll"
        with self.assertRaisesRegex(vendor.VendorError, "Invalid reviewed native exclusion"):
            vendor.read_wheel(package, self.cache)

    def test_source_patch_checks_both_hashes_retains_original_and_rebuilds_record(self):
        package = make_wheel(self.cache)
        original, _ = vendor.read_wheel(package, self.cache)
        path = "example/__init__.py"
        patched = original[path].replace(b'"fixture"', b'"reviewed"')
        patch = {"path": path, "original_sha256": hashlib.sha256(original[path]).hexdigest(),
                 "patched_sha256": hashlib.sha256(patched).hexdigest(),
                 "replacements": [{"old": '"fixture"', "new": '"reviewed"'}]}
        package["source_patches"] = [patch]
        files, provenance = vendor.read_wheel(package, self.cache)
        self.assertEqual(files[path], patched)
        saved = "example-1.0.dist-info/upstream_sources/" + path
        self.assertEqual(files[saved], original[path])
        self.assertEqual(files["example-1.0.dist-info/RECORD.upstream"], original["example-1.0.dist-info/RECORD"])
        rows = {row[0]: row[1:] for row in csv.reader(io.StringIO(files["example-1.0.dist-info/RECORD"].decode()))}
        self.assertIn(saved, rows)
        self.assertEqual(provenance["source_patches"][0]["patched_sha256"], patch["patched_sha256"])
        patch["patched_sha256"] = "0" * 64
        with self.assertRaisesRegex(vendor.VendorError, "patch output differs"):
            vendor.read_wheel(package, self.cache)
        patch["patched_sha256"] = hashlib.sha256(patched).hexdigest()
        patch["original_sha256"] = "0" * 64
        with self.assertRaisesRegex(vendor.VendorError, "patch input differs"):
            vendor.read_wheel(package, self.cache)

    def test_dependency_replacement_keeps_upstream_edge_and_requires_reviewed_patch(self):
        package = make_wheel(self.cache)
        original, _ = vendor.read_wheel(package, self.cache)
        content = original["example/__init__.py"]
        package["dependencies"] = ["native-example"]
        package["replaced_dependencies"] = {
            "native-example": {"module": "standalone.fixture_backend", "reason": "Reviewed actual consumer boundary"},
        }
        manifest_path = self.cache / "dependencies.json"
        manifest = {"format_version": 1, "packages": [package]}
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(vendor.VendorError, "upstream edge and reviewed source patch"):
            vendor.load_dependencies(manifest_path)
        package["source_patches"] = [{"path": "example/__init__.py", "original_sha256": hashlib.sha256(content).hexdigest(),
                                      "patched_sha256": hashlib.sha256(content + b"# reviewed\n").hexdigest(),
                                      "replacements": [{"old": content.decode(), "new": content.decode() + "# reviewed\n"}]}]
        manifest_path.write_text(json.dumps(manifest))
        self.assertEqual(vendor.load_dependencies(manifest_path)["packages"][0]["dependencies"], ["native-example"])
        package["replaced_dependencies"]["native-example"]["module"] = "native_example"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(vendor.VendorError, "explicit standalone module"):
            vendor.load_dependencies(manifest_path)

    def test_missing_cache_does_not_fetch(self):
        package = make_wheel(self.cache)
        (self.cache / package["filename"]).unlink()
        with mock.patch.object(vendor.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(vendor.VendorError, "builds never fetch"):
                vendor.read_wheel(package, self.cache)
            request.assert_not_called()

    def test_rejects_traversal_and_case_collisions(self):
        for path in ("../outside.py", "/absolute.py", "C:/file.py", "example\\hidden.py", "EXAMPLE/__init__.py"):
            with self.subTest(path=path):
                package = make_wheel(self.cache, [(path, b"bad")])
                with self.assertRaises(vendor.VendorError):
                    vendor.read_wheel(package, self.cache)
                self.assertFalse((self.cache.parent / "outside.py").exists())

    def test_rejects_nonportable_paths_before_caching_or_extraction(self):
        for path in (
            "example/con.txt", "example/NUL", "example/AUX/readme.txt",
            "example/com1.log", "example/LPT9.log", "example/e\u0301.txt",
            "example/trailing.", "example/trailing ", "example/control\x7f.txt",
        ):
            with self.subTest(path=path):
                package = make_wheel(self.cache, [(path, b"data")])
                with self.assertRaisesRegex(vendor.VendorError, "Unsafe archive path"):
                    vendor.read_wheel(package, self.cache)

    def test_rejects_directory_case_conflicts_and_files_used_as_directories(self):
        for first, second in (
            ("example/Branch/one.txt", "example/branch/two.txt"),
            ("example/parent", "example/parent/child.txt"),
            ("example/parent/child.txt", "example/parent"),
        ):
            with self.subTest(first=first, second=second):
                package = make_wheel(self.cache, [(first, b"one"), (second, b"two")])
                with self.assertRaisesRegex(vendor.VendorError, "Case-colliding|parent directory"):
                    vendor.read_wheel(package, self.cache)

    def test_regular_directory_entries_can_follow_their_children(self):
        directory = zipfile.ZipInfo("example/")
        directory.create_system = 3
        directory.external_attr = (stat.S_IFDIR | 0o755) << 16
        package = make_wheel(self.cache, [(directory, b"")])
        files, _ = vendor.read_wheel(package, self.cache)
        self.assertIn("example/__init__.py", files)
        self.assertNotIn("example/", files)

    def test_rejects_native_magic_even_with_pure_wheel_tag(self):
        for name, data in (
            ("example/extension.so", b"not even an ELF"),
            ("example/renamed.data", b"\x7fELF" + b"\0" * 32),
            ("example/renamed.txt", b"MZ" + b"\0" * 32),
            ("example/renamed.bin", b"\x00asm" + b"\0" * 32),
            ("example/extension.so.1", b"native"),
        ):
            with self.subTest(name=name):
                package = make_wheel(self.cache, [(name, data)])
                with self.assertRaisesRegex(vendor.VendorError, "compiled payload"):
                    vendor.read_wheel(package, self.cache)

    def test_rejects_symlinks_inside_wheel_and_in_cache(self):
        link = zipfile.ZipInfo("example/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        package = make_wheel(self.cache, [(link, b"/outside")])
        with self.assertRaisesRegex(vendor.VendorError, "symlink"):
            vendor.read_wheel(package, self.cache)
        package = make_wheel(self.cache)
        path = self.cache / package["filename"]
        moved = self.cache / "real-wheel"
        path.replace(moved)
        path.symlink_to(moved)
        with self.assertRaisesRegex(vendor.VendorError, "symlink"):
            vendor.read_wheel(package, self.cache)

    def test_requires_closed_reviewed_dependencies(self):
        package = make_wheel(self.cache)
        package["dependencies"] = ["missing"]
        manifest = self.cache / "dependencies.json"
        manifest.write_text(json.dumps({"format_version": 1, "packages": [package]}))
        with self.assertRaisesRegex(vendor.VendorError, "Incomplete reviewed dependency closure"):
            vendor.load_dependencies(manifest)

    def test_fetch_reuses_verified_cache_without_network(self):
        package = make_wheel(self.cache)
        with mock.patch.object(vendor.urllib.request, "urlopen") as request:
            result = vendor.fetch_dependencies({"packages": [package]}, self.cache)
            request.assert_not_called()
        self.assertEqual(result, [{"name": "example", "status": "verified"}])

    def test_failed_download_is_never_published(self):
        package = make_wheel(self.cache)
        path = self.cache / package["filename"]
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.unlink()
        response = io.BytesIO(content)
        response.geturl = lambda: package["url"]
        with mock.patch.object(vendor.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(vendor.VendorError, "SHA-256 mismatch"):
                vendor.fetch_dependencies({"packages": [package]}, self.cache)
        self.assertFalse(path.exists())
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_reviewed_sdist_preserves_sources_and_produces_verifiable_metadata(self):
        package = make_sdist(self.cache)
        with mock.patch.object(vendor.urllib.request, "urlopen") as request:
            files, provenance = vendor.read_wheel(package, self.cache)
            repeated, repeated_provenance = vendor.read_wheel(package, self.cache)
            request.assert_not_called()
        self.assertEqual(files, repeated)
        self.assertEqual(provenance, repeated_provenance)
        self.assertEqual(files["example/__init__.py"], b'VALUE = "upstream Python implementation"\n')
        self.assertEqual(files["example/data.json"], b'{"preserved": true}\n')
        self.assertNotIn("setup.py", files)
        self.assertNotIn("example/extension.c", files)
        extracted = self.cache / "extracted"
        for name, content in files.items():
            target = extracted / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        distribution = next(importlib.metadata.distributions(path=[str(extracted)]))
        self.assertEqual(distribution.metadata["License-Expression"], "MIT")
        self.assertEqual(distribution.read_text("licenses/LICENSE.txt"), "Original source distribution license\n")
        self.assertEqual({str(path) for path in distribution.files}, set(files))
        for path in distribution.files:
            if path.name == "RECORD":
                continue
            data = files[str(path)]
            digest = vendor.base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            self.assertEqual(path.hash.mode, "sha256")
            self.assertEqual(path.hash.value, digest)
            self.assertEqual(path.size, len(data))
        self.assertTrue(provenance["source_transformation"]["selected_source_bytes_unchanged"])
        self.assertEqual(provenance["license_files"], ["vendor/example-1.0.dist-info/licenses/LICENSE.txt"])

    def test_sdist_rejects_unsafe_unselected_members(self):
        link = tarfile.TarInfo("example-1.0/unselected-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/outside"
        for member, data in (
            ("../outside.py", b"outside"),
            ("other-root/unused.py", b"outside"),
            ("example-1.0/unused.bin", b"\x7fELFnative payload"),
            ("example-1.0/unused.so", b"native payload"),
            (link, b""),
        ):
            with self.subTest(member=str(member)):
                package = make_sdist(self.cache, [(member, data)])
                with self.assertRaises(vendor.VendorError):
                    vendor.read_wheel(package, self.cache)

    def test_sdist_requires_complete_explicit_layout(self):
        package = make_sdist(self.cache)
        package["source_layout"]["files"]["example/missing.py"] = "src/example/missing.py"
        with self.assertRaisesRegex(vendor.VendorError, "Reviewed source file is missing"):
            vendor.read_wheel(package, self.cache)
        package = make_sdist(self.cache)
        package["source_layout"]["files"]["startup.pth"] = "setup.py"
        with self.assertRaisesRegex(vendor.VendorError, "startup hooks"):
            vendor.read_wheel(package, self.cache)
        package = make_sdist(self.cache)
        package["source_layout"]["licenses"] = []
        with self.assertRaisesRegex(vendor.VendorError, "license notice"):
            vendor.read_wheel(package, self.cache)

    def test_sdist_metadata_must_match_the_pinned_identity(self):
        package = make_sdist(self.cache)
        package["version"] = "2.0"
        with self.assertRaisesRegex(vendor.VendorError, "name/version differs"):
            vendor.read_wheel(package, self.cache)

    def test_source_build_bytecode_requires_exact_review_and_never_enters_payload(self):
        path = "packaging/__pycache__/build.cpython-313.pyc"
        data = b"source archive build leftover"
        package = make_sdist(self.cache, [("example-1.0/" + path, data)])
        with self.assertRaisesRegex(vendor.VendorError, "compiled payload"):
            vendor.read_wheel(package, self.cache)
        package["source_layout"]["excluded_build_bytecode"] = {path: hashlib.sha256(data).hexdigest()}
        files, provenance = vendor.read_wheel(package, self.cache)
        self.assertFalse(any(name.endswith(".pyc") for name in files))
        self.assertEqual(provenance["source_transformation"]["omitted_build_bytecode"], {
            path: hashlib.sha256(data).hexdigest(),
        })
        package["source_layout"]["excluded_build_bytecode"][path] = "0" * 64
        with self.assertRaisesRegex(vendor.VendorError, "build-bytecode hash differs"):
            vendor.read_wheel(package, self.cache)


if __name__ == "__main__":
    unittest.main()
