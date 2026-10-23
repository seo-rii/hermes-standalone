"""Source archive restoration contracts, using independent temporary fixtures."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import struct
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import zlib

from scripts import standalone_assets as assets


ORIGINAL = (b"standalone model and dictionary fixture\x00\xff" * 4096) + "한글日本語".encode()
TARGET = "standalone/models/fixture.bin"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def archive(data=ORIGINAL):
    return lzma.compress(data, format=lzma.FORMAT_XZ, filters=[{
        "id": lzma.FILTER_LZMA2, "preset": 0, "dict_size": 1024 * 1024, "pb": 0,
    }])


def dictionary_property(data, value):
    """Change the standard XZ LZMA2 dictionary property and repair its header CRC."""
    size = (data[12] + 1) * 4
    header = bytearray(data[12:12 + size])
    if header[1:4] != b"\x00\x21\x01":
        raise AssertionError("Unexpected fixture XZ block header")
    header[4] = value
    header[-4:] = struct.pack("<I", zlib.crc32(header[:-4]))
    return data[:12] + header + data[12 + size:]


def write_manifest(root, manifest):
    path = root / "build-resources/manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def fixture(root, original=ORIGINAL, compressed=None, pieces=3, target=TARGET):
    compressed = archive(original) if compressed is None else compressed
    width = max(1, (len(compressed) + pieces - 1) // pieces)
    chunks = []
    for number, start in enumerate(range(0, len(compressed), width)):
        block = compressed[start:start + width]
        path = f"build-resources/fixture.xz.{number:03d}"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(block)
        chunks.append({"path": path, "size": len(block), "sha256": sha256(block)})
    manifest = {"format_version": 1, "compression": "xz", "assets": [{
        "path": target, "size": len(original), "sha256": sha256(original),
        "compressed_size": len(compressed), "compressed_sha256": sha256(compressed),
        "chunks": chunks,
    }]}
    write_manifest(root, manifest)
    return manifest


class SourceAssetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def assert_no_publication(self):
        self.assertFalse((self.root / TARGET).exists())
        self.assertEqual(list(self.root.rglob(".source-asset-*")), [])

    def test_multipart_round_trip_and_read_only_verification(self):
        manifest = fixture(self.root)
        self.assertEqual(assets.load_manifest(self.root), manifest)
        self.assertEqual(assets.restore_assets(self.root), {TARGET: "restored"})
        path = self.root / TARGET
        self.assertEqual(path.read_bytes(), ORIGINAL)
        before = path.stat().st_mtime_ns
        self.assertEqual(assets.verify_assets(self.root), {TARGET: "verified"})
        self.assertEqual(path.stat().st_mtime_ns, before)
        if os.name == "posix":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(self.root.rglob(".source-asset-*")), [])

    def test_48mib_dictionary_and_pb_zero_archive_restores_with_memory_limit(self):
        compressed = dictionary_property(archive(), 27)
        self.assertEqual(lzma.decompress(compressed, memlimit=64 * 1024 * 1024), ORIGINAL)
        fixture(self.root, compressed=compressed)
        self.assertEqual(assets.restore_assets(self.root), {TARGET: "restored"})
        self.assertEqual((self.root / TARGET).read_bytes(), ORIGINAL)

    def test_valid_existing_cache_keeps_mtime_and_never_decodes(self):
        fixture(self.root)
        path = self.root / TARGET
        path.parent.mkdir(parents=True)
        path.write_bytes(ORIGINAL)
        os.utime(path, ns=(1234567890000000000, 1234567890000000000))
        before = path.stat().st_mtime_ns
        with mock.patch.object(assets, "_decode", side_effect=AssertionError("cache decoded")):
            self.assertEqual(assets.restore_assets(self.root), {TARGET: "present"})
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertEqual(path.read_bytes(), ORIGINAL)

    def test_mismatched_existing_file_is_preserved(self):
        fixture(self.root)
        path = self.root / TARGET
        path.parent.mkdir(parents=True)
        for occupied in (b"occupied", bytes(len(ORIGINAL))):
            with self.subTest(size=len(occupied)):
                path.write_bytes(occupied)
                before = path.stat().st_mtime_ns
                with self.assertRaises(assets.AssetError):
                    assets.restore_assets(self.root)
                self.assertEqual(path.read_bytes(), occupied)
                self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertEqual(list(self.root.rglob(".source-asset-*")), [])

    def test_every_archive_is_authenticated_before_any_decoding(self):
        manifest = fixture(self.root)
        other = copy.deepcopy(manifest["assets"][0])
        other["path"] = "standalone/models/second.bin"
        for chunk in other["chunks"]:
            source = self.root / chunk["path"]
            chunk["path"] = chunk["path"].replace("fixture.xz", "second.xz")
            (self.root / chunk["path"]).write_bytes(source.read_bytes())
        other["compressed_sha256"] = "0" * 64
        manifest["assets"].append(other)
        write_manifest(self.root, manifest)
        with mock.patch.object(assets, "_decode", side_effect=AssertionError("unverified archive decoded")):
            with self.assertRaisesRegex(assets.AssetError, "Compressed asset SHA-256"):
                assets.restore_assets(self.root)
        self.assert_no_publication()

    def test_chunk_and_combined_hash_failures_refuse_publication(self):
        for kind in ("chunk", "combined", "bytes"):
            with self.subTest(kind=kind):
                manifest = fixture(self.root)
                asset = manifest["assets"][0]
                if kind == "chunk":
                    asset["chunks"][1]["sha256"] = "0" * 64
                elif kind == "combined":
                    asset["compressed_sha256"] = "0" * 64
                else:
                    path = self.root / asset["chunks"][1]["path"]
                    data = path.read_bytes()
                    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
                write_manifest(self.root, manifest)
                with self.assertRaisesRegex(assets.AssetError, "SHA-256"):
                    assets.restore_assets(self.root)
                self.assert_no_publication()

    def test_chunk_size_and_missing_chunk_fail_before_decoding(self):
        for kind in ("long", "short", "missing"):
            with self.subTest(kind=kind):
                manifest = fixture(self.root)
                path = self.root / manifest["assets"][0]["chunks"][0]["path"]
                data = path.read_bytes()
                if kind == "missing":
                    path.unlink()
                else:
                    path.write_bytes(data + b"x" if kind == "long" else data[:-1])
                with self.assertRaises(assets.AssetError):
                    assets.restore_assets(self.root)
                self.assert_no_publication()

    def test_decoder_refuses_truncation_corruption_and_all_trailing_streams(self):
        compressed = archive()
        cases = {
            "truncated": compressed[:-8], "corrupt": b"x" + compressed[1:],
            "trailing": compressed + b"trailing", "concatenated": compressed + archive(b"another"),
            "xz padding": compressed + bytes(4),
        }
        for label, data in cases.items():
            with self.subTest(case=label):
                fixture(self.root, compressed=data)
                with self.assertRaises(assets.AssetError):
                    assets.restore_assets(self.root)
                self.assert_no_publication()

    def test_trailing_data_in_a_separate_chunk_is_rejected(self):
        manifest = fixture(self.root, pieces=1)
        asset = manifest["assets"][0]
        trailing = b"trailing in a separate part"
        path = "build-resources/fixture.xz.001"
        (self.root / path).write_bytes(trailing)
        asset["chunks"].append({"path": path, "size": len(trailing), "sha256": sha256(trailing)})
        asset["compressed_size"] += len(trailing)
        asset["compressed_sha256"] = sha256(archive() + trailing)
        write_manifest(self.root, manifest)
        with self.assertRaisesRegex(assets.AssetError, "Trailing or concatenated"):
            assets.restore_assets(self.root)
        self.assert_no_publication()

    def test_decoder_enforces_64mib_memory_bound(self):
        fixture(self.root, compressed=dictionary_property(archive(), 40))
        with self.assertRaisesRegex(assets.AssetError, "memory limit"):
            assets.restore_assets(self.root)
        self.assert_no_publication()

    def test_decoded_size_and_original_digest_must_match(self):
        for kind in ("too large", "too small", "wrong hash"):
            with self.subTest(kind=kind):
                manifest = fixture(self.root)
                asset = manifest["assets"][0]
                if kind == "wrong hash":
                    asset["sha256"] = "0" * 64
                else:
                    asset["size"] += -1 if kind == "too large" else 1
                write_manifest(self.root, manifest)
                with self.assertRaises(assets.AssetError):
                    assets.restore_assets(self.root)
                self.assert_no_publication()

    def test_large_expansion_is_written_in_blocks_of_at_most_one_mib(self):
        original = bytes(3 * 1024 * 1024 + 17)
        fixture(self.root, original=original)
        lengths = []
        original_decode = assets._decode

        class Output:
            def __init__(self, output):
                self.output = output

            def write(self, block):
                lengths.append(len(block))
                return self.output.write(block)

        def decode(repo, asset, output):
            return original_decode(repo, asset, Output(output))

        with mock.patch.object(assets, "_decode", side_effect=decode):
            self.assertEqual(assets.restore_assets(self.root), {TARGET: "restored"})
        self.assertGreater(len(lengths), 3)
        self.assertLessEqual(max(lengths), 1024 * 1024)
        self.assertEqual((self.root / TARGET).read_bytes(), original)

    def test_second_pass_reauthenticates_resources_changed_after_first_pass(self):
        manifest = fixture(self.root)
        alternate = dictionary_property(archive(), 18)
        self.assertEqual(lzma.decompress(alternate), ORIGINAL)
        original_verify = assets._verify_archives

        def change_after_check(repo, source_assets):
            original_verify(repo, source_assets)
            position = 0
            for chunk in manifest["assets"][0]["chunks"]:
                (self.root / chunk["path"]).write_bytes(alternate[position:position + chunk["size"]])
                position += chunk["size"]

        with mock.patch.object(assets, "_verify_archives", side_effect=change_after_check):
            with self.assertRaisesRegex(assets.AssetError, "Chunk SHA-256"):
                assets.restore_assets(self.root)
        self.assert_no_publication()

    def test_verification_requires_both_originals_and_compressed_resources(self):
        manifest = fixture(self.root)
        with self.assertRaisesRegex(assets.AssetError, "Missing restored"):
            assets.verify_assets(self.root)
        assets.restore_assets(self.root)
        chunk = self.root / manifest["assets"][0]["chunks"][0]["path"]
        chunk.write_bytes(b"changed")
        for command in (assets.verify_assets, assets.restore_assets):
            with self.subTest(command=command.__name__):
                with self.assertRaises(assets.AssetError):
                    command(self.root)
        self.assertEqual((self.root / TARGET).read_bytes(), ORIGINAL)

    def test_strict_schema_rejects_unknown_fields_and_wrong_types(self):
        original = fixture(self.root)
        changes = [
            ((), "unexpected", None), ((), "format_version", True), ((), "format_version", 2),
            ((), "compression", "gzip"), ((), "assets", {}), ((), "assets", []),
            (("assets", 0), "unexpected", 1), (("assets", 0), "size", True),
            (("assets", 0), "size", 1.0), (("assets", 0), "sha256", "a" * 63),
            (("assets", 0), "sha256", "A" * 64), (("assets", 0), "sha256", 3),
            (("assets", 0), "compressed_sha256", "z" * 64),
            (("assets", 0), "chunks", []), (("assets", 0), "chunks", {}),
            (("assets", 0, "chunks", 0), "unexpected", 1),
            (("assets", 0, "chunks", 0), "size", False),
            (("assets", 0, "chunks", 0), "sha256", "bad"),
        ]
        for location, key, value in changes:
            with self.subTest(location=location, key=key, value=value):
                manifest = copy.deepcopy(original)
                item = manifest
                for part in location:
                    item = item[part]
                item[key] = value
                write_manifest(self.root, manifest)
                with self.assertRaises(assets.AssetError):
                    assets.load_manifest(self.root)

    def test_manifest_bounds_are_enforced_without_opening_resources(self):
        original = fixture(self.root)
        cases = []
        for field, value in (("size", 0), ("size", -1), ("size", 512 * 1024 * 1024 + 1),
                             ("compressed_size", 0), ("compressed_size", 512 * 1024 * 1024 + 1)):
            manifest = copy.deepcopy(original)
            manifest["assets"][0][field] = value
            cases.append(manifest)
        for value in (0, -1, 16 * 1024 * 1024 + 1):
            manifest = copy.deepcopy(original)
            manifest["assets"][0]["chunks"][0]["size"] = value
            cases.append(manifest)
        many_assets = copy.deepcopy(original)
        many_assets["assets"] *= 33
        cases.append(many_assets)
        many_chunks = copy.deepcopy(original)
        many_chunks["assets"][0]["chunks"] *= 65
        cases.append(many_chunks)
        for number, manifest in enumerate(cases):
            with self.subTest(case=number):
                write_manifest(self.root, manifest)
                with self.assertRaises(assets.AssetError):
                    assets.load_manifest(self.root)

    def test_total_chunk_and_compressed_bounds_apply_across_assets(self):
        original = fixture(self.root)
        for label, count, size in (("chunks", 33, 1), ("bytes", 17, 16 * 1024 * 1024)):
            with self.subTest(bound=label):
                manifest = copy.deepcopy(original)
                manifest["assets"] = []
                for number in range(2):
                    item = copy.deepcopy(original["assets"][0])
                    item["path"] = f"standalone/models/{number}.bin"
                    item["chunks"] = [{"path": f"build-resources/{number}.xz.{part:03d}",
                                       "size": size, "sha256": "0" * 64} for part in range(count)]
                    item["compressed_size"] = count * size
                    manifest["assets"].append(item)
                write_manifest(self.root, manifest)
                with self.assertRaises(assets.AssetError):
                    assets.load_manifest(self.root)

    def test_paths_are_canonical_confined_and_portable(self):
        original = fixture(self.root)
        targets = ["../outside", "/standalone/file", "standalone/../outside", "standalone//file",
                   "standalone/./file", "standalone/file/", "standalone\\file", "standalone/a\\b",
                   "standalone/c:file", "standalone/CON", "standalone/nul.bin", "standalone/a.",
                   "standalone/LPT1.bin", "standalone/COM9", "standalone/a b", "build-resources/a",
                   "standalone/" + "a" * 256, "standalone/" + "a/" * 260 + "file"]
        chunks = ["../outside", "standalone/fixture.xz.000", "build-resources/../outside.xz.000",
                  "build-resources/a/fixture.xz.000", "build-resources//fixture.xz.000",
                  "build-resources/fixture.xz.001", "build-resources/fixture.xz.00",
                  "build-resources/fixture.xz.000 ", "build-resources/CON.xz.000"]
        for kind, values in (("target", targets), ("chunk", chunks)):
            for value in values:
                with self.subTest(kind=kind, path=value):
                    manifest = copy.deepcopy(original)
                    item = manifest["assets"][0]
                    if kind == "chunk":
                        item = item["chunks"][0]
                    item["path"] = value
                    write_manifest(self.root, manifest)
                    with self.assertRaises(assets.AssetError):
                        assets.load_manifest(self.root)

    def test_duplicate_overlapping_and_reordered_paths_are_rejected(self):
        original = fixture(self.root)
        variants = []
        for paths in ((TARGET, TARGET), (TARGET, TARGET.upper().replace("STANDALONE", "standalone")),
                      ("standalone/a", "standalone/a-b", "standalone/a/b")):
            manifest = copy.deepcopy(original)
            manifest["assets"] = []
            for number, path in enumerate(paths):
                item = copy.deepcopy(original["assets"][0])
                item["path"] = path
                for chunk in item["chunks"]:
                    chunk["path"] = chunk["path"].replace("fixture.xz", f"fixture-{number}.xz")
                manifest["assets"].append(item)
            variants.append(manifest)
        for kind in ("duplicate", "order", "mixed stems", "sizes"):
            manifest = copy.deepcopy(original)
            item = manifest["assets"][0]
            if kind == "duplicate":
                item["chunks"][1]["path"] = item["chunks"][0]["path"]
            elif kind == "order":
                item["chunks"].reverse()
            elif kind == "mixed stems":
                item["chunks"][1]["path"] = "build-resources/other.xz.001"
            else:
                item["compressed_size"] += 1
            variants.append(manifest)
        for number, manifest in enumerate(variants):
            with self.subTest(case=number):
                write_manifest(self.root, manifest)
                with self.assertRaises(assets.AssetError):
                    assets.load_manifest(self.root)

    def test_invalid_duplicate_and_oversized_manifest_json_is_rejected(self):
        path = self.root / "build-resources/manifest.json"
        path.parent.mkdir()
        for data in (b"{", b"\xff", b"[]", b'{"format_version":1,"format_version":1}',
                     b"[" * 10000 + b"]" * 10000, b" " * (1024 * 1024 + 1)):
            with self.subTest(size=len(data)):
                path.write_bytes(data)
                with self.assertRaises(assets.AssetError):
                    assets.load_manifest(self.root)

    @unittest.skipUnless(hasattr(os, "symlink"), "Requires symlinks")
    def test_symlinked_resource_or_destination_is_never_followed(self):
        probe = self.root / "symlink-probe"
        try:
            probe.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation is unavailable on this host")
        probe.unlink()
        for kind in ("manifest", "chunk", "target", "target parent", "resource parent", "root"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "checkout"
                root.mkdir()
                outside = Path(directory) / "outside"
                outside.mkdir()
                manifest = fixture(root)
                if kind in ("manifest", "chunk"):
                    path = root / (assets.MANIFEST_PATH if kind == "manifest"
                                   else manifest["assets"][0]["chunks"][0]["path"])
                    saved = outside / "original"
                    path.rename(saved)
                    path.symlink_to(saved)
                elif kind == "target":
                    path = root / TARGET
                    path.parent.mkdir(parents=True)
                    saved = outside / "original"
                    saved.write_bytes(ORIGINAL)
                    path.symlink_to(saved)
                elif kind == "target parent":
                    (root / "standalone").symlink_to(outside, target_is_directory=True)
                elif kind == "resource parent":
                    (root / "build-resources").rename(outside / "resources")
                    (root / "build-resources").symlink_to(outside / "resources", target_is_directory=True)
                else:
                    alias = Path(directory) / "alias"
                    alias.symlink_to(root, target_is_directory=True)
                    root = alias
                before = {path.relative_to(outside): path.read_bytes()
                          for path in outside.rglob("*") if path.is_file()}
                with self.assertRaises(assets.AssetError):
                    assets.restore_assets(root)
                after = {path.relative_to(outside): path.read_bytes()
                         for path in outside.rglob("*") if path.is_file()}
                self.assertEqual(after, before)

    def test_windows_reparse_attribute_bit_is_rejected_on_portable_metadata(self):
        for attributes, expected in ((0, False), (0x10, False), (0x400, True), (0x410, True)):
            with self.subTest(attributes=attributes):
                info = SimpleNamespace(st_file_attributes=attributes)
                self.assertEqual(assets._is_reparse_point(info), expected)
        self.assertFalse(assets._is_reparse_point(SimpleNamespace()))

    def test_directory_destination_and_nondirectory_parent_are_preserved(self):
        fixture(self.root)
        path = self.root / TARGET
        path.mkdir(parents=True)
        with self.assertRaises(assets.AssetError):
            assets.restore_assets(self.root)
        self.assertTrue(path.is_dir())
        path.rmdir()
        path.parent.rmdir()
        path.parent.write_bytes(b"occupied parent")
        with self.assertRaises(assets.AssetError):
            assets.restore_assets(self.root)
        self.assertEqual(path.parent.read_bytes(), b"occupied parent")

    def test_concurrent_identical_publication_is_safe_and_idempotent(self):
        fixture(self.root)
        barrier = threading.Barrier(2, timeout=10)
        original_publish = assets._publish

        def publish(parent, temporary, asset):
            barrier.wait()
            return original_publish(parent, temporary, asset)

        with mock.patch.object(assets, "_publish", side_effect=publish):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(assets.restore_assets, self.root) for _ in range(2)]
                statuses = [future.result()[TARGET] for future in futures]
        self.assertEqual(sorted(statuses), ["present", "restored"])
        self.assertEqual((self.root / TARGET).read_bytes(), ORIGINAL)
        self.assertEqual(list(self.root.rglob(".source-asset-*")), [])

    def test_concurrently_occupied_destination_is_not_overwritten(self):
        fixture(self.root)
        original_publish = assets._publish
        occupied = b"concurrent authoring data"

        def publish(parent, temporary, asset):
            (self.root / TARGET).write_bytes(occupied)
            return original_publish(parent, temporary, asset)

        with mock.patch.object(assets, "_publish", side_effect=publish):
            with self.assertRaises(assets.AssetError):
                assets.restore_assets(self.root)
        self.assertEqual((self.root / TARGET).read_bytes(), occupied)
        self.assertEqual(list(self.root.rglob(".source-asset-*")), [])

    def test_cli_reports_success_and_bounded_user_facing_failure(self):
        fixture(self.root)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(assets.main(["restore", "--root", str(self.root)]), 0)
            self.assertEqual(assets.main(["--root", str(self.root), "verify"]), 0)
        self.assertIn("restored: " + TARGET, stdout.getvalue())
        self.assertIn("verified: " + TARGET, stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        (self.root / TARGET).write_bytes(b"occupied")
        with redirect_stderr(stderr):
            self.assertEqual(assets.main(["restore", "--root", str(self.root)]), 1)
        self.assertIn("Source assets:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertNotIn(str(self.root), stderr.getvalue())
