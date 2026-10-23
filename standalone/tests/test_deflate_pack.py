"""Independent gzip/DEFLATE vectors retain every original compressed bit."""

import gzip
import hashlib
import json
from pathlib import Path
import random
import struct
import unittest
from unittest.mock import patch
import zipfile
import zlib

from standalone import deflate_pack as codec


def _envelope(deflate, clear, *, extended=False):
    header = b"\x1f\x8b\x08\x00\x12\x34\x56\x78\x00\xff"
    if extended:
        header = b"\x1f\x8b\x08\x1e\x12\x34\x56\x78\x02\x03"
        header += b"\x06\x00extra!original-name.json\x00arbitrary comment \xff\x00"
        header += struct.pack("<H", zlib.crc32(header) & 65535)
    return header + deflate + struct.pack("<II", zlib.crc32(clear), len(clear) & 0xffffffff)


class DeflatePackTests(unittest.TestCase):
    def roundtrip(self, raw, clear=None):
        if clear is not None:
            self.assertEqual(gzip.decompress(raw), clear)
        packed = codec.pack(raw)
        self.assertIsInstance(packed, bytes)
        with patch.object(zlib, "compress", side_effect=AssertionError("No encoder during unpack")), \
             patch.object(zlib, "compressobj", side_effect=AssertionError("No encoder during unpack")), \
             patch.object(zlib, "decompress", side_effect=AssertionError("No decoder during unpack")), \
             patch.object(zlib, "decompressobj", side_effect=AssertionError("No decoder during unpack")), \
             patch.object(gzip, "compress", side_effect=AssertionError("No encoder during unpack")):
            self.assertEqual(codec.unpack(packed, max_output=len(raw)), raw)
        self.assertEqual(codec.pack(raw), packed)
        return packed

    def test_all_native_compression_levels_binary_and_text_vectors(self):
        rng = random.Random(19511952)
        values = [b"", b"x", b"hello gzip", b"abc"*2000, bytes(range(256))*100,
                  rng.randbytes(30000), json.dumps({"api": [{"name": "ExampleAPI", "description": "A repeated description", "index": i} for i in range(300)]}).encode()]
        for level in range(10):
            for clear in values:
                with self.subTest(level=level, size=len(clear)):
                    self.roundtrip(gzip.compress(clear, compresslevel=level, mtime=17), clear)

    def test_fixed_dynamic_stored_multiblock_and_history_reset(self):
        clear = (b"The quick brown fox jumps over the lazy dog.\n"*400 + bytes(range(256))*100)*2
        kinds = set()
        for level in (0, 1, 6, 9):
            for strategy in (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FIXED, zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE):
                for flush in (None, zlib.Z_SYNC_FLUSH, zlib.Z_FULL_FLUSH, zlib.Z_BLOCK):
                    compressor = zlib.compressobj(level=level, wbits=-15, strategy=strategy)
                    chunks = []
                    for start in range(0, len(clear), 19997):
                        chunks.append(compressor.compress(clear[start:start+19997]))
                        if flush is not None:
                            chunks.append(compressor.flush(flush))
                    chunks.append(compressor.flush())
                    deflate = b"".join(chunks)
                    kinds.add((deflate[0] >> 1) & 3)
                    with self.subTest(level=level, strategy=strategy, flush=flush):
                        self.roundtrip(_envelope(deflate, clear, extended=True), clear)
        self.assertEqual(kinds, {0, 1, 2})

    def test_concatenated_members_metadata_and_nonzero_final_padding(self):
        # RFC fixed tree: BFINAL=1, BTYPE=1 and seven-bit symbol 256.
        # The six unused upper bits are deliberately ones, not encoder defaults.
        empty_with_padding = _envelope(b"\x03\xfc", b"", extended=True)
        raw = empty_with_padding + gzip.compress(b"second member"*100, mtime=0) + gzip.compress(bytes(range(256)), compresslevel=0)
        self.roundtrip(raw, b"second member"*100 + bytes(range(256)))

    def test_corrupt_truncated_oversized_and_non_gzip_inputs(self):
        raw = gzip.compress(b"bound and integrity"*200, mtime=0)
        packed = self.roundtrip(raw)
        self.assertIsNone(codec.pack(b"ordinary bytes"))
        for index in (0, len(packed)//2, len(packed)-1):
            damaged = bytearray(packed)
            damaged[index] ^= 1
            with self.subTest(index=index), self.assertRaises(ValueError):
                codec.unpack(damaged)
        for cut in (0, 1, 9, 20, len(packed)-1):
            with self.subTest(cut=cut), self.assertRaises(ValueError):
                codec.unpack(packed[:cut])
        with self.assertRaises(ValueError):
            codec.unpack(packed, max_output=len(raw)-1)
        with self.assertRaises(ValueError):
            codec.pack(raw, max_clear=10)
        for malformed in (raw[:-1], raw+b"garbage", raw[:3]+b"\xe0"+raw[4:],
                          raw[:-8]+bytes([raw[-8]^1])+raw[-7:]):
            with self.assertRaises(ValueError):
                codec.pack(malformed)
        with self.assertRaises(ValueError):
            codec.pack(b"\x1f\x8b\x08\x08" + b"\0"*6 + b"x"*(codec.MAX_HEADER+1))
        for argument in (-1, True, "10"):
            with self.assertRaises(ValueError):
                codec.unpack(packed, max_output=argument)
        # Even if an attacker updates the envelope checksum, semantic framing
        # and original-byte digest checks still reject a forged recipe.
        forged = bytearray(packed)
        forged[-34] ^= 1
        forged[-32:] = hashlib.sha256(forged[:-32]).digest()
        with self.assertRaises(ValueError):
            codec.unpack(forged)

    def test_native_vectors_across_window_and_memlevel_settings(self):
        rng = random.Random(47)
        for window in (9, 12, 15):
            for memory in (1, 5, 9):
                clear = rng.randbytes(1000) + b"match phrase"*1000 + rng.randbytes(1000)
                compressor = zlib.compressobj(level=6, wbits=-window, memLevel=memory)
                self.roundtrip(_envelope(compressor.compress(clear)+compressor.flush(), clear), clear)

    def test_hash_pinned_botocore_resources_small_and_large(self):
        from standalone import vendor
        standalone = Path(__file__).resolve().parents[1]
        package = next((item for item in vendor.load_dependencies(standalone / "dependencies.json")["packages"] if item["name"] == "botocore"), None)
        if package is None or not (standalone / ".cache/downloads" / package["filename"]).is_file():
            self.skipTest("The reviewed pinned botocore archive is needed for resource vectors")
        import io
        with zipfile.ZipFile(io.BytesIO(vendor._verified_bytes(package, standalone / ".cache/downloads"))) as archive:
            resources = sorted((info for info in archive.infolist() if info.filename.startswith("botocore/data/") and info.filename.endswith(".json.gz")), key=lambda info: (info.file_size, info.filename))
            self.assertTrue(resources)
            for info in resources[:5]+resources[-3:]:
                with self.subTest(resource=info.filename, size=info.file_size):
                    self.roundtrip(archive.read(info))


if __name__ == "__main__":
    unittest.main()
