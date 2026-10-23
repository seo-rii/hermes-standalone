"""VP8 keyframe corruption, partition, and allocation boundary checks."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import random
import unittest

from standalone import raster_vp8 as vp8
from standalone.raster_webp import _chunks


class VP8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = json.loads((Path(__file__).with_name("fixtures") / "webp.json").read_text())["cases"]
        case = next(case for case in cases if case["name"] == "vp8-spatial-m6")
        cls.payload = next(payload for kind, payload in _chunks(base64.b64decode(case["webp_b64"])[12:])
                           if kind == b"VP8 ")
        cls.width, cls.height = case["width"], case["height"]

    def test_frame_dimensions_and_partition_lengths_precede_allocation(self):
        for maximum in (0, -1, True, self.width * self.height - 1):
            with self.subTest(maximum=maximum), self.assertRaises(vp8.VP8Error):
                vp8.decode(self.payload, max_pixels=maximum)
        changed = bytearray(self.payload)
        changed[6:10] = b"\xff\x3f\xff\x3f"
        with self.assertRaisesRegex(vp8.VP8Error, "pixel limit"):
            vp8.decode(changed, max_pixels=10000)
        changed = bytearray(self.payload)
        changed[:3] = b"\xf0\xff\xff"
        with self.assertRaisesRegex(vp8.VP8Error, "partition length"):
            vp8.decode(changed)
        changed[6:10] = b"\x00\x00\x00\x00"
        with self.assertRaises(vp8.VP8Error):
            vp8.decode(changed)

    def test_interframes_hidden_frames_and_reserved_versions_are_rejected(self):
        for bits, message in ((1, "keyframe"), (14, "version"), (16, "hidden")):
            changed = bytearray(self.payload)
            changed[0] = changed[0] | bits if bits != 16 else changed[0] & ~16
            with self.subTest(bits=bits), self.assertRaisesRegex(vp8.VP8Error, message):
                vp8.decode(changed)

    def test_truncated_headers_and_arithmetic_partitions_raise_codec_errors(self):
        for size in (*range(11), len(self.payload) // 4, len(self.payload) // 2, len(self.payload) * 3 // 4):
            with self.subTest(size=size), self.assertRaises(vp8.VP8Error):
                vp8.decode(self.payload[:size])
        partition = vp8._Partition(b"")
        with self.assertRaisesRegex(vp8.VP8Error, "partition"):
            partition.bit()

    def test_bit_corruption_is_bounded_and_does_not_escape_as_index_errors(self):
        rng = random.Random(6386)
        for _ in range(64):
            changed = bytearray(self.payload)
            for _ in range(3):
                index = rng.randrange(len(changed))
                changed[index] ^= 1 << rng.randrange(8)
            try:
                width, height, mode, pixels = vp8.decode(changed, max_pixels=10000)
            except vp8.VP8Error:
                continue
            self.assertEqual(mode, "RGB")
            self.assertEqual(len(pixels), width * height * 3)
            self.assertLessEqual(width * height, 10000)


if __name__ == "__main__":
    unittest.main()
