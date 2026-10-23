"""Independent libwebp oracle fixtures with an isolated stdlib runtime."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import struct
import unittest
import zlib

from standalone import raster_webp as webp


FIXTURES = Path(__file__).with_name("fixtures") / "webp.json"


def rgba(mode, pixels):
    if mode == "RGBA":
        return pixels
    return bytes(channel for index in range(0, len(pixels), 3)
                 for channel in (*pixels[index:index + 3], 255))


class WebPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(FIXTURES.read_text())

    def test_native_lossless_encoder_fixtures_decode_exactly(self):
        covered = set()
        for case in self.document["cases"]:
            with self.subTest(case=case["name"]):
                data = base64.b64decode(case["webp_b64"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), case["webp_sha256"])
                width, height, frames, info = webp.decode_frames(data)
                self.assertEqual((width, height), (case["width"], case["height"]))
                self.assertEqual(len(frames), len(case["frames"]))
                for actual, expected in zip(frames, case["frames"]):
                    mode, pixels, duration = actual
                    self.assertEqual(rgba(mode, pixels),
                                     zlib.decompress(base64.b64decode(expected["pixels_zlib_b64"])))
                    self.assertEqual(duration, expected["duration"])
                for key, value in case.get("info", {}).items():
                    self.assertEqual(list(info[key]) if isinstance(info[key], tuple) else info[key], value)
                first = webp.decode(data)
                self.assertEqual(first, (width, height, frames[0][0], frames[0][1]))
                covered.update(case["features"])
        self.assertTrue({"predictor", "color", "subtract-green", "palette", "cache", "animation",
                         "entropy-groups", "vp8", "alpha", "segmented", "filtered"} <= covered)

    def test_encoder_matches_committed_native_decoder_pixels_and_digest(self):
        for case in self.document["encoding_cases"]:
            with self.subTest(case=case["name"]):
                pixels = zlib.decompress(base64.b64decode(case["input_zlib_b64"]))
                data = webp.encode(case["width"], case["height"], case["mode"], pixels,
                                   lossless=True, quality=100, method=6, exact=True)
                self.assertEqual(hashlib.sha256(data).hexdigest(), case["webp_sha256"])
                self.assertEqual(len(data), case["encoded_size"])
                width, height, mode, decoded = webp.decode(data)
                self.assertEqual(rgba(mode, decoded),
                                 zlib.decompress(base64.b64decode(case["pixels_zlib_b64"])))
                self.assertEqual(data, webp.encode(width, height, case["mode"], pixels))

    def test_lz77_compresses_repeated_rows_and_preserves_invisible_colors(self):
        width, height = 257, 129
        pixels = bytes((123, 89, 54, 0)) * (width * height)
        data = webp.encode(width, height, "RGBA", pixels)
        self.assertLess(len(data), 500)
        self.assertEqual(webp.decode(data), (width, height, "RGBA", pixels))
        for mode, pixels, expected in (("L", b"\x71", b"\x71\x71\x71\xff"),
                                      ("LA", b"\x71\x03", b"\x71\x71\x71\x03")):
            decoded = webp.decode(webp.encode(1, 1, mode, pixels))
            self.assertEqual(rgba(decoded[2], decoded[3]), expected)

    def test_container_truncation_versions_and_reserved_fields_are_rejected(self):
        original = webp.encode(3, 2, "RGB", b"\x12\x34\x56" * 6)
        for size in range(len(original)):
            with self.subTest(size=size):
                with self.assertRaises(webp.WebPError):
                    webp.decode(original[:size])
        for malformed in (b"", b"GIF89a", original + b"\x00", b"RIFF\x00\x00\x00\x00WEBP"):
            with self.assertRaises(webp.WebPError):
                webp.decode(malformed)
        changed = bytearray(original)
        changed[24] |= 0xe0  # Header's three version bits.
        with self.assertRaisesRegex(webp.WebPError, "version"):
            webp.decode(changed)
        # RIFF padding is part of the container and must contain zero.
        chunk = b"JUNK" + struct.pack("<I", 1) + b"x\x01"
        malformed = b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk
        with self.assertRaisesRegex(webp.WebPError, "padding"):
            webp.decode(malformed)

    def test_pixel_frame_and_aggregate_limits_precede_frame_allocation(self):
        still = webp.encode(3, 2, "RGB", b"abc" * 6)
        for limit in (0, -1, True, 5):
            with self.subTest(limit=limit), self.assertRaises(webp.WebPError):
                webp.decode(still, max_pixels=limit)
        case = next(item for item in self.document["cases"] if "animation" in item["features"])
        animation = base64.b64decode(case["webp_b64"])
        with self.assertRaisesRegex(webp.WebPError, "aggregate"):
            webp.decode(animation, max_pixels=case["width"] * case["height"])
        with self.assertRaisesRegex(webp.WebPError, "frame"):
            webp.decode_frames(animation, max_frames=1)
        for limit in (0, -1, True):
            with self.assertRaises(webp.WebPError):
                webp.decode_frames(animation, max_frames=limit)
        # The final frame is always validated, even when only frame 0 is wanted.
        changed = bytearray(animation)
        marker = changed.rfind(b"VP8L")
        changed[marker + 8] = 0
        with self.assertRaisesRegex(webp.WebPError, "signature"):
            webp.decode(changed)

    def test_malformed_entropy_trees_cache_and_references_fail_closed(self):
        def stream(write_payload):
            writer = webp._Writer()
            writer.write(0, 14)
            writer.write(0, 14)
            writer.write(0, 4)
            writer.write(0, 1)
            write_payload(writer)
            payload = b"\x2f" + writer.finish()
            chunk = b"VP8L" + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) & 1 else b"")
            return b"RIFF" + struct.pack("<I", len(chunk) + 4) + b"WEBP" + chunk

        def invalid_cache(writer):
            writer.write(1, 1)
            writer.write(0, 4)

        with self.assertRaisesRegex(webp.WebPError, "cache"):
            webp.decode(stream(invalid_cache))

        def future_reference(writer):
            writer.write(0, 1)
            writer.write(0, 1)
            for alphabet, symbol in ((280, 256), (256, 0), (256, 0), (256, 255), (40, 0)):
                lengths = [0] * alphabet
                lengths[symbol] = 1
                webp._write_tree(writer, lengths)

        with self.assertRaisesRegex(webp.WebPError, "backward reference"):
            webp.decode(stream(future_reference))
        for lengths in ([0] * 4, [1] * 3, [2, 2, 0, 0], [16, 0, 0, 0]):
            with self.assertRaises(webp.WebPError):
                webp._Tree(lengths)

    def test_invalid_vp8_payload_and_bad_encoder_options_are_explicit(self):
        payload = b"VP8 " + struct.pack("<I", 2) + b"xx"
        data = b"RIFF" + struct.pack("<I", len(payload) + 4) + b"WEBP" + payload
        with self.assertRaisesRegex(webp.WebPError, "keyframe"):
            webp.decode(data)
        for args in ((0, 1, "RGB", b""), (1, 1, "P", b"x"), (1, 1, "RGB", b"x")):
            with self.assertRaises(webp.WebPError):
                webp.encode(*args)
        for options in ({"lossless": False}, {"method": 7}, {"method": True}, {"quality": -1},
                        {"quality": float("nan")}, {"quality": True}, {"unknown": 1}):
            with self.subTest(options=options), self.assertRaises(webp.WebPError):
                webp.encode(1, 1, "RGB", b"abc", **options)


if __name__ == "__main__":
    unittest.main()
