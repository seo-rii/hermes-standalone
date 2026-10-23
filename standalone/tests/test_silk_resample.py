"""Bit-exact PCM vectors authored with the independent BSD SILK SDK.

Shipped tests use only JSON, zlib and Python integer arithmetic. No compiler,
native library, external codec or platform emulation is used to validate them.
"""

import base64
import hashlib
import json
from pathlib import Path
import struct
import unittest
import zlib

from standalone.silk_resample import Resampler


FIXTURE = Path(__file__).with_name("fixtures") / "silk_resample.json"


def signal(kind, rate, lengths):
    values, state = [], 0x53A17C29 + rate
    for index in range(sum(lengths)):
        if kind == "silence":
            value = 0
        elif kind == "impulse":
            value = 32767 if index == 0 else -32768 if index == rate // 50 + 3 else 0
        elif kind == "extremes":
            value = (-32768, 32767, -32768, -32768, 32767, 0)[index % 6]
        elif kind == "noise":
            state = (1103515245 * state + 12345) & 0xffffffff
            value = (state >> 16) - 32768
        else:
            raise ValueError("Unknown resampler fixture signal")
        values.append(value)
    frames, offset = [], 0
    for length in lengths:
        frames.append(values[offset:offset + length])
        offset += length
    return frames


class SilkResamplerTests(unittest.TestCase):
    def test_stateful_fixed_point_pcm_matches_native_sdk_for_every_batch(self):
        reference = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(reference["source_archive_sha256"], "d4a1bcf93dc6ef5e95e0cfd728ed4ef4d49f9c0476d70816fecbe456cc762e7f")
        for vector in reference["vectors"]:
            with self.subTest(id=vector["id"]):
                resampler = Resampler(vector["in_rate"], vector["out_rate"])
                expected = zlib.decompress(base64.b64decode(vector["pcm_zlib_base64"], validate=True))
                self.assertEqual(hashlib.sha256(expected).hexdigest(), vector["pcm_sha256"])
                pieces = []
                for samples, count in zip(signal(vector["signal"], vector["in_rate"], vector["frame_lengths"]),
                                          vector["output_lengths"], strict=True):
                    output = resampler.process(samples)
                    self.assertEqual(len(output), count)
                    self.assertTrue(all(-32768 <= value <= 32767 for value in output))
                    pieces.append(struct.pack("<" + "h" * len(output), *output))
                    self.assertEqual(resampler.process([]), [])
                actual = b"".join(pieces)
                self.assertEqual(hashlib.sha256(actual).hexdigest(), vector["pcm_sha256"])
                self.assertEqual(actual, expected)

    def test_rate_changes_start_new_filter_state_and_invalid_input_cannot_advance_it(self):
        frames = signal("noise", 16000, [320] * 3)
        for output_rate in (8000, 11025, 12000, 16000, 24000, 48000):
            with self.subTest(output_rate=output_rate):
                resampler, expected = Resampler(16000, output_rate), Resampler(16000, output_rate)
                self.assertEqual(resampler.process(frames[0]), expected.process(frames[0]))
                for invalid in ([32768], [-32769], [1.0], [True], (1, 2), [0] * 16001):
                    with self.assertRaises(ValueError):
                        resampler.process(invalid)
                self.assertEqual(resampler.process(frames[1]), expected.process(frames[1]))
                fresh = Resampler(16000, output_rate)
                self.assertEqual(fresh.process(frames[0]), Resampler(16000, output_rate).process(frames[0]))
        for rates in ((7999, 8000), (8000, 48001), (16000.0, 24000), (16000, True), (0, 0)):
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                Resampler(*rates)
