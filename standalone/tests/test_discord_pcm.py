"""Discord mixing and WAV conversion use actual independent native vectors."""

import base64
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave
import zlib

from standalone import discord_pcm
from standalone.tests.test_silk_resample import signal


FIXTURES = Path(__file__).with_name("fixtures")


class DiscordPCMTests(unittest.TestCase):
    def test_float32_gain_addition_clipping_matches_native_fmaf(self):
        oracle = json.loads((FIXTURES / "discord-pcm.json").read_text())
        for index, case in enumerate(oracle["vectors"]):
            with self.subTest(index=index):
                first = discord_pcm.float_frame(base64.b64decode(case["first"]), case["first_gain"])
                second = discord_pcm.float_frame(base64.b64decode(case["second"]), case["second_gain"])
                output = discord_pcm.float_frame_bytes(discord_pcm.add_float_frames(first, second))
                self.assertEqual(output, base64.b64decode(case["expected"]))

    def test_pcm_wav_reader_and_stt_writer_match_native_resampling(self):
        oracle = json.loads((FIXTURES / "silk_resample.json").read_text())
        for in_rate, out_rate in ((16000, 48000), (48000, 16000)):
            case = next(case for case in oracle["vectors"]
                        if case["in_rate"] == in_rate and case["out_rate"] == out_rate and case["signal"] == "noise")
            samples = [sample for frame in signal(case["signal"], in_rate, case["frame_lengths"]) for sample in frame]
            expected = zlib.decompress(base64.b64decode(case["pcm_zlib_base64"]))
            with self.subTest(in_rate=in_rate), tempfile.TemporaryDirectory(prefix="hermes-discord-pcm-") as directory:
                path = Path(directory) / "recording.wav"
                if in_rate == 16000:
                    with wave.open(str(path), "wb") as output:
                        output.setparams((1, 2, in_rate, 0, "NONE", "not compressed"))
                        output.writeframes(struct.pack("<" + "h" * len(samples), *samples))
                    stereo = b"".join(struct.pack("<hh", sample[0], sample[0])
                                      for sample in struct.iter_unpack("<h", expected))
                    self.assertEqual(discord_pcm.decode_wav(path), stereo)
                else:
                    stereo = b"".join(struct.pack("<hh", sample, sample) for sample in samples)
                    discord_pcm.write_stt_wav(stereo, path, in_rate, 2)
                    with wave.open(str(path), "rb") as source:
                        self.assertEqual((source.getframerate(), source.getnchannels(), source.getsampwidth()), (16000, 1, 2))
                        self.assertEqual(source.readframes(source.getnframes()), expected)

    def test_ambient_spectrum_loop_length_and_deterministic_seed(self):
        first = discord_pcm.synth_ambient_pcm(.1)
        self.assertEqual(first, discord_pcm.synth_ambient_pcm(.1))
        self.assertEqual(len(first), 4800 * 4)
        samples = list(struct.iter_unpack("<hh", first))
        self.assertTrue(all(left == right for left, right in samples))
        self.assertGreater(max(abs(left) for left, _ in samples), 1000)
        self.assertLessEqual(max(abs(left) for left, _ in samples), 16384)


if __name__ == "__main__":
    unittest.main()
