"""Independent PyAV/mp3float and native minimp3 MPEG Layer III fixtures."""
import base64
import hashlib
import json
import lzma
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
import zlib

from standalone.local_whisper_audio_mp3 import (
    Mp3Error, _Bits, _scalefactors, decode_mp3, probe_mp3,
)


class Mp3DecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wrapper = json.loads((Path(__file__).parent / 'fixtures/local_whisper_audio_mp3.json').read_text())
        metadata = zlib.decompress(base64.b64decode(wrapper['metadata_zlib_base64'], validate=True))
        if hashlib.sha256(metadata).hexdigest() != wrapper['metadata_sha256']:
            raise AssertionError('Native MP3 metadata digest mismatch')
        cls.oracle = json.loads(metadata)
        shuffled = lzma.decompress(base64.b64decode(wrapper['native_pcm_shuffle_lzma_base64'], validate=True), memlimit=32 * 1024 * 1024)
        if len(shuffled) != wrapper['native_pcm_bytes'] or len(shuffled) % 4:
            raise AssertionError('Native MP3 PCM byte count mismatch')
        size = len(shuffled) // 4
        pcm = bytearray(len(shuffled))
        for byte in range(4):
            pcm[byte::4] = shuffled[byte * size:(byte + 1) * size]
        if hashlib.sha256(pcm).hexdigest() != wrapper['native_pcm_sha256']:
            raise AssertionError('Native MP3 float PCM digest mismatch')
        cls.native_pcm = pcm

    def _compare(self, case, **kwargs):
        actual = decode_mp3(base64.b64decode(case['data'], validate=True), **kwargs)
        self.assertEqual((actual[0], len(actual[1]), actual[2], actual[3]),
                         (case['rate'], case['channels'], 'fltp', 4 if case['channels'] == 1 else 3))
        count = case['samples']
        self.assertTrue(all(len(channel) == count for channel in actual[1]))
        expected = struct.unpack_from('<' + str(count * case['channels']) + 'f', self.native_pcm, case['native_f32le_offset'])
        error = 0.
        for channel, values in enumerate(actual[1]):
            self.assertTrue(all(math.isfinite(value) for value in values))
            error = max(error, max(abs(a - b) for a, b in zip(values, expected[channel * count:(channel + 1) * count])))
        self.assertLessEqual(error, case['tolerance'])

    def test_actual_encoder_all_nine_mpeg_rates_mono_stereo_and_signals(self):
        cases = [case for case in self.oracle['cases'] if case['group'] == 'native-comparison']
        self.assertEqual(len(cases), 90)
        for case in cases:
            with self.subTest(case=case['name']):
                self._compare(case)

    def test_native_huffman_count1_hybrid_windows_mixed_intensity_and_crc(self):
        cases = [case for case in self.oracle['cases'] if case['group'] in ('extra-fixtures', 'edge-fixtures')]
        self.assertEqual(len(cases), 95)
        for case in cases:
            with self.subTest(case=case['name']):
                self._compare(case)

    def test_actual_gapless_short_clips_vbr_abr_metadata_and_concatenation(self):
        cases = [case for case in self.oracle['cases'] if case['group'] == 'metadata-fixtures']
        self.assertEqual(len(cases), 98)
        for case in cases:
            with self.subTest(case=case['name']):
                self._compare(case)

    def test_native_scalefactor_compression_reuse_preflag_and_subblock_state(self):
        self.assertEqual(len(self.oracle['scalefactors']), 3168)
        for case in self.oracle['scalefactors']:
            with self.subTest(mpeg1=case['mpeg1'], kind=case['kind'], compression=case['compress'], intensity=case['intensity']):
                header = SimpleNamespace(mpeg1=case['mpeg1'], intensity=case['intensity'], ms=False)
                kind = case['kind']
                gr = SimpleNamespace(
                    n_long_sfb=0 if kind == 1 else (8 if case['mpeg1'] else 6) if kind == 2 else 22,
                    n_short_sfb=39 if kind == 1 else 30 if kind == 2 else 0,
                    scalefac_compress=case['compress'], global_gain=case['gain'],
                    scalefac_scale=case['scale'], preflag=case['preflag'],
                    scfsi=case['reuse'], subblock_gain=case['subblock'],
                )
                bits = _Bits(base64.b64decode(case['data'], validate=True))
                positions = list(case['positions'])
                actual = _scalefactors(header, positions, bits, gr, case['channel'])
                expected = struct.unpack_from('<' + str(case['native_scale_count']) + 'f', self.native_pcm, case['native_scale_offset'])
                self.assertEqual(bits.pos, case['native_consumed'])
                self.assertEqual(positions, case['native_positions'])
                self.assertEqual(len(actual), len(expected))
                self.assertTrue(all(abs(a - b) <= abs(b) * 4e-7 + 1e-20 for a, b in zip(actual, expected)))

    def test_free_format_superset_uses_separate_native_minimp3_golden(self):
        # Pinned PyAV's MP3 demuxer rejects these free-format streams. The
        # scalar minimp3 oracle independently decodes their exact native PCM.
        self.assertEqual(len(self.oracle['minimp3_cases']), 4)
        for case in self.oracle['minimp3_cases']:
            with self.subTest(case=case['name']):
                self._compare(case)

    def test_probe_leading_ape_junk_and_other_container_payload_boundaries(self):
        cases = [case for case in self.oracle['cases'] if case['group'] == 'prefix-fixtures']
        self.assertEqual(len(cases), 32)
        for case in cases:
            with self.subTest(case=case['name']):
                self.assertTrue(probe_mp3(base64.b64decode(case['data'])))
                self._compare(case)
        mp3 = base64.b64decode(self.oracle['cases'][0]['data'])
        containers = (b'RIFF' + bytes(24), b'OggS' + bytes(24), b'fLaC' + bytes(24),
                      b'\x1aE\xdf\xa3' + bytes(24), b'\xff\xf1' + bytes(24),
                      b'\x00\x00\x00\x18ftypisom' + bytes(12))
        for prefix in containers:
            with self.subTest(prefix=prefix):
                self.assertFalse(probe_mp3(prefix + mp3))
                with self.assertRaises(Mp3Error):
                    decode_mp3(prefix + mp3)
        self.assertFalse(probe_mp3(b'not MPEG audio' * 64))
        self.assertFalse(probe_mp3(None))

    def test_bad_crc_truncation_format_changes_and_source_rate_frame_budget(self):
        self.assertEqual(len(self.oracle['invalid']), 17)
        for case in self.oracle['invalid']:
            with self.subTest(case=case['name']), self.assertRaises(Mp3Error):
                decode_mp3(base64.b64decode(case['data'], validate=True))
        for name in ('gapless-8000-1', 'no-xing-full-untrimmed', 'vbr-44100-4'):
            case = next(case for case in self.oracle['cases'] if case['name'] == name)
            with self.subTest(case=name):
                self._compare(case, max_frames=case['samples'])
                with self.assertRaisesRegex(Mp3Error, 'budget'):
                    decode_mp3(base64.b64decode(case['data']), max_frames=case['samples'] - 1)
        with self.assertRaisesRegex(Mp3Error, 'budget'):
            decode_mp3(base64.b64decode(self.oracle['budget_forged_xing']), max_frames=3000)
        data = base64.b64decode(self.oracle['cases'][0]['data'])
        for budget in (-1, True, 2.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                decode_mp3(data, max_frames=budget)
        with self.assertRaises(TypeError):
            decode_mp3(None)


if __name__ == '__main__':
    unittest.main()
