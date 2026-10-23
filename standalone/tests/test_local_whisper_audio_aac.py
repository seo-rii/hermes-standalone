"""Independent pinned native AAC/AVTX golden fixtures; stdlib-only runtime."""
import base64
import hashlib
import json
import lzma
import math
from pathlib import Path
import struct
import unittest
import zlib

from standalone.local_whisper_audio_aac import AacError, UnsupportedAac, decode_aac, probe_aac
from standalone.local_whisper_audio_aac_codec import AacDecoder
from standalone.local_whisper_audio_aac_config import parse_config
from standalone.local_whisper_audio_aac_dsp import imdct, _window
from standalone.local_whisper_audio_aac_tables import POW43


class AacDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wrapper = json.loads((Path(__file__).parent / 'fixtures/local_whisper_audio_aac.json').read_text())
        metadata = zlib.decompress(base64.b64decode(wrapper['metadata_zlib_base64'], validate=True))
        if hashlib.sha256(metadata).hexdigest() != wrapper['metadata_sha256']:
            raise AssertionError('Native AAC metadata digest mismatch')
        cls.oracle = json.loads(metadata)
        shuffled = lzma.decompress(base64.b64decode(wrapper['native_pcm_shuffle_lzma_base64'], validate=True), memlimit=32 * 1024 * 1024)
        if len(shuffled) != wrapper['native_pcm_bytes'] or len(shuffled) % 4:
            raise AssertionError('Native AAC float byte count mismatch')
        count = len(shuffled) // 4
        pcm = bytearray(len(shuffled))
        for i in range(4):
            pcm[i::4] = shuffled[i * count:(i + 1) * count]
        if hashlib.sha256(pcm).hexdigest() != wrapper['native_pcm_sha256']:
            raise AssertionError('Native AAC float PCM digest mismatch')
        cls.native_pcm = pcm

    def _compare_pcm(self, case, actual):
        self.assertEqual((actual[0], len(actual[1]), actual[2], actual[3]),
                         (case['rate'], case['channels'], 'fltp', case['mask']))
        count = case['samples']
        self.assertTrue(all(len(values) == count for values in actual[1]))
        expected = struct.unpack_from('<' + str(count * case['channels']) + 'f', self.native_pcm, case['native_f32le_offset'])
        error = 0.
        for channel, values in enumerate(actual[1]):
            self.assertTrue(all(math.isfinite(value) for value in values))
            error = max(error, max(abs(a - b) for a, b in zip(values, expected[channel * count:(channel + 1) * count])))
        self.assertLessEqual(error, case['tolerance'])

    def test_actual_encoder_all_thirteen_lc_rates_pce_layouts_and_signals(self):
        cases = [case for case in self.oracle['files'] if case['group'] in ('native-files', 'expanded-files')]
        self.assertEqual(len(cases), 163)
        for case in cases:
            with self.subTest(case=case['name']):
                data = base64.b64decode(case['data'], validate=True)
                self.assertTrue(probe_aac(data))
                self._compare_pcm(case, decode_aac(data))

    def test_native_mp4_tables_fragments_edits_itunes_and_960_file_frames(self):
        cases = [case for case in self.oracle['files'] if case['group'] in ('container-files', 'synthetic-files')]
        self.assertEqual(len(cases), 54)
        for case in cases:
            with self.subTest(case=case['name']):
                self._compare_pcm(case, decode_aac(base64.b64decode(case['data'], validate=True)))

    def test_native_huffman_windows_noise_intensity_pulse_tns_and_coupling(self):
        self.assertEqual(len(self.oracle['raw']), 266)
        for case in self.oracle['raw']:
            with self.subTest(case=case['name']):
                decoder = AacDecoder(parse_config(base64.b64decode(case['asc'], validate=True)))
                actual, skip = [[] for _ in range(case['channels'])], 0
                for data in case['frames']:
                    values = decoder.decode(base64.b64decode(data, validate=True))
                    self.assertEqual(len(values), case['channels'])
                    skip = max(skip, decoder.skip_samples)
                    decoder.skip_samples = 0
                    start = min(skip, len(values[0]))
                    skip -= start
                    for target, channel in zip(actual, values):
                        target.extend(channel[start:])
                self._compare_pcm(case, (decoder.config.rate, actual, 'fltp', decoder.mask))

    def test_native_mdct_windows_and_all_8192_dequantization_words(self):
        cases = self.oracle['transforms']
        self.assertEqual(len(cases), 100)
        for case in cases:
            with self.subTest(length=case['length']):
                actual = imdct(case['values'])
                expected = struct.unpack_from('<' + str(case['length']) + 'f', self.native_pcm, case['output_offset'])
                error = max(abs(a - b) for a, b in zip(actual, expected))
                self.assertLessEqual(error, max(2e-12, max(abs(value) for value in expected) * 8e-7))
        self.assertEqual(len(self.oracle['windows']), 8)
        for case in self.oracle['windows']:
            with self.subTest(window=case['length'], shape=case['shape']):
                expected = struct.unpack_from('<' + str(case['length']) + 'f', self.native_pcm, case['window_offset'])
                actual = _window(case['length'], case['shape'])
                self.assertEqual(len(actual), len(expected))
                if case['shape']:
                    self.assertEqual(actual, expected)
                else:
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, expected)), 6e-8)
        offset = self.oracle['pow43_offset']
        self.assertEqual(struct.pack('<8192f', *POW43), self.native_pcm[offset:offset + 8192 * 4])

    def test_native_profile_refusal_and_malformed_input_outcomes(self):
        self.assertEqual(len(self.oracle['invalid']['cases']), 16)
        for case in self.oracle['invalid']['cases']:
            if case['candidate'] == 'decoded':
                self._compare_pcm(case, decode_aac(base64.b64decode(case['data'], validate=True)))
                continue
            expected = UnsupportedAac if case['candidate'] == 'UnsupportedAac' else AacError
            with self.subTest(case=case['name']), self.assertRaises(expected):
                decode_aac(base64.b64decode(case['data'], validate=True))
        self.assertEqual(len(self.oracle['invalid']['configs']), 12)
        for case in self.oracle['invalid']['configs']:
            if case['candidate'] == 'Config':
                config = parse_config(base64.b64decode(case['data'], validate=True))
                for name, value in case['expected'].items():
                    self.assertEqual(getattr(config, name), value)
                continue
            expected = UnsupportedAac if case['candidate'] == 'UnsupportedAac' else AacError
            with self.subTest(config=case['name']), self.assertRaises(expected):
                parse_config(base64.b64decode(case['data'], validate=True))

    def test_source_rate_budget_delay_work_and_other_container_boundaries(self):
        for name in ('adts-8000-1-tone', 'ipod-48000-2-tone',
                     'movflags-frag_keyframe+delay_moov-2', 'synthetic-mp4-cb11-960'):
            case = next(case for case in self.oracle['files'] if case['name'] == name)
            data = base64.b64decode(case['data'], validate=True)
            with self.subTest(case=name):
                self._compare_pcm(case, decode_aac(data, max_frames=case['samples'], max_seconds=1))
                with self.assertRaisesRegex(AacError, 'budget'):
                    decode_aac(data, max_frames=case['samples'] - 1)
                with self.assertRaisesRegex(AacError, 'budget'):
                    decode_aac(data, max_seconds=0)
        with self.assertRaisesRegex(AacError, 'work.*budget'):
            decode_aac(base64.b64decode(self.oracle['invalid']['forged_delay']), max_frames=1000)
        data = base64.b64decode(self.oracle['files'][0]['data'])
        for budget in (-1, True, 2.5):
            with self.subTest(budget=budget), self.assertRaises(AacError):
                decode_aac(data, max_frames=budget)
            with self.subTest(seconds=budget), self.assertRaises(AacError):
                decode_aac(data, max_seconds=budget)
        for prefix in (b'RIFF', b'OggS', b'fLaC', b'ADIF', b'\x1aE\xdf\xa3'):
            with self.subTest(prefix=prefix):
                self.assertFalse(probe_aac(prefix + bytes(24) + data))
                with self.assertRaises(AacError):
                    decode_aac(prefix + bytes(24) + data)
        self.assertFalse(probe_aac(b'not AAC' * 64))
        self.assertFalse(probe_aac(None))
        with self.assertRaises(AacError):
            decode_aac(None)


if __name__ == '__main__':
    unittest.main()
