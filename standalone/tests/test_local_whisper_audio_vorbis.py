"""Independent pinned original Vorbis native goldens; standard library only."""
import base64
import hashlib
import importlib.abc
import json
import lzma
import math
from pathlib import Path
import struct
import sys
import unittest
import zlib


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {
            'av', 'numpy', 'ctranslate2', 'tokenizers', 'onnxruntime', 'torch',
            'ctypes', 'subprocess', 'soundfile', 'pydub', 'audioread',
        }:
            raise AssertionError('Vorbis port attempted native or process import ' + fullname)


class VorbisDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guard = NoNative()
        sys.meta_path.insert(0, guard)
        cls.addClassCleanup(sys.meta_path.remove, guard)
        from standalone.local_whisper_audio_vorbis import decode_vorbis, probe_vorbis
        from standalone.local_whisper_audio_vorbis_bits import VorbisError, codes_from_lengths
        from standalone.local_whisper_audio_vorbis_codec import VorbisDecoder
        from standalone.local_whisper_audio_vorbis_floor import _render_line
        from standalone.local_whisper_audio_vorbis_tables import WINDOWS, INVERSE_DB
        cls.decode = staticmethod(decode_vorbis)
        cls.probe = staticmethod(probe_vorbis)
        cls.error = VorbisError
        cls.decoder = VorbisDecoder
        cls.codes = staticmethod(codes_from_lengths)
        cls.render = staticmethod(_render_line)
        cls.windows, cls.inverse_db = WINDOWS, INVERSE_DB
        wrapper = json.loads((Path(__file__).parent / 'fixtures/local_whisper_audio_vorbis.json').read_text())
        metadata = zlib.decompress(base64.b64decode(wrapper['metadata_zlib_base64'], validate=True))
        if hashlib.sha256(metadata).hexdigest() != wrapper['metadata_sha256']:
            raise AssertionError('Vorbis metadata digest mismatch')
        cls.oracle = json.loads(metadata)
        shuffled = lzma.decompress(base64.b64decode(wrapper['word_pool_shuffle_lzma_base64'], validate=True),
                                   memlimit=32 << 20)
        if len(shuffled) != wrapper['word_pool_bytes'] or len(shuffled) % 4:
            raise AssertionError('Vorbis PCM word count mismatch')
        words = len(shuffled) // 4
        cls.pool = bytearray(len(shuffled))
        for i in range(4):
            cls.pool[i::4] = shuffled[i * words:(i + 1) * words]
        if hashlib.sha256(cls.pool).hexdigest() != wrapper['word_pool_sha256']:
            raise AssertionError('Vorbis PCM word pool digest mismatch')
        for row in cls.oracle['blobs']:
            data = cls.pool[row['offset']:row['offset'] + row['bytes']]
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise AssertionError('Vorbis original native PCM blob mismatch')

    @classmethod
    def tearDownClass(cls):
        del cls.pool, cls.oracle

    def _bytes(self, ref):
        row = self.oracle['blobs'][ref['pcm_blob']]
        return self.pool[row['offset']:row['offset'] + row['bytes']]

    def _pcm(self, ref):
        data = self._bytes(ref)
        return struct.unpack('<' + str(len(data) // 4) + 'f', data)

    def _restore(self, value):
        if isinstance(value, dict):
            if set(value) == {'pcm_blob'}:
                return base64.b64encode(self._bytes(value)).decode()
            return {k: self._restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._restore(v) for v in value]
        return value

    def _assert_values(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        if all(math.isfinite(v) for v in expected):
            self.assertEqual(list(actual), list(expected))
            return
        for a, b in zip(actual, expected):
            if math.isnan(b):
                self.assertTrue(math.isnan(a))
            else:
                self.assertEqual(a, b)

    def _file(self, case):
        data = base64.b64decode(case['data'], validate=True)
        self.assertTrue(self.probe(data))
        rate, pcm, fmt, mask = self.decode(data)
        native_masks = {'mono': 4, 'stereo': 3, '3.0': 7, 'quad': 0x33,
                        '5.0': 0x37, '5.1': 0x3f, '6.1': 0x70f, '7.1': 0x63f}
        self.assertEqual((rate, len(pcm), fmt), (case['rate'], case['channels'], 'fltp'))
        if 'layout' in case:
            self.assertEqual(mask, native_masks[case['layout']])
        for row, target in zip(pcm, case['generic']):
            self.assertEqual(len(row), case['samples'])
            self._assert_values(row, self._pcm(target))

    def test_native_xiph_all_seven_rates_eight_channel_counts_three_signals(self):
        group = self.oracle['groups']['xiph']
        self.assertEqual(group['encoder_errors'], [])
        self.assertEqual(len(group['cases']), 168)
        for case in group['cases']:
            with self.subTest(case=case['name']):
                self._file(case)
                for generic, host in zip(case['generic'], case['host']):
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(self._pcm(generic), self._pcm(host))), 3e-7)

    def test_native_independent_ffmpeg_encoder_and_ogg_page_and_chain_history(self):
        for group, count in (('ffmpeg', 21), ('container', 71)):
            cases = self.oracle['groups'][group]['cases']
            self.assertEqual(len(cases), count)
            for case in cases:
                with self.subTest(group=group, case=case['name']):
                    self._file(case)

    def _syntax(self, group, count=276):
        cases = self.oracle['groups'][group]['cases']
        self.assertEqual(len(cases), count)
        for case in cases:
            with self.subTest(case=case['name']):
                headers = [base64.b64decode(v, validate=True) for v in case['headers']]
                decoder = self.decoder(headers[0], headers[2])
                expected = [[] for _ in range(case['channels'])]
                self.assertEqual(len(case['packets']), 8)
                for packet, target in zip(case['packets'], case['generic_frames']):
                    output = decoder.decode(base64.b64decode(packet, validate=True))
                    self.assertEqual(len(output), case['channels'])
                    for actual, reference, dest in zip(output, target, expected):
                        values = self._pcm(reference)
                        self._assert_values(actual, values)
                        dest.extend(values)
                data = base64.b64decode(case['data'], validate=True)
                self.assertEqual(hashlib.sha256(data).hexdigest(), case['sha256'])
                rate, pcm, fmt, mask = self.decode(data)
                self.assertEqual((rate, len(pcm), fmt), (case['rate'], case['channels'], 'fltp'))
                for actual, reference in zip(pcm, expected):
                    self._assert_values(actual, reference)

    def test_native_finite_floor0_floor1_residue012_and_channel_counts_to_nine(self):
        self._syntax('syntax_finite')

    def test_native_supplemental_blocks_complete_all_eight_legal_sizes(self):
        self._syntax('syntax_blocks', 42)

    def test_retained_native_singular_floor0_nan_and_infinity_classification(self):
        # The actual source emits NaNs for some singular LSP products. Preserve
        # all original words in the fixture; compare class, not a NaN payload.
        self._syntax('syntax_singular')

    def test_native_huffman_floor1_render_and_all_window_inverse_db_words(self):
        cases = self.oracle['groups']['units']
        self.assertEqual(len(cases['codebooks']), 1210)
        for case in cases['codebooks']:
            if case['codes'] is None:
                with self.assertRaises(self.error):
                    self.codes(case['lengths'])
            else:
                self.assertEqual(self.codes(case['lengths']), case['codes'])
        self.assertEqual(len(cases['render']), 1600)
        for case in cases['render']:
            values = [0.] * case['args'][2]
            self.render(*case['args'], values)
            self._assert_values(values, self._pcm(case['output']))
        values = [v for size in sorted(self.windows) for v in self.windows[size]] + list(self.inverse_db)
        self.assertEqual(len(values), 8416)
        self.assertEqual(hashlib.sha256(struct.pack('<8416f', *values)).hexdigest(), cases['tables_sha256'])

    def test_original_header_rejections_and_original_accepted_unknown_header(self):
        group = self.oracle['groups']['headers']
        self.assertEqual(len(group['errors']), 23)
        for case in group['errors']:
            with self.subTest(case=case['name']), self.assertRaises(self.error):
                self.decode(base64.b64decode(case['data'], validate=True))
        self.assertEqual(len(group['accepted']), 1)
        for case in group['accepted']:
            self._file(case)

    def test_original_demux_and_decoder_initialization_failure_phases(self):
        outcomes = {c['name']: c for c in self.oracle['groups']['frontend_headers']['cases']}
        cases = self.oracle['groups']['headers']['errors']
        self.assertEqual(sum('samples' in c['generic'] for c in outcomes.values()), 11)
        for case in cases:
            with self.subTest(case=case['name']):
                try:
                    self.decode(base64.b64decode(case['data'], validate=True))
                except self.error as exc:
                    self.assertEqual(exc.invalid_frame, 'samples' in outcomes[case['name']]['generic'])
                else:
                    self.fail('Original rejected header was accepted')

    def test_output_frame_and_seconds_budgets_and_non_vorbis_probe(self):
        self.assertFalse(self.probe(b'RIFF\x00\x00\x00\x00WAVE'))
        for name in ('xiph-44100-1-tone', 'xiph-96000-8-noise'):
            case = next(c for c in self.oracle['groups']['xiph']['cases'] if c['name'] == name)
            data = base64.b64decode(case['data'], validate=True)
            result = self.decode(data, max_frames=case['samples'], max_seconds=1)
            self.assertEqual(len(result[1][0]), case['samples'])
            with self.assertRaisesRegex(self.error, 'budget'):
                self.decode(data, max_frames=case['samples'] - 1)
            with self.assertRaisesRegex(self.error, 'budget'):
                self.decode(data, max_seconds=0)

    def test_lossless_fixture_restores_every_original_native_record(self):
        for name, group in self.oracle['groups'].items():
            restored = json.dumps(self._restore(group), sort_keys=True, separators=(',', ':')).encode()
            self.assertEqual(hashlib.sha256(restored).hexdigest(), self.oracle['sources'][name]['canonical_sha256'])


if __name__ == '__main__':
    unittest.main()
