"""Independent original FFmpeg GENERIC/host SBR/PS golden vectors; stdlib only."""
import base64
import hashlib
import importlib.abc
import json
import lzma
import math
from pathlib import Path
import random
import struct
import sys
from types import SimpleNamespace
import unittest
import zlib

from standalone.local_whisper_audio_aac import AacError, decode_aac, probe_aac
from standalone.local_whisper_audio_aac_bits import Bits
from standalone.local_whisper_audio_aac_dsp import f32
from standalone.local_whisper_audio_aac_ps import Ps
from standalone.local_whisper_audio_aac_sbr_bits import Sbr, Channel
from standalone.local_whisper_audio_aac_sbr_dsp import inverse_filter, dequant, _chirp, _estimate, _gain
from standalone.local_whisper_audio_aac_sbr_frequency import Frequency, Parameters
from standalone.local_whisper_audio_aac_sbr_qmf import Analysis, Synthesis
from standalone.local_whisper_audio_aac_sbr_transform import imdct


def _flat(rows):
    return [v for row in rows for v in row]


def _complex_flat(rows):
    return [v for row in rows for z in row for v in (z.real, z.imag)]


def _complex_rows(values, width):
    return [[complex(values[i + k], values[i + k + 1]) for k in range(0, width * 2, 2)]
            for i in range(0, len(values), width * 2)]


def _digest(values):
    # The physical PCM reference regards +0 and -0 as the same float value.
    values = [v if v else 0. for v in values]
    return hashlib.sha256(struct.pack('<' + str(len(values)) + 'f', *values)).hexdigest()


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {
            'av', 'numpy', 'ctranslate2', 'tokenizers', 'onnxruntime', 'torch',
            'ctypes', 'subprocess', 'pydub', 'soundfile',
        }:
            raise AssertionError('HE-AAC port attempted native or process import ' + fullname)


class HeAacDecodeTests(unittest.TestCase):
    def setUp(self):
        guard = NoNative()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    @classmethod
    def setUpClass(cls):
        wrapper = json.loads((Path(__file__).parent / 'fixtures/local_whisper_audio_he_aac.json').read_text())
        metadata = zlib.decompress(base64.b64decode(wrapper['metadata_zlib_base64'], validate=True))
        if hashlib.sha256(metadata).hexdigest() != wrapper['metadata_sha256']:
            raise AssertionError('HE-AAC metadata digest mismatch')
        cls.oracle = json.loads(metadata)
        shuffled = lzma.decompress(base64.b64decode(wrapper['word_pool_shuffle_lzma_base64'], validate=True), memlimit=32 * 1024 * 1024)
        if len(shuffled) != wrapper['word_pool_bytes'] or len(shuffled) % 4:
            raise AssertionError('HE-AAC native word pool byte count mismatch')
        count = len(shuffled) // 4
        pool = bytearray(len(shuffled))
        for i in range(4):
            pool[i::4] = shuffled[i * count:(i + 1) * count]
        if hashlib.sha256(pool).hexdigest() != wrapper['word_pool_sha256']:
            raise AssertionError('HE-AAC native word pool digest mismatch')
        cls.pool = pool

    def _references(self, case):
        count = case['samples'] * case['channels'] * 4
        width = case['samples'] * 4
        host_chunks = [self.pool[start:start + width] for start in case['host_offsets']]
        host = b''.join(host_chunks)
        generic_chunks = []
        for channel, chunk in enumerate(host_chunks):
            start = case.get('generic_xor_offsets', [None] * case['channels'])[channel]
            generic_chunks.append(chunk if start is None else bytes(a ^ b for a, b in zip(chunk, self.pool[start:start + width])))
        generic = b''.join(generic_chunks)
        self.assertEqual(hashlib.sha256(host).hexdigest(), case['host_sha256'])
        self.assertEqual(hashlib.sha256(generic).hexdigest(), case['generic_sha256'])
        fmt = '<' + str(count // 4) + 'f'
        return struct.unpack(fmt, host), struct.unpack(fmt, generic)

    def _file(self, case):
        data = base64.b64decode(case['data'], validate=True)
        self.assertTrue(probe_aac(data))
        rate, pcm, fmt, mask = decode_aac(data)
        self.assertEqual((rate, len(pcm), fmt, mask), (case['rate'], case['channels'], 'fltp', case['mask']))
        self.assertTrue(all(len(row) == case['samples'] for row in pcm))
        values = _flat(pcm)
        self.assertTrue(all(math.isfinite(v) for v in values))
        host, generic = self._references(case)
        if case['generic_exact']:
            self.assertEqual(values, list(generic))
        else:
            self.assertLessEqual(max(abs(a - b) for a, b in zip(values, generic)), 2e-6)
        if case['group'] != 'syntax':
            self.assertLessEqual(max(abs(a - b) for a, b in zip(values, host)), 2e-6)

    def test_native_he_hev2_nine_rates_and_multichannel(self):
        cases = [c for c in self.oracle['files'] if c['group'] in ('encoded', 'multichannel')]
        self.assertEqual(len(cases), 105)
        for case in cases:
            with self.subTest(case=case['name']):
                self._file(case)

    def test_native_explicit_sync_downsample_header_history_and_mp4_edits(self):
        cases = [c for c in self.oracle['files'] if c['group'] in ('container', 'edits', 'historical')]
        self.assertEqual(len(cases), 208)
        for case in cases:
            with self.subTest(case=case['name']):
                self._file(case)

    def test_all_retained_sbr_stress_streams_match_actual_generic(self):
        cases = [c for c in self.oracle['files'] if c['group'] == 'syntax']
        self.assertEqual(len(cases), 48)
        for case in cases:
            with self.subTest(case=case['name']):
                self.assertTrue(case['generic_exact'])
                self._file(case)

    def test_original_changed_format_and_output_rate_sample_budgets(self):
        self.assertEqual(len(self.oracle['changing_format']), 27)
        for case in self.oracle['changing_format']:
            with self.subTest(case=case['name']), self.assertRaisesRegex(AacError, 'format changes'):
                decode_aac(base64.b64decode(case['data'], validate=True))
        for name in ('aot5-16000-1-noise', 'aot29-48000-2-noise',
                     'aot29-48000-2-noise-mp4-downsample', 'aot5-44100-2-noise-mp4-explicit-edit-512'):
            case = next(c for c in self.oracle['files'] if c['name'] == name)
            data = base64.b64decode(case['data'], validate=True)
            with self.subTest(case=name):
                actual = decode_aac(data, max_frames=case['samples'], max_seconds=1)
                self.assertEqual(len(actual[1][0]), case['samples'])
                with self.assertRaisesRegex(AacError, 'budget'):
                    decode_aac(data, max_frames=case['samples'] - 1)
                with self.assertRaisesRegex(AacError, 'budget'):
                    decode_aac(data, max_seconds=0)

    def test_native_generic_mdct_qmf_and_frequency_tables(self):
        for case in self.oracle['transforms']:
            with self.subTest(mdct=case['length'], scale=case['scale']):
                self.assertEqual(imdct(case['input'], scale=case['scale']), case['output'])
        states = {}
        for case in self.oracle['qmf']:
            if case['kind'] == 'analysis':
                key = 'analysis', case['channel']
                state = states.setdefault(key, Analysis())
                self.assertEqual(_complex_flat(state.run(case['input'])), case['output'])
            else:
                key = 'synthesis', case['downsample'], case['stream']
                state = states.setdefault(key, Synthesis())
                self.assertEqual(state.run(_complex_rows(case['input'], 64), downsample=case['downsample']), case['output'])
                self.assertEqual(state.offset, case['offset'])
            self.assertEqual(state.samples, case['state'])
        for case in self.oracle['frequency']:
            rate, start, stop, scale, alter, crossover, noise, limiter = case['args']
            with self.subTest(frequency=case['args']):
                if case['expected'] is None:
                    with self.assertRaises(ValueError):
                        Frequency(rate, Parameters(start, stop, crossover, scale, alter, noise), limiter)
                    continue
                freq = Frequency(rate, Parameters(start, stop, crossover, scale, alter, noise), limiter)
                for name, value in case['expected'].items():
                    actual = getattr(freq, name)
                    self.assertEqual([list(pair) for pair in actual] if name == 'patches' else actual, value)

    def test_native_ps_all_parameters_phase_and_complete_dsp_history(self):
        fields = ('start', 'enable_iid', 'iid_quant', 'nr_iid_par', 'nr_ipdopd_par', 'enable_icc', 'icc_mode',
                  'nr_icc_par', 'enable_ext', 'frame_class', 'num_env_old', 'num_env', 'enable_ipdopd', 'is34bands', 'is34bands_old')
        contexts = {}
        for case in self.oracle['ps']:
            state = contexts.get(case['context'])
            if state is None:
                state = contexts[case['context']] = Ps()
            bits = Bits(base64.b64decode(case['data'], validate=True))
            bits.skip(case['start'])
            self.assertEqual(state.read(bits, case['left']), case['count'])
            actual = [getattr(state, k) for k in fields] + state.border_position
            actual += [v for k in ('iid_par', 'icc_par', 'ipd_par', 'opd_par') for row in getattr(state, k) for v in row]
            self.assertEqual(actual, case['state'])
            if 'input_seed' not in case:
                continue
            rng = random.Random(case['input_seed'])
            rows = [[complex(f32(rng.uniform(-1, 1)), f32(rng.uniform(-1, 1))) for _ in range(64)] for _ in range(38)]
            if case['input_zero']:
                rows = [[0j] * 64 for _ in range(38)]
            left, right = state.apply(rows, case['top'])
            self.assertEqual(_digest(_complex_flat(left) + _complex_flat(right)), case['output_sha256'])
            values = _complex_flat(state.in_buf) + _complex_flat(state.delay)
            values += [v for rows in state.ap_delay for v in _complex_flat(rows)]
            values += state.peak_decay_nrg + state.power_smooth + state.peak_decay_diff_smooth
            values += [v for component in state.h for part in component for row in part for v in row]
            self.assertEqual(_digest(values), case['dsp_sha256'])
            self.assertEqual(state.ipd_hist + state.opd_hist, case['phase'])

    def test_original_scalar_sbr_dsp_operations_on_identical_inputs(self):
        for case in self.oracle['dsp']:
            name = case['operation']
            with self.subTest(operation=name):
                if name == 'inverse':
                    values = inverse_filter(_complex_rows(case['input'], 40), case['count'])
                    actual = [v for pair in values for z in pair for v in (z.real, z.imag)]
                elif name == 'dequant':
                    values = iter(case['input'])
                    state = Sbr(24000)
                    state.coupling, low, high, noise = (next(values) for _ in range(4))
                    state.frequency = SimpleNamespace(low=[0] * (low + 1), high=[0] * (high + 1), noise=[0] * (noise + 1))
                    for channel in state.channels:
                        channel.count, channel.noise_count, channel.amp_res = (next(values) for _ in range(3))
                        channel.freq_res = [next(values) for _ in range(9)]
                        channel.env_q = [[next(values) for _ in range(48)] for _ in range(9)]
                        channel.noise_q = [[next(values) for _ in range(5)] for _ in range(3)]
                    actual = [v for group in dequant(state, case['kind']) for ch in group for row in ch for v in row]
                elif name == 'chirp':
                    channel = Channel()
                    channel.invf = [case['modes'][:5], case['modes'][5:]]
                    actual = case['input'][:]
                    _chirp(channel, actual, case['count'])
                elif name == 'energy':
                    values = iter(case['input'])
                    interpolate, kx, m, low, high, count = (next(values) for _ in range(6))
                    channel = Channel()
                    channel.count = count
                    channel.time = [next(values) for _ in range(9)]
                    channel.freq_res = [next(values) for _ in range(9)]
                    low_table = [next(values) for _ in range(25)]
                    high_table = [next(values) for _ in range(49)]
                    freq = SimpleNamespace(kx=kx, m=m, low=low_table[:low + 1], high=high_table[:high + 1])
                    actual = _flat(_estimate(_complex_rows(case['high'], 40), channel, SimpleNamespace(frequency=freq, interpolate=interpolate)))
                else:
                    self.assertEqual(name, 'gain')
                    values = iter(case['input'])
                    kx, m, limits, limiter, count = (next(values) for _ in range(5))
                    channel = Channel()
                    channel.count, channel.e_a = count, [next(values), next(values)]
                    table = [next(values) for _ in range(30)]
                    index = [[next(values) for _ in range(48)] for _ in range(9)]
                    sines = [[next(values) for _ in range(48)][:m] for _ in range(8)][:count]
                    source = iter(case['floats'])
                    energy, floors, estimated = ([[[next(source) for _ in range(48)][:m] for _ in range(8)][:count] for _ in range(3)])
                    state = SimpleNamespace(frequency=SimpleNamespace(kx=kx, m=m, limits=table[:limits + 1]), limiter_gain=limiter)
                    actual = [v for group in _gain(channel, SimpleNamespace(s_index=index), state, energy, floors, sines, estimated) for row in group for v in row]
                self.assertEqual(actual, case['expected'])


if __name__ == '__main__':
    unittest.main()
