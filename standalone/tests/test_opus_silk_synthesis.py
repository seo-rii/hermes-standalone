"""Independent Opus 1.3.1 C synthesis, PLC and comfort-noise golden states."""

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import zlib

from standalone import opus_silk_params as params
from standalone import opus_silk_synthesis as dsp


class OpusSilkSynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = json.loads((Path(__file__).parent / 'fixtures/opus_silk_synthesis.json').read_text())
        body = zlib.decompress(base64.b64decode(fixture['data_zlib_base64'], validate=True))
        if hashlib.sha256(body).hexdigest() != fixture['data_sha256']:
            raise AssertionError('Opus SILK synthesis fixture digest mismatch')
        cls.cases = json.loads(body)
        cls.fields = cls.cases['fields']
        cls.components = sum(1 if not shape else shape[0] if len(shape) == 1 else shape[0]*shape[1]
                             for _, shape in cls.fields)
        if cls.components != 1296:
            raise AssertionError('Incorrect native Opus SILK state dimensions')

    def restore(self, values):
        state, control = SimpleNamespace(), SimpleNamespace()
        offset = 0
        for name, shape in self.fields:
            parts = name.split('.')
            target = {'state': state, 'control': control}[parts[0]]
            for part in parts[1:-1]:
                if not hasattr(target, part):
                    setattr(target, part, SimpleNamespace())
                target = getattr(target, part)
            size = 1 if not shape else shape[0] if len(shape) == 1 else shape[0]*shape[1]
            items = list(values[offset:offset+size])
            offset += size
            if not shape:
                value = items[0]
            elif len(shape) == 1:
                value = items
            else:
                value = [items[i*shape[1]:(i+1)*shape[1]] for i in range(shape[0])]
            setattr(target, parts[-1], value)
        state.arch = 0
        return state, control

    def flatten(self, state, control):
        result = []
        for name, shape in self.fields:
            parts = name.split('.')
            value = {'state': state, 'control': control}[parts[0]]
            for part in parts[1:]:
                value = getattr(value, part)
            if not shape:
                result.append(value)
            elif len(shape) == 1:
                result.extend(value)
            else:
                result.extend(item for row in value for item in row)
        return result

    def test_native_initialization_preserves_parameter_state(self):
        self.assertEqual(len(self.cases['initialize']), 6)
        for case in self.cases['initialize']:
            with self.subTest(fs=case['fs'], subframes=case['sub']):
                state = params.initialize(case['fs'], case['sub'])
                indices = state.indices
                self.assertIs(dsp.initialize(state), state)
                self.assertIs(state.indices, indices)
                _, control = self.restore(case['output'])
                self.assertEqual(self.flatten(state, control), case['output'])

    def test_native_two_pass_energy_and_saturating_lpc_analysis(self):
        self.assertEqual(len(self.cases['energy']), 208)
        self.assertEqual(len(self.cases['analysis']), 288)
        for number, case in enumerate(self.cases['energy']):
            with self.subTest(operation='energy', number=number):
                self.assertEqual(list(dsp.sum_sqr_shift(case['input'])), case['output'])
        for number, case in enumerate(self.cases['analysis']):
            with self.subTest(operation='analysis', number=number):
                self.assertEqual(dsp.lpc_analysis_filter(case['input'], case['coefficients']), case['output'])

    def test_native_pcm_and_all_synthesis_plc_cng_state_components(self):
        self.assertGreaterEqual(len(self.cases['dsp']), 900)
        seen = set()
        for number, case in enumerate(self.cases['dsp']):
            with self.subTest(number=number, context=case['context']):
                state, control = self.restore(case['input'])
                mode = case['mode']
                seen.add(mode)
                tail = case['pcm'] + [0]*(320-len(case['pcm']))
                pcm = tail[:state.frame_length]
                if mode == 1:
                    pcm = dsp.decode_core(state, control, case['pulses'])
                elif mode == 4:
                    self.assertIs(dsp.cng(state, control, pcm), pcm)
                elif mode == 5:
                    self.assertIs(dsp.glue_frames(state, pcm), pcm)
                elif mode in (6, 7):
                    if mode == 6:
                        pcm = dsp.decode_core(state, control, case['pulses'])
                        self.assertIsNone(dsp.plc(state, control, False))
                        state.lossCnt = 0
                        state.prevSignalType = state.indices.signalType
                        state.first_frame_after_reset = 0
                    else:
                        state.indices.signalType = state.prevSignalType
                        pcm = dsp.plc(state, control, True)
                    move = state.ltp_mem_length-state.frame_length
                    state.outBuf[:move] = state.outBuf[state.frame_length:state.frame_length+move]
                    state.outBuf[move:move+state.frame_length] = pcm
                    self.assertIs(dsp.cng(state, control, pcm), pcm)
                    self.assertIs(dsp.glue_frames(state, pcm), pcm)
                    state.lagPrev = control.pitchL[state.nb_subfr-1]
                else:
                    self.fail('Unknown native DSP operation')
                actual = self.flatten(state, control) + pcm + tail[state.frame_length:]
                self.assertEqual(actual, case['output'])
        self.assertEqual(seen, {1, 4, 5, 6, 7})


if __name__ == '__main__':
    unittest.main()
