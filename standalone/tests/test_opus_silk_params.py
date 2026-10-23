"""Independent fixed Opus 1.3.1 integer/range golden results, without libopus."""
import base64
import hashlib
import json
from pathlib import Path
import unittest
import zlib

from standalone import opus_silk_lpc as lpc
from standalone import opus_silk_params as params
from standalone.opus_range import OpusRangeDecoder


class OpusSilkParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture=json.loads((Path(__file__).parent/'fixtures/opus_silk_params.json').read_text())
        data=zlib.decompress(base64.b64decode(fixture['data_zlib_base64'],validate=True))
        if hashlib.sha256(data).hexdigest()!=fixture['data_sha256']:
            raise AssertionError('Opus SILK fixture digest mismatch')
        cls.cases=json.loads(data)

    def test_native_nlsf_codebooks_lpc_fit_stability_and_chirps(self):
        self.assertEqual(len(self.cases['lpc']),2952)
        for number,case in enumerate(self.cases['lpc']):
            with self.subTest(number=number,kind=case['kind']):
                values=case['input'];kind=case['kind']
                if kind=='unpack':actual=[list(a) for a in lpc.unpack(*values)]
                elif kind=='decode':actual=lpc.decode_nlsf(values)
                elif kind=='nlsf2a':actual=lpc.nlsf2a(values)
                elif kind=='inverse':actual=lpc.inverse_gain(values)
                elif kind=='bw':actual=lpc.bwexpander32(list(values[0]),values[1])
                elif kind=='fit':
                    mutable=list(values);actual=[lpc.fit(mutable),mutable]
                else:self.fail('Unknown native LPC fixture operation')
                self.assertEqual(actual,case['output'])

    def test_native_indices_pulses_parameters_and_complete_range_state(self):
        self.assertEqual(len(self.cases['params']),720)
        fields=('storage','end_offs','end_window','nend_bits','nbits_total','offs','rng','val','ext','rem','error')
        for number,case in enumerate(self.cases['params']):
            with self.subTest(number=number):
                fs,sub,conditional,vad,lbrr,previous,lag,gain,first,loss=case['settings']
                state=params.initialize(fs,sub)
                state.VAD_flags[0]=vad;state.ec_prevSignalType=previous;state.ec_prevLagIndex=lag
                state.LastGainIndex=gain;state.first_frame_after_reset=first;state.lossCnt=loss
                state.prevNLSF_Q15[:]=case['previous_nlsf']
                decoder=OpusRangeDecoder(base64.b64decode(case['payload'],validate=True))
                params.decode_indices(state,decoder,0,lbrr,conditional)
                pulses=params.decode_pulses(decoder,state.indices.signalType,state.indices.quantOffsetType,state.frame_length)
                control=params.decode_parameters(state,conditional)
                indices=state.indices
                actual=indices.GainsIndices+indices.NLSFIndices+indices.LTPIndex
                actual+=[indices.lagIndex,indices.contourIndex,indices.PERIndex,indices.LTP_scaleIndex,
                         indices.NLSFInterpCoef_Q2,indices.Seed,indices.signalType,indices.quantOffsetType]
                actual+=control.Gains_Q16+control.PredCoef_Q12[0]+control.PredCoef_Q12[1]+control.pitchL+control.LTPCoef_Q14+[control.LTP_scale_Q14]
                actual+=state.prevNLSF_Q15+[state.ec_prevSignalType,state.ec_prevLagIndex,state.LastGainIndex]+pulses
                actual+=[getattr(decoder,name) for name in fields]+[decoder.tell(),decoder.tell_frac()]
                self.assertEqual(actual,case['output'])


if __name__=='__main__':
    unittest.main()
