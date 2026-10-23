"""Independent native Opus SILK delay/filter/stereo results with stdlib only."""
import base64
import hashlib
import json
from pathlib import Path
import unittest
import zlib

from standalone.opus_range import OpusRangeDecoder
from standalone.opus_silk_resample import Resampler
from standalone import opus_silk_stereo as stereo


class OpusSilkStereoResamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture=json.loads((Path(__file__).parent/'fixtures/opus_silk_stereo_resample.json').read_text())
        data=zlib.decompress(base64.b64decode(fixture['data_zlib_base64'],validate=True))
        if hashlib.sha256(data).hexdigest()!=fixture['data_sha256']:
            raise AssertionError('Opus SILK resampling fixture digest mismatch')
        cls.cases=json.loads(data)

    def test_native_all_thirty_rate_pairs_delays_partial_batches_and_filter_state(self):
        self.assertEqual(len(self.cases['resample']),30)
        fields=('resampler_function','batchSize','invRatio_Q16','FIR_Order','FIR_Fracs','Fs_in_kHz','Fs_out_kHz','inputDelay')
        for number,stream in enumerate(self.cases['resample']):
            candidate=Resampler(*stream['rates'],encoder=stream['encoder'])
            for index,frame in enumerate(stream['frames']):
                with self.subTest(stream=number,frame=index):
                    self.assertEqual(candidate.process(frame['input']),frame['output'])
                    state=candidate.sIIR+candidate.sFIR+candidate.delayBuf+[getattr(candidate,name) for name in fields]
                    self.assertEqual(state,frame['state'])

    def test_native_adaptive_stereo_saturation_and_delayed_predictor_state(self):
        self.assertEqual(len(self.cases['stereo']),240)
        for number,frame in enumerate(self.cases['stereo']):
            with self.subTest(frame=number):
                state=stereo.initialize();initial=frame['initial']
                state.pred_prev_Q13[:]=initial[:2];state.sMid[:]=initial[2:4];state.sSide[:]=initial[4:]
                output=stereo.ms_to_lr(state,list(frame['mid']),list(frame['side']),frame['prediction'],frame['fs'],frame['fs']*frame['duration'])
                self.assertEqual(list(output),frame['output'])
                self.assertEqual(state.pred_prev_Q13+state.sMid+state.sSide,frame['state'])

    def test_native_stereo_entropy_predictors_mid_only_and_range_state(self):
        self.assertEqual(len(self.cases['predictors']),180)
        fields=('storage','end_offs','end_window','nend_bits','nbits_total','offs','rng','val','ext','rem','error')
        for number,frame in enumerate(self.cases['predictors']):
            with self.subTest(frame=number):
                decoder=OpusRangeDecoder(base64.b64decode(frame['payload'],validate=True))
                result=stereo.decode_pred(decoder)+[stereo.mid_only(decoder)]
                result+=[getattr(decoder,name) for name in fields]+[decoder.tell(),decoder.tell_frac()]
                self.assertEqual(result,frame['output'])


if __name__=='__main__':
    unittest.main()
