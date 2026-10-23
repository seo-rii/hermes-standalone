"""Independent native PyAV/FFmpeg FLAC PCM fixtures and RFC integrity checks."""
import base64
import hashlib
import json
from pathlib import Path
import unittest
import zlib

from standalone.local_whisper_audio_flac import FlacError,decode_flac


class FlacDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wrapper=json.loads((Path(__file__).parent/'fixtures/local_whisper_audio_flac.json').read_text())
        raw=zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'],validate=True))
        if hashlib.sha256(raw).hexdigest()!=wrapper['data_sha256']:
            raise AssertionError('Native FLAC fixture digest mismatch')
        cls.oracle=json.loads(raw)

    def _compare(self,case,**kwargs):
        data=base64.b64decode(case['data'],validate=True)
        actual=decode_flac(data,**kwargs)
        self.assertEqual(actual,(case['sample_rate'],case['channels'],case['bits'],case['mask']))

    def test_actual_native_encoder_all_channels_and_original_integer_depths(self):
        count=0
        for case in self.oracle['cases']:
            if case['name'].startswith('native-'):
                with self.subTest(case=case['name']):
                    self._compare(case)
                    count+=1
        self.assertEqual(count,80)

    def test_native_decoder_predictors_rice_headers_masks_and_33bit_side(self):
        count=0
        for case in self.oracle['cases']:
            if not case['name'].startswith('native-'):
                with self.subTest(case=case['name']):
                    self._compare(case)
                    count+=1
        self.assertEqual(count,177)

    def test_rfc_36bit_sample_numbers_exceed_pinned_native_header_support(self):
        # RFC9639 extends the coded number to seven bytes. Pinned avcodec
        # rejects that prefix, so these are separate standard-derived cases.
        for case in self.oracle['rfc_36bit_numbers']:
            with self.subTest(case=case['name']):
                self._compare(case)

    def test_crc_metadata_truncation_sample_total_and_pcm_md5_fail_explicitly(self):
        for case in self.oracle['invalid']:
            with self.subTest(case=case['name']),self.assertRaises(FlacError):
                decode_flac(base64.b64decode(case['data'],validate=True))

    def test_decoded_frame_budget_checks_known_and_unknown_totals(self):
        for name in ('short-last-block','unknown-multiple-frames','wide-33-10-63'):
            case=next(case for case in self.oracle['cases'] if case['name']==name)
            count=len(case['channels'][0])
            with self.subTest(case=name):
                self._compare(case,max_frames=count)
                with self.assertRaisesRegex(FlacError,'budget'):
                    decode_flac(base64.b64decode(case['data']),max_frames=count-1)
        case=self.oracle['cases'][0]
        for budget in (-1,True,2.5):
            with self.subTest(budget=budget),self.assertRaises(ValueError):
                decode_flac(base64.b64decode(case['data']),max_frames=budget)


if __name__=='__main__':
    unittest.main()
