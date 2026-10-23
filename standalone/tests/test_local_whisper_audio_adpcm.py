"""Actual pinned FFmpeg ADPCM decoder and Faster-Whisper/SWR fixtures."""
import base64
import hashlib
import io
import json
from pathlib import Path
import unittest
import zlib

from standalone import local_whisper_audio as audio
from standalone.local_whisper_audio_adpcm import decode_adpcm


class WaveAdpcmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wrapper=json.loads((Path(__file__).parent/'fixtures/local_whisper_audio_adpcm.json').read_text())
        raw=zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'],validate=True))
        if hashlib.sha256(raw).hexdigest()!=wrapper['data_sha256']:
            raise AssertionError('Native WAVE ADPCM fixture digest mismatch')
        cls.oracle=json.loads(raw)

    def test_434_native_files_all_ima_widths_ms_predictors_channels_and_swr(self):
        self.assertEqual(len(self.oracle['cases']),434)
        self.assertEqual(self.oracle['native_encoded'],24)
        for case in self.oracle['cases']:
            with self.subTest(case=case['name']):
                output=decode_adpcm(base64.b64decode(case['payload']),case['kind'],case['channels'],case['bits'],case['block_align'])
                self.assertEqual(output,case['pcm'])
                actual=audio.decode_audio(io.BytesIO(base64.b64decode(case['input'])))
                self.assertEqual([round(v*32768)for v in actual],case['resampled'])

    def test_exact_decoded_budget_and_explicit_damaged_block_errors(self):
        for case in self.oracle['cases'][::31]:
            with self.subTest(case=case['name']):
                args=(base64.b64decode(case['payload']),case['kind'],case['channels'],case['bits'],case['block_align'])
                size=len(case['pcm'][0])
                self.assertEqual(decode_adpcm(*args,max_frames=size),case['pcm'])
                with self.assertRaises(ValueError):decode_adpcm(*args,max_frames=size-1)
        for budget in (-1,True,2.5):
            with self.subTest(budget=budget),self.assertRaises(ValueError):
                decode_adpcm(b'\0\0\0\0',17,1,4,4,max_frames=budget)
        for args in ((b'\0\0\x59\0',17,1,4,4),(b'\x07'+b'\0'*6,2,1,4,7),
                     (b'\0',17,1,4,4),(b'\0',2,1,4,7),
                     (b'',17,3,4,4),(b'',2,7,4,4),(b'',17,1,1,4),(b'',17,1,6,4),
                     (b'',2,1,3,4),(b'',17,1,4,0)):
            with self.subTest(args=args),self.assertRaises(ValueError):decode_adpcm(*args)


if __name__=='__main__':unittest.main()
