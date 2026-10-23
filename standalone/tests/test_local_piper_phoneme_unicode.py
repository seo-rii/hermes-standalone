"""Frozen observations from all original Unicode and permissive UTF-8 inputs."""
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest
import zlib

from standalone import local_piper_phoneme_unicode as u
ROOT=Path(__file__).resolve().parent/'fixtures'


class UnicodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads(zlib.decompress((ROOT/'local_piper_phoneme_unicode.json.zlib').read_bytes()))

    def test_all_original_unicode_primitives(self):
        digest=hashlib.sha256();pack=struct.Struct('<HBQ2I').pack
        for code in range(self.fixture['codepoints']):
            digest.update(pack(u.classes(code),u.category(code),u.properties(code),
                               u.to_lower(code),u.to_upper(code)))
        self.assertEqual(digest.hexdigest(),self.fixture['native_full_semantic_rows_sha256'])
        self.assertEqual(u.SOURCE_WCTYPE_PROFILE,self.fixture['source_wctype_profile'])
        self.assertEqual(self.fixture['unicode_source_version'],'11.0.0')
        self.assertEqual(u.to_lower_for_voice(73,True),0x131)
        self.assertEqual(u.to_lower_for_voice(73,False),105)
        self.assertFalse(u.is_wpunct(0x200d))
        self.assertEqual(self.fixture['consumer_macro_names']['iswalpha'],'ucd_isalpha')
        predicates=(u.is_alpha,u.is_digit,u.is_space,u.isspace2,u.is_walpha,u.is_wdigit,
                    u.is_wspace,u.is_wlower,u.is_wupper,u.is_wpunct,u.is_walnum,u.is_wblank)
        for code in (0,32,65,0x00e9,0x0301,0x0591,0x0903,0x1f600,0x10ffff):
            with self.subTest(code=code):
                self.assertEqual(sum(int(call(code))<<bit for bit,call in enumerate(predicates)),u.classes(code))
        for case in self.fixture['outside']:
            with self.subTest(code=case['code']):
                self.assertEqual([call(case['code']) for call in predicates],case['predicates'])
                self.assertEqual(u.utf8_out(case['code']),base64.b64decode(case['utf8']))
                self.assertEqual(u.to_lower(case['code']),case['code'])
                self.assertEqual(u.category(case['code']),5)

    def test_all_original_utf8_encodings(self):
        digest=hashlib.sha256()
        for code in range(self.fixture['codepoints']):
            encoded=u.utf8_out(code)
            digest.update(bytes((len(encoded),))+encoded)
        self.assertEqual(digest.hexdigest(),self.fixture['utf8_full_rows_sha256'])

    def test_original_permissive_and_reverse_utf8_decoding(self):
        for case in self.fixture['utf8_decodes']:
            with self.subTest(buffer=case['buffer'],cursor=case['cursor'],backwards=case['backwards']):
                self.assertEqual(u.utf8_in(base64.b64decode(case['buffer']),case['cursor'],case['backwards']),
                                 (case['code'],case['size']))


if __name__=='__main__':unittest.main(verbosity=2)
