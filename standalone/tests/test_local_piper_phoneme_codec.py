"""Untouched packaged bridge results for mnemonic codecs and table changes."""
import base64
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import zlib
HERE = Path(__file__).resolve().parent
ROOT = HERE / 'fixtures'
DATA = HERE.parent / 'assets/piper/espeak-ng-data'
import importlib.abc

class NoNative(importlib.abc.MetaPathFinder):

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'ctypes', 'subprocess', 'numpy', 'onnxruntime', 'piper', 'torch'}:
            raise AssertionError('eSpeak pronunciation attempted native dependency ' + fullname)

class NativeFreeTestCase(unittest.TestCase):

    def setUp(self):
        guard = NoNative()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)
from standalone.local_piper_phoneme_codec import encode_phonemes, decode_phonemes
from standalone.local_piper_phoneme_data import Phone, PhonemeData, PhonemeSelection
from dataclasses import asdict

class CodecTests(NativeFreeTestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(zlib.decompress((ROOT / 'codec_native.json.zlib').read_bytes()))
        cls.views = {item['table']: SimpleNamespace(phoneme_tab=[Phone(**p) if p is not None else None for p in item['phones']], n_phoneme_tab=item['count']) for item in cls.fixture['tables']}

    def test_original_encode_phonemes(self):
        differences = []
        for case in self.fixture['encodes']:
            result = encode_phonemes(self.views[case['table']], base64.b64decode(case['input']))
            actual = {'phones': base64.b64encode(result.phonemes).decode(), 'bad': result.bad_phoneme, 'cursor': result.cursor}
            if actual != {key: case[key] for key in actual}:
                differences.append({'case': case, 'actual': actual})
        self.assertFalse(differences, json.dumps(differences[:2], indent=2))

    def test_original_decode_phonemes(self):
        for case in self.fixture['decodes']:
            with self.subTest(table=case['table'], input=case['input']):
                self.assertEqual(decode_phonemes(self.views[case['table']], base64.b64decode(case['input'])), base64.b64decode(case['output']))
