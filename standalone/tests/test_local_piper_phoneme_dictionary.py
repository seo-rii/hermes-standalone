"""Actual pinned original compiled dictionary results, with full context flags."""
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
from standalone.local_piper_phoneme_data import Dictionary
from standalone.local_piper_phoneme_dictionary import DictionaryLookup, WordTab, transpose_alphabet

class LookupTests(NativeFreeTestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(zlib.decompress((ROOT / 'dictionary_lookup_native.json.zlib').read_bytes()))
        cls.languages = {item['name']: item['state'] for item in cls.fixture['languages']}
        cls.dictionaries = {name: Dictionary((DATA / (name + '_dict')).read_bytes()) for name in cls.languages}

    def test_original_transpose_alphabet(self):
        for case in self.fixture['transposes']:
            with self.subTest(language=case['language'], word=case['word']):
                tr = SimpleNamespace(**self.languages[case['language']])
                result, size = transpose_alphabet(tr, base64.b64decode(case['word']))
                self.assertEqual(size, case['size'])
                self.assertEqual(result, base64.b64decode(case['buffer']))

    def test_original_lookup_morphology_and_context(self):
        differences = []
        for case in self.fixture['cases']:
            state = dict(self.languages[case['language']])
            state.update(case['overrides'])
            state['langopts'] = SimpleNamespace(**state['langopts'])
            tr = SimpleNamespace(**state)
            primary = tr if state['primary_is_self'] else SimpleNamespace(clause_end=state['clause_end'], clause_terminator=state['clause_terminator'])
            lookup = DictionaryLookup(tr, self.dictionaries[case['language']], primary)
            lookup.dictionary_skipwords = 37
            wtab = None if case['word_flags'] is None else [WordTab(flags=case['word_flags'], length=4) for _ in range(4)]
            result = lookup.lookup_raw(base64.b64decode(case['word']), base64.b64decode(case['text']), flags=case['flags'], end_flags=case['end_flags'], word_table=wtab, phonemes=b'~')
            actual = {'found': result.found, 'phonemes': base64.b64encode(result.phonemes).decode(), 'flags': list(result.flags), 'end': result.end, 'skipwords': lookup.dictionary_skipwords}
            if actual != case['output']:
                differences.append({'case': case, 'actual': actual})
        self.assertFalse(differences, json.dumps(differences[:2], ensure_ascii=False, indent=2))
        print('native raw dictionary cases', len(self.fixture['cases']), flush=True)
