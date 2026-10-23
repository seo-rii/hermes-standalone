"""Compiled rule-chain matching against independent pinned native observations."""
import base64
import hashlib
import json
from pathlib import Path
import struct
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
from standalone.local_piper_phoneme_data import Dictionary, Phone
from standalone.local_piper_phoneme_dictionary import DictionaryLookup
from standalone.local_piper_phoneme_letters import is_letter
from standalone.local_piper_phoneme_rules import Rules
from standalone.local_piper_phoneme_stress import set_word_stress
from standalone.local_piper_numbers import NumberTranslator

class Context(SimpleNamespace):

    def is_letter(self, code, group):
        return is_letter(self, code, group)

    def lookup_dict_list(self, *args, **kwargs):
        return self.lookup_engine.lookup_dict_list(*args, **kwargs)

    def lookup(self, *args, **kwargs):
        return self.lookup_engine.lookup(*args, **kwargs)

    def lookup_english(self, word):
        return self.english.lookup(word)

    def lookup_accented_letter(self, letter):
        return self.numbers.lookup_accented_letter(self, letter)

    def set_word_stress(self, *args, **kwargs):
        return set_word_stress(self, *args, **kwargs)

class RuleTests(NativeFreeTestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(zlib.decompress((ROOT / 'rules_native.json.zlib').read_bytes()))
        cls.contexts = {}
        cls.engines = {}
        for voice in cls.fixture['voices']:
            state = dict(voice['state'])
            state['langopts'] = SimpleNamespace(**state['langopts'])
            state['phoneme_tab'] = [Phone(**p) if p is not None else None for p in state.pop('phoneme_table')]
            state.update(phonemes_repeat=b'', phonemes_repeat_count=0, option_sayas=0, clause_end=0, clause_terminator=524328)
            context = Context(**state)
            dictionary = Dictionary((DATA / (state['dictionary_name'] + '_dict')).read_bytes())
            context.lookup_engine = DictionaryLookup(context, dictionary)
            context.numbers = NumberTranslator(translator=context)
            cls.contexts[voice['name']] = context
            cls.engines[voice['name']] = Rules(context, dictionary)
        for context in cls.contexts.values():
            context.english = cls.contexts['en']

    def test_original_letter_classes(self):
        for item in self.fixture['letter_classes']:
            tr = self.contexts[item['voice']]
            digest = hashlib.sha256()
            for code in list(range(768)) + [tr.letter_bits_offset + i for i in range(256)] + [12288, 1114111, -1]:
                for group in range(8):
                    digest.update(struct.pack('<i', tr.is_letter(code, group)))
            self.assertEqual(digest.hexdigest(), item['sha256'], item['voice'])

    def test_original_rule_chains(self):
        differences = []
        for case in self.fixture['cases']:
            context = self.contexts[case['voice']]
            vars(context).update(case['overrides'])
            context.phonemes_repeat = b''
            context.phonemes_repeat_count = 0
            context.lookup_engine.dictionary_skipwords = 37
            text = base64.b64decode(case['text'])
            context.clause_end = len(text) - 1
            cursor, match = self.engines[case['voice']].match_rule(text, case['cursor'], case['start'], case['group_length'], case['rule'], case['word_flags'], case['dict_flags'])
            actual = {'cursor': cursor, 'points': match.points, 'phonemes': base64.b64encode(match.phonemes).decode(), 'end_type': match.end_type, 'del_fwd': match.del_fwd}
            if actual != case['output']:
                differences.append({'case': case, 'actual': actual})
        self.assertFalse(differences, json.dumps(differences[:2], ensure_ascii=False, indent=2))
        print('native rule-chain contexts', len(self.fixture['cases']), flush=True)
