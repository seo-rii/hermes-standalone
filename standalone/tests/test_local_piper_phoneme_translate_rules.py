"""Complete original rule translation and its mutation/counter contracts."""
import base64
import json
from pathlib import Path
import sys
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
from standalone.local_piper_phoneme_session import Session

class TranslateRulesTests(NativeFreeTestCase):

    def test_original_rule_translation_and_reentry(self):
        fixture = json.loads(zlib.decompress((ROOT / 'rules_translate_native.json.zlib').read_bytes()))
        session = Session(DATA)
        previous = None
        differences = []
        errors = []
        for case in fixture['cases']:
            if previous != case['voice']:
                tr = session.select_voice(case['voice'])
                previous = case['voice']
            flags = case['dict_flags'][:] if case['dict_flags'] is not None else None
            text = bytearray(base64.b64decode(case['text']))
            tr.option_sayas = case['option_sayas']
            tr.dict_condition = case['dict_condition']
            tr.expect_verb = case['expect_verb']
            tr.expect_past = tr.expect_noun = tr.expect_verb_s = 0
            tr.prev_dict_flags[:] = [0, 0]
            tr.phonemes_repeat = b''
            tr.phonemes_repeat_count = 0
            tr.clause_end = len(text) - 1
            tr.clause_terminator = 524328
            session.state.pre_pause = 7
            session.state.dictionary_skipwords = 37
            try:
                result = tr.rules.translate(text, 2, max_phonemes=case['max_phonemes'], end_phonemes=case['end_phonemes'], word_flags=case['word_flags'], dict_flags=flags, initial=base64.b64decode(case['initial']))
                actual = {'phonemes': base64.b64encode(result.phonemes).decode(), 'end_type': result.end_type, 'end_phonemes': base64.b64encode(result.end_phonemes).decode(), 'dict_flags': flags, 'text': base64.b64encode(text).decode(), 'word_vowel_count': tr.word_vowel_count, 'word_stressed_count': tr.word_stressed_count, 'pre_pause': session.state.pre_pause}
                if actual != case['output']:
                    differences.append({'case': case, 'actual': actual})
            except Exception as error:
                errors.append({'case': case, 'exception': type(error).__name__ + ': ' + str(error)})
        self.assertFalse(errors, json.dumps(errors[:2], ensure_ascii=False, indent=2))
        self.assertFalse(differences, json.dumps(differences[:2], ensure_ascii=False, indent=2))
        print('native full TranslateRules contexts', len(fixture['cases']), flush=True)
