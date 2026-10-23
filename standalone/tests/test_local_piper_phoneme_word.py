"""Whole original word output, mutation and state comparisons."""
import base64
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import traceback
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
from standalone.local_piper_phoneme_data import Phone
from standalone.local_piper_phoneme_dictionary import WordTab
from standalone.local_piper_phoneme_session import Session
from standalone.local_piper_phoneme_word import translate_word3, translate_word
b64 = lambda value: base64.b64encode(value).decode()
integers = ('expect_verb', 'expect_past', 'expect_verb_s', 'expect_noun', 'prev_last_stress', 'word_vowel_count', 'word_stressed_count', 'prepause_timeout', 'end_stressed_vowel', 'phonemes_repeat_count')

class WordTests(NativeFreeTestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(zlib.decompress((ROOT / 'word_native.json.zlib').read_bytes()))

    def test_original_dictionary_replacement_strings(self):
        session = Session(DATA)
        for voice, expected in self.fixture['replacements'].items():
            tr = session.select_voice(voice)
            actual = [[b64(first), b64(second)] for first, second in tr.langopts.replace_chars] if tr.langopts.replace_chars else None
            self.assertEqual(actual, expected, voice)

    def test_persistent_translator_slots_and_missing_alias_resources(self):
        session = Session(DATA)
        session.select_voice('en-us')
        for slot in (2, 3):
            for language in ('en', 'fr', 'en'):
                tr = session.translator(language, slot)
                tr.phonemes_repeat = b'repeated'
                tr.phonemes_repeat_count = 7
                same = session.translator(language, slot)
                self.assertIs(same, tr)
                self.assertEqual(tr.phonemes_repeat, b'')
                self.assertEqual(tr.phonemes_repeat_count, 7)
            for language in ('bs', 'hr', 'sr'):
                with self.assertRaises(FileNotFoundError):
                    session.translator(language, slot)

    def test_original_whole_word_translations(self):
        session = Session(DATA)
        session.select_voice(self.fixture['initial_voice'])
        previous = None
        differences = []
        errors = []
        difference_count = error_count = 0
        fields = Counter()
        exceptions = Counter()
        groups = Counter()
        views = {}

        def active_view():
            selection = session.selection
            key = (selection.current, selection.count, *map(id, selection.phones))
            if key not in views:
                value = {'current': selection.current, 'count': selection.count, 'phones': [asdict(p) if p is not None else None for p in selection.phones]}
                views[key] = hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()
            return views[key]

        def state():
            value = {key: getattr(tr, key) for key in integers}
            value.update(prev_dict_flags=list(tr.prev_dict_flags), phonemes_repeat=b64(tr.phonemes_repeat), pre_pause=session.state.pre_pause, dictionary_skipwords=session.state.dictionary_skipwords, any_stressed_words=session.state.any_stressed_words, active_view=active_view())
            return value
        for index, case in enumerate(self.fixture['cases']):
            if previous != case['voice']:
                tr = session.select_voice(case['voice'])
                previous = case['voice']
            options = case['options']
            text = bytearray(base64.b64decode(case['text']))
            original_wtab = self.fixture['word_tables'].get(str(case['variant'])) if options['wtab'] else None
            wtab = [WordTab(**item) for item in original_wtab] if original_wtab is not None else None
            for key in integers:
                setattr(tr, key, 0)
            tr.word_vowel_count = 3
            tr.word_stressed_count = 2
            tr.prev_last_stress = 4
            tr.clause_lower_count = 12
            tr.clause_upper_count = 2
            tr.dict_condition = case['dict_condition']
            tr.prev_dict_flags[:] = [0, 0]
            tr.phonemes_repeat = b''
            tr.clause_end = len(text) - 1
            tr.clause_terminator = 524328
            tr.option_sayas = options['sayas']
            session.state.pre_pause = 7
            session.state.dictionary_skipwords = 37
            session.state.option_tone_flags = case['tone_flags']
            session.state.option_capitals = 0
            session.state.any_stressed_words = False
            session.state.current_alphabet = None
            session.state.word_phonemes = b'STALE'
            session.selection.select(tr.phoneme_tab_ix)
            try:
                self.assertEqual(active_view(), case['active_view'], 'before active-map case ' + str(index))
                function = translate_word if options['wrapper'] else translate_word3
                result = function(tr, text, 3, wtab, word_out=options['word_out'])
                actual = {'flags': result.flags, 'phonemes': b64(result.phonemes), 'replacement': b64(result.replacement or b'') if options['word_out'] and result.flags & 536870912 else None, 'text': b64(text), 'word_table_changes': {str(i): asdict(item) for i, item in enumerate(wtab) if asdict(item) != original_wtab[i]} if wtab is not None else None, 'state': state()}
                expected = case['output']
                if actual != expected:
                    difference_count += 1
                    keys = [key for key in actual if actual[key] != expected[key]]
                    fields.update(keys)
                    groups.update(((case['voice'], base64.b64decode(case['text'])[3:].split(b' ', 1)[0].decode(), tuple(keys)),))
                    if len(differences) < 200:
                        differences.append({'index': index, 'case': case, 'actual': actual, 'fields': keys})
            except Exception as error:
                error_count += 1
                exceptions.update((type(error).__name__,))
                if len(errors) < 100:
                    errors.append({'index': index, 'case': case, 'exception': type(error).__name__ + ': ' + str(error), 'traceback': traceback.format_exc()})
        report = {'count': len(self.fixture['cases']), 'difference_count': difference_count, 'error_count': error_count, 'fields': dict(fields), 'exceptions': dict(exceptions), 'groups': [(list(key), count) for key, count in groups.items()], 'differences': differences, 'errors': errors}
        self.assertEqual(error_count, 0, json.dumps(errors[:1], ensure_ascii=False, indent=2))
        self.assertEqual(difference_count, 0, json.dumps(differences[:1], ensure_ascii=False, indent=2))
        print('native whole-word contexts', len(self.fixture['cases']), flush=True)
