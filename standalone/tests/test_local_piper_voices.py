"""Original packaged eSpeak voice selection and source configuration contracts."""
import base64
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import zlib

from standalone.local_piper_voices import ALPHABETS, VoiceLoader, alphabet_from_char, translator_config


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'numpy', 'onnxruntime', 'piper', 'ctypes', 'subprocess', 'torch'}:
            raise AssertionError('eSpeak configuration attempted native dependency ' + fullname)


def normalize(value):
    if isinstance(value, SimpleNamespace):
        return {key: normalize(item) for key, item in vars(value).items()}
    if isinstance(value, (tuple, list, bytes, bytearray)):
        return [normalize(item) for item in value]
    return value


class LocalPiperVoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wrapper = json.loads((Path(__file__).parent / 'fixtures/local_piper_voices.json').read_bytes())
        raw = zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'], validate=True))
        if hashlib.sha256(raw).hexdigest() != wrapper['data_sha256']:
            raise AssertionError('Native eSpeak configuration fixture digest mismatch')
        cls.oracle = json.loads(raw)

    def setUp(self):
        guard = NoNative()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    def assert_config(self, actual, expected, *, dictionary_loaded=False):
        actual = normalize(actual)
        for name, value in expected.items():
            if name == 'langopts' and dictionary_loaded:
                # Dictionary parsing owns the pointer-to-resource-offset field.
                # The voice parser supplies the selected dictionary and options.
                value = {k: v for k, v in value.items() if k != 'replace_chars'}
                for field, item in value.items():
                    self.assertEqual(actual[name][field], item, (name, field))
            else:
                self.assertEqual(actual[name], value, name)

    def test_original_defaults_all_languages_and_dynamic_names(self):
        cases = self.oracle['defaults']['language_defaults']
        self.assertEqual(len(cases), 214)
        for case in cases:
            with self.subTest(language=case['language']):
                actual = translator_config(case['language'])
                self.assertEqual(len(actual.langopts.param), 18)
                self.assert_config(actual, case['config'])
        before = translator_config('en')
        changed = translator_config('en')
        changed.langopts.param[0] = 999
        changed.letter_bits[97] = 0
        changed.stress_lengths[0] = -123
        self.assertEqual(normalize(before.langopts), normalize(translator_config('en').langopts))
        self.assertEqual(before.letter_bits, translator_config('en').letter_bits)
        self.assertEqual(before.stress_lengths, translator_config('en').stress_lengths)

    def test_all_unicode_alphabet_mappings_match_original_native_digest(self):
        oracle = self.oracle['defaults']
        self.assertEqual([vars(row) for row in ALPHABETS], oracle['alphabets'])
        indexes = {id(row): i + 1 for i, row in enumerate(ALPHABETS)}
        actual = bytearray()
        for code in range(0x110000):
            row = alphabet_from_char(code)
            actual.append(indexes[id(row)] if row is not None else 0)
        self.assertEqual(hashlib.sha256(actual).hexdigest(), oracle['native_alphabet_index_sha256'])

    def test_original_1061_voice_alias_variant_and_replacement_cases(self):
        loader = VoiceLoader()
        loader.select('en-us')  # The independently captured native fixture's initial voice.
        cases = self.oracle['voices']['cases']
        self.assertEqual(len(cases), 1061)
        for case in cases:
            with self.subTest(voice=case['name']):
                if not case['accepted']:
                    with self.assertRaises(ValueError):
                        loader.select(case['name'])
                    continue
                actual = loader.select(case['name'])
                self.assert_config(actual.config, case['config'], dictionary_loaded=True)
                self.assertEqual(actual.voice_id, case['voice_id'])
                self.assertEqual(actual.name, case['voice_name'])
                self.assertEqual(normalize(actual.languages), case['languages'])
                self.assertEqual(normalize(actual.replace_phonemes), case['replace_phonemes'])
                self.assertEqual(actual.option_tone_flags, case['option_tone_flags'])
                self.assertEqual(actual.phoneme_tab_ix, case['config']['phoneme_tab_ix'])
                self.assertEqual(actual.dictionary_name, case['config']['dictionary_name'])

    def test_original_438_language_option_overrides(self):
        loader = VoiceLoader()
        cases = self.oracle['options']['cases']
        self.assertEqual(len(cases), 438)
        for case in cases:
            with self.subTest(language=case['language'], key=case['key'], value=case['value']):
                config = loader.translator_config(case['language'])
                loader.option_tone_flags = 0
                loader._options(config, case['key'], case['value'].encode())
                self.assert_config(config, case['config'])
                self.assertEqual(loader.option_tone_flags, case['option_tone_flags'])

    def test_original_420_retained_phoneme_table_transitions(self):
        oracle = self.oracle['native_selections']
        loader = VoiceLoader()
        loader.select('en-us')
        self.assertEqual(loader._active_table, oracle['initial_table'])
        self.assertEqual(len(oracle['selections']), 420)
        for case in oracle['selections']:
            with self.subTest(table=case['table']):
                loader._select_table(case['table'])
                self.assertEqual(loader._active_count, case['count'])
                self.assertEqual([loader._active_phones.get(code) for code in range(256)], case['mnemonics'])

    def test_original_piper_disabled_klatt_build_and_80_named_variants(self):
        profile = self.oracle['native_build_profile']
        self.assertFalse(profile['klatt_build_enabled'])
        self.assertTrue(profile['packaged_binary_condition_observed'])
        self.assertEqual(len(profile['variants']), 19)
        self.assertEqual(len(profile['cases']), 80)
        loader = VoiceLoader()
        for case in profile['cases']:
            with self.subTest(voice=case['voice']):
                actual = loader.select(case['voice'])
                self.assertEqual(case['klattv'], [0] * 8)
                self.assertEqual(actual.klatt, case['klatt_condition'])
                self.assertFalse(case['condition_with_forced_voice_flag'])

    def test_invalid_names_and_malformed_option_boundaries(self):
        loader = VoiceLoader()
        for value in (None, 17, b'en'):
            with self.subTest(value=value), self.assertRaises(TypeError):
                loader.select(value)
        for value in ('absent-hermes-voice', '\0', '../en', '/en'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                loader.select(value)
        with self.assertRaises(ValueError):
            loader.translator_config('a' * 40)
        for key in ('numbers', 'dictrules', 'stressOpt'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                loader._options(loader.translator_config('en'), key, b'-1 /')
        with self.assertRaises(ValueError):
            VoiceLoader(select_table=lambda index: None)


if __name__ == '__main__':
    unittest.main()
