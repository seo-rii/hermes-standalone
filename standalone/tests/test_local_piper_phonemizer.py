"""Actual Piper bridge IPA clauses and untouched Python wrapper end-to-end."""
import importlib.abc
import json
from pathlib import Path
import sys
import traceback
import unittest
import zlib
HERE = Path(__file__).resolve().parent
ROOT = HERE / 'fixtures'
DATA = HERE.parent / 'assets/piper/espeak-ng-data'

class NoNative(importlib.abc.MetaPathFinder):

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'ctypes', 'subprocess', 'numpy', 'onnxruntime', 'piper', 'torch', 'espeakbridge'}:
            raise AssertionError('Native dependency in actual phonemizer ' + fullname)
from standalone.local_piper_phoneme import EspeakPhonemizer, _merge_known_vowel_clusters

class ObservedPhonemizer(EspeakPhonemizer):

    def _clauses(self, voice, text):
        result = super()._clauses(voice, text)
        self.observed_clauses = [list(item) for item in result]
        return result

class PhonemizerTests(unittest.TestCase):

    def setUp(self):
        guard = NoNative()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    def test_original_native_clauses_and_python_phonemize_wrapper(self):
        fixture = json.loads(zlib.decompress((ROOT / 'local_piper_phonemizer.json.zlib').read_bytes()))
        engine = ObservedPhonemizer(DATA)
        failures = []
        for index, case in enumerate(fixture['cases']):
            clusters = {tuple(item) for item in case['clusters']} if case['clusters'] else None
            try:
                actual = engine.phonemize(case['voice'], case['text'], clusters)
                if actual != case['phonemes']:
                    failures.append({'index': index, 'voice': case['voice'], 'text': case['text'][:200], 'actual': actual, 'expected': case['phonemes']})
                if engine.observed_clauses != case['raw']:
                    failures.append({'index': index, 'voice': case['voice'], 'text': case['text'][:200], 'raw_actual': engine.observed_clauses, 'raw_expected': case['raw']})
            except Exception as error:
                failures.append({'index': index, 'voice': case['voice'], 'text': case['text'][:200], 'error': type(error).__name__ + ': ' + str(error), 'traceback': traceback.format_exc()})
        self.assertFalse(failures, json.dumps(failures[:2], ensure_ascii=False, indent=2))
        for case in fixture['refusals']:
            voice = case['voice'].encode() if case.get('voice_is_bytes') else case['voice']
            text = case['text'].encode() if case.get('text_is_bytes') else case['text']
            with self.subTest(voice=voice, text=str(text)[:30]), self.assertRaises(Exception) as caught:
                engine.phonemize(voice, text)
            self.assertEqual(type(caught.exception).__name__, case['error'])
        print('native original Piper phonemizer exact', len(fixture['cases']), 'cases', flush=True)
