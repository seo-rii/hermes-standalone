"""Original GetVowelStress and SetWordStress outputs across actual voices."""
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
from standalone.local_piper_phoneme_data import Phone
from standalone.local_piper_phoneme_stress import vowel_stress, set_word_stress

class StressTests(NativeFreeTestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(zlib.decompress((ROOT / 'stress_native.json.zlib').read_bytes()))
        cls.voices = {}
        for voice in cls.fixture['voices']:
            state = {key: value for key, value in voice.items() if key not in ('name', 'dictionary', 'phoneme_table')}
            state['langopts'] = SimpleNamespace(**state['langopts'])
            state['phoneme_tab'] = [Phone(**value) if value is not None else None for value in voice['phoneme_table']]
            cls.voices[voice['name']] = SimpleNamespace(**state)

    def test_original_vowel_stress(self):
        differences = []
        for case in self.fixture['vowels']:
            phones, levels, count, stressed, maximum = vowel_stress(self.voices[case['voice']], base64.b64decode(case['phones']), case['primary'], case['control'])
            actual = {'output': base64.b64encode(phones).decode(), 'levels': levels[:count + 1], 'count': count, 'stressed': stressed, 'maximum': maximum}
            expected = {key: case[key] for key in actual}
            if actual != expected:
                differences.append({'case': case, 'actual': actual})
        self.assertFalse(differences, json.dumps(differences[:2], indent=2))

    def test_original_word_stress(self):
        differences = []
        for case in self.fixture['stress']:
            tr = self.voices[case['voice']]
            tr.prev_last_stress = 5
            flags = None if case['flags'] is None else [case['flags'], 0]
            actual = set_word_stress(tr, base64.b64decode(case['phones']), flags, case['tonic'], case['control'])
            output = {'output': base64.b64encode(actual).decode(), 'prev_last_stress': tr.prev_last_stress}
            expected = {key: case[key] for key in output}
            if output != expected:
                differences.append({'case': case, 'actual': output})
        self.assertFalse(differences, json.dumps(differences[:2], indent=2))
