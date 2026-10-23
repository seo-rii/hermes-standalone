"""Original GetTranslatedPhonemeString ASCII and IPA clause rendering."""
from dataclasses import asdict
import importlib.abc
import json
from pathlib import Path
import sys
import unittest
import zlib
from standalone.local_piper_phoneme_data import Phone,PhonemeData
from standalone.local_piper_phoneme_vm import Interpreter,Item
from standalone.local_piper_phoneme_render import render

class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0] in {'ctypes','subprocess','numpy','onnxruntime','piper','torch'}:
            raise AssertionError('eSpeak pronunciation attempted native dependency '+fullname)

class NativeFreeTestCase(unittest.TestCase):
    def setUp(self):
        guard=NoNative();sys.meta_path.insert(0,guard);self.addCleanup(sys.meta_path.remove,guard)

class RenderTests(NativeFreeTestCase):
    def test_original_ascii_and_ipa_rendered_clauses(self):
        here=Path(__file__).parent
        fixture=json.loads(zlib.decompress((here/'fixtures/phoneme_context_native.json.zlib').read_bytes()))
        data=PhonemeData(here.parent/'assets/piper/espeak-ng-data');count=0
        for case in fixture['cases']:
            vm=Interpreter(data,case['table'],phones=[Phone(**phone) if phone is not None else None for phone in case['global_phones']])
            sequence=[Item(**{**item,'ph':Phone(**item['ph'])}) for item in case['sequence']]
            for mode,expected in case['rendered'].items():
                self.assertEqual(render(vm,sequence,int(mode)),expected,(case['voice'],case['text'],mode));count+=1
        self.assertEqual(count,80)
