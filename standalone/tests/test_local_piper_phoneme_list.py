"""Original MakePhonemeList observations, with no native dependency or executable."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import unittest
import zlib

from standalone.local_piper_phoneme_data import Phone,PhonemeData,PhonemeSelection
from standalone.local_piper_phoneme_list import FirstItem,make_phoneme_list
from standalone.local_piper_phoneme_vm import Item
from standalone.local_piper_voices import VoiceLoader

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'assets/piper/espeak-ng-data'


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]in {'ctypes','subprocess','numpy','onnxruntime','piper','torch'}:
            raise AssertionError('Phoneme linking attempted native dependency '+fullname)


class PhonemeListTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads(zlib.decompress((ROOT/'fixtures/local_piper_phoneme_list.json.zlib').read_bytes()))
        cls.data=PhonemeData(DATA)
        cls.phone_dicts=[dict(zip(cls.fixture['phone_fields'],row))if row is not None else None for row in cls.fixture['phones']]
        cls.phones=[Phone(**row)if row is not None else None for row in cls.phone_dicts]

    def setUp(self):
        guard=NoNative();sys.meta_path.insert(0,guard);self.addCleanup(sys.meta_path.remove,guard)
        self.loader=VoiceLoader(DATA)

    def _replay(self,case,buffer=None):
        profile=self.loader.select(case['voice']);tr=profile.config
        for key,value in case['options'].items():setattr(tr.langopts,key,deepcopy(value))
        tr.phoneme_tab_ix=case['table'];selection=PhonemeSelection(self.data)
        initial=case['input_selection'];selection.phones[:]=[self.phones[index]for index in initial['phones']]
        selection.current=initial['current'];selection.count=initial['count']
        items=[FirstItem(**row)for row in case['items']]
        output=make_phoneme_list(tr,items,case['post_pause'],case['start_sentence'],selection=selection,phoneme_data=self.data,
                                 replacements=case['replacements'],option_wordgap=case['wordgap'],output_buffer=buffer)
        expected=[{**row,'ph':self.phone_dicts[row['ph']]}for row in case['output']]
        context=(case['voice'],case['text'])
        self.assertEqual([asdict(row)for row in output],expected,context)
        self.assertEqual([asdict(row)for row in items],case['mutated'],context)
        self.assertEqual(selection.current,case['final_selection']['current'],context)
        self.assertEqual(selection.count,case['final_selection']['count'],context)
        self.assertEqual([asdict(row)if row is not None else None for row in selection.phones],
                         [self.phone_dicts[index]for index in case['final_selection']['phones']],context)
        return output

    def test_original_1815_clause_derived_inputs_across_121_voices(self):
        cases=self.fixture['documents']['real']['cases']
        self.assertEqual(len(cases),1815);self.assertEqual(len({case['voice']for case in cases}),121)
        for case in cases:self._replay(case)

    def test_original_2076_voicing_replacement_switch_and_capacity_cases(self):
        cases=self.fixture['documents']['edges']['cases'];self.assertEqual(len(cases),2076)
        for case in cases:self._replay(case)

    def test_original_72_retained_buffer_calls_and_caller_mutations(self):
        groups=self.fixture['documents']['retained']['groups']
        self.assertEqual(len(groups),12);self.assertEqual(sum(len(group['cases'])for group in groups),72)
        for group in groups:
            buffer=[Item(**{**row,'ph':self.phones[row['ph']]})for row in group['initial_buffer']]
            for case in group['cases']:
                output=self._replay(case,buffer)
                self.assertTrue(all(item is buffer[index]for index,item in enumerate(output)))
                raw=json.dumps([asdict(row)for row in buffer],sort_keys=True,separators=(',',':')).encode()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),case['buffer_sha256'],(group['voice'],case['text']))
                for change in case['caller_updates']:
                    for key,value in change['fields'].items():setattr(output[change['index']],key,value)
                raw=json.dumps([asdict(row)for row in buffer],sort_keys=True,separators=(',',':')).encode()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),case['updated_buffer_sha256'])

    def test_pooled_observations_restore_all_original_native_document_bytes(self):
        self.assertEqual(self.fixture['format'],1);self.assertEqual(len(self.phone_dicts),2846)
        for name,pooled in self.fixture['documents'].items():
            document=deepcopy(pooled);cases=list(document.get('cases',[]))
            for group in document.get('groups',[]):
                for row in group['initial_buffer']:row['ph']=self.phone_dicts[row['ph']]
                cases.extend(group['cases'])
            for case in cases:
                for key in ('input_selection','final_selection'):
                    case[key]['phones']=[self.phone_dicts[index]for index in case[key]['phones']]
                for row in case['output']:row['ph']=self.phone_dicts[row['ph']]
            raw=json.dumps(document,sort_keys=True,separators=(',',':')).encode()
            expected=self.fixture['source_documents'][name]
            self.assertEqual(len(raw),expected['canonical_bytes'],name)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),expected['canonical_sha256'],name)

    def test_invalid_counts_and_truncated_output_buffer_are_rejected(self):
        profile=self.loader.select('en-us');tr=profile.config;tr.phoneme_tab_ix=profile.phoneme_tab_ix
        for size in (0,1,2,1001):
            with self.subTest(size=size),self.assertRaises(ValueError):
                make_phoneme_list(tr,[FirstItem()for _ in range(size)],0,False,
                                  selection=PhonemeSelection(self.data),phoneme_data=self.data)
        with self.assertRaises(ValueError):
            make_phoneme_list(tr,[FirstItem()for _ in range(3)],0,False,
                              selection=PhonemeSelection(self.data),phoneme_data=self.data,output_buffer=[Item()]*1000)


if __name__=='__main__':unittest.main(verbosity=2)
