"""Untouched eSpeak compiled-data and bytecode observations, with no native runtime."""
from dataclasses import asdict
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import zlib

from standalone.local_piper_phoneme_data import Dictionary,Phone,PhonemeData,PhonemeSelection,hash_dictionary
from standalone.local_piper_phoneme_vm import Interpreter,Item

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'assets/piper/espeak-ng-data'


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0] in {'ctypes','subprocess','numpy','onnxruntime','piper','torch'}:
            raise AssertionError('eSpeak primitive attempted native dependency '+fullname)


class PhonemeDataVMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle=json.loads(zlib.decompress((ROOT/'fixtures/local_piper_phoneme_data_vm.json.zlib').read_bytes()))
        cls.data=PhonemeData(DATA)

    def setUp(self):
        guard=NoNative();sys.meta_path.insert(0,guard);self.addCleanup(sys.meta_path.remove,guard)

    def test_original_140_tables_and_18595_isolated_programs(self):
        fixture=self.oracle['vm'];count=0
        for name,digest in fixture['asset_hashes'].items():
            self.assertEqual(hashlib.sha256((DATA/name).read_bytes()).hexdigest(),digest)
        self.assertEqual(len(self.data.tables),140)
        for table in fixture['tables']:
            actual=self.data.tables[table['number']]
            self.assertEqual(actual.name,table['name']);self.assertEqual(actual.includes,table['includes'])
            vm=Interpreter(self.data,table['number']);expected={entry['code']:entry for entry in table['entries']}
            self.assertEqual([index for index,phone in enumerate(vm.phones) if phone is not None],list(expected))
            for code,entry in expected.items():
                self.assertEqual(asdict(vm.phones[code]),entry['phone'])
                self.assertEqual(asdict(vm.isolated(code)),entry['output'],(table['name'],code));count+=1
        self.assertEqual(count,18595)

    def test_original_3030_context_and_reduction_programs(self):
        contexts=self.oracle['contexts'];count=0
        for case in contexts['cases']:
            vm=Interpreter(self.data,case['table'],phones=[Phone(**p) if p is not None else None for p in case['global_phones']])
            sequence=[Item(**{**item,'ph':Phone(**item['ph'])}) for item in case['sequence']]
            for index,expected in enumerate(case['outputs'],1):
                self.assertEqual(asdict(vm.run(sequence,index,start=index)),expected,(case['voice'],case['text'],index));count+=1
        for change in contexts['changes']:
            case=contexts['cases'][change['case']]
            vm=Interpreter(self.data,case['table'],phones=[Phone(**p) if p is not None else None for p in case['global_phones']])
            sequence=[Item(**{**item,'ph':Phone(**item['ph'])}) for item in case['sequence']];worddata={}
            for index,expected in enumerate(change['outputs'],1):
                self.assertEqual(asdict(vm.run(sequence,index,control=0x100,reduce=change['reduce'],worddata=worddata)),expected,
                                 (case['voice'],change['reduce'],index));count+=1
        self.assertEqual(count,3030)

    def test_original_2061_dictionary_hashes(self):
        self.assertEqual(len(self.oracle['vm']['hashes']),2061)
        for case in self.oracle['vm']['hashes']:
            self.assertEqual(hash_dictionary(case['word']),case['value'],case['word'])

    def test_all_118_compiled_dictionaries_and_replacement_pairs(self):
        expected={item['name']:item for item in self.oracle['indexes']['dictionaries']};total=0
        paths=sorted(DATA.glob('*_dict'));self.assertEqual(len(paths),118)
        for path in paths:
            content=path.read_bytes();dictionary=Dictionary(content);name=path.name[:-5]
            self.assertEqual(hashlib.sha256(content).hexdigest(),expected[name]['sha256'],name)
            self.assertEqual(dictionary.index_snapshot(),expected[name]['index'],name)
            if name in self.oracle['replacements']:
                actual=[[first.hex(),second.hex()] for first,second in dictionary.replacement_pairs] if dictionary.replacement_pairs else None
                self.assertEqual(actual,self.oracle['replacements'][name],name)
            total+=sum(map(len,dictionary.buckets))
        self.assertEqual(total,1269305)

    def test_original_420_retained_table_transitions_and_fresh_sentinel(self):
        selection=PhonemeSelection(self.data);self.assertEqual(selection.count,0)
        selection.select(0);self.assertEqual(selection.phones,[None]*256)
        oracle=self.oracle['selection'];selection.select(oracle['initial_table'])
        self.assertEqual(len(oracle['selections']),420)
        for case in oracle['selections']:
            selection.select(case['table']);self.assertEqual(selection.count,case['count'])
            self.assertEqual([asdict(p) if p is not None else None for p in selection.phones],case['phones'],case['table'])

    def test_malformed_binary_resources_and_table_boundaries(self):
        for content in (b'',b'\0'*1032,b'\xff'*1033):
            with self.subTest(length=len(content)),self.assertRaises(ValueError):Dictionary(content)
        for value in (-1,140,True,1.5,'absent-hermes-table'):
            with self.subTest(value=value),self.assertRaises(ValueError):self.data.number(value)
        with TemporaryDirectory(prefix='hermes-espeak-malformed-') as directory:
            root=Path(directory)
            for name in ('phontab','phonindex','phondata'):(root/name).write_bytes((DATA/name).read_bytes())
            for name,content in (('phondata',b'\0'*8),('phonindex',b'\0'),('phontab',b'')):
                original=(root/name).read_bytes();(root/name).write_bytes(content)
                with self.subTest(name=name),self.assertRaises(ValueError):PhonemeData(root)
                (root/name).write_bytes(original)


if __name__=='__main__':unittest.main(verbosity=2)
