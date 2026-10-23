"""Independent shipped eSpeak SDK vectors; callbacks are test dependencies."""
import base64
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import zlib

from standalone.local_piper_numbers import NumberTranslator
from standalone import local_piper_phoneme_unicode as unicode


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]in {'numpy','piper','ctypes','subprocess','espeakbridge'}:
            raise AssertionError('eSpeak number port attempted native/process import '+fullname)


class OriginalDictionaryEvents:
    """Finite explicit native dictionary/stress dependencies, only for tests."""
    def __init__(self,test,context,events):self.test=test;self.context=context;self.events=list(events)

    def event(self,kind,word):
        self.test.assertTrue(self.events,self.context);event=self.events.pop(0)
        self.test.assertEqual(event['kind'],kind,self.context);self.test.assertEqual(bytes.fromhex(event['word']),word,self.context)
        return event

    def lookup(self,word):
        event=self.event(0,word);return event['result'],bytes.fromhex(event['phones'])

    def lookup_english(self,word):
        event=self.event(5,word);return event['result'],bytes.fromhex(event['phones'])

    def lookup_dict_list(self,buffer,cursor=0,*,flags=(0,0),end_flags=0,word_table=None):
        event=self.event(1,bytes(buffer[cursor:]).split(b'\0',1)[0]);self.test.assertEqual(list(flags),event['before_flags'],self.context)
        return SimpleNamespace(found=event['result'],phonemes=bytes.fromhex(event['phones']),flags=event['flags'],end=cursor,replacement=None)

    def translate_word(self,word):
        event=self.event(2,word);self.prev_dict_flags[:]=event['flags'];return event['result'],bytes.fromhex(event['phones'])

    def translate_rules(self,word,*,max_phonemes=200,word_flags=0,dict_flags=None,initial=b''):
        before=self.event(7,word);self.test.assertEqual(before['phones'],initial.hex(),self.context);self.test.assertEqual(before['flags'],[max_phonemes,word_flags],self.context)
        event=self.event(3,word);self.test.assertEqual(event['flags'],[max_phonemes,word_flags],self.context)
        return SimpleNamespace(phonemes=bytes.fromhex(event['phones']))

    def set_word_stress(self,phones,flags,tonic=-1,control=0):
        event=self.event(4,phones);self.test.assertEqual(event['before_flags'],[tonic&0xffffffff,0],self.context);self.test.assertEqual(event['result'],control,self.context)
        return bytes.fromhex(event['phones'])


class LocalPiperNumberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wrapper=json.loads((Path(__file__).parent/'fixtures/local_piper_numbers.json').read_text())
        raw=zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'],validate=True))
        if hashlib.sha256(raw).hexdigest()!=wrapper['data_sha256']:raise AssertionError('Piper numeric fixture digest mismatch')
        cls.oracle=json.loads(raw)

    def setUp(self):
        guard=NoNative();sys.meta_path.insert(0,guard);self.addCleanup(sys.meta_path.remove,guard)

    def test_original_31_languages_11656_numeric_and_1206_plural_cases(self):
        count=0;fixture=self.oracle['helpers']
        for voice in fixture['voices']:
            tr=SimpleNamespace(langopts=SimpleNamespace(**voice['options']),translator_name=voice['translator_name'],phoneme_tab=[SimpleNamespace(type=t)for t in voice['table']])
            def lookup(key):
                flags,phones=voice['lookup'].get(key.decode(),[0,'']);return flags,bytes.fromhex(phones)
            tr.lookup=lookup
            for case in voice['cases']:
                with self.subTest(voice=voice['voice'],kind=case['kind'],args=case['args']):
                    state=NumberTranslator();state.number_control=case['control'];args=case['args']
                    if case['kind']=='two':result,phones=state.lookup_num2(tr,*args)
                    elif case['kind']=='three':result,phones=0,state.lookup_num3(tr,*args)
                    else:result,phones=state.lookup_thousands(tr,*args)
                    self.assertEqual(result,case['result']);self.assertEqual(phones.hex(),case['phones']);self.assertEqual(state.speak_missing_thousands,case['speak_missing_thousands']);count+=1
        self.assertEqual(count,11656)
        for mode,value,expected in fixture['variants']:
            tr.langopts.numbers2=mode;self.assertEqual(NumberTranslator().m_variant(tr,value),expected)
        self.assertEqual(len(fixture['variants']),1206)

    def test_original_24_languages_3624_full_number_ordinal_decimal_roman_states(self):
        fixture=self.oracle['full'];count=0
        for key,callback in [('IsAlpha',unicode.is_alpha),('iswalpha',unicode.is_walpha),('iswdigit',unicode.is_wdigit)]:
            for code,expected in fixture['unicode'][key].items():self.assertEqual(callback(int(code)),expected,(key,code))
        for voice in fixture['voices']:
            for case in voice['cases']:
                context=(voice['voice'],case['kind'],case['text'],case['word_flags'][:2])
                with self.subTest(context=context):
                    tr=OriginalDictionaryEvents(self,context,case['events']);tr.langopts=SimpleNamespace(**voice['options']);tr.translator_name=voice['translator_name'];tr.phoneme_tab=[SimpleNamespace(type=t)for t in voice['table']];tr.prev_dict_flags=[0,0]
                    tr.is_alpha=unicode.is_alpha;tr.is_walpha=unicode.is_walpha;tr.is_wdigit=unicode.is_wdigit
                    state=NumberTranslator(option_sayas=case['option_sayas']);buffer=bytearray.fromhex(case['buffer']);flags=[0,0];words=[SimpleNamespace(flags=f)for f in case['word_flags']]
                    if case['kind']=='number':result,phones=state.translate_number(tr,buffer,case['cursor'],flags,words,case['control'],max_output=case['max_output'])
                    else:result,phones=state.translate_roman(tr,buffer,case['cursor'],words,max_output=case['max_output'])
                    actual=dict(result=result,phones=phones.hex(),buffer=buffer.hex(),flags=flags,word_flags=[w.flags for w in words],prev_dict_flags=tr.prev_dict_flags,dictionary_skipwords=state.dictionary_skipwords)
                    self.assertEqual(actual,case['expected']);self.assertFalse(tr.events);count+=1
        self.assertEqual(count,3624)

    def test_original_20_languages_11180_letter_and_accent_states(self):
        fixture=self.oracle['letters'];count=0
        for code,expected in fixture['iswspace'].items():self.assertEqual(unicode.is_wspace(int(code)),expected,code)
        for voice in fixture['voices']:
            for case in voice['cases']:
                context=(voice['voice'],case['kind'],hex(case['letter']),case['next_byte'],case['control'])
                with self.subTest(context=context):
                    tr=OriginalDictionaryEvents(self,context,case['events']);tr.langopts=SimpleNamespace(accents=voice['accents']);tr.translator_name=voice['translator_name'];tr.is_wspace=unicode.is_wspace
                    state=NumberTranslator()
                    if case['kind']=='accent':phones=state.lookup_accented_letter(tr,case['letter'])
                    elif case['kind']=='letter2':phones=state.lookup_letter2(tr,case['letter'])
                    else:phones=state.lookup_letter(tr,case['letter'],case['next_byte'],case['control'])
                    self.assertEqual(phones.hex(),case['phones']);self.assertFalse(tr.events);count+=1
        self.assertEqual(count,11180)
        initial_symbols={case['letter']for voice in fixture['voices']for case in voice['cases']if any(event['kind']==7 and event['phones']for event in case['events'])}
        self.assertTrue({0xa7,0x1f600}<=initial_symbols)

    def test_original_600_spelling_stress_and_12291_super_subscript_values(self):
        fixture=self.oracle['letters'];tr=SimpleNamespace(langopts=SimpleNamespace())
        for case in fixture['stress']:
            tr.langopts.spelling_stress=case['spelling_stress'];actual=NumberTranslator.set_spelling_stress(tr,bytes.fromhex(case['input']),case['control'],case['n_chars'])
            self.assertEqual(actual.hex(),case['expected'])
        for code,expected in fixture['superscripts']:self.assertEqual(NumberTranslator.is_superscript(code),expected)
        self.assertEqual(len(fixture['stress']),600);self.assertEqual(len(fixture['superscripts']),12291)


if __name__=='__main__':unittest.main()
