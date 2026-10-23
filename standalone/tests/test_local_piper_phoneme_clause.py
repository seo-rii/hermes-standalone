"""Independent original eSpeak ReadClause output and state vectors."""
import base64
import hashlib
import importlib.abc
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
import zlib

from standalone.local_piper_phoneme_clause import ClauseReader,clause_type_from_codepoint


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]in {'ctypes','subprocess','numpy','piper','espeakbridge'}:
            raise AssertionError('eSpeak clause port attempted native/process import '+fullname)


class OriginalNameDependencies(ClauseReader):
    """Explicit native dictionary helper dependencies, only in tests."""
    def lookup_special(self,tr,word):
        data=tr.record['special'][word.decode()];tr.phonemes_repeat_count=data['repeat_count']
        return None if data['phones']is None else bytes.fromhex(data['phones'])

    def lookup_char_name(self,tr,c,only=False):
        data=tr.record['char_names'][str(c)+':'+str(int(only))];tr.phonemes_repeat_count=data['repeat_count']
        return bytes.fromhex(data['phones'])


class OriginalDictionaryEvents:
    """Finite original dictionary/stress/decode call transcripts for tests."""
    def __init__(self,test,events):self.test=test;self.events=list(events)

    def event(self,kind,word):
        self.test.assertTrue(self.events);event=self.events.pop(0)
        self.test.assertEqual(event['kind'],kind);self.test.assertEqual(bytes.fromhex(event['word']),word)
        return event

    def english_translator(self):return self

    def lookup_dict_list(self,word,cursor=0,*,flags=(0,0),end_flags=0,word_table=None):
        event=self.event(1,word[cursor:]);self.test.assertEqual(list(flags),event['before_flags'])
        return SimpleNamespace(found=event['result'],phonemes=bytes.fromhex(event['phones']),flags=event['flags'])

    def translate_rules(self,word,*,max_phonemes=200,word_flags=0,dict_flags=None,initial=b''):
        before=self.event(7,word);self.test.assertEqual(before['phones'],initial.hex());self.test.assertEqual(before['flags'],[max_phonemes,word_flags])
        event=self.event(3,word);self.test.assertEqual(event['flags'],[max_phonemes,word_flags])
        return SimpleNamespace(phonemes=bytes.fromhex(event['phones']))

    def set_word_stress(self,phones,flags,tonic=-1,control=0):
        event=self.event(4,phones);self.test.assertEqual(event['before_flags'],[tonic&0xffffffff,0]);self.test.assertEqual(event['result'],control)
        return bytes.fromhex(event['phones'])

    def decode_phonemes(self,phones):return bytes.fromhex(self.event(6,phones)['phones'])


class LocalPiperClauseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wrapper=json.loads((Path(__file__).parent/'fixtures/local_piper_phoneme_clause.json').read_text())
        raw=zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'],validate=True))
        if hashlib.sha256(raw).hexdigest()!=wrapper['data_sha256']:raise AssertionError('Piper clause fixture digest mismatch')
        cls.oracle=json.loads(raw)

    def setUp(self):
        guard=NoNative();sys.meta_path.insert(0,guard);self.addCleanup(sys.meta_path.remove,guard)

    @staticmethod
    def translator(voice):
        return SimpleNamespace(record=voice,langopts=SimpleNamespace(**voice['langopts']),translator_name=voice['translator_name'],chars_ignore=voice['chars_ignore'])

    def assert_clause(self,reader,tr,actual,expected,context):
        state={key:getattr(reader,key)for key in expected['state']};state['option_punctlist']=list(state['option_punctlist'])
        result=dict(buffer=actual.buffer.hex(),charix=list(actual.charix),charix_top=actual.charix_top,terminator=actual.terminator,tone=actual.tone,voice_change=actual.voice_change.hex(),cursor=actual.cursor,state=state,tr={key:getattr(tr,key)for key in expected['tr']})
        if result==expected:return
        differences={}
        for key,value in expected.items():
            if result[key]==value:continue
            if key in ('buffer','charix'):
                index=next((i for i,(a,b)in enumerate(zip(result[key],value))if a!=b),min(len(result[key]),len(value)))
                differences[key]=(index,result[key][max(0,index-8):index+32],value[max(0,index-8):index+32])
            else:differences[key]=(result[key],value)
        self.fail(str((context,differences)))

    def test_original_six_languages_8421_full_clause_states(self):
        count=0
        for voice in self.oracle['clauses']['voices']:
            for case in voice['cases']:
                reader=OriginalNameDependencies(case['text']);tr=self.translator(voice)
                for key,value in case['options'].items():
                    if key=='silence':reader.speech_parameters[0]=value
                    else:setattr(reader,key,value)
                for index,expected in enumerate(case['clauses']):
                    self.assert_clause(reader,tr,reader.read(tr,1024),expected,(voice['voice'],case['text'][:80],case['options'],index));count+=1
        self.assertEqual(count,8421)

    def test_original_all_unicode_clause_properties(self):
        fixture=self.oracle['clauses'];digest=hashlib.sha256()
        for cp in range(fixture['clause_type_unicode_values']):digest.update(struct.pack('<i',clause_type_from_codepoint(cp)))
        self.assertEqual(fixture['clause_type_unicode_values'],1114112);self.assertEqual(digest.hexdigest(),fixture['clause_type_sha256'])

    def test_original_2547_character_name_dictionary_callbacks(self):
        count=0
        for voice in self.oracle['helpers']['voices']:
            for case in voice['cases']:
                with self.subTest(voice=voice['voice'],kind=case['kind'],letter=case.get('letter'),only=case.get('only')):
                    reader=ClauseReader();tr=OriginalDictionaryEvents(self,case['events']);tr.translator_name=voice['translator_name']
                    actual=reader.lookup_char_name(tr,case['letter'],case['only'])if case['kind']=='char'else reader.lookup_special(tr,case['word'].encode())
                    self.assertEqual(None if actual is None else actual.hex(),case['phones']);self.assertFalse(tr.events);count+=1
        self.assertEqual(count,2547)

    def test_original_retained_binding_initialization_controls_and_caller_buffers(self):
        voices={voice['voice']:voice for voice in self.oracle['clauses']['voices']};count=0
        for case in self.oracle['states']['scenarios']:
            reader=OriginalNameDependencies(soundicons={int(k):v for k,v in case['soundicons'].items()});tr=self.translator(voices[case['voice']])
            for key,value in case['initial'].items():self.assertEqual(getattr(reader,key),tuple(value)if key=='option_punctlist'else value,(case['name'],key))
            for step in case['steps']:
                if step.get('initialize'):reader.initialize()
                for key,value in step.get('set',{}).items():setattr(reader,key,value)
                if 'text'in step:reader.bind_text(step['text'])
                for index,expected in enumerate(step['clauses']):
                    self.assert_clause(reader,tr,reader.read(tr,step.get('n_buf',800)),expected,(case['voice'],case['name'],index));count+=1
        self.assertEqual(count,self.oracle['state_clause_count'])
        self.assertEqual(len(self.oracle['states']['scenarios']),156)


if __name__=='__main__':unittest.main()
