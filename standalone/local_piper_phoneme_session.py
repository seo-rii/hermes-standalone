"""Per-session native-free eSpeak translators and persistent language slots."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

from standalone.local_piper_numbers import NumberTranslator
from standalone.local_piper_phoneme_data import Dictionary,PhonemeData,PhonemeSelection
from standalone.local_piper_phoneme_dictionary import DictionaryLookup
from standalone.local_piper_phoneme_letters import is_letter
from standalone.local_piper_phoneme_rules import Rules,append_phonemes
from standalone.local_piper_phoneme_stress import set_word_stress
from standalone.local_piper_phoneme_codec import encode_phonemes,decode_phonemes
from standalone.local_piper_phoneme_unicode import is_alpha,is_walpha,is_wdigit,is_wspace
from standalone.local_piper_voices import VoiceLoader,translator_config


@dataclass
class SpeechState:
    option_sayas:int=0
    pre_pause:int=0
    dictionary_skipwords:int=0
    option_tone_flags:int=0
    option_capitals:int=0
    any_stressed_words:bool=False
    current_alphabet:object=None
    native_calls_depth:int=0
    word_phonemes:bytes=b''
    lookup_outside_clause:bool=False


class Session:
    def __init__(self,data_dir):
        self.data=PhonemeData(data_dir);self.selection=PhonemeSelection(self.data)
        self.loader=VoiceLoader(data_dir,select_table=self.selection.select,lookup_phone=self.selection.lookup_phone)
        self.state=SpeechState()
        self.primary=None;self.secondary=None;self.tertiary=None
        self.numbers=NumberTranslator();self.dictionaries={}

    def dictionary(self,name):
        if name not in self.dictionaries:
            self.dictionaries[name]=Dictionary((self.data.directory/(name+'_dict')).read_bytes())
        return self.dictionaries[name]

    def select_voice(self,name):
        profile=self.loader.select(name)
        self.selection.select(profile.phoneme_tab_ix)
        self.primary=Translator(self,profile.config)
        self.primary.phoneme_tab_ix=profile.phoneme_tab_ix
        for alternate in (self.secondary,self.tertiary):
            if alternate is not None:alternate.lookup_engine.primary=self.primary
        self.numbers.translator=self.primary
        self.state.option_tone_flags=profile.option_tone_flags
        self.voice=profile
        return self.primary

    def translator(self,language,slot=2):
        attribute='secondary' if slot==2 else 'tertiary'
        current=getattr(self,attribute)
        if current is not None and current.requested_language==language:
            current.phonemes_repeat=b'';return current
        config=translator_config(language)
        try:config.phoneme_tab_ix=self.data.number(language)
        except ValueError:config.phoneme_tab_ix=0
        current=Translator(self,config,primary=self.primary);current.requested_language=language
        setattr(self,attribute,current)
        return current

    def alternate(self,language,slot=2):
        try:table=self.data.number(language)
        except ValueError:
            current=getattr(self,'secondary' if slot==2 else 'tertiary')
            if current is not None:current.phonemes_repeat=b''
            return -1,current
        self.selection.select(table)
        return table,self.translator(language,slot)

    def set_translator2(self,language):return self.alternate(language,2)[0]
    def set_translator3(self,language):return self.alternate(language,3)[0]

    @contextmanager
    def native_calls(self):
        self.state.native_calls_depth+=1
        try:yield
        finally:self.state.native_calls_depth-=1

    @contextmanager
    def outside_clause(self):
        previous=self.state.lookup_outside_clause;self.state.lookup_outside_clause=True
        try:yield
        finally:self.state.lookup_outside_clause=previous


class Translator:
    def __init__(self,session,config,*,primary=None):
        vars(self).update(vars(deepcopy(config)));self.session=session
        self.dictionary=session.dictionary(self.dictionary_name)
        self.langopts.replace_chars=self.dictionary.replacement_pairs
        self.lookup_engine=DictionaryLookup(self,self.dictionary,primary or self,session.state)
        self.rules=Rules(self,self.dictionary)

    @contextmanager
    def active_table(self):
        if self.session.state.native_calls_depth:
            yield;return
        selection=self.session.selection;previous=selection.current
        selection.select(self.phoneme_tab_ix)
        try:yield
        finally:selection.select(previous)

    @property
    def phoneme_tab(self):return self.session.selection.phones

    @property
    def n_phoneme_tab(self):return self.session.selection.count

    @property
    def option_sayas(self):return self.session.state.option_sayas

    @option_sayas.setter
    def option_sayas(self,value):self.session.state.option_sayas=value;self.session.numbers.option_sayas=value

    @property
    def pre_pause(self):return self.session.state.pre_pause

    @pre_pause.setter
    def pre_pause(self,value):self.session.state.pre_pause=value

    def english_translator(self):return self.session.translator('en',2)
    def lookup_english(self,word):
        _,foreign=self.session.alternate('en',3)
        try:return foreign.lookup(word)
        finally:self.session.selection.select(self.session.voice.phoneme_tab_ix)
    def is_letter(self,code,group):return is_letter(self,code,group)
    is_alpha=staticmethod(is_alpha)
    is_walpha=staticmethod(is_walpha)
    is_wdigit=staticmethod(is_wdigit)
    is_wspace=staticmethod(is_wspace)

    def lookup_dict_list(self,*args,**kwargs):
        with self.active_table():return self.lookup_engine.lookup_dict_list(*args,**kwargs)

    def lookup(self,word):
        with self.active_table():return self.lookup_engine.lookup(word)

    def lookup_accented_letter(self,letter):
        with self.active_table():return self.session.numbers.lookup_accented_letter(self,letter)

    def lookup_letter(self,letter,next_byte,control=0):
        with self.active_table():return self.session.numbers.lookup_letter(self,letter,next_byte,control)

    def set_word_stress(self,phones,dictionary_flags=None,tonic=-1,control=0):
        with self.active_table():return set_word_stress(self,phones,dictionary_flags,tonic,control)

    def append_phonemes(self,string,size,phones):
        with self.active_table():return append_phonemes(self,string,size,phones)

    def encode_phonemes(self,text,cursor=0):
        with self.active_table():return encode_phonemes(self,text,cursor)

    def decode_phonemes(self,phones):
        with self.active_table():return decode_phonemes(self,phones)

    def translate_rules(self,word,*,max_phonemes=200,end_phonemes=False,word_flags=0,dict_flags=None,initial=b''):
        buffer=bytearray(b'\0 '+bytes(word)+b' \0')
        with self.active_table():return self.rules.translate(buffer,2,max_phonemes=max_phonemes,end_phonemes=end_phonemes,word_flags=word_flags,dict_flags=dict_flags,initial=initial)

    def translate_word(self,word):
        from standalone.local_piper_phoneme_word import translate_word
        buffer=bytearray(b'\0  '+bytes(word).split(b'\0',1)[0]+b' \0')
        with self.active_table(),self.session.outside_clause():result=translate_word(self,buffer,3,word_out=False)
        return result.flags,result.phonemes

    def translate_word_buffer(self,buffer,cursor,word_table=None,index=0,*,word_out=True):
        from standalone.local_piper_phoneme_word import translate_word
        return translate_word(self,buffer,cursor,None if word_table is None else word_table[index:],word_out=word_out)
