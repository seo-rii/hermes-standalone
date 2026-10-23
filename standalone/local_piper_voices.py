"""Original eSpeak source defaults, language/voice files and ordered overrides.

Derived from voices.c/langopts.c/tr_languages.c, Copyright2005-2015 Jonathan
Duddington, Copyright2015-2022 Reece H.Dunn and Juho Hiltunen, GPL-3.0-or-later.
All default constants and original package data are embedded; no word/sentence
pronunciations are cached. See local-piper-voices-provenance.json and notices.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import io
import re
import struct
from types import SimpleNamespace

from .local_piper_voice_tables import DATA as _DATA
@dataclass(frozen=True)
class Alphabet:
    name:str
    offset:int
    range_min:int
    range_max:int
    language:int
    flags:int


ALPHABETS=tuple(Alphabet(**row)for row in _DATA['alphabets'])
_CONFIG_BY_ID={v['translator_name']:v for k,v in _DATA['profiles'].items()if k!='~default'}
_FIXED_DICTIONARIES={v['translator_name']:v['dictionary_name']for k,v in _DATA['profiles'].items()
                    if k!='~default'and v['dictionary_name']!=k}


def alphabet_from_char(code):
    code&=0xffffffff
    for row in ALPHABETS:
        if code<=row.range_max:
            return row if code>=row.range_min else None
    return None


def _signed(value,bits=32):
    value&=(1<<bits)-1
    return value-(1<<bits)if value&(1<<(bits-1))else value


def translator_config(language):
    encoded=language.encode('utf8').split(b'\0',1)[0]
    if len(encoded)>39:
        raise ValueError('eSpeak translator language exceeds its name field')
    name=encoded.decode('utf8')
    identifier=0
    for char in encoded:
        identifier=_signed((identifier<<8)+_signed(char,8))
    values=deepcopy(_CONFIG_BY_ID.get(identifier,_DATA['profiles']['~default']))
    values['translator_name']=identifier;values['dictionary_name']=_FIXED_DICTIONARIES.get(identifier,name)
    for field in ('ordinal_indicator','roman_suffix','length_mods','length_mods0'):
        if values['langopts'][field]is not None:values['langopts'][field]=bytes(values['langopts'][field])
    values['langopts']=SimpleNamespace(**values['langopts'])
    values['letter_bits']=bytearray(values['letter_bits'])
    if values['transpose_map']is not None:values['transpose_map']=bytes(values['transpose_map'])
    for field in ('phonemes_repeat_count','phoneme_tab_ix','expect_verb','expect_past','expect_verb_s','expect_noun',
                  'prev_last_stress','word_vowel_count','word_stressed_count','clause_upper_count','clause_lower_count',
                  'prepause_timeout','end_stressed_vowel','clause_terminator'):
        values[field]=0
    values['phonemes_repeat']=b'';values['prev_dict_flags']=[0,0];values['clause_end']=0
    values['alphabet_from_char']=alphabet_from_char
    return SimpleNamespace(**values)


def _lines(data,width):
    stream=io.BytesIO(data)
    while line:=stream.readline(width-1):
        line=line.split(b'\0',1)[0]
        if line.startswith(b'#'):continue
        line=line.rstrip(b' \t\r\n\v\f').split(b'//',1)[0]
        match=re.match(rb'([^ \t\r\n\v\f]+)[ \t\r\n\v\f]?(.*)',line)
        if match:yield match[1].decode('utf8',errors='surrogateescape'),match[2]


def _integers(data,limit):
    result=[];cursor=0
    while len(result)<limit:
        match=re.match(rb'[ \t\r\n\v\f]*([+-]?[0-9]+)',data[cursor:])
        if not match:break
        value=int(match[1]);value=max(-(1<<63),min((1<<63)-1,value))
        result.append(_signed(value));cursor+=match.end()
    return result


def _option_numbers(data):
    # The C loop does not advance on a punctuation token. Refuse malformed
    # configuration rather than entering its unbounded loop.
    result=[]
    for word in data.split():
        if not re.fullmatch(rb'[A-Za-z0-9]+',word):
            raise ValueError('Malformed eSpeak bit option')
        match=re.match(rb'[0-9]+',word)
        if match:result.append(int(match[0]))
    return result


@dataclass(frozen=True)
class VoiceProfile:
    config:SimpleNamespace
    phoneme_tab_ix:int
    dictionary_name:str
    voice_id:str
    replace_phonemes:tuple
    name:str
    languages:tuple
    option_tone_flags:int
    # The pinned Piper build disables USE_KLATT, including named variants.
    klatt:bool=False


@dataclass(frozen=True)
class _Voice:
    identifier:str
    name:bytes
    languages:tuple
    variants:int
    gender:int
    age:int


class VoiceLoader:
    def __init__(self,data_dir=None,*,select_table=None,lookup_phone=None):
        self.data_dir=Path(data_dir)if data_dir is not None else Path(__file__).with_name('assets')/'piper/espeak-ng-data'
        content=(self.data_dir/'phontab').read_bytes()
        if len(content)<4 or not 0<content[0]<=150:raise ValueError('Invalid eSpeak phoneme table count')
        self.tables=[];position=4
        for index in range(content[0]):
            if position+36>len(content):raise ValueError('Truncated eSpeak phoneme table header')
            count,includes=content[position:position+2]
            name=content[position+4:position+36].split(b'\0',1)[0].decode('ascii')
            end=position+36+count*16
            if end>len(content)or includes>index:raise ValueError('Invalid eSpeak phoneme table entries')
            phones=[struct.unpack_from('<IIH6B',content,position+36+i*16)for i in range(count)]
            self.tables.append((name,includes,phones));position=end
        if position!=len(content):raise ValueError('Trailing eSpeak phoneme table data')
        content=(self.data_dir/'intonations').read_bytes()
        if len(content)%68:raise ValueError('Truncated eSpeak intonation table')
        self.tune_names=[content[i:i+12].split(b'\0',1)[0]for i in range(0,len(content),68)]
        self._catalog=None;self.option_tone_flags=0
        if (select_table is None)!=(lookup_phone is None):raise ValueError('Both eSpeak active-table callbacks are required')
        self._active_table=0;self._active_count=0;self._active_phones={}
        self.select_table=select_table or self._select_table
        self.lookup_phone=lookup_phone or self._lookup_phone

    translator_config=staticmethod(translator_config)

    def _file(self,identifier):
        if '\\'in identifier or identifier.startswith('/')or any(part in ('.','..')for part in identifier.split('/')):
            raise ValueError('Invalid eSpeak voice identifier')
        for kind in ('voices','lang'):
            path=self.data_dir/kind/identifier
            if path.is_file()and path.stat().st_size:return path
        return None

    def _voices(self):
        if self._catalog is not None:return self._catalog
        result=[]
        for kind in ('voices','lang'):
            for path in sorted((self.data_dir/kind).rglob('*')):
                if not path.is_file()or any(p.startswith('.')for p in path.relative_to(self.data_dir/kind).parts):continue
                identifier=path.relative_to(self.data_dir/kind).as_posix()
                name=None;languages=[];variants=4;gender=1;age=0;length=0
                for key,value in _lines(path.read_bytes(),120):
                    words=value.split()
                    if key=='name':name=value.lstrip(b' \t\r\n\v\f')[:79]
                    elif key=='language'and words:
                        priority=_integers(b' '.join(words[1:]),1)or[5]
                        if len(words[0])+2<299-length:
                            languages.append((priority[0]&255,words[0]));length+=len(words[0])+2
                    elif key=='variants':
                        values=_integers(value,1)
                        if values:variants=values[0]
                    elif key=='gender'and words:
                        gender=2 if words[0]==b'female'else 1
                        values=_integers(b' '.join(words[1:]),1)
                        if values:age=values[0]
                if languages:result.append(_Voice(identifier,name or identifier.encode(),tuple(languages),variants,gender,age))
        result.sort(key=lambda v:(v.languages[0][1],v.languages[0][0],v.name))
        self._catalog=tuple(result)
        return self._catalog

    def _select_name(self,name):
        suffix=b'/'+name;exact=None;last=None
        for voice in self._voices():
            if voice.name.lower()==name:return voice.identifier
            identifier=voice.identifier.encode().lower()
            if identifier==name:exact=voice.identifier
            elif identifier.endswith(suffix):last=voice.identifier
        return exact or last

    def _table(self,name):
        return next((i for i,row in enumerate(self.tables)if row[0]==name),None)

    def _select_table(self,index):
        if self._active_table==index:return
        self._active_count=0
        def include(number):
            _,parent,rows=self.tables[number]
            if parent:include(parent-1)
            for row in rows:
                self._active_phones[row[3]]=row[0]
                if row[3]>self._active_count:
                    for code in range(self._active_count+1,row[3]):self._active_phones.pop(code,None)
                    self._active_count=row[3]
        include(index);self._active_count+=1;self._active_table=index

    def _lookup_phone(self,name):
        mnemonic=int.from_bytes(name.split(b'\0',1)[0][:4].ljust(4,b'\0'),'little')
        return next((code for code in range(self._active_count)if self._active_phones.get(code)==mnemonic),0)

    def _set_table_name(self,name):
        index=self._table(name)
        if index is not None:self.select_table(index)
        return index

    def _options(self,tr,key,value):
        if tr is None:return
        options=tr.langopts
        params={'apostrophe':16,'brackets':12,'bracketsAnnounced':17,'l_dieresis':0,'l_prefix':2,
                'l_regressive_v':3,'l_unpronouncable':4,'l_sonorant_min':5}
        if key in params:
            values=_integers(value,1)
            if values:options.param[params[key]]=values[0]
        elif key=='dict_min':
            values=_integers(value,1)
            if values:tr.dict_min_size=values[0]
        elif key in ('dictrules','stressOpt','numbers'):
            for n in _option_numbers(value):
                if n<=0:continue
                if key=='numbers':
                    if n<32:options.numbers=_signed(options.numbers|1<<n)
                    elif n<64:options.numbers2=_signed(options.numbers2|1<<(n-32))
                elif n<32:
                    if key=='dictrules':tr.dict_condition=_signed(tr.dict_condition|1<<n)
                    else:options.stress_flags=_signed(options.stress_flags|1<<n)
            if key=='numbers':
                if options.numbers&8:options.thousands_sep=46;options.decimal_sep=44
                if options.numbers&4:options.thousands_sep=0
        elif key in ('lowercaseSentence','spellingStress'):
            setattr(options,'lowercase_sentence'if key=='lowercaseSentence'else'spelling_stress',True)
        elif key in ('stressAdd','stressAmp','stressLength'):
            for i,n in enumerate(_integers(value,8)):
                if key=='stressAmp':tr.stress_amps[i]=n&255
                else:tr.stress_lengths[i]=_signed(n+(tr.stress_lengths[i]if key=='stressAdd'else 0),16)
        elif key in ('stressRule','words'):
            names=('stress_rule','unstressed_wd1','unstressed_wd2')if key=='stressRule'else('word_gap','vowel_pause')
            for name,n in zip(names,_integers(value,len(names))):setattr(options,name,n)
        elif key=='intonation':
            values=_integers(value,1)
            if values:
                self.option_tone_flags=values[0]
                if values[0]&255:options.intonation_group=values[0]&255
        elif key=='tunes':
            options.intonation_group=0
            for i,name in enumerate(value.split()[:6]):
                if name!=b'NULL'and name in self.tune_names:options.tunes[i]=self.tune_names.index(name)

    def select(self,voice):
        if not isinstance(voice,str):raise TypeError('eSpeak voice name must be text')
        if '\0'in voice:raise ValueError('embedded null character')
        encoded=voice.encode('utf8')[:59]
        name,separator,variant=encoded.partition(b'+')
        if separator and variant[:1].isdigit():
            value=_integers(variant,1)[0]
            variant=(('m'+str(value))if value<10 else('f'+str(value-10))).encode()if value>0 else b''
        name=name.lower();identifier=name[:39].decode('utf8')
        path=self._file(identifier)if identifier else None
        if path is None:
            identifier=self._select_name(name[:39])
            if identifier is None:raise ValueError('Unknown eSpeak voice: '+voice)
            path=self._file(identifier)
        if path is None:raise ValueError('Missing eSpeak voice file')
        tr=None;dictionary='en';phonemes='';languages=[];length=0;display=b'';replacements=[]
        def read(path,tone_only=False):
            nonlocal tr,dictionary,phonemes,display,length
            table_set=False
            for key,value in _lines(path.read_bytes(),4126):
                words=value.split()
                if key=='language'and not tone_only and words and words[0]!=b'variant':
                    priority=_integers(b' '.join(words[1:]),1)or[5]
                    if len(words[0])+2<99-length:
                        languages.append((priority[0]&255,words[0].decode()));length+=len(words[0])+2
                    if tr is None:
                        language=words[0].split(b'-',1)[0].decode();tr=translator_config(language)
                        dictionary=phonemes=language
                        self._set_table_name(language)
                elif key=='name'and not tone_only:display=value.lstrip(b' \t\r\n\v\f')[:39]
                elif key=='dictionary'and words:dictionary=words[0].decode()
                elif key=='phonemes'and words:phonemes=words[0].decode()
                elif key=='replace'and len(words)>=2 and len(replacements)<60:
                    flags=_integers(value,1)
                    if not flags:continue
                    if not table_set:
                        self._set_table_name(phonemes);table_set=True
                    old=self.lookup_phone(words[1])
                    if old:
                        new=self.lookup_phone(words[2]if len(words)>2 else b'NULL')
                        replacements.append((old,new,_signed(flags[0],8)))
                else:self._options(tr,key,value)
        read(path)
        if tr is None:tr=translator_config('en')
        table=self._table(phonemes);table=0 if table is None else table
        self.select_table(table)
        tr.phoneme_tab_ix=table;tr.dictionary_name=dictionary
        if variant:
            variant_path=self._file('!v/'+variant.decode('utf8'))
            if variant_path is not None:
                read(variant_path,True)
                identifier=identifier+'+'+variant.decode('utf8')
        return VoiceProfile(tr,table,tr.dictionary_name,identifier,tuple(replacements),display.decode('utf8'),tuple(languages),self.option_tone_flags)
