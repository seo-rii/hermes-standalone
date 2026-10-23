"""Owned translation of eSpeak's compiled dictionary lookup and morphology flags."""
from dataclasses import dataclass
from types import SimpleNamespace

from standalone.local_piper_phoneme_data import hash_dictionary
from standalone.local_piper_phoneme_unicode import utf8_in,utf8_out,is_alpha


@dataclass(frozen=True)
class LookupResult:
    found:bool
    phonemes:bytes
    flags:tuple[int,int]
    end:int|None=None
    replacement:bytes|None=None


@dataclass(frozen=True)
class RuleResult:
    phonemes:bytes
    end_type:int=0
    end_phonemes:bytes=b''
    dict_flags:int=0


@dataclass
class WordTab:
    flags:int=0
    start:int=0
    pre_pause:int=0
    sourceix:int=0
    length:int=0


def transpose_alphabet(tr,word):
    """Return the original mutated buffer and encoded word-length flags.

    The source copies packed bytes over the input without copying their NUL.
    Retaining the untouched suffix matters: HashDictionary reads it until the
    first NUL, while dictionary entry comparison uses the explicit packed size.
    """
    word=bytes(word).split(b'\0',1)[0]
    minimum=tr.transpose_min;maximum=tr.transpose_max
    mapped=[];cursor=0
    while True:
        code,size=utf8_in(word,cursor);cursor+=size
        if code==0:break
        if not minimum<=code<=maximum:return word,len(word)
        if tr.transpose_map is None:value=code-minimum+1
        else:
            value=tr.transpose_map[code-minimum]
            if value>=128:value-=256
            if value<=0:return word,len(word)
        mapped.append(value)
        if len(mapped)>=160:break
    packed=bytearray();accumulator=0;bits=0;cursor=0
    pairs=tr.frequent_pairs
    while cursor<len(mapped):
        code=mapped[cursor];cursor+=1
        if pairs is not None:
            pair=code+((mapped[cursor] if cursor<len(mapped) else 0)<<8)
            for index,value in enumerate(pairs):
                if pair<value:break
                if pair==value:
                    code=index+maximum-minimum+2;cursor+=1;break
        accumulator=(accumulator<<6)+(code&0x3f);bits+=6
        if bits>=8:
            bits-=8;packed.append(accumulator>>bits&0xff)
    if bits:packed.append(accumulator<<(8-bits)&0xff)
    return bytes(packed)+word[len(packed):],len(packed)|0x40


class DictionaryLookup:
    def __init__(self,tr,dictionary,primary=None,state=None):
        self.tr=tr;self.dictionary=dictionary
        self.primary=tr if primary is None else primary
        self.state=SimpleNamespace(dictionary_skipwords=0) if state is None else state

    @property
    def dictionary_skipwords(self):return self.state.dictionary_skipwords

    @dictionary_skipwords.setter
    def dictionary_skipwords(self,value):self.state.dictionary_skipwords=value

    def _candidate_flags(self,entry,position,text,next_cursor,word_table):
        tr=self.tr;first=second=0;end=next_cursor;failed=False
        while position<len(entry):
            flag=entry[position];position+=1
            if flag>=100:
                condition=tr.dict_condition&(1<<((flag-(132 if flag>=132 else 100))&31))
                if bool(condition)!=(flag<132):failed=True
            elif flag>80:
                count=flag-80
                if word_table:
                    for item in word_table[:count+1]:
                        if not item.length:break
                        if item.flags&0xc00:failed=True
                remainder=entry[position:]
                if text[next_cursor:next_cursor+len(remainder)].split(b'\0',1)[0]!=remainder.split(b'\0',1)[0]:failed=True
                if not failed:first|=0x80;self.dictionary_skipwords=count;end=next_cursor+len(remainder)
                break
            elif flag>64:
                first=(first&~15)|(flag&15)
                if flag&12==12:first|=0x200
            elif flag>=32:second|=(1<<(flag-32))&0xffffffff
            else:first|=1<<flag
        return first,second,end,failed

    def _context_matches(self,first,second,end,end_flags,wflags,lookup_symbol):
        if not end_flags&4 and second&0x10000:return False
        if end_flags&0x400 and second&0xc000:return False
        if end_flags&4 and (second&0x4000 or second&0x8000 and not end_flags&8):return False
        if second&0x200 and not wflags&2:return False
        if second&0x400 and not wflags&1:return False
        if first&0x2000000 and not wflags&0x10000:return False
        before_clause_end=end<self.primary.clause_end
        if getattr(self.state,'lookup_outside_clause',False):
            # The caller profile owns this ordering; see word provenance.
            before_clause_end=getattr(self.state,'clause_buffer',None) is not None
        if second&0x20000 and before_clause_end and not lookup_symbol:return False
        if second&0x40000 and not wflags&0x200:return False
        if second&0x2000 and not self.primary.clause_terminator&0x80000:return False
        return True

    def _grammar_matches(self,first,second,end_flags):
        tr=self.tr
        if second&0x10:
            if not(tr.expect_verb or tr.expect_verb_s and end_flags&8):return False
            if tr.translator_name==0x656e and tr.prev_dict_flags[0]&0x200000 and end_flags&8:return False
        if second&0x40 and not tr.expect_past:return False
        if second&0x20 and (not tr.expect_noun or end_flags&0x800):return False
        if second&0x80000 and tr is not self.primary:return False
        if first&0x10000 and tr.translator_name==0x6875 and not tr.prev_dict_flags[0]&0x8000:return False
        return True

    def lookup_raw(self,word,text=b'',next_cursor=0,*,flags=(0,0),end_flags=0,word_table=None,phonemes=b''):
        tr=self.tr;word=bytes(word).split(b'\0',1)[0];text=bytes(text)
        if tr.transpose_min>0:word,wlen=transpose_alphabet(tr,word[:159])
        else:wlen=len(word)
        flags=list(flags);lookup_symbol=flags[1]&0x40000000
        wflags=word_table[0].flags if word_table else 0
        for entry in self.dictionary.buckets[hash_dictionary(word)]:
            if entry[1]&0x7f!=wlen or entry[2:2+(wlen&0x3f)]!=word[:wlen&0x3f]:continue
            position=2+(entry[1]&0x3f)
            if entry[1]&0x80:phonemes=b''
            else:
                stop=entry.index(0,position);phonemes=entry[position:stop];position=stop+1
            first,second,end,failed=self._candidate_flags(entry,position,text,next_cursor,word_table)
            if failed:continue
            if not self._context_matches(first,second,end,end_flags,wflags,lookup_symbol):continue
            if not self._grammar_matches(first,second,end_flags):continue
            flags=[first|0x40000000,second]
            if not phonemes:return LookupResult(False,phonemes,tuple(flags))
            flags[0]|=0x80000000
            code,size=utf8_in(word)
            if (size>=len(word) or word[size]==0) and not is_alpha(code):flags[0]|=0x8000000
            return LookupResult(True,phonemes,tuple(flags),end)
        return LookupResult(False,phonemes,tuple(flags))

    def _abbreviation(self,text,cursor,flags,end_flags,word_table):
        word2=cursor;abbreviation=bytearray()
        def byte(index):return text[index] if 0<=index<len(text) else 0
        while True:
            first=byte(word2);size=1 if first<0x80 else 2 if first<0xe0 else 3 if first<0xf0 else 4
            if byte(word2+size)!=32 or byte(word2+size+1)!=46:break
            if len(abbreviation)+1>160:
                abbreviation.clear();break
            abbreviation.extend(text[word2:word2+size]);abbreviation.append(46);word2+=size+3
        if abbreviation:
            tail=word2
            while byte(tail) not in (0,32):tail+=1
            if len(abbreviation)+tail-word2+1<=160:
                result=self.lookup_raw(bytes(abbreviation)+text[word2:tail],text,word2,flags=flags,end_flags=end_flags,word_table=word_table)
                flags=result.flags
                if result.found:
                    self.dictionary_skipwords=len(abbreviation)
                    return LookupResult(True,result.phonemes,(flags[0]|0x80,flags[1]),result.end),flags
        return None,flags

    def lookup_dict_list(self,text,cursor=0,*,flags=(0,0),end_flags=0,word_table=None):
        text=bytes(text);tr=self.tr
        abbreviation,flags=self._abbreviation(text,cursor,flags,end_flags,word_table)
        if abbreviation is not None:return abbreviation
        def byte(index):return text[index] if 0<=index<len(text) else 0
        word=bytearray();word1=cursor
        for length in range(159):
            value=byte(word1);word1+=1
            if value in (0,32) or value==46 and word and 48<=word[-1]<=57:break
            word.append(value)
        result=self.lookup_raw(word,text,word1,flags=flags,end_flags=end_flags,word_table=word_table)
        flags=result.flags;phones=result.phonemes;found=result.found;end=result.end
        if flags[0]&0x8000000:
            if phones==tr.phonemes_repeat:
                tr.phonemes_repeat_count+=1
                if tr.phonemes_repeat_count>3:phones=b''
            else:tr.phonemes_repeat=phones[:19];tr.phonemes_repeat_count=1
        else:tr.phonemes_repeat_count=0
        if not found and flags[1]&0x800:
            start=1 if word[:1]==b'_' else 0;letter,size=utf8_in(word,start)
            phones=tr.lookup_accented_letter(letter);found=True;end=cursor+start+size
        if not found and len(word)>=2:
            phones=b''
            if end_flags&0x10 and word[-1]==101 or end_flags&0x1000 and word[-1]==word[-2]:
                result=self.lookup_raw(word[:-1],text,word1,flags=flags,end_flags=end_flags,word_table=word_table)
                flags=result.flags;phones=result.phonemes;found=result.found;end=result.end
        if found:
            if tr.langopts.textmode:flags=(flags[0]^0x20000000,flags[1])
            if flags[0]&0x20000000:
                replacement=phones+b' ' if end_flags&2 else None
                return LookupResult(False,b'',flags,end,replacement)
            return LookupResult(True,phones,flags,end)
        return LookupResult(False,b'',flags)

    def lookup(self,word):
        result=self.lookup_dict_list(word,flags=(0,0x40000000),end_flags=2)
        if result.flags[0]&0x20000000:
            # The translator owns option_sayas and the real replacement-word
            # controller. The original always restores say-as after this call.
            previous=self.tr.option_sayas;self.tr.option_sayas=0
            try:return self.tr.translate_word((result.replacement or bytes(word))[:76])
            finally:self.tr.option_sayas=previous
        flags=result.flags[0] if result.found else 0
        return flags-0x100000000 if flags&0x80000000 else flags,result.phonemes
