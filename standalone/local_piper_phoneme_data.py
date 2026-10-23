"""Pinned eSpeak binary phoneme/dictionary data, read with the standard library."""
from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass(frozen=True)
class Phone:
    mnemonic:int
    phflags:int
    program:int
    code:int
    type:int
    start_type:int
    end_type:int
    std_length:int
    length_mod:int

    @property
    def name(self):
        return self.mnemonic.to_bytes(4,'little').split(b'\0',1)[0].decode('ascii')


@dataclass(frozen=True)
class Table:
    name:str
    includes:int
    phones:tuple[Phone,...]


class PhonemeData:
    def __init__(self, directory):
        self.directory=Path(directory)
        phontab=(self.directory/'phontab').read_bytes()
        index=(self.directory/'phonindex').read_bytes()
        phondata=(self.directory/'phondata').read_bytes()
        if len(phondata)<8 or struct.unpack_from('<I',phondata)[0]!=0x014801:
            raise ValueError('eSpeak phondata version mismatch')
        self.sample_rate=struct.unpack_from('<I',phondata,4)[0]
        if len(index)%2:
            raise ValueError('Truncated eSpeak phonindex')
        self.program=struct.unpack('<'+str(len(index)//2)+'H',index)
        self.tables=[];position=4
        if len(phontab)<4 or not 0<phontab[0]<=150:
            raise ValueError('Invalid eSpeak phoneme table count')
        for number in range(phontab[0]):
            if position+36>len(phontab):
                raise ValueError('Truncated eSpeak phoneme table header')
            count,includes=phontab[position:position+2]
            name=phontab[position+4:position+36].split(b'\0',1)[0].decode('ascii')
            position+=36
            if position+count*16>len(phontab):
                raise ValueError('Truncated eSpeak phoneme entries')
            phones=tuple(Phone(*struct.unpack_from('<IIH6B',phontab,position+index*16)) for index in range(count))
            position+=count*16
            if includes>number:
                raise ValueError('eSpeak phoneme table inheritance cycle')
            self.tables.append(Table(name,includes,phones))
        if position!=len(phontab):
            raise ValueError('Trailing eSpeak phoneme table data')
        self.tables=tuple(self.tables)

    def number(self,name_or_number):
        if isinstance(name_or_number,str):
            number=next((index for index,table in enumerate(self.tables) if table.name==name_or_number),None)
            if number is None:
                raise ValueError('Unknown eSpeak phoneme table: '+name_or_number)
        else:
            number=name_or_number
        if type(number) is not int or not 0<=number<len(self.tables):
            raise ValueError('Invalid eSpeak phoneme table index')
        return number

    def select(self, name_or_number):
        number=self.number(name_or_number)
        phones={}
        def include(index):
            table=self.tables[index]
            if table.includes:
                include(table.includes-1)
            for phone in table.phones:
                phones[phone.code]=phone
        include(number)
        return tuple(phones.get(index) for index in range(max(phones,default=0)+1))


class PhonemeSelection:
    """Per-engine equivalent of the original retained 256-entry active view."""
    def __init__(self,data):
        # espeak_Initialize leaves the sentinel current table0 with no active
        # entries. Selection0 is initially a no-op, as in the native source.
        self.data=data;self.phones=[None]*256;self.current=0;self.count=0

    def select(self,name_or_number):
        number=self.data.number(name_or_number)
        if number==self.current:return
        maximum=0
        def include(index):
            nonlocal maximum
            table=self.data.tables[index]
            if table.includes:include(table.includes-1)
            for phone in table.phones:
                self.phones[phone.code]=phone
                if phone.code>maximum:
                    self.phones[maximum+1:phone.code]=[None]*(phone.code-maximum-1)
                    maximum=phone.code
        include(number);self.current=number;self.count=maximum+1

    def lookup_phone(self,name):
        name=bytes(name).split(b'\0',1)[0][:4]
        mnemonic=int.from_bytes(name.ljust(4,b'\0'),'little')
        for phone in self.phones[:self.count]:
            if phone is not None and phone.mnemonic==mnemonic:return phone.code
        return 0


def hash_dictionary(text):
    text=text.encode('utf8') if isinstance(text,str) else bytes(text)
    value=0;count=0
    for byte in text:
        if byte==0:
            break
        value=value*8+byte
        value=(value&0x3ff)^(value>>8)
        count+=1
    return (value+count)&0x3ff


class Dictionary:
    def __init__(self, content):
        self.content=bytes(content)
        if len(content)<=1032:
            raise ValueError('Empty eSpeak dictionary')
        count,offset=struct.unpack_from('<II',content)
        if count!=1024 or not 0<offset<=min(len(content),0x8000000):
            raise ValueError('Invalid eSpeak dictionary header')
        self.rules_offset=offset;self.buckets=[];self.bucket_offsets=[];cursor=8
        for bucket in range(1024):
            self.bucket_offsets.append(cursor)
            entries=[]
            while cursor<offset and content[cursor]:
                length=content[cursor]
                if length<2 or cursor+length>offset:
                    raise ValueError('Truncated eSpeak dictionary record')
                entries.append(bytes(content[cursor:cursor+length]))
                cursor+=length
            if cursor>=offset:
                raise ValueError('Missing eSpeak dictionary bucket terminator')
            cursor+=1;self.buckets.append(tuple(entries))
        if cursor!=offset:
            raise ValueError('eSpeak dictionary rules offset mismatch')
        self.buckets=tuple(self.buckets)
        self.bucket_offsets=tuple(self.bucket_offsets)
        self._init_groups()

    def _init_groups(self):
        content=self.content
        self.groups1=[None]*256;self.groups3=[None]*128;self.letter_groups=[None]*95
        self.groups2=[];self.groups2_name=[];self.groups2_count=[0]*256;self.groups2_start=[255]*256
        self.replacements=None;self.replacement_pairs=None;cursor=self.rules_offset
        if cursor>=len(content):
            raise ValueError('Missing eSpeak dictionary rules')
        if content[cursor]==7:
            return
        while cursor<len(content) and content[cursor]:
            if content[cursor]!=6:
                raise ValueError('Invalid eSpeak dictionary rule group')
            cursor+=1
            if cursor>=len(content):
                raise ValueError('Truncated eSpeak dictionary rule group')
            if content[cursor]==20:
                cursor=(cursor+4)&~3;self.replacements=cursor
                pairs=[]
                while cursor+4<=len(content) and content[cursor:cursor+4]!=b'\0'*4:
                    try:
                        middle=content.index(0,cursor);end=content.index(0,middle+1)
                    except ValueError as error:
                        raise ValueError('Unterminated eSpeak character replacement') from error
                    pairs.append((content[cursor:middle],content[middle+1:end]));cursor=end+1
                if cursor+4>len(content):
                    raise ValueError('Missing eSpeak character replacement terminator')
                self.replacement_pairs=tuple(pairs)
                cursor=content.index(7,cursor)+1
                continue
            if content[cursor]==18:
                index=(content[cursor+1]-65)&255;cursor+=2
                if index<95:
                    self.letter_groups[index]=cursor
            else:
                stop=content.index(0,cursor);name=content[cursor:stop];cursor=stop+1
                if len(name)==1:
                    self.groups1[name[0]]=cursor
                elif not name:
                    self.groups1[0]=cursor
                elif name[0]==1:
                    if not 1<=name[1]<=128:
                        raise ValueError('Invalid eSpeak alphabet rule group')
                    self.groups3[name[1]-1]=cursor
                else:
                    if len(self.groups2)>=120:
                        raise ValueError('Too many eSpeak paired rule groups')
                    if self.groups2_start[name[0]]==255:
                        self.groups2_start[name[0]]=len(self.groups2)
                    self.groups2_count[name[0]]+=1
                    self.groups2.append(cursor);self.groups2_name.append(name[0]|name[1]<<8)
            while cursor<len(content) and content[cursor]!=7:
                cursor=content.index(0,cursor)+1
            if cursor>=len(content):
                raise ValueError('Unterminated eSpeak dictionary rule group')
            cursor+=1

    def index_snapshot(self):
        return dict(buckets=list(self.bucket_offsets),groups1=self.groups1,groups2=self.groups2,groups3=self.groups3,
            letter_groups=self.letter_groups,groups2_name=self.groups2_name,groups2_count=self.groups2_count,
            groups2_start=self.groups2_start,replacements=self.replacements)
