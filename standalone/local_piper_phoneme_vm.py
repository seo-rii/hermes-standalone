"""eSpeak phoneme instruction interpreter; no synthesizer or native runtime."""
from copy import deepcopy
from dataclasses import dataclass,field


@dataclass
class Item:
    phcode:int=9
    stresslevel:int=0
    sourceix:int=0
    wordstress:int=0
    tone_ph:int=0
    synthflags:int=0
    type:int=0
    std_length:int=0
    phontab_addr:int=0
    sound_param:int=0
    ph:object=None
    length:int=0
    env:int=0
    prepause:int=0
    amp:int=0
    newword:int=0
    pitch1:int=0
    pitch2:int=0


@dataclass
class Result:
    pd_control:int=0
    pd_param:list[int]=field(default_factory=lambda:[0]*16)
    sound_addr:list[int]=field(default_factory=lambda:[0]*5)
    sound_param:list[int]=field(default_factory=lambda:[0]*5)
    vowel_transition:list[int]=field(default_factory=lambda:[0]*4)
    pitch_env:int=0
    amp_env:int=0
    ipa_string:str=''


def _signed(value,bits):
    return (value+(1<<(bits-1)))%(1<<bits)-(1<<(bits-1))


class Interpreter:
    def __init__(self,data,table,*,phones=None):
        self.data=data
        self.program=data.program
        self.phones=data.select(table) if phones is None else tuple(phones)

    def isolated(self,code):
        sequence=[Item(ph=self.phones[9]) for _ in range(4)]
        sequence[1].phcode=code;sequence[1].ph=self.phones[code]
        sequence[2].sourceix=1
        return self.run(sequence,1)

    def _words(self,pc):
        instruction=self.program[pc];kind=instruction>>12
        count=(0,1,0,0,1,1,0,1,1,2,4,0,0,0,0,0)[kind]
        if count:
            return count
        if kind==0:
            return 1+((instruction&255)+1)//2 if instruction&0xf00==0xd00 else 1
        if kind==6:
            return 12 if (instruction&0xf00)>>9 in (5,6) else 1
        if kind in (2,3):
            return 2 if instruction&0xf00 in (0x600,0xd00) else 1
        next_instruction=self.program[pc+2] if pc+2<len(self.program) else 0
        return 4 if next_instruction>>12==15 else 3 if next_instruction==2 else 2

    def _stress(self,sequence,index,condition,control,reduce):
        item=sequence[index]
        if self.phones[item.phcode].type==2:
            vowel=item
        elif self.phones[sequence[index+1].phcode].type==2:
            vowel=sequence[index+1]
        else:
            return False
        stress=vowel.stresslevel&15
        if reduce is not None:
            if control&1 and item.synthflags&0x10 and not reduce&1:
                return False
            if reduce&2 and stress>=vowel.wordstress:
                stress=4
        if condition==4:
            return stress>=vowel.wordstress
        if condition==3:
            return stress>3
        return stress<(1,2,4,15)[condition]

    def _condition_target(self,sequence,index,start,which,worddata):
        if which==4 and sequence[index+1].sourceix:
            return None
        if which==5 and sequence[index].sourceix:
            return None
        if which==6 and (sequence[index+1].sourceix or sequence[index+2].sourceix):
            return None
        endtype=False;previous=None
        if which in (0,5):
            if index<start+1:
                return None
            index-=1;endtype=True
        elif which in (2,4):
            index+=1
        elif which in (3,6):
            index+=2
        elif which==7:
            index+=1
            while True:
                if sequence[index].sourceix:
                    return None
                if self.phones[sequence[index].phcode].type==2:
                    break
                index+=1
        elif which==8:
            if worddata is None or worddata.get('prev_vowel') is None:
                return None
            previous=worddata['prev_vowel'];endtype=True
        elif which==9:
            if any(sequence[index+offset].sourceix for offset in (1,2,3)):
                return None
            index+=3
        elif which==10:
            if index<start+2 or sequence[index].sourceix or sequence[index-1].sourceix:
                return None
            index-=2;endtype=True
        return index,endtype,previous

    @staticmethod
    def _preceding_stress(sequence,index,start,item):
        if item.sourceix:return False
        while index>start:
            index-=1;item=sequence[index]
            if item.stresslevel&15>=4:return True
            if item.sourceix:break
        return False

    @staticmethod
    def _next_word_before_vowel(sequence,index):
        while True:
            index+=1;item=sequence[index]
            if item.sourceix:return True
            if item.ph.type==2:return False

    @staticmethod
    def _preceding_vowels(sequence,index,start):
        count=0
        while True:
            item=sequence[index];count+=item.ph.type==2
            if item.sourceix or index==start:return count
            index-=1

    def _word_condition(self,sequence,index,original,start,value,previous,reduce):
        item=sequence[index] if previous is None else previous;phone=item.ph
        if value in (0,1,2,3,4):
            if previous is not None:
                sequence=sequence[:index]+[previous]+sequence[index+1:]
            return self._stress(sequence,index,value,0,reduce)
        if value==17:return phone.type==0 or bool(sequence[original].synthflags&0x2000)
        if value==18:return item.sourceix!=0
        if value==19:return bool(sequence[index+1].sourceix or sequence[index+1].ph.type==0)
        if value==9:return self._preceding_stress(sequence,index,start,item)
        if value==10:return phone.type!=2
        if value==11:return self._next_word_before_vowel(sequence,index)
        if value==12:return phone.type in (2,3) or bool(phone.phflags&0x10)
        if value in (13,14):return self._preceding_vowels(sequence,index,start)==value-12
        if value==16:return bool(item.synthflags&0x10)
        return False

    def _condition(self,sequence,index,start,pc,control,reduce,worddata,klatt):
        instruction=self.program[pc]&0xfff;data=instruction&255;kind=instruction>>8
        if kind>=14:
            if kind!=15:return False
            return bool(control&1) if data==1 else bool(klatt) if data==2 else False
        which=self.program[pc+1] if kind%7==6 else kind%7;original=index
        target=self._condition_target(sequence,index,start,which,worddata)
        if target is None:return False
        index,endtype,previous=target;item=sequence[index] if previous is None else previous
        if which in (0,5) and item.phcode==1:
            if index<=start:
                return False
            index-=1;item=sequence[index]
        if control&0x100:
            item.ph=self.phones[item.phcode]
        phone=item.ph
        if kind<7:
            if self.phones[data].mnemonic==phone.mnemonic:
                return True
            return data==(phone.end_type if endtype and phone.type==2 else phone.start_type)
        value=instruction&31;group=instruction&0xe0
        if group==0:
            return phone.type==value
        if group==0x20:
            return phone.phflags>>16&15==value
        if group==0x40:
            return bool(phone.phflags&(1<<value))
        if group!=0x80:
            return False
        return self._word_condition(sequence,index,original,start,value,previous,reduce)

    def _simple_instruction(self,sequence,index,pc,control,end,result):
        instruction=self.program[pc];sub=instruction>>8&15;data=instruction&255
        if sub==0:
            if data==1:end=1
            elif data!=2:raise ValueError('Invalid eSpeak phoneme opcode')
        elif sub==5:
            if self.phones[sequence[index+1].phcode].type==2:result.pd_param[4]=data
        elif sub==12:result.pd_param[10]+=_signed(data,8)
        elif sub==13:
            name=bytearray()
            for offset in range(0,min(data,16),2):
                pc+=1;word=self.program[pc];name.extend((word>>8,word&255))
            result.ipa_string=bytes(name).split(b'\0',1)[0].decode('utf8')
        else:
            result.pd_param[sub]=data
            if sub==1 and control&0x100:end=1
        return pc,end

    def _conditional_instruction(self,sequence,index,start,pc,control,reduce,worddata,klatt):
        instruction=self.program[pc];truth=True;or_flag=False
        while instruction&0xe000==0x2000:
            value=self._condition(sequence,index,start,pc,control,reduce,worddata,klatt)
            pc+=self._words(pc)
            if self.program[pc]==3:value=not value;pc+=1
            truth=truth or value if or_flag else truth and value
            or_flag=bool(instruction&0x1000);instruction=self.program[pc]
        if not truth:
            if instruction&0xf800==0x6800:pc+=instruction&255
            else:
                pc+=self._words(pc)
                if self.program[pc]&0xfe00==0x6000:pc+=1
        return pc-1

    def _vowel_instruction(self,sequence,index,pc,result):
        instruction=self.program[pc];branch=(instruction>>8&15)>>1
        if branch==0:return pc+(instruction&255)-1
        if branch not in (5,6):return pc
        slot=2 if branch==5 else 3
        if slot==2:result.pd_control|=2;vowel=sequence[index+1].ph.start_type
        else:vowel=sequence[index-1].ph.end_type
        vowel-=28
        if 0<=vowel<6:
            offset=pc+vowel*2
            result.sound_addr[slot]=((self.program[offset+1]&15)<<16|self.program[offset+2])*4
            result.sound_param[slot]=_signed(self.program[offset+1]>>4&255,8)
        return pc+12

    def _sound_instruction(self,pc,end,result):
        instruction=self.program[pc];slot=(instruction>>12)-11
        result.sound_addr[slot]=(instruction&15)<<18|self.program[pc+1]<<2
        parameter=instruction>>4&255;result.sound_param[slot]=parameter;pc+=1
        following=self.program[pc+1] if pc+1<len(self.program) else 0
        if following!=2:
            if slot<2:end=2 if following>>12==15 else 1
            elif slot==4:end-=1
            if slot in (2,3):result.sound_param[slot]=_signed(parameter,8)
        return pc,end

    def run(self,sequence,index,*,start=0,control=0,reduce=None,worddata=None,klatt=False):
        item=sequence[index];phone=item.ph
        result=Result();result.pd_param[10]=phone.std_length;result.pd_param[9]=phone.length_mod
        if worddata is not None and item.sourceix:
            worddata['prev_vowel']=None
        if not phone.program:
            return result
        pc=phone.program;returns=[];end=0;steps=0
        while end!=1:
            steps+=1
            if steps>200000 or not 0<=pc<len(self.program):
                raise ValueError('Invalid or cyclic eSpeak phoneme program')
            instruction=self.program[pc];kind=instruction>>12;sub=instruction>>8&15;data=instruction&255
            if kind==0:
                pc,end=self._simple_instruction(sequence,index,pc,control,end,result)
            elif kind==1:
                if reduce is not None and sub<8 and self._stress(sequence,index,sub&7,1,reduce):
                    result.pd_param[1]=data;end=1
            elif kind in (2,3):
                pc=self._conditional_instruction(sequence,index,start,pc,control,reduce,worddata,klatt)
            elif kind==6:
                pc=self._vowel_instruction(sequence,index,pc,result)
            elif kind==9:
                address=(instruction&15)<<16|self.program[pc+1];pc+=1
                if sub==1 and len(returns)<10:
                    returns.append(pc);pc=address-1
                elif sub==2:result.pitch_env=address
                elif sub==3:result.amp_env=address
            elif kind==10:
                offset=0 if sub==1 else 2
                result.vowel_transition[offset]=(instruction&255)<<16|self.program[pc+1]
                result.vowel_transition[offset+1]=_signed(self.program[pc+2]<<16|self.program[pc+3],32)
                pc+=3
            elif kind>=11:
                pc,end=self._sound_instruction(pc,end,result)
            else:
                raise ValueError('Invalid eSpeak phoneme opcode')
            if end==1 and returns:
                end=0;pc=returns.pop()
            pc+=1
        if worddata is not None and item.type==2:
            worddata['prev_vowel']=deepcopy(item)
        item.std_length=result.pd_param[10]&255
        slot=0 if result.sound_addr[0] else 1
        item.phontab_addr=result.sound_addr[slot];item.sound_param=result.sound_param[slot]
        return result
