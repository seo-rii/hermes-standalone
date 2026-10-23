"""Original eSpeak phoneme linking, substitutions and clause buffer behavior.

Source: espeak-ng 724808c5a83f9ef95fdd0db886ba7ba537ff224a phonemelist.c.
Copyright (C) 2005-2014 Jonathan Duddington; 2015-2016 Reece H. Dunn.
GPL-3.0-or-later; see licenses/local-piper-phoneme-list-notices.txt.
"""
from copy import copy
from dataclasses import dataclass
from .local_piper_phoneme_vm import Interpreter,Item


@dataclass
class FirstItem:
    synthflags:int=0
    phcode:int=9
    stresslevel:int=0
    sourceix:int=0
    wordstress:int=0
    tone_ph:int=0


_FIRST=('synthflags','phcode','stresslevel','sourceix','wordstress','tone_ph')
_PAUSES=(0,23,10,9,24,19,24,24)


def _voicing(item,ph,regression,voicing):
    kind=ph.type
    stop=bool(regression&2 and ph.mnemonic&255 in (118,82))
    if stop and regression&0x10:voicing=0
    if kind in (4,6):
        if voicing==0 and regression&15:voicing=1
        elif voicing==2 and ph.end_type:item.phcode=ph.end_type
    elif kind in (5,7):
        if voicing==0 and regression&15:voicing=2
        elif voicing==1 and ph.end_type:item.phcode=ph.end_type
    elif not regression&8 or kind in (0,2):voicing=0
    return 0 if stop else voicing


def _regressive(tr,items,selection):
    regression=tr.langopts.param[3];voicing=0
    for j in range(len(items)-1,-1,-1):
        item=items[j]
        if item.phcode==21:
            k=j-1
            while k>=0 and items[k].phcode!=21:k-=1
            selection.select(items[k].tone_ph if k>=0 else tr.phoneme_tab_ix)
        ph=selection.phones[item.phcode]
        if ph is None:continue
        if item.synthflags&0x20:
            voicing=1 if regression&0x100 else 0
            continue
        voicing=_voicing(item,ph,regression,voicing)
        if item.sourceix:
            if regression&4:voicing=0
            if regression&0x100 and voicing==0:voicing=1


def _replace(item,selection,replacements,word_end):
    for old,new,flags in replacements:
        if item.phcode!=old:continue
        if flags&1 and not word_end:continue
        if flags&2 and item.stresslevel&7>3:continue
        if flags&4 and item.sourceix==0:continue
        item.phcode=new
        if item.stresslevel>1:
            ph=selection.phones[new]
            if ph is None:raise ValueError('Invalid stressed eSpeak replacement')
            if ph.phflags&2:item.stresslevel=0
        break


def _substitute(items,selection,replacements):
    output=[];deleted_source=-1;next_phone=None
    for i,item in enumerate(items):
        if len(output)>=1000:break
        if deleted_source!=-1:item.sourceix=deleted_source;deleted_source=-1
        if item.phcode==21:selection.select(item.tone_ph)
        if not item.synthflags&0x20:
            if i<len(items)-1:next_phone=selection.phones[items[i+1].phcode]
            word_end=i==len(items)-1 or items[i+1].sourceix or(next_phone is not None and next_phone.type==0)
            _replace(item,selection,replacements,word_end)
            if item.phcode==0:deleted_source=item.sourceix;continue
        ph=selection.phones[item.phcode]
        if ph is None:raise ValueError('Undefined eSpeak phoneme list entry')
        output.append(Item(**{name:getattr(item,name)for name in _FIRST},ph=ph,type=ph.type))
    return output


def _promote_last_stress(items):
    maximum=0;j=len(items)-3
    while j>=0:
        maximum=max(maximum,items[j].stresslevel&127)
        if items[j].sourceix:break
        j-=1
    if maximum<4:
        j-=1
        while j>=0:
            if items[j].synthflags&0x40:items[j].stresslevel=4;break
            if items[j].stresslevel>=4:break
            j-=1


def _compact_switches(tr,items):
    current=tr.phoneme_tab_ix;deleted=0;deleted_source=-1
    # Keep the original backing-array read/write order during compaction.
    for j in range(len(items)):
        if current!=tr.phoneme_tab_ix:items[j].synthflags|=0x20
        if deleted:
            items[j-deleted]=copy(items[j])
            if deleted_source!=-1:items[j-deleted].sourceix=deleted_source;deleted_source=-1
        if items[j].phcode==21:
            if not items[j].synthflags&2 and(items[j].tone_ph==current or items[j+1].phcode==21 or
                       (items[j+1].phcode==9 and items[j+2].phcode==21)):
                if deleted_source==-1 and items[j].sourceix:deleted_source=items[j].sourceix
                deleted+=1
            else:current=items[j].tone_ph
    if deleted:del items[len(items)-deleted:]
    return current


def _set_word_stress(stage,logical):
    j=0;word_stress=0
    while j<logical:
        if stage[j].sourceix:
            word_stress=0;following=j
            while following<logical:
                word_stress=max(word_stress,stage[following].stresslevel)
                following+=1
                if stage[following].sourceix:break
            for k in range(j,following):stage[k].wordstress=word_stress
            j=following
        else:j+=1
    return word_stress


def _boundary_pause(insert,ph,next_phone,following,pause):
    if pause and ph.type!=0:
        if ph.type!=2 and pause&0x200:insert=11
        if next_phone.type==2:
            if pause&12:insert=11 if pause&12==12 else 23
            if ph.type==2 and pause&3:insert=10 if pause&3==2 else 23
            if following.stresslevel>=4 and pause&0x100:insert=10
    return insert


class _Linker:
    """One original linking pass, including its mutable pointer/index state."""

    def __init__(self,tr,stage,selection,data,buffer,option_wordgap,klatt,start_sentence,word_stress):
        self.tr=tr;self.stage=stage;self.selection=selection;self.data=data;self.buffer=buffer
        self.option_wordgap=option_wordgap;self.klatt=klatt;self.start_sentence=start_sentence;self.word_stress=word_stress
        self.output=[];self.insert=0;self.inserted_pointer=None;self.word_start=1
        self.unstressed=0;self.clause_start=True;self.worddata={}

    def _interpret(self,index):
        vm=Interpreter(self.data,self.selection.current,phones=self.selection.phones)
        return vm.run(self.stage,index,control=0x100,reduce=self.tr.langopts.param[8],worddata=self.worddata,klatt=self.klatt)

    def _reinterpret(self,index,previous):
        item=self.stage[index]
        if item.ph.type==2:
            item.synthflags|=4
            if previous.type!=2:item.stresslevel=0
        else:item.synthflags&=~4
        return self._interpret(index)

    def _current(self,j):
        item=self.stage[j]
        if self.insert:
            next_phone=self.selection.phones[item.phcode];j-=1
            if j>0:
                if self.word_start>0:k=self.word_start;self.word_start-=1
                else:k=2
                while k<=j:self.stage[k-1]=copy(self.stage[k]);k+=1
            self.stage[j]=item=Item(phcode=self.insert,ph=self.selection.phones[self.insert])
            self.inserted_pointer=j;ph=item.ph;self.insert=0;inserted=True
        else:
            if item.sourceix:self.word_start=j
            item.ph=ph=self.selection.phones[item.phcode]
            if item.phcode==21:self.selection.select(item.tone_ph)
            next_phone=self.selection.phones[self.stage[j+1].phcode];self.stage[j+1].ph=next_phone
            inserted=False
        return j,item,ph,next_phone,inserted

    def _alternatives(self,j,item,ph,next_phone,inserted,data):
        deleted=False;alternative=data.pd_param[2]
        if alternative>0:
            next_phone=self.selection.phones[alternative]
            self.stage[j+1].ph=next_phone;self.stage[j+1].phcode=alternative;self.stage[j+1].type=next_phone.type
        alternative=data.pd_param[3]
        if alternative>0 and not inserted:
            previous=ph;self.insert=item.phcode;item.ph=ph=self.selection.phones[alternative];item.phcode=alternative
            data=self._reinterpret(j,previous)
        alternative=data.pd_param[1]
        if alternative>0:
            previous=ph;item.ph=ph=self.selection.phones[alternative];item.phcode=alternative
            if alternative==1:deleted=True
            else:data=self._reinterpret(j,previous)
        return ph,next_phone,data,deleted

    def _unstress_pair(self,j,item):
        k=j+1
        while self.stage[k].type!=0:
            following=self.stage[k]
            if following.type==2:
                if following.stresslevel<=1:
                    if item.wordstress<4:item.stresslevel=0
                    if following.wordstress<4:following.stresslevel=0
                break
            k+=1

    def _reduce(self,j,item,ph,deleted):
        if ph.type!=2 or deleted:return
        if item.stresslevel<=1:
            self.unstressed+=1
            if self.tr.langopts.stress_flags&8:self._unstress_pair(j,item)
            elif self.unstressed>1 and self.unstressed&1==0:
                if self.tr.langopts.stress_flags&2 or(self.word_stress>3 and self.stage[j+1].sourceix):self.unstressed=1
                else:item.stresslevel=0
        else:self.unstressed=0

    def _next_insert(self,j,next_phone,ph,data):
        following=self.stage[j+1]
        if following.synthflags&8 and j>0 and next_phone.type in (6,7,8,3,0):
            self.insert=next_phone.code;following.synthflags^=8
        if following.sourceix:
            self.insert=_boundary_pause(self.insert,ph,next_phone,following,self.tr.langopts.vowel_pause)
            if j!=self.inserted_pointer and self.output:
                gap=self.tr.langopts.word_gap&7
                if gap and(gap>1 or self.insert not in (10,11)):self.insert=_PAUSES[gap]
                if self.option_wordgap>0:self.insert=24
        self.stage[j+2].ph=self.selection.phones[self.stage[j+2].phcode]
        if self.insert==0 and data.pd_param[4]!=0:self.insert=data.pd_param[4]

    def _emit(self,j,item,ph,data):
        # Interpreter-owned fields retain the caller's existing slot values.
        target=self.buffer[len(self.output)]
        target.ph=ph;target.type=ph.type;target.env=0;target.synthflags=item.synthflags
        target.stresslevel=item.stresslevel&15;target.wordstress=item.wordstress
        target.tone_ph=item.tone_ph;target.sourceix=0;target.phcode=ph.code;target.newword=0
        if item.sourceix:
            target.sourceix=item.sourceix;target.newword=1
            if self.start_sentence:target.newword|=4;self.start_sentence=False
            if self.clause_start:target.newword|=8;self.clause_start=False
        target.length=(data.pd_param[10]*2)&0xffffffff
        if ph.code==24 and self.option_wordgap>0 and self.stage[j+1].sourceix:
            target.ph=self.selection.phones[10];target.length=(self.option_wordgap*14)&0xffffffff
        if ph.type in (2,3,8,5,7)or ph.phflags&0x2000000:target.length=128;target.env=0
        target.prepause=0;target.amp=20;target.pitch1=255;target.pitch2=255
        self.output.append(target)

    def run(self,logical):
        j=0
        while self.insert or(j<logical and len(self.output)<997):
            j,item,ph,next_phone,inserted=self._current(j)
            if ph is None:j+=1;continue
            data=self._interpret(j)
            ph,next_phone,data,deleted=self._alternatives(j,item,ph,next_phone,inserted,data)
            self._reduce(j,item,ph,deleted)
            self._next_insert(j,next_phone,ph,data)
            if not deleted:self._emit(j,item,ph,data)
            j+=1
        return self.output


def _terminate(output,buffer,selection,post_pause,end_source):
    for marker,code,length,source in ((2,9,post_pause&0xffffffff,end_source),(0,10,0,0)):
        target=buffer[len(output)]
        target.newword=marker;target.phcode=9;target.type=0;target.length=length
        target.sourceix=source;target.synthflags=0;target.prepause=0;target.ph=selection.phones[code]
        output.append(target)


def make_phoneme_list(tr,items,post_pause,start_sentence,*,selection,phoneme_data,
                      replacements=(),option_wordgap=0,klatt=False,output_buffer=None):
    """Mutate controller input and return the active prefix of owned output slots."""
    if not 3<=len(items)<=1000:raise ValueError('Invalid eSpeak first-stage phoneme count')
    if output_buffer is None:output_buffer=[Item(phcode=0)for _ in range(1001)]
    if len(output_buffer)<1001:raise ValueError('eSpeak output buffer needs 1001 owned slots')
    end_source=items[-1].sourceix
    _promote_last_stress(items)
    selection.select(_compact_switches(tr,items))
    if tr.langopts.param[3]:_regressive(tr,items,selection)
    selection.select(tr.phoneme_tab_ix)
    stage=_substitute(items,selection,replacements);logical=len(stage)-2
    if logical<1:raise ValueError('Missing eSpeak phoneme terminators')
    word_stress=_set_word_stress(stage,logical)
    stage[0].ph=selection.phones[9]
    selection.select(tr.phoneme_tab_ix)
    output=_Linker(tr,stage,selection,phoneme_data,output_buffer,option_wordgap,klatt,start_sentence,word_stress).run(logical)
    _terminate(output,output_buffer,selection,post_pause,end_source)
    selection.select(tr.phoneme_tab_ix)
    return output
