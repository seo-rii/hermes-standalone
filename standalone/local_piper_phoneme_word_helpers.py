"""Original eSpeak word morphology, spelling checks and stress adjustments."""
from standalone.local_piper_phoneme_letters import is_letter
from standalone.local_piper_phoneme_rules import byte
from standalone.local_piper_phoneme_stress import vowel_stress,STRESS_PHONEMES,NONSYLLABIC
from standalone.local_piper_phoneme_unicode import utf8_in,utf8_out,is_alpha,is_walpha


def remove_ending(tr,buffer,start,end_type,copy=True):
    end=start
    while byte(buffer,end)!=32:
        if byte(buffer,end)==69:buffer[end]=101
        end+=1
        if end>=len(buffer):raise ValueError('eSpeak suffix input must end with a space')
    original=bytes(buffer[start:start+min(end-start,159)]) if copy else None
    length=end_type&63;position=end
    for _ in range(length):
        position-=1
        while position>=start and byte(buffer,position)&0xc0==0x80:position-=1;length+=1
    ending=bytes(buffer[position:position+min(length,49)])
    buffer[position:position+len(ending)]=b' '*len(ending)
    last=position-1;flags=(end_type&0xfff0)|4
    if end_type&0x200 and byte(buffer,last)==105:buffer[last]=121
    if end_type&0x100:
        if tr.translator_name==0x6e6c:
            if not byte(buffer,last)&128 and not byte(buffer,last-1)&128 and is_letter(tr,byte(buffer,last-1),7) and is_letter(tr,byte(buffer,last),2) and not is_letter(tr,byte(buffer,last-2),7):
                buffer[last+1]=byte(buffer,last);buffer[last]=byte(buffer,last-1);buffer[last+2]=32
        elif tr.translator_name==0x656e:
            stem=bytes(buffer[start:last+1])
            if is_letter(tr,byte(buffer,last-1),7) and is_letter(tr,byte(buffer,last),1):
                if not stem.endswith(b'ion'):flags|=0x10
            elif any(stem.endswith(suffix) for suffix in (b'c',b'rs',b'ir',b'ur',b'ath',b'ns',b'u',b'spong',b'rang',b'larg')):flags|=0x10
        elif tr.langopts.suffix_add_e:flags|=0x10
        if flags&0x10:
            encoded=utf8_out(tr.langopts.suffix_add_e);buffer[last+1:last+1+len(encoded)]=encoded
    if end_type&0x800 and tr.expect_verb==0:tr.expect_verb=1
    if ending in (b's',b'es'):flags|=8
    if ending[:1]==b"'":flags&=~4
    return flags,original


def change_word_stress(tr,phones,new_stress):
    phones,levels,count,_,maximum=vowel_stress(tr,phones,0,0)
    if new_stress>=4:
        for index in range(1,count):
            if levels[index]>=maximum:levels[index]=new_stress;break
    else:
        for index in range(1,count):
            if levels[index]>new_stress:levels[index]=new_stress
    output=bytearray();index=1
    for code in phones:
        phone=tr.phoneme_tab[code]
        if phone.type==2 and not phone.phflags&NONSYLLABIC:
            if levels[index]==0 or levels[index]>1:output.append(STRESS_PHONEMES[levels[index]])
            index+=1
        output.append(code)
    return bytes(output)


def apply_special_attribute(tr,phones,flags):
    phones=bytearray(phones)
    if tr.langopts.param[11]&2:
        for index in range(len(phones)-1):
            if phones[index]!=6:continue
            pairs=((b'E',b'e'),(b'O',b'o')) if flags&0x10000 else ((b'e',b'E'),(b'o',b'O'))
            for before,after in pairs:
                if phones[index+1]==tr.session.selection.lookup_phone(before):phones[index+1]=tr.session.selection.lookup_phone(after)
            break
    return bytes(phones)


def add_plural_suffix(tr,flags,last_char,phones):
    if not flags&4:return phones
    last_char=(last_char+128)%256-128
    word=b'ss' if last_char==102 else b'iz' if last_char in (104,115,120) else b'zz'
    buffer=bytearray(b'\0 '+word+b'\0')
    return tr.rules.translate(buffer,2,initial=phones).phonemes


def check_dotted_abbrev(tr,buffer,start):
    cursor=start;output=bytearray();count=0
    while True:
        code,size=utf8_in(buffer,cursor);matched=0
        if byte(buffer,cursor+size)==32 and is_alpha(code):
            if byte(buffer,cursor+size+1)==46:
                if byte(buffer,cursor+size+2)==32:matched=1
                elif byte(buffer,cursor+size+2)==39 and byte(buffer,cursor+size+3)==115:size+=2;matched=2
            elif count>0:matched=2
        if matched==0:break
        output.extend(buffer[cursor:cursor+size]);count+=1
        if matched==2:cursor+=size;break
        cursor+=size+3
    if count>1:
        buffer[start:start+len(output)]=output
        buffer[start+len(output):cursor]=b' '*(cursor-start-len(output))
        tr.session.state.dictionary_skipwords=(count-1)*2
    return count


def non_ascii_number(letter):
    for base in (0x660,0x6f0,0x966,0x9e6,0xa66,0xae6,0xb66,0xbe6,0xc66,0xce6,0xd66,0xe50,0xed0,0xf20,0x1040,0x1090):
        if letter<base:break
        if letter<base+10:return letter-base+48
    return -1


def unpronounceable(tr,buffer,cursor,posn):
    first,_=utf8_in(buffer,cursor)
    if tr.letter_bits_offset>0 and first<0x241:return False
    alphabet=tr.alphabet_from_char(first)
    if alphabet is not None and alphabet.offset!=tr.letter_bits_offset:return False
    option=tr.langopts.param[4]
    if option==1 or byte(buffer,cursor) in (0,32,39):return False
    position=cursor;count=0;vowel_position=9;initial=0
    while True:
        code,size=utf8_in(buffer,position);position+=size
        if code in (0,32):break
        if code==39 and (count>1 or posn>0):break
        if count==0:initial=code
        if code!=39 or option!=3:count+=1
        if is_letter(tr,code,7):vowel_position=count;break
        if code!=39 and not is_walpha(code):return False
    if vowel_position>2 and option==2:
        previous=byte(buffer,cursor-1);buffer[cursor-1]=32
        result=tr.rules.translate(buffer,cursor,word_flags=0x80000000)
        buffer[cursor-1]=previous
        return result.end_type==0 or bool(result.end_type&0x8000)
    if initial==option:vowel_position-=1
    return vowel_position>tr.langopts.max_initial_consonants+1
