"""Original mnemonic/internal-phoneme conversion for eSpeak word controllers."""
from dataclasses import dataclass
from standalone.local_piper_phoneme_unicode import utf8_in


@dataclass(frozen=True)
class Encoded:
    phonemes:bytes
    cursor:int
    bad_phoneme:int=0


def encode_phonemes(tr,text,cursor=0):
    text=bytes(text);output=bytearray();table=tr.phoneme_tab
    def byte(index):return text[index] if index<len(text) else 0
    while byte(cursor) in (9,10,11,12,13,32):cursor+=1
    while byte(cursor) not in (0,9,10,11,12,13,32):
        if byte(cursor)==124 and byte(cursor+1)!=124:cursor+=1;continue
        maximum=-1;selected=0
        for index in range(1,tr.n_phoneme_tab):
            phone=table[index]
            if phone is None or phone.type==15:continue
            count=0;mnemonic=phone.mnemonic
            while byte(cursor+count)>32 and count<4 and byte(cursor+count)==(mnemonic>>(count*8)&255):count+=1
            if count>maximum and (count==4 or mnemonic>>(count*8)&255==0):maximum=count;selected=phone.code
        if selected==0:return Encoded(bytes(output),cursor+1,utf8_in(text,cursor)[0])
        cursor+=max(1,maximum);output.append(selected)
        if selected==21:
            start=len(output)
            while byte(cursor) not in (0,9,10,11,12,13,32):
                value=byte(cursor);cursor+=1;output.append(value+32 if 65<=value<=90 else value)
            if not byte(cursor):
                if output[start:]==b'en':del output[start:];return Encoded(bytes(output),cursor)
            else:output.append(124)
    return Encoded(bytes(output),cursor)


def decode_phonemes(tr,phonemes):
    phonemes=bytes(phonemes).split(b'\0',1)[0];output=bytearray();cursor=0
    while cursor<len(phonemes):
        code=phonemes[cursor];cursor+=1
        if code==255:continue
        phone=tr.phoneme_tab[code]
        if phone is None:continue
        if phone.type==1 and phone.std_length<=4 and phone.program==0:
            if phone.std_length>1:output.append(b'==,,\'*  '[phone.std_length])
        else:
            output.extend(phone.mnemonic.to_bytes(4,'little').split(b'\0',1)[0])
            if code==21:
                while cursor<len(phonemes) and (65<=phonemes[cursor]<=90 or 97<=phonemes[cursor]<=122):
                    output.append(phonemes[cursor]);cursor+=1
    return bytes(output)
