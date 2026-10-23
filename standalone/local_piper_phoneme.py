"""Original Piper eSpeak text-to-phonemes boundary using owned Python algorithms."""
from pathlib import Path
import re
from threading import RLock
import unicodedata

from standalone.local_piper_phoneme_session import Session
from standalone.local_piper_phoneme_vm import Interpreter
from standalone.local_piper_phoneme_render import render
from standalone.local_piper_phoneme_text import ClauseController

ESPEAK_DATA_DIR=Path(__file__).with_name('assets')/'piper/espeak-ng-data'
TERMINATORS={0x80028:'.',0x82028:'?',0x8302d:'!',0x41014:',',0x4001e:':',0x4101e:';'}


def _merge_known_vowel_clusters(phones,clusters):
    maximum=max(len(cluster) for cluster in clusters);output=[];index=0
    while index<len(phones):
        match=None
        for count in range(min(maximum,len(phones)-index),1,-1):
            candidate=tuple(phones[index:index+count])
            if candidate in clusters:match=candidate;break
        if match is None:output.append(phones[index]);index+=1
        else:output.append(''.join(match));index+=len(match)
    return output


class EspeakPhonemizer:
    def __init__(self,espeak_data_dir=None):
        self.session=Session(ESPEAK_DATA_DIR if espeak_data_dir is None else espeak_data_dir)
        self.controller=ClauseController.from_session(self.session)
        self.lock=RLock()

    def _clauses(self,voice,text):
        if not isinstance(voice,str):raise TypeError('voice must be str')
        if '\0' in voice:raise ValueError('embedded null character')
        try:tr=self.session.select_voice(voice)
        except ValueError as error:raise RuntimeError('Failed to set voice: '+voice) from error
        if not isinstance(text,str):raise TypeError('text must be str')
        if '\0' in text:raise ValueError('embedded null character')
        self.controller.reader.bind_text(text)
        clauses=[]
        while True:
            result=self.controller.translate_clause(tr)
            vm=Interpreter(self.session.data,self.session.selection.current,phones=self.session.selection.phones)
            phones=render(vm,result.phoneme_list,2);terminator=result.terminator&0xfffff
            clauses.append((phones,TERMINATORS.get(terminator,''),bool(terminator&0x80000)))
            if self.controller.reader.eof():break
        return clauses

    def phonemize(self,voice,text,vowel_clusters=None):
        with self.lock:
            clauses=self._clauses(voice,text)
        output=[];sentence=[]
        for phones,terminator,end_of_sentence in clauses:
            phones=re.sub(r'\([^)]+\)','',phones)+terminator
            if terminator in (',',':',';'):phones+=' '
            sentence.extend(unicodedata.normalize('NFD',phones))
            if end_of_sentence:
                if vowel_clusters:sentence=_merge_known_vowel_clusters(sentence,vowel_clusters)
                output.append(sentence);sentence=[]
        if sentence:
            if vowel_clusters:sentence=_merge_known_vowel_clusters(sentence,vowel_clusters)
            output.append(sentence)
        return output
