"""Original TranslateClause word/context controller, owned mutable state.

Adapted from eSpeakNG translate.c, Copyright 2005-2014 Jonathan Duddington
and Copyright 2015-2018 Reece H.Dunn, GPL-3.0-or-later. Original source and
independent native observations are recorded in local-piper-phoneme-text-
provenance.json; complete license notices are in licenses/. This qualified
implementation executes owned mutable state and standard-library algorithms.
"""
from dataclasses import dataclass
from types import SimpleNamespace
from standalone import local_piper_phoneme_unicode as u
from standalone.local_piper_phoneme_clause import ClauseReader
from .local_piper_phoneme_text_characters import Characters
from .local_piper_phoneme_text_tables import *
from .local_piper_phoneme_text_tokenizer import Tokenizer, WordTab, bytec

@dataclass
class FirstItem:
    synthflags: int = 0
    phcode: int = 0
    stresslevel: int = 0
    sourceix: int = 0
    wordstress: int = 0
    tone_ph: int = 0

@dataclass
class ClauseResult:
    tone: int
    terminator: int
    voice_change: object
    words: object
    buffer: bytearray
    phoneme_list: object

def initialize_state(state):
    values = dict(embedded_ix=0, embedded_read=0, pre_pause=0, any_stressed_words=False, count_words=0, count_sentences=0, clause_start_char=0, clause_start_word=0, max_clause_pause=0, skip_sentences=0, skip_words=0, option_sayas2=0, count_sayas_digits=0, word_emphasis=0, option_phoneme_input=0, n_ph_list2=0, dictionary_skipwords=0, option_endpause=0, n_phoneme_list=0, embedded_flag=0, option_sayas=0, option_emphasis=0, option_capitals=0, new_sentence=False, current_alphabet=None, word_phonemes=b'', phoneme_list=[], phoneme_buffer=None, ph_list2=[FirstItem() for _ in range(N_PHONEME_LIST)], embedded_list=[0] * N_EMBEDDED_LIST)
    for key, value in values.items():
        if not hasattr(state, key):
            setattr(state, key, value)
    return state

class ClauseController:

    def __init__(self, state=None, *, reader=None, word2=None, make_phoneme_list=None):
        self.state = initialize_state(SimpleNamespace() if state is None else state)
        self.reader = ClauseReader() if reader is None else reader
        self.characters = Characters()
        self.tokenizer = Tokenizer(self.state, self.characters)
        self.word2 = word2
        self.make_phoneme_list = make_phoneme_list

    @classmethod
    def from_session(cls, session, reader=None):
        from standalone.local_piper_phoneme_list import make_phoneme_list
        from standalone.local_piper_phoneme_vm import Item
        from .local_piper_phoneme_words import WordBuilder
        controller = cls(session.state, reader=reader)
        if controller.state.phoneme_buffer is None:
            controller.state.phoneme_buffer = [Item(phcode=0) for _ in range(N_PHONEME_LIST + 1)]

        def raw_word(tr, buffer, cursor, words, index, *, word_out):
            return tr.translate_word_buffer(buffer, cursor, words, index, word_out=word_out)
        controller.word2 = WordBuilder(controller, raw_word).translate

        def build_list(tr, items, pause, sentence):
            return make_phoneme_list(tr, items, pause, sentence, selection=session.selection, phoneme_data=session.data, replacements=session.voice.replace_phonemes, option_wordgap=getattr(session.state, 'option_wordgap', 0), klatt=session.voice.klatt, output_buffer=controller.state.phoneme_buffer)
        controller.make_phoneme_list = build_list
        return controller

    def _read(self, tr):
        state = self.state
        reader = self.reader
        state.embedded_ix = state.embedded_read = state.pre_pause = 0
        state.any_stressed_words = False
        state.clause_start_char = max(reader.count_characters, 0)
        state.clause_start_word = state.count_words + 1
        clause = reader.read(tr)
        for key in ('count_characters', 'clause_start_char', 'option_phoneme_input', 'option_capitals', 'option_punctuation', 'skipping_text', 'skip_characters', 'end_character_position', 'ungot_char', 'ungot_char2'):
            setattr(state, key, getattr(reader, key))
        charix = list(clause.charix)
        charix[clause.charix_top + 1:clause.charix_top + 4] = [0, 32767, 0]
        term = clause.terminator
        tone = clause.tone or (term & CLAUSE_INTONATION_TYPE) >> 12
        pause = (term & CLAUSE_PAUSE) * 10
        if term & CLAUSE_PAUSE_LONG:
            pause *= 32
        source = bytearray(clause.buffer)
        pointer = 0
        while source[pointer] and u.isspace2(bytec(source[pointer])):
            pointer += 1
        if source[pointer] == 0:
            pause = max(0, pause - state.max_clause_pause)
            if state.new_sentence:
                term |= CLAUSE_TYPE_SENTENCE
            state.max_clause_pause += pause
            sentence = False
        else:
            state.max_clause_pause = pause
            sentence = state.new_sentence
        tr.clause_terminator = term
        if sentence:
            state.count_sentences += 1
            if state.skip_sentences > 0:
                state.skip_sentences -= 1
                if state.skip_sentences == 0:
                    state.skipping_text = False
        state.ph_list2[0] = FirstItem(phcode=phonPAUSE_SHORT)
        state.n_ph_list2 = 1
        for field in ('prev_last_stress', 'prepause_timeout', 'expect_verb', 'expect_noun', 'expect_past', 'expect_verb_s', 'phonemes_repeat_count', 'end_stressed_vowel'):
            setattr(tr, field, 0)
        tr.prev_dict_flags[:] = [0, 0]
        result = self.tokenizer.tokenize(tr, source, charix, clause.charix_top, term)
        tr.clause_end = result.clause_end
        state.clause_buffer = result.buffer
        return (result, term, tone, pause, sentence, clause.voice_change)

    def word_embedded_command(self, tr):
        state = self.state
        while True:
            command = state.embedded_list[state.embedded_read]
            state.embedded_read += 1
            value = command >> 8
            code = command & 31
            if code == EMBED_Y:
                state.option_sayas = value
                tr.option_sayas = value
            elif code == EMBED_F:
                state.option_emphasis = value
            elif code == EMBED_B:
                state.pre_pause = 0 if value == 0 else state.pre_pause + value
            if command & 128 or state.embedded_read >= state.embedded_ix:
                break

    def _word2(self, tr, buffer, cursor, words, index, pause):
        if self.word2 is None:
            raise RuntimeError('The qualified original word translator is required')
        return self.word2(tr, buffer, cursor, words, index, pause)

    @staticmethod
    def _merge_thousands(tr, s, index):
        buffer = s.buffer
        cursor = s.words[index].start
        if not u.is_wdigit(bytec(buffer[cursor])) or tr.langopts.break_numbers == BREAK_THOUSANDS:
            return index
        output = bytearray()
        pointer = cursor
        while len(output) < 130:
            if u.is_wdigit(bytec(buffer[pointer])):
                output.append(buffer[pointer])
                pointer += 1
            elif buffer[pointer] == tr.langopts.thousands_sep and buffer[pointer + 1] == 32 and u.is_wdigit(bytec(buffer[pointer + 2])) and (buffer[pointer + 3] != 32) and (buffer[pointer + 4] != 32):
                pointer += 2
                index += 1
            else:
                buffer[cursor:pointer] = b' ' * (pointer - cursor)
                buffer[cursor:cursor + len(output)] = output
                break
        return index

    def _split_number(self, tr, s, index, cursor, n_digits):
        source = s.buffer
        number = bytearray(150)
        number[:3] = b'   '
        pointer = cursor
        end = 3
        remaining = n_digits
        count = 0
        words = [WordTab() for _ in range(N_CLAUSE_WORDS)]
        if n_digits > tr.langopts.max_digits or source[cursor] == 48:
            s.words[index].flags |= FLAG_INDIVIDUAL_DIGITS
        while end < 130 and count < N_CLAUSE_WORDS - 2:
            c = source[pointer]
            pointer += 1
            if not 48 <= c <= 57 and c != tr.langopts.decimal_sep:
                break
            number[end] = c
            end += 1
            remaining -= 1
            if remaining > 0 and tr.langopts.break_numbers & 1 << (remaining & 31):
                words[count] = WordTab(**vars(s.words[index]))
                count += 1
                if tr.langopts.thousands_sep != 32:
                    number[end] = tr.langopts.thousands_sep & 255
                    end += 1
                number[end] = 32
                end += 1
                if not s.words[index].flags & FLAG_INDIVIDUAL_DIGITS:
                    if tr.langopts.break_numbers & 1 << (remaining - 1 & 31):
                        number[end:end + 2] = b'00'
                        end += 2
                    if tr.langopts.break_numbers & 1 << (remaining - 2 & 31):
                        number[end] = 48
                        end += 1
        pointer -= 1
        words[count:count + 2] = [WordTab(**vars(w)) for w in s.words[index:index + 2]]
        for j in range(1, count + 1):
            words[j].flags &= ~(FLAG_MULTIPLE_SPACES | FLAG_EMBEDDED)
        suffix = bytes(source[pointer:pointer + 16]).split(b'\x00', 1)[0]
        number[end:end + 16] = suffix.ljust(16, b'\x00')
        number[end + 16] = 0
        cursor = 3
        count = 0
        flags = 0
        while cursor < end and count < N_CLAUSE_WORDS:
            flags = self._word2(tr, number, cursor, words, count, s.words[index].pre_pause)
            count += 1
            while cursor < end:
                c = number[cursor]
                cursor += 1
                if c == 32:
                    break
            s.words[index].pre_pause = 0
        return flags

    def _spell_word(self, tr, s, index, cursor):
        while s.buffer[cursor] != 32:
            number = bytearray(150)
            number[1:10] = b' ' * 9
            _, size = u.utf8_in(s.buffer, cursor)
            number[3:3 + size] = s.buffer[cursor:cursor + size]
            self._word2(tr, number, 3, s.words, index, 0)
            cursor += size

    def _normal_word(self, tr, s, index, cursor, term, tone, pause):
        state = self.state
        state.pre_pause = 0
        flags = self._word2(tr, s.buffer, cursor, s.words, index, s.words[index].pre_pause)
        if state.pre_pause > s.words[index + 1].pre_pause:
            s.words[index + 1].pre_pause = state.pre_pause & 255
            state.pre_pause = 0
        if flags & FLAG_SPELLWORD:
            self._spell_word(tr, s, index, cursor)
        if flags & (FLAG_ALLOW_DOT | FLAG_NEEDS_DOT) and index == s.word_count - 1 - state.dictionary_skipwords and term & CLAUSE_DOT_AFTER_LAST_WORD:
            pause = 10
            tone = 4
        return (flags, tone, pause)

    def _words(self, tr, s, term, tone, pause):
        state = self.state
        index = 0
        while index < s.word_count and state.n_ph_list2 < N_PHONEME_LIST - 9:
            state.count_words += 1
            if state.skip_words > 0:
                state.skip_words -= 1
                if state.skip_words == 0:
                    state.skipping_text = False
            if state.skipping_text:
                index += 1
                continue
            state.current_alphabet = None
            cursor = s.words[index].start
            index = self._merge_thousands(tr, s, index)
            count = 0
            while u.is_wdigit(bytec(s.buffer[cursor + count])):
                count += 1
            if 4 < count <= 32:
                flags = self._split_number(tr, s, index, cursor, count)
            else:
                flags, tone, pause = self._normal_word(tr, s, index, cursor, term, tone, pause)
            if flags & FLAG_SKIPWORDS:
                while state.dictionary_skipwords > 0:
                    s.words[index + state.dictionary_skipwords].flags |= FLAG_DELETE_WORD
                    state.dictionary_skipwords -= 1
            index += 1
        return (tone, pause)

    def _finish(self, tr, s, term, pause, sentence):
        state = self.state
        if state.embedded_read < state.embedded_ix:
            self.word_embedded_command(tr)
        for offset in range(2):
            item = state.ph_list2[state.n_ph_list2 + offset]
            item.phcode = phonPAUSE
            item.stresslevel = 0
            item.sourceix = s.source_index & 65535
            item.synthflags = 0
        state.n_ph_list2 += 2
        if self.reader.eof() and (s.word_count == 0 or state.option_endpause == 0):
            pause = 10
        if self.make_phoneme_list is None:
            raise RuntimeError('The qualified original phoneme-list constructor is required')
        items = state.ph_list2[:state.n_ph_list2]
        state.phoneme_list = self.make_phoneme_list(tr, items, pause, sentence)
        state.ph_list2[:len(items)] = items
        state.n_ph_list2 = len(items)
        state.n_phoneme_list = len(state.phoneme_list)
        if state.phoneme_buffer is not None:
            state.phoneme_buffer[N_PHONEME_LIST].ph = None
            state.phoneme_buffer[N_PHONEME_LIST].sourceix = 1
        if s.embedded_count:
            state.phoneme_list[-2].synthflags = SFLAG_EMBEDDED
            state.embedded_list[state.embedded_ix - 1] |= 128
            state.embedded_list[state.embedded_ix] = 128
        state.new_sentence = bool(term & CLAUSE_TYPE_SENTENCE)
        self.reader.skipping_text = state.skipping_text

    def translate_clause(self, tr):
        result, term, tone, pause, sentence, change = self._read(tr)
        tone, pause = self._words(tr, result, term, tone, pause)
        self._finish(tr, result, term, pause, sentence)
        return ClauseResult(tone, term, change if term & CLAUSE_TYPE_VOICE_CHANGE else None, result.words[:result.word_count + 1], result.buffer, self.state.phoneme_list)
