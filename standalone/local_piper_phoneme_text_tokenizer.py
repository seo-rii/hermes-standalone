"""Original TranslateClause UTF8-to-WORD_TAB state machine.

Adapted from eSpeakNG translate.c, Copyright 2005-2014 Jonathan Duddington
and Copyright 2015-2018 Reece H.Dunn, GPL-3.0-or-later. Original source and
independent native observations are recorded in local-piper-phoneme-text-
provenance.json; complete license notices are in licenses/. This qualified
implementation executes owned mutable state and standard-library algorithms.
"""
from dataclasses import dataclass
from types import SimpleNamespace
from standalone import local_piper_phoneme_unicode as u
from standalone.local_piper_phoneme_clause import is_bracket
from .local_piper_phoneme_text_characters import lower, upper_case_in_word
from .local_piper_phoneme_text_tables import *

@dataclass
class WordTab:
    flags: int = 0
    start: int = 0
    pre_pause: int = 0
    sourceix: int = 0
    length: int = 0

def bytec(value):
    return value if value < 128 else value - 256

def calc_word_length(source_index, charix_top, charix, words, index):
    while source_index < charix_top and charix[source_index] <= 0:
        source_index += 1
    words[index].sourceix = charix[source_index] & 65535
    count = 0
    while charix[source_index] != 0:
        if charix[source_index] != -1:
            count += 1
        source_index += 1
    return count & 255

def _put(buffer, cursor, data):
    if cursor < 0 or cursor + len(data) > len(buffer):
        raise ValueError('eSpeak word buffer write exceeds owned storage')
    buffer[cursor:cursor + len(data)] = data
    return cursor + len(data)

class Tokenizer:

    def __init__(self, state, characters):
        self.state = state
        self.characters = characters

    def embedded_command(self, source, cursor):
        state = self.state
        start = cursor
        sign = 0
        value = -1
        if source[cursor] in (43, 45):
            sign = 64 if source[cursor] == 43 else 96
            cursor += 1
        if 48 <= source[cursor] <= 57:
            start_number = cursor
            while 48 <= source[cursor] <= 57:
                cursor += 1
            value = int(source[start_number:cursor]) & 4294967295
        command = source[cursor]
        cursor += 1
        if state.embedded_ix >= N_EMBEDDED_LIST - 2:
            return (0, start)
        index = b'PSARHTIVYMUBF'.find(bytes((command,))) if command < 128 else -1
        if index < 0:
            return (0, start)
        command = index + 1
        if value == -1:
            value = EMBEDDED_DEFAULT[command]
            sign = 0
        if command == EMBED_Y:
            state.option_sayas2 = value
            state.count_sayas_digits = 0
        if command == EMBED_F:
            state.word_emphasis = FLAG_EMPHASIZED if value >= 3 else 0
        state.embedded_list[state.embedded_ix] = command + sign + (value << 8) & 4294967295
        state.embedded_ix += 1
        return (1, cursor)

    def _next(self, tr, s):
        s.prev_out2 = s.prev_out
        s.prev_out, _ = u.utf8_in(s.buffer, s.ix - 1, True)
        if tr.langopts.tone_numbers and 48 <= s.prev_out <= 57 and u.is_alpha(s.prev_out2):
            s.prev_out = 97
        if s.prev_in_save:
            s.prev_in = s.prev_in_save
            s.prev_in_save = 0
        elif s.source_index > 0:
            s.prev_in, _ = u.utf8_in(s.source, s.source_index - 1, True)
        s.prev_source_index = s.source_index
        if s.char_inserted:
            s.c = s.char_inserted
            s.char_inserted = 0
        else:
            s.cc, size = u.utf8_in(s.source, s.source_index)
            s.source_index += size
            s.c = s.cc
        if s.c == 0:
            s.finished = True
            s.c = s.next_in = 32
            s.next_in_nbytes = 0
        else:
            s.next_in, s.next_in_nbytes = u.utf8_in(s.source, s.source_index)

    def _embedded(self, s):
        if s.c != 1:
            return False
        index = s.source_index - 1
        if s.prev_in != 32:
            s.c = 32
            s.prev_in_save = s.c
            s.source_index -= 1
            return False
        count, s.source_index = self.embedded_command(s.source, s.source_index)
        s.embedded_count += count
        s.prev_in_save = s.prev_in
        s.source[index:s.source_index] = b' ' * (s.source_index - index)
        s.source_index = index
        return True

    def _sayas(self, tr, s):
        state = self.state
        if state.option_sayas2 == SAYAS_KEY and s.c != 32:
            if s.prev_in == 32 and s.next_in == 32:
                state.option_sayas2 = SAYAS_SINGLE_CHARS
            s.c = lower(tr, s.c)
        if s.phoneme_mode:
            s.all_upper_case = FLAG_PHONEMES
            if s.c == 93 and s.next_in == 93:
                s.phoneme_mode = False
                s.source_index += 1
                s.c = 32
            return False
        if state.option_sayas2 & 240 == SAYAS_DIGITS:
            if u.is_wdigit(s.c):
                state.count_sayas_digits += 1
                if state.count_sayas_digits > state.option_sayas2 & 15:
                    s.c = 32
                    s.space_inserted = True
                    state.count_sayas_digits = 0
            else:
                state.count_sayas_digits = 0
                if u.is_wdigit(s.prev_out):
                    s.c = 32
                    s.space_inserted = True
            return False
        return state.option_sayas2 & 16 == 0

    @staticmethod
    def _normalize_word_marks(s):
        if s.c in (146, 180, 8217, 8242):
            s.c = 39
        if s.c in (8216, 63) and u.is_alpha(s.prev_out) and u.is_alpha(s.next_in):
            s.c = 39
        if s.c == CHAR_EMPHASIS:
            s.c = 32
            s.word_flags |= FLAG_FOCUS
        if s.c == CHAR_COMMA_BREAK:
            s.c = 32
            s.word_flags |= FLAG_COMMA_AFTER

    def _word_characters(self, tr, s):
        self._normalize_word_marks(s)
        s.c, s.char_inserted, s.word_flags = self.characters.translate(tr, s.source, s.source_index, s.prev_in, s.c, s.next_in, s.word_flags)
        if s.c == 8:
            return True
        if s.char_inserted:
            s.next_in = s.char_inserted
        if not u.is_alpha(s.c) and (not u.is_space(s.c)) and (s.c not in tr.punct_within_word) and u.is_alpha(s.prev_out):
            if not (tr.langopts.tone_numbers and 48 <= s.c <= 57 and (not 48 <= s.next_in <= 57)):
                s.c = 32
                s.space_inserted = True
        if u.is_wdigit(s.prev_out):
            if not u.is_wdigit(s.c) and s.c not in (46, 44, 32):
                s.c = 32
                s.space_inserted = True
        elif s.prev_in == 44:
            if s.c == 44:
                s.c = 32
        else:
            s.decimal_sep_count = False
        if s.c == 91 and (s.next_in == 2 or (s.next_in == 91 and self.state.option_phoneme_input)):
            s.phoneme_mode = True
            s.source_index += 1
            return True
        return False

    def _alphabetic_start(self, tr, s):
        s.alpha_count += 1
        if u.is_alpha(s.prev_out) and (not (tr.langopts.ideographs and (s.c > 12352 or s.prev_out > 12352))):
            return
        if s.prev_out not in tr.punct_within_word:
            s.letter_count = 0
        if s.prev_out != 32 and s.prev_out not in tr.punct_within_word:
            s.c = 32
            s.space_inserted = True
            if not is_bracket(s.prev_out):
                s.next_word_flags |= FLAG_NOSPACE
        else:
            if u.is_wupper(s.c):
                s.word_flags |= FLAG_FIRST_UPPER
            if s.prev_out == 32 and u.is_wdigit(bytec(s.buffer[s.ix - 2])) and (not u.is_wdigit(s.prev_in)):
                s.buffer[s.ix] = 32
                s.ix += 1
                s.words[s.word_count].start += 1

    @staticmethod
    def _mix_alphabets(tr, s):
        if s.c == 32:
            return
        s.letter_count += 1
        if tr.letter_bits_offset <= 0:
            return
        mixed = s.c < 592 and s.prev_out >= tr.letter_bits_offset or (s.c >= tr.letter_bits_offset and s.letter_count > 1 and (s.prev_out < 592))
        if mixed and u.is_alpha(s.prev_out):
            s.c = 32
            s.space_inserted = True
            s.word_flags |= FLAG_HYPHEN_AFTER
            s.next_word_flags |= FLAG_HYPHEN

    def _uppercase(self, tr, s):
        s.c = lower(tr, s.c)
        if tr.langopts.param[LOPT_CAPS_IN_WORD]:
            if not s.syllable_marked:
                s.char_inserted = s.c
                s.c = 712
                s.syllable_marked = True
            return
        if u.is_wlower(s.prev_in):
            if s.prev_out != 32 and (not upper_case_in_word(tr, s.buffer, s.ix, s.c)):
                s.c = 32
                s.space_inserted = True
                s.prev_in_save = s.c
        elif s.c != 32 and u.is_wupper(s.prev_in) and u.is_wlower(s.next_in):
            next2, _ = u.utf8_in(s.source, s.source_index + s.next_in_nbytes)
            if tr.translator_name == 28268 and s.letter_count == 2 and (s.c == 106) and (s.prev_in == 73):
                return
            if s.prev_out != 32 and u.is_alpha(next2):
                s.c = 32
                s.space_inserted = True
                s.prev_in_save = s.c
                s.next_word_flags |= FLAG_NOSPACE

    def _alphabetic(self, tr, s):
        self._alphabetic_start(tr, s)
        self._mix_alphabets(tr, s)
        if u.is_wupper(s.c):
            self._uppercase(tr, s)
        elif s.all_upper_case and s.letter_count > 2:
            if tr.translator_name == 25966 and s.c == 115 and (s.next_in == 32):
                s.c = 32
                s.all_upper_case |= FLAG_HAS_PLURAL
                if s.buffer[s.ix - 1] == 39:
                    s.buffer[s.ix - 1] = 32
            else:
                s.all_upper_case = 0
        else:
            s.all_upper_case = 0

    @staticmethod
    def _hyphen(tr, s):
        if not u.is_space(s.prev_in) and u.is_alpha(s.next_in):
            if s.prev_out != 32:
                s.c = 32
                s.space_inserted = True
            else:
                s.word_flags |= FLAG_HYPHEN
                if s.word_count > 0:
                    s.words[s.word_count - 1].flags |= FLAG_HYPHEN_AFTER
                s.c = 32
        elif s.prev_in == 32 and s.next_in == 32:
            s.c = 32
            s.pre_pause_add = 4
        elif s.next_in == 45:
            s.source_index += 1
            s.c = 32
            s.pre_pause_add = 4
        elif s.prev_out == 32 and u.is_alpha(s.prev_out2) and (not u.is_alpha(s.prev_in)):
            s.buffer[s.ix] = 32
            s.ix += 1
            s.words[s.word_count].start += 1

    @staticmethod
    def _period(tr, s):
        if s.prev_out == 46:
            s.c = 32
            s.space_inserted = True
        elif s.word_count > 0 and (not s.words[s.word_count - 1].flags & FLAG_NOSPACE) and u.is_alpha(s.prev_in):
            s.words[s.word_count - 1].flags |= FLAG_HAS_DOT
            if u.is_space(s.next_in) or s.next_in == 45:
                s.c = 32

    @staticmethod
    def _apostrophe(tr, s):
        if (s.prev_in == 46 and s.next_in == 115 or u.is_walnum(s.prev_in)) and u.is_alpha(s.next_in):
            s.single_quoted = False
        elif tr.langopts.param[LOPT_APOSTROPHE] & 1 and u.is_alpha(s.next_in):
            s.single_quoted = False
        elif tr.langopts.param[LOPT_APOSTROPHE] & 2 and u.is_alpha(s.prev_in):
            s.single_quoted = False
        elif s.prev_in in tr.char_plus_apostrophe and s.prev_out2 == 32:
            s.single_quoted = False
            if s.next_in == 32:
                s.source_index += 1
        elif s.prev_out == 115 and (not s.single_quoted):
            s.c = 32
        else:
            s.single_quoted = bool(u.is_space(s.prev_out))
            s.pre_pause_add = 4
            s.c = 32

    @staticmethod
    def _digit(tr, s):
        if tr.langopts.tone_numbers and u.is_alpha(s.prev_out) and (not u.is_digit(s.next_in)):
            return
        if s.prev_out != 32 and (not u.is_wdigit(s.prev_out)):
            if s.prev_out != tr.langopts.decimal_sep or (s.decimal_sep_count and tr.langopts.decimal_sep == 44):
                s.c = 32
                s.space_inserted = True
            else:
                s.decimal_sep_count = True
        elif s.prev_out == 32 and u.is_alpha(s.prev_out2) and (not u.is_alpha(s.prev_in)):
            s.buffer[s.ix] = 32
            s.ix += 1
            s.words[s.word_count].start += 1

    def _word(self, tr, s):
        if self._word_characters(tr, s):
            return True
        if u.is_alpha(s.c):
            self._alphabetic(tr, s)
        elif s.c == 45:
            self._hyphen(tr, s)
        elif s.c == 46:
            self._period(tr, s)
        elif s.c == 39:
            self._apostrophe(tr, s)
        elif s.c == 95:
            s.c = 32
        elif u.is_wdigit(s.c):
            self._digit(tr, s)
        return False

    def _end_word(self, s):
        state = self.state
        word = s.words[s.word_count]
        if s.embedded_count:
            state.embedded_list[state.embedded_ix - 1] |= 128
            word.flags |= FLAG_EMBEDDED
            s.embedded_count = 0
        if s.alpha_count == 0:
            s.all_upper_case &= ~FLAG_ALL_UPPER
        word.pre_pause = state.pre_pause & 255
        word.flags = (word.flags | s.all_upper_case | s.word_flags | state.word_emphasis) & 4294967295
        if state.pre_pause > 0:
            for j in range(s.ix, word.start, -1):
                s.buffer[j] = s.buffer[j - 1]
            s.buffer[word.start] = 32
            word.start += 1
            s.ix += 1
        s.word_count += 1
        s.words[s.word_count].start = s.ix
        s.words[s.word_count].flags = 0
        s.words[s.word_count].length = calc_word_length(s.source_index, s.charix_top, s.charix, s.words, s.word_count)
        s.word_flags = s.next_word_flags
        s.next_word_flags = 0
        state.pre_pause = 0
        s.all_upper_case = FLAG_ALL_UPPER
        s.alpha_count = 0
        s.syllable_marked = False

    def _space(self, s):
        if s.prev_out == 32:
            s.word_flags |= FLAG_MULTIPLE_SPACES
            return True
        if s.cc in (9, 10):
            s.next_word_flags |= FLAG_MULTIPLE_SPACES
        if s.space_inserted:
            j = 0
            k = s.source_index - 1
            while k >= s.source_index_word and s.charix[k] != 0:
                if s.charix[k] > 0:
                    j += 1
                k -= 1
            s.words[s.word_count].length = j & 255
        s.source_index_word = s.source_index
        s.buffer[s.ix] = 32
        s.ix += 1
        if s.word_count < N_CLAUSE_WORDS - 1 and s.ix > s.words[s.word_count].start:
            self._end_word(s)
        if s.space_inserted:
            s.source_index = s.prev_source_index
            s.char_inserted = 0
            s.space_inserted = False
        return False

    def tokenize(self, tr, source, charix, charix_top, terminator):
        s = SimpleNamespace(source=source, charix=charix, charix_top=charix_top, buffer=bytearray(N_TR_SOURCE + 4), words=[WordTab() for _ in range(N_CLAUSE_WORDS)], ix=3, source_index=0, source_index_word=0, prev_in=32, prev_out=32, prev_in_save=0, char_inserted=0, cc=0, all_upper_case=FLAG_ALL_UPPER, alpha_count=0, finished=False, single_quoted=False, phoneme_mode=False, word_flags=0, next_word_flags=0, embedded_count=0, letter_count=0, space_inserted=False, syllable_marked=False, decimal_sep_count=False, pre_pause_add=0, word_count=0)
        s.buffer[1:3] = b'  '
        s.words[0].start = 3
        s.words[0].length = calc_word_length(0, charix_top, charix, s.words, 0)
        while not s.finished and s.ix < N_TR_SOURCE - 1:
            self._next(tr, s)
            if self._embedded(s):
                continue
            if self._sayas(tr, s) and self._word(tr, s):
                continue
            if u.is_space(s.c):
                if self._space(s):
                    continue
            elif s.ix < N_TR_SOURCE - 4:
                s.ix = _put(s.buffer, s.ix, u.utf8_out(s.c))
            self.state.pre_pause = max(self.state.pre_pause, s.pre_pause_add)
            s.pre_pause_add = 0
        self._finish(s, terminator)
        return s

    def _finish(self, s, terminator):
        if s.word_count == 0 and s.embedded_count:
            self.state.embedded_list[self.state.embedded_ix - 1] |= 128
            s.words[0].flags |= FLAG_EMBEDDED
            s.word_count = 1
        s.clause_end = s.ix - 1
        s.buffer[s.ix] = 0
        s.words[0].pre_pause = 0
        s.words[s.word_count].pre_pause = 8
        if s.word_count > 0:
            index = s.word_count - 1
            while index > 0 and is_bracket(bytec(s.buffer[s.words[index].start])):
                index -= 1
            s.words[index].flags |= FLAG_LAST_WORD
            if terminator & CLAUSE_DOT_AFTER_LAST_WORD and (not s.words[s.word_count - 1].flags & FLAG_NOSPACE):
                s.words[s.word_count - 1].flags |= FLAG_HAS_DOT
        s.words[0].flags |= FLAG_FIRST_WORD
