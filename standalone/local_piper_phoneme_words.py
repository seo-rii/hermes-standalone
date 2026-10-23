"""Original TranslateWord2 raw-phoneme and first-stage context construction.

Adapted from eSpeakNG translate.c, Copyright 2005-2014 Jonathan Duddington
and Copyright 2015-2018 Reece H.Dunn, GPL-3.0-or-later. Original source and
independent native observations are recorded in local-piper-phoneme-text-
provenance.json; complete license notices are in licenses/. This qualified
implementation executes owned mutable state and standard-library algorithms.
"""
from types import SimpleNamespace
from contextlib import nullcontext
from standalone import local_piper_phoneme_unicode as u
from .local_piper_phoneme_text_tables import *

def i32(value):
    return (value + 2147483648) % 4294967296 - 2147483648

class WordBuilder:

    def __init__(self, controller, translate_word):
        self.controller = controller
        self.state = controller.state
        self.translate_word = translate_word

    def set_item(self, index, code):
        item = self.state.ph_list2[index]
        item.phcode = code
        item.stresslevel = item.tone_ph = item.sourceix = 0
        item.synthflags = self.state.embedded_flag & 65535
        self.state.embedded_flag = 0
        return item

    def append_item(self, code):
        item = self.set_item(self.state.n_ph_list2, code)
        self.state.n_ph_list2 += 1
        return item

    def call_word(self, tr, buffer, cursor, words, index, word_out, *, temporary=False):
        temporary = temporary or (hasattr(self.state, 'clause_buffer') and buffer is not self.state.clause_buffer)
        with tr.session.outside_clause() if temporary else nullcontext():
            result = self.translate_word(tr, buffer, cursor, words, index, word_out=word_out)
        self.state.word_phonemes = result.phonemes
        return result

    def combine(self, tr, buffer, cursor, words, index, flags):
        state = self.state
        limit = tr.langopts.param[LOPT_COMBINE_WORDS]
        boundary = buffer.index(32, cursor)
        cp, _ = u.utf8_in(buffer, boundary + 1)
        if not u.is_walpha(cp):
            return flags
        old = state.word_phonemes
        second = self.call_word(tr, buffer, boundary + 1, words, index + 1, False).flags
        valid = not (second & FLAG_WAS_UNPRONOUNCABLE or state.word_phonemes[:1] == bytes((phonSWITCH,)))
        if limit & 256 and (not second & FLAG_ALT_TRANS):
            valid = False
        if limit & 512 and words[index + 1].flags & FLAG_LAST_WORD:
            valid = False
        if not valid:
            state.word_phonemes = old
            return flags
        buffer[boundary] = 45
        words[index].flags &= ~FLAG_ALL_UPPER
        flags = self.call_word(tr.session.primary, buffer, cursor, words, index, False).flags
        syllables = sum((tr.phoneme_tab[code].type == phVOWEL for code in state.word_phonemes))
        if limit > 0 and syllables > limit & 31:
            buffer[boundary] = 32
            return self.call_word(tr.session.primary, buffer, cursor, words, index, False).flags
        state.dictionary_skipwords = 1
        return i32((flags or second) | FLAG_SKIPWORDS)

    def switch_language(self, tr, buffer, cursor):
        start = cursor + 3
        end = start
        while buffer[end] not in (0, 32):
            end += 1
        language = bytes(buffer[start:end]).lower().decode('ascii')
        try:
            number = tr.session.data.number(language)
        except ValueError:
            return
        if number > 0:
            tr.session.selection.select(number)
            self.state.word_phonemes = bytes((phonSWITCH, number))

    def _pitch(self, s):
        state = self.state
        if not s.word_flags & FLAG_FIRST_UPPER or state.option_capitals <= 2 or state.embedded_ix >= N_EMBEDDED_LIST - 6:
            return
        if state.embedded_flag:
            state.embedded_list[state.embedded_ix - 1] &= ~128
        s.pitch_raised = 20 if state.option_capitals == 3 else state.option_capitals
        state.embedded_list[state.embedded_ix] = EMBED_P + 64 + 128 + (s.pitch_raised << 8) & 4294967295
        state.embedded_ix += 1
        state.embedded_flag = SFLAG_EMBEDDED

    def _start(self, tr, s):
        state = self.state
        word = s.words[s.index]
        s.word_flags = word.flags
        s.source_ix = word.sourceix & 2047 | min(word.length, 31) << 11
        if s.word_flags & FLAG_EMBEDDED:
            word.flags &= ~FLAG_EMBEDDED
            state.embedded_flag = SFLAG_EMBEDDED
            self.controller.word_embedded_command(tr)
        if state.n_ph_list2 >= N_PHONEME_LIST - 2:
            return False
        if s.buffer[s.cursor] == 0 or s.word_flags & FLAG_DELETE_WORD:
            if state.embedded_flag:
                self.append_item(phonEND_WORD).wordstress = 0
                state.embedded_flag = 0
            state.word_phonemes = b''
            return False
        if state.n_ph_list2 >= N_PHONEME_LIST - 9:
            return False
        if tr.prepause_timeout > 0:
            tr.prepause_timeout -= 1
        if state.option_sayas & 240 == 16 and (not s.word_flags & FLAG_FIRST_WORD):
            s.pause += 4
        self._pitch(s)
        return True

    def _alternate(self, tr, s):
        state = self.state
        s.old_dictionary_name = state.dictionary_name
        for _ in range(2):
            s.buffer[s.cursor:s.cursor + len(s.copy)] = s.copy
            language = state.word_phonemes[1:].split(b'\x00', 1)[0] or b'en'
            s.switch_phonemes = tr.session.set_translator2(language.decode('ascii'))
            if s.switch_phonemes >= 0:
                s.words[s.index].flags |= FLAG_TRANSLATOR2
                if s.replacement:
                    replacement = bytearray(b'\x00 ' + s.replacement + b'\x00')
                    s.flags = self.call_word(tr.session.secondary, replacement, 1, s.words, s.index, False, temporary=True).flags
                else:
                    result = self.call_word(tr.session.secondary, s.buffer, s.cursor, s.words, s.index, True)
                    s.flags = result.flags
                    s.replacement = result.replacement or b''
            if state.word_phonemes[:1] != bytes((phonSWITCH,)):
                break
        if state.word_phonemes[:1] == bytes((phonSWITCH,)):
            return False
        if s.switch_phonemes < 0:
            state.word_phonemes = bytes((phonSCHWA, phonSCHWA))
        if s.switch_phonemes == -1:
            state.dictionary_name = s.old_dictionary_name
            tr.session.selection.select(tr.session.voice.phoneme_tab_ix)
            s.switch_phonemes = tr.session.voice.phoneme_tab_ix
        return True

    @staticmethod
    def _word_pause(tr, s):
        if not s.word_flags & FLAG_HYPHEN:
            if s.flags & FLAG_PAUSE1:
                s.pause = max(s.pause, 1)
            word_flags = s.word_flags
            previous = s.words[s.index - 1].flags if s.index > 0 else 0
            if s.flags & FLAG_PREPAUSE and (not word_flags & (FLAG_LAST_WORD | FLAG_FIRST_WORD)) and (not previous & FLAG_FIRST_WORD) and (tr.prepause_timeout == 0):
                s.pause = max(s.pause, 4)
                tr.prepause_timeout = 3

    def _translate(self, tr, s):
        state = self.state
        if s.word_flags & FLAG_PHONEMES:
            if s.buffer[s.cursor:s.cursor + 3] == b'_^_':
                self.switch_language(tr, s.buffer, s.cursor)
            else:
                state.word_phonemes = tr.encode_phonemes(s.buffer, s.cursor).phonemes
            s.flags = i32(FLAG_FOUND)
            return True
        length = 0
        while s.buffer[s.cursor + length] not in (0, 32) and length < N_WORD_BYTES:
            length += 1
        s.copy = bytes(s.buffer[s.cursor:s.cursor + length])
        result = self.call_word(tr.session.primary, s.buffer, s.cursor, s.words, s.index, True)
        s.flags = result.flags
        s.replacement = result.replacement or b''
        if s.flags & FLAG_SPELLWORD:
            s.buffer[s.cursor:s.cursor + length] = s.copy
            return False
        if s.flags & FLAG_COMBINE and (not s.words[s.index + 1].flags & FLAG_PHONEMES):
            s.flags = self.combine(tr, s.buffer, s.cursor, s.words, s.index, s.flags)
        if state.word_phonemes[:1] == bytes((phonSWITCH,)) and (not self._alternate(tr, s)):
            s.flags = FLAG_SPELLWORD
            return False
        self._word_pause(tr, s)
        if state.option_emphasis >= 3:
            s.pause = max(s.pause, 1)
        return True

    def _pauses(self, tr, s):
        state = self.state
        while s.pause > 0 and state.n_ph_list2 < N_PHONEME_LIST - 9:
            if s.pause > 1:
                self.append_item(phonPAUSE)
                s.pause -= 2
            else:
                self.append_item(phonPAUSE_NOLINK)
                s.pause -= 1
            tr.end_stressed_vowel = 0
            tr.prev_dict_flags[:] = [0, 0]
        s.word_start = state.n_ph_list2
        if state.option_capitals == 1 and s.word_flags & FLAG_FIRST_UPPER:
            self.append_item(phonPAUSE_SHORT)
            self.append_item(phonCAPITAL)
            if s.word_flags & FLAG_ALL_UPPER and u.is_alpha(s.buffer[s.cursor + 1]):
                self.append_item(phonPAUSE_SHORT)
                self.append_item(phonCAPITAL)

    def _initial_switch(self, s):
        state = self.state
        if s.switch_phonemes < 0:
            return
        if s.phones[:2] == bytes((phonPAUSE, phonSWITCH)):
            if state.ph_list2[state.n_ph_list2 - 1].phcode == phonSWITCH:
                state.n_ph_list2 -= 1
        else:
            if state.ph_list2[state.n_ph_list2 - 1].phcode == phonSWITCH:
                state.n_ph_list2 -= 1
            else:
                self.set_item(state.n_ph_list2, phonSWITCH)
            state.ph_list2[state.n_ph_list2].tone_ph = s.switch_phonemes & 255
            state.n_ph_list2 += 1

    def _vowel(self, tr, s, item):
        state = self.state
        s.stress = s.next_stress
        s.next_stress = 1
        if s.stress >= 4:
            state.any_stressed_words = True
        if s.prev_vowel >= 0 and state.n_ph_list2 - 1 != s.prev_vowel:
            state.ph_list2[state.n_ph_list2 - 1].stresslevel = s.stress & 255
        item.synthflags |= SFLAG_SYLLABLE
        s.prev_vowel = state.n_ph_list2
        if s.stress > s.max_stress:
            s.max_stress = s.stress
            s.max_stress_ix = state.n_ph_list2
        if s.next_tone:
            item.tone_ph = s.next_tone & 255
            s.next_tone = 0

    def _phone(self, tr, s, code, phone):
        state = self.state
        item = state.ph_list2[state.n_ph_list2]
        item.phcode = code
        item.tone_ph = 0
        item.synthflags = (state.embedded_flag | s.found_dict_flag) & 65535
        state.embedded_flag = 0
        item.sourceix = s.srcix & 65535
        s.srcix = 0
        if phone.type == phVOWEL:
            self._vowel(tr, s, item)
        elif s.first_phoneme and tr.prev_dict_flags[0] & FLAG_DOUBLING:
            item.synthflags |= SFLAG_LENGTHEN
        item.stresslevel = s.stress & 255
        state.n_ph_list2 += 1
        s.first_phoneme = False

    def _special_phone(self, tr, s, code, phone):
        state = self.state
        if code == phonSWITCH:
            item = state.ph_list2[state.n_ph_list2]
            item.phcode = code
            item.stresslevel = item.sourceix = item.synthflags = 0
            item.tone_ph = s.phones[s.position] & 255
            state.n_ph_list2 += 1
            tr.session.selection.select(s.phones[s.position])
            s.position += 1
        elif phone.type == phSTRESS:
            if phone.program == 0:
                s.next_stress = phone.std_length
            elif s.prev_vowel >= 0:
                state.ph_list2[s.prev_vowel].tone_ph = code
            else:
                s.next_tone = code
        elif code == phonSYLLABIC:
            s.prev_vowel = state.n_ph_list2 - 1
            state.ph_list2[s.prev_vowel].synthflags |= SFLAG_SYLLABLE
            state.ph_list2[s.prev_vowel].stresslevel = s.next_stress & 255
        elif code == phonLENGTHEN:
            state.ph_list2[state.n_ph_list2 - 1].synthflags |= SFLAG_LENGTHEN
        elif code == phonEND_WORD:
            s.srcix = s.source_ix + 1
        elif code == phonX1:
            s.flags = i32(s.flags | FLAG_DOUBLING)
        else:
            self._phone(tr, s, code, phone)

    def _phones(self, tr, s):
        state = self.state
        s.phones = state.word_phonemes + b'\x00' * 4
        s.position = 0
        self._initial_switch(s)
        if s.word_flags & FLAG_HYPHEN and tr.phoneme_tab[s.phones[0]].type == phPAUSE:
            s.position += 1
        if s.phones[s.position] == 0 and state.embedded_flag:
            phones = bytearray(s.phones)
            phones[s.position:s.position + 2] = bytes((phonPAUSE_VSHORT, 0))
            s.phones = bytes(phones)
            state.word_phonemes = s.phones.split(b'\x00', 1)[0]
        while state.n_ph_list2 < N_PHONEME_LIST - 5:
            code = s.phones[s.position]
            s.position += 1
            if code == 0:
                break
            if code == 255:
                continue
            phone = tr.phoneme_tab[code]
            if phone is None:
                continue
            self._special_phone(tr, s, code, phone)

    def _finish(self, tr, s):
        state = self.state
        if s.word_flags & FLAG_COMMA_AFTER:
            self.append_item(phonPAUSE_CLAUSE)
        if not s.word_flags & FLAG_HYPHEN:
            state.ph_list2[s.word_start].sourceix = s.source_ix & 65535
        tr.end_stressed_vowel = int(s.stress >= 4 and tr.phoneme_tab[state.ph_list2[state.n_ph_list2 - 1].phcode].type == phVOWEL)
        if s.switch_phonemes >= 0:
            state.dictionary_name = s.old_dictionary_name
            tr.session.selection.select(tr.session.voice.phoneme_tab_ix)
            self.append_item(phonSWITCH).tone_ph = tr.session.voice.phoneme_tab_ix & 255
        if s.pitch_raised > 0:
            state.embedded_list[state.embedded_ix] = EMBED_P + 96 + 128 + (s.pitch_raised << 8) & 4294967295
            state.embedded_ix += 1
            self.append_item(phonPAUSE_SHORT).synthflags = SFLAG_EMBEDDED
        if s.flags & FLAG_STRESS_END2:
            state.ph_list2[s.max_stress_ix].synthflags |= SFLAG_PROMOTE_STRESS
        tr.prev_dict_flags[0] = i32(s.flags)

    def translate(self, tr, buffer, cursor, words, index, pause):
        if not hasattr(self.state, 'dictionary_name'):
            self.state.dictionary_name = tr.session.primary.dictionary_name
        s = SimpleNamespace(buffer=buffer, cursor=cursor, words=words, index=index, pause=pause, flags=0, next_tone=0, max_stress_ix=0, prev_vowel=-1, pitch_raised=0, switch_phonemes=-1, first_phoneme=True)
        if not self._start(tr, s):
            return 0
        if not self._translate(tr, s):
            return s.flags
        s.stress = 0
        s.next_stress = 1
        s.srcix = 0
        s.max_stress = -1
        s.found_dict_flag = SFLAG_DICTIONARY if s.flags & FLAG_FOUND and (not s.flags & FLAG_TEXTMODE) else 0
        self._pauses(tr, s)
        self._phones(tr, s)
        self._finish(tr, s)
        return i32(s.flags)
