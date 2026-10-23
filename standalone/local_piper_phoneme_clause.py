"""Original eSpeak ReadClause plain-text and embedded-control algorithm.

Adapted from eSpeakNG readclause.c, Copyright2005-2014 Jonathan Duddington
and Copyright2015-2018 Reece H.Dunn, GPL-3.0-or-later. See the complete source
provenance in local-piper-phoneme-clause-provenance.json and license notices
in licenses/local-piper-phoneme-clause-notices.txt.

The qualified owned Translator provides actual dictionary/rule/stress helpers.
No native library, external process or fixed phrase translation is used.
"""
from dataclasses import dataclass
from types import SimpleNamespace
from standalone import local_piper_phoneme_unicode as u
from .local_piper_phoneme_clause_tables import *
PROPERTY_MASK = 18442240474082181120
PUNCTUATION_TYPES = {ESPEAKNG_PROPERTY_FULL_STOP: CLAUSE_PERIOD, ESPEAKNG_PROPERTY_FULL_STOP | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER: CLAUSE_PERIOD | CLAUSE_OPTIONAL_SPACE_AFTER, ESPEAKNG_PROPERTY_QUESTION_MARK: CLAUSE_QUESTION, ESPEAKNG_PROPERTY_QUESTION_MARK | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER: CLAUSE_QUESTION | CLAUSE_OPTIONAL_SPACE_AFTER, ESPEAKNG_PROPERTY_QUESTION_MARK | ESPEAKNG_PROPERTY_PUNCTUATION_IN_WORD: CLAUSE_QUESTION | CLAUSE_PUNCTUATION_IN_WORD, ESPEAKNG_PROPERTY_EXCLAMATION_MARK: CLAUSE_EXCLAMATION, ESPEAKNG_PROPERTY_EXCLAMATION_MARK | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER: CLAUSE_EXCLAMATION | CLAUSE_OPTIONAL_SPACE_AFTER, ESPEAKNG_PROPERTY_EXCLAMATION_MARK | ESPEAKNG_PROPERTY_PUNCTUATION_IN_WORD: CLAUSE_EXCLAMATION | CLAUSE_PUNCTUATION_IN_WORD, ESPEAKNG_PROPERTY_COMMA: CLAUSE_COMMA, ESPEAKNG_PROPERTY_COMMA | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER: CLAUSE_COMMA | CLAUSE_OPTIONAL_SPACE_AFTER, ESPEAKNG_PROPERTY_COLON: CLAUSE_COLON, ESPEAKNG_PROPERTY_COLON | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER: CLAUSE_COLON | CLAUSE_OPTIONAL_SPACE_AFTER, ESPEAKNG_PROPERTY_SEMI_COLON: CLAUSE_SEMICOLON, ESPEAKNG_PROPERTY_EXTENDED_DASH: CLAUSE_SEMICOLON, ESPEAKNG_PROPERTY_SEMI_COLON | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER: CLAUSE_SEMICOLON | CLAUSE_OPTIONAL_SPACE_AFTER, ESPEAKNG_PROPERTY_QUESTION_MARK | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER | ESPEAKNG_PROPERTY_INVERTED_TERMINAL_PUNCTUATION: CLAUSE_SEMICOLON | CLAUSE_OPTIONAL_SPACE_AFTER, ESPEAKNG_PROPERTY_EXCLAMATION_MARK | ESPEAKNG_PROPERTY_OPTIONAL_SPACE_AFTER | ESPEAKNG_PROPERTY_INVERTED_TERMINAL_PUNCTUATION: CLAUSE_SEMICOLON | CLAUSE_OPTIONAL_SPACE_AFTER, ESPEAKNG_PROPERTY_ELLIPSIS: CLAUSE_SEMICOLON | CLAUSE_SPEAK_PUNCTUATION_NAME | CLAUSE_OPTIONAL_SPACE_AFTER, ESPEAKNG_PROPERTY_PARAGRAPH_SEPARATOR: CLAUSE_PARAGRAPH}

def clause_type_from_codepoint(c):
    return PUNCTUATION_TYPES.get(u.properties(c) & PROPERTY_MASK, CLAUSE_NONE)

def is_bracket(c):
    return 8212 <= c <= 8223 or c in (40, 41, 91, 93, 123, 125, 60, 62, 34, 39, 96, 171, 187, 12298, 12299, 57404)

def word_to_string(value):
    return bytes((c for ix in (24, 16, 8, 0) if (c := (value >> ix & 255))))

@dataclass(frozen=True)
class ClauseResult:
    buffer: bytes
    charix: tuple
    charix_top: int
    terminator: int
    tone: int
    voice_change: bytes
    cursor: object

class ClauseReader:

    def __init__(self, text=b'', *, soundicons=None, option_ssml=0):
        if option_ssml:
            raise ValueError('The qualified plain-text Piper reader does not enable SSML synthesis')
        self.option_ssml = 0
        self.option_linelength = 0
        self.option_capitals = 0
        self.option_punctuation = 0
        self.option_punctlist = ()
        self.option_phoneme_input = 0
        self.speech_parameters = list(DEFAULT_SPEECH_PARAMETERS)
        self.embedded_value = list(DEFAULT_EMBEDDED_VALUE)
        self.soundicons = {} if soundicons is None else dict(soundicons)
        self.ungot_char = self.ungot_char2 = 0
        self.count_characters = DEFAULT_COUNT_CHARACTERS
        self.clause_start_char = 0
        self.skip_characters = 0
        self.end_character_position = 0
        self.skipping_text = False
        self.ignore_text = self.audio_text = self.clear_skipping_text = False
        self.sayas_mode = self.sayas_start = 0
        self.bind_text(text)

    def initialize(self):
        self.ungot_char = self.ungot_char2 = 0
        self.speech_parameters = list(DEFAULT_PARAM_STACK)
        self.option_punctuation = self.speech_parameters[5]
        self.option_capitals = self.speech_parameters[6]
        self.ignore_text = self.audio_text = self.clear_skipping_text = False
        self.count_characters = -1
        self.sayas_mode = 0

    def bind_text(self, text, cursor=0):
        if isinstance(text, str):
            text = text.encode('utf8')
        text = bytes(text)
        text.decode('utf8')
        if b'\x00' in text:
            raise ValueError('Piper text contains an embedded null character')
        if not 0 <= cursor <= len(text):
            raise ValueError('Invalid text cursor')
        self.text = text + b'\x00'
        self.position = cursor

    def eof(self):
        return not self.ungot_char and self.position == len(self.text)

    def getc(self):
        if self.ungot_char:
            c = self.ungot_char
            self.ungot_char = 0
            return c
        self.count_characters += 1
        if self.position == len(self.text):
            return 0
        c, size = u.utf8_in(self.text, self.position)
        self.position += size
        return c

    def ungetc(self, c):
        self.ungot_char = c

    def peekc(self):
        if self.position == len(self.text):
            return 0
        return u.utf8_in(self.text, self.position)[0]

    @staticmethod
    def _write(s, data):
        if s.ix + len(data) > len(s.buffer):
            raise ValueError('Piper clause exceeds its caller buffer')
        s.buffer[s.ix:s.ix + len(data)] = data
        s.ix += len(data)

    def _terminate(self, s, index=None, unget=None):
        index = s.ix if index is None else index
        if index + 2 > len(s.buffer):
            raise ValueError('Piper clause has no terminator space')
        s.buffer[index:index + 2] = b' \x00'
        if unget is not None:
            self.ungetc(unget)

    @staticmethod
    def _decode_mode(tr, phones, flags, other=None):
        active = tr if other is None else other
        phones = active.set_word_stress(phones, list(flags), -1, 0)
        decoded = active.decode_phonemes(phones)
        if other is None:
            return b'[\x02' + decoded + b']]'
        return b'[\x02_^_en ' + decoded + b' _^_' + word_to_string(tr.translator_name) + b']]'

    def lookup_special(self, tr, word):
        result = tr.lookup_dict_list(word, flags=(0, 0), end_flags=0, word_table=None)
        return self._decode_mode(tr, result.phonemes, result.flags) if result.found else None

    def lookup_char_name(self, tr, c, only=False):
        symbol = u.utf8_out(c)
        flags = (0, 0)
        phones = b''
        other = None
        if only:
            result = tr.lookup_dict_list(symbol, flags=flags, end_flags=0, word_table=None)
            phones = result.phonemes
            flags = result.flags
        else:
            result = tr.lookup_dict_list(b'_' + symbol, flags=flags, end_flags=0, word_table=None)
            phones = result.phonemes
            flags = result.flags
            if not result.found:
                result = tr.lookup_dict_list(symbol, flags=flags, end_flags=0, word_table=None)
                phones = result.phonemes
                flags = result.flags
                if not result.found:
                    phones = tr.translate_rules(symbol, max_phonemes=60, dict_flags=None, word_flags=0, initial=phones).phonemes
            if (not phones or phones[0] == 21) and tr.translator_name != 25966:
                english = tr.english_translator()
                result = english.lookup_dict_list(b'_' + symbol, flags=flags, end_flags=0, word_table=None)
                phones = result.phonemes
                flags = result.flags
                if not result.found:
                    result = english.lookup_dict_list(symbol, flags=flags, end_flags=0, word_table=None)
                    phones = result.phonemes
                    flags = result.flags
                if phones:
                    other = english
        if phones:
            return self._decode_mode(tr, phones, flags, other)
        return b'' if only else b'[\x02(X1)(X1)(X1)]]'

    def _repeat_name(self, tr, s, name, end_clause):
        count = 1
        while not self.eof() and s.c2 == s.c1 and (s.c1 != 60):
            count += 1
            s.c2 = self.getc()
        if end_clause:
            self.ungetc(s.c2)
        if count == 1:
            return b' ' + name
        if count >= 4:
            return b' ' + name + b' ' + str(count).encode() + b' ' + name
        fast = self.embedded_value[2] < 300
        return (b'\x01+10S' if fast else b'') + (b' ' + name) * count + (b' \x01-10S' if fast else b'')

    def _announce(self, tr, s, end_clause):
        c1, c2 = (s.c1, s.c2)
        saved = s.ix
        buf = b''
        soundicon = self.soundicons.get(c1, -1)
        if soundicon >= 0:
            buf = b'\x01' + str(soundicon).encode() + b'I '
            self.ungetc(c2)
        else:
            name = self.lookup_special(tr, b'_.p') if c1 == 46 and end_clause and (c2 != 46) else None
            if name is None:
                name = self.lookup_char_name(tr, c1, False)
            if saved == 0 or not end_clause or tr.langopts.param[LOPT_ANNOUNCE_PUNCT] & 2:
                buf = self._repeat_name(tr, s, name, end_clause)
            else:
                self.ungot_char2 = c1
                buf = b' \x00'
                self.ungetc(c2)
                buf = buf.split(b'\x00', 1)[0]
        self._write(s, buf)
        if not end_clause:
            return -1
        if c1 == 45:
            return CLAUSE_NONE
        attrs = clause_type_from_codepoint(c1)
        short = CLAUSE_SHORTCOMMA if attrs & CLAUSE_INTONATION_TYPE == 4096 else CLAUSE_SHORTFALL
        if saved > 0 and (not tr.langopts.param[LOPT_ANNOUNCE_PUNCT] & 2):
            return CLAUSE_SHORTFALL if attrs & ~CLAUSE_OPTIONAL_SPACE_AFTER == CLAUSE_SEMICOLON else short
        return attrs if attrs & CLAUSE_TYPE_SENTENCE else short

    def _embedded(self, tr, s):
        if s.c1 != CTRL_EMBEDDED:
            return None
        if s.c2 == 86:
            self._write(s, b'\x00')
            while not self.eof():
                s.c1 = self.getc()
                if u.is_wspace(s.c1) or s.ix >= s.n_buf - 1:
                    break
                self._write(s, bytes((s.c1 & 255,)))
            self._write(s, b'\x00')
            return CLAUSE_VOICE
        if s.c2 != 66:
            return None
        self._write(s, b'   ')
        if not self.eof():
            s.c2 = self.getc()
        if not self.eof() and s.c2 == 48:
            self.option_punctuation = 0
        else:
            self.option_punctuation = 1
            self.option_punctlist = ()
            if s.c2 != 49:
                chars = []
                while not self.eof() and (not u.is_wspace(s.c2)) and (len(chars) < 59):
                    chars.append(s.c2)
                    s.c2 = self.getc()
                    self._write(s, b' ')
                self.option_punctlist = tuple(chars)
                self.option_punctuation = 2
        if not self.eof():
            s.c2 = self.getc()
        return 'continue'

    @staticmethod
    def _replace(tr, c):
        pairs = tr.chars_ignore
        if isinstance(pairs, dict):
            replacement = pairs.get(c, c)
            return (replacement == 1 and c in pairs, replacement)
        for ix in range(0, len(pairs), 2):
            if not pairs[ix]:
                break
            if pairs[ix] == c:
                return (pairs[ix + 1] == 1, pairs[ix + 1])
        return (False, c)

    def _case(self, tr, s):
        if u.is_walnum(s.c1):
            s.any_alnum = True
        else:
            if s.stressed_word:
                s.stressed_word = False
                s.c1 = CHAR_EMPHASIS
                self.ungetc(s.c2)
                s.c2 = 32
            if s.c1 == 3851:
                s.c1 = 32
            if s.c1 == 3405 and s.c2 == 8205:
                s.c1 = 3406
        if u.is_wupper(s.c1):
            tr.clause_upper_count += 1
            if self.option_capitals == 2 and self.sayas_mode == 0 and (not u.is_wupper(s.cprev)):
                text = self.lookup_special(tr, b'_cap')
                if text is not None and s.ix + len(text) < s.n_buf:
                    self._write(s, text)
        elif u.is_walpha(s.c1):
            tr.clause_lower_count += 1
        if self.option_phoneme_input:
            if s.phoneme_mode > 0:
                s.phoneme_mode -= 1
            elif s.c1 == 91 and s.c2 == 91:
                s.phoneme_mode = -1
            elif s.c1 == 93 and s.c2 == 93:
                s.phoneme_mode = 2

    def _newlines(self, tr, s):
        if s.c1 != 10:
            return None
        count = 0
        while not self.eof() and u.is_wspace(s.c2):
            if s.c2 == 10:
                count += 1
            s.c2 = self.getc()
        if count:
            self._terminate(s, unget=s.c2)
            return CLAUSE_PARAGRAPH - 30 + 30 * min(count, 3)
        if s.linelength <= self.option_linelength:
            self._terminate(s, unget=s.c2)
            return CLAUSE_COLON
        s.linelength = 0
        return None

    def _dot_sequence(self, s):
        if s.c1 == 46 and s.c2 == 46:
            while not self.eof():
                s.c_next = self.getc()
                if s.c_next != 46:
                    break
                s.c1 = 8230
                s.c2 = 32
            if s.c1 == 8230:
                s.c2 = s.c_next
            else:
                self.ungetc(s.c_next)

    def _punctuation_attributes(self, s):
        s.is_end_clause = False
        self._dot_sequence(s)
        s.punct_data = clause_type_from_codepoint(s.c1)
        if s.punct_data != CLAUSE_NONE:
            if s.punct_data & (CLAUSE_QUESTION | CLAUSE_EXCLAMATION):
                while not self.eof() and clause_type_from_codepoint(s.c2) & (CLAUSE_QUESTION | CLAUSE_EXCLAMATION):
                    s.c_next = self.getc()
                    s.c2 = s.c_next
            if s.punct_data & CLAUSE_PUNCTUATION_IN_WORD:
                s.stressed_word = True
                s.tone = s.punct_data >> 12 & 15
                return 'continue'
            if u.is_wspace(s.c2) or s.punct_data & CLAUSE_OPTIONAL_SPACE_AFTER or is_bracket(s.c2) or (s.c2 == 63) or self.eof() or (s.c2 == CTRL_EMBEDDED):
                s.is_end_clause = True
        return None

    def _punctuation(self, tr, s):
        result = self._punctuation_attributes(s)
        if result is not None:
            return result
        if s.c1 == 57404:
            s.c1 = 60
        if self.option_punctuation and u.is_wpunct(s.c1) and (not self.audio_text):
            if self.option_punctuation == 1 or s.c1 in self.option_punctlist:
                tr.phonemes_repeat_count = 0
                terminator = self._announce(tr, s, s.is_end_clause)
                if terminator >= 0:
                    return terminator
                s.announced_punctuation = s.c1
        if s.punct_data & CLAUSE_SPEAK_PUNCTUATION_NAME and (not s.announced_punctuation):
            name = self.lookup_char_name(tr, s.c1, True)
            if name:
                self._write(s, name)
                s.announced_punctuation = s.c1
                s.punct_data &= ~CLAUSE_INTONATION_TYPE
        if s.is_end_clause:
            return self._end_clause(tr, s)
        return None

    def _end_clause(self, tr, s):
        nl_count = 0
        s.c_next = s.c2
        while not self.eof() and u.is_wspace(s.c_next):
            if s.c_next == 10:
                nl_count += 1
            s.c_next = self.getc()
        if s.c1 == 46 and nl_count < 2:
            s.punct_data |= CLAUSE_DOT_AFTER_LAST_WORD
        if not nl_count:
            self._abbreviation(tr, s)
        if s.is_end_clause:
            self._terminate(s, unget=s.c_next)
            if u.is_wdigit(s.cprev) and (not u.is_alpha(s.c_next)):
                s.punct_data &= ~CLAUSE_DOT_AFTER_LAST_WORD
            if nl_count > 1:
                return s.punct_data + 35 if s.punct_data in (CLAUSE_QUESTION, CLAUSE_EXCLAMATION) else CLAUSE_PARAGRAPH
            return s.punct_data
        if not self.eof() and u.is_wspace(s.c2):
            self.ungetc(s.c_next)
        return None

    @staticmethod
    def _ordinal_dot(tr, s):
        roman = lambda c: c in (73, 86, 88, 76)
        if not tr.langopts.numbers & NUM_ORDINAL_DOT:
            return
        preceding = u.is_wdigit(s.cprev) or (roman(s.cprev) and (roman(s.cprev2) or u.is_wspace(s.cprev2)))
        if not preceding:
            return
        if not u.is_wdigit(s.cprev) or u.is_wlower(s.c_next) or s.c_next == 45:
            s.is_end_clause = False

    def _abbreviation(self, tr, s):
        if s.c1 == 44 and s.cprev == 46 and (tr.translator_name == 26741) and u.is_wdigit(s.cprev2) and (u.is_wdigit(s.c_next) or u.is_wlower(s.c_next)):
            s.c1 = CHAR_COMMA_BREAK
            s.is_end_clause = False
        if s.c1 == 46 and s.c_next == 39 and (self.peekc() == 115):
            s.is_end_clause = False
        if s.c1 == 46:
            self._ordinal_dot(tr, s)
            if u.is_wlower(s.c_next) and (not tr.langopts.lowercase_sentence):
                s.is_end_clause = False
            if not s.any_alnum:
                s.c1 = 32
                s.is_end_clause = False
        elif not s.any_alnum:
            s.is_end_clause = False

    def _append(self, tr, s):
        if self.speech_parameters[0] == 1:
            return None
        if s.c1 == s.announced_punctuation:
            if is_bracket(s.c1):
                s.c1 = 57384
            elif s.c1 != 45:
                s.c1 = 32
        old = s.ix
        if s.c1 == 57404:
            s.c1 = 60
        self._write(s, u.utf8_out(s.c1))
        if not u.is_wspace(s.c1) and (not is_bracket(s.c1)):
            s.charix[s.ix] = (self.count_characters - self.clause_start_char + 32768) % 65536 - 32768
            for j in range(old + 1, s.ix):
                s.charix[j] = -1
        s.charix_top = s.ix
        if s.ix > s.n_buf - 75 and (not u.is_alpha(s.c1)) and (not u.is_wdigit(s.c1)) or s.ix >= s.n_buf - 4:
            self._terminate(s, unget=s.c2)
            return CLAUSE_NONE
        return None

    def _step(self, tr, s):
        if not u.is_walnum(s.c1):
            if self.end_character_position > 0 and self.count_characters > self.end_character_position:
                return CLAUSE_EOF
            if self.skip_characters > 0 and self.count_characters >= self.skip_characters:
                self.clear_skipping_text = True
                self.skip_characters = 0
                self.ungetc(s.c2)
                return CLAUSE_NONE
        s.cprev2 = s.cprev
        s.cprev = s.c1
        s.c1 = s.c2
        s.c2 = 32 if self.eof() else self.getc()
        self.ungot_char2 = 0
        if self.ignore_text:
            return 'continue'
        if s.c2 == 10 and self.option_linelength == -1:
            terminator = clause_type_from_codepoint(s.c1)
            if terminator == CLAUSE_NONE:
                s.charix[s.ix] = (self.count_characters - self.clause_start_char + 32768) % 65536 - 32768
                s.charix_top = s.ix
                self._write(s, u.utf8_out(s.c1))
                terminator = CLAUSE_PERIOD
            self._terminate(s)
            return terminator
        result = self._embedded(tr, s)
        if result is not None:
            return result
        s.linelength += 1
        ignored, s.c1 = self._replace(tr, s.c1)
        if ignored:
            return 'continue'
        self._case(tr, s)
        result = self._newlines(tr, s)
        if result is not None:
            return result
        s.announced_punctuation = 0
        if s.phoneme_mode == 0 and self.sayas_mode == 0:
            result = self._punctuation(tr, s)
            if result is not None:
                return result
        return self._append(tr, s)

    def read(self, tr, n_buf=800):
        if self.option_ssml:
            raise ValueError('The qualified plain-text Piper reader does not enable SSML synthesis')
        if n_buf < 80:
            raise ValueError('Piper clause buffer must allow the original75-byte margin')
        if self.clear_skipping_text:
            self.skipping_text = False
            self.clear_skipping_text = False
        self.clause_start_char = max(self.count_characters, 0)
        tr.phonemes_repeat_count = tr.clause_upper_count = tr.clause_lower_count = 0
        s = SimpleNamespace(buffer=bytearray(n_buf + 40), charix=[0] * (n_buf + 4), charix_top=0, tone=0, voice_change=b'', ix=0, n_buf=n_buf, c1=32, cprev=32, c_next=0, linelength=0, phoneme_mode=0, any_alnum=False, stressed_word=False)
        s.c2 = self.ungot_char2 if self.ungot_char2 else 0 if self.eof() else self.getc()
        terminator = None
        while not self.eof() or self.ungot_char or self.ungot_char2:
            result = self._step(tr, s)
            if result == 'continue':
                continue
            if result is not None:
                terminator = result
                break
        if terminator is None:
            if s.stressed_word:
                self._write(s, u.utf8_out(CHAR_EMPHASIS))
            self._terminate(s)
            terminator = CLAUSE_EOF
        cursor = None if self.position == len(self.text) else self.position
        return ClauseResult(bytes(s.buffer), tuple(s.charix), s.charix_top, terminator, s.tone, s.voice_change, cursor)
