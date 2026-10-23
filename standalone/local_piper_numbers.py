"""Original eSpeak number, ordinal, Roman and individual-letter algorithms.

Adapted from eSpeakNG numbers.c, Copyright2005-2015 Jonathan Duddington and
Copyright2015-2016,2020 Reece H.Dunn, GPL-3.0-or-later. Full source provenance
and notices: local-piper-numbers-provenance.json and licenses/local-piper-numbers-notices.txt.
Dictionary, stress, Unicode and translator state are provided by the qualified
owned eSpeak controller; this module contains no cached phrase translation.
"""
from types import SimpleNamespace
from .local_piper_numbers_tables import *
from standalone.local_piper_phoneme_unicode import utf8_in as utf8_at

def digit(value):
    return 48 <= value <= 57

def language(name):
    value = 0
    for c in name.encode('ascii'):
        value = (value << 8) + c
    return value

def integer_at(buffer, cursor):
    start = cursor
    while buffer[cursor] in (9, 10, 11, 12, 13, 32):
        cursor += 1
    sign = -1 if buffer[cursor] == 45 else 1
    if buffer[cursor] in (43, 45):
        cursor += 1
    digits = cursor
    value = 0
    while digit(buffer[cursor]):
        value = value * 10 + buffer[cursor] - 48
        cursor += 1
    return (value * sign, cursor) if cursor > digits else (0, start)

class NumberTranslator:

    def __init__(self, *, option_sayas=0, translator=None):
        self.option_sayas = option_sayas
        self.translator = translator
        self.n_digit_lookup = 0
        self.digit_lookup = b''
        self.speak_missing_thousands = 0
        self.number_control = 0
        self.ph_ordinal2 = b''
        self.ph_ordinal2x = b''
        self.dictionary_skipwords = 0

    @staticmethod
    def is_superscript(letter):
        for index in range(0, len(derived_letters), 2):
            c = derived_letters[index]
            if c == 0 or c > letter:
                break
            if c == letter:
                return derived_letters[index + 1]
        return 0

    @staticmethod
    def set_spelling_stress(tr, phones, control, n_chars):
        count_stress = 0
        prev = 0
        for c in phones:
            if c == phonSTRESS_P and prev != phonSWITCH:
                count_stress += 1
            prev = c
        result = bytearray()
        count = prev = 0
        for ix, c in enumerate(phones):
            if c == phonSTRESS_P and n_chars > 1 and (prev != phonSWITCH):
                count += 1
                if tr.langopts.spelling_stress:
                    if count > 1:
                        c = phonSTRESS_3
                elif count != count_stress and (count % 3 or count == count_stress - 1):
                    c = phonSTRESS_3
            elif c == 255:
                if control < 2 or ix == 0:
                    continue
                c = phonPAUSE_NOLINK if count % 3 == 0 or control > 2 else phonPAUSE_VSHORT
            result.append(c)
            prev = c
        if control >= 2:
            result.append(phonPAUSE_NOLINK)
        return bytes(result)

    @staticmethod
    def lookup_letter2(tr, letter):
        text = chr(letter).encode('utf8') + b' '
        found, phones = tr.lookup(b'_' + text)
        if not found:
            found, phones = tr.lookup(text)
            if not found:
                phones = tr.translate_rules(text, max_phonemes=20, dict_flags=None, word_flags=0, initial=phones).phonemes
        return phones

    def lookup_accented_letter(self, tr, letter, phones=b''):
        data = 0
        accent1 = accent2 = letter2 = 0
        ph_accent2 = b''
        if 224 <= letter < 383:
            data = letter_accents_0e0[letter - 224]
        elif 592 <= letter <= 680:
            data = letter_accents_250[letter - 592]
        if not data:
            return phones
        basic = (data & 63) + 59
        if basic < 97:
            basic = non_ascii_tab[basic - 59]
        if data & 32768:
            letter2 = (data >> 6 & 63) + 59
            accent2 = data >> 12 & 7
        else:
            accent1 = data >> 6 & 31
            accent2 = data >> 11 & 15
        if not accent1 and (not data & 32768):
            return phones
        flags1, ph_accent1 = tr.lookup(accents_tab[accent1][0].encode())
        if not flags1:
            return phones
        ph_letter1 = self.lookup_letter2(tr, basic)
        if not ph_letter1:
            return phones
        prefix = b''
        if accent2:
            flags2, ph_accent2 = tr.lookup(accents_tab[accent2][0].encode())
            if flags2 & FLAG_ACCENT_BEFORE:
                prefix = ph_accent2
                ph_accent2 = b''
        if letter2:
            ph_letter2 = self.lookup_letter2(tr, letter2)
            return prefix + ph_accent1 + bytes((phonPAUSE_VSHORT,)) + ph_letter1 + bytes((phonSTRESS_P,)) + ph_letter2 + ph_accent2
        if not accent1:
            return prefix + ph_letter1
        if tr.langopts.accents & 1 or flags1 & FLAG_ACCENT_BEFORE or accents_tab[accent1][1] & 1:
            return prefix + ph_accent1 + bytes((phonPAUSE_VSHORT, phonSTRESS_P)) + ph_letter1
        return prefix + bytes((phonSTRESS_2,)) + ph_letter1 + bytes((phonPAUSE_VSHORT,)) + ph_accent1 + bytes((phonPAUSE_VSHORT,))

    def lookup_letter(self, tr, letter, next_byte, control=0):
        text = chr(letter).encode('utf8') + b' '
        if next_byte == -1:
            found, phones = tr.lookup(text)
            if found:
                return phones
            found, _ = tr.lookup(b'_' + text)
            if found or tr.translator_name == language('en'):
                return phones
            found, _ = tr.lookup_english(text)
            return bytes((phonSWITCH,)) if found else phones
        if letter <= 32 or tr.is_wspace(letter):
            return tr.lookup(f'_#{letter} '.encode())[1]
        text += bytes((32 if next_byte == 32 else RULE_SPELLING,))
        found, phones = tr.lookup(b'_' + text)
        if not found:
            found, phones = tr.lookup(text)
            if not found:
                phones = tr.translate_rules(text, max_phonemes=N_PHONEME_BYTES, dict_flags=None, word_flags=FLAG_NO_TRACE, initial=phones).phonemes
        if not phones:
            phones = self.lookup_accented_letter(tr, letter, phones)
        if not phones or phones[0] == phonSWITCH:
            return phones
        return tr.set_word_stress(phones, [0, 0], -1, control & 1)

    def m_variant(self, tr, value):
        teens = 10 < value % 100 < 20
        variant = (tr if self.translator is None else self.translator).langopts.numbers2 & NUM2_THOUSANDS_VAR_BITS
        if variant == NUM2_THOUSANDS_VAR1:
            if not teens:
                if value % 10 == 1:
                    return '1MA'
                if 2 <= value % 10 <= 4:
                    return '0MA'
        elif variant == NUM2_THOUSANDS_VAR2:
            if 2 <= value <= 4:
                return '0MA'
        elif variant == NUM2_THOUSANDS_VAR3:
            if not teens and 2 <= value % 10 <= 4:
                return '0MA'
        elif variant == NUM2_THOUSANDS_VAR4:
            if teens or value % 10 == 0:
                return '0MB'
            if value % 10 == 1:
                return '0MA'
        elif variant == NUM2_THOUSANDS_VAR5:
            if not teens:
                if value % 10 == 1:
                    return '1M'
                if 2 <= value % 10 <= 4:
                    return '0MA'
        return '0M'

    def lookup_thousands(self, tr, value, thousandplex, thousands_exact):
        s = SimpleNamespace(value=value, thousandplex=thousandplex, thousands_exact=thousands_exact)
        s.found_value = 0
        s.ph_of = b''
        s.ph_thousands = b''
        if s.value > 0:
            if s.thousands_exact & 1:
                if s.thousands_exact & 2:
                    s.found_value, s.ph_thousands = tr.lookup(f'_{s.value}M{s.thousandplex}o'.encode())
                if not s.found_value and self.number_control & 1:
                    s.found_value, s.ph_thousands = tr.lookup(f'_{s.value}M{s.thousandplex}e'.encode())
                if not s.found_value:
                    s.found_value, s.ph_thousands = tr.lookup(f'_{s.value}M{s.thousandplex}x'.encode())
            if not s.found_value:
                s.found_value, s.ph_thousands = tr.lookup(f'_{s.value}M{s.thousandplex}'.encode())
        self._thousands_fallback(tr, s)
        if s.value == 1 and s.thousandplex == 1 and tr.langopts.numbers & NUM_OMIT_1_THOUSAND:
            return (1, s.ph_of + s.ph_thousands)
        return (s.found_value, s.ph_of + s.ph_thousands)

    def lookup_num2(self, tr, value, thousandplex, control):
        s = SimpleNamespace(value=value, thousandplex=thousandplex, control=control)
        s.units = s.value % 10
        s.tens = s.value // 10
        s.found = 0
        s.used_and = 0
        s.found_ordinal = 0
        s.ph_ordinal = s.ph_tens = s.ph_digits = s.ph_and = b''
        s.ord_type = 'q' if s.control & 32 else 'o'
        s.is_ordinal = s.control & 1
        s.n = tr.langopts.numbers
        s.n2 = tr.langopts.numbers2
        if s.control & 2 and self.n_digit_lookup == 2:
            s.ph_out = self.digit_lookup
        else:
            self._pair_forms(tr, s)
            self._pair_leading(tr, s)
            self._pair_ordinal(tr, s)
            self._pair_combine(tr, s)
        if s.n & NUM_SINGLE_STRESS_L:
            s.found = False
            s.out = bytearray(s.ph_out)
            for s.ix, s.c in enumerate(s.out):
                if s.c == phonSTRESS_P:
                    if s.found:
                        s.out[s.ix] = phonSTRESS_3
                    else:
                        s.found = True
            s.ph_out = bytes(s.out)
        elif s.n & NUM_SINGLE_STRESS:
            s.found = False
            s.out = bytearray(s.ph_out)
            for s.ix in reversed(range(len(s.out))):
                if s.out[s.ix] == phonSTRESS_P:
                    if s.found:
                        s.out[s.ix] = phonSTRESS_3
                    else:
                        s.found = True
            s.ph_out = bytes(s.out)
        return (s.used_and, s.ph_out)

    def lookup_num3(self, tr, value, suppress_null, thousandplex, control):
        s = SimpleNamespace(value=value, suppress_null=suppress_null, thousandplex=thousandplex, control=control)
        s.ordinal = s.control & 34
        s.hundreds = s.value // 100
        s.tensunits = s.value % 100
        s.buf1 = s.ph_thousands = s.ph_thousand_and = b''
        s.n = tr.langopts.numbers
        s.n2 = tr.langopts.numbers2
        s.say_zero = bool(s.n & NUM_ZERO_HUNDRED and (s.control & 1 or s.hundreds >= 10))
        self._num3_hundreds(tr, s)
        s.ph_hundred_and = b''
        self._num3_and(tr, s)
        s.buf2 = b''
        self._num3_remainder(tr, s)
        return s.buf1 + s.ph_hundred_and + bytes((phonEND_WORD,)) + s.buf2

    @staticmethod
    def hu_number_e(buffer, cursor, thousandplex, value):
        if buffer[cursor] in (97, 101):
            if buffer[cursor + 1] in (32, 122) or (buffer[cursor + 1] == 116 and buffer[cursor + 2] == 116):
                return False
            if (thousandplex == 1 or value % 1000 == 0) and buffer[cursor + 1] == 108:
                return False
            return True
        return False

    @staticmethod
    def check_thousands_group(buffer, cursor, length):
        return all((digit(buffer[cursor + i]) for i in range(length))) and (not digit(buffer[cursor + length])) and (not digit(buffer[cursor - 1]))

    @staticmethod
    def check_dot_ordinal(tr, buffer, start, end, wtab, roman):
        s = SimpleNamespace(buffer=buffer, start=start, end=end, wtab=wtab, roman=roman)
        s.ordinal = 0
        if tr.langopts.numbers & NUM_ORDINAL_DOT and (s.buffer[s.end] == 46 or s.wtab[0].flags & FLAG_HAS_DOT) and (not s.wtab[1].flags & FLAG_NOSPACE):
            if s.roman or not s.wtab[1].flags & FLAG_FIRST_UPPER:
                s.c2, _ = utf8_at(s.buffer, s.end + 2 if s.buffer[s.end] == 46 else s.end)
                if s.buffer[s.end] and s.buffer[s.end + 1] and (s.c2 == 0 or s.wtab[0].flags & FLAG_COMMA_AFTER or tr.is_alpha(s.c2)):
                    s.ordinal = 2
                    if s.buffer[s.end] == 46:
                        s.buffer[s.end] = 32
                    NumberTranslator._ordinal_hungarian(tr, s)
        return s.ordinal

    def translate_roman(self, tr, buffer, cursor, wtab, *, max_output=200):
        s = SimpleNamespace(buffer=buffer, cursor=cursor, wtab=wtab, max_output=max_output)
        s.acc = s.prev = s.repeat = 0
        s.subtract = 32767
        s.num_control = 0
        s.start = s.cursor
        s.n = tr.langopts.numbers
        _early = self._roman_candidate(tr, s)
        if _early is not None:
            return _early
        _early = self._roman_accumulate(tr, s)
        if _early is not None:
            return _early
        if digit(s.buffer[s.cursor]):
            return (0, b'')
        s.acc += s.prev
        if not tr.langopts.min_roman <= s.acc <= tr.langopts.max_roman:
            return (0, b'')
        _, s.ph_roman = tr.lookup(b'_roman')
        s.prefix = b'' if s.n & NUM_ROMAN_AFTER else s.ph_roman
        s.suffix = tr.langopts.roman_suffix or b''
        if isinstance(s.suffix, str):
            s.suffix = s.suffix.encode()
        s.number = bytearray(b'  ' + str(s.acc).encode() + b' ' + s.suffix + b'    \x00\x00\x00')
        if s.buffer[s.cursor] == 46:
            return (0, s.prefix)
        if self.check_dot_ordinal(tr, s.buffer, s.start, s.cursor, s.wtab, True):
            s.wtab[0].flags |= FLAG_ORDINAL
        if s.n & NUM_ROMAN_ORDINAL:
            if tr.translator_name == language('hu'):
                if not s.wtab[0].flags & FLAG_ORDINAL:
                    if s.wtab[0].flags & FLAG_HYPHEN_AFTER and self.hu_number_e(s.buffer, s.cursor, 0, s.acc):
                        s.num_control |= 1
                    else:
                        return (0, s.prefix)
            else:
                s.wtab[0].flags |= FLAG_ORDINAL
        tr.prev_dict_flags[:] = [0, 0]
        s.flags = [0, 0]
        _, s.phones = self.translate_number(tr, s.number, 2, s.flags, s.wtab, s.num_control, max_output=s.max_output - len(s.ph_roman))
        return (1, s.prefix + s.phones + (s.ph_roman if s.n & NUM_ROMAN_AFTER else b''))

    def translate_number(self, tr, buffer, cursor, flags, wtab, control=0, *, max_output=200):
        s = SimpleNamespace(buffer=buffer, cursor=cursor, flags=flags, wtab=wtab, control=control, max_output=max_output)
        _early = self._number_prefix(tr, s)
        if _early is not None:
            return _early
        self._number_previous(tr, s)
        self._number_ordinal(tr, s)
        _early = self._number_leading_zeros(tr, s)
        if _early is not None:
            return _early
        self._number_groups(tr, s)
        self._number_thousands(tr, s)
        _early = self._number_dictionary(tr, s)
        if _early is not None:
            return _early
        s.phones = self.lookup_num3(tr, s.value, s.suppress_null, s.thousandplex, s.prev_thousands | s.ordinal | s.decimal_point)
        s.phones = s.prefix + s.ph_zeros + (s.ph_append + bytes((phonEND_WORD,)) + s.phones if s.thousandplex > 0 and s.n2 & NUM2_SWAP_THOUSANDS else s.phones + bytes((phonEND_WORD,)) + s.ph_append)
        if s.decimal_point:
            s.phones, s.n_digits = self._decimal(tr, s.buffer, s.cursor, s.n_digits, s.phones, s.max_output)
        if s.phones and s.phones[0] != phonSWITCH:
            s.next_char, s.length = utf8_at(s.buffer, s.cursor + s.n_digits + 1)
            if s.n & NUM_NOPAUSE and s.next_char == 32:
                s.next_char, _ = utf8_at(s.buffer, s.cursor + s.n_digits + 1 + s.length)
            if not tr.is_walpha(s.next_char) and (not s.thousands_exact):
                s.phones = s.phones[:s.max_output - 2] + bytes((phonPAUSE_NOLINK,))
        s.flags[0] |= FLAG_FOUND
        self.speak_missing_thousands -= 1
        if s.skipwords:
            self.dictionary_skipwords = s.skipwords
        return (1, s.phones)

    def _decimal(self, tr, buffer, cursor, n_digits, phones, max_output):
        s = SimpleNamespace(buffer=buffer, cursor=cursor, n_digits=n_digits, phones=phones, max_output=max_output)
        s.c = 0
        while True:
            s.n_digits += 1
            s.count = 0
            while digit(s.buffer[s.cursor + s.n_digits + s.count]):
                s.count += 1
            s.mode = tr.langopts.numbers & NUM_DFRACTION_BITS
            s.max_count = 5 if s.mode == NUM_DFRACTION_4 else 2
            _early = self._decimal_fraction(tr, s)
            if _early is not None:
                return _early
            while True:
                s.c = s.buffer[s.cursor + s.n_digits]
                if not digit(s.c) or len(s.phones) >= N_WORD_PHONEMES - 10:
                    break
                s.value = s.c - 48
                s.n_digits += 1
                _, s.part = self.lookup_num2(tr, s.value, 0, 2)
                if len(s.phones) + len(s.part) + 2 > s.max_output:
                    return (s.phones, s.n_digits)
                s.phones += bytes((phonEND_WORD,)) + s.part
            s.found, s.part = tr.lookup(b'_dpt2')
            if s.found:
                if len(s.phones) + len(s.part) + 1 > s.max_output:
                    return (s.phones, s.n_digits)
                s.phones += s.part
            if s.c == tr.langopts.decimal_sep and digit(s.buffer[s.cursor + s.n_digits + 1]):
                _, s.part = tr.lookup(b'_dpt')
                if len(s.phones) + len(s.part) + 1 > s.max_output:
                    return (s.phones, s.n_digits)
                s.phones += s.part
            else:
                return (s.phones, s.n_digits)

    def _thousands_fallback(self, tr, s):
        if not s.found_value:
            if s.value % 100 >= 20:
                _, s.ph_of = tr.lookup(b'_0of')
            s.found = 0
            if s.thousands_exact & 1:
                if s.thousands_exact & 2:
                    s.found, s.ph_thousands = tr.lookup(f'_{self.m_variant(tr, s.value)}{s.thousandplex}o'.encode())
                if not s.found and self.number_control & 1:
                    s.found, s.ph_thousands = tr.lookup(f'_{self.m_variant(tr, s.value)}{s.thousandplex}e'.encode())
                if not s.found:
                    s.found, s.ph_thousands = tr.lookup(f'_{self.m_variant(tr, s.value)}{s.thousandplex}x'.encode())
            if not s.found:
                s.found, s.ph_thousands = tr.lookup(f'_{self.m_variant(tr, s.value)}{s.thousandplex}'.encode())
                if not s.found:
                    if s.thousandplex > 3:
                        s.previous, _ = tr.lookup(f'_0M{s.thousandplex - 1}'.encode())
                        if not s.previous:
                            _, s.ph_thousands = tr.lookup(b'_0M2')
                            self.speak_missing_thousands = 3
                    if not s.ph_thousands:
                        s.found_value, s.ph_thousands = tr.lookup(f'_{s.value}M1'.encode())
                        if not s.found_value:
                            _, s.ph_thousands = tr.lookup(b'_0M1')
                        self.speak_missing_thousands = 2

    def _pair_units(self, tr, s):
        if s.units > 0:
            s.found = 0
            if s.control & 2 and self.digit_lookup:
                s.ph_digits = self.digit_lookup
                s.found_ordinal = 1
                s.ph_ordinal = b''
            else:
                if s.control & 8:
                    s.found, s.ph_digits = tr.lookup(f'_{s.units}f'.encode())
                if s.is_ordinal and (not s.n & NUM_SWAP_TENS):
                    s.found, s.ph_digits = tr.lookup(f'_{s.units}{s.ord_type}'.encode())
                    if s.found:
                        s.found_ordinal = 1
                if not s.found:
                    if self.number_control & 1 and s.control & 2:
                        s.found, s.ph_digits = tr.lookup(f'_{s.units}e'.encode())
                    elif not s.control & 2 or s.n & NUM_SWAP_TENS:
                        s.found, s.ph_digits = tr.lookup((f'_{s.units}' + ('o' if s.n2 & NUM2_ORDINAL_AND_THOUSANDS and s.thousandplex <= 1 else 'a')).encode())
                if not s.found:
                    _, s.ph_digits = tr.lookup(f'_{s.units}'.encode())

    def _pair_tens(self, tr, s):
        if s.is_ordinal:
            s.ordinal_found, s.ph_tens = tr.lookup(f'_{s.tens}X{s.ord_type}'.encode())
            if s.ordinal_found:
                s.found_ordinal = 1
                if s.units and s.n2 & NUM2_MULTIPLE_ORDINAL:
                    s.ph_tens += self.ph_ordinal2
        if not s.found_ordinal:
            _, s.ph_tens = tr.lookup((f'_{s.tens}X' + ('f' if s.control & 512 else '')).encode())
        if not s.ph_tens and s.n & NUM_VIGESIMAL:
            s.units = s.value % 20
            _, s.ph_tens = tr.lookup(f'_{s.tens & 254}X'.encode())
        s.ph_digits = b''
        self._pair_units(tr, s)

    def _pair_combine(self, tr, s):
        if s.n & (NUM_SWAP_TENS | NUM_AND_UNITS) and s.ph_tens and s.ph_digits:
            _, s.ph_and = tr.lookup(b'_0and')
            if s.is_ordinal and s.n2 & NUM2_ORDINAL_NO_AND:
                s.ph_and = b''
            s.ph_out = (s.ph_digits + s.ph_and + s.ph_tens if s.n & NUM_SWAP_TENS else s.ph_tens + s.ph_and + s.ph_digits) + s.ph_ordinal
            s.used_and = 1
        else:
            if s.n & NUM_SINGLE_VOWEL and s.ph_tens and s.ph_digits:
                s.nexttype = tr.phoneme_tab[s.ph_digits[0]].type
                if s.nexttype == phSTRESS:
                    s.nexttype = tr.phoneme_tab[s.ph_digits[1]].type
                if tr.phoneme_tab[s.ph_tens[-1]].type == phVOWEL and s.nexttype == phVOWEL:
                    s.ph_tens = s.ph_tens[:-1]
            s.ph_out = s.ph_tens + s.ph_digits
            if s.n2 & NUM2_ORDINAL_DROP_VOWEL and s.ph_ordinal and s.ph_out and (tr.phoneme_tab[s.ph_out[-1]].type == phVOWEL):
                s.ph_out = s.ph_out[:-1]
            s.ph_out += s.ph_ordinal

    def _pair_ordinal(self, tr, s):
        if s.is_ordinal and (not s.found_ordinal) and (not s.ph_ordinal):
            if s.value >= 20 and (s.value % 10 == 0 or s.n & NUM_SWAP_TENS):
                _, s.ph_ordinal = tr.lookup(b'_ord20')
            if not s.ph_ordinal:
                _, s.ph_ordinal = tr.lookup(b'_ord')

    def _pair_leading(self, tr, s):
        if s.value < 10 and s.control & 16:
            _, s.ph_tens = tr.lookup(b'_0')
        elif not s.found:
            self._pair_tens(tr, s)

    def _pair_forms(self, tr, s):
        if not self.digit_lookup:
            if s.control & 8:
                s.found, s.ph_digits = tr.lookup(f'_{s.value}fx'.encode())
                if not s.found:
                    s.found, s.ph_digits = tr.lookup(f'_{s.value}f'.encode())
            elif s.is_ordinal:
                s.ph_ordinal = self.ph_ordinal2
                if s.control & 4:
                    s.found, s.ph_digits = tr.lookup(f'_{s.value}{s.ord_type}x'.encode())
                    if s.found and self.ph_ordinal2x:
                        s.ph_ordinal = self.ph_ordinal2x
                if not s.found:
                    s.found, s.ph_digits = tr.lookup(f'_{s.value}{s.ord_type}'.encode())
                s.found_ordinal = s.found
            if not s.found:
                if s.control & 2:
                    if self.number_control & 1:
                        s.found, s.ph_digits = tr.lookup(f'_{s.value}e'.encode())
                else:
                    s.found, s.ph_digits = tr.lookup((f'_{s.value}' + ('o' if s.n2 & NUM2_ORDINAL_AND_THOUSANDS and s.thousandplex <= 1 else 'a')).encode())
                if not s.found and (not (s.is_ordinal and s.n2 & NUM2_NO_TEEN_ORDINALS)):
                    s.found, s.ph_digits = tr.lookup(f'_{s.value}'.encode())

    def _hundred_exact(self, tr, s):
        if not s.hundreds and s.say_zero:
            _, s.ph_digits = tr.lookup(b'_0')
        else:
            if not (s.hundreds == 1 and s.n2 & NUM2_OMIT_1_HUNDRED_ONLY and (not s.control & 1)):
                if not s.found and (not s.tensunits):
                    s.found, s.ph_digits = tr.lookup(f'_{s.hundreds}C0'.encode())
                if not s.found:
                    s.found, s.ph_digits = tr.lookup(f'_{s.hundreds}C'.encode())
            if s.found:
                s.ph_100 = b''
            elif not (s.hundreds == 1 and s.n & NUM_OMIT_1_HUNDRED):
                _, s.ph_digits = self.lookup_num2(tr, s.hundreds, s.thousandplex, 0)

    def _hundred_digits(self, tr, s):
        if s.hundreds > 0 or s.say_zero:
            if s.n & NUM_AND_HUNDRED and (s.control & 1 or s.ph_thousands):
                _, s.ph_thousand_and = tr.lookup(b'_0and')
            s.suppress_null = True
            s.found = 0
            if s.ordinal and (s.tensunits == 0 or s.n2 & NUM2_MULTIPLE_ORDINAL):
                s.found, s.ph_digits = tr.lookup(f'_{s.hundreds}Co'.encode())
                if s.n2 & NUM2_MULTIPLE_ORDINAL and s.tensunits > 0:
                    s.ph_digits += self.ph_ordinal2
            self._hundred_exact(tr, s)

    def _hundreds_prepare(self, tr, s):
        s.found = 0
        s.ph_100 = b''
        if s.ordinal and s.tensunits == 0:
            s.found, s.ph_100 = tr.lookup(b'_0Co')
        if not s.found:
            if not s.tensunits:
                s.found, s.ph_100 = tr.lookup(b'_0C0')
            if not s.found:
                _, s.ph_100 = tr.lookup(b'_0C')
        if s.n & NUM_1900 and s.hundreds == 19:
            pass
        elif s.hundreds >= 10:
            s.ph_digits = b''
            s.exact = int(s.value % 1000 == 0)
            s.tplex = 0 if s.n2 & NUM2_MYRIADS else s.thousandplex + 1
            s.found, s.ph_10t = self.lookup_thousands(tr, s.hundreds // 10, s.tplex, s.exact | s.ordinal)
            if not s.found:
                s.x = 8 if s.n2 & 1 << s.tplex and s.tplex <= 3 else 0
                if tr.translator_name == language('ml'):
                    s.x = 520
                _, s.ph_digits = self.lookup_num2(tr, s.hundreds // 10, s.thousandplex, s.x)
            s.ph_thousands = (s.ph_10t + bytes((phonEND_WORD,)) + s.ph_digits if s.n2 & NUM2_SWAP_THOUSANDS else s.ph_digits + bytes((phonEND_WORD,)) + s.ph_10t) + bytes((phonEND_WORD,))
            s.hundreds %= 10
            if not s.hundreds and (not s.say_zero):
                s.ph_100 = b''
            s.suppress_null = True
            s.control |= 1

    def _num3_remainder(self, tr, s):
        if s.tensunits or not s.suppress_null:
            s.x = 0
            if s.thousandplex == 0:
                s.x = 3 if s.ordinal else 2
                if s.value < 100 and (not s.control & 1):
                    s.x |= 4
                if s.ordinal & 32:
                    s.x |= 32
            elif s.n2 & 1 << s.thousandplex and s.thousandplex <= 3:
                s.x = 8
            if tr.translator_name == language('ml') and s.thousandplex == 1:
                s.x |= 520
            if s.n2 & NUM2_ZERO_TENS and (s.control & 1 or s.hundreds > 0):
                s.x |= 16
            s.used_and, s.buf2 = self.lookup_num2(tr, s.tensunits, s.thousandplex, s.x | s.control & 256)
            if s.used_and and s.n & NUM_SINGLE_AND:
                s.ph_hundred_and = b''
        elif self.ph_ordinal2:
            if s.buf1 and s.buf1[-1] == phonPAUSE_SHORT:
                s.buf1 = s.buf1[:-1]
            s.buf2 = self.ph_ordinal2

    def _num3_and(self, tr, s):
        if s.tensunits > 0 and (not (s.control & 2 and s.n2 & NUM2_MULTIPLE_ORDINAL)):
            if s.value > 100 or (s.control & 1 and s.thousandplex == 0):
                if s.n & NUM_HUNDRED_AND or (s.n & NUM_HUNDRED_AND_DIGIT and s.tensunits < 10):
                    _, s.ph_hundred_and = tr.lookup(b'_0and')
            if s.n & NUM_THOUSAND_AND and s.hundreds == 0 and (s.control & 1 or s.ph_thousands):
                _, s.ph_hundred_and = tr.lookup(b'_0and')

    def _num3_hundreds(self, tr, s):
        if s.hundreds > 0 or s.say_zero:
            self._hundreds_prepare(tr, s)
            s.ph_digits = b''
            self._hundred_digits(tr, s)
            s.buf1 = s.ph_thousands + s.ph_thousand_and + s.ph_digits + s.ph_100

    @staticmethod
    def _ordinal_hungarian(tr, s):
        if not s.roman and tr.translator_name == language('hu'):
            s.nextflags = tr.translate_word(bytes(s.buffer[s.end + 2:]).split(b'\x00', 1)[0])[0] if tr.is_alpha(s.c2) else 0
            if tr.prev_dict_flags[0] & FLAG_ALT_TRANS and (s.c2 == 0 or s.wtab[0].flags & FLAG_COMMA_AFTER or tr.is_wdigit(s.c2)):
                s.ordinal = 0
            if s.nextflags & FLAG_ALT_TRANS:
                s.ordinal = 0
            if s.nextflags & FLAG_ALT3_TRANS:
                if s.buffer[s.start - 2] == 45:
                    s.ordinal = 0
                if tr.prev_dict_flags[0] & (FLAG_ALT_TRANS | FLAG_ALT3_TRANS):
                    s.ordinal = 34

    def _roman_accumulate(self, tr, s):
        while True:
            s.c = s.buffer[s.cursor]
            s.cursor += 1
            if s.c in (0, 32):
                break
            if s.c not in s.roman:
                return (0, b'')
            s.value = s.roman[s.c]
            if s.value == s.prev:
                s.repeat += 1
                if s.repeat >= 3:
                    return (0, b'')
            else:
                s.repeat = 0
            if s.prev > 1 and s.prev not in (10, 100) and (s.value >= s.prev):
                return (0, b'')
            if s.prev and s.prev < s.value:
                if s.acc % 10 or s.prev * 10 < s.value:
                    return (0, b'')
                s.subtract = s.prev
                s.value -= s.subtract
            elif s.value >= s.subtract:
                return (0, b'')
            else:
                s.acc += s.prev
            s.prev = s.value

    def _roman_candidate(self, tr, s):
        if s.n & NUM_ROMAN_CAPITALS and (not s.wtab[0].flags & FLAG_ALL_UPPER) or digit(s.buffer[s.cursor - 2]):
            return (0, b'')
        if s.buffer[s.cursor + 1] == 32 and (not (s.n & (NUM_ROMAN_CAPITALS | NUM_ROMAN_ORDINAL | NUM_ORDINAL_DOT) and s.wtab[0].flags & FLAG_HAS_DOT)):
            return (0, b'')
        s.roman = dict(zip(b'ixcmvld', (1, 10, 100, 1000, 5, 50, 500)))

    def _number_dictionary(self, tr, s):
        if not s.thousandplex:
            s.p = s.cursor
            while digit(s.buffer[s.p + 1]):
                s.p += 1
            if digit(s.buffer[s.p - 1]):
                s.result = tr.lookup_dict_list(s.buffer, s.p - 1, flags=tuple(s.flags), end_flags=FLAG_SUFX, word_table=s.wtab)
                s.flags[:] = s.result.flags
                self.digit_lookup = s.result.phonemes
                if s.result.found:
                    self.n_digit_lookup = 2
            if not self.digit_lookup and s.buffer[s.p] != 48:
                s.result = tr.lookup_dict_list(s.buffer, s.p, flags=tuple(s.flags), end_flags=FLAG_SUFX, word_table=s.wtab)
                s.flags[:] = s.result.flags
                self.digit_lookup = s.result.phonemes
                if s.result.found:
                    self.n_digit_lookup = 1
            if not s.prev_thousands:
                if not s.decimal_point and (not s.ordinal):
                    s.found, s.phones = tr.lookup(f'_{s.value}n'.encode())
                    if s.found:
                        return (1, s.phones)
                if s.n2 & NUM2_PERCENT_BEFORE:
                    s.p2 = s.cursor
                    while s.buffer[s.p2] not in (0, 32):
                        s.p2 += 1
                    if s.buffer[s.p2 + 1] == 37:
                        _, s.prefix = tr.lookup(b'%')
                        s.buffer[s.p2 + 1] = 32

    def _number_thousands(self, tr, s):
        s.decimal_point = 0
        if s.buffer[s.cursor + s.n_digits] == tr.langopts.decimal_sep and digit(s.buffer[s.cursor + s.n_digits + 1]):
            _, s.ph_append = tr.lookup(b'_dpt')
            s.decimal_point = 256
        elif not s.suppress_null:
            if s.thousands_inc and s.thousandplex:
                s.found, s.ph_append = self.lookup_thousands(tr, s.value, s.thousandplex, s.thousands_exact)
                if s.found:
                    s.value = 0
                    s.suppress_null = True
        elif self.speak_missing_thousands == 1:
            s.found, _ = tr.lookup(f'_0M{s.thousandplex + 1}'.encode())
            if not s.found:
                _, s.ph_append = tr.lookup(f'_0M{s.thousandplex}'.encode())
        if not s.ph_append and s.buffer[s.cursor + s.n_digits] == 46 and (not s.thousandplex):
            _, s.ph_append = tr.lookup(b'_.')
        s.prefix = b''

    def _number_groups(self, tr, s):
        s.thousands_inc = 1 if s.n & NUM_ALLOW_SPACE and s.buffer[s.cursor + s.n_digits] == 32 else 2 if s.buffer[s.cursor + s.n_digits] == tr.langopts.thousands_sep else 0
        s.suffix_ix = s.n_digits + 2
        if s.thousands_inc:
            s.digix = s.n_digits + s.thousands_inc
            while not s.wtab[s.thousandplex + 1].flags & FLAG_MULTIPLE_SPACES and self.check_thousands_group(s.buffer, s.cursor + s.digix, s.group_len):
                if any((s.buffer[s.cursor + s.digix + i] != 48 for i in range(s.group_len))):
                    s.thousands_exact = 0
                s.thousandplex += 1
                s.digix += s.group_len
                if s.buffer[s.cursor + s.digix] == tr.langopts.thousands_sep or (s.n & NUM_ALLOW_SPACE and s.buffer[s.cursor + s.digix] == 32):
                    s.suffix_ix = s.digix + 2
                    s.digix += s.thousands_inc
                else:
                    break
        if s.value == 0 and s.prev_thousands:
            s.suppress_null = True
        if tr.translator_name == language('hu') and s.wtab[s.thousandplex].flags & FLAG_HYPHEN_AFTER and (s.thousands_exact == 1) and self.hu_number_e(s.buffer, s.cursor + s.suffix_ix, s.thousandplex, s.value):
            self.number_control |= 1

    def _number_leading_zeros(self, tr, s):
        if s.buffer[s.cursor] == 48 and (not s.prev_thousands) and (s.buffer[s.cursor + 1] not in (32, tr.langopts.decimal_sep)):
            if not (s.n_digits == 2 and s.buffer[s.cursor + 3] == 58 and digit(s.buffer[s.cursor + 5]) and (s.buffer[s.cursor + 7] in (9, 10, 11, 12, 13, 32))):
                if s.n_digits > 3:
                    s.flags[0] &= ~FLAG_SKIPWORDS
                    return (0, b'')
                s.ix = 0
                while s.buffer[s.cursor + s.ix] == 48 and s.ix < s.n_digits - 1:
                    _, s.zero = tr.lookup(b'_0')
                    s.ph_zeros += s.zero
                    s.ix += 1

    def _number_ordinal(self, tr, s):
        if s.buffer[s.cursor + s.ix] == 46 and (not digit(s.buffer[s.cursor + s.ix + 1])) and (not digit(s.buffer[s.cursor + s.ix + 2])) and (not s.wtab[1].flags & FLAG_NOSPACE):
            s.buffer[s.cursor + s.ix] = 0
        if not s.ordinal or tr.translator_name == language('hu'):
            s.ix += 1
            s.suffix = bytearray()
            if s.wtab[0].flags & FLAG_HYPHEN_AFTER:
                s.suffix.append(45)
                s.ix += 1
            while s.buffer[s.cursor + s.ix] not in (0, 32) and s.ix < 29:
                s.suffix.append(s.buffer[s.cursor + s.ix])
                s.ix += 1
            if s.suffix:
                s.indicator = tr.langopts.ordinal_indicator
                if isinstance(s.indicator, str):
                    s.indicator = s.indicator.encode()
                if s.indicator is not None and s.suffix == s.indicator:
                    s.ordinal = 2
                elif not digit(s.suffix[0]):
                    s.found, self.ph_ordinal2 = tr.lookup(b'_#' + s.suffix)
                    if s.found:
                        s.ordinal = 2
                        s.flags[0] |= FLAG_SKIPWORDS
                        s.skipwords = 1
                        _, self.ph_ordinal2x = tr.lookup(b'_x#' + s.suffix)
        if s.wtab[0].flags & FLAG_ORDINAL:
            s.ordinal = 2
        s.ph_append = b''

    def _number_previous(self, tr, s):
        s.group_len = 4 if s.n2 & NUM2_MYRIADS else 3
        s.prev_thousands = int(s.n_digits == s.group_len and s.buffer[s.cursor - 2] == tr.langopts.thousands_sep and digit(s.buffer[s.cursor - 3]))
        if not s.prev_thousands and (tr.langopts.thousands_sep == 32 or s.n & NUM_ALLOW_SPACE):
            if s.n_digits == 3 and (not s.wtab[0].flags & FLAG_MULTIPLE_SPACES) and digit(s.buffer[s.cursor - 2]):
                s.prev_thousands = 1
        if not s.prev_thousands:
            self.speak_missing_thousands = 0
        self.ph_ordinal2 = b''
        s.ph_zeros = b''
        s.ordinal = 0
        s.skipwords = 0
        s.thousandplex = 0
        s.thousands_exact = 1
        s.suppress_null = False
        if s.prev_thousands or s.buffer[s.cursor] != 48:
            s.ordinal = self.check_dot_ordinal(tr, s.buffer, s.cursor, s.cursor + s.ix, s.wtab, False)

    def _number_prefix(self, tr, s):
        if self.option_sayas == SAYAS_DIGITS1 or s.wtab[0].flags & FLAG_INDIVIDUAL_DIGITS or (not tr.langopts.numbers):
            return (0, b'')
        self.n_digit_lookup = 0
        self.digit_lookup = b''
        self.number_control = s.control
        s.flags[0] = 0
        s.n = tr.langopts.numbers
        s.n2 = tr.langopts.numbers2
        s.ix = 0
        while digit(s.buffer[s.cursor + s.ix]):
            s.ix += 1
        s.n_digits = s.ix
        s.value, s.end = integer_at(s.buffer, s.cursor)
        if s.end == s.cursor or s.value > 2147483647 or s.value < -9223372036854775808:
            return (0, b'')

    def _fraction_compact(self, tr, s):
        while s.buffer[s.cursor + s.n_digits] == 48:
            _, s.zero = tr.lookup(b'_0')
            if len(s.phones) + len(s.zero) + 1 > s.max_output:
                return (s.phones, s.n_digits)
            s.phones += s.zero
            s.count -= 1
            s.n_digits += 1
        if s.count <= s.max_count and digit(s.buffer[s.cursor + s.n_digits]):
            s.value, _ = integer_at(s.buffer, s.cursor + s.n_digits)
            s.part = self.lookup_num3(tr, s.value, False, 0, 0)
            if len(s.phones) + len(s.part) + 1 > s.max_output:
                return (s.phones, s.n_digits)
            s.phones += s.part
            s.n_digits += s.count

    def _fraction_named(self, tr, s):
        s.value, _ = integer_at(s.buffer, s.cursor + s.n_digits)
        s.part = self.lookup_num3(tr, s.value, False, 0, 0)
        s.append = True
        if s.buffer[s.cursor + s.n_digits] == 48 or s.mode != NUM_DFRACTION_1:
            s.found, s.suffix = tr.lookup(f'_0Z{s.count}'.encode())
            if not s.found:
                s.append = False
            elif s.mode == NUM_DFRACTION_6:
                if len(s.phones) + len(s.suffix) + 1 > s.max_output:
                    return (s.phones, s.n_digits)
                s.phones += s.suffix
            else:
                s.part += s.suffix
        if s.append:
            if len(s.phones) + len(s.part) + 1 > s.max_output:
                return (s.phones, s.n_digits)
            s.phones += s.part
            s.n_digits += s.count

    def _fraction_romanian(self, tr, s):
        if s.count <= 4 and s.buffer[s.cursor + s.n_digits] != 48:
            s.value, _ = integer_at(s.buffer, s.cursor + s.n_digits)
            s.part = self.lookup_num3(tr, s.value, False, 0, 0)
            if len(s.phones) + len(s.part) + 1 > s.max_output:
                return (s.phones, s.n_digits)
            s.phones += s.part
            s.n_digits += s.count

    def _fraction_digits(self, tr, s):
        while s.count > 1:
            s.count -= 1
            s.found, s.part = tr.lookup(b'_' + bytes((s.buffer[s.cursor + s.n_digits],)) + b'd')
            if not s.found:
                break
            s.n_digits += 1
            if len(s.phones) + len(s.part) + 1 > s.max_output:
                return (s.phones, s.n_digits)
            s.phones += s.part

    def _decimal_fraction(self, tr, s):
        if s.mode in (NUM_DFRACTION_2, NUM_DFRACTION_4):
            _early = self._fraction_compact(tr, s)
            if _early is not None:
                return _early
        elif s.mode in (NUM_DFRACTION_1, NUM_DFRACTION_5, NUM_DFRACTION_6):
            _early = self._fraction_named(tr, s)
            if _early is not None:
                return _early
        elif s.mode == NUM_DFRACTION_3:
            _early = self._fraction_romanian(tr, s)
            if _early is not None:
                return _early
        elif s.mode == NUM_DFRACTION_7:
            _early = self._fraction_digits(tr, s)
            if _early is not None:
                return _early
