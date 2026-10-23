"""Actual eSpeak spelling, foreign alphabet and Unicode character names."""
from standalone.local_piper_phoneme_rules import byte
from standalone.local_piper_phoneme_unicode import utf8_in, utf8_out, is_wupper, is_walpha, is_wspace, to_lower_for_voice
from standalone.local_piper_phoneme_word_helpers import non_ascii_number

def _terminated(data):
    return bytes(data).split(b'\x00', 1)[0]

class _Letter:

    def __init__(self, tr, buffer, cursor, phonemes, control, current_alphabet):
        self.tr = tr
        self.buffer = buffer
        self.cursor = cursor
        self.phonemes = phonemes
        self.control = control
        self.current_alphabet = current_alphabet

    def lookup(self):
        self.session = self.tr.session
        self.primary = self.session.primary
        self.phone_table = self.primary.phoneme_tab_ix
        self.capital = b''
        self.alphabet_phones = b''
        self.phones = b''
        self.letter, self.size = utf8_in(self.buffer, self.cursor)
        if self.letter & 1048320 == 57344:
            self.letter &= 255
        if self.control & 2 and is_wupper(self.letter):
            _, self.capital = self.tr.lookup(b'_cap')
        self.letter = to_lower_for_voice(self.letter, self.tr.langopts.dotless_i)
        self.phones = self.tr.lookup_letter(self.letter, byte(self.buffer, self.cursor + self.size), self.control & 1)
        if not self.phones:
            derived = self.session.numbers.is_superscript(self.letter)
            if derived:
                self.letter = derived & 16383
                modifier = (None, b'_sub', b'_sup', None)[derived >> 14]
                if self.control & 4 and modifier is not None:
                    _, self.capital = self.tr.lookup(modifier)
                    if not self.capital:
                        other_table, foreign = self.session.alternate('en', 3)
                        _, other = foreign.lookup(modifier)
                        if other:
                            self.capital = b'\t\x15' + bytes((other_table & 255,)) + other + b'\x15' + bytes((self.phone_table & 255,))
            self.phones = self.tr.lookup_letter(self.letter, byte(self.buffer, self.cursor + self.size), self.control & 1)

    def alphabet_name(self):
        self.number = non_ascii_number(self.letter)
        if not self.phones and self.number > 0:
            self.phones = self.tr.lookup_letter(self.number, 0, self.control & 1)
        self.alphabet = self.tr.alphabet_from_char(self.letter)
        self.offset = 0 if self.alphabet is None else self.alphabet.offset
        self.flags = 0 if self.alphabet is None else self.alphabet.flags
        if self.alphabet is not self.current_alphabet:
            if self.alphabet is not None and (not self.flags & 1) and (self.offset != self.primary.letter_bits_offset):
                if self.offset not in (self.primary.langopts.alt_alphabet, self.primary.langopts.our_alphabet):
                    found, self.alphabet_phones = self.primary.lookup(self.alphabet.name.encode())
                    other = b''
                    if not found:
                        other_table, foreign = self.session.alternate('en', 3)
                        _, other = foreign.lookup(self.alphabet.name.encode())
                    elif self.primary is not self.tr:
                        self.phone_table = self.tr.phoneme_tab_ix
                        other = self.alphabet_phones
                        other_table = self.primary.phoneme_tab_ix
                    if other:
                        self.alphabet_phones = b'\t\x15' + bytes((other_table & 255,)) + other + b'\x15' + bytes((self.phone_table & 255,))

    def foreign(self):
        language = self.primary.langopts.alt_alphabet_lang if self.offset and self.offset == self.primary.langopts.alt_alphabet else self.alphabet.language if self.alphabet is not None and self.alphabet.language and (not self.flags & 2) else 25966
        if language != self.tr.translator_name or language == 27503:
            name = (language & 4294967295).to_bytes(4, 'big').lstrip(b'\x00').decode('ascii')
            other_table, foreign = self.session.alternate(name, 3)
            if foreign is None:
                other = b''
            else:
                code = self.letter - 44032
                if code >= 0 and self.letter <= 55215:
                    initial = code // 28 // 21
                    text = b'' if initial == 11 else utf8_out(initial + 4352)
                    text += utf8_out(code // 28 % 21 + 4449) + utf8_out(code % 28 + 4519) + b' \x00'
                    other = foreign.translate_rules(text, max_phonemes=77).phonemes
                    other = foreign.set_word_stress(other, None, -1, 0)
                else:
                    other = foreign.lookup_letter(self.letter, byte(self.buffer, self.cursor + self.size), self.control & 1)
                if other[:1] == b'\x15':
                    name = other[1:].decode('ascii')
                    other_table, foreign = self.session.alternate(name, 3)
                    other = foreign.lookup_letter(self.letter, byte(self.buffer, self.cursor + self.size), self.control & 1)
                self.session.selection.select(self.session.voice.phoneme_tab_ix)
                if other:
                    self.phones = b'\t\x15' + bytes((other_table & 255,)) + other + b'\x15' + bytes((self.tr.phoneme_tab_ix & 255,))

    def unknown(self):
        if not self.flags & 16:
            if is_walpha(self.letter):
                _, self.phones = self.primary.lookup(b'_?A')
            if not self.phones and (not is_wspace(self.letter)):
                _, self.phones = self.primary.lookup(b'_??')
            if not self.phones:
                self.phones = self.tr.encode_phonemes(b"l'et@").phonemes
        if self.control & 4 or not self.flags & 8:
            digits = bytes((49 + bit for bit in range(8) if self.letter & 1 << bit)) if self.offset == 10240 else format(self.letter, 'x').encode()
            for digit in digits:
                self.phones += b'\x17'
                part = self.primary.lookup_letter(digit, 0, 1)
                if (not part or part[:1] == b'\x15') and digit >= 97:
                    part = self.tr.encode_phonemes((b"'e:j", b"b'i:", b"s'i:", b"d'i:", b"'i:", b"'ef")[digit - 97]).phonemes
                self.phones += part
            self.phones += b'\t'

    def finish(self):
        self.alphabet_phones = _terminated(self.alphabet_phones)
        self.capital = _terminated(self.capital)
        self.phones = _terminated(self.phones)
        tail = b'\xff' + self.alphabet_phones + (self.phones + self.capital if self.tr.langopts.accents & 2 else self.capital + self.phones)
        if len(self.phonemes) + len(tail) < 200:
            self.phonemes += tail
        return (self.size, _terminated(self.phonemes))

def translate_letter(tr, buffer, cursor, phonemes=b'', control=0, current_alphabet=None):
    work = _Letter(tr, buffer, cursor, phonemes, control, current_alphabet)
    work.lookup()
    if work.phones[:1] == b'\x15':
        return (0, work.phones)
    work.alphabet_name()
    if not work.phones:
        work.foreign()
    if not work.phones:
        work.unknown()
    return work.finish()

def speak_individual_letters(tr, buffer, cursor, phonemes=b'', spell_word=0, current_alphabet=None):
    count = 0
    control = 2 if spell_word > 2 else 0
    if spell_word > 1:
        control |= 4
    while byte(buffer, cursor) not in (0, 32):
        size, phonemes = translate_letter(tr, buffer, cursor, phonemes, control | int(count > 0), current_alphabet)
        cursor += size
        count += 1
        if phonemes[:1] == b'\x15':
            return (None, phonemes)
    return (cursor, tr.session.numbers.set_spelling_stress(tr, phonemes, spell_word, count))
