"""Original eSpeak word orchestration: dictionary, spelling, morphology, stress."""
from dataclasses import dataclass, field
from standalone.local_piper_phoneme_dictionary import WordTab
from standalone.local_piper_phoneme_rules import byte
from standalone.local_piper_phoneme_unicode import utf8_in, utf8_out, is_alpha, is_walpha, is_wdigit, is_wupper
from standalone.local_piper_phoneme_word_helpers import remove_ending, change_word_stress, apply_special_attribute, add_plural_suffix, check_dotted_abbrev, unpronounceable
from standalone.local_piper_phoneme_word_letter import speak_individual_letters, translate_letter

def signed(value):
    value &= 4294967295
    return value - 4294967296 if value & 2147483648 else value

@dataclass(frozen=True)
class WordResult:
    flags: int
    phonemes: bytes
    replacement: bytes | None = None

@dataclass
class Work:
    tr: object
    buffer: bytearray
    start: int
    wtab: list | None
    word_out: bool
    flags: list[int] = field(default_factory=lambda: [0, 0])
    flags2: list[int] = field(default_factory=lambda: [0, 0])
    phones: bytes = b''
    prefix_phones: bytes = b''
    unpron_phones: bytes = b''
    end_phones: bytes = b''
    end_type: int = 0
    end_type1: int = 0
    prefix_type: int = 0
    prefix_flags: int = 0
    spell: int = 0
    emphasis: int = 0
    was_unpronounceable: int = 0
    found: bool = False

    def __post_init__(self):
        if self.wtab is None:
            self.wtab = [WordTab() for _ in range(8)]
        self.wflags = self.wtab[0].flags
        self.tr.session.state.dictionary_skipwords = 0
        self.word1 = self.start + int(byte(self.buffer, self.start) == 32)
        self.wordx = self.word1
        self.first, _ = utf8_in(self.buffer, self.wordx)
        self.last = 0
        self.length = 0
        while byte(self.buffer, self.wordx) not in (0, 32):
            self.last, size = utf8_in(self.buffer, self.wordx)
            self.wordx += size
            self.length += 1
        self.saved = bytes(self.buffer[self.start:self.start + min(self.wordx - self.start, 159)])
        self.alphabet = self.tr.session.state.current_alphabet

    def result(self, flags=None, phones=None, replacement=None, restore=False):
        if phones is not None:
            self.tr.session.state.word_phonemes = bytes(phones).split(b'\x00', 1)[0]
        if restore:
            self.buffer[self.start:self.start + len(self.saved)] = self.saved
        return WordResult(signed(self.flags[0] if flags is None else flags), self.tr.session.state.word_phonemes, replacement)

    def lookup(self, cursor, flags, end_flags=0, buffer=None):
        if buffer is None:
            result = self.tr.lookup_dict_list(self.buffer, cursor, flags=tuple(flags), end_flags=end_flags, word_table=self.wtab)
        else:
            with self.tr.session.outside_clause():
                result = self.tr.lookup_dict_list(buffer, cursor, flags=tuple(flags), end_flags=end_flags, word_table=self.wtab)
        flags[:] = result.flags
        return result

    def rules(self, cursor, *, flags=None, word_flags=None, end=True, initial=None):
        return self.tr.rules.translate(self.buffer, cursor, initial=self.phones if initial is None else initial, end_phonemes=end, word_flags=self.wflags if word_flags is None else word_flags, dict_flags=self.flags if flags is None else flags)

    def initial_lookup(self):
        tr = self.tr
        session = tr.session
        if self.length == 1 and self.wflags & 4194304:
            following, _ = utf8_in(self.buffer, self.wordx + 1)
            if not is_alpha(following) or tr.alphabet_from_char(self.last) is not tr.alphabet_from_char(following):
                self.spell = 1
        if tr.option_sayas == 36:
            if self.length == 1:
                self.spell = 4
            else:
                self.word1 -= 1
                self.buffer[self.word1] = 95
                lookup = self.lookup(self.word1, self.flags)
                self.found = lookup.found
                self.phones = lookup.phonemes
        if tr.option_sayas & 16:
            self.spell = tr.option_sayas & 15
            return None
        if not self.found:
            lookup = self.lookup(self.word1, self.flags, 2)
            self.found = lookup.found
            self.phones = lookup.phonemes
        else:
            lookup = None
        if self.flags[0] & 50331648 and byte(self.buffer, self.wordx + 1) == 46:
            self.buffer[self.wordx + 1] = 32
        if self.flags[0] & 536870912:
            return self.result(replacement=lookup.replacement if self.word_out and lookup is not None else None)
        if not self.found and self.flags[0] & 128 and (not self.flags[0] & 8192):
            cursor = self.word1
            count = 0
            while count < session.state.dictionary_skipwords:
                if byte(self.buffer, cursor) == 32:
                    self.buffer[cursor] = 45
                    count += 1
                cursor += 1
        if self.length == 1 and session.state.dictionary_skipwords == 0 and check_dotted_abbrev(tr, self.buffer, self.word1):
            self.flags[:] = [128 if session.state.dictionary_skipwords else 0, 0]
            self.spell = 1
        if self.phones[:1] == b'\x15':
            return self.result(0, self.phones)
        if not self.found and self.flags[0] & 8192:
            self.spell = 1
        return self.numerals_and_capitals()

    def spell_word(self):
        cursor, phones = speak_individual_letters(self.tr, self.buffer, self.word1, b'', self.spell, self.alphabet)
        if cursor is None:
            return self.result(4096 if self.length > 1 else 0, phones)
        if self.wflags & 4194304:
            return self.result(0, phones)
        phones = add_plural_suffix(self.tr, self.wflags, self.last, phones)
        return self.result(self.flags[0] & 128, phones)

    def pronounce(self):
        tr = self.tr
        posn = 0
        length = 999
        self.wordx = self.word1
        while 0 < length < 3 or (self.length > 1 and unpronounceable(tr, self.buffer, self.wordx, posn)):
            self.was_unpronounceable = 67108864
            self.emphasis = 0
            if byte(self.buffer, self.wordx) == 39:
                break
            size, self.unpron_phones = translate_letter(tr, self.buffer, self.wordx, self.unpron_phones, int(posn > 0), self.alphabet)
            self.wordx += size
            posn += 1
            if self.unpron_phones[:1] == b'\x15':
                return self.result(4096 if self.unpron_phones[1:] == b'en' else 0, self.unpron_phones)
            length = 0
            while byte(self.buffer, self.wordx + length) != 32:
                length += 1
        self.unpron_phones = tr.session.numbers.set_spelling_stress(tr, self.unpron_phones, 0, posn)
        if byte(self.buffer, self.wordx) == 32:
            return None
        if self.unpron_phones and byte(self.buffer, self.wordx) != 39:
            self.buffer[self.wordx - 1] = 32
        result = self.rules(self.wordx)
        self.phones = result.phonemes
        self.end_phones = result.end_phonemes
        self.end_type = result.end_type
        if self.phones[:1] == b'\x15':
            return self.result(0, self.phones)
        if not self.phones and (not self.end_phones):
            code, _ = utf8_in(self.buffer, self.wordx)
            if self.length == 1 and (is_alpha(code) or tr.session.numbers.is_superscript(code)):
                cursor, phones = speak_individual_letters(tr, self.buffer, self.wordx, self.phones, self.spell, self.alphabet)
                return self.result(0, phones)
        self.previous = byte(self.buffer, self.wordx - 1)
        result = self.prefixes()
        if result is not None:
            return result
        result = self.suffixes()
        if result is not None:
            return result
        self.buffer[self.wordx - 1] = self.previous
        return None

    def prefixes(self):
        tr = self.tr
        confirm = True
        for _ in range(50):
            if not self.end_type & 1024:
                break
            if confirm and (not self.end_type & 131072):
                check = self.rules(self.wordx, initial=b'', word_flags=self.wflags | 805306368)
                if check.end_type:
                    _, original = remove_ending(tr, self.buffer, self.wordx, check.end_type)
                    result = self.rules(self.wordx, word_flags=self.wflags | 268435456)
                    self.phones = result.phonemes
                    self.end_type = result.end_type
                    self.end_phones = result.end_phonemes
                    self.buffer[self.wordx:self.wordx + len(original)] = original
                    if not self.end_type & 1024:
                        self.phones = check.phonemes
                        self.end_type = check.end_type
                        self.end_phones = check.end_phonemes
                    confirm = False
                    continue
            self.prefix_type = self.end_type
            if self.prefix_type & 2048:
                tr.expect_verb = 1
            self.buffer[self.wordx - 1] = self.previous
            if not self.prefix_type & 131072:
                for _ in range(self.prefix_type & 15):
                    self.wordx += 1
                    while byte(self.buffer, self.wordx) & 192 == 128:
                        self.wordx += 1
            else:
                length = self.prefix_type & 63
                prefix_chars = b'\x00' + bytes(self.buffer[self.wordx:self.wordx + length - 1]) + b'\x00'
                self.wordx += length
            self.previous = byte(self.buffer, self.wordx - 1)
            self.buffer[self.wordx - 1] = 32
            confirm = True
            self.wflags |= 8388608
            if self.prefix_type & 131072:
                self.prefix_phones = self.phones
                lookup = self.lookup(1, self.flags, buffer=prefix_chars)
                self.found = lookup.found
                self.phones = lookup.phonemes
                if self.found:
                    self.prefix_phones = self.phones
                if self.flags[0] & 8192:
                    _, self.prefix_phones = speak_individual_letters(tr, bytearray(prefix_chars), 1, b'', 1, self.alphabet)
            else:
                self.prefix_phones += self.end_phones
            self.end_phones = b''
            self.end_type = 0
            lookup = self.lookup(self.wordx, self.flags2, 1024)
            self.found = lookup.found
            self.phones = lookup.phonemes
            if self.flags[0] == 0:
                self.flags[:] = self.flags2
            else:
                self.prefix_flags = 1
            if not self.found:
                result = self.rules(self.wordx, word_flags=self.wflags & 8404992)
                self.phones = result.phonemes
                self.end_type = result.end_type
                self.end_phones = result.end_phonemes
                if self.phones[:1] == b'\x15':
                    self.buffer[self.wordx - 1] = self.previous
                    return self.result(0, self.phones)
        return None

    def suffixes(self):
        if not self.end_type or self.end_type & 1024:
            return None
        tr = self.tr
        self.end_type1 = self.end_type
        before = self.phones
        end_flags, original = remove_ending(tr, self.buffer, self.wordx, self.end_type)
        more = True
        while more:
            more = False
            self.phones = b''
            if self.prefix_phones:
                self.buffer[self.wordx - 1] = self.previous
                lookup = self.lookup(self.word1, self.flags2, end_flags)
                self.found = lookup.found
                self.phones = lookup.phonemes
                self.buffer[self.wordx - 1] = 32
                if self.phones[:1] == b'\x15':
                    self.buffer[self.wordx:self.wordx + len(original)] = original
                    return self.result(0, self.phones)
                if self.flags[0] == 0:
                    self.flags[:] = self.flags2
                if self.found:
                    self.prefix_phones = b''
                if not self.found and self.flags2[0] != 0:
                    self.prefix_flags = 1
            if not self.found:
                lookup = self.lookup(self.wordx, self.flags2, end_flags)
                self.found = lookup.found
                self.phones = lookup.phonemes
                if self.phones[:1] == b'\x15':
                    self.buffer[self.wordx:self.wordx + len(original)] = original
                    return self.result(0, self.phones)
                if self.flags[0] == 0:
                    self.flags[:] = self.flags2
            if not self.found:
                if self.end_type & 16384:
                    self.phones = before
                else:
                    if end_flags & 4:
                        self.wflags |= 8192
                    if self.end_type & 262144:
                        self.wflags |= 134217728
                    if self.end_type & 524288:
                        tail = self.end_phones
                        result = self.rules(self.wordx)
                        self.phones = result.phonemes
                        self.end_type = result.end_type
                        self.end_phones = result.end_phonemes + tail
                        if self.end_type and (not self.end_type & 1024):
                            end_flags, _ = remove_ending(tr, self.buffer, self.wordx, self.end_type, False)
                            more = True
                    else:
                        result = self.rules(self.wordx, end=False)
                        self.phones = result.phonemes
                        self.end_type = 0
                    if self.phones[:1] == b'\x15':
                        self.buffer[self.wordx:self.wordx + len(original)] = original
                        self.buffer[self.wordx - 1] = self.previous
                        return self.result(0, self.phones)
        if not self.end_type1 & 65536:
            self.phones = tr.append_phonemes(self.phones, 200, self.end_phones)
            self.end_phones = b''
        self.buffer[self.wordx:self.wordx + len(original)] = original
        return None

    def stress(self):
        tr = self.tr
        tr.session.state.word_phonemes = add_plural_suffix(tr, self.wflags, self.last, tr.session.state.word_phonemes)
        self.wflags |= self.emphasis
        prefix_stress = next((code for code in reversed(self.prefix_phones) if code in (6, 7)), 0)
        if self.prefix_flags or prefix_stress:
            prefix = self.prefix_phones
            if tr.langopts.param[2] or self.prefix_type & 65536:
                self.phones = tr.set_word_stress(self.phones, self.flags, 3, 0)
                first = True
                output = bytearray()
                for code in prefix:
                    if code == 6:
                        if first:
                            first = False
                        else:
                            code = 5
                    output.append(code)
                prefix = bytes(output)
            phones = (self.unpron_phones + prefix + self.phones)[:199]
            phones = tr.set_word_stress(phones, self.flags, -1, 0)
        else:
            self.phones = tr.set_word_stress(self.phones, self.flags, -1, 2 if self.end_phones else 0)
            phones = (self.unpron_phones + self.prefix_phones + self.phones)[:199]
        if self.end_phones:
            phones += self.end_phones[:199 - len(phones)]
        if self.wflags & 16:
            self.flags[0] &= ~268435456
        if self.wflags & 128 and tr.langopts.stress_flags & 1048576:
            phones = change_word_stress(tr, phones, 3)
        elif self.wflags & 3072:
            phones = change_word_stress(tr, phones, 6)
            if self.wflags & 2048:
                self.flags[0] |= 268435456
        elif self.wtab[tr.session.state.dictionary_skipwords].flags & 16:
            if self.flags[0] & 1536:
                phones = change_word_stress(tr, phones, 4)
            elif self.flags[0] & 2048:
                phones = change_word_stress(tr, phones, 3)
        self.following_word_expectations()
        if self.length == 1 and tr.translator_name == 25966 and is_walpha(self.first) and (self.first != 105):
            self.flags[0] |= 16777216
        if tr.langopts.param[11] & 2 and self.flags[0] & 98304:
            phones = apply_special_attribute(tr, phones, self.flags[0])
        self.flags[0] |= self.was_unpronounceable
        return self.result(phones=phones, replacement=b'' if self.flags[0] & 536870912 and self.word_out else None, restore=True)

    def numerals_and_capitals(self):
        tr = self.tr
        session = tr.session
        if not self.found and is_wdigit(self.first):
            _, phones = tr.lookup(b'_0lang')
            session.state.word_phonemes = phones
            if phones[:1] == b'\x15':
                return self.result(0)
            if tr.langopts.numbers2 & 32768 and (not self.wtab[0].flags & 2097152):
                return self.result(0, b'\x15')
            session.numbers.dictionary_skipwords = session.state.dictionary_skipwords
            self.found, self.phones = session.numbers.translate_number(tr, self.buffer, self.word1, self.flags, self.wtab, 0)
            session.state.dictionary_skipwords = session.numbers.dictionary_skipwords
        if not self.found and self.wflags & 3 != 2:
            if tr.langopts.numbers & 16777216 or (tr.langopts.numbers & 33554432 and self.wflags & 1):
                if self.wflags & 16 or not self.wtab[1].flags & 256:
                    session.numbers.dictionary_skipwords = session.state.dictionary_skipwords
                    self.found, self.phones = session.numbers.translate_roman(tr, self.buffer, self.word1, self.wtab)
                    session.state.dictionary_skipwords = session.numbers.dictionary_skipwords
                    if self.found:
                        self.flags[0] |= 8192
        if self.wflags & 1 and self.length > 1 and is_walpha(self.first):
            if session.state.option_tone_flags & 256 and (not self.flags[0] & 8192):
                self.emphasis = 2048
            elif not self.found and (not self.flags[0] & 128) and (self.length < 4) and (tr.clause_lower_count > 3) and (tr.clause_upper_count <= tr.clause_lower_count):
                self.spell = 1
        return None

    def following_word_expectations(self):
        tr = self.tr
        if self.end_type1 & 8192:
            tr.expect_verb = tr.expect_verb_s = 2
        if self.flags[1] & 8:
            tr.expect_past = 3
            tr.expect_verb = tr.expect_noun = 0
        elif self.flags[1] & 1:
            tr.expect_verb = 2
            tr.expect_verb_s = tr.expect_noun = 0
        elif self.flags[1] & 2:
            tr.expect_verb = 0
            tr.expect_verb_s = 2
            tr.expect_past = tr.expect_noun = 0
        elif self.flags[1] & 4:
            tr.expect_noun = 2
            tr.expect_verb = tr.expect_verb_s = tr.expect_past = 0
        if byte(self.buffer, self.wordx) != 0 and (not self.flags[1] & 256):
            for key in ('expect_verb', 'expect_verb_s', 'expect_noun', 'expect_past'):
                if getattr(tr, key) > 0:
                    setattr(tr, key, getattr(tr, key) - 1)

def translate_word3(tr, buffer, cursor, word_table=None, *, word_out=True):
    with tr.session.native_calls():
        work = Work(tr, buffer, cursor, word_table, word_out)
        result = work.initial_lookup()
        if result is not None:
            return result
        if work.spell > 0:
            return work.spell_word()
        if not work.found:
            result = work.pronounce()
            if result is not None:
                return result
        return work.stress()

def translate_word(tr, buffer, cursor, word_table=None, *, word_out=True):
    result = translate_word3(tr, buffer, cursor, word_table, word_out=word_out)
    if not result.flags & 536870912 or not word_out:
        return result
    replacement = result.replacement or b''
    text = bytearray(b'\x00 ' + replacement + b'\x00')
    position = 2
    output = bytearray()
    available = 200
    first = True
    while byte(text, position) and available > 1:
        code, _ = utf8_in(text, position)
        if word_table is None:
            raise ValueError('eSpeak replacement expansion requires its word table')
        if is_wupper(code):
            word_table[0].flags |= 2
            lower = code + 32 if 65 <= code <= 90 else code
            encoded = utf8_out(lower)
            text[position:position + len(encoded)] = encoded
        else:
            word_table[0].flags &= ~2
        saved = tr.session.state.dictionary_skipwords
        with tr.session.outside_clause():
            word = translate_word3(tr, text, position, word_table, word_out=False)
        part = (b'' if first else b'\x0f') + word.phonemes
        first = False
        output.extend(part[:max(0, available - 1)])
        available -= len(part)
        skips = tr.session.state.dictionary_skipwords + 1
        for _ in range(skips):
            while byte(text, position) not in (9, 10, 11, 12, 13, 32):
                if not byte(text, position):
                    raise ValueError('eSpeak replacement must have its trailing separator')
                position += 1
            while byte(text, position) in (9, 10, 11, 12, 13, 32):
                position += 1
        tr.session.state.dictionary_skipwords = saved
    if output:
        tr.session.state.word_phonemes = bytes(output[:199])
    return WordResult(result.flags, tr.session.state.word_phonemes, replacement)
