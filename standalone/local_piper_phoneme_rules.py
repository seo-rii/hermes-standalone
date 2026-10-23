"""Interpreter for original eSpeak compiled pronunciation-rule chains."""
_PRE_HANDLERS = {17: '_pre_group', 18: '_pre_group', 25: '_pre_letter', 11: '_pre_letter', 15: '_pre_letter', 16: '_pre_letter', 28: '_pre_dollar', 21: '_pre_syllables', 10: '_pre_word_state', 26: '_pre_word_state', 19: '_pre_word_state', 29: '_pre_nonvowel', 46: '_pre_dot', 45: '_pre_hyphen', 23: '_pre_skip'}
_POST_HANDLERS = {17: '_post_group', 18: '_post_group', 25: '_post_nonvowel', 15: '_post_digit', 16: '_post_letter', 11: '_post_letter', 28: '_post_dollar', 45: '_post_hyphen', 21: '_post_syllables', 29: '_post_syllables', 23: '_post_skip', 12: '_post_score', 60: '_post_score', 13: '_post_deleted', 14: '_post_ending', 24: '_post_word_start'}
from dataclasses import dataclass, field
from standalone.local_piper_phoneme_unicode import utf8_in, is_digit, is_walpha, is_alpha
from standalone.local_piper_phoneme_dictionary import RuleResult
from standalone.local_piper_phoneme_letters import REMOVE_ACCENT, is_bracket

def byte(buffer, index):
    return buffer[index] if 0 <= index < len(buffer) else 0

def append_phonemes(tr, string, size, phonemes):
    phonemes = bytes(phonemes).split(b'\x00', 1)[0]
    if len(phonemes) + len(string) >= size:
        return string
    unstressed = False
    for code in phonemes:
        if code >= tr.n_phoneme_tab:
            continue
        phone = tr.phoneme_tab[code]
        if phone is None:
            continue
        if phone.type == 1:
            if phone.std_length < 4:
                unstressed = True
        elif phone.type == 2:
            if not phone.phflags & 2 and (not unstressed):
                tr.word_stressed_count += 1
            unstressed = False
            tr.word_vowel_count += 1
    return string + phonemes

@dataclass
class Match:
    points: int = 0
    phonemes: bytes = b''
    end_type: int = 0
    del_fwd: int | None = None

@dataclass
class Frame:
    buffer: bytes | bytearray
    cursor: int
    start: int
    group_length: int
    rule: int
    word_flags: int
    dict_flags: int
    pre: int = 0
    post: int = 0
    left: int = -2
    right: int = -6
    letter: int = 0
    wide: int = 0
    last: int = 0
    extra: int = 0
    consumed: int = 0
    failed: int = 0
    mode: int = 0
    atstart: bool = False
    ignore: int = 0
    match: Match = field(default_factory=lambda: Match(points=1))

class Rules:

    def __init__(self, tr, dictionary):
        self.tr = tr
        self.dictionary = dictionary
        self.data = dictionary.content

    def letter_group(self, buffer, cursor, group, pre=False):
        position = self.dictionary.letter_groups[group]
        if position is None:
            return -1
        while self.data[position] != 7:
            if self.data[position] == 126:
                return 0
            stop = self.data.index(0, position)
            length = stop - position
            start = cursor
            if pre:
                if not byte(buffer, start):
                    position = stop + 1
                    continue
                for _ in range(length - 1):
                    start -= 1
                    if not byte(buffer, start):
                        break
                else:
                    if bytes(buffer[start:start + length]) == self.data[position:stop]:
                        return length
                position = stop + 1
                continue
            if bytes(buffer[start:start + length]) == self.data[position:stop]:
                return length
            position = stop + 1
        return -1

    def dollar(self, frame, command):
        size = frame.cursor - frame.start + frame.consumed + frame.group_length + 1
        if size + 2 > 160:
            frame.failed = 1
            return 0
        word = frame.buffer[frame.start - 1:frame.start - 1 + size] + b' \x00'
        flags = self.tr.lookup_dict_list(word, 1, flags=(0, 0)).flags
        if command == 3 and flags[0] & 2147483648 and (not flags[1] & 16384):
            return 23
        if flags[0] & 1 << 14 + (command & 15):
            return 23
        frame.failed = 1
        return 0

    def _operator(self, f, opcode, common):
        if opcode == 0:
            position = common
            if position is None:
                f.match.phonemes = b''
            else:
                while True:
                    value = self.data[position]
                    position += 1
                    if value in (0, 3):
                        break
                    if value == 5:
                        position += 1
                    if value == 9:
                        position += 2
                f.match.phonemes = self.data[position:self.data.index(0, position)]
            f.rule -= 1
            f.failed = 2
        elif opcode == 8:
            f.atstart = True
            f.ignore = 0
            f.mode = 1
        elif opcode == 1:
            f.mode = 1
            if f.word_flags & 2147483648:
                f.failed = 1
        elif opcode == 2:
            f.mode = 2
        elif opcode == 3:
            f.match.phonemes = self.data[f.rule:self.data.index(0, f.rule)]
            f.failed = 2
        elif opcode == 4:
            common = f.rule
        elif opcode == 5:
            condition = self.data[f.rule]
            f.rule += 1
            present = bool(self.tr.dict_condition & 4294967295 & 1 << (condition - (32 if condition >= 32 else 0) & 63))
            if present == (condition < 32):
                f.match.points += 1
            else:
                f.failed = 1
        elif opcode == 9:
            f.rule += 2
        return common

    def _post(self, f, opcode):
        f.right = min(19, f.right + 6)
        f.last = f.wide
        if not byte(f.buffer, f.post - 1):
            f.failed = 1
            return 0
        f.wide, size = utf8_in(f.buffer, f.post)
        f.extra = size - 1
        f.letter = byte(f.buffer, f.post)
        f.post += 1
        handler = _POST_HANDLERS.get(opcode)
        if handler is not None:
            return getattr(self, handler)(f, opcode)
        if f.letter == opcode:
            return 21 - f.right if f.letter & 192 != 128 else 0
        f.failed = 1
        return 0

    def _pre(self, f, opcode):
        f.left = min(19, f.left + 2)
        if not byte(f.buffer, f.pre):
            f.failed = 1
            return 0
        f.last, _ = utf8_in(f.buffer, f.pre)
        f.pre -= 1
        f.wide, size = utf8_in(f.buffer, f.pre, True)
        f.extra = size - 1
        f.letter = byte(f.buffer, f.pre)
        handler = _PRE_HANDLERS.get(opcode)
        if handler is not None:
            return getattr(self, handler)(f, opcode)
        if f.letter == opcode:
            return 4 if f.letter == 32 else 21 - f.left if f.letter & 192 != 128 else 0
        f.failed = 1
        return 0

    def match_rule(self, buffer, cursor, start, group_length, rule, word_flags=0, dict_flags=0):
        if rule is None:
            return (cursor + 1, Match())
        best = Match()
        consumed = 0
        common = None
        while self.data[rule] != 7:
            f = Frame(buffer, cursor, start, group_length, rule, word_flags, dict_flags, pre=cursor, post=cursor + group_length, ignore=word_flags & 2147483648)
            while not f.failed:
                opcode = self.data[f.rule]
                f.rule += 1
                points = 0
                if opcode <= 9:
                    common = self._operator(f, opcode, common)
                    continue
                if f.mode == 0:
                    value = byte(buffer, f.post)
                    f.post += 1
                    if value == opcode or (value == 69 and opcode == 101):
                        if value & 192 != 128:
                            points = 21
                        f.consumed += 1
                    else:
                        f.failed = 1
                elif f.mode == 2:
                    points = self._post(f, opcode)
                else:
                    points = self._pre(f, opcode)
                if not f.failed:
                    f.match.points += points
            if f.failed == 2 and (not f.ignore) and (not f.atstart or byte(buffer, f.pre - 1) == 32):
                if f.atstart:
                    f.match.points += 4
                if f.match.points >= best.points:
                    best = f.match
                    consumed = f.consumed
            rule = self.data.index(0, f.rule) + 1
        return (cursor + (consumed + group_length or 1), best)

    def translate(self, buffer, start=0, *, max_phonemes=200, end_phonemes=False, word_flags=0, dict_flags=None, initial=b''):
        return _Translation(self, buffer, start, max_phonemes, end_phonemes, word_flags, dict_flags, initial).run()

    def _post_group(self, f, opcode):
        group = self.data[f.rule] - 65 & 255
        f.rule += 1
        if opcode == 17:
            if not self.tr.is_letter(f.wide, group):
                f.failed = 1
                return 0
            f.post += f.extra
            return (19 if group == 2 else 20) - f.right
        size = self.letter_group(f.buffer, f.post - 1, group)
        if size < 0:
            f.failed = 1
            return 0
        f.post += size - 1
        return 20 - f.right

    def _post_nonvowel(self, f, opcode):
        if self.tr.is_letter(f.wide, 0) or (f.wide == 32 and f.word_flags & 134217728):
            f.failed = 1
            return 0
        f.post += f.extra
        return 20 - f.right

    def _post_digit(self, f, opcode):
        if is_digit(f.wide):
            f.post += f.extra
            return 20 - f.right
        if self.tr.langopts.tone_numbers:
            f.post -= 1
            return 20 - f.right
        f.failed = 1
        return 0

    def _post_letter(self, f, opcode):
        if not is_walpha(f.wide) if opcode == 16 else f.wide == f.last:
            f.post += f.extra
            return 21 - f.right
        f.failed = 1
        return 0

    def _post_dollar(self, f, opcode):
        f.post -= 1
        command = self.data[f.rule]
        f.rule += 1
        if command == 1:
            f.match.end_type = 32768
        elif command == 2:
            if f.word_flags & 8388608:
                f.failed = 1
            else:
                return 1
        elif command & 240 == 16:
            if f.dict_flags & 1 << 14 + (command & 15):
                return 23
            f.failed = 1
        elif command & 240 == 32 or command == 3:
            return self.dollar(f, command)
        return 0

    def _post_hyphen(self, f, opcode):
        if f.letter == 45 or (f.letter == 32 and f.word_flags & 16384):
            return 22 - f.right
        f.failed = 1
        return 0

    def _post_syllables(self, f, opcode):
        position = f.post + f.extra
        wide = f.wide
        vowels = 0
        prior = False
        syllables = 1
        while self.data[f.rule] == 21 and opcode == 21:
            f.rule += 1
            syllables += 1
        while wide not in (0, 32):
            vowel = bool(self.tr.is_letter(wide, 7))
            if opcode == 29 and vowel:
                f.failed = 1
                return 0
            if not prior and vowel:
                vowels += 1
            prior = vowel
            wide, size = utf8_in(f.buffer, position)
            position += size
        if opcode == 29:
            return 19 - f.right
        if syllables <= vowels:
            return 18 + syllables - f.right
        f.failed = 1
        return 0

    def _post_skip(self, f, opcode):
        position = f.post - 1
        previous = position
        wide = f.wide
        group_bytes = -1
        rule_wide, _ = utf8_in(self.data, f.rule)
        while wide != rule_wide and wide not in (0, 32) and (group_bytes == -1):
            if rule_wide == 18:
                group_bytes = self.letter_group(f.buffer, position, self.data[f.rule + 1] - 65 & 255)
            previous = position
            wide, size = utf8_in(f.buffer, position)
            position += size
        if wide == rule_wide or group_bytes >= 0:
            f.post = previous
        return 0

    def _post_score(self, f, opcode):
        f.post -= 1
        return 20 if opcode == 12 else -20

    def _post_deleted(self, f, opcode):
        for position in range(f.cursor + f.group_length, f.post):
            if byte(f.buffer, position) == 101:
                f.match.del_fwd = position
                break
        return 0

    def _post_ending(self, f, opcode):
        high = self.data[f.rule]
        high = high - 256 if high >= 128 else high
        ending = (high << 16) + ((self.data[f.rule + 1] & 127) << 8) + (self.data[f.rule + 2] & 127)
        if self.tr.word_vowel_count == 0 and (not ending & 1024) and self.tr.langopts.param[15] & 1:
            f.failed = 1
        else:
            f.match.end_type = ending
            f.rule += 3
        return 0

    def _post_word_start(self, f, opcode):
        if f.word_flags & 8192:
            f.failed = 1
            return 0
        f.post -= 1
        return 1

    def _pre_group(self, f, opcode):
        group = self.data[f.rule] - 65 & 255
        f.rule += 1
        if opcode == 17:
            if not self.tr.is_letter(f.wide, group):
                f.failed = 1
                return 0
            f.pre -= f.extra
            return (19 if group == 2 else 20) - f.left
        size = self.letter_group(f.buffer, f.pre, group, True)
        if size < 0:
            f.failed = 1
            return 0
        f.pre -= size - 1
        return 20 - f.right

    def _pre_letter(self, f, opcode):
        valid = {25: lambda: not self.tr.is_letter(f.wide, 0), 11: lambda: f.wide == f.last, 15: lambda: is_digit(f.wide), 16: lambda: not is_walpha(f.wide)}[opcode]()
        if not valid:
            f.failed = 1
            return 0
        f.pre -= f.extra
        return (20 if opcode == 25 else 21) - (f.right if opcode == 16 else f.left)

    def _pre_dollar(self, f, opcode):
        f.pre += 1
        command = self.data[f.rule]
        f.rule += 1
        return self.dollar(f, command) if command == 3 or command & 240 == 32 else 0

    def _pre_syllables(self, f, opcode):
        syllables = 1
        while self.data[f.rule] == 21:
            f.rule += 1
            syllables += 1
        if syllables <= self.tr.word_vowel_count:
            return 18 + syllables - f.left
        f.failed = 1
        return 0

    def _pre_word_state(self, f, opcode):
        f.pre += 1
        valid = self.tr.word_stressed_count > 0 if opcode == 10 else bool(self.tr.expect_verb) if opcode == 26 else bool(f.word_flags & 2)
        if valid:
            return 19 if opcode == 10 else 1
        f.failed = 1
        return 0

    def _pre_nonvowel(self, f, opcode):
        position = f.pre - f.extra
        wide = f.wide
        while wide != 32:
            if self.tr.is_letter(wide, 7) or wide == 0:
                f.failed = 1
                return 0
            wide, size = utf8_in(f.buffer, position - 1, True)
            position -= size
        return 3

    def _pre_dot(self, f, opcode):
        position = f.pre
        while byte(f.buffer, position) not in (0, 32):
            if byte(f.buffer, position) == 46:
                return 50
            position -= 1
        f.failed = 1
        return 0

    def _pre_hyphen(self, f, opcode):
        if f.letter == 45 or (f.letter == 32 and f.word_flags & 128):
            return 22 - f.right
        f.failed = 1
        return 0

    def _pre_skip(self, f, opcode):
        position = f.pre + 1
        previous = position
        group_bytes = -1
        while byte(f.buffer, position) != self.data[f.rule] and byte(f.buffer, position) not in (0, 32) and (group_bytes == -1):
            previous = position
            position -= 1
            if self.data[f.rule] == 18:
                group_bytes = self.letter_group(f.buffer, previous, self.data[f.rule + 1] - 65 & 255, True)
        if byte(f.buffer, position) == self.data[f.rule]:
            f.pre = previous
        if group_bytes >= 0:
            f.pre = previous + 1
        return 0

class _Translation:

    def __init__(self, rules, buffer, start, max_phonemes, end_phonemes, word_flags, dict_flags, initial):
        self.rules = rules
        self.tr = rules.tr
        self.dictionary = rules.dictionary
        self.buffer = buffer
        self.start = start
        self.max_phonemes = max_phonemes
        self.end_phonemes = end_phonemes
        self.word_flags = word_flags
        self.dict_flags = dict_flags
        self.output = bytes(initial)
        self.ending = b''
        self.dict_flags0 = 0 if dict_flags is None else dict_flags[0]
        self.original = bytes(buffer[start:start + 159]).split(b'\x00', 1)[0]
        self.cursor = start
        self.tr.word_vowel_count = 0
        self.tr.word_stressed_count = 0
        self.any_alpha = 0
        self.digit_count = 0
        self.match = Match()

    def finish(self, end_type=0, restore=True):
        if restore:
            self.buffer[self.start:self.start + len(self.original)] = self.original
        if self.dict_flags is not None:
            self.dict_flags[0] = self.dict_flags0
        return RuleResult(self.output, end_type, self.ending, self.dict_flags0)

    def digit(self):
        _, phones = self.tr.lookup(b'_' + bytes(self.buffer[self.cursor:self.cursor + self.wide_bytes]))
        self.digit_count += 1
        if self.digit_count >= 2:
            phones += b'\x0b'
            self.digit_count = 0
        self.output = append_phonemes(self.tr, self.output, self.max_phonemes, phones)
        self.cursor += self.wide_bytes

    def match_rule(self, length, rule):
        return self.rules.match_rule(self.buffer, self.cursor, self.start, length, rule, self.word_flags, self.dict_flags0)

    def select_match(self):
        self.found = False
        self.fallback = False
        offset = self.wide - self.tr.letter_bits_offset
        if 0 <= offset < 128 and self.dictionary.groups3[offset] is not None:
            self.cursor, self.match = self.match_rule(self.wide_bytes, self.dictionary.groups3[offset])
            self.found = True
        groups = self.dictionary.groups2_count[self.value]
        if not self.found and groups > 0:
            pair = self.value + (byte(self.buffer, self.cursor + 1) << 8)
            first = self.dictionary.groups2_start[self.value]
            for group in range(first, first + groups):
                if self.dictionary.groups2_name[group] != pair:
                    continue
                self.found = True
                cursor2, match2 = self.match_rule(2, self.dictionary.groups2[group])
                if match2.points > 0:
                    match2.points += 35
                self.cursor, self.match = self.match_rule(1, self.dictionary.groups1[self.value])
                if match2.points >= self.match.points:
                    self.match = match2
                    self.cursor = cursor2
        if not self.found:
            if self.dictionary.groups1[self.value] is not None:
                self.cursor, self.match = self.match_rule(1, self.dictionary.groups1[self.value])
            else:
                self.cursor, self.match = self.match_rule(0, self.dictionary.groups1[0])
                self.fallback = True

    def remove_accent(self, letter, extra, replacement):
        previous = self.cursor - 1
        self.buffer[self.cursor - 1] = replacement
        while True:
            self.buffer[self.cursor] = byte(self.buffer, self.cursor + extra)
            if self.buffer[self.cursor] == 32:
                break
            self.cursor += 1
        for _ in range(extra):
            self.buffer[self.cursor] = 32
            self.cursor += 1
        if self.tr.langopts.param[0] and letter in (228, 235, 239, 246, 252, 255):
            self.cursor = previous
            return
        self.output = b''
        self.cursor = self.start
        self.tr.word_vowel_count = 0
        self.tr.word_stressed_count = 0

    def fallback_character(self):
        letter, size = utf8_in(self.buffer, self.cursor - 1)
        extra = size - 1
        if self.tr.letter_bits_offset > 0 and letter <= 577 and is_walpha(letter):
            self.output = b'\x15en'
            return 'foreign'
        if letter == 57384:
            self.tr.pre_pause = max(self.tr.pre_pause, self.tr.langopts.param[17])
        if is_bracket(letter):
            self.tr.pre_pause = max(self.tr.pre_pause, self.tr.langopts.param[12])
        replacement = REMOVE_ACCENT[letter - 192] if 192 <= letter < 606 else 0
        if replacement and (byte(self.buffer, self.cursor - 2) != 32 or byte(self.buffer, self.cursor + extra) != 32):
            self.remove_accent(letter, extra, replacement)
            return 'retry'
        alphabet = self.tr.alphabet_from_char(letter)
        if alphabet is not None and alphabet.offset != self.tr.letter_bits_offset:
            language = self.tr.langopts.alt_alphabet_lang if self.tr.langopts.alt_alphabet == alphabet.offset else alphabet.language if alphabet.flags & 4 else None
            if language is not None:
                self.output = b'\x15' + (language & 4294967295).to_bytes(4, 'big').lstrip(b'\x00')
                return 'foreign'
        return None

    def unmatched(self):
        if self.fallback and self.match.points == 0 and (not self.tr.option_sayas & 16):
            action = self.fallback_character()
            if action is not None:
                return action
        if self.match.points == 0:
            if 768 <= self.wide <= 879:
                pass
            elif is_alpha(self.wide):
                following = byte(self.buffer, self.cursor + self.wide_bytes - 1)
                if self.any_alpha > 1 or 32 < following < 128:
                    self.output = b''
                    if self.dict_flags is not None:
                        self.dict_flags0 |= 4096
                    return 'break'
            else:
                phones = self.tr.lookup_letter(self.wide, -1)
                if phones:
                    self.match.phonemes = phones
                    self.match.points = 1
            self.cursor += self.wide_bytes - 1
        else:
            self.tr.phonemes_repeat_count = 0
        return None

    def accepted(self):
        if self.word_flags & 2147483648:
            return self.finish(self.match.end_type | 1, False)
        if self.match.phonemes[:1] == b'\x15' and (not self.word_flags & 4096):
            self.output = self.match.phonemes
            return self.finish(restore=False)
        self.match.end_type &= ~32768
        if self.match.end_type and self.end_phonemes:
            if not (self.match.end_type & 1024 and self.word_flags & 536870912):
                if self.match.end_type & 1024 and (not self.match.end_type & 127):
                    self.match.end_type |= self.cursor - self.start
                self.ending = self.match.phonemes
                return self.finish(self.match.end_type)
        if self.match.del_fwd is not None:
            self.buffer[self.match.del_fwd] = 69
        self.output = append_phonemes(self.tr, self.output, self.max_phonemes, self.match.phonemes)
        return None

    def run(self):
        while byte(self.buffer, self.cursor) not in (0, 32):
            self.value = byte(self.buffer, self.cursor)
            self.wide, self.wide_bytes = utf8_in(self.buffer, self.cursor)
            if is_alpha(self.wide):
                self.any_alpha += 1
            if is_digit(self.wide) and (not self.tr.langopts.tone_numbers or not self.any_alpha):
                self.digit()
                continue
            self.digit_count = 0
            self.select_match()
            if not self.found:
                action = self.unmatched()
                if action == 'retry':
                    continue
                if action == 'foreign':
                    return self.finish(restore=False)
                if action == 'break':
                    break
            if self.match.points <= 0:
                continue
            result = self.accepted()
            if result is not None:
                return result
        return self.finish()
