"""Original TranslateChar, .replace and upper-case word exceptions.

Adapted from eSpeakNG translate.c, Copyright 2005-2014 Jonathan Duddington
and Copyright 2015-2018 Reece H.Dunn, GPL-3.0-or-later. Original source and
independent native observations are recorded in local-piper-phoneme-text-
provenance.json; complete license notices are in licenses/. This qualified
implementation executes owned mutable state and standard-library algorithms.
"""
from standalone import local_piper_phoneme_unicode as u
from .local_piper_phoneme_text_tables import FLAG_CHAR_REPLACED
HANGUL_COMPATIBILITY = (0, 0, 1, 170, 2, 172, 173, 3, 4, 5, 176, 177, 178, 179, 180, 180, 182, 6, 7, 8, 185, 9, 10, 188, 12, 13, 14, 15, 16, 17, 18, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117)
UCASE_GA = (b'bp', b'bhf', b'dt', b'gc', b'hA', b'mb', b'nd', b'ng', b'ts', b'tA', b'nA')

def lower(tr, c):
    return u.to_lower_for_voice(c, tr.langopts.dotless_i)

def upper_case_in_word(tr, buffer, cursor, c):
    if tr.translator_name != 26465:
        return False
    for rule in UCASE_GA:
        size = len(rule)
        if buffer[cursor - size] == 32 and buffer[cursor - size + 1:cursor] == rule[:-1]:
            if c == rule[-1] or (rule[-1] == 65 and tr.is_letter(c, 7)):
                return True
    return False

class Characters:

    def __init__(self):
        self.ignore_next_n = 0

    @staticmethod
    def replacement_pairs(data):
        if not isinstance(data, (bytes, bytearray)):
            return () if data is None else data
        pairs = []
        cursor = 0
        while data[cursor:cursor + 4] != b'\x00' * 4:
            end = data.index(0, cursor)
            original = bytes(data[cursor:end])
            cursor = end + 1
            end = data.index(0, cursor)
            replacement = bytes(data[cursor:end])
            cursor = end + 1
            pairs.append((original, replacement))
        return pairs

    @staticmethod
    def _match(tr, original, c, buffer, cursor):
        current, size = u.utf8_in(original, 0)
        if current != c:
            return None
        matched = True
        count = 0
        position = cursor
        while size < len(original):
            current, used = u.utf8_in(original, size)
            size += used
            following, used = u.utf8_in(buffer, position)
            position += used
            if lower(tr, following) != current:
                matched = False
            else:
                count += 1
        return count if matched else None

    def substitute(self, tr, c, next_in, buffer, cursor, flags):
        if self.ignore_next_n:
            self.ignore_next_n -= 1
            return (8, 0, flags)
        if c == 0:
            return (c, 0, flags)
        uppercase = bool(u.is_wupper(c))
        key = lower(tr, c) if uppercase else c
        for original, replacement in self.replacement_pairs(tr.langopts.replace_chars):
            count = self._match(tr, original, key, buffer, cursor)
            if count is None:
                continue
            self.ignore_next_n = count
            new_c, size = u.utf8_in(replacement + b'\x00', 0)
            insert = 0
            if size < len(replacement):
                insert, _ = u.utf8_in(replacement, size)
                if uppercase and u.is_wupper(next_in):
                    insert = u.to_upper(insert)
            if uppercase:
                new_c = u.to_upper(new_c)
            return (new_c, insert, flags | FLAG_CHAR_REPLACED)
        return (c, 0, flags)

    def translate(self, tr, buffer, cursor, prev_in, c, next_in, flags):
        code = c - 44032
        if 0 <= code and c <= 55215:
            initial = code // 28 // 21
            medial = code // 28 % 21
            final = code % 28
            if initial == 11:
                return (medial + 4449, final + 4519 if final else 0, flags)
            return (initial + 4352, 11 * 28 * 21 + medial * 28 + final + 44032, flags)
        if 0 <= c - 12592 < 52:
            return (HANGUL_COMPATIBILITY[c - 12592] + 4352, 0, flags)
        if tr.translator_name in (24934, 28268) and c == 39 and (not u.is_walpha(prev_in)):
            next2, _ = u.utf8_in(buffer, cursor + 1)
            if u.is_space(next2):
                if next_in == 110 and tr.translator_name == 24934:
                    buffer[cursor] = 32
                    return (601, 0, flags)
                if next_in in (110, 116):
                    return (601, 0, flags)
        return self.substitute(tr, c, next_in, buffer, cursor, flags)
