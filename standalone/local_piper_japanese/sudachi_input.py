"""Owned Sudachi character categories, source-position edits and input plugins."""
from bisect import bisect_right
from pathlib import Path
import json
import re
import zlib


NAMES = ('DEFAULT', 'SPACE', 'KANJI', 'SYMBOL', 'NUMERIC', 'ALPHA', 'HIRAGANA', 'KATAKANA',
         'KANJINUMERIC', 'GREEK', 'CYRILLIC', 'USER1', 'USER2', 'USER3', 'USER4')
BITS = {name: 1 << index for index, name in enumerate(NAMES)}
BITS.update(NOOOVBOW=1 << 30, NOOOVEOW=1 << 31, ALL=(1 << 30) - 1)


class Categories:
    def __init__(self, path):
        ranges = []
        self.properties = {}
        for line in Path(path).read_text().splitlines():
            fields = line.split('#', 1)[0].split()
            if not fields:
                continue
            if not fields[0].startswith('0x'):
                self.properties[BITS[fields[0]]] = tuple(map(int, fields[1:4]))
                continue
            bounds = fields[0].split('..')
            start = int(bounds[0], 16)
            end = int(bounds[-1], 16) + 1
            category = 0
            for name in fields[1:]:
                category |= BITS[name]
            ranges.append((start, end, category))
        self.boundaries = sorted({value for row in ranges for value in row[:2]})
        self.categories = [1]
        for start in self.boundaries:
            value = 0
            for lower, upper, mask in ranges:
                if lower <= start < upper:
                    value |= mask
            self.categories.append(value or 1)

    def get(self, char):
        return self.categories[bisect_right(self.boundaries, ord(char))]


class Input:
    def __init__(self, text):
        self.original = text
        self.original_bytes = text.encode()
        if len(self.original_bytes) > 49149:
            raise ValueError('Sudachi input exceeds 49149 bytes')
        self.current = text
        self.mapping = list(range(len(self.original_bytes) + 1))

    def edit(self, edits):
        if not edits:
            return
        source = self.current.encode()
        output = bytearray()
        mapping = []
        start = 0
        for begin, end, replacement in edits:
            output.extend(source[start:begin])
            mapping.extend(self.mapping[start:begin])
            replacement = replacement.encode()
            output.extend(replacement)
            if replacement:
                mapping.append(self.mapping[begin])
                mapping.extend([self.mapping[end]] * (len(replacement) - 1))
            start = end
        output.extend(source[start:])
        mapping.extend(self.mapping[start:])
        mapping[0] = 0
        if len(output) > 65535:
            raise ValueError('Sudachi normalized input exceeds 65535 bytes')
        self.current = output.decode()
        self.mapping = mapping

    def build(self, categories):
        self.chars = list(self.current)
        self.byte_offsets = [0]
        for char in self.chars:
            self.byte_offsets.append(self.byte_offsets[-1] + len(char.encode()))
        self.char_offsets = {value: index for index, value in enumerate(self.byte_offsets)}
        self.cats = [categories.get(char) for char in self.chars]
        self.bow = []
        previous = 0
        for category in self.cats:
            self.bow.append(not (category & (BITS['ALPHA'] | BITS['GREEK'] | BITS['CYRILLIC']) and category & previous))
            previous = category
        self.bow.append(True)
        self._continuity()

    def _continuity(self):
        self.continuity = []
        length = 1
        for start, category in enumerate(self.cats):
            if length > 1:
                length -= 1
                self.continuity.append(length)
                continue
            length = 1
            common = category
            while start + length < len(self.cats):
                common &= self.cats[start + length]
                if not common:
                    break
                length += 1
            self.continuity.append(length)

    def original_slice(self, begin, end):
        start = self.mapping[self.byte_offsets[begin]]
        end = self.mapping[self.byte_offsets[end]]
        return self.original_bytes[start:end].decode()

    def category(self, begin, end):
        if begin == end:
            return 0
        result = 0xffffffff
        for category in self.cats[begin:end]:
            result &= category
        return result

    def oov_bow(self, position):
        return not (self.cats[position] & BITS['NOOOVBOW'] or position and self.cats[position - 1] & BITS['NOOOVEOW'])


class Rewriter:
    def __init__(self, resource_directory, categories):
        self.categories = categories
        self.ignored = set()
        self.replacements = {}
        for line in (Path(resource_directory) / 'rewrite.def').read_text().splitlines():
            fields = line.split()
            if not fields or fields[0].startswith('#'):
                continue
            if len(fields) == 1:
                self.ignored.add(fields[0])
            else:
                self.replacements[fields[0]] = fields[1]
        self.starts = {}
        for key in sorted(self.replacements, key=lambda value: (-len(value), value)):
            self.starts.setdefault(key[0], []).append(key)
        self.unicode = json.loads(zlib.decompress((Path(__file__).with_name('sudachi-unicode.json.zlib')).read_bytes()))

    def default_edits(self, text):
        edits = []
        offset = 0
        index = 0
        while index < len(text):
            char = text[index]
            key = next((key for key in self.starts.get(char, ()) if text.startswith(key, index)), None)
            if key is not None:
                source, replacement = key, self.replacements[key]
            else:
                source = char
                table = self.unicode['lower'] if char in self.ignored else self.unicode['normalized']
                replacement = table.get(char, char)
            size = len(source.encode())
            if source != replacement:
                edits.append((offset, offset + size, replacement))
            offset += size
            index += len(source)
        return edits

    def yomigana_edits(self, text):
        edits = []
        offsets = [0]
        for char in text:
            offsets.append(offsets[-1] + len(char.encode()))
        index = 0
        while index + 3 < len(text):
            if not (self.categories.get(text[index]) & BITS['KANJI']) or text[index + 1] not in '(（':
                index += 1
                continue
            end = index + 2
            while end < len(text) and end < index + 6 and self.categories.get(text[end]) & (BITS['HIRAGANA'] | BITS['KATAKANA']):
                end += 1
            if index + 2 < end < len(text) and text[end] in ')）':
                edits.append((offsets[index + 1], offsets[end + 1], ''))
                index = end + 1
            else:
                index += 1
        return edits

    def rewrite(self, text):
        data = Input(text)
        data.edit(self.default_edits(data.current))
        raw = data.current
        offsets = [0]
        for char in raw:
            offsets.append(offsets[-1] + len(char.encode()))
        data.edit([(offsets[match.start()], offsets[match.end()], 'ー') for match in re.finditer('[ー⁓〜〰-]{2,}', raw)])
        data.edit(self.yomigana_edits(data.current))
        data.build(self.categories)
        return data
