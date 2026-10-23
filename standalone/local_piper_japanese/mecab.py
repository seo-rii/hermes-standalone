"""Owned reader and Viterbi interpreter for original OpenJTalk MeCab data."""
from dataclasses import dataclass
from pathlib import Path
import struct
from .numeral_boundaries import numeral_boundaries


@dataclass
class Node:
    start: int = 0
    length: int = 0
    rlength: int = 0
    left: int = 0
    right: int = 0
    posid: int = 0
    word_cost: int = 0
    feature: str = ''
    unknown: bool = False
    char_type: int = 0
    dictionary: int = 0
    cost: int = 0
    previous: object = None


class Dictionary:
    def __init__(self, path):
        self.content = Path(path).read_bytes()
        if len(self.content) < 72:
            raise ValueError('Truncated OpenJTalk dictionary')
        header = struct.unpack_from('<10I', self.content)
        magic, self.version, self.kind, self.lexsize, self.left_size, self.right_size, trie_size, token_size, feature_size, _ = header
        if magic ^ 0xef718f77 != len(self.content) or self.version != 102:
            raise ValueError('Invalid OpenJTalk dictionary header')
        if self.content[40:72].split(b'\0', 1)[0].upper().replace(b'-', b'') != b'UTF8':
            raise ValueError('Only the pinned UTF-8 OpenJTalk dictionaries are supported')
        if trie_size % 8 or token_size % 16 or 72 + trie_size + token_size + feature_size != len(self.content):
            raise ValueError('Invalid OpenJTalk dictionary section lengths')
        self.trie_offset = 72
        self.trie_size = trie_size
        self.token_offset = 72 + trie_size
        self.token_count = token_size // 16
        self.feature_offset = self.token_offset + token_size

    def _unit(self, index):
        if index < 0 or index * 8 + 8 > self.trie_size:
            raise ValueError('Invalid OpenJTalk double-array offset')
        return struct.unpack_from('<iI', self.content, self.trie_offset + index * 8)

    def prefixes(self, text, start=0, end=None):
        if end is None:
            end = len(text)
        base, _ = self._unit(0)
        matches = []
        for position in range(start, end):
            value, check = self._unit(base)
            if check == base and value < 0:
                matches.append((position - start, -value - 1))
            next_index = base + text[position] + 1
            next_base, check = self._unit(next_index)
            if check != base:
                return matches
            base = next_base
        value, check = self._unit(base)
        if check == base and value < 0:
            matches.append((end - start, -value - 1))
        return matches

    def exact(self, text):
        for length, value in self.prefixes(text):
            if length == len(text):
                return value
        raise ValueError('Missing OpenJTalk unknown-word category')

    def tokens(self, value):
        first, count = value >> 8, value & 255
        if first + count > self.token_count:
            raise ValueError('Invalid OpenJTalk dictionary token offset')
        for index in range(first, first + count):
            left, right, posid, cost, feature, _ = struct.unpack_from('<HHHhII', self.content, self.token_offset + index * 16)
            cursor = self.feature_offset + feature
            if not self.feature_offset <= cursor < len(self.content):
                raise ValueError('Invalid OpenJTalk dictionary feature offset')
            end = self.content.index(0, cursor)
            yield left, right, posid, cost, self.content[cursor:end].decode('utf-8')


@dataclass(frozen=True)
class Character:
    types: int
    default: int
    length: int
    group: bool
    invoke: bool

    def matches(self, other):
        return bool(self.types & other.types)


def utf8_to_ucs2(text, cursor, end):
    if cursor >= end:
        return 0, 1
    first = text[cursor]
    if first < 128:
        return first, 1
    available = end - cursor
    if first & 0xe0 == 0xc0 and available >= 2:
        return (first & 31) << 6 | text[cursor + 1] & 63, 2
    if first & 0xf0 == 0xe0 and available >= 3:
        return (first & 15) << 12 | (text[cursor + 1] & 63) << 6 | text[cursor + 2] & 63, 3
    for width in (4, 5, 6):
        mask = (0xff << (7 - width)) & 0xff
        if available >= width and first & mask == (mask << 1) & 0xff:
            return 0, width
    return 0, 1


class Characters:
    def __init__(self, path):
        self.content = Path(path).read_bytes()
        count, = struct.unpack_from('<I', self.content)
        self.offset = 4 + count * 32
        if self.offset + 0xffff * 4 != len(self.content):
            raise ValueError('Invalid OpenJTalk character table size')
        self.names = tuple(self.content[4 + index * 32:4 + (index + 1) * 32].split(b'\0', 1)[0] for index in range(count))
        self.cache = {}
        self.space = self.get(32)

    def get(self, code):
        if not 0 <= code < 0xffff:
            raise ValueError('OpenJTalk UCS-2 character is outside its original defined table')
        value, = struct.unpack_from('<I', self.content, self.offset + code * 4)
        if value not in self.cache:
            self.cache[value] = Character(value & 0x3ffff, value >> 18 & 255, value >> 26 & 15, bool(value & 0x40000000), bool(value & 0x80000000))
        return self.cache[value]

    def read(self, text, cursor, end):
        code, width = utf8_to_ucs2(text, cursor, end)
        return self.get(code), width

    def seek_other(self, text, cursor, end, kind):
        count = 0
        info = self.get(0)
        width = 1
        while cursor != end:
            info, width = self.read(text, cursor, end)
            if not kind.matches(info):
                break
            cursor += width
            count += 1
            kind = info
        return cursor, info, width, count


class Mecab:
    def __init__(self, directory):
        directory = Path(directory)
        self.dictionary = Dictionary(directory / 'sys.dic')
        self.unknown = Dictionary(directory / 'unk.dic')
        self.characters = Characters(directory / 'char.bin')
        self.unknown_tokens = tuple(tuple(self.unknown.tokens(self.unknown.exact(name))) for name in self.characters.names)
        self.matrix = (directory / 'matrix.bin').read_bytes()
        self.left_size, self.right_size = struct.unpack_from('<HH', self.matrix)
        if self.left_size * self.right_size * 2 + 4 != len(self.matrix):
            raise ValueError('Invalid OpenJTalk connection matrix size')
        if (self.left_size, self.right_size) != (self.dictionary.left_size, self.dictionary.right_size):
            raise ValueError('OpenJTalk dictionary and connection matrix differ')

    def _unknown_nodes(self, nodes, begin, surface, end, info):
        for left, right, posid, cost, feature in self.unknown_tokens[info.default]:
            nodes.append(Node(surface, end - surface, end - begin, left, right, posid, cost, feature, True, info.default, 255))

    def _unknown_lengths(self, text, begin, surface, end, info, width, nodes):
        cursor = surface + width
        grouped = None
        if cursor > end:
            self._unknown_nodes(nodes, begin, surface, cursor, info)
            if nodes:
                return
        if info.group:
            grouped, _, _, count = self.characters.seek_other(text, cursor, end, info)
            if count <= 24:
                self._unknown_nodes(nodes, begin, surface, grouped, info)
        for _ in range(1, info.length + 1):
            if cursor > end:
                break
            if cursor == grouped:
                continue
            self._unknown_nodes(nodes, begin, surface, cursor, info)
            following, width = self.characters.read(text, cursor, end)
            if not info.matches(following):
                break
            cursor += width
        if not nodes:
            self._unknown_nodes(nodes, begin, surface, cursor, info)

    def candidates(self, text, begin, end):
        end = min(end, begin + 65535)
        surface, info, width, _ = self.characters.seek_other(text, begin, end, self.characters.space)
        nodes = []
        for length, value in self.dictionary.prefixes(text, surface, end):
            for left, right, posid, cost, feature in self.dictionary.tokens(value):
                nodes.append(Node(surface, length, surface - begin + length, left, right, posid, cost, feature, False, info.default))
        if not nodes or info.invoke:
            self._unknown_lengths(text, begin, surface, end, info, width, nodes)
        return reversed(nodes)

    def _connect(self, position, node, ends):
        best_cost = 2147483647
        previous = None
        for left in ends[position]:
            transition, = struct.unpack_from('<h', self.matrix, 4 + (left.right + self.left_size * node.left) * 2)
            cost = left.cost + transition + node.word_cost
            if cost < best_cost:
                best_cost = cost
                previous = left
        if previous is None:
            raise RuntimeError('OpenJTalk sentence cost exceeds the original range')
        node.cost = best_cost
        node.previous = previous
        return node

    def _parse(self, text, boundaries=()):
        ends = [[] for _ in range(len(text) + 1)]
        ends[0] = [Node()]
        for position in range(len(text)):
            if not ends[position]:
                continue
            stop = next((boundary for boundary in boundaries if boundary > position), len(text))
            for node in self.candidates(text, position, stop):
                self._connect(position, node, ends)
                end = position + node.rlength
                if end <= len(text):
                    ends[end].insert(0, node)
        position = next(index for index in range(len(text), -1, -1) if ends[index])
        terminal = self._connect(position, Node(start=len(text)), ends)
        nodes = []
        node = terminal.previous
        while node.previous is not None:
            nodes.append(node)
            node = node.previous
        nodes.reverse()
        return nodes

    def parse(self, text):
        text = bytes(text).split(b'\0', 1)[0]
        nodes = self._parse(text)
        boundaries = numeral_boundaries(text, nodes)
        return self._parse(text, boundaries) if boundaries else nodes

    def features(self, text):
        text = bytes(text).split(b'\0', 1)[0]
        return [text[node.start:node.start + node.length].decode('utf-8') + ',' + node.feature for node in self.parse(text)]
