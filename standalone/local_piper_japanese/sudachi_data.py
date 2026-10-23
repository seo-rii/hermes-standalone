"""Owned Sudachi V1 dictionary, compacted UTF-16 strings and XOR trie reader."""
from dataclasses import dataclass
import mmap
from pathlib import Path
import struct


def varint(data, position):
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError('Truncated Sudachi integer')
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            if value >= 1 << 64:
                raise ValueError('Sudachi integer exceeds64bits')
            return value, position
    raise ValueError('Invalid Sudachi integer')


def utf8_string(data, position):
    size, position = varint(data, position)
    end = position + size
    if end > len(data):
        raise ValueError('Truncated Sudachi string')
    return data[position:end].decode(), end


def string_pointer(encoded):
    base = encoded >> 27
    extra = max(0, base - 19)
    variable = (encoded & 0x07ff0000) >> (28 - extra)
    implicit = 4096 >> (13 - extra)
    length = base - extra + (variable | implicit)
    alignment = max(0, extra - 1)
    offset = (encoded & (0x07ffffff >> alignment)) << alignment
    return length, offset


@dataclass(frozen=True)
class WordInfo:
    word_id: int
    left: int
    right: int
    cost: int
    pos_id: int
    headword: str
    reading: str
    normalized_id: int
    dictionary_id: int
    index_length: int
    c_split: tuple
    b_split: tuple
    a_split: tuple
    structure: tuple
    synonyms: tuple
    user_data: str


class SudachiBinary:
    def __init__(self, path):
        self.file = Path(path).open('rb')
        try:
            self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self.file.close()
            raise
        try:
            self._description()
            self._pos()
            start, size = self.blocks['ConnMatrix']
            self.num_left, self.num_right = struct.unpack_from('<hh', self.data, start)
            if self.num_left * self.num_right * 2 + 4 > size:
                raise ValueError('Truncated Sudachi connection matrix')
        except BaseException:
            self.close()
            raise

    def close(self):
        self.data.close()
        self.file.close()

    def _description(self):
        if self.data[:16] != b'SudachiBinaryDic' or struct.unpack_from('<Q', self.data, 16)[0] != 1:
            raise ValueError('Expected the reviewed Sudachi V1 binary format')
        self.created, self.flags = struct.unpack_from('<QQ', self.data, 24)
        self.comment, position = utf8_string(self.data, 40)
        self.signature, position = utf8_string(self.data, position)
        self.reference, position = utf8_string(self.data, position)
        self.indexed, position = varint(self.data, position)
        self.total, position = varint(self.data, position)
        count, position = varint(self.data, position)
        self.blocks = {}
        for _ in range(count):
            name, position = utf8_string(self.data, position)
            start, position = varint(self.data, position)
            size, position = varint(self.data, position)
            if start + size > len(self.data) or name in self.blocks:
                raise ValueError('Invalid Sudachi dictionary block')
            self.blocks[name] = start, size
        if self.reference:
            raise ValueError('This owned reader currently accepts a Sudachi system dictionary')

    def _pos(self):
        position, size = self.blocks['POS']
        end = position + size
        count, = struct.unpack_from('<h', self.data, position)
        position += 2
        self.pos = []
        for _ in range(count):
            row = []
            for _ in range(6):
                length, = struct.unpack_from('<h', self.data, position)
                position += 2
                if length < 0 or position + length * 2 > end:
                    raise ValueError('Invalid Sudachi part-of-speech string')
                row.append(self.data[position:position + length * 2].decode('utf-16-le'))
                position += length * 2
            self.pos.append(tuple(row))

    def _u32(self, block, index):
        start, size = self.blocks[block]
        if index < 0 or index * 4 + 4 > size:
            raise ValueError('Invalid Sudachi table offset')
        return struct.unpack_from('<I', self.data, start + index * 4)[0]

    def _string(self, pointer):
        length, offset = string_pointer(pointer)
        start, size = self.blocks['Strings']
        if (offset + length) * 2 > size:
            raise ValueError('Invalid Sudachi compacted string pointer')
        return self.data[start + offset * 2:start + (offset + length) * 2].decode('utf-16-le')

    def params(self, word_id):
        start, size = self.blocks['Entries']
        offset = (word_id & 0x0fffffff) * 8
        if offset < 32 or offset + 32 > size:
            raise ValueError('Invalid Sudachi entry offset')
        return struct.unpack_from('<hhh', self.data, start + offset)

    def info(self, word_id):
        word_id &= 0x0fffffff
        left, right, cost = self.params(word_id)
        start, size = self.blocks['Entries']
        position = start + word_id * 8
        pos_id, headword, reading, normalized, dictionary, length, *counts = struct.unpack_from('<hIIIIh6b', self.data, position + 6)
        position += 32
        lists = []
        for index, count in enumerate(counts[:5]):
            if count < 0:
                if index not in (1, 2, 3):
                    raise ValueError('Invalid Sudachi split-array length')
                lists.append(lists[-1])
                continue
            if position + count * 4 > start + size:
                raise ValueError('Truncated Sudachi word arrays')
            code = 'i' if index == 4 else 'I'
            values = struct.unpack_from('<' + str(count) + code, self.data, position)
            lists.append(tuple(value if index == 4 else value & 0x0fffffff for value in values))
            position += count * 4
        user_data = ''
        if counts[5] == 1:
            units, = struct.unpack_from('<h', self.data, position)
            position += 2
            user_data = self.data[position:position + units * 2].decode('utf-16-le')
        elif counts[5] != 0:
            raise ValueError('Invalid Sudachi user-data flag')
        return WordInfo(word_id, left, right, cost, pos_id, self._string(headword), self._string(reading),
                        normalized & 0x0fffffff, dictionary & 0x0fffffff, length, *lists, user_data)

    def headword(self, word_id):
        self.params(word_id)
        start, _ = self.blocks['Entries']
        pointer, = struct.unpack_from('<I', self.data, start + (word_id & 0x0fffffff) * 8 + 8)
        return self._string(pointer)

    def entry_values(self, word_id):
        info = self.info(word_id)
        return {'word_id': word_id, 'surface': info.headword, 'reading': info.reading, 'pos_id': info.pos_id,
                'pos': self.pos[info.pos_id], 'normalized': self.headword(info.normalized_id),
                'dictionary': self.headword(info.dictionary_id), 'synonyms': info.synonyms, 'user_data': info.user_data}

    def entries(self):
        start, size = self.blocks['Entries']
        offset = 32
        for _ in range(self.total):
            if offset + 32 > size:
                raise ValueError('Truncated Sudachi dictionary entry')
            yield offset // 8
            counts = struct.unpack_from('<6b', self.data, start + offset + 26)
            if counts[0] < 0 or counts[4] < 0 or counts[5] not in (0, 1):
                raise ValueError('Invalid Sudachi entry length')
            length = 32 + 4 * sum(max(0, value) for value in counts[:5])
            if counts[5]:
                units, = struct.unpack_from('<h', self.data, start + offset + length)
                if units < 0:
                    raise ValueError('Invalid Sudachi user-data string')
                length += 2 + units * 2
            offset += (length + 7) & ~7

    def lookup(self, data, offset=0):
        unit = self._u32('TrieIndex', 0)
        node = (unit >> 10) << ((unit & 512) >> 6)
        pointers_start, pointers_size = self.blocks['WordPointers']
        for position in range(offset, len(data)):
            key = data[position]
            node ^= key
            unit = self._u32('TrieIndex', node)
            if unit & 0x800000ff != key:
                return
            node ^= (unit >> 10) << ((unit & 512) >> 6)
            if unit & 256:
                pointer = self._u32('TrieIndex', node) & 0x7fffffff
                cursor = pointers_start + pointer
                count, cursor = varint(self.data, cursor)
                total = 0
                for _ in range(count):
                    value, cursor = varint(self.data, cursor)
                    total += value
                    yield total, position + 1

    def connection(self, left, right):
        if not 0 <= left < self.num_left or not 0 <= right < self.num_right:
            raise ValueError('Invalid Sudachi connection index')
        start, _ = self.blocks['ConnMatrix']
        return struct.unpack_from('<h', self.data, start + 4 + 2 * (right * self.num_left + left))[0]
