"""Owned Sudachi source-order dictionary/OOV lattice and C-unit tokenizer."""
from dataclasses import dataclass
from pathlib import Path

from .sudachi_data import SudachiBinary
from .sudachi_input import BITS, Categories, Input, Rewriter

OOV = 0xf0000000
MAX_COST = 0x7fffffff


@dataclass(eq=False)
class Node:
    begin: int
    end: int
    left: int
    right: int
    cost: int
    word_id: int
    total: int = MAX_COST
    previous: object = None
    info: dict | None = None

    def is_oov(self):
        return self.word_id >> 28 == 15


class Morpheme:
    def __init__(self, node, text, pos):
        self.node = node
        self.text = text
        self.pos = pos

    def surface(self):
        return self.text.original_slice(self.node.begin, self.node.end)

    def begin(self):
        offset = self.text.mapping[self.text.byte_offsets[self.node.begin]]
        return len(self.text.original_bytes[:offset].decode())

    def end(self):
        offset = self.text.mapping[self.text.byte_offsets[self.node.end]]
        return len(self.text.original_bytes[:offset].decode())

    def reading_form(self):
        return self.node.info['reading']

    def normalized_form(self):
        return self.node.info['normalized']

    def dictionary_form(self):
        return self.node.info['dictionary']

    def part_of_speech_id(self):
        return self.node.info['pos_id']

    def part_of_speech(self):
        return self.pos[self.part_of_speech_id()]

    def word_id(self):
        return self.node.word_id

    def dictionary_id(self):
        return -1 if self.node.is_oov() else self.node.word_id >> 28

    def is_oov(self):
        return self.node.is_oov()

    def synonym_group_ids(self):
        return self.node.info['synonyms']

    def user_data(self):
        return self.node.info['user_data']


class Tokenizer:
    def __init__(self, dictionary_path, resource_directory):
        self.dictionary = SudachiBinary(dictionary_path)
        self.categories = Categories(Path(resource_directory) / 'char.def')
        self.rewriter = Rewriter(resource_directory, self.categories)
        self.pos_ids = {value: index for index, value in enumerate(self.dictionary.pos)}
        self.oovs = self._unknowns(Path(resource_directory) / 'unk.def')
        self.simple_pos = self.pos_ids[('補助記号', '一般', '*', '*', '*', '*')]
        self.katakana_pos = self.pos_ids[('名詞', '普通名詞', '一般', '*', '*', '*')]
        self.numeric_pos = self.pos_ids[('名詞', '数詞', '*', '*', '*', '*')]

    def close(self):
        self.dictionary.close()

    def _unknowns(self, path):
        result = {}
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split(',')
            category = BITS[fields[0]]
            left, right, cost = map(int, fields[1:4])
            pos = self.pos_ids[tuple(fields[4:10])]
            result.setdefault(category, []).append((left, right, cost, pos))
        return result

    def _mecab_oovs(self, text, begin, created):
        category = text.cats[begin]
        length = text.continuity[begin]
        for bit, (invoke, group, maximum) in sorted(self.categories.properties.items()):
            if not category & bit or created and not invoke:
                continue
            lengths = [length] if group else []
            limit = length - bool(group)
            for size in range(1, maximum + 1):
                size = min(size, len(text.chars) - begin)
                if size > limit:
                    break
                lengths.append(size)
            for size in lengths:
                for left, right, cost, pos in self.oovs.get(bit, ()):
                    yield Node(begin, begin + size, left, right, cost, OOV | pos)

    def _simple_oov(self, text, begin):
        end = begin + 1
        while end < len(text.chars) and not text.bow[end]:
            end += 1
        return Node(begin, end, 5968, 5968, 3857, OOV | self.simple_pos)

    def _connect(self, node, ends):
        for previous in ends[node.begin]:
            if previous.total == MAX_COST:
                continue
            cost = previous.total + self.dictionary.connection(previous.right, node.left) + node.cost
            cost = ((cost + (1 << 31)) & 0xffffffff) - (1 << 31)
            if cost < node.total:
                node.previous, node.total = previous, cost

    def _insert(self, node, ends):
        self._connect(node, ends)
        ends[node.end].append(node)

    def lattice(self, text):
        ends = [[] for _ in range(len(text.chars) + 1)]
        ends[0].append(Node(0, 0, 0, 0, 0, OOV | 0x0fffffff, 0))
        raw = text.current.encode()
        for begin, offset in enumerate(text.byte_offsets[:-1]):
            if not ends[begin]:
                continue
            created = False
            for word_id, end_byte in self.dictionary.lookup(raw, offset):
                end = text.char_offsets[end_byte]
                if not text.bow[end]:
                    continue
                left, right, cost = self.dictionary.params(word_id)
                self._insert(Node(begin, end, left, right, cost, word_id), ends)
                created = True
            if text.oov_bow(begin):
                for node in self._mecab_oovs(text, begin, created):
                    self._insert(node, ends)
                    created = True
            if not created:
                self._insert(self._simple_oov(text, begin), ends)
        eos = Node(len(text.chars), len(text.chars), 0, 0, 0, OOV | 0x0ffffffe)
        self._connect(eos, ends)
        if eos.previous is None:
            raise ValueError('Sudachi BOS/EOS are disconnected')
        return ends, eos

    def node_info(self, node, text):
        if not node.is_oov():
            return self.dictionary.entry_values(node.word_id)
        spelling = ''.join(text.chars[node.begin:node.end])
        return {'word_id': node.word_id, 'surface': spelling, 'reading': spelling,
                'pos_id': node.word_id & 0x0fffffff, 'normalized': spelling, 'dictionary': spelling,
                'synonyms': (), 'user_data': ''}

    def raw_path(self, text, rewrite=True):
        data = self.rewriter.rewrite(text) if rewrite else Input(text)
        if not rewrite:
            data.build(self.categories)
        if not data.chars:
            return data, [], [[]]
        ends, eos = self.lattice(data)
        path = []
        current = eos.previous
        while current.previous is not None:
            current.info = self.node_info(current, data)
            path.append(current)
            current = current.previous
        path.reverse()
        return data, path, ends

    def tokenize_raw(self, text, rewrite=True):
        data, path, _ = self.raw_path(text, rewrite)
        return [Morpheme(node, data, self.dictionary.pos) for node in path]

    def tokenize(self, text):
        from .sudachi_rewrite import NumericRewriter, katakana_rewrite
        data, path, lattice = self.raw_path(text)
        path = NumericRewriter(self.numeric_pos, data, path).rewrite()
        path = katakana_rewrite(self, data, path, lattice)
        return [Morpheme(node, data, self.dictionary.pos) for node in path]
