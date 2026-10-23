"""Owned OpenJTalk NJD nodes and exact MeCab feature loading."""
from copy import copy
from dataclasses import dataclass
import re

STRING_FIELDS = ('string', 'pos', 'pos_group1', 'pos_group2', 'pos_group3', 'ctype', 'cform', 'orig', 'read', 'pron', 'chain_rule')


def _integer(value):
    match = re.match(r'\s*[+-]?\d+', value)
    return int(match.group()) if match else 0


@dataclass(eq=False)
class NJDNode:
    string: str | None = None
    pos: str | None = None
    pos_group1: str | None = None
    pos_group2: str | None = None
    pos_group3: str | None = None
    ctype: str | None = None
    cform: str | None = None
    orig: str | None = None
    read: str | None = None
    pron: str | None = None
    acc: int = 0
    mora_size: int = 0
    chain_rule: str | None = None
    chain_flag: int = -1
    prev: object = None
    next: object = None

    def get(self, name):
        value = getattr(self, name)
        return '*' if value is None else value

    def set(self, name, value):
        if name in ('acc', 'mora_size'):
            value = max(0, value)
        elif name in STRING_FIELDS and not value:
            value = None
        setattr(self, name, value)

    def add(self, name, value):
        if isinstance(value, str):
            setattr(self, name, (getattr(self, name) or '') + value)
        elif value is not None:
            self.set(name, getattr(self, name) + value)

    def values(self):
        return {name: getattr(self, name) for name in (*STRING_FIELDS, 'acc', 'mora_size', 'chain_flag')}


def _tokens(text, delimiter=','):
    # The original tokenizer bounds each field at 1023 bytes, then consumes
    # the remaining bytes through its separator before starting the next one.
    fields = text.split('\0', 1)[0].split(delimiter)
    return [part.encode('utf-8')[:1023].decode('utf-8') for part in fields]


def _accent(node, text):
    accent, separator, size = text.partition('/')
    node.set('acc', _integer(accent))
    node.set('mora_size', _integer(size.split(':', 1)[0]))


def _compound(node, surface, original, reading, pronunciation, accent):
    count = accent.count('/')
    originals = _tokens(original, ':')
    readings = _tokens(reading, ':')
    pronunciations = _tokens(pronunciation, ':')
    accents = _tokens(accent, ':')
    aligned = len(originals) == count
    index = 0
    for part in originals[:count - 1]:
        if not part or not surface.startswith(part, index):
            aligned = False
            break
        index += len(part)
    aligned = aligned and index < len(surface)
    result = []
    index = 0
    for position in range(count):
        current = node if position == 0 else copy(node)
        if position:
            current.chain_flag = 0
        original = originals[position] if position < len(originals) else ''
        current.set('orig', original)
        if aligned:
            current.set('string', original if position + 1 < count else surface[index:])
            index += len(original)
        else:
            current.set('string', surface if position == 0 else original)
        current.set('read', readings[position] if position < len(readings) else '')
        current.set('pron', pronunciations[position] if position < len(pronunciations) else '')
        _accent(current, accents[position] if position < len(accents) else '')
        result.append(current)
    return result


def load_feature(feature):
    node = NJDNode()
    if len(feature.encode()) >= 6144:
        return [node]
    fields = _tokens(feature)
    fields += [''] * max(0, 13 - len(fields))
    for index, name in enumerate(('pos', 'pos_group1', 'pos_group2', 'pos_group3', 'ctype', 'cform'), 1):
        node.set(name, fields[index])
    node.set('chain_rule', fields[11])
    if fields[12] in ('0', '1'):
        node.chain_flag = int(fields[12])
    surface, original, reading, pronunciation, accent = fields[0], fields[7], fields[8], fields[9], fields[10]
    if '*' not in accent and accent.count('/') > 1:
        return _compound(node, surface, original, reading, pronunciation, accent)
    for name, value in (('string', surface), ('orig', original), ('read', reading), ('pron', pronunciation)):
        node.set(name, value)
    if '*' not in accent and '/' in accent:
        _accent(node, accent)
    return [node]


class NJD:
    def __init__(self, features=()):
        self.head = self.tail = None
        for feature in features:
            for node in load_feature(feature):
                self.append(node)

    @classmethod
    def from_values(cls, rows):
        result = cls()
        for row in rows:
            result.append(NJDNode(**row))
        return result

    def append(self, node):
        node.prev = self.tail
        node.next = None
        if self.tail is None:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node

    def remove(self, node):
        following = node.next
        if node.prev is None:
            self.head = following
        else:
            node.prev.next = following
        if following is None:
            self.tail = node.prev
        else:
            following.prev = node.prev
        node.prev = node.next = None
        return following

    def remove_silent(self):
        node = self.head
        while node is not None:
            node = self.remove(node) if node.get('pron') == '*' else node.next

    def nodes(self):
        node = self.head
        while node is not None:
            yield node
            node = node.next

    def values(self):
        return [node.values() for node in self.nodes()]
