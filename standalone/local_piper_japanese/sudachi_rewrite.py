"""Sudachi source-order numeric and Katakana OOV path rewrite plugins."""
from dataclasses import replace
from .sudachi_input import BITS
from .sudachi_numeric import NumericParser
from .sudachi_tokenizer import Node, OOV


def concat(path, begin, end, pos, normalized=None, only_headword=False):
    group = path[begin:end]
    headword = ''.join(node.info['surface'] for node in group)
    reading = headword if only_headword else ''.join(node.info['reading'] for node in group)
    dictionary = headword if only_headword else ''.join(node.info['dictionary'] for node in group)
    if normalized is None:
        normalized = headword if only_headword else ''.join(node.info['normalized'] for node in group)
    node = Node(group[0].begin, group[-1].end, 65535, 65535, 32767, OOV | pos, group[-1].total)
    node.info = {'surface': headword, 'reading': reading, 'normalized': normalized, 'dictionary': dictionary,
                 'pos_id': pos, 'synonyms': (), 'user_data': ''}
    path[begin:end] = [node]


class NumericRewriter:
    def __init__(self, pos, text, path):
        self.pos = pos
        self.text = text
        self.path = path
        self.begin = -1
        self.comma = self.period = True
        self.parser = NumericParser()

    def _concat(self, begin, end):
        first = self.path[begin]
        if first.info['pos_id'] != self.pos:
            return
        normalized = self.parser.normalized()
        if end - begin > 1 or normalized != first.info['normalized']:
            concat(self.path, begin, end, self.pos, normalized)

    def _numeric(self, index, spelling):
        if self.begin < 0:
            self.parser.clear()
            self.begin = index
        for char in spelling:
            if self.parser.append(char):
                continue
            if self.begin >= 0:
                if self.parser.error == 'Comma':
                    self.comma = False
                    index = self.begin - 1
                elif self.parser.error == 'Point':
                    self.period = False
                    index = self.begin - 1
                self.begin = -1
            break
        return index

    def _trailing_separator(self, index):
        spelling = self.path[index - 1].info['normalized']
        return self.parser.error == 'Comma' and spelling == ',' or self.parser.error == 'Point' and spelling == '.'

    def _separator(self, index, spelling):
        if self.begin >= 0:
            if self.parser.done():
                self._concat(self.begin, index)
                index = self.begin + 1
            elif self._trailing_separator(index):
                self._concat(self.begin, index - 1)
                index = self.begin + 2
        self.begin = -1
        if not self.comma and spelling != ',':
            self.comma = True
        if not self.period and spelling != '.':
            self.period = True
        return index

    def rewrite(self):
        index = -1
        while index < len(self.path) - 1:
            index += 1
            node = self.path[index]
            spelling = node.info['normalized']
            category = self.text.category(node.begin, node.end)
            numeric = category & (BITS['NUMERIC'] | BITS['KANJINUMERIC'])
            if numeric or self.comma and spelling == ',' or self.period and spelling == '.':
                index = self._numeric(index, spelling)
            else:
                index = self._separator(index, spelling)
        if self.begin >= 0:
            end = len(self.path)
            if self.parser.done():
                self._concat(self.begin, end)
            elif self._trailing_separator(end):
                self._concat(self.begin, end - 1)
        return self.path


def _katakana(text, node):
    return bool(text.category(node.begin, node.end) & BITS['KATAKANA'])


def _katakana_group(text, path, index):
    begin = index
    while begin > 0 and _katakana(text, path[begin - 1]):
        begin -= 1
    end = index + 1
    while end < len(path) and _katakana(text, path[end]):
        end += 1
    while begin != end and text.cats[path[begin].begin] & BITS['NOOOVBOW']:
        begin += 1
    return begin, end


def _join_katakana(tokenizer, text, path, lattice, begin, end):
    first, last = path[begin], path[end - 1]
    choices = [node for node in lattice[last.end] if node.begin == first.begin]
    if choices:
        # The source chooses the first minimum word cost, independently of the
        # accumulated path cost, and can recover a hidden dictionary candidate.
        node = replace(min(choices, key=lambda node: node.cost))
        node.info = tokenizer.node_info(node, text)
        path[begin:end] = [node]
    else:
        concat(path, begin, end, tokenizer.katakana_pos, only_headword=True)


def katakana_rewrite(tokenizer, text, path, lattice):
    index = 0
    while index < len(path):
        node = path[index]
        if not (node.is_oov() or node.end - node.begin < 3) or not _katakana(text, node):
            index += 1
            continue
        begin, end = _katakana_group(text, path, index)
        if end - begin > 1:
            _join_katakana(tokenizer, text, path, lattice, begin, end)
            index = begin + 1
        index += 1
    return path
