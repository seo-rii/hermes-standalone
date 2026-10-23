"""Original OpenJTalk three-mora look-ahead and vowel devoicing state."""
from dataclasses import dataclass
from . import unvoiced_tables as tables

_MORAS = {}
for _text in tables.NJD_SET_UNVOICED_VOWEL_MORA_LIST:
    _encoded = _text.encode()
    _MORAS.setdefault(_encoded[0], []).append(_encoded)
_CANDIDATES = tuple((frozenset(getattr(tables, 'NJD_SET_UNVOICED_VOWEL_CANDIDATE_LIST' + str(index))),
                     getattr(tables, 'NJD_SET_UNVOICED_VOWEL_NEXT_MORA_LIST' + str(index))) for index in (1, 2, 3))
del _text, _encoded


@dataclass
class Mora:
    text: bytes | None = None
    node: object = None
    flag: int = -1
    size: int = 0
    index: int = 0
    accent: int = 0


def mora_information(node, index, mora_index, accent):
    while node is not None:
        text = node.get('pron').encode()
        if index < len(text):
            break
        index -= len(text)
        node = node.next
    if node is None:
        return Mora()
    if index == 0 and node.chain_flag != 1:
        mora_index = 0
        accent = node.acc
    result = Mora(node=node, index=mora_index, accent=accent)
    if text.decode() in ('、', '？', '！'):
        result.text = text
        result.flag = 0
        result.size = len(text)
        return result
    for mora in _MORAS.get(text[index], ()):
        if text.startswith(mora, index):
            result.text = mora
            result.size = len(mora)
            break
    if text.startswith('’'.encode(), index + result.size):
        result.flag = 1
        result.size += 3
    return result


def apply_unvoice_rule(current, following):
    if following is None:
        return 0
    current = current.decode()
    following = following.decode()
    if current in ('チ', 'チュ', 'ツ', 'ツィ') and following.startswith(('サ', 'シ', 'ス', 'セ', 'ソ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ')):
        return 0
    for candidates, prefixes in _CANDIDATES:
        if current in candidates:
            return int(following.startswith(prefixes))
    return -1


class _Devoicing:
    def __init__(self):
        self.first = Mora()
        self.second = Mora(index=1)
        self.third = Mora(index=2)

    def look_ahead(self, node, index):
        if self.first.text is None:
            self.first = mora_information(node, index, self.first.index, self.first.accent)
        if self.first.text is None:
            return False
        if self.second.text is None:
            self.second = mora_information(node, index + self.first.size, self.first.index + 1, self.first.accent)
        if self.third.text is None:
            self.third = mora_information(node, index + self.first.size + self.second.size, self.second.index + 1, self.second.accent)
        return True

    def auxiliaries(self):
        first, second, third = self.first, self.second, self.third
        if second.text is not None and third.text is not None and first.node is second.node and second.node is not third.node and first.text.decode() in ('マ', 'デ') and second.text.decode() == 'ス' and second.node.get('pos') in ('動詞', '助動詞', '感動詞'):
            second.flag = 0 if third.node.get('pron') in ('？', 'ー') else 1

    def shi(self):
        first, second, third = self.first, self.second, self.third
        if first.flag == 1 or second.flag != -1 or third.flag == 1 or second.text is None:
            return
        if second.node.get('pron') != 'シ' or second.node.get('pos') not in ('動詞', '助動詞', '助詞'):
            return
        second.flag = 0 if second.accent == second.index + 1 else apply_unvoice_rule(second.text, third.text)
        if second.flag == 1:
            if first.flag == -1:
                first.flag = 0
            if third.flag == -1:
                third.flag = 0

    def estimate(self):
        first, second = self.first, self.second
        if first.flag == -1:
            if first.node.get('pos') == 'フィラー' or second.flag == 1 or first.accent == first.index + 1:
                first.flag = 0
            else:
                first.flag = apply_unvoice_rule(first.text, second.text)
        if first.flag == 1 and second.flag == -1:
            second.flag = 0

    def advance(self):
        size = self.first.size
        self.first = self.second
        self.second = self.third
        self.third = Mora()
        return size


def set_unvoiced_vowel(njd):
    work = _Devoicing()
    for node in njd.nodes():
        length = len(node.get('pron').encode())
        if length * 2 >= 1023:
            continue
        output = bytearray()
        index = 0
        while index < length:
            if not work.look_ahead(node, index):
                return
            work.auxiliaries()
            work.shi()
            work.estimate()
            part = work.first.text + ('’'.encode() if work.first.flag == 1 else b'')
            if len(output) + len(part) >= 1024:
                break
            output.extend(part)
            index += work.advance()
        node.set('pron', output.decode())


def set_long_vowel(njd):
    """The pinned source unconditionally disables its deprecated estimator."""
    return None
