"""Original OpenJTalk missing-reading, kana-chain and auxiliary pronunciation."""
from .pronunciation_tables import NJD_SET_PRONUNCIATION_LIST, NJD_SET_PRONUNCIATION_SYMBOL_LIST

_READINGS = {}
for _surface, _reading, _mora in NJD_SET_PRONUNCIATION_LIST:
    _READINGS.setdefault(_surface[0], []).append((_surface, _reading, int(_mora)))
_SYMBOLS = dict(NJD_SET_PRONUNCIATION_SYMBOL_LIST)
_KANA = frozenset(row[0] for row in NJD_SET_PRONUNCIATION_LIST)
del _surface, _reading, _mora


def _fill_reading(node):
    node.read = node.pron = None
    text = node.get('string')
    position = 0
    while position < len(text):
        for surface, reading, mora in _READINGS.get(text[position], ()):
            if text.startswith(surface, position):
                position += len(surface)
                node.add('read', reading)
                node.add('pron', reading)
                node.add('mora_size', mora)
                break
        else:
            position += 1
    if node.mora_size:
        node.pos = 'フィラー'
        node.pos_group1 = node.pos_group2 = node.pos_group3 = None
    if node.get('orig') == '*':
        node.set('orig', text)
    if node.get('pron') == '*' and text in _SYMBOLS:
        node.set('read', _SYMBOLS[text])
        node.set('pron', _SYMBOLS[text])
    if node.get('pron') == '*':
        node.read = node.pron = '、'
        node.pos = '記号'
        node.pos_group1 = '読点'
        node.pos_group2 = node.pos_group3 = node.ctype = node.cform = '*'


def _chain_kana(njd):
    head = None
    for node in njd.nodes():
        if node.get('pos') != 'フィラー' or node.get('string') not in _KANA:
            head = None
            continue
        if head is None:
            head = node
        else:
            for name in ('string', 'orig', 'read', 'pron'):
                head.add(name, node.get(name))
            head.add('mora_size', node.mora_size)
            node.pron = None


def _auxiliaries(njd):
    for node in njd.nodes():
        following = node.next
        if following is None:
            continue
        if following.get('pron') == 'ウ' and following.get('pos') == '助動詞' and node.get('pos') in ('動詞', '助動詞') and node.mora_size > 0:
            following.pron = 'ー'
        if node.get('pos') == '助動詞' and following.get('string') in ('？', '！'):
            if node.get('string') == 'です':
                node.pron = 'デス'
            elif node.get('string') == 'ます':
                node.pron = 'マス'


def set_pronunciation(njd):
    for node in njd.nodes():
        if node.mora_size == 0:
            _fill_reading(node)
    njd.remove_silent()
    _chain_kana(njd)
    njd.remove_silent()
    _auxiliaries(njd)
