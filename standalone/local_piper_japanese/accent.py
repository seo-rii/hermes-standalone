"""Original OpenJTalk ordered accent-phrase and accent-nucleus rules."""
from .njd import _integer


def _basic_boundary(node, previous):
    pos, before = node.get('pos'), previous.get('pos')
    node.chain_flag = 1
    if before == '形容詞' and pos in ('名詞', '動詞', '形容詞'):
        node.chain_flag = 0
    if before == '動詞' and pos in ('形容詞', '名詞'):
        node.chain_flag = 0
    if pos in ('副詞', '接続詞', '連体詞') or before in ('副詞', '接続詞', '連体詞'):
        node.chain_flag = 0
    if before == '名詞' and previous.get('pos_group1') == '副詞可能':
        node.chain_flag = 0
    if pos == '名詞' and node.get('pos_group1') == '副詞可能':
        node.chain_flag = 0
    if pos in ('助動詞', '助詞'):
        node.chain_flag = 1
    if before in ('助動詞', '助詞') and pos not in ('助動詞', '助詞'):
        node.chain_flag = 0
    if previous.get('pos_group1') == '接尾' and pos == '名詞':
        node.chain_flag = 0


def _dependent_boundary(node, previous):
    pos, before = node.get('pos'), previous.get('pos')
    if pos == '形容詞' and node.get('pos_group1') == '非自立':
        if before in ('動詞', '形容詞') and previous.get('cform').startswith('連用'):
            node.chain_flag = 1
        elif before == '助詞' and previous.get('pos_group1') == '接続助詞' and previous.get('string') in ('て', 'で'):
            node.chain_flag = 1
    if before == '名詞' and (pos in ('動詞', '形容詞') or node.get('pos_group1') == '形容動詞語幹'):
        node.chain_flag = 0
    if pos == '動詞' and node.get('pos_group1') == '非自立':
        if before == '動詞' and previous.get('cform').startswith('連用'):
            node.chain_flag = 1
        elif before == '名詞' and previous.get('pos_group1') in ('サ変接続', '形容動詞語幹'):
            node.chain_flag = 1
        elif before == '助詞':
            node.chain_flag = 1


def _name_boundary(node, previous):
    pos, before = node.get('pos'), previous.get('pos')
    if pos == '記号' or before == '記号':
        node.chain_flag = 0
    if pos == '接頭詞':
        node.chain_flag = 0
    if previous.get('pos_group3') == '姓' and node.get('pos_group3') == '名':
        node.chain_flag = 0
    if before == '名詞' and node.get('pos_group3') == '名':
        node.chain_flag = 0
    if node.get('pos_group1') == '接尾':
        node.chain_flag = 1
    if pos == '名詞' and node.get('pos_group1') == '非自立' and before in ('助動詞', '連体詞', '助詞', '形容詞', '動詞'):
        node.chain_flag = 1
    if node.get('pos_group1') == '固有名詞':
        node.chain_flag = 0
    if before == '名詞' and previous.get('pos_group1') in ('非自立', '代名詞') and pos == '名詞' and node.get('pos_group1') in ('一般', 'サ変接続'):
        node.chain_flag = 0
    if pos == '感動詞' or before == '感動詞':
        node.chain_flag = 0


def set_accent_phrase(njd):
    for node in njd.nodes():
        if node.prev is None or node.chain_flag >= 0:
            continue
        _basic_boundary(node, node.prev)
        _dependent_boundary(node, node.prev)
        _name_boundary(node, node.prev)


class _RuleTokens:
    def __init__(self, text):
        self.text = text or ''
        self.position = 0

    def next(self):
        start = self.position
        while self.position < len(self.text) and self.text[self.position] not in '%@/':
            self.position += 1
        token = self.text[start:self.position].encode()[:1023].decode()
        separator = self.text[self.position] if self.position < len(self.text) else '\0'
        if separator != '\0':
            self.position += 1
        return token, separator


def accent_rule(text, previous_pos):
    tokens = _RuleTokens(text)
    separator = ' '
    while separator != '\0':
        token, separator = tokens.next()
        if separator == '%' and token in previous_pos or separator in ('@', '/', '\0'):
            if separator == '%':
                token, separator = tokens.next()
            if separator == '@':
                addition, _ = tokens.next()
                return token, _integer(addition)
            return token, 0
        while separator in ('%', '@'):
            _, separator = tokens.next()
    return '*', 0


def _chain_accent(top, node, rule, addition, mora):
    direct = {'F4': mora + addition, 'F5': 0, 'C1': mora + node.acc, 'C2': mora + 1, 'C3': mora, 'C4': 0, 'P6': 0}
    if rule in direct:
        top.set('acc', direct[rule])
    elif rule == 'F2' and top.acc == 0 or rule == 'F3' and top.acc != 0:
        top.set('acc', mora + addition)
    elif rule == 'P1':
        top.set('acc', 0 if node.acc == 0 else mora + node.acc)
    elif rule == 'P14' and node.acc != 0:
        top.set('acc', mora + node.acc)


def _tens_accent(node, previous):
    before = previous.get('string')
    if before in ('三', '四', '九', '何', '数'):
        previous.acc = 1
    elif before in ('五', '六', '八'):
        previous.acc = previous.mora_size + 1
    elif before == '七':
        previous.acc = previous.mora_size
    else:
        previous.acc = 1
    if before in ('五', '六', '八') and node.next is not None and node.next.get('string') in ('一', '二', '三', '四', '五', '六', '七', '八', '九'):
        previous.acc = 0


def _numeric_accent(node):
    previous = node.prev
    if previous is None or node.chain_flag != 1 or previous.get('pos_group1') != '数' or node.get('pos_group1') != '数':
        return
    word, before = node.get('string'), previous.get('string')
    if word == '十':
        _tens_accent(node, previous)
    elif word == '百':
        if before == '七':
            previous.acc = 2
        elif before in ('三', '四', '九', '何'):
            previous.acc = 1
        else:
            previous.acc = previous.mora_size + node.mora_size
    elif word in ('千', '万'):
        previous.acc = previous.mora_size + 1
    elif word == '億':
        previous.acc = 2 if before in ('一', '六', '七', '八', '幾') else 1
    elif word == '兆':
        previous.acc = 2 if before in ('六', '七') else 1


def _finish_number(top, mora):
    if top is not None and top.get('pos_group1') == '数' and top.acc > mora:
        top.acc = 0


def set_accent_type(njd):
    top = None
    mora = original_accent = original_mora = 0
    for node in njd.nodes():
        if node.prev is None or node.chain_flag != 1:
            _finish_number(top, mora)
            top = node
            mora = 0
            original_accent = node.acc
            original_mora = node.mora_size
        else:
            rule, addition = accent_rule(node.get('chain_rule'), node.prev.get('pos'))
            if top.get('chain_rule') == 'P2' and original_accent != 0 and original_mora > original_accent:
                top.acc = original_accent
            else:
                _chain_accent(top, node, rule, addition, mora)
        _numeric_accent(node)
        if node.get('string') == '十' and node.chain_flag != 1 and node.next is not None and node.next.chain_flag == 1 and node.next.get('pos_group1') == '数':
            node.acc = 0
        mora += node.mora_size
    _finish_number(top, mora)
