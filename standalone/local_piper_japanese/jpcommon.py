"""Original OpenJTalk mora, word, accent phrase and breath-group structure."""
from dataclasses import dataclass, field
from . import jpcommon_tables as tables


def _rows(values, width):
    for index in range(0, len(values), width):
        if values[index] is None:
            break
        yield values[index:index + width]


_MORAS = {}
for _mora, _first, _second in _rows(tables.JPCOMMON_MORA_LIST, 3):
    _MORAS.setdefault(_mora[0], []).append((_mora, _first, _second))
_UNVOICE = dict(_rows(tables.JPCOMMON_UNVOICE_LIST, 2))
_POS = dict(_rows(tables.JPCOMMON_POS_LIST, 2))
_CTYPE = dict(_rows(tables.JPCOMMON_CTYPE_LIST, 2))
_CFORM = dict(_rows(tables.JPCOMMON_CFORM_LIST, 2))
_NJD_POS = {tuple(row[:4]): row[4] for row in _rows(tables.NJD2JPCOMMON_POS_LIST, 5)}
_NJD_CTYPE = dict(_rows(tables.NJD2JPCOMMON_CTYPE_LIST, 2))
_NJD_CFORM = dict(_rows(tables.NJD2JPCOMMON_CFORM_LIST, 2))
del _mora, _first, _second


@dataclass(eq=False)
class Phone:
    text: str
    index: int
    mora: object = None


@dataclass(eq=False)
class Mora:
    text: str
    index: int
    word: object
    phones: list = field(default_factory=list)


@dataclass(eq=False)
class Word:
    pron: str
    pos: str
    ctype: str
    cform: str
    index: int
    moras: list = field(default_factory=list)
    phrase: object = None


@dataclass(eq=False)
class Phrase:
    accent: int
    index: int
    breath: object
    words: list = field(default_factory=list)
    emotion: str | None = None
    exclamation: str | None = None

    @property
    def head(self):
        return self.words[0].moras[0]

    @property
    def tail(self):
        return self.words[-1].moras[-1]

    @property
    def count(self):
        return self.tail.index - self.head.index + 1


@dataclass(eq=False)
class Breath:
    index: int
    phrases: list = field(default_factory=list)

    @property
    def head(self):
        return self.phrases[0].head

    @property
    def count(self):
        return self.phrases[-1].tail.index - self.head.index + 1


class Label:
    def __init__(self):
        self.phones = []
        self.moras = []
        self.words = []
        self.phrases = []
        self.breaths = []
        self.short_pause = False

    def pause(self):
        if self.short_pause:
            if self.phones:
                if self.phones[-1].text == 'pau':
                    return
                self.phones.append(Phone('pau', len(self.phones)))
            self.short_pause = False

    def punctuation(self, pron):
        if pron not in ('、', '？', '！'):
            return False
        if pron in ('？', '！') and self.phones:
            phone = self.phones[-1]
            if phone.text == 'pau':
                phone = self.phones[-2]
            phrase = phone.mora.word.phrase
            if pron == '？' and phrase.emotion is None:
                phrase.emotion = '1'
            if pron == '！' and phrase.exclamation is None:
                phrase.exclamation = '1'
        self.short_pause = True
        return True

    def append_mora(self, text, word, phones):
        mora = Mora(text, len(self.moras), word)
        self.moras.append(mora)
        word.moras.append(mora)
        for name in phones:
            phone = Phone(name, len(self.phones), mora)
            self.phones.append(phone)
            mora.phones.append(phone)

    def append_word(self, pron, pos, ctype, cform):
        word = Word(pron, _POS.get(pos, tables.JPCOMMON_POS_LIST[1]),
                    _CTYPE.get(ctype, tables.JPCOMMON_CTYPE_LIST[1]),
                    _CFORM.get(cform, tables.JPCOMMON_CFORM_LIST[1]), len(self.words))
        self.words.append(word)
        return word

    def phrase(self, word, accent, chain):
        if len(self.words) > 1 and chain == 1:
            phrase = self.phrases[-1]
        else:
            pause = len(self.words) > 1 and self.phones[self.words[-2].moras[-1].phones[-1].index + 1].text == 'pau'
            if not self.breaths or pause:
                self.breaths.append(Breath(len(self.breaths)))
            breath = self.breaths[-1]
            phrase = Phrase(accent, len(self.phrases), breath)
            self.phrases.append(phrase)
            breath.phrases.append(phrase)
        phrase.words.append(word)
        word.phrase = phrase

    def push_word(self, pron, pos, ctype, cform, accent, chain):
        if self.punctuation(pron):
            return
        position = 0
        word = None
        while position < len(pron):
            character = pron[position]
            if character == 'ー':
                if self.phones and not self.short_pause:
                    self.append_mora('ー', self.words[-1], (self.phones[-1].text,))
                position += 1
                continue
            if character == '’':
                if self.phones and word is not None:
                    self.phones[-1].text = _UNVOICE.get(self.phones[-1].text, self.phones[-1].text)
                position += 1
                continue
            for mora, first, second in _MORAS.get(character, ()):
                if pron.startswith(mora, position):
                    self.pause()
                    if word is None:
                        word = self.append_word(pron[position:], pos, ctype, cform)
                    self.append_mora(mora, word, (first,) if second is None else (first, second))
                    position += len(mora)
                    break
            else:
                break
        if word is not None:
            self.phrase(word, accent, chain)


def from_njd(njd):
    label = Label()
    for node in njd.nodes():
        groups = tuple(node.get(name) for name in ('pos', 'pos_group1', 'pos_group2', 'pos_group3'))
        pron = node.get('pron')
        if groups[3] == '数の区切り':
            pron = ''
            groups = (*groups[:3], '*')
        pos = _NJD_POS.get(groups, tables.NJD2JPCOMMON_POS_LIST[4])
        ctype = _NJD_CTYPE.get(node.get('ctype'), tables.NJD2JPCOMMON_CTYPE_LIST[1])
        cform = _NJD_CFORM.get(node.get('cform'), tables.NJD2JPCOMMON_CFORM_LIST[1])
        label.push_word(pron, pos, ctype, cform, node.acc, node.chain_flag)
    return label
