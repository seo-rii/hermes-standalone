"""Original complete OpenJTalk full-context label fields A through K."""
from .jpcommon import from_njd


def _limit(value, maximum, minimum=1):
    return min(maximum, max(minimum, value))


def _neighbor(sequence, index):
    return sequence[index] if 0 <= index < len(sequence) else None


def _phrase_numbers(phrase):
    return (_limit(phrase.count, 49), _limit(phrase.count if phrase.accent == 0 else phrase.accent, 49),
            phrase.emotion or '0', phrase.exclamation or '0')


class _Context:
    def __init__(self, label, phone, position):
        self.label = label
        self.phone = phone
        self.leading = position == 0
        self.trailing = position == len(label.phones) + 1
        self.pause = phone.text == 'pau'
        self.boundary = self.leading or self.trailing or self.pause
        self.word = None if self.pause else phone.mora.word

    def accent_position(self):
        if self.boundary:
            return '/A:xx+xx+xx'
        phrase = self.word.phrase
        index = self.phone.mora.index - phrase.head.index + 1
        nucleus = phrase.count if phrase.accent == 0 else phrase.accent
        return '/A:%d+%d+%d' % (_limit(index - nucleus, 49, -49), _limit(index, 49), _limit(phrase.count - index + 1, 49))

    def word_fields(self):
        if self.pause:
            before = self.label.phones[self.phone.index - 1].mora.word
            after = self.label.phones[self.phone.index + 1].mora.word
        else:
            before = None if self.word.index == 0 else self.word if self.trailing else self.label.words[self.word.index - 1]
            after = None if self.word.index + 1 == len(self.label.words) else self.word if self.leading else self.label.words[self.word.index + 1]
        b = '/B:xx-xx_xx' if before is None else '/B:%s-%s_%s' % (before.pos, before.ctype, before.cform)
        c = '/C:xx_xx+xx' if self.boundary else '/C:%s_%s+%s' % (self.word.pos, self.word.ctype, self.word.cform)
        d = '/D:xx+xx_xx' if after is None else '/D:%s+%s_%s' % (after.pos, after.ctype, after.cform)
        return b + c + d

    def adjacent_phrases(self):
        if self.pause:
            return (self.label.phones[self.phone.index - 1].mora.word.phrase,
                    self.label.phones[self.phone.index + 1].mora.word.phrase)
        phrase = self.word.phrase
        before = phrase if self.trailing else _neighbor(self.label.phrases, phrase.index - 1)
        after = phrase if self.leading else _neighbor(self.label.phrases, phrase.index + 1)
        return before, after

    def previous_phrase(self, phrase):
        value = '/E:xx_xx!xx_xx' if phrase is None else '/E:%d_%d!%s_%s' % _phrase_numbers(phrase)
        if self.boundary or phrase is None:
            return value + '-xx'
        following = self.label.phones[phrase.tail.phones[-1].index + 1]
        return value + ('-0' if following.text == 'pau' else '-1')

    def current_phrase(self):
        if self.boundary:
            return '/F:xx_xx#xx_xx@xx_xx|xx_xx'
        phrase = self.word.phrase
        breath = phrase.breath
        index = phrase.index - breath.phrases[0].index + 1
        mora = phrase.head.index - breath.head.index + 1
        return '/F:%d_%d#%s_%s@%d_%d|%d_%d' % (*_phrase_numbers(phrase), _limit(index, 49),
                _limit(len(breath.phrases) - index + 1, 49), _limit(mora, 99), _limit(breath.count - mora + 1, 99))

    def following_phrase(self, phrase):
        value = '/G:xx_xx%xx_xx' if phrase is None else '/G:%d_%d%%%s_%s' % _phrase_numbers(phrase)
        if self.boundary or phrase is None:
            return value + '_xx'
        previous = self.label.phones[phrase.head.phones[0].index - 1]
        return value + ('_0' if previous.text == 'pau' else '_1')

    def phrase_fields(self):
        before, after = self.adjacent_phrases()
        return self.previous_phrase(before) + self.current_phrase() + self.following_phrase(after)

    def adjacent_breaths(self):
        if self.pause:
            return (self.label.phones[self.phone.index - 1].mora.word.phrase.breath,
                    self.label.phones[self.phone.index + 1].mora.word.phrase.breath)
        breath = self.word.phrase.breath
        before = breath if self.trailing else _neighbor(self.label.breaths, breath.index - 1)
        after = breath if self.leading else _neighbor(self.label.breaths, breath.index + 1)
        return before, after

    def current_breath(self):
        if self.boundary:
            return '/I:xx-xx@xx+xx&xx-xx|xx+xx'
        breath = self.word.phrase.breath
        breath_index = breath.index + 1
        phrase_index = breath.phrases[0].index + 1
        mora_index = breath.head.index + 1
        return '/I:%d-%d@%d+%d&%d-%d|%d+%d' % (
            _limit(len(breath.phrases), 49), _limit(breath.count, 99), _limit(breath_index, 19),
            _limit(len(self.label.breaths) - breath_index + 1, 19), _limit(phrase_index, 49),
            _limit(len(self.label.phrases) - phrase_index + 1, 49), _limit(mora_index, 199),
            _limit(len(self.label.moras) - mora_index + 1, 199))

    def breath_fields(self):
        before, after = self.adjacent_breaths()
        h = '/H:xx_xx' if before is None else '/H:%d_%d' % (_limit(len(before.phrases), 49), _limit(before.count, 99))
        j = '/J:xx_xx' if after is None else '/J:%d_%d' % (_limit(len(after.phrases), 49), _limit(after.count, 99))
        k = '/K:%d+%d-%d' % (_limit(len(self.label.breaths), 19), _limit(len(self.label.phrases), 49), _limit(len(self.label.moras), 199))
        return h + self.current_breath() + j + k


def make_label(njd):
    label = from_njd(njd)
    if not label.phones:
        return []
    phones = ['xx', 'xx', 'sil', *(phone.text for phone in label.phones), 'sil', 'xx', 'xx']
    output = []
    for position in range(len(label.phones) + 2):
        phone = label.phones[min(max(position - 1, 0), len(label.phones) - 1)]
        context = _Context(label, phone, position)
        value = '%s^%s-%s+%s=%s' % tuple(phones[position:position + 5])
        value += context.accent_position() + context.word_fields() + context.phrase_fields() + context.breath_fields()
        output.append(value[:1023])
    return output
