"""Original OpenJTalk kanji-numeral constraints for its MeCab best path."""
_DIGITS = frozenset(character.encode() for character in '〇一二三四五六七八九')
_PLACES = frozenset(character.encode() for character in '十百千万億兆')
_HYPHENS = tuple(character.encode() for character in '−－‐‑‒–—ー-')


def _kanji(text, position, digits_only=False):
    part = text[position:position + 3]
    return part in _DIGITS or not digits_only and part in _PLACES


def _word(text, node):
    return node.length > 0 and node.length % 3 == 0 and all(_kanji(text, point) for point in range(node.start, node.start + node.length, 3))


def _numeric(node):
    return node.feature.startswith('名詞,数,')


def _digit(text, position):
    part = text[position:position + 3]
    return len(part) == 3 and (part[:2] == b'\xef\xbc' and 0x90 <= part[2] <= 0x99 or part in _DIGITS)


def _hyphen(text, position, before=False):
    for hyphen in _HYPHENS:
        start = position - len(hyphen) if before else position
        if start >= 0 and text[start:start + len(hyphen)] == hyphen:
            return len(hyphen)
    return 0


def _junctions(text, nodes):
    boundaries = []
    previous = None
    chain_start = 0
    for node in nodes:
        position = node.start
        adjacent = previous is not None and previous.start + previous.length == position
        if adjacent and _word(text, previous) and _kanji(text, position):
            if _numeric(previous) and node.length > 3 or _numeric(node) and previous.length > 3:
                end = position
                while end < position + node.length and _kanji(text, end):
                    end += 3
                boundaries.extend(range(chain_start, end + 1, 3))
        if _word(text, node) and (not adjacent or not _word(text, previous)):
            chain_start = position
        previous = node
    return boundaries


def _telephone_runs(text):
    boundaries = []
    position = 0
    while position < len(text):
        if not _kanji(text, position, True):
            position += 1
            continue
        start = position
        while _kanji(text, position, True):
            position += 3
        before = _hyphen(text, start, True)
        after = _hyphen(text, position)
        if before and _digit(text, start - before - 3) or after and _digit(text, position + after):
            boundaries.extend(range(start, position + 1, 3))
    return boundaries


def numeral_boundaries(text, nodes):
    return tuple(sorted(set(_junctions(text, nodes) + _telephone_runs(text))))
