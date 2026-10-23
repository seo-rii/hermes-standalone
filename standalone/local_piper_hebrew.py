"""Hebrew diacritization (niqqud) with Nakdimon.

Hebrew is normally written without niqqud (vowel points), but espeak-ng needs the
points to produce correct phonemes. This module restores them with the Nakdimon
model, mirroring the Arabic ``libtashkeel`` path in :mod:`piper.tashkeel`.

Inference-only port of Nakdimon (https://github.com/elazarg/nakdimon, MIT).
The character tables and merge logic are vendored from ``nakdimon/hebrew.py`` and
``nakdimon/dataset.py`` so the package is self-contained (no upstream/TensorFlow
dependency at runtime). See ``LICENSE`` and ``SOURCE`` in this directory.
"""
from .local_piper_nakdimon import NakdimonModel
import re
from pathlib import Path
from typing import List, Optional, Union
HEBREW_DIR = Path(__file__).with_name('assets') / 'piper' / 'hebrew'
DEFAULT_MODEL_PATH = HEBREW_DIR / 'nakdimon.onnx'
RAFE = 'ֿ'
HEBREW_LETTERS = [chr(c) for c in range(1488, 1514 + 1)]
NIQQUD = [RAFE] + [chr(c) for c in range(1456, 1468 + 1)] + ['ַ']
SHIN_YEMANIT = 'ׁ'
SHIN_SMALIT = 'ׂ'
NIQQUD_SIN = [RAFE, SHIN_YEMANIT, SHIN_SMALIT]
DAGESH_LETTER = 'ּ'
DAGESH = [RAFE, DAGESH_LETTER]
VALID_LETTERS = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?'] + HEBREW_LETTERS
SPECIAL_TOKENS = ['H', 'O', '5']
ENDINGS_TO_REGULAR = dict(zip('ךםןףץ', 'כמנפצ'))
_NIQQUD_PATTERN = re.compile('[ְ-ׇּֿׁׂ]')

def remove_niqqud(text: str) -> str:
    """Strip any existing points so the model sees bare consonants."""
    return _NIQQUD_PATTERN.sub('', text)

def normalize(c: str) -> str:
    """Map a character to the model's input alphabet."""
    if c in VALID_LETTERS:
        return c
    if c in ENDINGS_TO_REGULAR:
        return ENDINGS_TO_REGULAR[c]
    if c in ('\n', '\t'):
        return ' '
    if c in ('־', '‒', '–', '—', '―', '−'):
        return '-'
    if c == '[':
        return '('
    if c == ']':
        return ')'
    if c in ('´', '‘', '’'):
        return "'"
    if c in ('“', '”', '״'):
        return '"'
    if c.isdigit():
        return '5'
    if c == '…':
        return ','
    if c in ('ײ', 'װ', 'ױ'):
        return 'H'
    return 'O'

def can_dagesh(letter: str) -> bool:
    return letter in 'בגדהוזטיכלמנספצקשת' + 'ךף'

def can_sin(letter: str) -> bool:
    return letter == 'ש'

def can_niqqud(letter: str) -> bool:
    return letter in 'אבגדהוזחטיכלמנסעפצקרשת' + 'ךן'
_LETTER_CHARS = [''] + SPECIAL_TOKENS + VALID_LETTERS
_NIQQUD_CHARS = [''] + NIQQUD
_DAGESH_CHARS = [''] + DAGESH
_SIN_CHARS = [''] + NIQQUD_SIN
_CHAR_TO_ID = {c: i for i, c in enumerate(_LETTER_CHARS)}

class NakdimonDiacritizer:
    """Add niqqud to Hebrew text with Nakdimon."""

    def __init__(self, model_path: Union[str, Path]=DEFAULT_MODEL_PATH) -> None:
        self.model = NakdimonModel(model_path)

    def __call__(self, text: str, taskeen_threshold: Optional[float]=None) -> str:
        """Add niqqud (``taskeen_threshold`` accepted for API parity, unused)."""
        return self.diacritize(text)

    def diacritize(self, text: str) -> str:
        """Return ``text`` with niqqud restored."""
        text = remove_niqqud(text)
        letters = list(text)
        if not letters:
            return text
        input_ids = [_CHAR_TO_ID[normalize(c)] for c in letters]
        n_out, d_out, s_out = self.model.run(input_ids)
        niqqud = [max(range(len(row)), key=row.__getitem__) for row in n_out]
        dagesh = [max(range(len(row)), key=row.__getitem__) for row in d_out]
        sin = [max(range(len(row)), key=row.__getitem__) for row in s_out]
        out: List[str] = []
        for letter, n_id, d_id, s_id in zip(letters, niqqud, dagesh, sin):
            out.append(letter)
            if can_dagesh(letter):
                out.append(_DAGESH_CHARS[d_id])
            if can_sin(letter):
                out.append(_SIN_CHARS[s_id])
            if can_niqqud(letter):
                out.append(_NIQQUD_CHARS[n_id])
        return ''.join(out).replace(RAFE, '')
