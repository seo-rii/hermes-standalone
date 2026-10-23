from __future__ import annotations
import re as _re
import unicodedata as _ud
from dataclasses import dataclass
from typing import List, Tuple
TAAMIM = '[\\u0591-\\u05AF]'
NIQQUD_ALL = '[\\u0591-\\u05C7]'
DAGESH = 'ּ'
SHIN_DOT = 'ׁ'
SIN_DOT = 'ׂ'
GERESH = '׳'
GERSHAYIM = '״'
SHEVA = 'ְ'
HATAF_SEGOL = 'ֱ'
HATAF_PATAH = 'ֲ'
HATAF_QAMATS = 'ֳ'
HIRIQ = 'ִ'
TSERE = 'ֵ'
SEGOL = 'ֶ'
PATAH = 'ַ'
QAMATS = 'ָ'
HOLAM = 'ֹ'
QUBUTZ = 'ֻ'
QAMATS_QATAN = 'ׇ'
VOWEL_MARKS = {SHEVA, HATAF_SEGOL, HATAF_PATAH, HATAF_QAMATS, HIRIQ, TSERE, SEGOL, PATAH, QAMATS, HOLAM, QUBUTZ, QAMATS_QATAN}
ALEF = 'א'
BET = 'ב'
GIMEL = 'ג'
DALET = 'ד'
HE = 'ה'
VAV = 'ו'
ZAYIN = 'ז'
HET = 'ח'
TET = 'ט'
YOD = 'י'
KAF = 'כ'
KAF_FINAL = 'ך'
LAMED = 'ל'
MEM = 'מ'
MEM_FINAL = 'ם'
NUN = 'נ'
NUN_FINAL = 'ן'
SAMEKH = 'ס'
AYIN = 'ע'
PE = 'פ'
PE_FINAL = 'ף'
TSADI = 'צ'
TSADI_FINAL = 'ץ'
QOF = 'ק'
RESH = 'ר'
SHIN = 'ש'
TAV = 'ת'
FINAL_FORM_BASE = {KAF_FINAL: KAF, MEM_FINAL: MEM, NUN_FINAL: NUN, PE_FINAL: PE, TSADI_FINAL: TSADI}
BEGADKEFAT = {BET, KAF, PE, TAV, DALET, GIMEL}
GERESH_DIGRAPHS = {GIMEL + GERESH: 'd͡ʒ', ZAYIN + GERESH: 'ʒ', TSADI + GERESH: 't͡ʃ'}

@dataclass
class Glyph:
    base: str
    marks: List[str]

    def has(self, mark: str) -> bool:
        return mark in self.marks

    def any(self, marks: set[str]) -> bool:
        return any((m in marks for m in self.marks))

def _nfc(s: str) -> str:
    return _ud.normalize('NFC', s)

def _strip_taamim(s: str) -> str:
    return _re.sub(TAAMIM, '', s)

def _iter_glyphs(word: str) -> List[Glyph]:
    """Split a Hebrew word into base letters + combining marks (niqqud/dots)."""
    word = _nfc(_strip_taamim(word))
    glyphs: List[Glyph] = []
    for ch in word:
        if _ud.combining(ch):
            if glyphs:
                glyphs[-1].marks.append(ch)
        else:
            glyphs.append(Glyph(ch, []))
    return glyphs

def _apply_geresh_digraphs(glyphs: List[Glyph]) -> List[Glyph]:
    """Collapse ג׳ ז׳ צ׳ into placeholder IPA tokens via special base tags."""
    out: List[Glyph] = []
    i = 0
    while i < len(glyphs):
        g = glyphs[i]
        if i + 1 < len(glyphs) and glyphs[i + 1].base == GERESH and (g.base + glyphs[i + 1].base in (GIMEL + GERESH, ZAYIN + GERESH, TSADI + GERESH)):
            ipa = GERESH_DIGRAPHS[g.base + glyphs[i + 1].base]
            out.append(Glyph(f'<IPA:{ipa}>', []))
            i += 2
            continue
        out.append(g)
        i += 1
    return out

def _map_consonant(base: str, marks: List[str], is_final: bool, next_is_vowel: bool) -> str:
    """Return IPA (possibly empty) for a consonant letter considering dots/dagesh."""
    b = FINAL_FORM_BASE.get(base, base)
    if b in (ALEF, AYIN):
        return '<GLT>'
    if b == HE:
        if is_final and DAGESH not in marks:
            return ''
        return 'h'
    if b == YOD:
        return 'j'
    if b == VAV:
        return 'v'
    if b == SHIN:
        if SHIN_DOT in marks:
            return 'ʃ'
        if SIN_DOT in marks:
            return 's'
        return 'ʃ'
    if b == BET:
        return 'b' if DAGESH in marks else 'v'
    if b == KAF:
        if DAGESH in marks:
            return 'k'
        return 'χ'
    if b == PE:
        return 'p' if DAGESH in marks else 'f'
    if b == GIMEL:
        return 'g'
    if b == DALET:
        return 'd'
    if b == HET:
        return 'χ'
    if b == TET:
        return 't'
    if b == LAMED:
        return 'l'
    if b == MEM:
        return 'm'
    if b == NUN:
        return 'n'
    if b == SAMEKH:
        return 's'
    if b == TSADI:
        return 't͡s'
    if b == QOF:
        return 'k'
    if b == RESH:
        return 'ʁ'
    if b == TAV:
        return 't'
    if b == ZAYIN:
        return 'z'
    return ''

def _is_vowel_mark(m: str) -> bool:
    return m in VOWEL_MARKS

def _has_vowel_marks(marks: List[str]) -> bool:
    return any((_is_vowel_mark(m) for m in marks))

def _is_hiriq_yod(curr: Glyph, nxt: Glyph | None) -> bool:
    return curr.has(HIRIQ) and nxt is not None and (nxt.base == YOD) and (not _has_vowel_marks(nxt.marks))

def _is_holam_male(g: Glyph) -> bool:
    return g.base == VAV and g.has(HOLAM) and (not g.has(DAGESH))

def _is_shuruk(curr: Glyph) -> bool:
    return curr.base == VAV and curr.has(DAGESH) and (not any((m in {HOLAM, HIRIQ, TSERE, SEGOL, PATAH, QAMATS, QUBUTZ, QAMATS_QATAN, SHEVA, HATAF_SEGOL, HATAF_PATAH, HATAF_QAMATS} for m in curr.marks)))

def _map_basic_vowel(g: Glyph) -> Tuple[str, bool]:
    """Return (ipa, is_vocalic) from marks on current glyph (not including mater cases)."""
    if g.has(QAMATS_QATAN):
        return ('o', True)
    if g.has(QUBUTZ):
        return ('u', True)
    if g.has(HIRIQ):
        return ('i', True)
    if g.has(TSERE):
        return ('e', True)
    if g.has(SEGOL):
        return ('e', True)
    if g.has(PATAH):
        return ('a', True)
    if g.has(QAMATS):
        return ('a', True)
    if g.has(HATAF_PATAH):
        return ('a', True)
    if g.has(HATAF_SEGOL):
        return ('e', True)
    if g.has(HATAF_QAMATS):
        return ('o', True)
    if g.has(SHEVA):
        return ('ə', False)
    return ('', False)

@dataclass
class Segment:
    onset: List[str]
    nucleus: str
    coda: List[str]
    dagesh: bool = False

def _word_to_segments(word: str) -> List[Segment]:
    """Parse a diacritized word into syllable-like segments with rough nucleus detection."""
    glyphs = _apply_geresh_digraphs(_iter_glyphs(word))
    segs: List[Segment] = []
    onset: List[str] = []
    i = 0
    while i < len(glyphs):
        g = glyphs[i]
        nxt = glyphs[i + 1] if i + 1 < len(glyphs) else None
        is_final = i == len(glyphs) - 1
        if _is_shuruk(g):
            if not onset:
                onset = []
            segs.append(Segment(onset, 'u', []))
            onset = []
            i += 1
            continue
        if _is_hiriq_yod(g, nxt):
            cons = _map_consonant(g.base, g.marks, is_final=False, next_is_vowel=True)
            if cons and cons != '<GLT>':
                onset.append(cons)
            segs.append(Segment(onset, 'i', []))
            onset = []
            i += 2
            continue
        if _is_holam_male(g):
            segs.append(Segment(onset, 'o', []))
            onset = []
            i += 1
            continue
        cons = _map_consonant(g.base, g.marks, is_final=is_final, next_is_vowel=False)
        v, is_voc = _map_basic_vowel(g)
        if is_voc:
            if cons and cons != '<GLT>':
                onset.append(cons)
            segs.append(Segment(onset, v, []))
            onset = []
        elif g.has(SHEVA):
            if cons and cons != '<GLT>':
                onset.append(cons)
            segs.append(Segment(onset, 'ə', [], g.has(DAGESH)))
            onset = []
        elif cons == '<GLT>':
            onset.append('ʔ')
        elif cons:
            onset.append(cons)
        i += 1
    if onset and segs:
        segs[-1].coda.extend(onset)
        onset = []
    return segs

def _resolve_sheva_and_qamats(segs: List[Segment]) -> List[Segment]:
    """Resolve each shva ('ə' nucleus) to na (pronounced /e/) or nach (silent).

    Modern Israeli Hebrew: a shva is *na* (realized as /e/, not schwa) only in a
    few positions; otherwise it is *nach* (silent, closing the previous syllable).
    We treat as na:
      - word-initial shva (prefixes be-/le-/ve-/she-/ke-, etc.),
      - a shva under a consonant with dagesh chazak (gemination),
      - the second of two consecutive shvas (the first is nach).
    Everything else is nach. Defaulting to nach matches how the marks are actually
    spoken (careful modern reading) and avoids the spurious schwas that made words
    like nir'eit / le'eineinu unintelligible.
    """
    prev_silenced_shva = False
    for i, s in enumerate(segs):
        if s.nucleus != 'ə':
            prev_silenced_shva = False
            continue
        dagesh_chazak = s.dagesh and i > 0
        na = i == 0 or dagesh_chazak or prev_silenced_shva
        if na:
            s.nucleus = 'e'
            prev_silenced_shva = False
        else:
            if i - 1 >= 0:
                segs[i - 1].coda.extend(s.onset)
            s.onset = []
            s.nucleus = ''
            prev_silenced_shva = True
    merged: List[Segment] = []
    for s in segs:
        if s.nucleus == '':
            if merged:
                merged[-1].coda.extend(s.onset)
            else:
                merged.append(s)
            continue
        merged.append(s)
    segs = merged
    return segs

def _syllabify_to_ipa(segs: List[Segment]) -> Tuple[str, List[Tuple[int, int]]]:
    """Join segments to IPA string and compute stress placement.
    Returns (ipa, syllable_spans) where spans are offsets (start,end) in the string for each syllable nucleus.
    """
    syls: List[str] = []
    for s in segs:
        syl = ''.join(s.onset) + s.nucleus + ''.join(s.coda)
        syls.append(syl)
    stress_index = len(syls) - 1
    if len(syls) == 2 and segs[-1].coda:
        stress_index = 0
    pieces = []
    for idx, s in enumerate(segs):
        syl = ''.join(s.onset) + s.nucleus + ''.join(s.coda)
        if idx == stress_index:
            before = ''.join(s.onset)
            after = ''.join(s.coda)
            syl = before + 'ˈ' + s.nucleus + after
        pieces.append(syl)
    ipa = ''.join(pieces)
    return (ipa, [])

def hebrew_word_to_ipa(word: str) -> str:
    """Convert a single fully-diacritized Hebrew word to IPA."""
    segs = _word_to_segments(word)
    segs = _resolve_sheva_and_qamats(segs)
    ipa, _ = _syllabify_to_ipa(segs)
    ipa = _re.sub('ʔ(?=ˈ?[aeiouə])', 'ʔ', ipa)
    ipa = _re.sub('(?<=^)ʔ', 'ʔ', ipa)
    ipa = _re.sub('ʔ(?=[^aeiouəˈ]|$)', '', ipa)
    ipa = ipa.replace('͡', '')
    return ipa

def hebrew_to_ipa(text: str) -> str:
    """Convert a sentence (space-separated words) to IPA."""
    text = _nfc(_strip_taamim(text))
    words = text.split()
    out = []
    for w in words:
        out.append(hebrew_word_to_ipa(w))
    return ' '.join(out)
