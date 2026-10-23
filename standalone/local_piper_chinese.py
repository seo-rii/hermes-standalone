"""Owned original Piper Chinese number/pinyin/ID source (Apache-2.0)."""
from __future__ import annotations
import logging, re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, Union
from .local_piper_const import BOS, EOS, PAD
from .local_piper_phoneme_ids import DEFAULT_PHONEME_ID_MAP
_LOGGER = logging.getLogger(__name__)
_LOGGER = logging.getLogger(__name__)
PINYIN_INITIALS = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']
PHONEME_TO_ID: dict[str, list[int]] = {PAD: DEFAULT_PHONEME_ID_MAP[PAD], BOS: DEFAULT_PHONEME_ID_MAP[BOS], EOS: DEFAULT_PHONEME_ID_MAP[EOS], 'Ø': [3], 'b': [4], 'p': [5], 'm': [6], 'f': [7], 'd': [8], 't': [9], 'n': [10], 'l': [11], 'g': [12], 'k': [13], 'h': [14], 'j': [15], 'q': [16], 'x': [17], 'zh': [18], 'ch': [19], 'sh': [20], 'r': [21], 'z': [22], 'c': [23], 's': [24], 'y': [25], 'w': [26], 'a': [27], 'o': [28], 'e': [29], 'ai': [30], 'ei': [31], 'ao': [32], 'ou': [33], 'an': [34], 'en': [35], 'ang': [36], 'eng': [37], 'ong': [38], 'i': [39], 'ia': [40], 'ie': [41], 'iao': [42], 'iu': [43], 'ian': [44], 'in': [45], 'iang': [46], 'ing': [47], 'iong': [48], 'u': [49], 'ua': [50], 'uo': [51], 'uai': [52], 'ui': [53], 'uan': [54], 'un': [55], 'uang': [56], 'ueng': [57], 'v': [58], 've': [59], 'van': [60], 'vn': [61], 'er': [62], 'ue': [63], '1': [64], '2': [65], '3': [66], '4': [67], '5': [68], '。': [69], '.': [69], '？': [70], '?': [70], '！': [71], '!': [71], '—': [72], '…': [72], '、': [72], '，': [72], ',': [72], '：': [72], ':': [72], '；': [72], ';': [72], ' ': [72]}
GROUP_END_PHONEMES = {'1', '2', '3', '4', '5', '。', '？', '！', '.', '?', '!', '—', '…', '、', '，', '：', '；', ',', ':', ';', ' '}
G2PW_URL = 'https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/zh/zh_CN/_resources/g2pw.tar.gz?download=true'
TEMP_PATTERN = re.compile('(?P<sign>[-−])?(?P<num>\\d+)\\s*(?:°\\s*C|℃)')
PERCENT_PATTERN = re.compile('(?P<num>-?\\d+(?:\\.\\d+)?|[零〇一二三四五六七八九十百千万亿两点]+)\\s*(?:%|％)')

class ChineseFrontend:
    """Phonemize Chinese text using g2pW."""

    def __init__(self, converter):
        from unicode_rbnf import RbnfEngine
        self.g2p = converter
        self.number_engine = RbnfEngine.for_language('zh')

    def phonemize(self, text: str) -> list[list[str]]:
        """Turn text into phonemes per sentence."""
        from .local_piper_chinese_sentence import stream_to_sentences
        text = re.sub('[“”"]', '', text)
        all_phonemes: list[list[str]] = []
        for sentence in stream_to_sentences([text]):
            sentence = self._numbers_to_words(sentence)
            sylls = self.g2p(sentence)[0]
            sentence_phonemes = []
            for syl, syl_char in zip(sylls, sentence):
                if syl is None:
                    if syl_char in PHONEME_TO_ID:
                        sentence_phonemes.append(syl_char)
                    continue
                syl = _normalize_g2pw_syllable(syl)
                ini_p, fin_p, tone = _split_initial_final_tone(syl)
                if not fin_p:
                    sentence_phonemes.append(syl)
                    continue
                if not ini_p:
                    ini_p = 'Ø'
                for sym in (ini_p, fin_p, tone):
                    assert sym, sym
                    sentence_phonemes.append(sym)
            all_phonemes.append(sentence_phonemes)
        return all_phonemes

    def _numbers_to_words(self, text: str) -> str:

        def replace_temp(m: re.Match) -> str:
            sign = m.group('sign')
            num_str = m.group('num')
            num_words = self._zh_number(num_str)
            if sign:
                return f'零下{num_words}度'
            return f'{num_words}度'
        text = TEMP_PATTERN.sub(replace_temp, text)

        def replace_percent(m: re.Match) -> str:
            num_str = m.group('num')
            if re.fullmatch('-?\\d+(?:\\.\\d+)?', num_str):
                num_words = self._zh_number(num_str)
            else:
                num_words = num_str
            return f'百分之{num_words}'
        text = PERCENT_PATTERN.sub(replace_percent, text)
        return re.sub('-?\\d+(?:\\.\\d+)?', lambda m: self._zh_number(m.group(0)), text)

    def _zh_number(self, text: str) -> str:
        return self.number_engine.format_number(text).text

def phonemes_to_ids(phonemes: list[str], id_map: Optional[Mapping[str, Sequence[int]]]=None) -> list[int]:
    """Get phoneme ids for phonemes.

    Padding is done after a group of phonemes representing one pinyin group
    instead of between each phoneme.
    """
    if not id_map:
        id_map = PHONEME_TO_ID
    ids: list[int] = []
    ids.extend(id_map[BOS])
    for phoneme in phonemes:
        if phoneme not in id_map:
            _LOGGER.warning('Missing phoneme from id map: %s', phoneme)
            continue
        ids.extend(id_map[phoneme])
        if phoneme in GROUP_END_PHONEMES:
            ids.extend(id_map[PAD])
    ids.extend(id_map[EOS])
    return ids

def _normalize_g2pw_syllable(syl: str) -> str:
    """
    - Keep only syllables like [a-züv:]+[1-5]
    - Convert g2pW's 'u:' and 'ü' -> 'v' (ü-family)
      nu:3   -> nv3        (n + v)
      lu:e4  -> lve4       (l + ve)
      ju:an3 -> jvan3      (j + van)
      ju:n3  -> jvn3       (j + vn)
    """
    m = re.match('^([a-züv:]+?)([1-5])$', syl)
    if not m:
        return syl
    base, tone = (m.group(1), m.group(2))
    base = base.replace('u:', 'v').replace('ü', 'v')
    return base + tone

def _split_initial_final_tone(syl: str):
    """
    'hang2' -> ('h', 'ang', '2')
    'ai3'   -> ('', 'ai', '3')
    """
    m = re.match('^([a-zvü]+?)([1-5])$', syl)
    if not m:
        return ('', '', None)
    base, tone = (m.group(1), m.group(2))
    ini = ''
    for cand in PINYIN_INITIALS:
        if base.startswith(cand):
            ini = cand
            break
    fin = base[len(ini):] if ini else base
    return (ini, fin, tone)

class ChinesePhonemizer(ChineseFrontend):
    """Run the original Chinese frontend using the owned quantized BERT graph."""

    def __init__(self, model_dir):
        from .local_piper_chinese_controller import G2PWTextController
        from .local_piper_chinese_model import G2PWModel
        model_dir = Path(model_dir)
        download_model(model_dir)
        resources = Path(__file__).with_name('piper_chinese_data')
        converter = G2PWTextController(model_dir, resources / 'bert-base-chinese', resources / 'g2pw', G2PWModel(model_dir / 'g2pw.onnx'), style='pinyin', enable_non_tradional_chinese=True)
        super().__init__(converter)

def download_model(model_dir):
    """Populate the original per-voice directory from embedded reviewed data."""
    import shutil
    model_dir = Path(model_dir)
    if (model_dir / 'g2pw.onnx').exists():
        _LOGGER.debug('Found g2pW model at %s', model_dir / 'g2pw.onnx')
        return
    resources = Path(__file__).with_name('piper_chinese_data') / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ('g2pw.onnx', 'config.py', 'POLYPHONIC_CHARS.txt', 'MONOPHONIC_CHARS.txt', 'version'):
        shutil.copyfile(resources / name, model_dir / name)
