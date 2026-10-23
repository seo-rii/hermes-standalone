"""Owned source-order OpenJTalk native-boundary replacement for Piper."""
from functools import wraps
from pathlib import Path
from threading import RLock

from ..accent import set_accent_phrase, set_accent_type
from ..digit import set_digit
from ..jpcommon_labels import make_label as create_labels
from ..mecab import Mecab
from ..njd import NJD, NJDNode, STRING_FIELDS
from ..normalize import text_to_mecab
from ..pronunciation import set_pronunciation
from ..unvoiced import set_long_vowel, set_unvoiced_vowel

from ._known_symbols import KNOWN_SYMBOL_FEATURES
from .original_pipeline import bound_features, expand_features, unbound_features
from .original_rules import (
    _apply_original_rule_before_chaining, _expand_symbol_feature,
    _mark_numeral_space_boundaries, _restore_unknown_katakana_features,
)

ROOT = Path(__file__).resolve().parent.parent / 'data'


def locked(function):
    @wraps(function)
    def invoke(self, *args, **kwargs):
        with self._lock:
            return function(self, *args, **kwargs)
    return invoke


def export(njd):
    result = []
    for node in njd.nodes():
        values = {name: node.get(name) for name in STRING_FIELDS}
        values.update(acc=node.acc, mora_size=node.mora_size, chain_flag=node.chain_flag,
                      is_reading_protected=False, is_reading_rewritten=False)
        result.append(values)
    return result


def from_features(features):
    njd = NJD()
    for feature in features:
        node = NJDNode()
        for name in STRING_FIELDS:
            value = feature[name]
            if not isinstance(value, str):
                raise TypeError('NJD feature field must be str: ' + name)
            if '\0' in value:
                raise ValueError('NJD feature field contains null character: ' + name)
            node.set(name, value)
        for name in ('acc', 'mora_size', 'chain_flag'):
            value = feature[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError('NJD feature field must be int: ' + name)
            if not -(1 << 31) <= value < 1 << 31:
                raise OverflowError('value too large to convert to int')
            node.set(name, value)
        njd.append(node)
    return njd


def morph(node, raw):
    surface = raw[node.start:node.start + node.length].decode('utf-8', 'replace')
    begin = len(raw[:node.start].decode('utf-8'))
    end = len(raw[:node.start + node.length].decode('utf-8'))
    return dict(surface=surface, features=(surface + ',' + node.feature).split(','),
                char_span=(begin, end), pos_id=node.posid, left_id=node.left, right_id=node.right,
                word_cost=node.word_cost, link_cost=node.cost - node.previous.cost, node_cost=node.cost,
                is_unknown=node.unknown, is_ignored='記号,空白' in node.feature, dictionary_index=node.dictionary,
                is_tsqyomi_selected=False, is_tsqyomi_target=False)


def expand_morph(node, value):
    surface = value['surface']
    if not (value['is_unknown'] and len(surface) > 1 and all(not char.isalnum() for char in surface)):
        return [value]
    result = []
    for index, char in enumerate(surface):
        known = KNOWN_SYMBOL_FEATURES.get(char)
        left, right, cost, feature = known or (node.left, node.right, node.word_cost, node.feature)
        begin = value['char_span'][0] + index
        result.append(dict(surface=char, features=(char + ',' + feature).split(','), char_span=(begin, begin + 1),
                           pos_id=node.posid, left_id=left, right_id=right, word_cost=cost,
                           link_cost=value['link_cost'] if index == 0 else 0, node_cost=value['node_cost'],
                           is_unknown=known is None, is_ignored='記号,空白' in feature,
                           dictionary_index=0 if known else value['dictionary_index'],
                           is_tsqyomi_selected=False, is_tsqyomi_target=False))
    return result


class OpenJTalk:
    def __init__(self, dn_mecab=None):
        self.mecab = Mecab(dn_mecab or ROOT / 'openjtalk/dictionary')
        self.userdic_reading_protection = ()
        self.userdic_paths = ()
        self._lock = RLock()

    def normalize_for_mecab(self, text):
        if isinstance(text, str):
            text = text.encode('utf-8')
        status, output = text_to_mecab(text.replace(b'\0', b''), 16384)
        if status == 2:
            raise RuntimeError('Input text is too long after normalization')
        if status:
            raise RuntimeError('Invalid arguments for text2mecab')
        return output.decode('utf-8')

    @locked
    def run_mecab(self, text):
        raw = self.normalize_for_mecab(text).encode()
        features = [part for row in self.mecab.features(raw) for part in _expand_symbol_feature(row)]
        return _mark_numeral_space_boundaries(features)

    @locked
    def run_mecab_detailed(self, text):
        raw = self.normalize_for_mecab(text).encode()
        nodes = self.mecab.parse(raw)
        features = []
        morphs = []
        for node in nodes:
            value = morph(node, raw)
            features.extend(_expand_symbol_feature(','.join(value['features'])))
            morphs.extend(expand_morph(node, value))
        return _mark_numeral_space_boundaries(features), morphs

    @locked
    def run_njd_from_mecab(self, mecab_features, restore_unknown_katakana=False, modify_numeral_reading=True):
        if not mecab_features:
            return []
        for feature in mecab_features:
            if not isinstance(feature, str):
                raise TypeError('Each MeCab feature must be str')
            if '\0' in feature:
                raise ValueError('MeCab feature must not contain null characters')
        njd = NJD(expand_features(mecab_features))
        set_pronunciation(njd)
        feature = export(njd)
        if restore_unknown_katakana:
            feature = _restore_unknown_katakana_features(feature, mecab_features)
        feature = _apply_original_rule_before_chaining(feature, modify_numeral_reading=modify_numeral_reading)
        bounded, has_boundary = bound_features(feature)
        njd = from_features(bounded)
        set_digit(njd)
        if has_boundary:
            njd = from_features(unbound_features(export(njd)))
        set_accent_phrase(njd)
        set_accent_type(njd)
        set_unvoiced_vowel(njd)
        set_long_vowel(njd)
        return export(njd)

    @locked
    def run_frontend(self, text, restore_unknown_katakana=False, modify_numeral_reading=True):
        return self.run_njd_from_mecab(self.run_mecab(text), restore_unknown_katakana, modify_numeral_reading)

    @locked
    def run_frontend_detailed(self, text, restore_unknown_katakana=False, modify_numeral_reading=True):
        features, morphs = self.run_mecab_detailed(text)
        return self.run_njd_from_mecab(features, restore_unknown_katakana, modify_numeral_reading), morphs

    @locked
    def set_unvoiced_vowel(self, features):
        features = list(features)
        njd = from_features(features)
        set_unvoiced_vowel(njd)
        result = export(njd)
        for current, previous in zip(result, features, strict=True):
            current['is_reading_protected'] = previous.get('is_reading_protected', False)
            current['is_reading_rewritten'] = previous.get('is_reading_rewritten', False)
        return result

    @locked
    def make_label(self, features):
        return create_labels(from_features(feature for feature in features if feature['pron'] != ''))

    @locked
    def analyze_mecab_candidates(self, text, target_spans):
        raw = self.normalize_for_mecab(text).encode()
        nodes = []
        for begin, end in target_spans:
            offset = len(raw.decode()[:begin].encode())
            for node in self.mecab.candidates(raw, offset, len(raw)):
                surface = raw[node.start:node.start + node.length].decode()
                if begin + len(surface) != end:
                    continue
                nodes.append({'surface': surface, 'feature': surface + ',' + node.feature,
                              'is_unknown': node.unknown, 'word_cost': node.word_cost})
        return {'nodes': nodes}
