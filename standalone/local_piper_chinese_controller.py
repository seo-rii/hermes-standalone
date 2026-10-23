"""Owned original Piper g2pW text/controller source (Apache-2.0)."""
from __future__ import annotations
import json, logging, math, os, re
from pathlib import Path
from typing import Any, Optional, Sequence, Union
from .local_piper_tensor import Tensor
from .local_piper_chinese_tokenizer import BertWordPiece
_LOGGER = logging.getLogger(__name__)
_CONFIG_DEFAULTS = {'model_source': 'bert-base-chinese', 'window_size': 32, 'batch_size': 256, 'use_mask': True, 'use_char_phoneme': False}
POS_TAGS = ['UNK', 'A', 'C', 'D', 'I', 'N', 'P', 'T', 'V', 'DE', 'SHI']

def _pad_stack(sequences):
    width = max((len(sequence) for sequence in sequences))
    return Tensor.make((len(sequences), width), (item for sequence in sequences for item in [*sequence, *[0] * (width - len(sequence))]), 7)

def _collate(samples):
    return {'input_ids': _pad_stack([row['input_ids'] for row in samples]), 'token_type_ids': _pad_stack([row['token_type_ids'] for row in samples]), 'attention_mask': _pad_stack([row['attention_mask'] for row in samples]), 'phoneme_mask': Tensor.make((len(samples), len(samples[0]['phoneme_mask'])), (value for row in samples for value in row['phoneme_mask'])), 'char_ids': Tensor.make((len(samples),), [row['char_id'] for row in samples], 7), 'position_ids': Tensor.make((len(samples),), [row['position_id'] for row in samples], 7)}

def _argmax_rows(tensor):
    width = tensor.shape[-1]
    if width == 0:
        raise ValueError('attempt to get argmax of an empty sequence')
    result = []
    for start in range(0, len(tensor.values), width):
        best = 0
        value = tensor.values[start]
        for index in range(1, width):
            if math.isnan(value):
                break
            other = tensor.values[start + index]
            if math.isnan(other) or other > value:
                best = index
                value = other
        result.append(best)
    return result

def wordize_and_map(text: str):
    """Split text into words, mapping indices both ways.

    Runs of ASCII alphanumerics become one word, every other character becomes
    its own word, and spaces map to None.
    """
    words: list[str] = []
    index_map_from_text_to_word: list[Optional[int]] = []
    index_map_from_word_to_text: list[tuple[int, int]] = []
    while len(text) > 0:
        match_space = re.match('^ +', text)
        if match_space:
            space_str = match_space.group(0)
            index_map_from_text_to_word += [None] * len(space_str)
            text = text[len(space_str):]
            continue
        match_en = re.match('^[a-zA-Z0-9]+', text)
        if match_en:
            en_word = match_en.group(0)
            word_start_pos = len(index_map_from_text_to_word)
            word_end_pos = word_start_pos + len(en_word)
            index_map_from_word_to_text.append((word_start_pos, word_end_pos))
            index_map_from_text_to_word += [len(words)] * len(en_word)
            words.append(en_word)
            text = text[len(en_word):]
        else:
            word_start_pos = len(index_map_from_text_to_word)
            word_end_pos = word_start_pos + 1
            index_map_from_word_to_text.append((word_start_pos, word_end_pos))
            index_map_from_text_to_word += [len(words)]
            words.append(text[0])
            text = text[1:]
    return (words, index_map_from_text_to_word, index_map_from_word_to_text)

def tokenize_and_map(tokenizer, text: str):
    """Tokenize text, returning tokens and index maps to and from the text."""
    words, text2word, word2text = wordize_and_map(text)
    tokens: list[str] = []
    index_map_from_token_to_text: list[tuple[int, int]] = []
    for word, (word_start, word_end) in zip(words, word2text):
        word_tokens = tokenizer.tokenize(word)
        if len(word_tokens) == 0 or word_tokens == ['[UNK]']:
            index_map_from_token_to_text.append((word_start, word_end))
            tokens.append('[UNK]')
        else:
            current_word_start = word_start
            for word_token in word_tokens:
                word_token_len = len(re.sub('^##', '', word_token))
                index_map_from_token_to_text.append((current_word_start, current_word_start + word_token_len))
                current_word_start = current_word_start + word_token_len
                tokens.append(word_token)
    index_map_from_text_to_token = text2word
    for i, (token_start, token_end) in enumerate(index_map_from_token_to_text):
        for token_pos in range(token_start, token_end):
            index_map_from_text_to_token[token_pos] = i
    return (tokens, index_map_from_text_to_token, index_map_from_token_to_text)

def get_phoneme_labels(polyphonic_chars: Sequence[Sequence[str]]):
    """Get bare phoneme labels, plus char to candidate label ids."""
    labels = sorted({phoneme for char, phoneme in polyphonic_chars})
    char2phonemes: dict[str, list[int]] = {}
    for char, phoneme in polyphonic_chars:
        char2phonemes.setdefault(char, []).append(labels.index(phoneme))
    return (labels, char2phonemes)

def get_char_phoneme_labels(polyphonic_chars: Sequence[Sequence[str]]):
    """Get "<char> <phoneme>" labels, plus char to candidate label ids."""
    labels = sorted({f'{char} {phoneme}' for char, phoneme in polyphonic_chars})
    char2phonemes: dict[str, list[int]] = {}
    for char, phoneme in polyphonic_chars:
        char2phonemes.setdefault(char, []).append(labels.index(f'{char} {phoneme}'))
    return (labels, char2phonemes)

class _FeatureBuilder:
    """Builds one set of model inputs per (text, query char).

    Kept as a class with ``__len__``/``__getitem__`` so upstream's recursive
    skip-on-unusable-text behaviour ports across directly.
    """

    def __init__(self, tokenizer, labels: list[str], char2phonemes: dict[str, list[int]], chars: list[str], texts: list[str], query_ids: list[int], use_mask: bool=False, window_size: Optional[int]=None, max_len: int=512) -> None:
        self.tokenizer = tokenizer
        self.labels = labels
        self.char2phonemes = char2phonemes
        self.chars = chars
        self.texts = texts
        self.query_ids = query_ids
        self.use_mask = use_mask
        self.window_size = window_size
        self.max_len = max_len
        if window_size is not None:
            self.truncated_texts, self.truncated_query_ids = self._truncate_texts(window_size, texts, query_ids)

    @staticmethod
    def _truncate_texts(window_size: int, texts, query_ids):
        """Keep a window of characters centered on each query character."""
        truncated_texts = []
        truncated_query_ids = []
        for text, query_id in zip(texts, query_ids):
            start = max(0, query_id - window_size // 2)
            end = min(len(text), query_id + window_size // 2)
            truncated_texts.append(text[start:end])
            truncated_query_ids.append(query_id - start)
        return (truncated_texts, truncated_query_ids)

    def _truncate(self, max_len, text, query_id, tokens, text2token, token2text):
        """Center the token window on the query when the text exceeds max_len."""
        truncate_len = max_len - 2
        if len(tokens) <= truncate_len:
            return (text, query_id, tokens, text2token, token2text)
        token_position = text2token[query_id]
        token_start = token_position - truncate_len // 2
        token_end = token_start + truncate_len
        font_exceed_dist = -token_start
        back_exceed_dist = token_end - len(tokens)
        if font_exceed_dist > 0:
            token_start += font_exceed_dist
            token_end += font_exceed_dist
        elif back_exceed_dist > 0:
            token_start -= back_exceed_dist
            token_end -= back_exceed_dist
        start = token2text[token_start][0]
        end = token2text[token_end - 1][1]
        return (text[start:end], query_id - start, tokens[token_start:token_end], [i - token_start if i is not None else None for i in text2token[start:end]], [(s - start, e - start) for s, e in token2text[token_start:token_end]])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = (self.truncated_texts if self.window_size else self.texts)[idx].lower()
        query_id = (self.truncated_query_ids if self.window_size else self.query_ids)[idx]
        try:
            tokens, text2token, token2text = tokenize_and_map(self.tokenizer, text)
        except Exception:
            _LOGGER.warning('Skipping unusable text: %s', text)
            return self[(idx + 1) % len(self)]
        text, query_id, tokens, text2token, token2text = self._truncate(self.max_len, text, query_id, tokens, text2token, token2text)
        processed_tokens = ['[CLS]'] + tokens + ['[SEP]']
        query_char = text[query_id]
        phoneme_mask = [1 if i in self.char2phonemes[query_char] else 0 for i in range(len(self.labels))] if self.use_mask else [1] * len(self.labels)
        return {'input_ids': self.tokenizer.convert_tokens_to_ids(processed_tokens), 'token_type_ids': [0] * len(processed_tokens), 'attention_mask': [1] * len(processed_tokens), 'phoneme_mask': phoneme_mask, 'char_id': self.chars.index(query_char), 'position_id': text2token[query_id] + 1}

def _read_lines(path: Path) -> list[str]:
    """Read a file and split it into stripped lines."""
    with open(path, 'r', encoding='utf-8') as text_file:
        return text_file.read().strip().split('\n')

class G2PWTextController:

    def __init__(self, model_dir, tokenizer_dir, data_dir, session, style='bopomofo', batch_size=None, enable_non_tradional_chinese=False):
        model_dir = Path(model_dir)
        data_dir = Path(data_dir)
        self.config = self._load_config(model_dir / 'config.py')
        self.batch_size = batch_size or self.config['batch_size']
        self.model_source = self.config['model_source']
        self.tokenizer = BertWordPiece(tokenizer_dir)
        self.session_g2pw = session
        self.polyphonic_chars = [line.split('\t') for line in _read_lines(model_dir / 'POLYPHONIC_CHARS.txt')]
        self.monophonic_chars = [line.split('\t') for line in _read_lines(model_dir / 'MONOPHONIC_CHARS.txt')]
        label_func = get_char_phoneme_labels if self.config['use_char_phoneme'] else get_phoneme_labels
        self.labels, self.char2phonemes = label_func(self.polyphonic_chars)
        self.chars = sorted(self.char2phonemes)
        self.pos_tags = POS_TAGS
        self.bopomofo_convert_dict = json.loads((data_dir / 'bopomofo_to_pinyin_wo_tune_dict.json').read_text())
        self.style_convert_func = {'bopomofo': lambda value: value, 'pinyin': self._convert_bopomofo_to_pinyin}[style]
        self.char_bopomofo_dict = json.loads((data_dir / 'char_bopomofo_dict.json').read_text())
        self.enable_non_tradional_chinese = enable_non_tradional_chinese
        self.s2t_dict = {}
        if enable_non_tradional_chinese:
            for line in _read_lines(data_dir / 'bert-base-chinese_s2t_dict.txt'):
                source, target = line.split('\t')
                self.s2t_dict[source] = target

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        """Read the settings this path uses out of the model's config.py."""
        import importlib.util
        spec = importlib.util.spec_from_file_location('g2pw_config', config_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {key: getattr(module, key, default) for key, default in _CONFIG_DEFAULTS.items()}

    def _convert_bopomofo_to_pinyin(self, bopomofo: str) -> Optional[str]:
        """Convert one bopomofo syllable to pinyin, keeping the tone digit."""
        tone = bopomofo[-1]
        assert tone in '12345'
        component = self.bopomofo_convert_dict.get(bopomofo[:-1])
        if component:
            return component + tone
        _LOGGER.warning('Cannot convert bopomofo to pinyin: %s', bopomofo)
        return None

    def _convert_s2t(self, sentence: str) -> str:
        """Map simplified characters to traditional ones."""
        return ''.join((self.s2t_dict.get(char, char) for char in sentence))

    def __call__(self, sentences: Union[str, list[str]]) -> list[list[Optional[str]]]:
        """Get a phoneme (or None for punctuation) per character per sentence."""
        if isinstance(sentences, str):
            sentences = [sentences]
        if self.enable_non_tradional_chinese:
            translated_sentences = []
            for sentence in sentences:
                translated_sentence = self._convert_s2t(sentence)
                assert len(translated_sentence) == len(sentence)
                translated_sentences.append(translated_sentence)
            sentences = translated_sentences
        texts, query_ids, sent_ids, partial_results = self._prepare_data(sentences)
        if not texts:
            return partial_results
        features = _FeatureBuilder(self.tokenizer, self.labels, self.char2phonemes, self.chars, texts, query_ids, use_mask=self.config['use_mask'], window_size=self.config['window_size'])
        preds: list[str] = []
        for start in range(0, len(features), self.batch_size):
            batch = [features[i] for i in range(start, min(start + self.batch_size, len(features)))]
            probs = self.session_g2pw.run(_collate(batch))[0]
            preds += [self.labels[pred] for pred in _argmax_rows(probs)]
        if self.config['use_char_phoneme']:
            preds = [pred.split(' ')[1] for pred in preds]
        results = partial_results
        for sent_id, query_id, pred in zip(sent_ids, query_ids, preds):
            results[sent_id][query_id] = self.style_convert_func(pred)
        return results

    def _prepare_data(self, sentences: list[str]):
        """Resolve what the lookup tables can, and collect the rest for the model."""
        polyphonic_chars = set(self.chars)
        monophonic_chars_dict = dict(self.monophonic_chars)
        texts: list[str] = []
        query_ids: list[int] = []
        sent_ids: list[int] = []
        partial_results: list[list[Optional[str]]] = []
        for sent_id, sentence in enumerate(sentences):
            partial_result: list[Optional[str]] = [None] * len(sentence)
            for i, char in enumerate(sentence):
                if char in polyphonic_chars:
                    texts.append(sentence)
                    query_ids.append(i)
                    sent_ids.append(sent_id)
                elif char in monophonic_chars_dict:
                    partial_result[i] = self.style_convert_func(monophonic_chars_dict[char])
                elif char in self.char_bopomofo_dict:
                    partial_result[i] = self.style_convert_func(self.char_bopomofo_dict[char][0])
            partial_results.append(partial_result)
        return (texts, query_ids, sent_ids, partial_results)
