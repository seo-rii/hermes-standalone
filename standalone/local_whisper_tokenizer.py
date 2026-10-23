"""Deterministic Whisper tokenizer.json ByteLevel BPE, without a Rust module.

This qualified model-data consumer follows tokenizers 0.23.1's ByteLevel
pre-tokenizer, stable BPE merge heap and byte decoder (Apache-2.0). It accepts
the tokenizer graph carried by the original Whisper models, without claiming
to implement unrelated tokenizer graph types, normalizers or random dropout.
"""

import heapq
import itertools
import json
from pathlib import Path
import re

from .local_whisper_unicode import category, word


def _byte_alphabet():
    # The extra Unicode characters are assigned in increasing byte order.
    next_char = 256
    result = {value: chr(value) for value in list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))}
    for value in range(256):
        if value not in result:
            result[value] = chr(next_char)
            next_char += 1
    return result


_BYTE_CHARS = _byte_alphabet()
_CHAR_BYTES = {value: key for key, value in _BYTE_CHARS.items()}
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _pieces(text):
    """The pinned GPT-2 regex, including its whitespace backtracking rule."""
    index = 0
    while index < len(text):
        contraction = next((value for value in _CONTRACTIONS if text.startswith(value, index)), None)
        if contraction is not None:
            yield contraction
            index += len(contraction)
            continue
        start = index
        offset = index + int(text[index] == " " and index + 1 < len(text))
        kind = category(text[offset])
        if kind != "S":
            index = offset + 1
            while index < len(text) and category(text[index]) == kind:
                index += 1
        else:
            index += 1
            while index < len(text) and category(text[index]) == "S":
                index += 1
            if index < len(text) and index - start > 1:
                index -= 1
        yield text[start:index]


class ByteBPETokenizer:
    def __init__(self, data):
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.stat().st_size > 16 * 1024**2:
                raise ValueError("Whisper tokenizer exceeds its JSON byte bound")
            data = path.read_bytes()
        if not isinstance(data, bytes) or len(data) > 16 * 1024**2:
            raise ValueError("Whisper tokenizer must be bounded tokenizer.json bytes or a path")
        try:
            graph = json.loads(data)
        except (ValueError, RecursionError) as exc:
            raise ValueError("Invalid Whisper tokenizer JSON") from exc
        if not isinstance(graph, dict):
            raise ValueError("Invalid Whisper tokenizer graph")
        model, pre, decoder = (graph.get(key) for key in ("model", "pre_tokenizer", "decoder"))
        if (not isinstance(model, dict) or model.get("type") != "BPE" or model.get("dropout") is not None
                or model.get("unk_token") is not None or model.get("continuing_subword_prefix") not in (None, "")
                or model.get("end_of_word_suffix") not in (None, "") or model.get("fuse_unk", False)
                or model.get("byte_fallback", False) or model.get("ignore_merges", False)
                or graph.get("normalizer") is not None or graph.get("truncation") is not None
                or graph.get("padding") is not None
                or not isinstance(pre, dict) or pre.get("type") != "ByteLevel"
                or not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel"):
            raise ValueError("Expected the deterministic Whisper ByteLevel BPE tokenizer graph")
        self.add_prefix_space = bool(pre.get("add_prefix_space", True))
        self.use_regex = bool(pre.get("use_regex", True))
        vocab, merges = model.get("vocab"), model.get("merges")
        if not isinstance(vocab, dict) or not 256 <= len(vocab) <= 262144 or not isinstance(merges, list) or len(merges) > 262144:
            raise ValueError("Invalid Whisper tokenizer vocabulary or merge bound")
        self._vocab, self._tokens = {}, {}
        for token, index in vocab.items():
            self._register(token, index)
        if any(char not in self._vocab for char in _BYTE_CHARS.values()):
            raise ValueError("Whisper tokenizer lacks a complete byte vocabulary")
        self._merges = {}
        for rank, merge in enumerate(merges):
            if isinstance(merge, str):
                merge = merge.split(" ")
            if (not isinstance(merge, (tuple, list)) or len(merge) != 2
                    or any(not isinstance(value, str) or value not in self._vocab for value in merge)
                    or "".join(merge) not in self._vocab):
                raise ValueError("Invalid Whisper BPE merge")
            pair = tuple(self._vocab[value] for value in merge)
            if pair in self._merges:
                raise ValueError("Duplicate Whisper BPE merge pair")
            self._merges[pair] = rank, self._vocab["".join(merge)]
        added = graph.get("added_tokens", [])
        if not isinstance(added, list) or len(added) > 8192:
            raise ValueError("Whisper tokenizer exceeds its added token bound")
        self._added, self._special = {}, set()
        for item in added:
            if not isinstance(item, dict) or any(not isinstance(item.get(key, False), bool)
                                                  for key in ("special", "single_word", "lstrip", "rstrip", "normalized")):
                raise ValueError("Invalid Whisper added token")
            token, index = item.get("content"), item.get("id")
            self._register(token, index, added=True)
            if token in self._added:
                raise ValueError("Duplicate Whisper added token")
            self._added[token] = item
            if item.get("special", False):
                self._special.add(index)
        self._added_re = re.compile("|".join(re.escape(value) for value in sorted(self._added, key=len, reverse=True))) if added else None
        self._template = graph.get("post_processor")
        self._cache = {}

    def _register(self, token, index, added=False):
        if (not isinstance(token, str) or added and not token or len(token) > 4096 or isinstance(index, bool)
                or not isinstance(index, int) or not 0 <= index < 2**20
                or index in self._tokens and self._tokens[index] != token
                or token in self._vocab and (not added or self._vocab[token] != index)):
            raise ValueError("Invalid or duplicate Whisper vocabulary entry")
        self._vocab[token], self._tokens[index] = index, token

    def token_to_id(self, token):
        return self._vocab.get(token)

    def _bpe(self, text):
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        symbols = [self._vocab[_BYTE_CHARS[value]] for value in text.encode("utf-8")]
        previous = list(range(-1, len(symbols) - 1))
        following = list(range(1, len(symbols))) + [-1]
        active = [True] * len(symbols)
        heap = []

        def offer(left):
            if left < 0 or not active[left]:
                return
            right = following[left]
            if right < 0:
                return
            pair = symbols[left], symbols[right]
            merge = self._merges.get(pair)
            if merge is not None:
                rank, replacement = merge
                heapq.heappush(heap, (rank, left, right, *pair, replacement))

        for index in range(max(0, len(symbols) - 1)):
            offer(index)
        while heap:
            _, left, right, first, second, replacement = heapq.heappop(heap)
            if not active[left] or not active[right] or following[left] != right or (symbols[left], symbols[right]) != (first, second):
                continue
            symbols[left], active[right] = replacement, False
            following[left] = following[right]
            if following[right] >= 0:
                previous[following[right]] = left
            offer(previous[left])
            offer(left)
        result = tuple(symbol for symbol, keep in zip(symbols, active, strict=True) if keep)
        if len(text) <= 4096:
            if len(self._cache) >= 4096:
                self._cache.clear()
            self._cache[text] = result
        return result

    def _normal(self, text):
        if not text:
            return ()
        if self.add_prefix_space and not text.startswith(" "):
            text = " " + text
        return tuple(token for piece in (_pieces(text) if self.use_regex else (text,)) for token in self._bpe(piece))

    def encode(self, text, *, add_special_tokens=False):
        if not isinstance(text, str) or len(text) > 2**20:
            raise ValueError("Whisper tokenizer text exceeds its one-million-character bound")
        result, offset = [], 0
        if self._added_re is not None:
            for match in self._added_re.finditer(text):
                content, start, end = match.group(), match.start(), match.end()
                if start < offset:
                    continue
                item = self._added[content]
                if item.get("single_word", False) and ((start > 0 and word(text[start - 1])) or (end < len(text) and word(text[end]))):
                    continue
                if item.get("lstrip", False):
                    while start > offset and category(text[start - 1]) == "S":
                        start -= 1
                if item.get("rstrip", False):
                    while end < len(text) and category(text[end]) == "S":
                        end += 1
                result.extend(self._normal(text[offset:start]))
                result.append(item["id"])
                offset = end
        result.extend(self._normal(text[offset:]))
        if add_special_tokens and self._template is not None:
            result = self._apply_template(result)
        return result

    def _apply_template(self, tokens):
        template = self._template
        if not isinstance(template, dict) or template.get("type") != "TemplateProcessing":
            raise ValueError("Expected a Whisper single-sequence token template")
        result, seen = [], False
        for piece in template.get("single", ()):
            if not isinstance(piece, dict) or len(piece) != 1:
                raise ValueError("Invalid Whisper single-sequence token template")
            if "Sequence" in piece and piece["Sequence"].get("id") == "A" and not seen:
                result.extend(tokens)
                seen = True
            elif "SpecialToken" in piece:
                special = template.get("special_tokens", {}).get(piece["SpecialToken"].get("id"), {})
                ids = special.get("ids", ())
                if not ids or any(index not in self._tokens for index in ids):
                    raise ValueError("Invalid Whisper template special token")
                result.extend(ids)
            else:
                raise ValueError("Unsupported Whisper single-sequence template element")
        if not seen:
            raise ValueError("Whisper token template has no input sequence")
        return result

    def decode(self, tokens, *, skip_special_tokens=True):
        values = list(itertools.islice(tokens, 2**20 + 1))
        if len(values) > 2**20 or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32 for value in values):
            raise ValueError("Invalid or excessive Whisper token IDs")
        raw = bytearray()
        for index in values:
            if skip_special_tokens and index in self._special:
                continue
            for char in self._tokens.get(index, ""):
                value = _CHAR_BYTES.get(char)
                raw.extend(char.encode("utf-8") if value is None else bytes((value,)))
        return raw.decode("utf-8", errors="replace")
