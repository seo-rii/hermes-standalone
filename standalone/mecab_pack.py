"""Lossless field-column packing for the pinned UTF-8 MeCab system dictionary.

The original header, trie, tokens and every feature byte are restored before
the dictionary is exposed to the runtime. Only the feature fields are permuted.
"""
import struct as _mecab_struct


_mecab_MAX_BYTES = 256 * 1024 * 1024
_mecab_MAX_ROWS = 3000000
_mecab_COLUMNS = 11
_mecab_HEADER_BYTES = 72
_mecab_MAGIC = 0xef718f77


def _mecab_frame(data):
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError('Expected MeCab dictionary bytes')
    if not _mecab_HEADER_BYTES <= len(data) <= _mecab_MAX_BYTES:
        raise ValueError('MeCab dictionary size exceeds reviewed bounds')
    magic, version, kind, total, left, right, trie, tokens, features, _ = _mecab_struct.unpack_from('<10I', data)
    if magic != len(data) ^ _mecab_MAGIC or version != 102 or kind != 0:
        raise ValueError('Expected reviewed MeCab V102 system dictionary')
    if not 0 < total <= _mecab_MAX_ROWS or not 0 < left <= 65535 or not 0 < right <= 65535:
        raise ValueError('MeCab dictionary counts exceed reviewed bounds')
    if trie < 8 or trie % 8 or tokens != total * 16:
        raise ValueError('Invalid MeCab trie or token framing')
    charset = data[40:_mecab_HEADER_BYTES]
    if b'\0' not in charset or charset.partition(b'\0')[0].lower() != b'utf-8':
        raise ValueError('Expected a NUL-terminated UTF-8 MeCab charset')
    start = _mecab_HEADER_BYTES + trie + tokens
    if features < _mecab_COLUMNS or start + features != len(data) or data[-1] != 0:
        raise ValueError('Invalid MeCab feature framing or NUL termination')
    return start


def _mecab_pack(data, start):
    columns = [bytearray() for _ in range(_mecab_COLUMNS)]
    source = memoryview(data)
    position, rows = start, 0
    while position < len(data):
        end = data.find(b'\0', position)
        if end < 0:
            raise ValueError('Truncated MeCab feature record')
        for column in range(_mecab_COLUMNS):
            following = data.find(b',', position, end)
            if column == _mecab_COLUMNS - 1:
                if following >= 0:
                    raise ValueError('MeCab feature record has too many fields')
                following = end
            elif following < 0:
                raise ValueError('MeCab feature record has too few fields')
            columns[column].extend(source[position:following])
            columns[column].append(0)
            position = following + 1
        rows += 1
        if rows > _mecab_MAX_ROWS:
            raise ValueError('MeCab feature rows exceed reviewed bounds')
    return b''.join([source[:start], *columns])


def _mecab_column_offsets(data, start):
    cells = data.count(b'\0', start)
    rows, extra = divmod(cells, _mecab_COLUMNS)
    if extra or not 0 < rows <= _mecab_MAX_ROWS:
        raise ValueError('Invalid packed MeCab column count or row bound')
    if data.find(b',', start) >= 0:
        raise ValueError('Invalid comma in packed MeCab feature field')
    offsets, position = [], start
    for _ in range(_mecab_COLUMNS):
        offsets.append(position)
        for _ in range(rows):
            position = data.find(b'\0', position) + 1
    return offsets, rows


def _mecab_unpack(data, start):
    offsets, rows = _mecab_column_offsets(data, start)
    source = memoryview(data)
    output = bytearray(len(data))
    output[:start] = source[:start]
    position = start
    for _ in range(rows):
        for column in range(_mecab_COLUMNS):
            begin = offsets[column]
            end = data.find(b'\0', begin)
            if end < 0:
                raise ValueError('Truncated packed MeCab feature field')
            length = end - begin
            output[position:position + length] = source[begin:end]
            output[position + length] = 0 if column == _mecab_COLUMNS - 1 else 44
            position += length + 1
            offsets[column] = end + 1
    if position != len(data):
        raise ValueError('Packed MeCab feature length changed')
    return bytes(output)


def transform_mecab(data, *, inverse=False):
    """Pack or restore every original byte of a bounded MeCab V102 dictionary."""
    start = _mecab_frame(data)
    output = _mecab_unpack(data, start) if inverse else _mecab_pack(data, start)
    if len(output) != len(data):
        raise ValueError('Packed MeCab dictionary size changed')
    return output
