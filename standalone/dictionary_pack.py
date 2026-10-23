"""Lossless Sudachi V1 block packing; restores every original dictionary byte.

The format is pinned by piper-japanese-resources-provenance.json. This module is
a build helper; its stdlib-only functions are inserted into the bootstrap so
the original resource is restored before a cache directory is published.
"""
from array import array
from itertools import accumulate
import struct
import sys


def _dict_integer(data, position):
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError('Truncated packed dictionary integer')
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            if value >= 1 << 64:
                raise ValueError('Packed dictionary integer exceeds64bits')
            return value, position
    raise ValueError('Invalid packed dictionary integer')


def _dict_string(data, position):
    size, position = _dict_integer(data, position)
    following = position + size
    if size > 4096 or following > len(data):
        raise ValueError('Invalid packed dictionary header string')
    return data[position:following].decode('utf8'), following


def _dict_blocks(data):
    if len(data) < 40 or data[:16] != b'SudachiBinaryDic' or struct.unpack_from('<Q', data, 16)[0] != 1:
        raise ValueError('Expected reviewed Sudachi V1 dictionary data')
    position = 40
    for _ in range(3):
        _, position = _dict_string(data, position)
    _, position = _dict_integer(data, position)
    total, position = _dict_integer(data, position)
    count, position = _dict_integer(data, position)
    if total > 3000000 or count > 32:
        raise ValueError('Packed dictionary header exceeds reviewed bounds')
    blocks = {}
    for _ in range(count):
        name, position = _dict_string(data, position)
        start, position = _dict_integer(data, position)
        size, position = _dict_integer(data, position)
        if start + size > len(data) or name in blocks:
            raise ValueError('Invalid packed dictionary block')
        blocks[name] = start, size
    return total, blocks


def _dict_planes(data, width, inverse=False):
    if len(data) % width:
        raise ValueError('Unaligned packed dictionary plane')
    if not inverse:
        return b''.join(data[offset::width] for offset in range(width))
    output, size = bytearray(len(data)), len(data) // width
    for offset in range(width):
        output[offset::width] = data[offset * size:(offset + 1) * size]
    return bytes(output)


def _dict_matrix(data, inverse):
    if len(data) < 4 or len(data) % 2:
        raise ValueError('Invalid packed dictionary matrix')
    words = array('H')
    words.frombytes(_dict_planes(data[4:], 2, True) if inverse else data[4:])
    if sys.byteorder != 'little':
        words.byteswap()
    if inverse:
        values = (value & 65535 for value in accumulate(words))
    else:
        values = ((value - (words[index - 1] if index else 0)) & 65535 for index, value in enumerate(words))
    result = array('H', values)
    if sys.byteorder != 'little':
        result.byteswap()
    raw = result.tobytes()
    return data[:4] + (raw if inverse else _dict_planes(raw, 2))


def _dict_tail(data, position, header):
    counts = struct.unpack_from('<6b', header, 26)
    length = 4 * sum(max(0, count) for count in counts[:5])
    if counts[5]:
        if counts[5] != 1 or position + length + 2 > len(data):
            raise ValueError('Invalid packed dictionary user-data count')
        units, = struct.unpack_from('<h', data, position + length)
        if units < 0:
            raise ValueError('Invalid packed dictionary user-data length')
        length += 2 + 2 * units
    length = ((32 + length + 7) & ~7) - 32
    if position + length > len(data):
        raise ValueError('Truncated packed dictionary entry tail')
    return length


def _dict_entries(data, total, inverse):
    limit = 32 + total * 32
    if limit > len(data):
        raise ValueError('Invalid packed dictionary entry count')
    if inverse:
        headers = _dict_planes(data[32:limit], 32, True)
        output, position = bytearray(data[:32]), limit
        for index in range(total):
            header = headers[index * 32:(index + 1) * 32]
            length = _dict_tail(data, position, header)
            output.extend(header)
            output.extend(data[position:position + length])
            position += length
        output.extend(data[position:])
        return bytes(output)
    headers, tails, position = bytearray(total * 32), bytearray(), 32
    for index in range(total):
        if position + 32 > len(data):
            raise ValueError('Truncated dictionary entry header')
        header = data[position:position + 32]
        headers[index * 32:(index + 1) * 32] = header
        length = _dict_tail(data, position + 32, header)
        tails.extend(data[position + 32:position + 32 + length])
        position += 32 + length
    return data[:32] + _dict_planes(headers, 32) + tails + data[position:]


def transform_dictionary(data, *, inverse=False):
    """Permute/delta-encode four known blocks without changing file length."""
    total, blocks = _dict_blocks(data)
    output = bytearray(data)
    required = ('ConnMatrix', 'TrieIndex', 'Strings', 'Entries')
    if any(name not in blocks for name in required):
        raise ValueError('Missing packed dictionary block')
    intervals = sorted(blocks[name] for name in required)
    if any(start + size > following for (start, size), (following, _) in zip(intervals, intervals[1:])):
        raise ValueError('Overlapping packed dictionary blocks')
    for name in required:
        start, size = blocks[name]
        block = data[start:start + size]
        if name == 'ConnMatrix':
            block = _dict_matrix(block, inverse)
        elif name == 'Entries':
            block = _dict_entries(block, total, inverse)
        else:
            block = _dict_planes(block, 2 if name == 'Strings' else 4, inverse)
        if len(block) != size:
            raise ValueError('Packed dictionary block size changed')
        output[start:start + size] = block
    return bytes(output)
