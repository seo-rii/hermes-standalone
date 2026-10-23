"""Original dictionary bytes and independently constructed format boundaries."""
import hashlib
from pathlib import Path
import struct
import unittest

from standalone import dictionary_pack


def fixture_dictionary():
    def integer(value):
        data = bytearray()
        while value >= 128:
            data.append((value & 127) | 128)
            value >>= 7
        return bytes(data) + bytes([value])

    def string(value):
        data = value.encode()
        return integer(len(data)) + data

    def entry(counts, tail):
        header = bytearray(32)
        struct.pack_into('<6b', header, 26, *counts)
        return bytes(header) + tail + bytes((-len(tail)) % 8)

    first = entry((2, -1, -1, 0, 1, 1), struct.pack('<3Ih', 4, 7, 11, 2) + '한字'.encode('utf-16-le'))
    second = entry((1, 0, 1, 2, 0, 0), struct.pack('<4I', 1, 2, 3, 4))
    blocks = {
        'ConnMatrix': struct.pack('<hh6h', 2, 3, -32768, -1, 0, 1, 255, 32767),
        'TrieIndex': bytes(range(256)) * 16,
        'Strings': 'ヘルメス 한글 original data'.encode('utf-16-le'),
        'Entries': bytes(32) + first + second + b'original-trailing-bytes',
    }
    header = b'SudachiBinaryDic' + struct.pack('<QQQ', 1, 0, 0)
    header += string('fixture') + string('') + string('') + integer(2) + integer(2) + integer(len(blocks))
    data, position = bytearray(), 4096
    for name, block in blocks.items():
        header += string(name) + integer(position) + integer(len(block))
        data.extend(block)
        position += len(block)
    return header + bytes(4096 - len(header)) + data


class DictionaryPackingTests(unittest.TestCase):
    def test_all_integer_pairs_and_original_entry_tails_round_trip(self):
        values = b''.join(bytes((left, right)) for left in range(256) for right in range(256))
        original = bytes(4) + values
        packed = dictionary_pack._dict_matrix(original, False)
        self.assertEqual(dictionary_pack._dict_matrix(packed, True), original)
        data = fixture_dictionary()
        packed = dictionary_pack.transform_dictionary(data)
        self.assertEqual(len(packed), len(data))
        self.assertNotEqual(packed, data)
        self.assertEqual(dictionary_pack.transform_dictionary(packed, inverse=True), data)
        self.assertEqual(dictionary_pack.transform_dictionary(data), packed)

    def test_invalid_headers_and_truncated_blocks_refuse_before_restoration(self):
        for data in (b'', b'not a dictionary', fixture_dictionary()[:-40]):
            with self.subTest(size=len(data)):
                with self.assertRaises(ValueError):
                    dictionary_pack.transform_dictionary(data, inverse=True)
        data = fixture_dictionary().replace(b'Strings', b'Entries', 1)
        with self.assertRaisesRegex(ValueError, 'Invalid packed dictionary block'):
            dictionary_pack.transform_dictionary(data)

    def test_complete_original_sudachi_dictionary_matches_independent_packing_digest(self):
        path = Path(__file__).resolve().parents[1] / 'local_piper_japanese/data/sudachi/system.dic'
        original = path.read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         'e815659f6ad6f6118725103d47d317d660276ff3a15f668f8fc123ec3c4e6646')
        packed = dictionary_pack.transform_dictionary(original)
        self.assertEqual(len(packed), len(original))
        self.assertEqual(hashlib.sha256(packed).hexdigest(),
                         'd4ac1fecc71bd2ad73a073b19c9afa2643aec65bb1003e38ca401f86ef3aa80c')
        restored = dictionary_pack.transform_dictionary(packed, inverse=True)
        self.assertEqual(restored, original)
