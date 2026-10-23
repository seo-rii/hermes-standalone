"""Independent MeCab field-column fixtures and malformed-format boundaries."""
import hashlib
from pathlib import Path
import struct
import unittest
from unittest import mock

from standalone import mecab_pack


def fixture_mecab(rows=None, *, payload=None):
    if rows is None:
        rows = [(b'word', b'noun', b'', b'*', b'word', b'0', b'written', b'reading', b'phonemes', b'1/2', b'*'),
                (b'longer word', b'verb', b'', b'', b'base', b'100', b'\xe6\xbc\xa2\xe5\xad\x97',
                 b'\xff\xfe', b'p', b'0/1', b'')]
    trie = struct.pack('<4i', 1, 0, -123456, 7654321)
    tokens = bytes(range(16)) * len(rows)
    features = b''.join(b','.join(row) + b'\0' for row in rows) if payload is None else payload
    size = 72 + len(trie) + len(tokens) + len(features)
    header = struct.pack('<10I', size ^ 0xef718f77, 102, 0, len(rows), 1, 2,
                         len(trie), len(tokens), len(features), 0xa1b2c3d4)
    charset = b'utf-8\0' + bytes(range(26))
    return header + charset + trie + tokens + features


def replace_header(data, index, value):
    output = bytearray(data)
    struct.pack_into('<I', output, index * 4, value)
    return bytes(output)


class MeCabPackingTests(unittest.TestCase):
    def test_independent_column_layout_and_original_prefix_round_trip(self):
        rows = [tuple(f'row{row}-column{column}'.encode() for column in range(11)) for row in range(3)]
        original = fixture_mecab(rows)
        prefix = original[:72 + 16 + 16 * len(rows)]
        expected = prefix + b''.join(rows[row][column] + b'\0'
                                      for column in range(11) for row in range(len(rows)))
        packed = mecab_pack.transform_mecab(original)
        self.assertEqual(packed, expected)
        self.assertEqual(len(packed), len(original))
        self.assertNotEqual(packed, original)
        self.assertEqual(mecab_pack.transform_mecab(expected, inverse=True), original)

    def test_empty_fields_nontext_bytes_and_variable_lengths_are_preserved(self):
        cases = [[tuple(b'' for _ in range(11))],
                 [tuple(bytes([128 + column]) * (column * 13) for column in range(11))],
                 [tuple((b'' if (row + column) % 3 else b'x' * (row + column))
                        for column in range(11)) for row in range(31)]]
        for rows in cases:
            with self.subTest(rows=len(rows)):
                data = fixture_mecab(rows)
                packed = mecab_pack.transform_mecab(data)
                self.assertEqual(mecab_pack.transform_mecab(packed, inverse=True), data)
                self.assertEqual(mecab_pack.transform_mecab(data), packed)
        data = fixture_mecab()
        self.assertEqual(mecab_pack.transform_mecab(
            mecab_pack.transform_mecab(data), inverse=True), data)

    def test_bytearray_input_is_not_mutated_and_result_is_bytes(self):
        data = bytearray(fixture_mecab())
        original = bytes(data)
        packed = mecab_pack.transform_mecab(data)
        self.assertIsInstance(packed, bytes)
        self.assertEqual(bytes(data), original)
        self.assertEqual(mecab_pack.transform_mecab(bytearray(packed), inverse=True), original)

    def test_invalid_header_version_kind_counts_charset_and_block_sizes_refuse(self):
        original = fixture_mecab()
        cases = [b'', b'x' * 71, original[:-1], original + b'x',
                 replace_header(original, 0, 0), replace_header(original, 1, 101),
                 replace_header(original, 2, 1), replace_header(original, 3, 0),
                 replace_header(original, 3, 3000001), replace_header(original, 4, 0),
                 replace_header(original, 5, 65536), replace_header(original, 6, 0),
                 replace_header(original, 6, 15), replace_header(original, 7, 31),
                 replace_header(original, 8, 10), replace_header(original, 8, 0xffffffff),
                 original[:40] + b'ascii\0' + original[46:],
                 original[:40] + b'x' * 32 + original[72:]]
        for number, data in enumerate(cases):
            with self.subTest(case=number):
                for inverse in (False, True):
                    with self.assertRaises(ValueError):
                        mecab_pack.transform_mecab(data, inverse=inverse)

    def test_invalid_feature_record_counts_and_unterminated_records_refuse(self):
        payloads = [b'a,' * 9 + b'a\0', b'a,' * 11 + b'a\0',
                    b'a,' * 10 + b'a', b'a,' * 10 + b'a\0\0',
                    b'a,' * 5 + b'\0' + b'a,' * 5 + b'a\0',
                    b'a,' * 10 + b'a\0' + b'a,' * 9 + b'a\0']
        for payload in payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    mecab_pack.transform_mecab(fixture_mecab(payload=payload))

    def test_invalid_packed_column_count_comma_and_terminal_nul_refuse(self):
        payloads = [bytes(12), bytes(21), b',' + bytes(22),
                    bytes(10) + b'last', b'x' * 11]
        for payload in payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    mecab_pack.transform_mecab(fixture_mecab(payload=payload), inverse=True)

    def test_size_and_feature_row_caps_apply_in_both_directions(self):
        data = fixture_mecab()
        packed = mecab_pack.transform_mecab(data)
        with mock.patch.object(mecab_pack, '_mecab_MAX_BYTES', len(data) - 1):
            for inverse, value in ((False, data), (True, packed)):
                with self.subTest(bound='bytes', inverse=inverse):
                    with self.assertRaises(ValueError):
                        mecab_pack.transform_mecab(value, inverse=inverse)
        more_features = fixture_mecab([tuple(b'cell' for _ in range(11))] * 3)
        # Keep the token count within the cap so the feature rows enforce their own bound.
        start = 72 + 16 + 3 * 16
        many_rows = fixture_mecab(payload=more_features[start:])
        many_packed = mecab_pack.transform_mecab(many_rows)
        with mock.patch.object(mecab_pack, '_mecab_MAX_ROWS', 2):
            for inverse, value in ((False, many_rows), (True, many_packed)):
                with self.subTest(bound='rows', inverse=inverse):
                    with self.assertRaises(ValueError):
                        mecab_pack.transform_mecab(value, inverse=inverse)

    def test_unsupported_input_types_refuse_without_conversion(self):
        for value in (None, 'text', 3, memoryview(b'data')):
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises(ValueError):
                    mecab_pack.transform_mecab(value)

    def test_complete_dictionary_matches_independently_measured_pack_digest(self):
        path = Path(__file__).resolve().parents[1] / 'local_piper_japanese/data/openjtalk/dictionary/sys.dic'
        original = path.read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         '23287ce5d518164b8b97683b1d9534df095556e446f3470a6c52bd737be74527')
        packed = mecab_pack.transform_mecab(original)
        self.assertEqual(len(packed), len(original))
        self.assertEqual(hashlib.sha256(packed).hexdigest(),
                         '6f75989d8f7980ac4f48e9e021d1e7af91b464cb60d3a508f345560da9346fc2')
        restored = mecab_pack.transform_mecab(packed, inverse=True)
        self.assertEqual(restored, original)
