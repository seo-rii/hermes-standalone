"""Pickle memo operand vectors are parsed as data, never executed."""
import unittest
from unittest import mock

from standalone import pickle_pack


class PicklePackingTests(unittest.TestCase):
    def test_nonunit_memo_indexes_wrap_and_preserve_unrelated_operands(self):
        prefix = b'\x80\x03]q\xff'
        original_memos = (b'r\x00\x01\x00\x00r\x01\x04\x00\x00r\x07\x00\x00\x00'
                          b'r\xff\xff\xff\xffr\x00\x02\x00\x00')
        expected_deltas = (b'r\x00\x01\x00\x00r\x01\x03\x00\x00r\x06\xfc\xff\xff'
                           b'r\xf8\xff\xff\xffr\x01\x02\x00\x00')
        # The apparent memo opcode inside BINBYTES and the LONG_BINGET stay literal.
        suffix = b'C\x05r\x00\x01\x00\x00aj\x01\x04\x00\x000.'
        original = prefix + original_memos + suffix
        expected = prefix + expected_deltas + suffix
        packed = pickle_pack.transform_pickle_memo(original)
        self.assertEqual(packed, expected)
        self.assertEqual(len(packed), len(original))
        self.assertEqual(pickle_pack.transform_pickle_memo(expected, inverse=True), original)
        self.assertEqual(pickle_pack.transform_pickle_memo(original), packed)

    def test_global_and_reduce_are_never_executed(self):
        payload = b'\x80\x03cos\nsystem\nr\x01\x04\x00\x00X\x05\x00\x00\x00never\x85R.'
        with mock.patch('os.system', side_effect=AssertionError('Pickle execution is forbidden')) as command:
            packed = pickle_pack.transform_pickle_memo(payload)
            self.assertEqual(pickle_pack.transform_pickle_memo(packed, inverse=True), payload)
            command.assert_not_called()

    def test_protocol_three_stream_without_long_memos_stays_identical(self):
        payload = b'\x80\x03]q\x00(K\x01K\x02e.'
        self.assertEqual(pickle_pack.transform_pickle_memo(payload), payload)
        self.assertEqual(pickle_pack.transform_pickle_memo(payload, inverse=True), payload)

    def test_truncation_unknown_opcodes_and_unexpected_framing_refuse(self):
        invalid = (
            b'', b'\x80', b'\x80\x03', b'\x80\x02N.', b'\x80\x04N.',
            b'\x80\x03\x80\x03N.', b'\x80\x03N.N.', b'\x80\x03N.\x80\x03N.',
            b'\x80\x03\x95' + bytes(8) + b'N.', b'\x80\x03r\x01\x00.',
            b'\x80\x03r\x01\x00\x00\x00', b'\x80\x03X\x05\x00\x00\x00hi.',
            b'\x80\x03cos\nsystem.', b'\x80\x03\xff.',
            b'\x80\x03B\xff\xff\xff\xff.',
            b'\x80\x03\x8e\xff\xff\xff\xff\xff\xff\xff\xff.',
        )
        for payload in invalid:
            for inverse in (False, True):
                with self.subTest(payload=payload, inverse=inverse):
                    with self.assertRaises(ValueError):
                        pickle_pack.transform_pickle_memo(payload, inverse=inverse)

    def test_resource_size_limit_refuses_before_opcode_parsing(self):
        with self.assertRaisesRegex(ValueError, 'size'):
            pickle_pack.transform_pickle_memo(bytes((32 << 20) + 1))
