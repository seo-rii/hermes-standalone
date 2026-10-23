"""Independent byte vectors for the reviewed model's reversible transpose."""
import unittest

from standalone import model_pack


class ModelPackingTests(unittest.TestCase):
    def test_rectangular_embedding_bytes_match_independent_column_vector(self):
        original = b'\x00abc\xffDEF\x80ghi'
        expected = b'\x00\xff\x80aDgbEhcFi'
        packed = model_pack._model_transpose(original, 3, 4)
        self.assertEqual(packed, expected)
        self.assertEqual(model_pack._model_transpose(expected, 3, 4, inverse=True), original)
        self.assertEqual(model_pack._model_transpose(original, 3, 4), packed)

    def test_single_row_and_column_preserve_all_byte_values(self):
        original = bytes(range(256))
        for rows, columns in ((1, 256), (256, 1)):
            for inverse in (False, True):
                with self.subTest(rows=rows, inverse=inverse):
                    self.assertEqual(model_pack._model_transpose(original, rows, columns, inverse=inverse), original)

    def test_invalid_dimensions_and_resource_bounds_refuse(self):
        for data, rows, columns in (
            (b'', 0, 4), (b'', -1, 4), (b'', 1, 0),
            (b'abcdefgh', 3, 4), (b'', 1, 4097),
            (b'', (32 << 20) + 1, 1),
        ):
            for inverse in (False, True):
                with self.subTest(rows=rows, columns=columns, inverse=inverse):
                    with self.assertRaises(ValueError):
                        model_pack._model_transpose(data, rows, columns, inverse=inverse)

    def test_unreviewed_model_lengths_refuse_before_any_permutation(self):
        for data in (b'', b'not a reviewed ONNX model', bytes(128)):
            for inverse in (False, True):
                with self.subTest(size=len(data), inverse=inverse):
                    with self.assertRaises(ValueError):
                        model_pack.transform_g2pw_embedding(data, inverse=inverse)
