"""Independent Opus 1.3.1 C oracle bytes and coder states; no native runtime."""
import base64
import hashlib
import json
from pathlib import Path
import unittest
import zlib

from standalone.opus_range import OpusRangeDecoder, OpusRangeEncoder, OpusRangeError


def encode(context, operation):
    kind = operation['kind']
    if kind == 'range':
        context.encode(operation['low'], operation['high'], operation['total'])
    elif kind == 'bin':
        context.encode_bin(operation['low'], operation['high'], operation['precision'])
    elif kind == 'bit':
        context.bit_logp(operation['value'], operation['logp'])
    elif kind == 'icdf':
        context.icdf(operation['value'], operation['table'], operation.get('ftb', 8))
    elif kind == 'uint':
        context.uint(operation['value'], operation['total'])
    elif kind == 'bits':
        context.bits(operation['value'], operation['count'])
    elif kind == 'patch':
        context.patch_initial_bits(operation['value'], operation['count'])
    elif kind == 'shrink':
        context.shrink(operation['size'])
    elif kind == 'laplace':
        return context.laplace(operation['value'], operation['fs'], operation['decay'])
    else:
        raise AssertionError(operation)
    return 0


def decode(context, operation):
    kind = operation['kind']
    if kind == 'range':
        value = context.decode(operation['total'])
        context.update(operation['low'], operation['high'], operation['total'])
    elif kind == 'bin':
        value = context.decode_bin(operation['precision'])
        context.update(value, value + 1, 1 << operation['precision'])
    elif kind == 'bit':
        value = context.bit_logp(operation['logp'])
    elif kind == 'icdf':
        value = context.icdf(operation['table'], operation.get('ftb', 8))
    elif kind == 'uint':
        value = context.uint(operation['total'])
    elif kind == 'bits':
        value = context.bits(operation['count'])
    elif kind == 'laplace':
        value = context.laplace(operation['fs'], operation['decay'])
    else:
        raise AssertionError(operation)
    return value


class OpusRangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = json.loads((Path(__file__).parent / 'fixtures/opus_range.json').read_text())
        payload = zlib.decompress(base64.b64decode(fixture['data_zlib_base64'], validate=True))
        if hashlib.sha256(payload).hexdigest() != fixture['data_sha256']:
            raise AssertionError('Corrupted native Opus range oracle fixture')
        cls.data = json.loads(payload)

    def assert_context(self, context, expected):
        actual = {name: getattr(context, name)() if name in ('tell', 'tell_frac') else getattr(context, name)
                  for name in expected}
        self.assertEqual(actual, expected)

    def test_native_encoder_bytes_and_every_internal_state(self):
        for vector in self.data['vectors']:
            with self.subTest(vector=vector['id']):
                encoder = OpusRangeEncoder(vector['storage'])
                for operation, state, expected_result in zip(vector['operations'], vector['encoder_trace'], vector['encoder_results']):
                    self.assertEqual(encode(encoder, operation), expected_result)
                    self.assert_context(encoder, state)
                if vector['expect_error']:
                    with self.assertRaises(OpusRangeError):
                        encoder.done()
                    self.assertEqual(bytes(encoder.buf[:encoder.storage]), base64.b64decode(vector['encoded']))
                else:
                    self.assertEqual(encoder.done(), base64.b64decode(vector['encoded']))
                self.assert_context(encoder, vector['encoder_final'])

    def test_native_decoder_symbols_and_every_internal_state(self):
        for vector in self.data['vectors'] + self.data['decoder_only']:
            if vector.get('expect_error'):
                continue
            with self.subTest(vector=vector['id']):
                decoder = OpusRangeDecoder(base64.b64decode(vector['encoded']))
                operations = [op for op in vector.get('decode_operations', vector['operations'])
                              if op['kind'] not in ('patch', 'shrink')]
                self.assertEqual(len(operations), len(vector['decoder_trace']))
                for operation, symbol, state in zip(operations, vector['symbols'], vector['decoder_trace']):
                    self.assertEqual(decode(decoder, operation), symbol)
                    self.assert_context(decoder, state)

    def test_finished_encoder_rejects_mutation(self):
        for operation in ({'kind':'range','low':0,'high':1,'total':2},
                          {'kind':'bin','low':0,'high':1,'precision':1},
                          {'kind':'bit','value':1,'logp':1},
                          {'kind':'icdf','value':0,'table':[0]},
                          {'kind':'uint','value':1,'total':2},
                          {'kind':'bits','value':1,'count':1},
                          {'kind':'patch','value':1,'count':1},
                          {'kind':'shrink','size':1},
                          {'kind':'laplace','value':1,'fs':6000,'decay':8000}):
            with self.subTest(operation=operation):
                encoder = OpusRangeEncoder(4)
                original = encoder.done()
                before = vars(encoder).copy()
                before['buf'] = bytes(encoder.buf)
                with self.assertRaises(OpusRangeError):
                    encode(encoder, operation)
                after = vars(encoder).copy()
                after['buf'] = bytes(encoder.buf)
                self.assertEqual(after, before)
                self.assertEqual(bytes(encoder.buf), original)

    def test_zero_probability_symbol_is_rejected_without_hanging(self):
        encoder = OpusRangeEncoder(8)
        with self.assertRaises(OpusRangeError):
            encoder.icdf(1, [128, 128, 0])
        with self.assertRaises(OpusRangeError):
            encoder.icdf(0, [256, 0])
        self.assertEqual(encoder.tell(), 1)


if __name__ == '__main__':
    unittest.main()
