"""Native Opus 1.3.1 CELT fixtures; all runtime tests use the stdlib alone."""
import base64
import hashlib
import json
from pathlib import Path
import struct
import unittest
import zlib

from standalone.opus_celt import CeltDecoder
from standalone.opus_celt_alloc import allocation
from standalone.opus_celt_energy import coarse
from standalone.opus_celt_bands import decode_bands
from standalone.opus_celt_pvq import encode_pulses, unquant
from standalone.opus_celt_plc import autocorr, lpc_coefficients
from standalone.opus_range import OpusRangeDecoder, OpusRangeEncoder, OpusRangeError


def fixture(name):
    wrapper = json.loads((Path(__file__).parent / 'fixtures' / name).read_text())
    raw = zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'], validate=True))
    if hashlib.sha256(raw).hexdigest() != wrapper['data_sha256']:
        raise AssertionError('Corrupted native Opus CELT fixture: ' + name)
    return json.loads(raw)


def floats(value):
    raw = base64.b64decode(value)
    return struct.unpack('<' + 'f' * (len(raw) // 4), raw)


class OpusCeltTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.received = fixture('opus_celt_pcm.json')
        cls.loss = fixture('opus_celt_loss.json')
        cls.components = fixture('opus_celt_components.json')
        cls.extra = fixture('opus_celt_extra.json')

    def assert_context(self, context, expected):
        actual = {name:getattr(context,name)() if name in ('tell','tell_frac') else getattr(context,name)
                  for name in expected}
        self.assertEqual(actual, expected)

    def assert_float_values(self, actual, expected, tolerance):
        self.assertEqual(len(actual), len(expected))
        self.assertLessEqual(max((abs(a-b) for a,b in zip(actual,expected)), default=0.), tolerance)

    def test_native_received_celt_packets_and_int16_output(self):
        for index, stream in enumerate(self.received):
            with self.subTest(stream=index):
                decoder = CeltDecoder(stream['channels'])
                integer_decoder = CeltDecoder(stream['channels']) if index % 8 == 0 else None
                for frame in stream['frames']:
                    packet = base64.b64decode(frame['packet'])
                    options = {'stream_channels':frame['stream_channels'], 'end':frame['end']}
                    expected = floats(frame['pcm_f32'])
                    actual = decoder.decode_float(packet[1:], stream['duration'], **options)
                    self.assert_float_values(actual,expected,2/32768)
                    self.assertEqual(decoder.rng,frame['rng'])
                    if integer_decoder is not None:
                        pcm = integer_decoder.decode(packet[1:],stream['duration'],**options)
                        integers = [max(-32768,min(32767,round(value*32768))) for value in expected]
                        self.assertLessEqual(max(abs(a-b) for a,b in zip(pcm,integers)),1)

    def test_native_loss_pitch_noise_recovery_rates_and_channel_changes(self):
        for index, stream in enumerate(self.loss):
            with self.subTest(stream=index):
                decoder = CeltDecoder(stream['channels'],stream['sample_rate'])
                for frame in stream['frames']:
                    packet = base64.b64decode(frame['packet']) if frame['packet'] is not None else None
                    if packet is None:
                        # Loss uses the previously received band's/channel format.
                        actual = decoder.decode_float(None,stream['duration'])
                    else:
                        actual = decoder.decode_float(packet[1:],stream['duration'],
                                                      stream_channels=frame['stream_channels'],end=frame['end'])
                    self.assert_float_values(actual,floats(frame['pcm_f32']),8/32768)
                    self.assertEqual({name:getattr(decoder,name) for name in frame['state']},frame['state'])

    def test_native_coarse_energy_allocation_recursive_bands_and_state(self):
        for index, vector in enumerate(self.components['bands']):
            with self.subTest(vector=index):
                decoder = OpusRangeDecoder(base64.b64decode(vector['data']))
                old = vector['initial_energy'][:]
                coarse(vector['start'],vector['end'],old,vector['intra'],decoder,vector['channels'],vector['lm'])
                self.assert_float_values(old,vector['coarse_energy'],2e-5)
                self.assert_context(decoder,vector['coarse_context'])
                result = allocation(vector['start'],vector['end'],vector['offsets'],vector['cap'],
                                    vector['trim'],vector['total'],vector['channels'],vector['lm'],decoder)
                self.assertEqual(list(result),vector['allocation'])
                self.assert_context(decoder,vector['allocation_context'])
                coded,intensity,dual,balance,pulses,_,_ = result
                values,masks,seed = decode_bands(vector['start'],vector['end'],vector['channels'],pulses,
                                                 vector['short_blocks'],vector['spread'],dual,intensity,vector['tf'],
                                                 len(decoder.buf)*64,balance,decoder,vector['lm'],coded,
                                                 vector['seed'],vector['disable_inv'])
                self.assertEqual(masks,vector['masks'])
                self.assertEqual(seed,vector['final_seed'])
                self.assert_context(decoder,vector['final_context'])
                for actual,expected in zip(values,vector['bands']):
                    self.assert_float_values(actual[:len(expected)],expected,2e-5)

    def test_native_pyramid_codebooks_rotation_and_collapse_masks(self):
        for index, vector in enumerate(self.components['pvq']):
            with self.subTest(vector=index):
                decoder = OpusRangeDecoder(base64.b64decode(vector['data']))
                actual,mask = unquant(decoder,vector['dimensions'],vector['pulses'],
                                     vector['spread'],vector['blocks'],vector['gain'])
                self.assert_float_values(actual,vector['values'],2e-5)
                self.assertEqual(mask,vector['mask'])
                self.assert_context(decoder,vector['context'])

    def test_native_float32_autocorrelation_and_lpc(self):
        for index, vector in enumerate(self.extra['helpers']):
            with self.subTest(vector=index):
                self.assertEqual(autocorr(list(floats(vector['samples_f32'])),vector['order'],vector['window']),
                                 vector['autocorr'])
                self.assertEqual(lpc_coefficients(vector['autocorr'],vector['order']),vector['lpc'])

    def test_native_pyramid_encoder_codebook_indices_and_range_bytes(self):
        for index, vector in enumerate(self.components['pulse_encoders']):
            with self.subTest(vector=index):
                encoder = OpusRangeEncoder(16)
                encode_pulses(encoder,vector['integers'])
                self.assert_context(encoder,vector['encoder_context'])
                self.assertEqual(encoder.done(),base64.b64decode(vector['encoded']))
                self.assert_context(encoder,vector['encoder_final'])

    def test_native_arbitrary_payloads_and_hybrid_shared_range_continuation(self):
        for index, vector in enumerate(self.extra['packets']):
            with self.subTest(vector=index):
                data = base64.b64decode(vector['data'])
                decoder = OpusRangeDecoder(data)
                for total in vector['prefix_uint_totals']:
                    decoder.uint(total)
                celt = CeltDecoder(vector['channels'],vector['sample_rate'])
                if vector['result'] < 0:
                    with self.assertRaises(OpusRangeError):
                        celt.decode_float(data,vector['duration'],stream_channels=vector['stream_channels'],
                                          start=vector['start'],end=vector['end'],range_decoder=decoder)
                else:
                    actual = celt.decode_float(data,vector['duration'],stream_channels=vector['stream_channels'],
                                               start=vector['start'],end=vector['end'],range_decoder=decoder)
                    expected = floats(vector['pcm_f32'])
                    scale = max(1.,max(abs(value) for value in expected))
                    self.assert_float_values(actual,expected,max(2/32768,scale*5e-6))
                self.assert_context(decoder,vector['context'])
                self.assertEqual({name:getattr(celt,name) for name in vector['state']},vector['state'])

    def test_reset_retains_format_and_restores_cold_decode(self):
        stream = next(item for item in self.received if len(base64.b64decode(item['frames'][0]['packet']))>2)
        frame = stream['frames'][0]
        payload = base64.b64decode(frame['packet'])[1:]
        decoder = CeltDecoder(stream['channels'])
        original = decoder.decode_float(payload,stream['duration'],stream_channels=frame['stream_channels'],end=frame['end'])
        decoder.decode_float(payload,stream['duration'])
        decoder.reset()
        self.assertEqual((decoder.stream_channels,decoder.start,decoder.end),
                         (frame['stream_channels'],0,frame['end']))
        self.assertEqual(decoder.decode_float(payload,stream['duration']),original)


if __name__ == '__main__':
    unittest.main()
