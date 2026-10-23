"""Standard-library audio packets independently decoded by native Opus 1.3.1."""
import base64
import hashlib
import json
from pathlib import Path
import struct
import unittest
import zlib

from standalone.opus import parse_packet, samples_per_frame
from standalone.opus_celt import CeltDecoder
from standalone.opus_celt_encoder import CeltEncoder, mdct


def fixture(name):
    wrapper = json.loads((Path(__file__).parent/'fixtures'/name).read_text())
    payload = zlib.decompress(base64.b64decode(wrapper['data_zlib_base64'],validate=True))
    if hashlib.sha256(payload).hexdigest() != wrapper['data_sha256']:
        raise AssertionError('Corrupted Opus encoder interoperability fixture')
    return json.loads(payload)


def floats(encoded):
    data = base64.b64decode(encoded)
    return struct.unpack('<'+'f'*(len(data)//4),data)


class OpusCeltEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture('opus_celt_encoder.json')
        cls.controls = fixture('opus_celt_encoder_extra.json')

    def assert_native_pcm(self,decoder,packet,frame):
        parsed = parse_packet(packet)
        end = (13,17,19,21)[(parsed.toc>>5)&3]
        actual = []
        for payload in parsed.frames:
            actual.extend(decoder.decode_float(payload,samples_per_frame(parsed.toc),
                                               stream_channels=parsed.channels,end=end))
        expected = floats(frame['decoded_f32'])
        self.assertEqual(len(actual),len(expected))
        self.assertLessEqual(max(abs(a-b) for a,b in zip(actual,expected)),2/32768)
        self.assertEqual(decoder.rng,frame['rng'])

    def test_native_interoperable_packets_are_deterministic_for_all_rates_and_durations(self):
        for index,stream in enumerate(self.data['streams']):
            with self.subTest(stream=index):
                encoder = CeltEncoder(stream['channels'],stream['bitrate'])
                decoder = CeltDecoder(stream['channels'])
                for frame in stream['frames']:
                    packet = encoder.encode(base64.b64decode(frame['pcm_s16']),stream['duration'])
                    self.assertEqual(packet,base64.b64decode(frame['packet']))
                    self.assertEqual(encoder.rng,frame['rng'])
                    self.assert_native_pcm(decoder,packet,frame)
                    self.assertLessEqual(max(abs(a-b) for a,b in zip(encoder.old_energy,decoder.old_energy)),1e-10)

    def test_native_bandwidth_bitrate_controls_and_multi_frame_packets(self):
        for index,stream in enumerate(self.controls):
            with self.subTest(stream=index):
                encoder = CeltEncoder(stream['channels'])
                decoder = CeltDecoder(stream['channels'])
                for frame in stream['frames']:
                    encoder.set_bandwidth(frame['bandwidth'])
                    self.assertEqual(encoder.set_bitrate(frame['bitrate']),frame['bitrate'])
                    packet = encoder.encode(base64.b64decode(frame['pcm_s16']),stream['duration'])
                    self.assertEqual(packet,base64.b64decode(frame['packet']))
                    self.assertEqual(encoder.rng,frame['rng'])
                    self.assert_native_pcm(decoder,packet,frame)

    def test_forward_mdct_matches_independent_c_reference(self):
        for index,vector in enumerate(self.data['mdct']):
            with self.subTest(vector=index):
                actual = mdct(list(floats(vector['input_f32'])),vector['lm'])
                expected = floats(vector['output_f32'])
                self.assertLessEqual(max(abs(a-b) for a,b in zip(actual,expected)),.0012)

    def test_float_and_signed_pcm16_entrypoints_and_reset(self):
        stream = self.data['streams'][-1]
        pcm = base64.b64decode(stream['frames'][2]['pcm_s16'])
        values = struct.unpack('<'+'h'*(len(pcm)//2),pcm)
        first = CeltEncoder(stream['channels'],stream['bitrate'],'wide')
        second = CeltEncoder(stream['channels'],stream['bitrate'],'wide')
        expected = first.encode(pcm,stream['duration'])
        self.assertEqual(second.encode_float([value/32768 for value in values],stream['duration']),expected)
        second.encode(pcm,stream['duration'])
        second.reset()
        self.assertEqual((second.bitrate,second.bandwidth),(stream['bitrate'],'wide'))
        self.assertEqual(second.encode(pcm,stream['duration']),expected)
        with self.assertRaises(ValueError):
            second.encode_float([float('nan')]*(stream['duration']*stream['channels']),stream['duration'])
        with self.assertRaises(ValueError):
            second.encode(pcm[:-2],stream['duration'])


if __name__ == '__main__':
    unittest.main()
