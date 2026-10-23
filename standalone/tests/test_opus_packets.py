"""Native Opus parser fixtures including self-delimited multiplexed streams."""
import base64
import hashlib
import json
from pathlib import Path
import unittest
import zlib

from standalone.opus import OpusError, pack_frames, parse_packet, samples_per_frame


class OpusPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = json.loads((Path(__file__).parent / 'fixtures/opus_packets.json').read_text())
        data = zlib.decompress(base64.b64decode(fixture['data_zlib_base64'], validate=True))
        if hashlib.sha256(data).hexdigest() != fixture['data_sha256']:
            raise AssertionError('Corrupted native Opus framing fixture')
        cls.vectors = json.loads(data)

    def test_native_valid_invalid_and_self_delimited_packets(self):
        for index, vector in enumerate(self.vectors):
            with self.subTest(case=index):
                data = base64.b64decode(vector['data'])
                if vector['result'] < 0:
                    with self.assertRaises(OpusError):
                        parse_packet(data, self_delimited=vector['self_delimited'])
                    continue
                packet = parse_packet(data, self_delimited=vector['self_delimited'])
                self.assertEqual(len(packet.frames), vector['result'])
                self.assertEqual([len(frame) for frame in packet.frames], vector['sizes'])
                self.assertEqual(packet.payload_offset, vector['payload_offset'])
                self.assertEqual(packet.packet_offset, vector['packet_offset'])
                position = vector['payload_offset']
                for frame, size in zip(packet.frames, vector['sizes']):
                    self.assertEqual(frame, data[position:position + size])
                    position += size

    def test_canonical_packer_preserves_native_validated_original_payloads(self):
        for index, vector in enumerate(self.vectors):
            if not vector['authored']:
                continue
            with self.subTest(case=index):
                data = base64.b64decode(vector['data'])
                frames = []
                position = vector['payload_offset']
                for size in vector['sizes']:
                    frames.append(data[position:position + size])
                    position += size
                padding = vector['packet_offset'] - position
                self.assertEqual(pack_frames(data[0], frames, padding=padding,
                                             self_delimited=vector['self_delimited']),
                                 data[:vector['packet_offset']])

    def test_duration_limit_and_known_toc_modes(self):
        self.assertEqual([samples_per_frame(toc) for toc in (0,8,16,24,96,104,128,136,144,152)],
                         [480,960,1920,2880,480,960,120,240,480,960])
        packet = parse_packet(pack_frames(128, [b'a'] * 48))
        self.assertEqual(packet.duration_ms, 120)
        self.assertEqual((packet.mode,packet.bandwidth,packet.channels), ('celt','narrowband',1))
        with self.assertRaises(OpusError):
            pack_frames(128, [b'a'] * 49)
        with self.assertRaises(OpusError):
            pack_frames(24, [b'a'] * 3)


if __name__ == '__main__':
    unittest.main()
