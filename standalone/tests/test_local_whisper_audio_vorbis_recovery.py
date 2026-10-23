"""Original native malformed-packet state and Faster Whisper prefix semantics."""
import base64
import hashlib
import importlib.abc
import io
import json
import lzma
from pathlib import Path
import struct
import sys
import unittest
import zlib


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {
            'av', 'numpy', 'ctypes', 'subprocess', 'torch', 'ctranslate2',
            'soundfile', 'onnxruntime', 'tokenizers', 'pydub', 'audioread',
        }:
            raise AssertionError('Vorbis recovery attempted native or process import ' + fullname)


class VorbisRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guard = NoNative()
        sys.meta_path.insert(0, guard)
        cls.addClassCleanup(sys.meta_path.remove, guard)
        from standalone.local_whisper_audio import decode_audio
        from standalone.local_whisper_audio_vorbis import decode_vorbis
        from standalone.local_whisper_audio_vorbis_bits import VorbisError
        from standalone.local_whisper_audio_vorbis_codec import VorbisDecoder
        cls.frontend, cls.decode = staticmethod(decode_audio), staticmethod(decode_vorbis)
        cls.error, cls.decoder = VorbisError, VorbisDecoder
        path = Path(__file__).parent / 'fixtures/local_whisper_audio_vorbis_recovery.json'
        cls.wrapper = json.loads(path.read_bytes())
        metadata = zlib.decompress(base64.b64decode(cls.wrapper['metadata_zlib_base64'], validate=True))
        if hashlib.sha256(metadata).hexdigest() != cls.wrapper['metadata_sha256']:
            raise AssertionError('Recovery metadata digest mismatch')
        cls.metadata = json.loads(metadata)
        shuffled = lzma.decompress(base64.b64decode(cls.wrapper['pool_planes_xz_base64'], validate=True),
                                   memlimit=32 << 20)
        if len(shuffled) != cls.wrapper['pool_bytes'] or len(shuffled) % 4:
            raise AssertionError('Recovery native byte pool size mismatch')
        words = len(shuffled) // 4
        cls.pool = bytearray(len(shuffled))
        for i in range(4):
            cls.pool[i::4] = shuffled[i * words:(i + 1) * words]
        if hashlib.sha256(cls.pool).hexdigest() != cls.wrapper['pool_sha256']:
            raise AssertionError('Recovery native byte pool digest mismatch')
        for blob in cls.metadata['blobs']:
            raw = cls.pool[blob['offset']:blob['offset'] + blob['bytes']]
            if len(raw) != blob['bytes'] or hashlib.sha256(raw).hexdigest() != blob['sha256']:
                raise AssertionError('Recovery original native blob mismatch')
        cls.cases = cls.metadata['document']['cases']

    @classmethod
    def tearDownClass(cls):
        del cls.pool, cls.metadata, cls.cases, cls.wrapper

    def _bytes(self, ref):
        blob = self.metadata['blobs'][ref['binary_blob']]
        return bytes(self.pool[blob['offset']:blob['offset'] + blob['bytes']])

    def _values(self, ref, kind):
        raw = self._bytes(ref)
        width = 4 if kind == 'f' else 2
        return struct.unpack('<' + str(len(raw) // width) + kind, raw)

    def _restore(self, value):
        if isinstance(value, dict):
            if set(value) == {'binary_blob'}:
                return base64.b64encode(self._bytes(value)).decode()
            return {k: self._restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._restore(v) for v in value]
        return value

    def _packet_pcm(self, event, channels):
        rows = [[] for _ in range(channels)]
        for frame in event['frames']:
            for row, ref in zip(rows, frame['pcm'], strict=True):
                row.extend(self._values(ref, 'f'))
        return rows

    def test_direct_packet_decoder_matches_native_state_before_and_after_errors(self):
        errors, accepted_short = 0, 0
        self.assertEqual(len(self.cases), 64)
        for case in self.cases:
            with self.subTest(case=case['name']):
                headers = [self._bytes(v) for v in case['headers']]
                decoder = self.decoder(headers[0], headers[2])
                for packet, event in zip(case['packets'], case['profiles']['generic']['packets'], strict=True):
                    if 'error' in event:
                        errors += 1
                        with self.assertRaises(self.error):
                            decoder.decode(self._bytes(packet['data']))
                    else:
                        raw = self._bytes(packet['data'])
                        accepted_short += len(raw) == 1
                        self.assertEqual(decoder.decode(raw), self._packet_pcm(event, case['channels']))
        self.assertEqual(errors, 14)
        self.assertGreater(accepted_short, 0)

    def _frontend_case(self, case, event):
        source = io.BytesIO(self._bytes(case['data']))
        self.assertEqual(hashlib.sha256(source.getvalue()).hexdigest(), case['input_sha256'])
        output = self.frontend(source, event['rate'], event['split'])
        self.assertFalse(source.closed)
        rows = list(output) if event['split'] else [output]
        for actual, ref, count in zip(rows, event['pcm'], event['samples'], strict=True):
            expected = self._values(ref, 'h')
            self.assertEqual(len(actual), count)
            self.assertEqual(len(expected), count)
            delta = max((abs(round(float(a) * 32768) - b) for a, b in zip(actual, expected, strict=True)), default=0)
            self.assertLessEqual(delta, 1)

    def test_original_frontend_retains_prefix_and_stops_at_first_invalid_packet(self):
        count, empty, nonempty = 0, 0, 0
        for case in self.cases:
            for event in case['profiles']['generic']['frontend']:
                with self.subTest(case=case['name'], rate=event['rate'], split=event['split']):
                    self._frontend_case(case, event)
                    count += 1
                    empty += not event['samples'][0]
                    nonempty += bool(event['samples'][0])
        self.assertEqual(count, 128)
        self.assertGreater(empty, 0)
        self.assertGreater(nonempty, 0)

    def test_strict_file_refuses_invalid_packets_and_opted_prefix_is_native_exact(self):
        count = 0
        for case in self.cases:
            events = case['profiles']['generic']['packets']
            if not any('error' in event for event in events):
                continue
            with self.subTest(case=case['name']):
                data = self._bytes(case['data'])
                with self.assertRaises(self.error):
                    self.decode(data)
                rate, rows, fmt, mask = self.decode(data, stop_on_invalid=True)
                expected = [[] for _ in range(case['channels'])]
                for event in events:
                    if 'error' in event:
                        break
                    for target, channel in zip(expected, self._packet_pcm(event, case['channels']), strict=True):
                        target.extend(channel)
                self.assertEqual((rate, len(rows), fmt), (case['rate'], case['channels'], 'fltp'))
                self.assertEqual(rows, expected)
                count += 1
        self.assertEqual(count, 14)

    def test_lossless_pool_restores_the_complete_original_native_document(self):
        document = self._restore(self.metadata['document'])
        raw = json.dumps(document, separators=(',', ':')).encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), self.wrapper['source_sha256'])
        self.assertEqual(self.wrapper['source_sha256'], '9ac9c49e1f2d462fffa0bd6df455bcd5e4ee81afa32a9bdc72a05f382d0db077')
        self.assertTrue(all(set(c['profiles']) == {'generic', 'host'} for c in document['cases']))


if __name__ == '__main__':
    unittest.main()
