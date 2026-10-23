"""Real Silero ONNX and original faster-whisper VAD consumer golden cases."""

from array import array
import base64
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zlib

from standalone import local_whisper_vad as vad
from standalone.local_whisper_vad_model import MODEL_BYTES, MODEL_SHA256, SileroVADModel


class LocalWhisperVadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = json.loads((Path(__file__).parent/'fixtures/local_whisper_vad.json').read_text())
        data = zlib.decompress(base64.b64decode(fixture['data_zlib_base64'], validate=True))
        if hashlib.sha256(data).hexdigest() != fixture['data_sha256']:
            raise AssertionError('Silero VAD native fixture digest mismatch')
        cls.cases = json.loads(data)
        cls.model = SileroVADModel()

    def test_original_asset_and_strict_model_reader(self):
        asset = Path(vad.__file__).with_name('assets')/'silero_vad_v6.onnx'
        self.assertEqual(asset.stat().st_size, MODEL_BYTES)
        self.assertEqual(hashlib.sha256(asset.read_bytes()).hexdigest(), MODEL_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'wrong.onnx'
            path.write_bytes(b'not the reviewed model')
            with self.assertRaisesRegex(ValueError, 'reviewed'):
                SileroVADModel(path)
        with self.assertRaises(ValueError):
            self.model([0.0]*511)
        with self.assertRaises(ValueError):
            self.model([0.0]*512, context_size_samples=32)

    def test_native_onnx_probabilities_and_recurrent_states(self):
        self.assertEqual(len(self.cases['model_steps']), 60)
        hidden = cell = None
        for number, case in enumerate(self.cases['model_steps']):
            with self.subTest(window=number):
                if case['reset']:
                    hidden = cell = None
                probability, hidden, cell = self.model.step(case['input'], hidden, cell)
                self.assertLessEqual(abs(probability-case['probability']), 1e-5)
                self.assertLessEqual(max(abs(a-b) for a, b in zip(hidden, case['hidden'], strict=True)), 5e-5)
                self.assertLessEqual(max(abs(a-b) for a, b in zip(cell, case['cell'], strict=True)), 5e-5)

    def test_original_ten_thousand_window_batch_boundary(self):
        case = self.cases['large_batch_boundary']
        probability, hidden, cell = self.model.step(case['input'], case['hidden'], case['cell'])
        self.assertLessEqual(abs(probability-case['probability']), 1e-5)
        for actual, expected in zip(hidden+cell, case['next_hidden']+case['next_cell'], strict=True):
            # Large carried LSTM cells need a relative term: the native
            # 10,000-window state reaches 6,883 and one float32 ULP is 0.000488.
            self.assertLessEqual(abs(actual-expected), 5e-5 + 1e-6*abs(expected))

    def test_actual_speech_context_padding_probability_and_segment_consumer(self):
        self.assertEqual(len(self.cases['audio']), 5)
        for case in self.cases['audio']:
            with self.subTest(audio=case['name']):
                raw = base64.b64decode(case['pcm_s16le_base64'], validate=True)
                audio = [value[0]/32768 for value in struct.iter_unpack('<h', raw)]
                padded = audio + [0.0]*(512-len(audio)%512)
                probabilities = self.model(padded)
                self.assertEqual(len(probabilities), len(case['probabilities']))
                self.assertLessEqual(max(abs(a-b) for a, b in zip(probabilities, case['probabilities'], strict=True)), 1e-5)
                with mock.patch.object(vad, 'get_vad_model', return_value=lambda _: probabilities):
                    actual = vad.get_speech_timestamps(audio, vad.VadOptions(**case['options']))
                self.assertEqual(actual, case['segments'])
                if case['name'] == 'tail':
                    # A real whole-audio call also exercises the asset/cache binding.
                    self.assertEqual(vad.get_speech_timestamps(audio, **case['options']), case['segments'])

    def test_original_threshold_hysteresis_duration_and_padding(self):
        self.assertEqual(len(self.cases['segments']), 125)
        for number, case in enumerate(self.cases['segments']):
            with self.subTest(case=number):
                audio = array('f', [0.0])*case['length']
                with mock.patch.object(vad, 'get_vad_model', return_value=lambda _: case['probabilities']):
                    actual = vad.get_speech_timestamps(audio, vad.VadOptions(**case['options']))
                self.assertEqual(actual, case['output'])

    def test_original_chunk_audio_metadata_and_timestamp_mapping(self):
        self.assertEqual(len(self.cases['chunks']), 12)
        self.assertEqual(len(self.cases['timestamps']), 96)
        raw = base64.b64decode(self.cases['chunk_audio_f32le_base64'], validate=True)
        audio = array('f', (value[0] for value in struct.iter_unpack('<f', raw)))
        for number, case in enumerate(self.cases['chunks']):
            with self.subTest(chunks=number):
                actual, metadata = vad.collect_chunks(audio, case['chunks'], max_duration=case['max_duration'])
                self.assertEqual([list(chunk) for chunk in actual], case['output'])
                self.assertEqual(metadata, case['metadata'])
        for number, case in enumerate(self.cases['timestamps']):
            with self.subTest(timestamp=number):
                mapping = vad.SpeechTimestampsMap(case['chunks'], 16000, case['precision'])
                self.assertEqual(mapping.get_chunk_index(case['time'], case['is_end']), case['index'])
                self.assertEqual(mapping.get_original_time(case['time'], is_end=case['is_end']), case['output'])


if __name__ == '__main__':
    unittest.main()
