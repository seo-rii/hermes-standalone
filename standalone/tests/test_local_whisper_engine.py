"""Independent original CT2 fixtures for stdlib Whisper inference and search."""
import base64
import gc
import hashlib
import importlib.abc
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import weakref
import zlib

ROOT = Path(__file__).resolve().parent
from standalone import local_whisper_float
from standalone.local_whisper_engine import WhisperEngine
from standalone.local_whisper_features import MelFeatures
from standalone.local_whisper_search import SearchOptions, WhisperSearch
from standalone.local_whisper_tensor import EncodedAudio


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'numpy', 'av', 'tokenizers', 'ctranslate2', 'onnxruntime', 'torch'}:
            raise AssertionError('Whisper inference attempted to import ' + fullname)


def raw(value):
    return zlib.decompress(base64.b64decode(value, validate=True))


def floats(value):
    value = raw(value)
    return struct.unpack('<' + str(len(value) // 4) + 'f', value)


def audio(vector):
    values = floats(vector['audio_zlib_base64'])
    width = vector['audio_shape'][-1]
    return EncodedAudio(tuple(values[start:start + width] for start in range(0, len(values), width)))


class WhisperEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((ROOT / 'fixtures/local_whisper_engine.json').read_text())

    def setUp(self):
        guard = NoNative()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    def model_files(self, *, model=None, config=None, vocabulary=None):
        model = self.fixture['synthetic_model'] if model is None else model
        return {'model.bin': raw(model['model_zlib_base64']),
                'config.json': json.dumps(model['config'] if config is None else config).encode(),
                'vocabulary.json': json.dumps(model['vocabulary'] if vocabulary is None else vocabulary).encode()}

    def engine(self, compute_type='float32', **kwargs):
        engine = WhisperEngine('in-memory-original-model', device='cpu', compute_type=compute_type,
                               files=self.model_files(**kwargs))
        self.addCleanup(engine.close)
        return engine

    def test_original_float32_elementary_kernels(self):
        for vector in self.fixture['elementary']['vectors']:
            function = getattr(local_whisper_float, vector['operation'])
            with self.subTest(operation=vector['operation']):
                self.assertEqual(tuple(map(function, floats(vector['input']))), floats(vector['output']))

    def test_original_memory_filelike_cursor_and_compute_resolver(self):
        for vector in self.fixture['model_io']['vectors']:
            files = self.model_files()
            if vector['kind'] == 'filelike':
                files = {name: io.BytesIO(b'prefix' + data) for name, data in files.items()}
                for stream in files.values():
                    stream.seek(6)
            engine = WhisperEngine('in-memory-original-model', device=vector['device'],
                                   compute_type=vector['compute_type'], files=files)
            try:
                with self.subTest(vector=vector):
                    self.assertEqual({name: getattr(engine, name) for name in vector['expected']}, vector['expected'])
                    if vector['kind'] == 'filelike':
                        self.assertTrue(all(stream.tell() == len(stream.getvalue()) for stream in files.values()))
            finally:
                engine.close()

    def test_missing_memory_files_preserve_the_original_reader_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            files = self.model_files()
            for name, data in files.items():
                (path / name).write_bytes(data)
            for vector in self.fixture['model_io']['missing']:
                memory = {name: data for name, data in files.items() if name not in vector['missing']} if vector['missing'] else {}
                with self.subTest(missing=vector['missing']), self.assertRaises(RuntimeError) as captured:
                    WhisperEngine(path, device='cpu', files=memory)
                self.assertEqual(str(captured.exception), vector['message'].replace('{model_path}', str(path)))
        for files in ([], {1: b'bad'}, {'model.bin': 'bad'}, {'model.bin': io.StringIO('bad')}):
            with self.subTest(content_type=type(files).__name__), self.assertRaises(TypeError):
                WhisperEngine('unused-path', files=files)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "Unable to open file 'model.bin'"):
                WhisperEngine(directory)

    def test_idle_unload_and_explicit_close_release_the_mapped_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, data in self.model_files().items():
                (path / name).write_bytes(data)
            engine = WhisperEngine(path, device='cpu', compute_type='float32')
            weights, storage, reference = engine.weights, engine.weights._storage, weakref.ref(engine)
            self.assertFalse(storage.closed)
            del engine
            gc.collect()
            self.assertIsNone(reference())
            self.assertTrue(storage.closed)
            with self.assertRaisesRegex(ValueError, 'closed'):
                weights['decoder/embeddings/weight'].read(0, 1)
            engine = WhisperEngine(path, device_index=[0, 2], compute_type='default')
            self.assertEqual(engine.device_index, [0, 2])
            storage = engine.weights._storage
            engine.close()
            engine.close()
            self.assertTrue(storage.closed)

    def test_original_convolution_attention_ffn_and_encoder_lengths(self):
        engines = {kind: self.engine(kind) for kind in ('float32', 'int8_float32')}
        for vector in self.fixture['encoder_vectors']:
            values, length = floats(vector['input_zlib_base64']), vector['input_shape'][-1]
            features = tuple(values[start:start + length] for start in range(0, len(values), length))
            result = engines[vector['compute_type']].encode(MelFeatures(features))
            with self.subTest(vector=vector['id']):
                self.assertEqual(result.shape, tuple(vector['output_shape']))
                actual = tuple(value for row in result.rows for value in row)
                self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, floats(vector['output_zlib_base64']), strict=True)), 2e-5)

    def test_original_decoder_prefill_causality_immutable_kv_and_logits(self):
        engines = {kind: self.engine(kind) for kind in ('float32', 'int8_float32')}
        for vector in self.fixture['decoder_vectors']:
            decoder, memory, state = engines[vector['compute_type']].decoder, audio(vector), None
            for batch in vector['steps']:
                previous = state
                result = decoder.forward(batch['tokens'], memory, state)
                state = result.state
                logits = decoder.logits(result, range(len(batch['tokens'])))
                with self.subTest(vector=vector['id'], tokens=batch['tokens']):
                    actual = tuple(value for row in result.hidden for value in row)
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, floats(batch['hidden_zlib_base64']), strict=True)), 2e-5)
                    actual = tuple(value for row in logits for value in row)
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, floats(batch['logits_zlib_base64']), strict=True)), 2e-5)
                    self.assertEqual([max(range(len(row)), key=row.__getitem__) for row in logits], batch['top_tokens'])
                    if previous is not None:
                        self.assertEqual(len(previous.layers[0]['self']['keys']), previous.position)
                        self.assertEqual(state.position, previous.position + len(batch['tokens']))
        engine = engines['float32']
        memory = EncodedAudio(((0.0,) * 8,))
        result = engine.decoder.forward((1,), memory)
        with self.assertRaisesRegex(ValueError, 'another model or audio'):
            engine.decoder.forward((2,), EncodedAudio(memory.rows), result.state)
        with self.assertRaisesRegex(ValueError, 'another model or audio'):
            self.engine().decoder.forward((2,), memory, result.state)

    def test_original_beam_patience_prefix_penalties_and_timestamps(self):
        engines = {kind: self.engine(kind) for kind in ('float32', 'int8_float32')}
        for vector in self.fixture['search_vectors']:
            actual = engines[vector['compute_type']].generate(audio(vector), [vector['prompt']], **vector['options'])[0]
            errors = [abs(a - float(b)) if math.isfinite(a) and math.isfinite(float(b)) else 0.0 if a == float(b) else math.inf
                      for a, b in zip(actual.scores, vector['scores'], strict=True)]
            with self.subTest(vector=vector['id']):
                self.assertEqual(actual.sequences_ids, vector['sequences_ids'])
                self.assertLessEqual(max(errors), 2e-5)
                self.assertLessEqual(abs(actual.no_speech_prob - vector['no_speech_prob']), 2e-6)

    def test_original_language_detection(self):
        engines = {kind: self.engine(kind, vocabulary=self.fixture['multilingual_vocabulary'])
                   for kind in ('float32', 'int8_float32')}
        for vector in self.fixture['language_vectors']:
            actual = engines[vector['compute_type']].detect_language(audio(vector))
            with self.subTest(vector=vector['id']):
                self.assertEqual([item[0] for item in actual[0]], [item[0] for item in vector['expected'][0]])
                self.assertLessEqual(max(abs(a[1] - b[1]) for a, b in zip(actual[0], vector['expected'][0], strict=True)), 2e-6)

    def test_original_alignment_heads_text_probabilities_median_and_dtw(self):
        engines = {kind: self.engine(kind, config=self.fixture['align_config'], vocabulary=self.fixture['align_vocabulary'])
                   for kind in ('float32', 'int8_float32')}
        for vector in self.fixture['alignment_vectors']:
            actual = engines[vector['compute_type']].align(audio(vector), vector['start_sequence'], vector['text_tokens'],
                      vector['num_frames'], median_filter_width=vector['median_filter_width'])[0]
            with self.subTest(vector=vector['id']):
                self.assertEqual([list(pair) for pair in actual.alignments], vector['alignments'])
                self.assertLessEqual(max(abs(a - b) for a, b in zip(actual.text_token_probs, vector['text_token_probs'], strict=True)), 2e-7)

    def test_original_sampling_distribution_and_topk_support(self):
        fixture = self.fixture['sampling']
        search = WhisperSearch(None, self.fixture['synthetic_model']['vocabulary'], {})
        for vector in fixture['vectors']:
            search.random.seed(631278)
            options = SearchOptions(beam_size=1, sampling_topk=vector['sampling_topk'],
                                    sampling_temperature=vector['sampling_temperature'])
            candidates = [(value, 0, index) for index, value in enumerate(fixture['logits'])]
            counts = [0] * len(candidates)
            for _ in range(vector['iterations']):
                counts[search.sample(candidates, 1, options)[0][2]] += 1
            for index, (actual, expected) in enumerate(zip(counts, vector['native_counts'], strict=True)):
                with self.subTest(topk=vector['sampling_topk'], temperature=vector['sampling_temperature'], token=index):
                    combined = actual + expected
                    probability = combined / (2 * vector['iterations'])
                    self.assertLessEqual(abs(actual - expected), 6 * math.sqrt(combined * (1 - probability)) + 12)
                    if vector['sampling_topk'] and expected == 0:
                        self.assertEqual(actual, 0)
        with self.assertRaises(ValueError):
            SearchOptions(beam_size=1, num_hypotheses=2, sampling_topk=1)
        self.assertEqual(SearchOptions(beam_size=1, num_hypotheses=3, sampling_topk=0).num_hypotheses, 3)
        for kwargs in ({'beam_size': 0}, {'patience': 0}, {'repetition_penalty': 0}, {'sampling_temperature': 0},
                       {'max_length': 0}, {'beam_size': 2, 'num_hypotheses': 3}):
            with self.subTest(options=kwargs), self.assertRaises(ValueError):
                SearchOptions(**kwargs)

    def test_original_saved_half_int8_int16_and_cpu_compute_conversion(self):
        fixture = self.fixture['stored_types']
        values, length = floats(fixture['features']), fixture['feature_shape'][-1]
        features = tuple(values[start:start + length] for start in range(0, len(values), length))
        for model in fixture['models']:
            self.assertEqual(hashlib.sha256(raw(model['model_zlib_base64'])).hexdigest(), model['model_sha256'])
            for vector in model['vectors']:
                engine = self.engine(vector['requested_compute'], model=model)
                encoded = engine.encode(features)
                actual = tuple(value for row in encoded.rows for value in row)
                result = engine.generate(encoded, [[1, 9]], **vector['options'])[0]
                with self.subTest(saved=model['saved_type'], compute=vector['requested_compute']):
                    self.assertEqual(engine.compute_type, vector['effective_compute'])
                    self.assertEqual(encoded.shape, tuple(vector['shape']))
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, floats(vector['encoded']), strict=True)), 2e-5)
                    self.assertEqual(result.sequences_ids, vector['sequences_ids'])
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(result.scores, vector['scores'], strict=True)), 2e-5)
                    self.assertLessEqual(abs(result.no_speech_prob - vector['no_speech_prob']), 2e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
