"""Independent native fixtures for the stdlib Whisper model-data frontend."""
import base64
import hashlib
import importlib.abc
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures/local_whisper_frontend.json"


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"numpy", "av", "tokenizers", "ctranslate2", "onnxruntime", "torch"}:
            raise AssertionError("A stdlib Whisper unit attempted to import " + fullname)


from standalone.local_whisper_features import FeatureExtractor
from standalone.local_whisper_tokenizer import ByteBPETokenizer
from standalone.local_whisper_weights import ModelWeights


def raw(value):
    return zlib.decompress(base64.b64decode(value, validate=True))


def floats(value):
    value = raw(value)
    return struct.unpack("<" + str(len(value) // 4) + "f", value)


class FrontendTests(unittest.TestCase):
    def setUp(self):
        guard = NoNative()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())

    def test_original_serializer_all_weight_types_and_aliases(self):
        fixture = self.fixture["weights"]
        data = raw(fixture["model_zlib_base64"])
        self.assertEqual(hashlib.sha256(data).hexdigest(), fixture["model_sha256"])
        for on_disk in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "model.bin"
                path.write_bytes(data)
                with ModelWeights(path if on_disk else data) as model:
                    self.assertEqual((model.binary_version, model.specification, model.revision), (6, "WhisperSpec", 3))
                    for name, vector in fixture["variables"].items():
                        tensor = model[name]
                        self.assertEqual(tensor.shape, tuple(vector["shape"]))
                        self.assertEqual(tensor.dtype, vector["dtype"])
                        self.assertEqual(tensor.read(), tuple(vector["values"]))
                        if tensor.size > 1:
                            self.assertEqual(tensor.read(1, 1), tuple(vector["values"][1:2]))
                    for alias, name in fixture["aliases"].items():
                        self.assertIs(model[alias], model[name])
                    self.assertIs(model["quantized_alias_scale"], model["int8_scale"])
                    self.assertEqual(model["int8"].floats(), (-1, 0, 1, -1, 0, 1))
                    self.assertEqual(model["int16"].floats(), (-1, -0.5, 0.5, 1))
                    tensor = model["float32"]
                    self.assertEqual(tensor.row(1), tensor.read(3, 3))
                    with self.assertRaises(ValueError):
                        tensor.read(0, 2**30)
                with self.assertRaises(ValueError):
                    tensor.read(0, 1)

    def test_truncated_corrupt_models_and_closed_views_refuse(self):
        data = raw(self.fixture["weights"]["model_zlib_base64"])
        bad = [data[:index] for index in range(0, min(len(data), 70))]
        bad += [data[:-1], data + b"x", b"\x07\0\0\0" + data[4:], data[:4] + b"\xff\xff" + data[6:]]
        for value in bad:
            with self.subTest(length=len(value)), self.assertRaises(ValueError):
                ModelWeights(value)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.bin"
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                ModelWeights(path, max_bytes=len(data) - 1)

    def test_original_logmel_frontend_native_numpy_vectors(self):
        maximum = 0
        cache = {}
        for vector in self.fixture["features"]:
            config = tuple(sorted(vector["config"].items()))
            if config not in cache:
                cache[config] = FeatureExtractor(**vector["config"])
            frontend = cache[config]
            actual = frontend(floats(vector["input_zlib_base64"]), padding=vector["padding"])
            expected = floats(vector["output_zlib_base64"])
            with self.subTest(vector=vector["id"]):
                self.assertEqual(actual.shape, tuple(vector["shape"]))
                values = tuple(value for row in actual.rows for value in row)
                error = max(abs(first - second) for first, second in zip(values, expected, strict=True))
                maximum = max(maximum, error)
                self.assertLessEqual(error, 2e-5)
                trimmed = actual.pad_or_trim(5, start=1)
                self.assertEqual(trimmed.shape, (actual.shape[0], 5))
                self.assertEqual(trimmed.rows[0][:min(5, max(0, actual.shape[1] - 1))], actual.rows[0][1:6])
        print(json.dumps({"whisper_logmel_vectors": len(self.fixture["features"]), "maximum_error": maximum}), flush=True)

    def test_original_rust_tokenizer_ids_decoding_and_templates(self):
        cases = 0
        for model in self.fixture["tokenizers"]:
            tokenizer = ByteBPETokenizer(raw(model["graph_zlib_base64"]))
            for vector in model["vectors"]:
                with self.subTest(model=model["model"], text=vector["text"][:50]):
                    self.assertEqual(tokenizer.encode(vector["text"]), vector["ids"])
                    self.assertEqual(tokenizer.encode(vector["text"], add_special_tokens=True), vector["with_special_ids"])
                    self.assertEqual(tokenizer.decode(vector["ids"]), vector["decoded"])
                    self.assertEqual(tokenizer.decode(vector["with_special_ids"], skip_special_tokens=False), vector["decoded_all"])
                cases += 1
            for vector in model["decode_vectors"]:
                self.assertEqual(tokenizer.decode(vector["ids"]), vector["default"])
                self.assertEqual(tokenizer.decode(vector["ids"], skip_special_tokens=False), vector["all"])
        print(json.dumps({"whisper_tokenizer_vectors": cases}), flush=True)

    def test_unsupported_graphs_and_nonfinite_audio_refuse(self):
        for data in (b"", b"{}", b"[0]", b"{not json}"):
            with self.assertRaises(ValueError):
                ByteBPETokenizer(data)
        graph = json.loads(raw(self.fixture["tokenizers"][0]["graph_zlib_base64"]))
        graph["model"]["dropout"] = 0.5
        with self.assertRaises(ValueError):
            ByteBPETokenizer(json.dumps(graph).encode())
        frontend = FeatureExtractor(n_fft=16, feature_size=4)
        for waveform in ([float("nan")], [float("inf")], []):
            with self.assertRaises(ValueError):
                frontend(waveform, padding=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
