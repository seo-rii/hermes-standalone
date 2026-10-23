"""Independent native wake-model fixtures, PCM and original listener lifecycle."""

import base64
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import unittest
import zlib

from standalone import wake_audio
from standalone.audio_samples import PCM16, from_samples
from standalone.wake_openword import OpenWakeWord, OpenWakeWordFeatures
from standalone.wake_tflite import TFLiteModel, Tensor
from standalone.tests.test_entrypoints import BundledSourceTestCase, support


PROJECT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures")
MODELS = {"classifier": PROJECT / "tools/wakewords/hey_hermes.tflite",
          "mel": PROJECT / "standalone/wake_models/melspectrogram.tflite",
          "embedding": PROJECT / "standalone/wake_models/embedding_model.tflite"}


def _floats(value):
    raw = zlib.decompress(base64.b64decode(value, validate=True))
    return struct.unpack("<" + "f" * (len(raw) // 4), raw)


class WakeModelTests(unittest.TestCase):
    def test_original_model_weights_match_independent_native_cpu_outputs(self):
        fixture = json.loads((FIXTURES / "wake_tflite.json").read_text())
        for name, path in MODELS.items():
            model = TFLiteModel.from_file(path)
            for vector in (value for value in fixture["vectors"] if value["model"] == name):
                with self.subTest(vector=vector["id"]):
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), vector["model_sha256"])
                    result = model.invoke(_floats(vector["input_zlib_base64"]), vector["input_shape"])
                    self.assertEqual(result.shape, tuple(vector["output_shape"]))
                    expected = _floats(vector["output_zlib_base64"])
                    tolerance = {"classifier": 1e-7, "mel": 5e-5, "embedding": 2e-4}[name]
                    self.assertEqual(len(result.values), len(expected))
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(result.values, expected)), tolerance)

    def test_incremental_convolution_preserves_cold_float32_tensors(self):
        path = MODELS["embedding"]
        fixture = json.loads((FIXTURES / "wake_tflite.json").read_text())
        vector = next(value for value in fixture["vectors"] if value["id"] == "embedding-noise")
        values = list(_floats(vector["input_zlib_base64"]))
        model = TFLiteModel.from_file(path)
        model.invoke(values)
        for frame in range(3):
            values = values[256:] + [(frame * 13 + index % 47) / 10 for index in range(256)]
            actual = model.invoke(values, temporal_shift=8)
            expected = TFLiteModel.from_file(path).invoke(values)
            self.assertEqual(actual.values, expected.values)
        # A caller's overlap hint cannot reuse a stale prefix after real input
        # changes, including a capture reset or a skipped frame.
        values[0] += 1
        self.assertEqual(model.invoke(values, temporal_shift=8).values,
                         TFLiteModel.from_file(path).invoke(values).values)

    def test_corrupt_models_and_invalid_inputs_fail_before_inference(self):
        valid = MODELS["classifier"].read_bytes()
        for raw in (b"", valid[:7], valid[:19], b"\xff" * 4 + valid[4:], valid[:4] + b"BAD!" + valid[8:]):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                TFLiteModel(raw)
        model = TFLiteModel(valid)
        for values, shape in (([0], None), (itertools.repeat(0), (1,)), ([math.nan], (1,)), ([math.inf], (1,))):
            with self.assertRaises(ValueError):
                model.invoke(values, shape)
        with self.assertRaises(ValueError):
            Tensor((2**30,), ())
        with self.assertRaises(ValueError):
            Tensor((1,), [2**31], 2)

    def test_pcm_frame_splitting_padding_peak_and_original_resampling_rules(self):
        source = from_samples([-32768, -1, 0, 1, 32767])
        frames = list(wake_audio.frames(source, 3))
        self.assertEqual([list(frame.values()) for frame in frames], [[-32768, -1, 0], [1, 32767, 0]])
        self.assertEqual(wake_audio.peak(source), 32768)
        cases = [([0, 1, 2, 3, 4, 5], 3, [0, 2, 4]),
                 ([-32768, 32767], 1, [0]), ([0, 10], 5, [0, 2, 5, 8, 10]),
                 ([12], 4, [12] * 4), ([], 2, [0, 0])]
        for samples, count, expected in cases:
            with self.subTest(samples=samples, count=count):
                self.assertEqual(list(wake_audio.resample_frame(from_samples(samples), count).values()), expected)
        for value in (0, -1, 65537):
            with self.assertRaises(ValueError):
                list(wake_audio.frames(b"\0\0", value))
        with self.assertRaises(ValueError):
            list(wake_audio.frames(b"x", 1280))


class WakeStreamingTests(unittest.TestCase):
    @unittest.skipIf(sys.version_info < (3, 14), "Full Hermes runtime targets Python 3.14; portable operators are tested separately")
    def test_twenty_second_original_stream_embeddings_scores_and_trigger_sequences(self):
        fixture = json.loads((FIXTURES / "wake_stream.json").read_text())
        audio = zlib.decompress(base64.b64decode(fixture["audio_zlib_base64"], validate=True))
        self.assertEqual(hashlib.sha256(audio).hexdigest(), fixture["audio_sha256"])
        features = OpenWakeWordFeatures.from_builtin()
        model = OpenWakeWord.from_model(MODELS["classifier"])
        all_scores, elapsed, maximum = [], [], 0.0
        try:
            for index, expected in enumerate(fixture["records"]):
                start = time.monotonic()
                embeddings = list(features.process_streaming(audio[index * 2560:(index + 1) * 2560]))
                scores = [score for block in embeddings for score in model.process_streaming(block)]
                elapsed.append(time.monotonic() - start)
                rows = [row for block in embeddings for row in block]
                self.assertEqual(len(rows), len(expected["embeddings"]), index)
                error = max((abs(a - b) for row, reference in zip(rows, expected["embeddings"])
                             for a, b in zip(row, reference, strict=True)), default=0.0)
                maximum = max(maximum, error)
                self.assertLessEqual(error, 2e-4, index)
                self.assertEqual(len(scores), len(expected["scores"]), index)
                self.assertLessEqual(max((abs(a - b) for a, b in zip(scores, expected["scores"])), default=0), 1e-7, index)
                all_scores.append(scores)
            for sequence in fixture["trigger_sequences"]:
                streak, actual = 0, []
                for index, scores in enumerate(all_scores):
                    if any(value >= sequence["threshold"] for value in scores):
                        streak += 1
                        if streak >= sequence["confirmation_frames"]:
                            actual.append(index)
                            streak = 0
                    else:
                        streak = 0
                self.assertEqual(actual, sequence["indices"], sequence["threshold"])
            print(json.dumps({"wake_stream_frames": len(all_scores), "embedding_max_absolute_error": maximum,
                              "first_frame_seconds": elapsed[0], "steady_median_seconds": sorted(elapsed[1:])[len(elapsed[1:]) // 2],
                              "steady_max_seconds": max(elapsed[1:]), "audio_frame_seconds": .08}), flush=True)
            features.reset()
            model.reset()
            self.assertEqual(list(features.process_streaming(b"")), [])
            for invalid in (b"x", b"\0" * (320000 + 2), "not PCM"):
                with self.assertRaises(ValueError):
                    list(features.process_streaming(invalid))
        finally:
            features.close()
            model.close()
        self.assertEqual(list(features.process_streaming(b"\0\0")), [])
        self.assertEqual(list(model.process_streaming([(0.0,) * 96])), [])


class WakeOriginalConsumerTests(BundledSourceTestCase):
    def test_client_pcm_reaches_original_engine_listener_and_bounded_queue_without_native_packages(self):
        worker = r'''
import importlib.abc,json,os,pathlib,sys,threading,time
context=json.loads(sys.argv[1]); scratch=pathlib.Path(sys.argv[2]); source=pathlib.Path(context["root"])/"app"
sys.path[:0]=[str(source),str(source.parent/"vendor")]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in ("numpy","sounddevice","pyopen_wakeword","tensorflow","onnxruntime"):
            raise AssertionError("Native inference dependency reached: "+fullname)
sys.meta_path.insert(0,NoNative())
from standalone.agent_probe_worker import ProbeGuard
guard=ProbeGuard(source,source.parent/"vendor",scratch);sys.addaudithook(guard)
import pm
def no_install(*args,**kwargs):raise AssertionError("Runtime installation attempted")
pm.ensure_import=no_install
from tools import wake_word as ww
from standalone.audio_samples import PCM16
engine=ww._OpenWakeWordEngine({"provider":"openwakeword","capture":"client","sensitivity":0.0,"confirmation_frames":3})
fired=threading.Event(); callbacks=[]
def on_wake():callbacks.append(True);fired.set()
detector=ww.WakeWordDetector(engine,on_wake,cooldown=0,external_audio=True)
assert not detector.running
detector.feed(b"\0"*(1280*2*70))
assert detector._audio_q.qsize()==64
assert all(isinstance(frame,PCM16) for frame in list(detector._audio_q.queue))
detector.start()
try:
    detector.feed(b"\0"*(1280*2*3))
    assert fired.wait(20),"Original listener did not trigger after three actual model frames"
    assert len(callbacks)==1
    detector.pause();fired.clear()
    detector.feed(b"\0"*(1280*2*6))
    detector.resume()  # stale queued frames must be discarded before the new arm
    assert not fired.wait(.1)
    detector.feed(b"\0"*(1280*2*3))
    assert fired.wait(20)
    assert len(callbacks)==2
finally:detector.stop()
assert not detector.running
assert "numpy" not in sys.modules and "pyopen_wakeword" not in sys.modules
assert not guard.events,guard.events
print(json.dumps({"original_listener_triggers":len(callbacks),"bounded_queue":64,"blocked_events":guard.events}),flush=True)
'''
        with __import__("tempfile").TemporaryDirectory(prefix="hermes-wake-original-") as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", worker,
                                     json.dumps(self.context), str(scratch)], cwd=scratch / "work", env=environment,
                                    capture_output=True, text=True, encoding="utf-8", timeout=75)
        self.assertEqual(result.returncode, 0, result.stderr[-5000:] + result.stdout[-2000:])
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["original_listener_triggers"], 2)
