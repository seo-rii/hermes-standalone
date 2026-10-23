"""The pinned pyopen-wakeword streaming algorithm with stdlib model inference.

The original 16 kHz PCM, 8-second silence prefill, mel normalization, 76-row
embedding windows, 8-row stride and classifier history are preserved. Model
bytes are unchanged. This qualified consumer never selects a hosted provider,
loads a shared library, installs a package or downloads a model.

Derived from pyopen-wakeword 1.1.0 openwakeword.py (Apache-2.0);
notice and source/model hashes: licenses/pyopen-wakeword.txt,
wake-provenance.json. Embeddings are explicit rows of 96 floats, not arrays.
"""

from array import array
from collections import deque
import math
from pathlib import Path
import struct

from standalone.wake_tflite import TFLiteModel


_MODELS = Path(__file__).with_name("wake_models")
_MAX_AUDIO = 160000
_MAX_MELS = 970
_MAX_EMBEDDINGS = 80


class OpenWakeWord:
    def __init__(self, model_path):
        self.tflite_model = Path(model_path)
        self.id = self.tflite_model.stem
        self._model = TFLiteModel.from_file(self.tflite_model)
        shape = self._model.input_shape
        if len(shape) != 3 or shape[0] != 1 or not 1 <= shape[1] <= _MAX_EMBEDDINGS or shape[2] != 96:
            raise ValueError("Wake classifier must consume [1, windows, 96] float embeddings")
        self.input_windows, self._closed = shape[1], False
        self.reset()

    @classmethod
    def from_model(cls, model_path):
        return cls(model_path)

    def process_streaming(self, rows):
        if self._closed:
            return
        if not isinstance(rows, (list, tuple)) or len(rows) > _MAX_EMBEDDINGS:
            raise ValueError("Wake embedding batch exceeds ten seconds")
        for row in rows:
            if len(row) != 96 or any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in row):
                raise ValueError("Wake embeddings must contain 96 finite values")
        for row in rows:
            self._embeddings.append(tuple(array("f", row)))
        self._new_embeddings = min(_MAX_EMBEDDINGS, self._new_embeddings + len(rows))
        while self._new_embeddings >= self.input_windows:
            start = len(self._embeddings) - self._new_embeddings
            window = list(self._embeddings)[start:start + self.input_windows]
            self._new_embeddings -= 1
            result = self._model.invoke((value for row in window for value in row))
            if len(result.values) != 1 or not math.isfinite(result.values[0]):
                raise ValueError("Wake classifier must produce one finite score")
            yield result.values[0]

    def reset(self):
        self._embeddings = deque([(0.0,) * 96] * _MAX_EMBEDDINGS, maxlen=_MAX_EMBEDDINGS)
        self._new_embeddings = 0

    def close(self):
        self.reset()
        self._closed, self._model = True, None


class OpenWakeWordFeatures:
    def __init__(self, melspectrogram_model, embedding_model):
        self._mel = TFLiteModel.from_file(melspectrogram_model)
        self._embedding = TFLiteModel.from_file(embedding_model)
        if self._embedding.input_shape != (1, 76, 32, 1):
            raise ValueError("Wake embedding model must consume [1, 76, 32, 1]")
        self._closed = False
        self.reset()

    @classmethod
    def from_builtin(cls, models_dir=_MODELS):
        models_dir = Path(models_dir)
        return cls(models_dir / "melspectrogram.tflite", models_dir / "embedding_model.tflite")

    def process_streaming(self, audio_chunk):
        if self._closed:
            return
        if not isinstance(audio_chunk, bytes) or len(audio_chunk) % 2 or len(audio_chunk) > 2 * _MAX_AUDIO:
            raise ValueError("Wake audio must be at most ten seconds of mono PCM16 bytes")
        if not audio_chunk:
            return
        self._audio.extend(audio_chunk)
        if len(self._audio) > _MAX_AUDIO * 2:
            del self._audio[:-_MAX_AUDIO * 2]
        offset = 0
        while len(self._audio) - offset >= 1760 * 2:
            samples = struct.unpack_from("<1760h", self._audio, offset)
            offset += 1280 * 2
            mels = self._mel.invoke(samples, (1, 1760))
            if len(mels.values) % 32:
                raise ValueError("Wake mel model must produce rows of 32 float values")
            scaled = array("f", (value / 10 for value in mels.values))
            transformed = tuple(array("f", (value + 2 for value in scaled)))
            self._mels.extend(transformed[index:index + 32] for index in range(0, len(transformed), 32))
            while len(self._mels) >= 76:
                rows = list(self._mels)[:76]
                embedding = self._embedding.invoke((value for row in rows for value in row),
                                                    temporal_shift=8)
                for _ in range(8):
                    self._mels.popleft()
                if len(embedding.values) % 96:
                    raise ValueError("Wake embedding model must produce rows of 96 float values")
                yield tuple(embedding.values[index:index + 96] for index in range(0, len(embedding.values), 96))
        del self._audio[:offset]

    def reset(self):
        # The original SDK initializes a ten-second zero array but admits eight
        # seconds of it before the first real frame; the queue stores that
        # identical pending prefix directly.
        self._audio = bytearray(8 * 16000 * 2)
        self._mels = deque(maxlen=_MAX_MELS)

    def close(self):
        self._audio.clear()
        self._mels.clear()
        self._closed, self._mel, self._embedding = True, None, None
