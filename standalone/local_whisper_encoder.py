"""Stdlib Whisper convolutional and Transformer encoder over original model weights."""
import math

from .local_whisper_features import MelFeatures
from .local_whisper_tensor import EncodedAudio, WhisperMath, f32, row


class WhisperEncoder:
    def __init__(self, weights, config=None, compute_type="float32", *, int8_saturating_pairs=False):
        self.math = WhisperMath(weights, config, compute_type, int8_saturating_pairs=int8_saturating_pairs)
        self.weights = weights
        self.layers = self.math.layer_count("encoder")
        self.heads = int(self.math.attribute("encoder/num_heads", 8))
        first = weights["encoder/conv1/weight"]
        positions = weights["encoder/position_encodings/encodings"]
        if len(first.shape) != 3 or len(positions.shape) != 2 or first.shape[0] != positions.shape[1]:
            raise ValueError("Invalid Whisper encoder convolution/position shapes")
        self.n_mels, self.width, self.max_frames = first.shape[1], first.shape[0], positions.shape[0] * 2

    def encode(self, features):
        source = features.rows if isinstance(features, MelFeatures) else features
        if not isinstance(source, (tuple, list)) or len(source) != self.n_mels:
            raise ValueError("Whisper encoder features have the wrong mel count")
        length = len(source[0])
        if not 1 <= length <= self.max_frames or any(len(value) != length for value in source):
            raise ValueError("Whisper encoder features exceed their uniform time bound")
        channels = tuple(row(value) for value in source)
        if any(not math.isfinite(value) for channel in channels for value in channel):
            raise ValueError("Whisper encoder features must be finite")
        channels = self.math.conv(channels, "encoder/conv1", 1)
        channels = self.math.conv(channels, "encoder/conv2", 2)
        positions = self.weights["encoder/position_encodings/encodings"]
        hidden = tuple(tuple(f32(value + offset) for value, offset in zip(column, positions.row(index), strict=True))
                       for index, column in enumerate(zip(*channels, strict=True)))
        for layer in range(self.layers):
            scope = "encoder/layer_" + str(layer)
            hidden, _ = self.math.attention(hidden, scope + "/self_attention", self.heads)
            hidden = self.math.feedforward(hidden, scope + "/ffn")
        return EncodedAudio(self.math.normalize(hidden, "encoder/layer_norm"))

    def close(self):
        self.math.close()
