"""The faster-whisper 1.2.1 log-mel frontend using Python's standard library.

Input PCM floats, periodic Hann window, centered reflected STFT, Slaney mel
filters and float32 normalization follow the original FeatureExtractor (MIT).
This module does not import NumPy or change sample rates or model settings.
"""

from array import array
from dataclasses import dataclass
import itertools
import math
import struct


_sumprod = getattr(math, "sumprod", lambda a, b: sum(x * y for x, y in zip(a, b, strict=True)))
_F32 = struct.Struct("<f")


def _f32(value):
    return _F32.unpack(_F32.pack(value))[0]


@dataclass(frozen=True)
class MelFeatures:
    rows: tuple[tuple[float, ...], ...]

    @property
    def shape(self):
        return len(self.rows), len(self.rows[0]) if self.rows else 0

    def pad_or_trim(self, length=3000, *, start=0):
        if not isinstance(length, int) or isinstance(length, bool) or not 0 < length <= 60000:
            raise ValueError("Invalid Whisper feature window length")
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ValueError("Invalid Whisper feature window offset")
        return MelFeatures(tuple(row[start:start + length] + (0.0,) * max(0, length - max(0, min(len(row) - start, length)))
                                 for row in self.rows))


class FeatureExtractor:
    def __init__(self, feature_size=80, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400):
        for value, lower, upper in ((feature_size, 1, 256), (sampling_rate, 1, 192000),
                                    (hop_length, 1, 4096), (chunk_length, 1, 600), (n_fft, 2, 4096)):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError("Invalid Whisper feature extractor configuration")
        self.feature_size, self.sampling_rate, self.hop_length = feature_size, sampling_rate, hop_length
        self.n_fft, self.chunk_length = n_fft, chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.time_per_frame = hop_length / sampling_rate
        self._window = tuple(array("f", (0.5 - 0.5 * math.cos(2 * math.pi * index / n_fft)
                                        for index in range(n_fft))))
        self._filters = self.get_mel_filters(sampling_rate, n_fft, feature_size)
        # Direct real DFT dot products run in C through math.sumprod on the
        # Hermes interpreter; coefficients are cached once for every frame.
        self._cos = tuple(tuple(math.cos(-2 * math.pi * frequency * index / n_fft) for index in range(n_fft))
                          for frequency in range(n_fft // 2 + 1))
        self._sin = tuple(tuple(math.sin(-2 * math.pi * frequency * index / n_fft) for index in range(n_fft))
                          for frequency in range(n_fft // 2 + 1))

    @staticmethod
    def get_mel_filters(sr, n_fft, n_mels=128):
        mels = [45.245640471924965 * index / (n_mels + 1) for index in range(n_mels + 2)]
        freqs = [(200.0 / 3) * mel if mel < 15 else 1000 * math.exp(math.log(6.4) / 27 * (mel - 15))
                 for mel in mels]
        rows = []
        for mel in range(n_mels):
            low, center, high = freqs[mel:mel + 3]
            weights = []
            indices = []
            for index in range(n_fft // 2 + 1):
                frequency = index * sr / n_fft
                weight = _f32(max(0.0, min((frequency - low) / (center - low),
                                           (high - frequency) / (high - center))) * 2 / (high - low))
                if weight:
                    indices.append(index)
                    weights.append(weight)
            rows.append((tuple(indices), tuple(weights)))
        return tuple(rows)

    def __call__(self, waveform, padding=160, chunk_length=None):
        if chunk_length is not None:
            if isinstance(chunk_length, bool) or not isinstance(chunk_length, int) or not 1 <= chunk_length <= 600:
                raise ValueError("Invalid Whisper audio chunk length")
            self.n_samples = chunk_length * self.sampling_rate
            self.nb_max_frames = self.n_samples // self.hop_length
        if isinstance(padding, bool) or not isinstance(padding, int) or not 0 <= padding <= self.sampling_rate * 600:
            raise ValueError("Invalid Whisper audio padding")
        limit = self.sampling_rate * 3600
        try:
            samples = tuple(array("f", itertools.islice(waveform, limit + 1)))
        except (TypeError, OverflowError, ValueError) as exc:
            raise ValueError("Whisper waveform must contain finite PCM floats") from exc
        if len(samples) > limit or any(not math.isfinite(value) for value in samples):
            raise ValueError("Whisper waveform exceeds its finite one-hour bound")
        samples += (0.0,) * padding
        if not samples:
            raise ValueError("Whisper reflected STFT requires a nonempty waveform")
        half = self.n_fft // 2
        period = max(1, 2 * (len(samples) - 1))

        def reflected(index):
            index %= period
            return samples[index if index < len(samples) else period - index]

        padded = tuple(reflected(index) for index in range(-half, len(samples) + half))
        frames = max(0, (len(padded) - self.n_fft) // self.hop_length)
        if frames == 0:
            raise ValueError("Whisper waveform does not contain a complete feature frame")
        rows = [[] for _ in self._filters]
        for index in range(frames):
            start = index * self.hop_length
            window = tuple(array("f", (value * coefficient for value, coefficient in
                                        zip(padded[start:start + self.n_fft], self._window, strict=True))))
            magnitudes = []
            for cosines, sines in zip(self._cos, self._sin, strict=True):
                real, imaginary = _f32(_sumprod(window, cosines)), _f32(_sumprod(window, sines))
                magnitude = _f32(math.hypot(real, imaginary))
                magnitudes.append(_f32(magnitude * magnitude))
            for row, (indices, weights) in zip(rows, self._filters, strict=True):
                mel = _f32(_sumprod(tuple(magnitudes[index] for index in indices), weights))
                row.append(_f32(math.log10(max(mel, _f32(1e-10)))))
        floor = _f32(max(max(row) for row in rows) - 8.0)
        return MelFeatures(tuple(tuple(_f32(_f32(max(value, floor) + 4.0) / 4.0) for value in row)
                                 for row in rows))
