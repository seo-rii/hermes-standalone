"""Bounded signed 16-bit PCM, levels, fades and voice cues using stdlib math."""

from __future__ import annotations

from array import array
import math
import struct
import sys


MAX_BYTES = 64 * 1024 * 1024


def format_size(rate, channels=1):
    if type(rate) is not int or not 8000 <= rate <= 192000 or type(channels) is not int or not 1 <= channels <= 8:
        raise ValueError("Audio requires 8–192 kHz and one to eight PCM channels")
    return rate, channels


class PCM16:
    """Immutable interleaved little-endian PCM frames; no array-library facade."""

    def __init__(self, data=b"", channels=1):
        if type(channels) is not int or not 1 <= channels <= 8:
            raise ValueError("PCM channel count must be between one and eight")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("PCM16 requires encoded sample bytes")
        length = data.nbytes if isinstance(data, memoryview) else len(data)
        if length > MAX_BYTES or length % (channels * 2):
            raise ValueError("PCM data is oversized or does not contain complete frames")
        self._data, self.channels = bytes(data), channels

    def __len__(self):
        return len(self._data) // (2 * self.channels)

    @property
    def nbytes(self):
        return len(self._data)

    @property
    def shape(self):
        return len(self), self.channels

    def copy(self):
        return self

    def tobytes(self):
        return self._data

    def values(self):
        return (item[0] for item in struct.iter_unpack("<h", self._data))


def from_samples(values, channels=1):
    result = array("h")
    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("PCM samples must be finite numbers")
        if len(result) * 2 >= MAX_BYTES:
            raise ValueError("PCM exceeds the sample byte limit")
        result.append(max(-32768, min(32767, int(value))))
    if sys.byteorder != "little":
        result.byteswap()
    return PCM16(result.tobytes(), channels)


def as_pcm(data, channels=None):
    if isinstance(data, PCM16):
        if channels is not None and data.channels != channels:
            raise ValueError("PCM channel counts disagree")
        return data
    if isinstance(data, (bytes, bytearray, memoryview)):
        return PCM16(data, channels or 1)
    if hasattr(data, "tobytes"):
        shape = getattr(data, "shape", ())
        inferred = shape[1] if len(shape) == 2 else 1
        return PCM16(data.tobytes(), channels or inferred)
    return from_samples(data, channels or 1)


def concatenate(blocks, channels=None):
    result, selected = bytearray(), channels
    for block in blocks:
        block = as_pcm(block, selected)
        selected = block.channels
        if block.nbytes > MAX_BYTES - len(result):
            raise ValueError("Combined PCM exceeds the recording byte limit")
        result.extend(block.tobytes())
    return PCM16(result, selected or 1)


def rms(data):
    pcm = as_pcm(data)
    count = pcm.nbytes // 2
    return math.sqrt(sum(value * value for value in pcm.values()) / count) if count else 0.0


def percentile(values, percentage):
    values = sorted(float(value) for value in values)
    if not values or any(not math.isfinite(value) for value in values) or not 0 <= percentage <= 100:
        raise ValueError("Percentile requires finite values and a percentage in 0–100")
    position = (len(values) - 1) * (percentage / 100.0)
    left = int(position)
    fraction = position - left
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * fraction


def silence(frames, channels=1):
    format_size(16000, channels)
    if type(frames) is not int or frames < 0 or frames * channels * 2 > MAX_BYTES:
        raise ValueError("Silence exceeds the PCM byte limit")
    return PCM16(b"\0" * (frames * channels * 2), channels)


def beep(frequency=880, duration=.12, count=1, *, rate=16000, volume=.3):
    format_size(rate)
    if (not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 <= duration <= 10
            or not isinstance(frequency, (int, float)) or not math.isfinite(frequency) or not 0 < frequency < rate / 2
            or type(count) is not int or not 1 <= count <= 100 or not 0 <= volume <= 1):
        raise ValueError("Invalid beep duration, frequency, count or volume")
    count_per_beep = int(rate * duration)
    fade = min(int(rate * .01), count_per_beep // 4)
    def value(index):
        level = 1.0
        if fade > 0:
            if index < fade:
                level *= index / max(1, fade - 1)
            if index >= count_per_beep - fade:
                level *= (count_per_beep - 1 - index) / max(1, fade - 1)
        return math.sin(2 * math.pi * frequency * index * duration / count_per_beep) * level * volume * 32767
    tone, gap = from_samples(value(index) for index in range(count_per_beep)), silence(int(rate * .06))
    return concatenate(block for index in range(count) for block in ((tone, gap) if index < count - 1 else (tone,)))


def thinking_blip(frequency, *, rate=16000, volume=.15):
    format_size(rate)
    if not 0 < frequency < rate / 2 or not 0 <= volume <= 1:
        raise ValueError("Invalid thinking cue frequency or volume")
    duration, phase, values = .16, 0.0, []
    count, attack = int(rate * .16), int(rate * .02)
    for index in range(count):
        phase += frequency * (1.0 - .28 * index / (count - 1))
        angle = 2 * math.pi * phase / rate
        envelope = min(1.0, index / max(1, attack - 1)) * math.exp(-index * duration / count * 14.0)
        values.append((.8 * math.sin(angle) + .2 * math.sin(angle / 2)) * envelope * volume * 32767)
    return from_samples(values)


def warmup(data, rate):
    pcm = as_pcm(data)
    format_size(rate, pcm.channels)
    fade = min(len(pcm), int(.1 * rate))
    values = (value * (min(index // pcm.channels, fade - 1) / max(1, fade - 1)) if index // pcm.channels < fade else value
              for index, value in enumerate(pcm.values()))
    return concatenate((silence(int(.1 * rate), pcm.channels), from_samples(values, pcm.channels),
                        silence(int(.05 * rate), pcm.channels)))
