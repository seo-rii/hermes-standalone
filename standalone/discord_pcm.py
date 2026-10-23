"""Explicit PCM operations used by the original Discord mixer and recorder.

Mixing retains float32 rounding at every multiply and addition. PCM WAV
conversion uses the reviewed integer SILK resampler instead of a subprocess.
Ambient synthesis keeps the original spectrum and seed; its Gaussian stream
uses stdlib Random rather than NumPy's PCG64/Ziggurat stream.
"""

from array import array
import math
import random
import struct
import wave

from standalone.audio_samples import MAX_BYTES, PCM16, from_samples


def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_frame(data, gain):
    pcm = PCM16(data, 2)
    gain = _float32(gain)
    if not math.isfinite(gain):
        raise ValueError("Discord mixer gain must be finite")
    return [_float32(value * gain) for value in pcm.values()]


def add_float_frames(first, second):
    if len(first) != len(second):
        raise ValueError("Discord mixer frames must have equal lengths")
    return [_float32(a + b) for a, b in zip(first, second)]


def float_frame_bytes(values):
    return from_samples(values, 2).tobytes()


def _convert_pcm(data, channels, out_channels, states):
    pcm = PCM16(data, channels)
    values = list(pcm.values())
    if channels not in (1, 2) or out_channels not in (1, 2):
        raise ValueError("Discord WAV conversion supports mono and stereo PCM")
    if out_channels == 1:
        streams = [[int(sum(values[position:position + channels]) / channels)
                    for position in range(0, len(values), channels)]]
    elif channels == 1:
        streams = [values, values]
    else:
        streams = [values[::2], values[1::2]]
    streams = [state.process(stream) for state, stream in zip(states, streams)]
    return from_samples((sample for frame in zip(*streams) for sample in frame), out_channels).tobytes()


def write_stt_wav(data, output_path, rate=48000, channels=2):
    from standalone.silk_resample import Resampler

    pcm = PCM16(data, channels)
    if channels not in (1, 2):
        raise ValueError("Discord recordings support mono and stereo PCM")
    state = [Resampler(rate, 16000)]
    with wave.open(str(output_path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        block = rate * channels * 2
        for offset in range(0, pcm.nbytes, block):
            output.writeframesraw(_convert_pcm(data[offset:offset + block], channels, 1, state))


def decode_wav(path):
    from standalone.silk_resample import Resampler

    output = bytearray()
    with wave.open(str(path), "rb") as source:
        rate, channels, width = source.getframerate(), source.getnchannels(), source.getsampwidth()
        if channels not in (1, 2) or width not in (1, 2, 3, 4) or source.getcomptype() != "NONE":
            raise ValueError("Discord standalone WAV input requires mono/stereo integer PCM")
        if source.getnframes() * channels * width > MAX_BYTES:
            raise ValueError("Discord WAV input exceeds the PCM byte limit")
        states = [Resampler(rate, 48000), Resampler(rate, 48000)]
        while data := source.readframes(rate):
            if len(data) % (width * channels):
                raise ValueError("Truncated Discord WAV sample frame")
            if width == 1:
                pcm = from_samples(((value - 128) << 8 for value in data), channels).tobytes()
            elif width == 2:
                pcm = data
            else:
                pcm = from_samples((int.from_bytes(data[index:index + width], "little", signed=True)
                                    >> (width * 8 - 16) for index in range(0, len(data), width)), channels).tobytes()
            frame = _convert_pcm(pcm, channels, 2, states)
            if len(frame) > MAX_BYTES - len(output):
                raise ValueError("Discord WAV output exceeds the PCM byte limit")
            output.extend(frame)
    return bytes(output)


def synth_ambient_pcm(seconds=4.0):
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Discord ambient duration must be positive and finite")
    count = int(48000 * seconds)
    if count < 64 or count * 4 > MAX_BYTES:
        raise ValueError("Discord ambient duration exceeds the PCM limits")
    f1, f2, tremolo = (max(1, round(frequency * seconds)) / seconds for frequency in (110.0, 110.5, .5))
    generator, prefix = random.Random(7), array("d", [0.0])
    for _ in range(count):
        prefix.append(prefix[-1] + generator.gauss(0.0, 1.0))
    signal = array("d")
    peak = 0.0
    for index in range(count):
        time = index / 48000
        pad = .55 * math.sin(2 * math.pi * f1 * time) + .45 * math.sin(2 * math.pi * f2 * time)
        modulation = .6 + .4 * (.5 * (1 + math.sin(2 * math.pi * tremolo * time)))
        noise = (prefix[min(count, index + 32)] - prefix[max(0, index - 32)]) / 64
        value = pad * modulation + .08 * noise
        signal.append(value)
        peak = max(peak, abs(value))
    peak = peak or 1.0
    return from_samples((value / peak * .5 * 32767 for value in signal for _ in range(2)), 2).tobytes()
