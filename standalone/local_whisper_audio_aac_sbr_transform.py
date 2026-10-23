"""Portable float32 AVTX split-radix inverse MDCT, pinned FFmpeg 8.1.2 LGPL.

Every multiply, sum, and butterfly follows the original generic C operation
order. HE's predictor is ill-conditioned, so a double-precision FFT followed
by one float32 rounding is not an equivalent numerical profile here.
"""
from functools import lru_cache
import math
from standalone.local_whisper_audio_aac_dsp import f32 as f, _mdct_rotations
from .local_whisper_audio_aac_sbr_transform_tables import WINDOWS


@lru_cache(maxsize=10)
def _cosines(length):
    frequency = 2 * math.pi / length
    return tuple(f(math.cos(i * frequency)) for i in range(length // 4)) + (0.,)


@lru_cache(maxsize=10)
def _permutation(length):
    def permute(i, size):
        size >>= 1
        if size <= 1:
            return i & 1
        if not i & size:
            return permute(i, size) * 2
        size >>= 1
        return permute(i, size) * 4 + 1 - 2 * (bool(not i & size) ^ True)
    return tuple(-permute(i, length) & (length - 1) for i in range(length))


def _butterfly(a0, a1, t1, t2, t5, t6):
    t3, t5 = f(t5 - t1), f(t5 + t1)
    t4, t6 = f(t2 - t6), f(t2 + t6)
    return (complex(f(a0.real + t5), f(a0.imag + t6)),
            complex(f(a1.real + t4), f(a1.imag + t3)),
            complex(f(a0.real - t5), f(a0.imag - t6)),
            complex(f(a1.real - t4), f(a1.imag - t3)))


def _transform(a0, a1, a2, a3, re, im):
    t1, t2 = f(f(a2.real * re) - f(a2.imag * -im)), f(f(a2.real * -im) + f(a2.imag * re))
    t5, t6 = f(f(a3.real * re) - f(a3.imag * im)), f(f(a3.real * im) + f(a3.imag * re))
    return _butterfly(a0, a1, t1, t2, t5, t6)


def _fft(values):
    length = len(values)
    if length == 2:
        a, b = values
        return [complex(f(a.real + b.real), f(a.imag + b.imag)), complex(f(a.real - b.real), f(a.imag - b.imag))]
    if length == 4:
        a, b, c, d = values
        t3, t1 = f(a.real - b.real), f(a.real + b.real)
        t8, t6 = f(d.real - c.real), f(d.real + c.real)
        t4, t2 = f(a.imag - b.imag), f(a.imag + b.imag)
        t7, t5 = f(c.imag - d.imag), f(c.imag + d.imag)
        return [complex(f(t1 + t6), f(t2 + t5)), complex(f(t3 + t7), f(t4 + t8)),
                complex(f(t1 - t6), f(t2 - t5)), complex(f(t3 - t7), f(t4 - t8))]
    if length == 8:
        result = _fft(values[:4]) + [0j] * 4
        a, b, c, d = values[4:]
        result[5] = complex(f(a.real - b.real), f(a.imag - b.imag))
        result[7] = complex(f(c.real - d.real), f(c.imag - d.imag))
        result[0], result[2], result[4], result[6] = _butterfly(
            result[0], result[2], f(a.real + b.real), f(a.imag + b.imag), f(c.real + d.real), f(c.imag + d.imag))
        result[1], result[3], result[5], result[7] = _transform(
            result[1], result[3], result[5], result[7], _cosines(8)[1], _cosines(8)[1])
        return result
    quarter = length // 4
    result = _fft(values[:2 * quarter]) + _fft(values[2 * quarter:3 * quarter]) + _fft(values[3 * quarter:])
    cosines = _cosines(length)
    for i in range(quarter):
        indices = (i, quarter + i, 2 * quarter + i, 3 * quarter + i)
        a0, a1, a2, a3 = (result[k] for k in indices)
        if length == 16 and not i:
            values4 = _butterfly(a0, a1, a2.real, a2.imag, a3.real, a3.imag)
        else:
            values4 = _transform(a0, a1, a2, a3, cosines[i], cosines[quarter - i])
        for index, value in zip(indices, values4):
            result[index] = value
    return result


def imdct(values, *, scale=None):
    length = len(values)
    if length < 4 or length & (length - 1):
        raise ValueError('HE AAC generic MDCT needs a power-of-two length')
    values = list(map(f, values))
    scale = f(1 / (length * 32768) if scale is None else scale)
    rotations = _mdct_rotations(length, scale)
    transformed = []
    for k in _permutation(length // 2):
        a, b, rotation = values[length - 1 - 2 * k], values[2 * k], rotations[k]
        transformed.append(complex(f(f(a * rotation.real) - f(b * rotation.imag)),
                                   f(f(a * rotation.imag) + f(b * rotation.real))))
    transformed = _fft(transformed)
    output = [0.] * length
    for i in range(length // 4):
        upper, lower = length // 4 + i, length // 4 - i - 1
        a, b = transformed[lower], transformed[upper]
        ar, ai, br, bi = rotations[lower].imag, rotations[lower].real, rotations[upper].imag, rotations[upper].real
        output[2 * lower] = f(f(a.imag * ar) - f(a.real * ai))
        output[2 * upper + 1] = f(f(a.imag * ai) + f(a.real * ar))
        output[2 * upper] = f(f(b.imag * br) - f(b.real * bi))
        output[2 * lower + 1] = f(f(b.imag * bi) + f(b.real * br))
    return output


def _multiply_window(first, second, window, length):
    output = [0.] * (2 * length)
    for i in range(length):
        j = 2 * length - i - 1
        a, b, w1, w2 = first[i], second[length - i - 1], window[i], window[j]
        output[i] = f(f(a * w2) - f(b * w1))
        output[j] = f(f(a * w1) + f(b * w2))
    return output


def windowing(values, previous, sequence, previous_sequence, shape, previous_shape, frame_length):
    if frame_length != 1024:
        raise ValueError('The original generic HE SBR decoder uses 1024-sample frames')
    short, half_short, half_long, offset = 128, 64, 512, 448
    swindow = WINDOWS[short, shape]
    lwindow_previous = WINDOWS[frame_length, previous_shape]
    swindow_previous = WINDOWS[short, previous_shape]
    if sequence == 2:
        transformed = [value for i in range(8) for value in imdct(values[i * short:(i + 1) * short])]
    else:
        transformed = imdct(values[:frame_length])
    if previous_sequence in (0, 3) and sequence in (0, 1):
        output = _multiply_window(previous, transformed, lwindow_previous, half_long)
    else:
        output = previous[:offset] + [0.] * (frame_length - offset)
        if sequence == 2:
            for i in range(4):
                first = previous[offset:] if i == 0 else transformed[(i - 1) * short + half_short:]
                window = swindow_previous if i == 0 else swindow
                piece = _multiply_window(first, transformed[i * short:], window, half_short)
                output[offset + i * short:offset + (i + 1) * short] = piece
            temp = _multiply_window(transformed[3 * short + half_short:], transformed[4 * short:], swindow, half_short)
            output[offset + 4 * short:] = temp[:half_short]
        else:
            output[offset:offset + short] = _multiply_window(previous[offset:], transformed, swindow_previous, half_short)
            output[offset + short:] = transformed[half_short:half_short + offset]
    if sequence == 2:
        new = temp[half_short:]
        for i in range(3):
            new.extend(_multiply_window(transformed[(4 + i) * short + half_short:], transformed[(5 + i) * short:], swindow, half_short))
        new.extend(transformed[7 * short + half_short:])
    elif sequence == 1:
        new = transformed[half_long:half_long + offset] + transformed[7 * short + half_short:]
    else:
        new = transformed[half_long:]
    previous[:] = new
    return output
