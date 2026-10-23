"""AAC-LC scalar spectral transform, following pinned FFmpeg's AVTX (LGPL)."""
import cmath
from functools import lru_cache
import math
import struct


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


@lru_cache(maxsize=32)
def _twiddles(length):
    return tuple(cmath.exp(2j * math.pi * i / length) for i in range(length))


@lru_cache(maxsize=16)
def _bit_reverse(length):
    bits = length.bit_length() - 1
    return tuple(int(format(i, '0' + str(bits) + 'b')[::-1], 2) for i in range(length))


def _fft(values):
    """Unnormalized inverse complex FFT, with radix 2/3/5 factorizations."""
    length = len(values)
    if length == 1:
        return list(values)
    if not length & (length - 1):
        output = [values[i] for i in _bit_reverse(length)]
        width = 2
        while width <= length:
            half = width // 2
            roots = _twiddles(width)
            for start in range(0, length, width):
                for i in range(half):
                    a = output[start + i]
                    b = output[start + i + half] * roots[i]
                    output[start + i], output[start + i + half] = a + b, a - b
            width *= 2
        return output
    radix = 3 if length % 3 == 0 else 5 if length % 5 == 0 else None
    if radix is None:
        raise ValueError('Unsupported AAC transform factor')
    size = length // radix
    transformed = [_fft(values[i::radix]) for i in range(radix)]
    roots = _twiddles(length)
    result = [0j] * length
    for k in range(size):
        for q in range(radix):
            j = k + q * size
            result[j] = sum(transformed[r][k] * roots[(r * j) % length] for r in range(radix))
    return result


@lru_cache(maxsize=8)
def _mdct_rotations(length, scale):
    amplitude = math.sqrt(abs(scale))
    theta = (length // 2 if scale < 0 else 0) + .125
    return tuple(complex(f32(math.cos(math.pi * (i + theta) / length) * amplitude),
                         f32(math.sin(math.pi * (i + theta) / length) * amplitude))
                 for i in range(length // 2))


def imdct(values, *, scale=None):
    """Return the N nonredundant inverse-MDCT samples, in native AVTX order."""
    length = len(values)
    if length < 4 or length % 4:
        raise ValueError('AAC inverse MDCT length must be a positive multiple of four')
    values = struct.unpack('<' + str(length) + 'f', struct.pack('<' + str(length) + 'f', *values))
    scale = f32(1 / (length * 32768) if scale is None else scale)
    rotations = _mdct_rotations(length, scale)
    transformed = _fft([complex(values[length - 1 - 2 * i], values[2 * i]) * rotations[i]
                        for i in range(length // 2)])
    output = [0.] * length
    for i in range(length // 4):
        upper, lower = length // 4 + i, length // 4 - i - 1
        a, b = transformed[lower], transformed[upper]
        a = complex(a.imag, a.real) * complex(rotations[lower].imag, rotations[lower].real)
        b = complex(b.imag, b.real) * complex(rotations[upper].imag, rotations[upper].real)
        output[2 * lower], output[2 * upper + 1] = a.real, a.imag
        output[2 * upper], output[2 * lower + 1] = b.real, b.imag
    return list(struct.unpack('<' + str(length) + 'f', struct.pack('<' + str(length) + 'f', *output)))


def _i0(value):
    term = total = 1.
    square = value * value / 4
    index = 1
    while True:
        term *= square / (index * index)
        total += term
        if term < total * 1e-18:
            return total
        index += 1


@lru_cache(maxsize=8)
def _window(length, shape):
    if not shape:
        return tuple(f32(math.sin(f32((i + .5) * (math.pi / (2 * length))))) for i in range(length))
    alpha = 6 if length in (120, 128) else 4
    ratio = alpha * math.pi / length
    squared = 4 * ratio * ratio
    temp = [_i0(math.sqrt(i * (length - i) * squared)) for i in range(length // 2 + 1)]
    scale = 1 / (sum(value * (2 if 0 < i < length // 2 else 1) for i, value in enumerate(temp)) + 1)
    total, window = 0., []
    for i in range(length):
        total += temp[i if i <= length // 2 else length - i]
        window.append(f32(math.sqrt(total * scale)))
    return tuple(window)


def _multiply_window(first, second, window, length):
    output = [0.] * (2 * length)
    for i in range(length):
        j = 2 * length - i - 1
        a, b = first[i], second[length - i - 1]
        w1, w2 = window[i], window[j]
        output[i] = f32(a * w2 - b * w1)
        output[j] = f32(a * w1 + b * w2)
    return output


def windowing(values, previous, sequence, previous_sequence, shape, previous_shape, frame_length):
    """Stateful AAC-LC 1024/960 long and eight-short window overlap."""
    short = frame_length // 8
    half_short, half_long = short // 2, frame_length // 2
    offset = half_long - half_short
    swindow = _window(short, shape)
    lwindow_previous = _window(frame_length, previous_shape)
    swindow_previous = _window(short, previous_shape)
    if sequence == 2:
        transformed = [value for i in range(8) for value in imdct(values[i * 128:i * 128 + short])]
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
