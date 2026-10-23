"""Portable float32 elementary operations from reviewed CTranslate2 4.8.1 CPU kernels.

These are numerical algorithms, not instruction set emulation or native calls.
The exponential follows the pinned avx_mathfun.h (MIT/zlib; see licenses/avx_mathfun.txt)
and erf follows CTranslate2 vec_erf. Each vector operation rounds to float32.
"""
import math
import struct

_FLOAT = struct.Struct("<f")
_UINT = struct.Struct("<I")
_fma = getattr(math, "fma", lambda a, b, c: a * b + c)


def f32(value):
    try:
        return _FLOAT.unpack(_FLOAT.pack(value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def fma32(a, b, c):
    return f32(_fma(a, b, c))


_EXP_CONSTANTS = tuple(map(f32, (1.44269504088896341, 0.693359375, -2.12194440e-4,
                                 1.9875691500e-4, 1.3981999507e-3, 8.3334519073e-3,
                                 4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1)))
_ERF_CONSTANTS = tuple(map(f32, (0.3275911, 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)))


def exp(value):
    value = max(-88.3762626647949, min(88.3762626647949, value))
    log2, c1, c2, *coefficients = _EXP_CONSTANTS
    exponent = math.floor(fma32(value, log2, 0.5))
    value = fma32(-exponent, c1, value)
    value = fma32(-exponent, c2, value)
    square = f32(value * value)
    result = coefficients[0]
    for coefficient in coefficients[1:]:
        result = fma32(result, value, coefficient)
    result = f32(fma32(result, square, value) + 1.0)
    power = _FLOAT.unpack(_UINT.pack((exponent + 127) << 23))[0]
    return f32(result * power)


def erf(value):
    p, p1, p2, p3, p4, p5 = _ERF_CONSTANTS
    inverse = f32(1.0 / fma32(p, abs(value), 1.0))
    result = fma32(p5, inverse, p4)
    for coefficient in (p3, p2, p1):
        result = fma32(result, inverse, coefficient)
    exponential = exp(-f32(value * value))
    result = fma32(f32(-exponential * inverse), result, 1.0)
    return math.copysign(result, value)


def gelu(value):
    intermediate = erf(f32(0.7071067690849304 * value))
    return f32(f32(value * f32(1.0 + intermediate)) * 0.5)


def total(values):
    if len(values) <= 8:
        result = 0.0
        for value in values:
            result = f32(result + value)
        return result
    lanes = [0.0] * 8
    complete = len(values) - len(values) % 8
    for start in range(0, complete, 8):
        lanes = [f32(first + second) for first, second in zip(lanes, values[start:start + 8], strict=True)]
    a, b, c, d = (f32(lanes[index] + lanes[index + 4]) for index in range(4))
    result = f32(f32(a + c) + f32(b + d))
    for value in values[complete:]:
        result = f32(result + value)
    return result


def softmax(values):
    maximum = max(values)
    exponentials = tuple(exp(f32(value - maximum)) for value in values)
    inverse = f32(1.0 / total(exponentials))
    return tuple(f32(value * inverse) for value in exponentials)


def normalize(source, gamma, beta, epsilon):
    """CT2 AVX2 reduction and fused affine rounding from the reviewed kernel."""
    width = len(source)
    count = width - width % 8
    sums, squares = [0.0] * 8, [0.0] * 8
    for start in range(0, count, 8):
        for lane in range(8):
            value = source[start + lane]
            sums[lane] = f32(sums[lane] + value)
            squares[lane] = f32(squares[lane] + f32(value * value))
    sums = [f32(sums[i] + sums[i + 4]) for i in range(4)]
    squares = [f32(squares[i] + squares[i + 4]) for i in range(4)]
    if width - count >= 4:
        for lane in range(4):
            value = source[count + lane]
            sums[lane] = f32(sums[lane] + value)
            squares[lane] = fma32(value, value, squares[lane])
        count += 4
    total = f32(f32(sums[0] + sums[2]) + f32(sums[1] + sums[3]))
    sum_squares = f32(f32(squares[0] + squares[2]) + f32(squares[1] + squares[3]))
    for value in source[count:]:
        total = f32(total + value)
        sum_squares = fma32(value, value, sum_squares)
    inverse_depth = f32(1.0 / width)
    mean = f32(total * inverse_depth)
    variance = max(fma32(sum_squares, inverse_depth, -f32(mean * mean)), 0.0)
    denominator = f32(math.sqrt(f32(variance + epsilon)))
    inverse = f32(1.0 / denominator) if denominator else math.inf
    return tuple(fma32(f32(value - mean), f32(inverse * gain), offset)
                 for value, gain, offset in zip(source, gamma, beta, strict=True))

