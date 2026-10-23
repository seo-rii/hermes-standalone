"""CELT pyramid codebook and inverse spreading, adapted from Opus 1.3.1.

Copyright (c) 2007–2008 CSIRO, (c) 2007–2011 Xiph.Org Foundation,
Timothy B. Terriberry and Jean-Marc Valin. Original notices: licenses/opus*.txt.
"""
from functools import lru_cache
from math import comb, cos, pi, sqrt


def lcg(seed):
    return (1664525 * seed + 1013904223) & 0xffffffff


@lru_cache(maxsize=8192)
def u(dimensions, pulses):
    if dimensions == 0:
        return int(pulses == 0)
    if pulses == 0:
        return 0
    return sum((1 << k) * comb(dimensions - 1, k) * comb(pulses - 1, k)
               for k in range(min(dimensions, pulses)))


def decode_pulses(decoder, dimensions, pulses):
    index = decoder.uint(u(dimensions, pulses) + u(dimensions, pulses + 1))
    result = []
    for dimension in range(dimensions, 0, -1):
        boundary = u(dimension, pulses + 1)
        sign = -int(index >= boundary)
        index -= boundary & sign
        before = pulses
        boundary = u(dimension, pulses)
        while boundary > index:
            pulses -= 1
            boundary = u(dimension, pulses)
        index -= boundary
        result.append((before - pulses + sign) ^ sign)
    return result


def encode_pulses(encoder, values):
    dimensions = len(values)
    pulses = abs(values[-1])
    index = int(values[-1] < 0)
    for j in range(dimensions - 2, -1, -1):
        index += u(dimensions - j, pulses)
        pulses += abs(values[j])
        if values[j] < 0:
            index += u(dimensions - j, pulses + 1)
    encoder.uint(index, u(dimensions, pulses) + u(dimensions, pulses + 1))


def renormalise(values, gain=1.):
    scale = gain / sqrt(1e-15 + sum(value * value for value in values))
    for i in range(len(values)):
        values[i] *= scale


def rotation1(values, offset, length, stride, cosine, sine):
    for i in range(length - stride):
        first, second = values[offset + i], values[offset + i + stride]
        values[offset + i + stride] = cosine * second + sine * first
        values[offset + i] = cosine * first - sine * second
    for i in range(length - 2 * stride - 1, -1, -1):
        first, second = values[offset + i], values[offset + i + stride]
        values[offset + i + stride] = cosine * second + sine * first
        values[offset + i] = cosine * first - sine * second


def rotation(values, direction, stride, pulses, spread):
    length = len(values)
    if 2 * pulses >= length or spread == 0:
        return
    factor = (15,10,5)[spread - 1]
    gain = length / (length + factor * pulses)
    theta = .5 * gain * gain
    cosine = cos(.5 * pi * theta)
    sine = cos(.5 * pi * (1 - theta))
    stride2 = 0
    if length >= 8 * stride:
        stride2 = 1
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < length:
            stride2 += 1
    block = length // stride
    for i in range(stride):
        if direction < 0:
            if stride2:
                rotation1(values, i * block, block, stride2, sine, cosine)
            rotation1(values, i * block, block, 1, cosine, sine)
        else:
            rotation1(values, i * block, block, 1, cosine, -sine)
            if stride2:
                rotation1(values, i * block, block, stride2, sine, -cosine)


def unquant(decoder, dimensions, pulses, spread, blocks, gain):
    integers = decode_pulses(decoder, dimensions, pulses)
    scale = gain / sqrt(sum(value * value for value in integers))
    values = [scale * value for value in integers]
    rotation(values, -1, blocks, pulses, spread)
    if blocks <= 1:
        mask = 1
    else:
        size = dimensions // blocks
        mask = sum(int(any(integers[i * size:(i + 1) * size])) << i for i in range(blocks))
    return values, mask
