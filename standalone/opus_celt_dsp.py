"""Scalar CELT transforms, overlap/add and pitch postfilter (Opus 1.3.1).

Copyright (c) 2003–2004 Mark Borgerding, (c) 2005–2011 Xiph.Org
Foundation, (c) 2007–2008 CSIRO, Jean-Marc Valin. Original notices:
licenses/opus*.txt. The mixed-radix FFT uses Python complex arithmetic.
"""
from functools import lru_cache
from math import cos, pi, sin, sqrt
from .opus_tables import EBANDS, MDCT_TRIG, WINDOW, ENERGY_MEAN
from .opus_celt_pvq import renormalise, lcg


@lru_cache(maxsize=32)
def _twiddles(length):
    return tuple(complex(cos(-2 * pi * k / length), sin(-2 * pi * k / length)) for k in range(length))


def fft(values):
    length = len(values)
    if length == 1:
        return list(values)
    if length == 2:
        return [values[0] + values[1],values[0] - values[1]]
    if length == 3:
        first,second,third = values
        combined = second + third
        base = first - .5 * combined
        side = -0.8660254037844386j * (second - third)
        return [first + combined,base + side,base - side]
    if length == 5:
        first,second,third,fourth,fifth = values
        outer,inner = second + fifth,third + fourth
        outer_difference,inner_difference = second - fifth,third - fourth
        base1 = first + .30901699437494745 * outer - .8090169943749473 * inner
        side1 = -1j * (.9510565162951535 * outer_difference + .5877852522924732 * inner_difference)
        base2 = first - .8090169943749473 * outer + .30901699437494745 * inner
        side2 = -1j * (.5877852522924732 * outer_difference - .9510565162951535 * inner_difference)
        return [first + outer + inner,base1 + side1,base2 + side2,base2 - side2,base1 - side1]
    twiddles = _twiddles(length)
    if length % 2 == 0:
        size = length >> 1
        even,odd = fft(values[::2]),fft(values[1::2])
        result = even + [0j] * size
        for k in range(size):
            value = odd[k] * twiddles[k]
            result[k],result[k + size] = even[k] + value,even[k] - value
        return result
    if length % 3 == 0:
        size = length // 3
        parts = [fft(values[index::3]) for index in range(3)]
        result = [0j] * length
        for k in range(size):
            first,second,third = parts[0][k],parts[1][k] * twiddles[k],parts[2][k] * twiddles[2 * k]
            combined = second + third
            base = first - .5 * combined
            side = -0.8660254037844386j * (second - third)
            result[k],result[k + size],result[k + 2 * size] = first + combined,base + side,base - side
        return result
    factor = 5 if length % 5 == 0 else length
    size = length // factor
    parts = [fft(values[index::factor]) for index in range(factor)]
    return [sum(parts[index][k % size] * twiddles[index * k % length] for index in range(factor)) for k in range(length)]


def imdct(coefficients, output, offset, shift, stride=1):
    """Inverse CELT transform, retaining the next frame's 60 overlap values."""
    length = 1920 >> shift
    half, quarter = length >> 1, length >> 2
    trig_offset = sum(1920 >> i for i in range(1, shift + 1))
    trig = MDCT_TRIG[trig_offset:trig_offset + half]
    rotated = []
    for i in range(quarter):
        first, second = coefficients[2 * stride * i], coefficients[stride * (half - 1 - 2 * i)]
        real = first * trig[i] - second * trig[quarter + i]
        imaginary = second * trig[i] + first * trig[quarter + i]
        rotated.append(complex(real, imaginary))
    transformed = fft(rotated)
    for i, value in enumerate(transformed):
        output[offset + 60 + 2 * i] = value.imag * trig[i] + value.real * trig[quarter + i]
        output[offset + 60 + half - 1 - 2 * i] = value.imag * trig[quarter + i] - value.real * trig[i]
    for i in range(60):
        first, second = output[offset + 119 - i], output[offset + i]
        output[offset + i] = WINDOW[119 - i] * second - WINDOW[i] * first
        output[offset + 119 - i] = WINDOW[i] * second + WINDOW[119 - i] * first


def denormalise(values, energy, start, end, lm, downsample=1, silence=False):
    multiplier = 1 << lm
    size = 120 * multiplier
    bound = min(multiplier * EBANDS[end], size // downsample)
    if silence:
        return [0.] * size
    result = [0.] * size
    for band in range(start, end):
        gain = 2. ** min(32., energy[band] + ENERGY_MEAN[band])
        for index in range(multiplier * EBANDS[band], multiplier * EBANDS[band + 1]):
            result[index] = values[index] * gain
    result[bound:] = [0.] * (size - bound)
    return result


def synthesis(spectra, histories, energy, start, end, channels, transient, lm, downsample=1, silence=False):
    size = 120 << lm
    stream_channels = len(spectra)
    frequencies = [denormalise(spectra[c], energy[c * 21:c * 21 + 21], start, end, lm, downsample, silence) for c in range(stream_channels)]
    if channels == 2 and stream_channels == 1:
        frequencies = [frequencies[0], frequencies[0][:]]
    elif channels == 1 and stream_channels == 2:
        frequencies = [[.5 * (first + second) for first, second in zip(*frequencies)]]
    blocks = 1 << lm if transient else 1
    block_size = 120 if transient else size
    shift = 3 if transient else 3 - lm
    for channel in range(channels):
        for block in range(blocks):
            imdct(frequencies[channel][block:], histories[channel], 2048 - size + block_size * block, shift, blocks)


_GAINS = ((.3066406250,.2170410156,.1296386719), (.4638671875,.2680664062,0.), (.7998046875,.1000976562,0.))


def comb_filter(values, offset, old_period, period, size, old_gain, gain, old_tapset, tapset, overlap=120, *, inplace=True):
    if old_gain == gain == 0:
        return values if inplace else values[:]
    output = values if inplace else values[:]
    old_period, period = max(old_period, 15), max(period, 15)
    old = [old_gain * value for value in _GAINS[old_tapset]]
    new = [gain * value for value in _GAINS[tapset]]
    if old_gain == gain and old_period == period and old_tapset == tapset:
        overlap = 0
    for i in range(size):
        current = offset + i
        if i < overlap:
            fade = WINDOW[i] * WINDOW[i]
            delayed = current - old_period
            first = old[0] * values[delayed] + old[1] * (values[delayed + 1] + values[delayed - 1]) + old[2] * (values[delayed + 2] + values[delayed - 2])
            delayed = current - period
            second = new[0] * values[delayed] + new[1] * (values[delayed + 1] + values[delayed - 1]) + new[2] * (values[delayed + 2] + values[delayed - 2])
            output[current] = values[current] + (1 - fade) * first + fade * second
        elif gain:
            delayed = current - period
            output[current] = values[current] + new[0] * values[delayed] + new[1] * (values[delayed + 1] + values[delayed - 1]) + new[2] * (values[delayed + 2] + values[delayed - 2])

    return output


def anti_collapse(spectra, masks, lm, start, end, energy, previous1, previous2, pulses, seed):
    channels, multiplier = len(spectra), 1 << lm
    for band in range(start, end):
        width = EBANDS[band + 1] - EBANDS[band]
        depth = (1 + pulses[band]) // width >> lm
        threshold = .5 * 2. ** (-.125 * depth)
        inverse_sqrt = 1 / sqrt(width << lm)
        for channel in range(channels):
            index = channel * 21 + band
            prev1, prev2 = previous1[index], previous2[index]
            if channels == 1:
                prev1 = max(prev1, previous1[21 + band])
                prev2 = max(prev2, previous2[21 + band])
            difference = max(0., energy[index] - min(prev1, prev2))
            gain = 2. * 2. ** -difference
            if lm == 3:
                gain *= 1.41421356
            gain = min(threshold, gain) * inverse_sqrt
            offset = EBANDS[band] << lm
            changed = False
            for block in range(multiplier):
                if not masks[band * channels + channel] & 1 << block:
                    for j in range(width):
                        seed = lcg(seed)
                        spectra[channel][offset + (j << lm) + block] = gain if seed & 0x8000 else -gain
                    changed = True
            if changed:
                values = spectra[channel][offset:offset + (width << lm)]
                renormalise(values)
                spectra[channel][offset:offset + (width << lm)] = values
