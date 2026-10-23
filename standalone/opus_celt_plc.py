"""CELT noise and pitch loss concealment, from scalar Opus 1.3.1.

Copyright (c) 2007–2011 Xiph.Org Foundation and CSIRO, Jean-Marc Valin.
Original notices are retained in licenses/opus*.txt.
"""
from math import sqrt
from struct import Struct

_FLOAT32 = Struct("<f")

def f32(value):
    return _FLOAT32.unpack(_FLOAT32.pack(value))[0]
from .opus_celt_pvq import lcg, renormalise
from .opus_celt_dsp import synthesis, comb_filter
from .opus_tables import EBANDS, WINDOW


def autocorr(samples, order, window=False):
    values = [f32(value) for value in samples]
    size = len(values)
    if window:
        for i in range(120):
            values[i] = f32(values[i] * WINDOW[i])
            values[size - i - 1] = f32(values[size - i - 1] * WINDOW[i])
    fast_size = size - order
    result = []
    for lag in range(order + 1):
        total = 0.
        for i in range(fast_size):
            total = f32(total + f32(values[i] * values[i + lag]))
        tail = 0.
        for i in range(lag + fast_size, size):
            tail = f32(tail + f32(values[i] * values[i - lag]))
        result.append(f32(total + tail))
    return result


def lpc_coefficients(correlations, order):
    result = [0.] * order
    error = correlations[0]
    if not error:
        return result
    for i in range(order):
        total = 0.
        for j in range(i):
            total = f32(total + f32(result[j] * correlations[i - j]))
        total = f32(total + correlations[i + 1])
        reflection = -f32(total / error)
        result[i] = reflection
        for j in range((i + 1) >> 1):
            first, second = result[j], result[i - 1 - j]
            result[j] = f32(first + f32(reflection * second))
            result[i - 1 - j] = f32(second + f32(reflection * first))
        error = f32(error - f32(f32(reflection * reflection) * error))
        if error < f32(f32(.001) * correlations[0]):
            break
    return result


def pitch_downsample(histories):
    low = [0.] * 1024
    for history in histories:
        low[0] += .5 * (.5 * history[1] + history[0])
        for i in range(1,1024):
            low[i] += .5 * (.5 * (history[2 * i - 1] + history[2 * i + 1]) + history[2 * i])
    correlations = autocorr(low,4)
    correlations[0] = f32(correlations[0] * f32(1.0001))
    for i in range(1,5):
        lag = f32(f32(.008) * i)
        correlations[i] = f32(correlations[i] - f32(f32(correlations[i] * lag) * lag))
    coefficients = lpc_coefficients(correlations,4)
    for i in range(4):
        coefficients[i] *= .9 ** (i + 1)
    fir = [coefficients[0] + .8] + [coefficients[i] + .8 * coefficients[i - 1] for i in range(1,4)] + [.8 * coefficients[3]]
    original = low[:]
    for i in range(1024):
        low[i] += sum(fir[j] * original[i - j - 1] for j in range(min(i,5)))
    return low


def best_pitch(correlations, values, size, maximum):
    energy = 1 + sum(value * value for value in values[:size])
    numerators, denominators, pitches = [-1.,-1.], [0.,0.], [0,1]
    for i in range(maximum):
        if correlations[i] > 0:
            numerator = (correlations[i] * 1e-12) ** 2
            if numerator * denominators[1] > numerators[1] * energy:
                if numerator * denominators[0] > numerators[0] * energy:
                    numerators[1], denominators[1], pitches[1] = numerators[0], denominators[0], pitches[0]
                    numerators[0], denominators[0], pitches[0] = numerator, energy, i
                else:
                    numerators[1], denominators[1], pitches[1] = numerator, energy, i
        energy = max(1., energy + values[i + size] ** 2 - values[i] ** 2)
    return pitches


def pitch_search(histories):
    low = pitch_downsample(histories)
    values = low[360:]
    size, maximum = 1328, 620
    fourth, y_fourth = values[:size >> 1:2], low[:(size + maximum) >> 1:2]
    correlations = [sum(x * y for x,y in zip(fourth,y_fourth[i:i + (size >> 2)])) for i in range(maximum >> 2)]
    coarse = best_pitch(correlations, y_fourth, size >> 2, maximum >> 2)
    correlations = [0.] * (maximum >> 1)
    for i in range(maximum >> 1):
        if abs(i - 2 * coarse[0]) <= 2 or abs(i - 2 * coarse[1]) <= 2:
            correlations[i] = max(-1., sum(values[j] * low[i + j] for j in range(size >> 1)))
    best = best_pitch(correlations,low,size >> 1,maximum >> 1)[0]
    offset = 0
    if 0 < best < (maximum >> 1) - 1:
        before, current, after = correlations[best - 1:best + 2]
        if after - before > .7 * (current - before):
            offset = 1
        elif before - after > .7 * (current - after):
            offset = -1
    return 720 - (2 * best - offset)


def conceal(state, size, lm, start, end):
    loss_count = state.loss_count
    if loss_count >= 5 or start != 0 or state.skip_plc:
        decay = 1.5 if loss_count == 0 else .5
        spectra = [[0.] * size for _ in range(state.channels)]
        for channel in range(state.channels):
            for band in range(start,end):
                index = channel * 21 + band
                state.old_energy[index] = max(state.background_energy[index], state.old_energy[index] - decay)
                offset = EBANDS[band] << lm
                width = (EBANDS[band + 1] - EBANDS[band]) << lm
                values = []
                for _ in range(width):
                    state.rng = lcg(state.rng)
                    values.append(float(((state.rng ^ 0x80000000) - 0x80000000) >> 20))
                renormalise(values)
                spectra[channel][offset:offset + width] = values
        for history in state.histories:
            history[:2048 - size + 60] = history[size:2108]
        synthesis(spectra, state.histories, state.old_energy, start, end, state.channels, False, lm, state.downsample)
    else:
        if loss_count == 0:
            state.last_pitch_index = pitch_search(state.histories)
            fade = 1.
        else:
            fade = .8
        pitch = state.last_pitch_index
        excitation_length = min(2 * pitch,1024)
        for channel, history in enumerate(state.histories):
            excitation = history[1000:2048]
            if loss_count == 0:
                correlations = autocorr(excitation[24:],24,True)
                correlations[0] = f32(correlations[0] * f32(1.0001))
                for i in range(1,25):
                    product = f32(correlations[i] * f32(f32(.008) * f32(.008)))
                    product = f32(f32(product * i) * i)
                    correlations[i] = f32(correlations[i] - product)
                state.lpc[channel] = lpc_coefficients(correlations,24)
            coefficients = state.lpc[channel]
            begin = 1048 - excitation_length
            filtered = [excitation[begin + i] + sum(coefficients[j] * excitation[begin + i - j - 1] for j in range(24)) for i in range(excitation_length)]
            excitation[begin:] = filtered
            decay_length = excitation_length >> 1
            energy1 = 1 + sum(value * value for value in excitation[1048 - decay_length:])
            energy2 = 1 + sum(value * value for value in excitation[1048 - 2 * decay_length:1048 - decay_length])
            decay = sqrt(min(energy1,energy2) / energy2)
            history[:2048 - size] = history[size:2048]
            extrapolation_offset, extrapolation_length = 1024 - pitch, size + 120
            attenuation = fade * decay
            original_energy, j = 0.,0
            for i in range(extrapolation_length):
                if j >= pitch:
                    j -= pitch
                    attenuation *= decay
                history[2048 - size + i] = attenuation * excitation[24 + extrapolation_offset + j]
                previous = history[2048 - 1024 - size + extrapolation_offset + j]
                original_energy += previous * previous
                j += 1
            memory = [history[2048 - size - 1 - i] for i in range(24)]
            for i in range(extrapolation_length):
                index = 2048 - size + i
                value = history[index] - sum(coefficients[j] * memory[j] for j in range(24))
                memory[1:] = memory[:-1]
                memory[0], history[index] = value,value
            extrapolated_energy = sum(value * value for value in history[2048 - size:2168])
            if not original_energy > .2 * extrapolated_energy:
                history[2048 - size:2168] = [0.] * extrapolation_length
            elif original_energy < extrapolated_energy:
                ratio = sqrt((original_energy + 1) / (extrapolated_energy + 1))
                for i in range(extrapolation_length):
                    gain = 1 - WINDOW[i] * (1 - ratio) if i < 120 else ratio
                    history[2048 - size + i] *= gain
            filtered = comb_filter(history,2048,state.postfilter_period,state.postfilter_period,120,-state.postfilter_gain,-state.postfilter_gain,state.postfilter_tapset,state.postfilter_tapset,0,inplace=False)
            for i in range(60):
                history[2048 + i] = WINDOW[i] * filtered[2048 + 119 - i] + WINDOW[119 - i] * filtered[2048 + i]
    state.loss_count = loss_count + 1
