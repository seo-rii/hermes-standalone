"""CELT decoder energy and time/frequency flags, from Opus 1.3.1.

Copyright (c) 2007–2008 CSIRO, (c) 2007–2009 Xiph.Org Foundation.
Original redistribution notices are retained in licenses/opus*.txt.
"""
from .opus_tables import ENERGY_PROB

_PRED = (29440/32768, 26112/32768, 21248/32768, 16384/32768)
_BETA = (30147/32768, 22282/32768, 12124/32768, 6554/32768)
_TF = ((0,-1,0,-1,0,-1,0,-1), (0,-1,0,-2,1,0,1,-1), (0,-2,0,-3,2,0,1,-1), (0,-2,0,-3,3,0,1,-1))


def coarse(start, end, old, intra, decoder, channels, lm):
    previous = [0., 0.]
    coefficient = 0 if intra else _PRED[lm]
    beta = 4915/32768 if intra else _BETA[lm]
    probabilities = ENERGY_PROB[lm][intra]
    budget = decoder.storage * 8
    for band in range(start, end):
        for channel in range(channels):
            left = budget - decoder.tell()
            if left >= 15:
                index = 2 * min(band, 20)
                delta = decoder.laplace(probabilities[index] << 7, probabilities[index + 1] << 6)
            elif left >= 2:
                delta = decoder.icdf((2,1,0), 2)
                delta = (delta >> 1) ^ -(delta & 1)
            elif left >= 1:
                delta = -decoder.bit_logp(1)
            else:
                delta = -1
            index = band + channel * 21
            old[index] = coefficient * max(-9., old[index]) + previous[channel] + delta
            previous[channel] += (1 - beta) * delta


def fine_energy(start, end, old, fine, decoder, channels):
    for band in range(start, end):
        if fine[band] <= 0:
            continue
        for channel in range(channels):
            value = decoder.bits(fine[band])
            old[band + channel * 21] += (value + .5) / (1 << fine[band]) - .5


def final_energy(start, end, old, fine, priority, bits_left, decoder, channels):
    for pass_priority in (0, 1):
        for band in range(start, end):
            if bits_left < channels:
                break
            if fine[band] >= 8 or priority[band] != pass_priority:
                continue
            for channel in range(channels):
                value = decoder.bits(1)
                old[band + channel * 21] += (value - .5) / (1 << (fine[band] + 1))
                bits_left -= 1


def tf_decode(start, end, transient, lm, decoder):
    budget, tell = decoder.storage * 8, decoder.tell()
    logp = 2 if transient else 4
    reserve = int(lm > 0 and tell + logp + 1 <= budget)
    budget -= reserve
    changed = current = 0
    result = [0] * 21
    for band in range(start, end):
        if tell + logp <= budget:
            current ^= decoder.bit_logp(logp)
            tell = decoder.tell()
            changed |= current
        result[band] = current
        logp = 4 if transient else 5
    selection = 0
    if reserve and _TF[lm][4 * transient + changed] != _TF[lm][4 * transient + 2 + changed]:
        selection = decoder.bit_logp(1)
    for band in range(start, end):
        result[band] = _TF[lm][4 * transient + 2 * selection + result[band]]
    return result
