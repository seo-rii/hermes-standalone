"""Integer SILK NLSF stabilization and LPC prediction filters.

Adapted from the Skype SILK SDK, copyright 2006-2012 Skype Limited.
The original notice is retained in licenses/skype-silk-sdk.txt; source hashes
are recorded in silk-provenance.json. Intermediate operations retain the
SDK's explicit signed widths and fixed-point rounding.
"""

from standalone.silk_fixed import (
    cdiv, clz32, i16, i32, inverse32_varq, rshift_round, sat16, smmul, smulww,
)
from standalone.silk_tables import SKP_Silk_LSFCosTab_FIX_Q12


def nlsf_stabilize(values, deltas):
    """Mutate one NLSF vector to meet its codebook's minimum Q15 distances."""
    order = len(values)
    if not 1 <= order <= 16 or len(deltas) != order + 1 or deltas[-1] < 1:
        raise ValueError("Invalid SILK NLSF stabilization dimensions")
    if any(value < 0 for value in deltas) or sum(deltas) > 32768:
        raise ValueError("Invalid SILK NLSF minimum distances")
    for _ in range(20):
        smallest, selected = i32(values[0] - deltas[0]), 0
        for index in range(1, order):
            difference = i32(values[index] - i32(values[index - 1] + deltas[index]))
            if difference < smallest:
                smallest, selected = difference, index
        difference = i32(32768 - i32(values[-1] + deltas[-1]))
        if difference < smallest:
            smallest, selected = difference, order
        if smallest >= 0:
            return values
        if selected == 0:
            values[0] = deltas[0]
        elif selected == order:
            values[-1] = 32768 - deltas[-1]
        else:
            half = deltas[selected] >> 1
            minimum = sum(deltas[:selected]) + half
            maximum = 32768 - sum(deltas[selected + 1:]) - (deltas[selected] - half)
            center = rshift_round(i32(values[selected - 1] + values[selected]), 1)
            center = max(minimum, min(maximum, center))
            values[selected - 1] = center - half
            values[selected] = values[selected - 1] + deltas[selected]
    # The SDK bounds iterative pair adjustment, then sorts and enforces both
    # ends. Keeping this fallback also handles severely disordered vectors.
    values[:] = sorted(values)
    values[0] = max(values[0], deltas[0])
    for index in range(1, order):
        values[index] = max(values[index], values[index - 1] + deltas[index])
    values[-1] = min(values[-1], 32768 - deltas[-1])
    for index in range(order - 2, -1, -1):
        values[index] = min(values[index], values[index + 1] - deltas[index + 1])
    return values


def bwexpander(coefficients, chirp_q16):
    """Mutate Q12 int16 LPC coefficients using the SDK's rounded chirp."""
    if not coefficients:
        raise ValueError("SILK bandwidth expansion requires coefficients")
    chirp_q16 = i32(chirp_q16)
    difference = i32(chirp_q16 - 65536)
    for index in range(len(coefficients) - 1):
        coefficients[index] = i16(rshift_round(i32(chirp_q16 * i16(coefficients[index])), 16))
        chirp_q16 = i32(chirp_q16 + rshift_round(i32(chirp_q16 * difference), 16))
    coefficients[-1] = i16(rshift_round(i32(chirp_q16 * i16(coefficients[-1])), 16))
    return coefficients


def bwexpander32(coefficients, chirp_q16):
    """Mutate int32 LPC coefficients with the SDK's full-word multiply."""
    if not coefficients:
        raise ValueError("SILK bandwidth expansion requires coefficients")
    current = i32(chirp_q16)
    for index in range(len(coefficients) - 1):
        coefficients[index] = smulww(coefficients[index], current)
        current = smulww(chirp_q16, current)
    coefficients[-1] = smulww(coefficients[-1], current)
    return coefficients


def _nlsf_polynomial(cosines):
    order = len(cosines)
    result = [1 << 20, -cosines[0]] + [0] * (order - 1)
    for index in range(1, order):
        cosine = cosines[index]
        result[index + 1] = i32(i32(result[index - 1] << 1)
                                 - i32(rshift_round(cosine * result[index], 20)))
        for position in range(index, 1, -1):
            result[position] = i32(result[position] + result[position - 2]
                                   - i32(rshift_round(cosine * result[position - 1], 20)))
        result[1] = i32(result[1] - cosine)
    return result


def nlsf2a(nlsf):
    """Convert an even Q15 NLSF vector to bounded Q12 whitening coefficients."""
    order = len(nlsf)
    if order < 2 or order > 16 or order & 1 or any(not 0 <= value <= 32767 for value in nlsf):
        raise ValueError("Invalid SILK NLSF vector")
    cosines = []
    for value in nlsf:
        integer, fraction = value >> 8, value & 255
        start = SKP_Silk_LSFCosTab_FIX_Q12[integer]
        delta = SKP_Silk_LSFCosTab_FIX_Q12[integer + 1] - start
        cosines.append((start << 8) + delta * fraction)
    even, odd = _nlsf_polynomial(cosines[::2]), _nlsf_polynomial(cosines[1::2])
    result = [0] * order
    for index in range(order // 2):
        p, q = i32(even[index + 1] + even[index]), i32(odd[index + 1] - odd[index])
        result[index] = i32(-rshift_round(i32(p + q), 9))
        result[order - index - 1] = rshift_round(i32(q - p), 9)
    for _ in range(10):
        maximum, selected = 0, 0
        for index, value in enumerate(result):
            if abs(value) > maximum:
                maximum, selected = abs(value), index
        if maximum <= 32767:
            break
        maximum = min(maximum, 98369)
        scale = 65470 - cdiv((65470 >> 2) * (maximum - 32767), maximum * (selected + 1) >> 2)
        bwexpander32(result, scale)
    else:
        result = [sat16(value) for value in result]
    return [i16(value) for value in result]


def lpc_inverse_pred_gain(coefficients, *, q=12):
    """Return (unstable, inverse gain Q30), including the early-failure gain."""
    order = len(coefficients)
    if not 1 <= order <= 16 or q not in (12, 24):
        raise ValueError("Invalid SILK LPC inverse-gain dimensions")
    scratch = [[0] * order, [0] * order]
    current = scratch[order & 1]
    for index, value in enumerate(coefficients):
        current[index] = i32(i16(value) << 4) if q == 12 else rshift_round(i32(value), 8)
    gain = 1 << 30
    for index in range(order - 1, 0, -1):
        if not -65520 <= current[index] <= 65520:
            return 1, gain
        reflection = i32(-(current[index] << 15))
        multiplier = 1073741823 - smmul(reflection, reflection)
        reciprocal = inverse32_varq(multiplier, 46)
        gain = i32(smmul(gain, multiplier) << 2)
        previous, current = current, scratch[index & 1]
        headroom = clz32(reciprocal) - 1
        reciprocal = i32(reciprocal << headroom)
        for position in range(index):
            temporary = i32(previous[position] - i32(smmul(previous[index - position - 1], reflection) << 1))
            current[position] = i32(smmul(temporary, reciprocal) << (16 - headroom))
    if not -65520 <= current[0] <= 65520:
        return 1, gain
    reflection = i32(-(current[0] << 15))
    multiplier = 1073741823 - smmul(reflection, reflection)
    return 0, i32(smmul(gain, multiplier) << 2)


def nlsf2a_stable(nlsf):
    """Use the decoder's 20-step bandwidth expansion to stabilize Q12 LPCs."""
    coefficients = nlsf2a(nlsf)
    for iteration in range(20):
        if not lpc_inverse_pred_gain(coefficients)[0]:
            return coefficients
        bwexpander(coefficients, 65536 - (10 + iteration) * iteration)
    return [0] * len(coefficients)
