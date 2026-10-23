"""CELT's deterministic decoder allocation (Opus 1.3.1 rate.c/rate.h).

Copyright (c) 2007–2008 CSIRO, (c) 2007–2009 Xiph.Org Foundation.
Original redistribution notices are retained in licenses/opus*.txt.
"""
from .opus_tables import EBANDS, ALLOC_VECTORS, LOG_N, CACHE_INDEX, CACHE_BITS, CACHE_CAPS

_LOG2_FRAC = (0,8,13,16,19,21,23,24,26,27,28,29,30,31,32,32,33,34,34,35,36,36,37,37)


def cdiv(numerator, denominator):
    return numerator // denominator if numerator >= 0 else -((-numerator) // denominator)


def get_pulses(index):
    return index if index < 8 else (8 + (index & 7)) << ((index >> 3) - 1)


def bits2pulses(band, lm, bits):
    offset = CACHE_INDEX[(lm + 1) * 21 + band]
    low, high = 0, CACHE_BITS[offset]
    bits -= 1
    for _ in range(6):
        middle = (low + high + 1) >> 1
        if CACHE_BITS[offset + middle] >= bits:
            high = middle
        else:
            low = middle
    return low if bits - (-1 if low == 0 else CACHE_BITS[offset + low]) <= CACHE_BITS[offset + high] - bits else high


def pulses2bits(band, lm, pulses):
    return 0 if pulses == 0 else CACHE_BITS[CACHE_INDEX[(lm + 1) * 21 + band] + pulses] + 1


def init_caps(lm, channels):
    return [(CACHE_CAPS[21 * (2 * lm + channels - 1) + i] + 64) * channels * ((EBANDS[i + 1] - EBANDS[i]) << lm) >> 2 for i in range(21)]


def allocation(start, end, offsets, cap, trim, total, channels, lm, decoder):
    """Return coded bands, intensity, dual stereo, balance, PVQ/fine allocations."""
    total = max(total, 0)
    skip_start = start
    skip_reserve = 8 if total >= 8 else 0
    total -= skip_reserve
    intensity_reserve = dual_reserve = 0
    if channels == 2:
        intensity_reserve = _LOG2_FRAC[end - start]
        if intensity_reserve > total:
            intensity_reserve = 0
        else:
            total -= intensity_reserve
            dual_reserve = 8 if total >= 8 else 0
            total -= dual_reserve
    threshold, trim_offset = [0] * 21, [0] * 21
    for band in range(start, end):
        width = EBANDS[band + 1] - EBANDS[band]
        threshold[band] = max(channels << 3, 3 * width << lm << 3 >> 4)
        trim_offset[band] = channels * width * (trim - 5 - lm) * (end - band - 1) * (1 << (lm + 3)) >> 6
        if width << lm == 1:
            trim_offset[band] -= channels << 3
    low, high = 1, len(ALLOC_VECTORS) // 21 - 1
    while low <= high:
        middle = (low + high) >> 1
        summed, done = 0, False
        for band in range(end - 1, start - 1, -1):
            width = EBANDS[band + 1] - EBANDS[band]
            bits = channels * width * ALLOC_VECTORS[middle * 21 + band] << lm >> 2
            if bits > 0:
                bits = max(0, bits + trim_offset[band])
            bits += offsets[band]
            if bits >= threshold[band] or done:
                done = True
                summed += min(bits, cap[band])
            elif bits >= channels << 3:
                summed += channels << 3
        if summed > total:
            high = middle - 1
        else:
            low = middle + 1
    high, low = low, low - 1
    bits1, bits2 = [0] * 21, [0] * 21
    for band in range(start, end):
        width = EBANDS[band + 1] - EBANDS[band]
        first = channels * width * ALLOC_VECTORS[low * 21 + band] << lm >> 2
        second = cap[band] if high >= len(ALLOC_VECTORS) // 21 else channels * width * ALLOC_VECTORS[high * 21 + band] << lm >> 2
        if first > 0:
            first = max(0, first + trim_offset[band])
        if second > 0:
            second = max(0, second + trim_offset[band])
        if low > 0:
            first += offsets[band]
        second += offsets[band]
        if offsets[band] > 0:
            skip_start = band
        bits1[band], bits2[band] = first, max(0, second - first)
    floor = channels << 3
    stereo = int(channels > 1)
    low, high = 0, 64
    for _ in range(6):
        middle = (low + high) >> 1
        summed, done = 0, False
        for band in range(end - 1, start - 1, -1):
            value = bits1[band] + (middle * bits2[band] >> 6)
            if value >= threshold[band] or done:
                done = True
                summed += min(value, cap[band])
            elif value >= floor:
                summed += floor
        if summed > total:
            high = middle
        else:
            low = middle
    bits, fine, priority = [0] * 21, [0] * 21, [0] * 21
    summed, done = 0, False
    for band in range(end - 1, start - 1, -1):
        value = bits1[band] + (low * bits2[band] >> 6)
        if value < threshold[band] and not done:
            value = floor if value >= floor else 0
        else:
            done = True
        bits[band] = min(value, cap[band])
        summed += bits[band]
    coded = end
    while True:
        band = coded - 1
        if band <= skip_start:
            total += skip_reserve
            break
        left = total - summed
        per_coefficient = left // (EBANDS[coded] - EBANDS[start])
        left -= (EBANDS[coded] - EBANDS[start]) * per_coefficient
        remainder = max(left - (EBANDS[band] - EBANDS[start]), 0)
        width = EBANDS[coded] - EBANDS[band]
        band_bits = bits[band] + per_coefficient * width + remainder
        if band_bits >= max(threshold[band], floor + 8):
            if decoder.bit_logp(1):
                break
            summed += 8
            band_bits -= 8
        summed -= bits[band] + intensity_reserve
        if intensity_reserve > 0:
            intensity_reserve = _LOG2_FRAC[band - start]
        summed += intensity_reserve
        bits[band] = floor if band_bits >= floor else 0
        summed += bits[band]
        coded -= 1
    intensity = start + decoder.uint(coded + 1 - start) if intensity_reserve > 0 else 0
    if intensity <= start:
        total += dual_reserve
        dual_reserve = 0
    dual = decoder.bit_logp(1) if dual_reserve > 0 else 0
    left = total - summed
    per_coefficient = left // (EBANDS[coded] - EBANDS[start])
    left -= (EBANDS[coded] - EBANDS[start]) * per_coefficient
    for band in range(start, coded):
        bits[band] += per_coefficient * (EBANDS[band + 1] - EBANDS[band])
    for band in range(start, coded):
        remainder = min(left, EBANDS[band + 1] - EBANDS[band])
        bits[band] += remainder
        left -= remainder
    balance = 0
    for band in range(start, coded):
        width = (EBANDS[band + 1] - EBANDS[band]) << lm
        bit = bits[band] + balance
        if width > 1:
            excess = max(bit - cap[band], 0)
            bits[band] = bit - excess
            denominator = channels * width + int(channels == 2 and width > 2 and not dual and band < intensity)
            n_log_n = denominator * (LOG_N[band] + (lm << 3))
            offset = (n_log_n >> 1) - denominator * 21
            if width == 2:
                offset += denominator << 3 >> 2
            if bits[band] + offset < denominator * 2 << 3:
                offset += n_log_n >> 2
            elif bits[band] + offset < denominator * 3 << 3:
                offset += n_log_n >> 3
            fine[band] = max(0, bits[band] + offset + (denominator << 2)) // denominator >> 3
            if channels * fine[band] > bits[band] >> 3:
                fine[band] = bits[band] >> stereo >> 3
            fine[band] = min(fine[band], 8)
            priority[band] = int(fine[band] * (denominator << 3) >= bits[band] + offset)
            bits[band] -= channels * fine[band] << 3
        else:
            excess = max(0, bit - (channels << 3))
            bits[band] = bit - excess
            fine[band], priority[band] = 0, 1
        if excess > 0:
            extra = min(excess >> (stereo + 3), 8 - fine[band])
            fine[band] += extra
            extra_bits = extra * channels << 3
            priority[band] = int(extra_bits >= excess - balance)
            excess -= extra_bits
        balance = excess
    for band in range(coded, end):
        fine[band] = bits[band] >> stereo >> 3
        bits[band] = 0
        priority[band] = int(fine[band] < 1)
    return coded, intensity, dual, balance, bits, fine, priority
