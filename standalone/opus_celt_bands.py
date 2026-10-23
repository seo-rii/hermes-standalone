"""CELT recursive band decoder, adapted from Opus 1.3.1 bands.c.

Copyright (c) 2007–2008 CSIRO, (c) 2007–2009 Xiph.Org Foundation,
(c) 2008–2009 Gregory Maxwell. Original notices: licenses/opus*.txt.
"""
from math import isqrt, sqrt
from .opus_celt_alloc import bits2pulses, pulses2bits, get_pulses, cdiv
from .opus_celt_pvq import unquant, lcg, renormalise
from .opus_tables import EBANDS, LOG_N, CACHE_BITS, CACHE_INDEX

_ORDERY = (1,0,3,0,2,1,7,0,4,3,6,1,5,2,15,0,8,7,12,3,11,4,14,1,9,6,13,2,10,5)
_INTERLEAVE = (0,1,1,1,2,3,3,3,2,3,3,3,2,3,3,3)
_DEINTERLEAVE = (0x00,0x03,0x0c,0x0f,0x30,0x33,0x3c,0x3f,0xc0,0xc3,0xcc,0xcf,0xf0,0xf3,0xfc,0xff)


def frac_mul(a, b):
    a, b = (a + 32768) % 65536 - 32768, (b + 32768) % 65536 - 32768
    return (16384 + a * b) >> 15


def bitexact_cos(value):
    squared = (4096 + value * value) >> 13
    return 1 + (32767 - squared) + frac_mul(squared, -7651 + frac_mul(squared, 8277 + frac_mul(-626, squared)))


def bitexact_log2tan(sine, cosine):
    lc, ls = cosine.bit_length(), sine.bit_length()
    cosine <<= 15 - lc
    sine <<= 15 - ls
    return (ls - lc) * 2048 + frac_mul(sine, frac_mul(sine, -2597) + 7932) - frac_mul(cosine, frac_mul(cosine, -2597) + 7932)


def haar(values, n0, stride):
    for i in range(stride):
        for j in range(n0 >> 1):
            first = .70710678 * values[stride * 2 * j + i]
            second = .70710678 * values[stride * (2 * j + 1) + i]
            values[stride * 2 * j + i] = first + second
            values[stride * (2 * j + 1) + i] = first - second


def interleave(values, n0, stride, hadamard, inverse=False):
    result = [0.] * len(values)
    for i in range(stride):
        ordered = _ORDERY[stride - 2 + i] if hadamard else i
        for j in range(n0):
            if inverse:
                result[j * stride + i] = values[ordered * n0 + j]
            else:
                result[ordered * n0 + j] = values[j * stride + i]
    values[:] = result


def compute_qn(dimensions, bits, offset, pulse_cap, stereo):
    degrees = 2 * dimensions - 1 - int(stereo and dimensions == 2)
    qb = min(cdiv(bits + degrees * offset, degrees), bits - pulse_cap - 32, 64)
    if qb < 4:
        return 1
    qn = (16384,17866,19483,21247,23170,25267,27554,30048)[qb & 7] >> (14 - (qb >> 3))
    return (qn + 1) >> 1 << 1


class BandContext:
    def __init__(self, decoder, intensity, spread, seed, disable_inv):
        self.decoder, self.intensity, self.spread = decoder, intensity, spread
        self.seed, self.disable_inv = seed, disable_inv
        self.band = self.tf_change = self.remaining_bits = 0

    def theta(self, dimensions, bits, blocks, original_blocks, lm, stereo, fill):
        pulse_cap = LOG_N[self.band] + lm * 8
        offset = (pulse_cap >> 1) - (16 if stereo and dimensions == 2 else 4)
        qn = compute_qn(dimensions, bits, offset, pulse_cap, stereo)
        if stereo and self.band >= self.intensity:
            qn = 1
        before = self.decoder.tell_frac()
        angle, inverse = 0, 0
        if qn != 1:
            if stereo and dimensions > 2:
                half = qn // 2
                total = 3 * (half + 1) + half
                symbol = self.decoder.decode(total)
                angle = symbol // 3 if symbol < (half + 1) * 3 else half + 1 + symbol - (half + 1) * 3
                low = 3 * angle if angle <= half else angle - 1 - half + (half + 1) * 3
                high = 3 * (angle + 1) if angle <= half else angle - half + (half + 1) * 3
                self.decoder.update(low, high, total)
            elif original_blocks > 1 or stereo:
                angle = self.decoder.uint(qn + 1)
            else:
                total = ((qn >> 1) + 1) ** 2
                symbol = self.decoder.decode(total)
                if symbol < ((qn >> 1) * ((qn >> 1) + 1) >> 1):
                    angle = (isqrt(8 * symbol + 1) - 1) >> 1
                    frequency = angle + 1
                    low = angle * (angle + 1) >> 1
                else:
                    angle = (2 * (qn + 1) - isqrt(8 * (total - symbol - 1) + 1)) >> 1
                    frequency = qn + 1 - angle
                    low = total - ((qn + 1 - angle) * (qn + 2 - angle) >> 1)
                self.decoder.update(low, low + frequency, total)
            angle = angle * 16384 // qn
        elif stereo:
            if bits > 16 and self.remaining_bits > 16:
                inverse = self.decoder.bit_logp(2)
            if self.disable_inv:
                inverse = 0
        allocated = self.decoder.tell_frac() - before
        bits -= allocated
        if angle == 0:
            middle, side, delta = 32767, 0, -16384
            fill &= (1 << blocks) - 1
        elif angle == 16384:
            middle, side, delta = 0, 32767, 16384
            fill &= ((1 << blocks) - 1) << blocks
        else:
            middle, side = bitexact_cos(angle), bitexact_cos(16384 - angle)
            delta = frac_mul((dimensions - 1) << 7, bitexact_log2tan(side, middle))
        return bits, fill, inverse, middle / 32768., side / 32768., delta, angle, allocated

    def n1(self, stereo):
        values = []
        for _ in range(1 + stereo):
            sign = 0
            if self.remaining_bits >= 8:
                sign = self.decoder.bits(1)
                self.remaining_bits -= 8
            values.append([-1. if sign else 1.])
        return values

    def partition(self, dimensions, bits, blocks, lowband, lm, gain, fill):
        offset = CACHE_INDEX[(lm + 1) * 21 + self.band]
        if lm != -1 and bits > CACHE_BITS[offset + CACHE_BITS[offset]] + 12 and dimensions > 2:
            half = dimensions >> 1
            lm -= 1
            original_blocks = blocks
            if blocks == 1:
                fill = (fill & 1) | (fill << 1)
            blocks = (blocks + 1) >> 1
            bits, fill, _, middle, side, delta, angle, allocated = self.theta(half, bits, blocks, original_blocks, lm, False, fill)
            if original_blocks > 1 and angle & 0x3fff:
                if angle > 8192:
                    delta -= delta >> (4 - lm)
                else:
                    delta = min(0, delta + (half << 3 >> (5 - lm)))
            middle_bits = max(0, min(bits, cdiv(bits - delta, 2)))
            side_bits = bits - middle_bits
            self.remaining_bits -= allocated
            first_low = lowband[:half] if lowband is not None else None
            second_low = lowband[half:] if lowband is not None else None
            before = self.remaining_bits
            if middle_bits >= side_bits:
                first, mask = self.partition(half, middle_bits, blocks, first_low, lm, gain * middle, fill)
                rebalance = middle_bits - (before - self.remaining_bits)
                if rebalance > 24 and angle != 0:
                    side_bits += rebalance - 24
                second, side_mask = self.partition(half, side_bits, blocks, second_low, lm, gain * side, fill >> blocks)
            else:
                second, side_mask = self.partition(half, side_bits, blocks, second_low, lm, gain * side, fill >> blocks)
                rebalance = side_bits - (before - self.remaining_bits)
                if rebalance > 24 and angle != 16384:
                    middle_bits += rebalance - 24
                first, mask = self.partition(half, middle_bits, blocks, first_low, lm, gain * middle, fill)
            return first + second, mask | (side_mask << (original_blocks >> 1))
        pulses = bits2pulses(self.band, lm, bits)
        used = pulses2bits(self.band, lm, pulses)
        self.remaining_bits -= used
        while self.remaining_bits < 0 and pulses > 0:
            self.remaining_bits += used
            pulses -= 1
            used = pulses2bits(self.band, lm, pulses)
            self.remaining_bits -= used
        if pulses:
            return unquant(self.decoder, dimensions, get_pulses(pulses), self.spread, blocks, gain)
        mask_limit = (1 << blocks) - 1
        fill &= mask_limit
        if not fill:
            return [0.] * dimensions, 0
        values = []
        for i in range(dimensions):
            self.seed = lcg(self.seed)
            if lowband is None:
                signed = (self.seed ^ 0x80000000) - 0x80000000
                values.append(float(signed >> 20))
            else:
                values.append(lowband[i] + (1/256 if self.seed & 0x8000 else -1/256))
        renormalise(values, gain)
        return values, mask_limit if lowband is None else fill

    def band_decode(self, dimensions, bits, blocks, lowband, lm, gain, fill):
        if dimensions == 1:
            values = self.n1(False)[0]
            return values, 1, values[:]
        original_dimensions, original_blocks = dimensions, blocks
        long_blocks = original_blocks == 1
        n_block = dimensions // blocks
        recombine = max(0, self.tf_change)
        tf_change = self.tf_change
        lowband = lowband[:] if lowband is not None else None
        for k in range(recombine):
            if lowband is not None:
                haar(lowband, dimensions >> k, 1 << k)
            fill = _INTERLEAVE[fill & 15] | (_INTERLEAVE[fill >> 4] << 2)
        blocks >>= recombine
        n_block <<= recombine
        time_divide = 0
        while not n_block & 1 and tf_change < 0:
            if lowband is not None:
                haar(lowband, n_block, blocks)
            fill |= fill << blocks
            blocks <<= 1
            n_block >>= 1
            time_divide += 1
            tf_change += 1
        changed_blocks, changed_n_block = blocks, n_block
        if blocks > 1 and lowband is not None:
            interleave(lowband, n_block >> recombine, blocks << recombine, long_blocks)
        values, mask = self.partition(dimensions, bits, blocks, lowband, lm, gain, fill)
        if changed_blocks > 1:
            interleave(values, n_block >> recombine, blocks << recombine, long_blocks, inverse=True)
        n_block, blocks = changed_n_block, changed_blocks
        for _ in range(time_divide):
            blocks >>= 1
            n_block <<= 1
            mask |= mask >> blocks
            haar(values, n_block, blocks)
        for k in range(recombine):
            mask = _DEINTERLEAVE[mask]
            haar(values, original_dimensions >> k, 1 << k)
        blocks <<= recombine
        scale = sqrt(original_dimensions)
        return values, mask & ((1 << blocks) - 1), [scale * value for value in values]

    def stereo(self, dimensions, bits, blocks, lowband, lm, fill):
        if dimensions == 1:
            first, second = self.n1(True)
            return first, second, 1, first[:]
        original_fill = fill
        bits, fill, inverse, middle, side, delta, angle, allocated = self.theta(dimensions, bits, blocks, blocks, lm, True, fill)
        if dimensions == 2:
            side_bits = 8 if angle not in (0, 16384) else 0
            middle_bits = bits - side_bits
            self.remaining_bits -= allocated + side_bits
            sign = 1 - 2 * self.decoder.bits(1) if side_bits else 1
            primary, mask, lowout = self.band_decode(dimensions, middle_bits, blocks, lowband, lm, 1., original_fill)
            secondary = [-sign * primary[1], sign * primary[0]]
            first, second = (secondary, primary) if angle > 8192 else (primary, secondary)
            first = [middle * value for value in first]
            second = [side * value for value in second]
            first, second = ([x - y for x, y in zip(first, second)], [x + y for x, y in zip(first, second)])
        else:
            middle_bits = max(0, min(bits, cdiv(bits - delta, 2)))
            side_bits = bits - middle_bits
            self.remaining_bits -= allocated
            before = self.remaining_bits
            if middle_bits >= side_bits:
                first, mask, lowout = self.band_decode(dimensions, middle_bits, blocks, lowband, lm, 1., fill)
                rebalance = middle_bits - (before - self.remaining_bits)
                if rebalance > 24 and angle != 0:
                    side_bits += rebalance - 24
                second, side_mask, _ = self.band_decode(dimensions, side_bits, blocks, None, lm, side, fill >> blocks)
            else:
                second, side_mask, _ = self.band_decode(dimensions, side_bits, blocks, None, lm, side, fill >> blocks)
                rebalance = side_bits - (before - self.remaining_bits)
                if rebalance > 24 and angle != 16384:
                    middle_bits += rebalance - 24
                first, mask, lowout = self.band_decode(dimensions, middle_bits, blocks, lowband, lm, 1., fill)
            mask |= side_mask
            product = middle * sum(x * y for x, y in zip(first, second))
            energy = sum(y * y for y in second)
            left, right = middle * middle + energy - 2 * product, middle * middle + energy + 2 * product
            if min(left, right) < 6e-4:
                second = first[:]
            else:
                lgain, rgain = 1 / sqrt(left), 1 / sqrt(right)
                first, second = ([lgain * (middle * x - y) for x, y in zip(first, second)], [rgain * (middle * x + y) for x, y in zip(first, second)])
        if inverse:
            second = [-value for value in second]
        return first, second, mask, lowout


def decode_bands(start, end, channels, pulses, short_blocks, spread, dual, intensity, tf, total_bits, balance, decoder, lm, coded, seed, disable_inv):
    multiplier = 1 << lm
    blocks = multiplier if short_blocks else 1
    norm_offset = multiplier * EBANDS[start]
    norm_size = multiplier * EBANDS[20] - norm_offset
    norm, norm2 = [0.] * norm_size, [0.] * norm_size
    first, second = [0.] * (120 * multiplier), [0.] * (120 * multiplier) if channels == 2 else None
    masks = [0] * (21 * channels)
    ctx = BandContext(decoder, intensity, spread, seed, disable_inv)
    lowband_offset, update_lowband = 0, True
    for band in range(start, end):
        ctx.band = band
        last = band == end - 1
        offset = multiplier * EBANDS[band]
        dimensions = multiplier * (EBANDS[band + 1] - EBANDS[band])
        tell = decoder.tell_frac()
        if band != start:
            balance -= tell
        remaining = total_bits - tell - 1
        ctx.remaining_bits = remaining
        bits = max(0, min(16383, remaining + 1, pulses[band] + cdiv(balance, min(3, coded - band)))) if band < coded else 0
        if (offset - dimensions >= multiplier * EBANDS[start] or band == start + 1) and (update_lowband or lowband_offset == 0):
            lowband_offset = band
        if band == start + 1:
            n1 = multiplier * (EBANDS[start + 1] - EBANDS[start])
            n2 = multiplier * (EBANDS[start + 2] - EBANDS[start + 1])
            norm[n1:n2] = norm[2 * n1 - n2:n1]
            if dual:
                norm2[n1:n2] = norm2[2 * n1 - n2:n1]
        ctx.tf_change = tf[band]
        effective = -1
        if lowband_offset != 0 and (spread != 3 or blocks > 1 or tf[band] < 0):
            effective = max(0, multiplier * EBANDS[lowband_offset] - norm_offset - dimensions)
            fold_start = lowband_offset - 1
            while multiplier * EBANDS[fold_start] > effective + norm_offset:
                fold_start -= 1
            fold_end = lowband_offset
            while fold_end < band and multiplier * EBANDS[fold_end] < effective + norm_offset + dimensions:
                fold_end += 1
            x_mask = y_mask = 0
            for previous in range(fold_start, fold_end):
                x_mask |= masks[previous * channels]
                y_mask |= masks[previous * channels + channels - 1]
        else:
            x_mask = y_mask = (1 << blocks) - 1
        if dual and band == intensity:
            dual = 0
            for j in range(offset - norm_offset):
                norm[j] = .5 * (norm[j] + norm2[j])
        low = norm[effective:effective + dimensions] if effective != -1 else None
        if dual:
            x, x_mask, lowout = ctx.band_decode(dimensions, bits // 2, blocks, low, lm, 1., x_mask)
            low2 = norm2[effective:effective + dimensions] if effective != -1 else None
            y, y_mask, lowout2 = ctx.band_decode(dimensions, bits // 2, blocks, low2, lm, 1., y_mask)
            if not last:
                norm2[offset - norm_offset:offset - norm_offset + dimensions] = lowout2
        elif channels == 2:
            x, y, x_mask, lowout = ctx.stereo(dimensions, bits, blocks, low, lm, x_mask | y_mask)
            y_mask = x_mask
        else:
            x, x_mask, lowout = ctx.band_decode(dimensions, bits, blocks, low, lm, 1., x_mask | y_mask)
            y_mask = x_mask
        first[offset:offset + dimensions] = x
        if channels == 2:
            second[offset:offset + dimensions] = y
        if not last:
            norm[offset - norm_offset:offset - norm_offset + dimensions] = lowout
        masks[band * channels], masks[band * channels + channels - 1] = x_mask, y_mask
        balance += pulses[band] + tell
        update_lowband = bits > dimensions << 3
    return [first] if channels == 1 else [first, second], masks, ctx.seed
