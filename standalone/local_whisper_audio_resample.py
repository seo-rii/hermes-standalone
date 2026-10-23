"""Default FFmpeg 8.1.2 swresample filter and sample conversion (LGPL-2.1+).

Algorithm source: libswresample/{resample,rematrix,audioconvert}.c.
"""
import math
import struct
from functools import lru_cache

_FLOAT = struct.Struct('<f')


def f32(value):
    try:
        return _FLOAT.unpack(_FLOAT.pack(value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _i0(x):
    p = (-2.2335582639474375249e15, -5.5050369673018427753e14,
         -3.2940087627407749166e13, -8.4925101247114157499e11,
         -1.1912746104985237192e10, -1.0313066708737980747e8,
         -5.9545626019847898221e5, -2.4125195876041896775e3,
         -7.093534744921054919, -.015453977791786851041,
         -.000025172644670688975051, -3.0517226450451067446e-8,
         -2.6843448573468483278e-11, -1.5982226675653184646e-14,
         -5.2487866627945699800e-18)
    q = (-2.2335582639474375245e15, 7.8858692566751002988e12,
         -1.2207067397808979846e10, 1.0377081058062166144e7,
         -4852.7560179962773045, 1.)
    y = x * x
    a = p[-1]
    for value in reversed(p[:-1]):
        a = a * y + value
    b = q[-1]
    for value in reversed(q[:-1]):
        b = b * y + value
    return a / b if x else 1.


@lru_cache(maxsize=16)
def _filters(input_rate, output_rate, precision):
    factor = min(output_rate * .97 / input_rate, 1.)
    length = max(math.ceil(32 / factor), 1)
    length += length > 1 and length % 2
    center = (length - 1) // 2
    phases = min(1024, output_rate // math.gcd(input_rate, output_rate))
    # SWR uses the exact numerator only when it fits the original bank.
    if output_rate // math.gcd(input_rate, output_rate) > 1024:
        phases = 1024
    count = phases if phases % 2 else phases // 2 + 1
    bank = [None] * (phases + 1)
    norm = 0.
    for phase in range(count):
        tab = []
        sine = math.sin(math.pi * phase / phases) * (1 if center & 1 else -1)
        for i in range(length):
            x = math.pi * ((i - center) - phase / phases) * factor
            value = 1. if x == 0 else sine / x if factor == 1 else math.sin(x) / x
            w = 2 * x / (factor * length * math.pi)
            value *= _i0(9 * math.sqrt(max(1 - w * w, 0)))
            tab.append(value)
            sine = -sine
            if phase == 0:
                norm += value
        if precision == 's16':
            coeffs = [max(-32768, min(32767, round(f32(v * 32768 / norm)))) for v in tab]
        else:
            coeffs = [f32(v / norm) if precision == 'f32' else v / norm for v in tab]
        coeffs += [0.] * ((-length) % 8)
        bank[phase] = coeffs
        if phases % 2 == 0 and phase:
            bank[phases - phase] = coeffs[:length][::-1] + coeffs[length:]
    # The extra phase is a one-sample shift, with the first source phase's
    # padding participating exactly as in resample_init.
    bank[phases] = [bank[0][-1]] + bank[0][:-1]
    return bank, length, center, phases


def _sum_float(samples, offset, coeffs):
    lanes = [0.] * 8
    for i, coeff in enumerate(coeffs):
        lane = i & 7
        lanes[lane] = f32(samples[offset + i] * coeff + lanes[lane])
    lanes = [f32(lanes[i] + lanes[i + 4]) for i in range(4)]
    return f32(f32(lanes[0] + lanes[2]) + f32(lanes[1] + lanes[3]))


def _sum_double(samples, offset, coeffs):
    even = odd = 0.
    for i in range(0, len(coeffs), 2):
        even += samples[offset + i] * coeffs[i]
        if i + 1 < len(coeffs):
            odd += samples[offset + i + 1] * coeffs[i + 1]
    return even + odd


def _sum_s16(samples, offset, coeffs):
    value = 16384 + sum(samples[offset + i] * coeff for i, coeff in enumerate(coeffs) if coeff)
    value = (value + (1 << 31)) % (1 << 32) - (1 << 31)
    return value


def resample(samples, input_rate, output_rate, precision='f32'):
    """Resample one complete planar stream, including SWR's flush reflection."""
    if input_rate == output_rate:
        return samples
    if not samples:
        return []
    bank, length, center, phases = _filters(input_rate, output_rate, precision)
    size = len(samples)
    remaining = size
    if size >= length + 1:
        boundary = size + center - length + 1
        emitted = (boundary * output_rate + input_rate - 1) // input_rate
        remaining += center - emitted * input_rate // output_rate
    reflection = (min(remaining, length) + 1) // 2
    if size + reflection < length + 1:
        return []
    extended = list(samples) + list(samples[-reflection:][::-1])
    extended = extended[1:center + 1][::-1] + extended
    last = len(extended) - length + 1
    extended += [0.] * 7
    numerator = input_rate * phases
    denominator = output_rate
    common = math.gcd(numerator, denominator)
    src_incr, dst_incr = denominator // common, numerator // common
    while src_incr < (1 << 20) and dst_incr < (1 << 20):
        src_incr *= 2
        dst_incr *= 2
    div, mod = divmod(dst_incr, src_incr)
    index = frac = offset = 0
    output = []
    summer = {'f32': _sum_float, 'f64': _sum_double, 's16': _sum_s16}[precision]
    convert = f32 if precision == 'f32' else float
    inverse = convert(1. / convert(src_incr))
    while offset < last:
        value = summer(extended, offset, bank[index])
        if frac and precision == 's16':
            second = summer(extended, offset, bank[index + 1])
            delta = (second - value) * frac
            value += abs(delta) // src_incr * (1 if delta >= 0 else -1)
        elif frac:
            second = summer(extended, offset, bank[index + 1])
            # The optimized SWR linear kernel rounds each scalar operation.
            delta = convert(convert(second - value) * inverse)
            value = convert(value + convert(delta * convert(frac)))
        output.append(max(-32768, min(32767, value >> 15)) if precision == 's16' else value)
        frac += mod
        index += div
        if frac >= src_incr:
            frac -= src_incr
            index += 1
        advance, index = divmod(index, phases)
        offset += advance
    return output


def to_s16(value):
    """Default no-dither normalized float to signed PCM16 conversion."""
    if not math.isfinite(value) or abs(value) >= (1 << 48):
        return -32768
    return max(-32768, min(32767, round(value * 32768)))


DEFAULT_LAYOUTS = {1: 4, 2: 3, 3: 11, 4: 0x107, 5: 0x37,
                   6: 0x3f, 7: 0x70f, 8: 0x63f}


def _stereo_coeff(channel, mask, center, half):
    if channel in (0, 1):
        return (1., 0.) if channel == 0 else (0., 1.)
    if channel == 2:
        value = center if mask & 3 else half
        return value, value
    if channel in (4, 9):
        return center, 0.
    if channel in (5, 10):
        return 0., center
    if channel == 8:
        return center * half, center * half
    if channel in (6, 7):
        return (1., 0.) if channel == 6 else (0., 1.)
    if channel in (12, 14):
        value = half if mask & 1 else 1.
        return (value, 0.) if channel == 12 else (0., value)
    return 0., 0.


def _mono_coeff(channel, mask, center, half):
    if channel == 2:
        return center * math.sqrt(2) if mask & 3 else 1.
    if channel in (0, 1, 6, 7, 12, 14):
        return half
    if channel in (4, 5, 8, 9, 10):
        return center * half
    return 0.


def _matrix(mask, count, stereo):
    if count == 1:
        mask = 4
    if mask is None or mask == 0:
        mask = DEFAULT_LAYOUTS.get(count)
    if mask is None or not 1 <= count < 64 or mask.bit_count() != count or mask >> 64:
        raise ValueError('Unsupported or inconsistent audio channel layout')
    if not mask & 7 or any((mask & pair) not in (0, pair) for pair in (3, 0x30, 0x600, 0xc0, 0x5000)):
        raise ValueError('Unsupported asymmetric audio channel layout')
    # swr_build_matrix2 deliberately treats 22.2 as 7.1(wide-back)+BC
    # when downmixing. Its compact first nine matrix columns consume the
    # first nine input planes; all fifteen remaining planes stay zero.
    channels = list(range(9)) + [-1] * 15 if mask == 2164663779327 else [i for i in range(64) if mask & (1 << i)]
    half = math.sqrt(.5)
    center = f32(half)
    if stereo:
        rows = [list(row) for row in zip(*(_stereo_coeff(ch, mask, center, half) for ch in channels))]
    else:
        rows = [[_mono_coeff(ch, mask, center, half) for ch in channels]]
    maximum = max(map(lambda row: sum(map(abs, row)), rows))
    if maximum > 1.:
        rows = [[v / maximum for v in row] for row in rows]
    return rows


def _mix(channels, matrix, precision, surround=False):
    result = []
    for row in matrix:
        active = [(ch, value) for ch, value in enumerate(row) if value]
        if len(active) == 1 and active[0][1] == 1.:
            result.append(channels[active[0][0]])
            continue
        if precision == 's16':
            coeffs = [(ch, round(f32(value * 32768))) for ch, value in active]
            # Two-input and one-input kernels retain coefficient quantization
            # error between columns; generic multi-input uses matrix32.
            if len(active) <= 2 or surround:
                rem = 0.
                coeffs = []
                for ch, value in enumerate(row):
                    target = value * 32768 + rem
                    quant = round(f32(target))
                    rem += target - quant
                    if value:
                        coeffs.append((ch, quant))
            output = []
            for i in range(len(channels[0])):
                value = (sum(channels[ch][i] * c for ch, c in coeffs) + 16384) >> 15
                output.append(((value + 32768) & 65535) - 32768 if len(active) > 2 and not surround else max(-32768, min(32767, value)))
            result.append(output)
        elif precision == 'f32':
            coeffs = [(ch, f32(value)) for ch, value in active]
            output = []
            for i in range(len(channels[0])):
                value = 0.
                for ch, coeff in coeffs:
                    value = f32(value + f32(channels[ch][i] * coeff))
                output.append(value)
            result.append(output)
        else:
            result.append([sum(channels[ch][i] * c for ch, c in active) for i in range(len(channels[0]))])
    return result


def convert_audio(channels, input_rate, output_rate=16000, sample_format='s16', mask=None, stereo=False):
    """Apply original default format conversion, rematrix and resampling order."""
    if type(input_rate) is not int or input_rate < 1 or type(output_rate) is not int or output_rate < 1:
        raise ValueError('Audio sample rates must be positive integers')
    matrix = _matrix(mask, len(channels), stereo)
    effective_mask = mask or DEFAULT_LAYOUTS.get(len(channels))
    surround = stereo and ((len(channels) == 6 and effective_mask in (0x3f, 0x60f)) or
                           (len(channels) == 8 and effective_mask == 0x63f))
    identity = len(channels) == len(matrix) and all(
        value == (i == j) for i, row in enumerate(matrix) for j, value in enumerate(row))
    if identity and input_rate == output_rate:
        if sample_format == 's16':
            return channels
        if sample_format == 's32':
            return [[v >> 16 for v in ch] for ch in channels]
        if sample_format == 's64':
            return [[v >> 48 for v in ch] for ch in channels]
        if sample_format == 'u8':
            return [[(v - 128) << 8 for v in ch] for ch in channels]
        return [[to_s16(v) for v in ch] for ch in channels]
    precision = 's16' if sample_format == 'u8' or (sample_format == 's16' and input_rate == output_rate) else 'f64' if sample_format in ('dbl', 's64') else 'f32'
    if precision == 's16':
        working = [[(v - 128) << 8 for v in ch] for ch in channels] if sample_format == 'u8' else channels
    else:
        scale = {'u8': 1 / 128, 's16': 1 / 32768, 's32': 1 / 2147483648, 's64': 1 / 9223372036854775808, 'flt': 1., 'dbl': 1.}[sample_format]
        working = [[f32(v * scale) if precision == 'f32' else v * scale for v in ch] for ch in channels]
    resample_first = len(matrix) // len(channels) - 1 < output_rate / f32(input_rate) - 1.
    if resample_first:
        working = [resample(ch, input_rate, output_rate, precision) for ch in working]
        working = _mix(working, matrix, precision, surround)
    else:
        working = _mix(working, matrix, precision, surround)
        working = [resample(ch, input_rate, output_rate, precision) for ch in working]
    return working if precision == 's16' else [[to_s16(v) for v in ch] for ch in working]
