"""Skype SILK fixed-point operations with explicit signed widths.

Derived from SKP_Silk_macros.h, SKP_Silk_SigProc_FIX.h and
SKP_Silk_Inlines.h; original notice: licenses/skype-silk-sdk.txt.
"""


def i16(value):
    return ((value + 32768) & 65535) - 32768


def i32(value):
    return ((value + 2147483648) & 0xFFFFFFFF) - 2147483648


def u32(value):
    return value & 0xFFFFFFFF


def sat16(value):
    return max(-32768, min(32767, value))


def sat32(value):
    return max(-2147483648, min(2147483647, value))


def clz32(value):
    return 32 - (value & 0xFFFFFFFF).bit_length()


def lshift_sat32(value, shift):
    if not 0 <= shift < 32:
        raise ValueError("SILK saturating shift must be below 32")
    return max(-2147483648 >> shift, min(2147483647 >> shift, value)) << shift


def rshift_round(value, shift):
    if shift < 0:
        raise ValueError("SILK rounding shift must be nonnegative")
    return value if not shift else ((value >> (shift - 1)) + 1) >> 1


def smulwb(a, b):
    return i32(i32(a) * i16(b) >> 16)


def smlawb(a, b, c):
    return i32(a + smulwb(b, c))


def smulww(a, b):
    return i32(i32(a) * i32(b) >> 16)


def smlaww(a, b, c):
    return i32(a + smulww(b, c))


def smmul(a, b):
    return i32(i32(a) * i32(b) >> 32)


def cdiv(a, b):
    if not b:
        raise ValueError("SILK division by zero")
    value = abs(a) // abs(b)
    return -value if (a < 0) != (b < 0) else value


def div32_varq(a, b, q):
    a, b = i32(a), i32(b)
    if not b or a == -2147483648 or b == -2147483648 or not 0 <= q <= 64:
        raise ValueError("Invalid SILK fixed-point division")
    ah, bh = clz32(abs(a)) - 1, clz32(abs(b)) - 1
    an, bn = i32(a << ah), i32(b << bh)
    inverse = cdiv(2147483647 >> 2, bn >> 16)
    result = smulwb(an, inverse)
    an = i32(an - i32(smmul(bn, result) << 3))
    result = smlawb(result, an, inverse)
    shift = 29 + ah - bh - q
    if shift <= 0:
        return lshift_sat32(result, -shift)
    return result >> shift if shift < 32 else 0


def inverse32_varq(b, q):
    b = i32(b)
    if not b or b == -2147483648 or not 0 < q <= 96:
        raise ValueError("Invalid SILK fixed-point inverse")
    headroom = clz32(abs(b)) - 1
    bn = i32(b << headroom)
    inverse = cdiv(2147483647 >> 2, bn >> 16)
    result = i32(inverse << 16)
    error = i32(-smulwb(bn, inverse) << 3)
    result = smlaww(result, error, inverse)
    shift = 61 - headroom - q
    if shift <= 0:
        return lshift_sat32(result, -shift)
    return result >> shift if shift < 32 else 0


def log2lin(value):
    if value < 0:
        return 0
    if value >= 3968:
        return 2147483647
    out, fraction = 1 << (value >> 7), value & 127
    polynomial = smlawb(fraction, fraction * (128 - fraction), -174)
    return i32(out + ((out * polynomial) >> 7) if value < 2048 else out + (out >> 7) * polynomial)
