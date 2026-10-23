"""Bounded SILK range and shell decoder, derived from the Skype SDK.

The original notice is in licenses/skype-silk-sdk.txt. Arithmetic state is
unsigned 32-bit; input exhaustion follows the SDK's zero padding and final
length/padding checks, rather than reading outside the packet.
"""

from standalone import silk_tables as T


class SilkError(ValueError):
    pass


class RangeDecoder:
    def __init__(self, packet):
        if not isinstance(packet, bytes) or not 0 < len(packet) <= 1024:
            raise SilkError("Invalid SILK entropy payload length")
        self.packet = packet
        self.buffer = packet + b"\0" * 4
        self.base = int.from_bytes(self.buffer[:4], "big")
        self.range = 65535
        self.index = 0

    def symbol(self, cdf, middle=1):
        if not 0 <= middle < len(cdf):
            raise SilkError("Invalid SILK probability index")
        high = cdf[middle]
        product = self.range * high
        if product > self.base:
            while True:
                middle -= 1
                if middle < 0:
                    raise SilkError("SILK probability interval underflow")
                low = cdf[middle]
                if self.range * low <= self.base:
                    break
                high = low
                if not high:
                    raise SilkError("SILK probability interval underflow")
        else:
            while True:
                low = high
                middle += 1
                if middle >= len(cdf):
                    raise SilkError("SILK probability interval overflow")
                high = cdf[middle]
                if self.range * high > self.base:
                    middle -= 1
                    break
                if high == 65535:
                    raise SilkError("SILK probability interval overflow")
        self.base = (self.base - self.range * low) & 0xFFFFFFFF
        product = self.range * (high - low)
        if product & 0xFF000000:
            self.range = product >> 16
        else:
            if product & 0xFFFF0000:
                self.range = product >> 8
                if self.base >> 24:
                    raise SilkError("SILK entropy normalization failed")
            else:
                self.range = product
                if self.base >> 16:
                    raise SilkError("SILK entropy normalization failed")
                self._byte()
            self._byte()
        if not self.range:
            raise SilkError("SILK entropy interval is empty")
        return middle

    def _byte(self):
        self.base = self.base << 8 & 0xFFFFFFFF
        if self.index < len(self.packet):
            self.base |= self.buffer[self.index + 4]
            self.index += 1

    def length(self):
        bits = self.index * 8 + 32 - (self.range - 1).bit_length() - 14
        return bits, (bits + 7) >> 3

    def finish(self):
        bits, size = self.length()
        if not 0 < size <= len(self.packet):
            raise SilkError("SILK entropy stream is truncated")
        if bits & 7:
            mask = 0xFF >> (bits & 7)
            if self.packet[size - 1] & mask != mask:
                raise SilkError("Invalid SILK entropy padding")
        return size


def _shell(decoder, total, level):
    if not total:
        return [0] * (2 << level)
    if not 0 < total < len(T.SKP_Silk_shell_code_table_offsets):
        raise SilkError("SILK pulse count is out of range")
    table = getattr(T, "SKP_Silk_shell_code_table" + str(level))
    offset = T.SKP_Silk_shell_code_table_offsets[total]
    left = decoder.symbol(table[offset:offset + total + 2], total >> 1)
    if not level:
        return [left, total - left]
    return _shell(decoder, left, level - 1) + _shell(decoder, total - left, level - 1)


def pulses(decoder, frame_length, signal_type, offset_type):
    if frame_length not in (160, 240, 320, 480) or signal_type not in (0, 1) or offset_type not in (0, 1):
        raise SilkError("Invalid SILK pulse frame parameters")
    rate = decoder.symbol(T.SKP_Silk_rate_levels_CDF[signal_type], T.SKP_Silk_rate_levels_CDF_offset)
    sums, shifts = [], []
    for _ in range(frame_length // 16):
        total = decoder.symbol(T.SKP_Silk_pulses_per_block_CDF[rate], T.SKP_Silk_pulses_per_block_CDF_offset)
        shift = 0
        while total == 19:
            shift += 1
            if shift > 10:
                raise SilkError("Excessive SILK pulse shift")
            total = decoder.symbol(T.SKP_Silk_pulses_per_block_CDF[9], T.SKP_Silk_pulses_per_block_CDF_offset)
        sums.append(total)
        shifts.append(shift)
    result = []
    for total in sums:
        result.extend(_shell(decoder, total, 3))
    for block, shift in enumerate(shifts):
        if shift:
            for i in range(block * 16, block * 16 + 16):
                for _ in range(shift):
                    result[i] = (result[i] << 1) + decoder.symbol(T.SKP_Silk_lsb_CDF)
    threshold = T.SKP_Silk_sign_CDF[9 * ((signal_type << 1) + offset_type) + rate]
    cdf = (0, threshold, 65535)
    for i, value in enumerate(result):
        if value:
            result[i] *= (decoder.symbol(cdf) << 1) - 1
    return result, rate
