"""Opus 1.3.1 range coding, including the reverse raw-bit channel.

Adapted from entcode.c, entdec.c, entenc.c and laplace.c, copyright (c) 2001–2011
Timothy B. Terriberry and (c) 2008–2009 Xiph.Org Foundation. Redistribution
terms and the original disclaimer are retained in licenses/opus.txt.
This is the Opus coder, not the incompatible Skype SILK v3 range coder.
"""

from __future__ import annotations


class OpusRangeError(ValueError):
    pass


_TOP = 1 << 31
_BOT = _TOP >> 8
_CORRECTION = (35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535)


def _table(table, precision):
    if type(precision) is not int or not 0 <= precision <= 8 or not 1 <= len(table) <= 256 or table[-1] != 0:
        raise OpusRangeError("Invalid Opus inverse CDF dimensions")
    previous = 1 << precision
    for value in table:
        if type(value) is not int or not 0 <= value <= previous or value == 1 << precision:
            raise OpusRangeError("Invalid Opus inverse CDF probability")
        previous = value


class _Context:
    def tell(self):
        return self.nbits_total - self.rng.bit_length()

    def tell_frac(self):
        length = self.rng.bit_length()
        scaled = self.rng >> (length - 16)
        fraction = (scaled >> 12) - 8
        fraction += scaled > _CORRECTION[fraction]
        return (self.nbits_total << 3) - ((length << 3) + fraction)


class OpusRangeDecoder(_Context):
    """Bounded range decoder with the SDK's defined zero-extension at EOF.

    A codec must also enforce its frame bit budget using tell()/tell_frac();
    entropy read-ahead and zero-extension alone are not packet validation.
    """

    def __init__(self, data, storage=None):
        self.buf = bytes(data)
        self.storage = len(self.buf) if storage is None else storage
        if type(self.storage) is not int or not 0 <= self.storage <= min(len(self.buf), 65535):
            raise OpusRangeError("Invalid Opus range input size")
        self.end_offs = self.end_window = self.nend_bits = self.offs = self.error = self.ext = 0
        self.nbits_total, self.rng = 9, 128
        self.rem = self._read()
        self.val = self.rng - 1 - (self.rem >> 1)
        self._normalize()

    def _read(self):
        if self.offs >= self.storage:
            return 0
        value = self.buf[self.offs]
        self.offs += 1
        return value

    def _read_end(self):
        if self.end_offs >= self.storage:
            return 0
        self.end_offs += 1
        return self.buf[self.storage - self.end_offs]

    def _normalize(self):
        while self.rng <= _BOT:
            self.nbits_total += 8
            self.rng <<= 8
            symbol = self.rem
            self.rem = self._read()
            symbol = ((symbol << 8) | self.rem) >> 1
            self.val = ((self.val << 8) + (255 & ~symbol)) & (_TOP - 1)

    def decode(self, total):
        if type(total) is not int or not 1 <= total <= self.rng:
            raise OpusRangeError("Invalid Opus range frequency total")
        self.ext = self.rng // total
        return total - min(self.val // self.ext + 1, total)

    def decode_bin(self, precision):
        if type(precision) is not int or not 0 <= precision <= 23:
            raise OpusRangeError("Invalid Opus binary frequency precision")
        self.ext = self.rng >> precision
        return (1 << precision) - min(self.val // self.ext + 1, 1 << precision)

    def update(self, low, high, total):
        if not 0 <= low < high <= total or self.ext <= 0:
            raise OpusRangeError("Invalid Opus range update interval")
        scaled = self.ext * (total - high)
        self.val -= scaled
        self.rng = self.ext * (high - low) if low else self.rng - scaled
        if self.val < 0 or self.rng <= 0:
            raise OpusRangeError("Opus range update does not contain the decoded value")
        self._normalize()

    def bit_logp(self, logp):
        if type(logp) is not int or not 0 <= logp <= 23:
            raise OpusRangeError("Invalid Opus binary probability")
        split = self.rng >> logp
        value = int(self.val < split)
        if not value:
            self.val -= split
        self.rng = split if value else self.rng - split
        self._normalize()
        return value

    def icdf(self, table, ftb=8):
        _table(table, ftb)
        step, upper, value = self.rng >> ftb, self.rng, self.val
        for index, probability in enumerate(table):
            lower = step * probability
            if value >= lower:
                self.val, self.rng = value - lower, upper - lower
                self._normalize()
                return index
            upper = lower
        raise OpusRangeError("Opus inverse CDF cannot represent the decoded value")

    def uint(self, total):
        if type(total) is not int or not 2 <= total <= 0xFFFFFFFF:
            raise OpusRangeError("Invalid Opus unsigned integer total")
        maximum = total - 1
        precision = maximum.bit_length()
        if precision > 8:
            precision -= 8
            upper = (maximum >> precision) + 1
            value = self.decode(upper)
            self.update(value, value + 1, upper)
            decoded = (value << precision) | self.bits(precision)
            if decoded > maximum:
                self.error = 1
                return maximum
            return decoded
        value = self.decode(total)
        self.update(value, value + 1, total)
        return value

    def bits(self, count):
        if type(count) is not int or not 0 <= count <= 25:
            raise OpusRangeError("Invalid Opus raw-bit count")
        window, available = self.end_window, self.nend_bits
        if available < count:
            while True:
                window = (window | (self._read_end() << available)) & 0xFFFFFFFF
                available += 8
                if available > 24:
                    break
        result = window & ((1 << count) - 1)
        self.end_window = window >> count
        self.nend_bits = available - count
        self.nbits_total += count
        return result

    def laplace(self, fs, decay):
        """Decode CELT's signed Laplace energy delta (15-bit probability)."""
        if not 1 <= fs <= 32736 or not 0 <= decay <= 16384:
            raise OpusRangeError("Invalid Opus Laplace probability")
        value, low = 0, 0
        symbol = self.decode_bin(15)
        if symbol >= fs:
            value, low = 1, fs
            fs = ((32768 - 32 - fs) * (16384 - decay) >> 15) + 1
            while fs > 1 and symbol >= low + 2 * fs:
                fs *= 2
                low += fs
                fs = ((fs - 2) * decay >> 15) + 1
                value += 1
            if fs <= 1:
                delta = (symbol - low) >> 1
                value += delta
                low += 2 * delta
            if symbol < low + fs:
                value = -value
            else:
                low += fs
        self.update(low, min(low + fs, 32768), 32768)
        return value


class OpusRangeEncoder(_Context):
    """Deterministic Opus encoder; finish raises if the buffer was exhausted."""

    def __init__(self, storage=1275):
        if type(storage) is not int or not 1 <= storage <= 65535:
            raise OpusRangeError("Invalid Opus range output size")
        self.buf = bytearray(storage)
        self.storage, self.rng, self.nbits_total, self.rem = storage, _TOP, 33, -1
        self.end_offs = self.end_window = self.nend_bits = self.offs = self.val = self.ext = self.error = 0
        self._finished = False

    def _write(self, value, *, end=False):
        if self.offs + self.end_offs >= self.storage:
            self.error = -1
            return
        if end:
            self.end_offs += 1
            self.buf[self.storage - self.end_offs] = value & 255
        else:
            self.buf[self.offs] = value & 255
            self.offs += 1

    def _carry(self, value):
        if value == 255:
            self.ext += 1
            return
        carry = value >> 8
        if self.rem >= 0:
            self._write(self.rem + carry)
        while self.ext:
            self._write((255 + carry) & 255)
            self.ext -= 1
        self.rem = value & 255

    def _normalize(self):
        if self._finished:
            raise OpusRangeError("Opus range encoder is already finalized")
        while self.rng <= _BOT:
            if not self.rng:
                raise OpusRangeError("Opus range interval has zero probability")
            self._carry(self.val >> 23)
            self.val = (self.val << 8) & (_TOP - 1)
            self.rng <<= 8
            self.nbits_total += 8

    def encode(self, low, high, total):
        if self._finished:
            raise OpusRangeError("Opus range encoder is already finalized")
        if not 0 <= low < high <= total <= self.rng:
            raise OpusRangeError("Invalid Opus encoding interval")
        step = self.rng // total
        if low:
            self.val = (self.val + self.rng - step * (total - low)) & 0xFFFFFFFF
            self.rng = step * (high - low)
        else:
            self.rng -= step * (total - high)
        self._normalize()

    def encode_bin(self, low, high, precision):
        if self._finished:
            raise OpusRangeError("Opus range encoder is already finalized")
        if type(precision) is not int or not 0 <= precision <= 23:
            raise OpusRangeError("Invalid Opus binary encoding precision")
        total = 1 << precision
        if not 0 <= low < high <= total:
            raise OpusRangeError("Invalid Opus binary encoding interval")
        step = self.rng >> precision
        if low:
            self.val = (self.val + self.rng - step * (total - low)) & 0xFFFFFFFF
            self.rng = step * (high - low)
        else:
            self.rng -= step * (total - high)
        self._normalize()

    def bit_logp(self, value, logp):
        if self._finished:
            raise OpusRangeError("Opus range encoder is already finalized")
        if value not in (0, 1) or type(logp) is not int or not 0 <= logp <= 23 or (not logp and not value):
            raise OpusRangeError("Invalid Opus binary encoding probability")
        split = self.rng >> logp
        self.rng -= split
        if value:
            self.val = (self.val + self.rng) & 0xFFFFFFFF
            self.rng = split
        self._normalize()

    def icdf(self, value, table, ftb=8):
        if self._finished:
            raise OpusRangeError("Opus range encoder is already finalized")
        _table(table, ftb)
        if not 0 <= value < len(table):
            raise OpusRangeError("Invalid Opus inverse CDF symbol")
        if table[value] == (table[value - 1] if value else 1 << ftb):
            raise OpusRangeError("Opus inverse CDF symbol has zero probability")
        step = self.rng >> ftb
        if value:
            self.val = (self.val + self.rng - step * table[value - 1]) & 0xFFFFFFFF
            self.rng = step * (table[value - 1] - table[value])
        else:
            self.rng -= step * table[0]
        self._normalize()

    def uint(self, value, total):
        if type(total) is not int or not 2 <= total <= 0xFFFFFFFF or not 0 <= value < total:
            raise OpusRangeError("Invalid Opus unsigned encoding value")
        precision = (total - 1).bit_length()
        if precision > 8:
            precision -= 8
            upper, symbol = ((total - 1) >> precision) + 1, value >> precision
            self.encode(symbol, symbol + 1, upper)
            self.bits(value & ((1 << precision) - 1), precision)
        else:
            self.encode(value, value + 1, total)

    def bits(self, value, count):
        if type(count) is not int or not 1 <= count <= 25 or not 0 <= value < 1 << count:
            raise OpusRangeError("Invalid Opus raw encoding bits")
        if self._finished:
            raise OpusRangeError("Opus range encoder is already finalized")
        window, used = self.end_window, self.nend_bits
        if used + count > 32:
            while True:
                self._write(window & 255, end=True)
                window >>= 8
                used -= 8
                if used < 8:
                    break
        self.end_window = (window | (value << used)) & 0xFFFFFFFF
        self.nend_bits = used + count
        self.nbits_total += count

    def patch_initial_bits(self, value, count):
        if self._finished:
            raise OpusRangeError("Opus range encoder is already finalized")
        if not 0 <= count <= 8 or not 0 <= value < 1 << count:
            raise OpusRangeError("Invalid Opus initial-bit patch")
        shift = 8 - count
        mask = ((1 << count) - 1) << shift
        if self.offs:
            self.buf[0] = (self.buf[0] & ~mask) | (value << shift)
        elif self.rem >= 0:
            self.rem = (self.rem & ~mask) | (value << shift)
        elif self.rng <= _TOP >> count:
            self.val = (self.val & ~(mask << 23)) | (value << (23 + shift))
        else:
            self.error = -1

    def laplace(self, value, fs, decay):
        """Encode an energy delta and return its representable, clamped value."""
        if type(value) is not int or not 1 <= fs <= 32736 or not 0 <= decay <= 16384:
            raise OpusRangeError("Invalid Opus Laplace encoding probability")
        low, result = 0, value
        if value:
            sign = -int(value < 0)
            value = (value + sign) ^ sign
            low = fs
            fs = (32768 - 32 - fs) * (16384 - decay) >> 15
            index = 1
            while fs > 0 and index < value:
                fs *= 2
                low += fs + 2
                fs = fs * decay >> 15
                index += 1
            if not fs:
                limit = (32768 - low - sign) >> 1
                delta = min(value - index, limit - 1)
                low += 2 * delta + 1 + sign
                fs = min(1, 32768 - low)
                result = (index + delta + sign) ^ sign
            else:
                fs += 1
                low += fs & ~sign
        self.encode_bin(low, low + fs, 15)
        return result

    def shrink(self, size):
        if type(size) is not int or not self.offs + self.end_offs <= size <= self.storage or self._finished:
            raise OpusRangeError("Invalid Opus range output shrink")
        if self.end_offs:
            self.buf[size - self.end_offs:size] = self.buf[self.storage - self.end_offs:self.storage]
        self.storage = size

    def done(self):
        if self._finished:
            raise OpusRangeError("Opus range encoder is already finalized")
        length = 32 - self.rng.bit_length()
        mask = (_TOP - 1) >> length
        end = (self.val + mask) & ~mask
        if (end | mask) >= self.val + self.rng:
            length += 1
            mask >>= 1
            end = (self.val + mask) & ~mask
        while length > 0:
            self._carry(end >> 23)
            end = (end << 8) & (_TOP - 1)
            length -= 8
        if self.rem >= 0 or self.ext:
            self._carry(0)
        window, used = self.end_window, self.nend_bits
        while used >= 8:
            self._write(window & 255, end=True)
            window >>= 8
            used -= 8
        if not self.error:
            self.buf[self.offs:self.storage - self.end_offs] = b"\0" * (self.storage - self.end_offs - self.offs)
            if used:
                if self.end_offs >= self.storage:
                    self.error = -1
                else:
                    length = -length
                    if self.offs + self.end_offs >= self.storage and length < used:
                        window &= (1 << length) - 1
                        self.error = -1
                    self.buf[self.storage - self.end_offs - 1] |= window & 255
        self._finished = True
        if self.error:
            raise OpusRangeError("Opus range output buffer was exhausted")
        return bytes(self.buf[:self.storage])
