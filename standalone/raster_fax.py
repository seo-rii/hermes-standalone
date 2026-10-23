"""TIFF CCITT Modified Huffman, T.4 and T.6 bilevel decoding.

Code words are the format tables in RFC 804 and ITU-T T.4 (07/96), tables 2-4.
The implementation uses scanline transitions and bounded Huffman traversal.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right


_WHITE = """
00110101 000111 0111 1000 1011 1100 1110 1111 10011 10100 00111 01000 001000 000011 110100 110101
101010 101011 0100111 0001100 0001000 0010111 0000011 0000100 0101000 0101011 0010011 0100100
0011000 00000010 00000011 00011010 00011011 00010010 00010011 00010100 00010101 00010110 00010111
00101000 00101001 00101010 00101011 00101100 00101101 00000100 00000101 00001010 00001011 01010010
01010011 01010100 01010101 00100100 00100101 01011000 01011001 01011010 01011011 01001010 01001011
00110010 00110011 00110100
11011 10010 010111 0110111 00110110 00110111 01100100 01100101 01101000 01100111 011001100
011001101 011010010 011010011 011010100 011010101 011010110 011010111 011011000 011011001
011011010 011011011 010011000 010011001 010011010 011000 010011011
""".split()
_BLACK = """
0000110111 010 11 10 011 0011 0010 00011 000101 000100 0000100 0000101 0000111 00000100 00000111
000011000 0000010111 0000011000 0000001000 00001100111 00001101000 00001101100 00000110111
00000101000 00000010111 00000011000 000011001010 000011001011 000011001100 000011001101
000001101000 000001101001 000001101010 000001101011 000011010010 000011010011 000011010100
000011010101 000011010110 000011010111 000001101100 000001101101 000011011010 000011011011
000001010100 000001010101 000001010110 000001010111 000001100100 000001100101 000001010010
000001010011 000000100100 000000110111 000000111000 000000100111 000000101000 000001011000
000001011001 000000101011 000000101100 000001011010 000001100110 000001100111
0000001111 000011001000 000011001001 000001011011 000000110011 000000110100 000000110101
0000001101100 0000001101101 0000001001010 0000001001011 0000001001100 0000001001101 0000001110010
0000001110011 0000001110100 0000001110101 0000001110110 0000001110111 0000001010010 0000001010011
0000001010100 0000001010101 0000001011010 0000001011011 0000001100100 0000001100101
""".split()
_EXTENDED = """00000001000 00000001100 00000001101 000000010010 000000010011 000000010100
000000010101 000000010110 000000010111 000000011100 000000011101 000000011110 000000011111""".split()


def _table(codes):
    runs = list(range(64)) + list(range(64, 1729, 64)) + list(range(1792, 2561, 64))
    return {(len(code), int(code, 2)): run for code, run in zip(codes + _EXTENDED, runs)}


_RUNS = (_table(_WHITE), _table(_BLACK))
_MODES = {(len(code), int(code, 2)): value for code, value in
          (("1", 0), ("011", 1), ("010", -1), ("000011", 2), ("000010", -2),
           ("0000011", 3), ("0000010", -3), ("001", "H"), ("0001", "P"))}
_REVERSE = bytes(int(f"{value:08b}"[::-1], 2) for value in range(256))


class _Bits:
    def __init__(self, data):
        self.data, self.offset = data, 0

    def bit(self):
        if self.offset >= len(self.data) * 8:
            raise ValueError("Truncated CCITT pixel stream")
        value = self.data[self.offset // 8] >> (7 - self.offset % 8) & 1
        self.offset += 1
        return value

    def code(self, table, limit):
        value = 0
        for width in range(1, limit + 1):
            value = value * 2 + self.bit()
            if (width, value) in table:
                return table[width, value]
        raise ValueError("Invalid or unsupported CCITT code word")

    def run(self, color, remaining):
        total = 0
        while True:
            count = self.code(_RUNS[color], 13)
            total += count
            if total > remaining:
                raise ValueError("CCITT run exceeds the image row")
            if count < 64:
                return total

    def eol(self):
        zeros = 0
        while not self.bit():
            zeros += 1
        if zeros < 11:
            raise ValueError("Missing CCITT end-of-line marker")


def _line1(bits, width):
    row, x, color, operations = bytearray(width), 0, 0, 0
    while x < width:
        count = bits.run(color, width - x)
        if color:
            row[x:x + count] = b"\1" * count
        x, color, operations = x + count, color ^ 1, operations + 1
        if operations > width * 2 + 4:
            raise ValueError("CCITT row contains excessive empty runs")
    return row


def _line2(bits, width, reference):
    row, a0, color, initial, operations = bytearray(width), 0, 0, True, 0
    while a0 < width:
        mode = bits.code(_MODES, 7)
        if mode == "H":
            a1 = a0 + bits.run(color, width - a0)
            a2 = a1 + bits.run(color ^ 1, width - a1)
            if color:
                row[a0:a1] = b"\1" * (a1 - a0)
            else:
                row[a1:a2] = b"\1" * (a2 - a1)
            a0 = a2
        else:
            index = bisect_left(reference, a0) if initial else bisect_right(reference, a0)
            index += (index & 1) ^ color
            b1 = reference[index] if index < len(reference) else width
            if mode == "P":
                end = reference[index + 1] if index + 1 < len(reference) else width
            else:
                end = b1 + mode
            if not a0 <= end <= width or mode == "P" and end == a0:
                raise ValueError("CCITT reference transition exceeds the row")
            if color:
                row[a0:end] = b"\1" * (end - a0)
            a0 = end
            if mode != "P":
                color ^= 1
        initial, operations = False, operations + 1
        if operations > width * 2 + 4:
            raise ValueError("CCITT row contains excessive empty transitions")
    return row


def decode(data, width, height, compression, t4_options=0, t6_options=0, fill=1):
    from standalone.raster import checked_size
    checked_size((width, height))
    if compression not in (2, 3, 4) or t4_options & ~5 or t6_options or fill not in (1, 2):
        raise ValueError("Unsupported CCITT fax options")
    bits = _Bits(data.translate(_REVERSE) if fill == 2 else data)
    stride, reference = (width + 7) // 8, [width, width]
    output = bytearray(stride * height)
    for y in range(height):
        one_dimensional = compression != 4
        if compression == 3:
            bits.eol()
            if t4_options & 1:
                one_dimensional = bool(bits.bit())
                if y == 0 and not one_dimensional:
                    raise ValueError("CCITT T.4 strip must begin with a one-dimensional row")
        row = _line1(bits, width) if one_dimensional else _line2(bits, width, reference)
        reference = [x for x in range(width) if row[x] != (row[x - 1] if x else 0)] + [width, width]
        for x in range(width):
            if row[x]:
                output[y * stride + x // 8] |= 128 >> (x % 8)
        if compression == 2:
            bits.offset = (bits.offset + 7) & ~7
    # Pixel rows are fully decoded before optional RTC/EOFB markers and byte padding.
    if compression == 2:
        if any(bits.data[bits.offset // 8:]):
            raise ValueError("Unexpected data after CCITT Huffman rows")
    elif compression == 4:
        remaining = len(bits.data) * 8 - bits.offset
        if remaining > 31:
            raise ValueError("Unexpected data after CCITT Group 4 rows")
        suffix = 0
        for _ in range(remaining):
            suffix = suffix * 2 + bits.bit()
        padding = remaining - 24
        if suffix and not (0 <= padding <= 7 and suffix == 0x1001 << padding):
            raise ValueError("Invalid CCITT Group 4 end-of-block marker")
    return bytes(output)
