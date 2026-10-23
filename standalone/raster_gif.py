"""GIF87a/89a palettes, LZW, interlace and animation disposal in Python.

Format reference: https://www.w3.org/Graphics/GIF/spec-gif89a.txt
"""

from __future__ import annotations

import struct

from standalone.raster import MAX_FRAMES, MAX_IMAGE_PIXELS, checked_size, new


class Reader:
    def __init__(self, data):
        self.data, self.offset = data, 0

    def read(self, count):
        result = self.data[self.offset:self.offset + count]
        if len(result) != count:
            raise ValueError("Truncated GIF stream")
        self.offset += count
        return result

    def byte(self):
        return self.read(1)[0]

    def blocks(self):
        result = bytearray()
        while True:
            count = self.byte()
            if not count:
                return bytes(result)
            result.extend(self.read(count))


def _lzw(data, minimum, expected):
    if not 2 <= minimum <= 8:
        raise ValueError("Invalid GIF LZW minimum code size")
    clear, end = 1 << minimum, (1 << minimum) + 1
    table, bits, width, previous, result = [], 0, minimum + 1, None, bytearray()
    while bits + width <= len(data) * 8:
        start = bits // 8
        code = (int.from_bytes(data[start:start + 3], "little") >> (bits % 8)) & ((1 << width) - 1)
        bits += width
        if code == clear:
            table, width, previous = [bytes([i]) for i in range(clear)] + [None, None], minimum + 1, None
            continue
        if not table:
            raise ValueError("GIF LZW stream must start with a clear code")
        if code == end:
            if len(result) != expected:
                raise ValueError("GIF pixel count does not match the frame")
            return result
        if code < len(table) and table[code] is not None:
            entry = table[code]
        elif code == len(table) and previous is not None:
            entry = previous + previous[:1]
        else:
            raise ValueError("Invalid GIF LZW dictionary reference")
        if len(result) + len(entry) > expected:
            raise ValueError("GIF LZW stream exceeds the frame dimensions")
        result.extend(entry)
        if previous is not None and len(table) < 4096:
            table.append(previous + entry[:1])
            if len(table) == (1 << width) and width < 12:
                width += 1
        previous = entry
    raise ValueError("Truncated GIF LZW stream or missing end code")


def decode(data):
    reader = Reader(data)
    if reader.read(6) not in (b"GIF87a", b"GIF89a"):
        raise ValueError("Invalid GIF signature")
    width, height, packed, background, _aspect = struct.unpack("<HHBBB", reader.read(7))
    checked_size((width, height))
    palette = reader.read(3 * (2 << (packed & 7))) if packed & 128 else b""
    if palette and background >= len(palette) // 3:
        raise ValueError("GIF background palette index is out of range")
    canvas, frames = None, []
    control, loop, previous_disposal = None, None, None
    while True:
        marker = reader.byte()
        if marker == 0x3B:
            if not frames or reader.offset != len(data):
                raise ValueError("Invalid GIF end marker")
            result = frames[0].copy()
            result.format, result._frames = "GIF", frames
            if loop is not None:
                result.info["loop"] = loop
            return result
        if marker == 0x21:
            label, extension = reader.byte(), reader.blocks()
            if label == 0xF9:
                if len(extension) != 4:
                    raise ValueError("Invalid GIF graphics control")
                flags, delay, transparent = struct.unpack("<BHB", extension)
                dispose = (flags >> 2) & 7
                if flags & 224 or dispose > 3:
                    raise ValueError("Invalid GIF disposal method")
                control = (dispose, delay * 10, transparent if flags & 1 else None)
            elif label == 0xFF and extension[:11] in (b"NETSCAPE2.0", b"ANIMEXTS1.0"):
                if len(extension) >= 14 and extension[11] == 1:
                    loop = int.from_bytes(extension[12:14], "little")
            elif label == 0x01:
                # Rendering text needs the stream's external character-cell font.
                raise ValueError("GIF plain-text rendering is not supported")
            continue
        if marker != 0x2C:
            raise ValueError("Invalid GIF block marker")
        if len(frames) >= MAX_FRAMES or (len(frames) + 1) * width * height > MAX_IMAGE_PIXELS:
            raise ValueError("GIF animation exceeds the decoded frame/pixel limit")
        left, top, fw, fh, flags = struct.unpack("<HHHHB", reader.read(9))
        checked_size((fw, fh))
        if left + fw > width or top + fh > height or flags & 24:
            raise ValueError("Invalid GIF frame rectangle or flags")
        local_palette = reader.read(3 * (2 << (flags & 7))) if flags & 128 else palette
        if not local_palette:
            raise ValueError("GIF frame has no color table")
        dispose, delay, transparent = control or (0, 0, None)
        if transparent is not None and transparent >= len(local_palette) // 3:
            raise ValueError("GIF transparent palette index is out of range")
        if canvas is None:
            bg = tuple(palette[background * 3:background * 3 + 3]) + (255,) if palette else (0, 0, 0, 255)
            canvas = new("RGBA", (width, height), (0, 0, 0, 0) if transparent is not None else bg)
        if previous_disposal is not None:
            method, box, previous, color = previous_disposal
            if method == 2:
                canvas.paste(color, box)
            elif method == 3:
                canvas = previous
        previous = canvas.copy() if dispose == 3 else None
        minimum = reader.byte()
        indices = _lzw(reader.blocks(), minimum, fw * fh)
        rows = list(range(fh))
        if flags & 64:
            rows = [y for start, step in ((0, 8), (4, 8), (2, 4), (1, 2)) for y in range(start, fh, step)]
        for sy, dy in enumerate(rows):
            for x in range(fw):
                index = indices[sy * fw + x]
                if index >= len(local_palette) // 3:
                    raise ValueError("GIF color index is out of range")
                if index != transparent:
                    offset = ((top + dy) * width + left + x) * 4
                    canvas._pixels[offset:offset + 4] = local_palette[index * 3:index * 3 + 3] + b"\xff"
        frame = canvas.copy()
        frame.info.update(duration=delay, disposal=dispose)
        frames.append(frame)
        background_color = ((0, 0, 0, 0) if transparent is not None else
                            tuple(palette[background * 3:background * 3 + 3]) + (255,) if palette else (0, 0, 0, 255))
        previous_disposal = dispose, (left, top, left + fw, top + fh), previous, background_color
        control = None


def encode(image, *, duration=0, loop=None):
    # Reset the dictionary before each 254 literal pixels. A fixed 9-bit code
    # width is valid GIF LZW, deterministic, and bounds the encoder's memory.
    palette_image = image.quantize(colors=255) if image.mode != "P" else image
    indices = bytearray(palette_image._pixels)
    transparent = None
    if "A" in image.mode:
        alpha = image.getchannel("A")._pixels
        if any(value < 128 for value in alpha):
            transparent = 255
            for index, value in enumerate(alpha):
                if value < 128:
                    indices[index] = transparent
    output = bytearray(b"GIF89a" + struct.pack("<HHBBB", *image.size, 0xF7, 0, 0))
    output.extend(bytes(palette_image._palette[:768]).ljust(768, b"\0"))
    if loop is not None:
        output.extend(b"\x21\xff\x0bNETSCAPE2.0\x03\x01" + struct.pack("<H", loop) + b"\0")
    output.extend(b"\x21\xf9\x04" + struct.pack("<BHB", int(transparent is not None), round(duration / 10), transparent or 0) + b"\0")
    output.extend(b"\x2c" + struct.pack("<HHHHB", 0, 0, *image.size, 0) + b"\x08")
    codes = []
    for offset in range(0, len(indices), 254):
        codes.append(256)
        codes.extend(indices[offset:offset + 254])
    codes.append(257)
    compressed, accumulator, bits = bytearray(), 0, 0
    for code in codes:
        accumulator |= code << bits
        bits += 9
        while bits >= 8:
            compressed.append(accumulator & 255)
            accumulator >>= 8
            bits -= 8
    if bits:
        compressed.append(accumulator)
    for offset in range(0, len(compressed), 255):
        block = compressed[offset:offset + 255]
        output.extend(bytes([len(block)]) + block)
    return bytes(output + b"\x00\x3b")
