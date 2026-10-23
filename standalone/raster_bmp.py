"""Windows/OS2 BMP palettes, bitfields and RLE4/RLE8 decoding."""

from __future__ import annotations

import struct

from standalone.raster import Raster, checked_size


def _rle(data, width, height, bits):
    pixels, x, y, offset = bytearray(width * height), 0, 0, 0
    while offset + 2 <= len(data):
        count, value = data[offset:offset + 2]
        offset += 2
        if count:
            values = [value if bits == 8 else (value >> 4 if i % 2 == 0 else value & 15) for i in range(count)]
        elif value == 0:
            x, y = 0, y + 1
            if y > height:
                raise ValueError("BMP RLE row exceeds the image")
            continue
        elif value == 1:
            return pixels
        elif value == 2:
            if offset + 2 > len(data):
                raise ValueError("Truncated BMP RLE delta")
            dx, dy = data[offset:offset + 2]
            offset += 2
            x, y = x + dx, y + dy
            if x > width or y >= height:
                raise ValueError("BMP RLE delta exceeds the image")
            continue
        else:
            count = value
            size = count if bits == 8 else (count + 1) // 2
            if offset + size + (size % 2) > len(data):
                raise ValueError("Truncated BMP RLE literals")
            values = [data[offset + i] if bits == 8 else
                      (data[offset + i // 2] >> (4 if i % 2 == 0 else 0)) & 15 for i in range(count)]
            offset += size + size % 2
        if y >= height or x + count > width:
            raise ValueError("BMP RLE run exceeds the image")
        pixels[y * width + x:y * width + x + count] = bytes(values)
        x += count
    raise ValueError("Missing BMP RLE end marker")


def _mask_info(mask, bits):
    if not mask:
        return 0, 0
    shift = (mask & -mask).bit_length() - 1
    maximum = mask >> shift
    if mask >> bits or maximum & (maximum + 1):
        raise ValueError("Invalid BMP channel mask")
    return shift, maximum


def decode(data):
    if len(data) < 26 or data[:2] != b"BM":
        raise ValueError("Invalid or truncated BMP header")
    declared, offset, header = struct.unpack_from("<IxxxxII", data, 2)
    if declared and declared > len(data) or offset > len(data) or header > len(data) - 14:
        raise ValueError("Truncated BMP data")
    if header == 12:
        width, signed_height, planes, bits = struct.unpack_from("<HHHH", data, 18)
        compression, used, entry_size = 0, 0, 3
    elif header >= 40:
        width, signed_height, planes, bits, compression = struct.unpack_from("<iiHHI", data, 18)
        used, entry_size = struct.unpack_from("<I", data, 46)[0], 4
    else:
        raise ValueError("Unsupported BMP DIB header")
    height = abs(signed_height)
    checked_size((width, height))
    if planes != 1 or bits not in (1, 4, 8, 16, 24, 32):
        raise ValueError("Unsupported BMP planes or pixel depth")
    if compression not in (0, 1, 2, 3, 6) or compression == 1 and bits != 8 or compression == 2 and bits != 4:
        raise ValueError("Unsupported BMP compression")
    if compression in (1, 2) and signed_height < 0:
        raise ValueError("RLE BMP cannot be stored top down")
    palette_start = 14 + header
    masks = (0x7C00, 0x03E0, 0x001F, 0) if bits == 16 else (0xFF0000, 0xFF00, 0xFF, 0)
    if compression in (3, 6):
        if bits not in (16, 32):
            raise ValueError("BMP bitfields require 16 or 32 bits")
        count = 4 if header >= 56 or compression == 6 else 3
        start = 54 if header >= 52 else palette_start
        if start + count * 4 > min(offset, len(data)):
            raise ValueError("Truncated BMP channel masks")
        masks = tuple(struct.unpack_from("<" + "I" * count, data, start))
        masks += (0,) * (4 - len(masks))
        if header == 40:
            palette_start += count * 4
        if not all(masks[:3]) or any(a & b for index, a in enumerate(masks) for b in masks[index + 1:]):
            raise ValueError("Overlapping or missing BMP channel masks")
    mask_info = [_mask_info(mask, bits) for mask in masks] if bits in (16, 32) else []
    palette = []
    if bits <= 8:
        count = used or 1 << bits
        if count > 1 << bits or palette_start + count * entry_size > offset:
            raise ValueError("Invalid BMP palette")
        for index in range(count):
            b, g, r = data[palette_start + index * entry_size:palette_start + index * entry_size + 3]
            palette.append(bytes((r, g, b)))
    elif palette_start > offset:
        raise ValueError("BMP pixels overlap the header")
    alpha = bool(masks[3])
    mode, bands = ("RGBA", 4) if alpha else ("RGB", 3)
    pixels = bytearray(width * height * bands)
    stride = ((width * bits + 31) // 32) * 4
    indices = _rle(data[offset:], width, height, bits) if compression in (1, 2) else None
    if indices is None and offset + stride * height > len(data):
        raise ValueError("Truncated BMP pixel stream")
    for stored_y in range(height):
        y = stored_y if signed_height < 0 else height - stored_y - 1
        row = offset + stored_y * stride
        for x in range(width):
            if bits <= 8:
                index = indices[stored_y * width + x] if indices is not None else (data[row + x * bits // 8] >> (8 - bits - x * bits % 8)) & ((1 << bits) - 1)
                if index >= len(palette):
                    raise ValueError("BMP palette index is out of range")
                value = palette[index]
            elif bits == 24:
                value = data[row + x * 3:row + x * 3 + 3][::-1]
            else:
                packed = int.from_bytes(data[row + x * (bits // 8):row + (x + 1) * (bits // 8)], "little")
                value = bytes(((packed >> shift) & maximum) * 255 // maximum for shift, maximum in mask_info[:bands])
            start = (y * width + x) * bands
            pixels[start:start + bands] = value
    return Raster(mode, (width, height), pixels, format="BMP")


def encode(image):
    image = image.convert("RGB")
    stride = (image.width * 3 + 3) & ~3
    pixels = bytearray(stride * image.height)
    for y in range(image.height):
        for x in range(image.width):
            source, target = (y * image.width + x) * 3, (image.height - y - 1) * stride + x * 3
            pixels[target:target + 3] = image._pixels[source:source + 3][::-1]
    header = b"BM" + struct.pack("<IHHI", 54 + len(pixels), 0, 0, 54)
    header += struct.pack("<IiiHHIIiiII", 40, *image.size, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    return header + pixels
