"""ICO/CUR PNG and DIB resources, with validation of every directory entry."""

from __future__ import annotations

import struct

from standalone.raster import MAX_FRAMES, MAX_IMAGE_PIXELS, Raster, checked_size


def _dib(data, width, height):
    from standalone import raster_bmp

    if len(data) < 12:
        raise ValueError("Truncated ICO bitmap header")
    header = int.from_bytes(data[:4], "little")
    if header == 12:
        stored_width, doubled, planes, depth = struct.unpack_from("<HHHH", data, 4)
        compression, used, entry_size = 0, 0, 3
    elif 40 <= header <= len(data):
        stored_width, doubled, planes, depth, compression = struct.unpack_from("<iiHHI", data, 4)
        used, entry_size = struct.unpack_from("<I", data, 32)[0], 4
    else:
        raise ValueError("Unsupported ICO bitmap header")
    if (stored_width, doubled, planes) != (width, height * 2, 1):
        raise ValueError("ICO directory and bitmap dimensions disagree")
    if depth not in (1, 4, 8, 16, 24, 32) or compression not in (0, 3, 6):
        raise ValueError("Unsupported ICO bitmap depth or compression")
    palette_count = used or (1 << depth if depth <= 8 else 0)
    if palette_count > (1 << depth if depth <= 8 else 256):
        raise ValueError("Invalid ICO bitmap palette")
    mask_bytes = (16 if compression == 6 else 12) if header == 40 and compression in (3, 6) else 0
    start = header + mask_bytes + palette_count * entry_size
    stride, and_stride = ((width * depth + 31) // 32) * 4, ((width + 31) // 32) * 4
    end = start + stride * height
    if end > len(data):
        raise ValueError("Truncated ICO bitmap pixels")
    dib = bytearray(data[:end])
    struct.pack_into("<H" if header == 12 else "<i", dib, 6 if header == 12 else 8, height)
    bitmap = b"BM" + struct.pack("<IHHI", 14 + len(dib), 0, 0, 14 + start) + dib
    bitmap_image = raster_bmp.decode(bitmap)
    image = bitmap_image.convert("RGBA")
    if depth == 32 and compression == 0:
        # Windows' 32-bit icon channel is authoritative even when every alpha is zero.
        for y in range(height):
            row = start + (height - 1 - y) * stride
            image._pixels[(y * width) * 4 + 3:(y + 1) * width * 4:4] = data[row + 3:row + width * 4:4]
    elif bitmap_image.mode != "RGBA":
        if end + and_stride * height > len(data):
            raise ValueError("Truncated ICO transparency mask")
        for y in range(height):
            row = end + (height - 1 - y) * and_stride
            for x in range(width):
                image._pixels[(y * width + x) * 4 + 3] = 0 if data[row + x // 8] & (128 >> (x % 8)) else 255
    return image, depth


def decode(data):
    from standalone import raster_png

    if len(data) < 6:
        raise ValueError("Truncated ICO directory")
    reserved, kind, count = struct.unpack_from("<HHH", data)
    if reserved or kind not in (1, 2) or not 1 <= count <= MAX_FRAMES or 6 + count * 16 > len(data):
        raise ValueError("Invalid or oversized ICO directory")
    best, sizes, total = None, set(), 0
    for index in range(count):
        w, h, colors, unused, planes, depth, length, offset = struct.unpack_from("<BBBBHHII", data, 6 + index * 16)
        width, height = w or 256, h or 256
        checked_size((width, height))
        total += width * height
        if total > MAX_IMAGE_PIXELS or unused or offset < 6 + count * 16 or not length or length > len(data) - offset:
            raise ValueError("Invalid ICO resource bounds")
        resource = data[offset:offset + length]
        if resource.startswith(raster_png.SIGNATURE):
            image = raster_png.decode(resource)
            if image.size != (width, height) or image.n_frames != 1:
                raise ValueError("ICO PNG dimensions or frame count are invalid")
            image, bits = image.convert("RGBA"), 32
        else:
            image, bits = _dib(resource, width, height)
        sizes.add((width, height))
        rank = (width * height, bits)
        if best is None or rank > best[0]:
            best = rank, image, (planes, depth)
    image = best[1]
    image.format = "ICO" if kind == 1 else "CUR"
    image.info["sizes"] = sizes
    if kind == 2:
        image.info["hotspot"] = best[2]
    return image


def encode(image, sizes=None):
    from standalone import raster_png

    sizes = sizes if sizes is not None else [(n, n) for n in (16, 24, 32, 48, 64, 128, 256)
                                            if n <= image.width and n <= image.height]
    sizes = sorted(set(tuple(size) for size in sizes))
    if not sizes or len(sizes) > MAX_FRAMES:
        raise ValueError("ICO output requires at least one bounded image size")
    entries, resources, offset = [], [], 6 + 16 * len(sizes)
    for size in sizes:
        width, height = checked_size(size)
        if width > 256 or height > 256 or width > image.width or height > image.height:
            raise ValueError("ICO dimensions must fit the source and be at most 256")
        scaled = image.convert("RGBA") if size == image.size else image.convert("RGBA").resize(size)
        payload = raster_png.encode(scaled)
        entries.append(struct.pack("<BBBBHHII", width % 256, height % 256, 0, 0, 1, 32, len(payload), offset))
        resources.append(payload)
        offset += len(payload)
    return struct.pack("<HHH", 0, 1, len(entries)) + b"".join(entries) + b"".join(resources)
