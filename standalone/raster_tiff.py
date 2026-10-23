"""Bounded TIFF/BigTIFF strips, tiles and page directories using stdlib codecs."""

from __future__ import annotations

import math
import struct
import zlib

from standalone.raster import MAX_FRAMES, MAX_IMAGE_PIXELS, Raster, checked_size


_TYPES = {1: (1, "B"), 2: (1, None), 3: (2, "H"), 4: (4, "I"), 5: (8, None),
          6: (1, "b"), 7: (1, None), 8: (2, "h"), 9: (4, "i"), 10: (8, None),
          11: (4, "f"), 12: (8, "d"), 13: (4, "I"), 16: (8, "Q"), 17: (8, "q"), 18: (8, "Q")}
_REVERSE = bytes(int(f"{value:08b}"[::-1], 2) for value in range(256))


class _Reader:
    def __init__(self, data):
        self.data = data
        if len(data) < 8 or data[:2] not in (b"II", b"MM"):
            raise ValueError("Invalid or truncated TIFF header")
        self.endian = "<" if data[:2] == b"II" else ">"
        version = self.unpack("H", 2)[0]
        self.big = version == 43
        if version not in (42, 43):
            raise ValueError("Unsupported TIFF version")
        if self.big and self.unpack("HH", 4) != (8, 0):
            raise ValueError("Unsupported BigTIFF offset size or reserved field")
        self.offset = self.unpack("Q" if self.big else "I", 8 if self.big else 4)[0]

    def unpack(self, code, offset):
        size = struct.calcsize(self.endian + code)
        if offset < 0 or size > len(self.data) - offset:
            raise ValueError("Truncated TIFF directory")
        return struct.unpack_from(self.endian + code, self.data, offset)

    def span(self, offset, length):
        if offset < 0 or length < 0 or length > len(self.data) - offset:
            raise ValueError("TIFF field or pixel data exceeds the file")
        return self.data[offset:offset + length]

    def directory(self, offset):
        size, step, integer = (8, 20, "Q") if self.big else (2, 12, "I")
        count = self.unpack("Q" if self.big else "H", offset)[0]
        if count > 4096:
            raise ValueError("TIFF directory entry count exceeds the validation limit")
        self.span(offset + size, count * step + (8 if self.big else 4))
        tags, aggregate = {}, 0
        for index in range(count):
            entry = offset + size + index * step
            tag, kind = self.unpack("HH", entry)
            amount = self.unpack(integer, entry + 4)[0]
            if tag in tags or kind not in _TYPES:
                raise ValueError("Duplicate TIFF field or unsupported field type")
            unit, code = _TYPES[kind]
            aggregate += amount
            if amount > 1_000_000 or aggregate > 2_000_000:
                raise ValueError("TIFF metadata exceeds the validation limit")
            length, value_offset, inline = unit * amount, entry + (12 if self.big else 8), 8 if self.big else 4
            start = value_offset if length <= inline else self.unpack(integer, value_offset)[0]
            raw = self.span(start, length)
            tags[tag] = tuple(item[0] for item in struct.iter_unpack(self.endian + code, raw)) if code else raw
        following = self.unpack(integer, offset + size + count * step)[0]
        return tags, following


def _one(tags, tag, default=None):
    value = tags.get(tag, (default,))
    if not isinstance(value, tuple) or len(value) != 1 or type(value[0]) is not int:
        raise ValueError("Invalid TIFF scalar field " + str(tag))
    return value[0]


def _lzw(data, expected):
    offset, width, table, previous, result = 0, 9, [bytes((n,)) for n in range(256)] + [b"", b""], None, bytearray()
    while offset + width <= len(data) * 8:
        byte, shift = divmod(offset, 8)
        code = (int.from_bytes(data[byte:byte + 3], "big") >> (len(data[byte:byte + 3]) * 8 - shift - width)) & ((1 << width) - 1)
        offset += width
        if code == 256:
            width, table, previous = 9, [bytes((n,)) for n in range(256)] + [b"", b""], None
            continue
        if code == 257:
            if len(result) != expected:
                raise ValueError("Incorrect TIFF LZW decoded size")
            return bytes(result)
        if code < len(table):
            value = table[code]
        elif code == len(table) and previous is not None:
            value = previous + previous[:1]
        else:
            raise ValueError("Invalid TIFF LZW dictionary code")
        if not value or len(value) > expected - len(result):
            raise ValueError("TIFF LZW exceeds the decoded pixel size")
        result.extend(value)
        if previous is not None and len(table) < 4096:
            table.append(previous + value[:1])
            if len(table) == (1 << width) - 1 and width < 12:
                width += 1
        previous = value
    raise ValueError("Missing TIFF LZW end marker")


def _packbits(data, expected):
    result, offset = bytearray(), 0
    while offset < len(data):
        value = data[offset]
        offset += 1
        if value == 128:
            continue
        count = value + 1 if value < 128 else 257 - value
        needed = count if value < 128 else 1
        if needed > len(data) - offset or count > expected - len(result):
            raise ValueError("Truncated or oversized TIFF PackBits run")
        result.extend(data[offset:offset + count] if value < 128 else data[offset:offset + 1] * count)
        offset += needed
    if len(result) != expected:
        raise ValueError("Incorrect TIFF PackBits decoded size")
    return bytes(result)


def _decompress(data, compression, expected):
    if compression == 1:
        if len(data) != expected:
            raise ValueError("Incorrect uncompressed TIFF pixel size")
        return data
    if compression == 5:
        return _lzw(data, expected)
    if compression == 32773:
        return _packbits(data, expected)
    if compression in (8, 32946):
        inflater = zlib.decompressobj()
        raw = inflater.decompress(data, expected + 1)
        if len(raw) != expected or not inflater.eof or inflater.unconsumed_tail or inflater.unused_data:
            raise ValueError("Truncated or oversized TIFF Deflate pixels")
        return raw
    if compression == 34925:
        import lzma
        inflater = lzma.LZMADecompressor(memlimit=64 * 1024 * 1024)
        raw = inflater.decompress(data, max_length=expected + 1)
        if len(raw) != expected or not inflater.eof or inflater.unused_data:
            raise ValueError("Truncated or oversized TIFF LZMA pixels")
        return raw
    raise ValueError("Unsupported TIFF compression: " + str(compression))


def _samples(raw, width, rows, channels, depth, endian, predictor, sample_format):
    stride = (width * channels * depth + 7) // 8
    if predictor not in (1, 2) or predictor == 2 and (depth not in (8, 16, 32) or sample_format == 3):
        raise ValueError("Unsupported TIFF differencing predictor")
    if sample_format not in (1, 2, 3) or sample_format == 3 and depth not in (32, 64):
        raise ValueError("Unsupported TIFF sample format")
    if sample_format == 2 and depth not in (8, 16, 32):
        raise ValueError("Unsupported signed TIFF depth")
    maximum = (1 << depth) - 1
    byte_order = "little" if endian == "<" else "big"
    for y in range(rows):
        row = raw[y * stride:(y + 1) * stride]
        if depth < 8:
            values = [(row[n * depth // 8] >> (8 - depth - n * depth % 8)) & maximum for n in range(width * channels)]
        elif sample_format == 3:
            values = [value[0] for value in struct.iter_unpack(endian + ("f" if depth == 32 else "d"), row)]
        else:
            values = [int.from_bytes(row[n:n + depth // 8], byte_order) for n in range(0, len(row), depth // 8)]
        if predictor == 2:
            for n in range(channels, len(values)):
                values[n] = (values[n] + values[n - channels]) & maximum
        if sample_format == 2:
            values = [value - (1 << depth) if value >> (depth - 1) else value for value in values]
        yield values


def _orientation(image, orientation):
    if orientation == 1:
        return image
    width, height, bands = image.width, image.height, len(image.mode)
    out_width, out_height = (height, width) if orientation >= 5 else (width, height)
    output = bytearray(len(image._pixels))
    for y in range(height):
        for x in range(width):
            target_x, target_y = {
                2: (width - 1 - x, y), 3: (width - 1 - x, height - 1 - y),
                4: (x, height - 1 - y), 5: (y, x), 6: (height - 1 - y, x),
                7: (height - 1 - y, width - 1 - x), 8: (y, width - 1 - x),
            }[orientation]
            source, target = (y * width + x) * bands, (target_y * out_width + target_x) * bands
            output[target:target + bands] = image._pixels[source:source + bands]
    return Raster(image.mode, (out_width, out_height), output, format="TIFF", info=image.info)


def _pixel(values, photo, depth, sample_format, palette, extra):
    def scale(value, gray=False):
        if not isinstance(value, int) and not math.isfinite(value):
            return 255 if value > 0 else 0
        if sample_format != 1 or gray and depth > 8:
            return max(0, min(255, int(value)))
        return value >> (depth - 8) if depth > 8 else value * 255 // ((1 << depth) - 1)

    if photo in (0, 1):
        value = scale(values[0], gray=True)
        color = (255 - value if photo == 0 else value,)
    elif photo == 2:
        color = tuple(scale(value) for value in values[:3])
    elif photo == 3:
        index, length = values[0], len(palette) // 3
        if not 0 <= index < length:
            raise ValueError("TIFF palette index is out of range")
        color = tuple(palette[index + channel * length] >> 8 for channel in range(3))
    elif photo == 5:
        c, m, y, k = (scale(value) for value in values[:4])
        color = tuple(((255 - value) * (255 - k) + 127) // 255 for value in (c, m, y))
    elif photo == 6:
        y, cb, cr = values[:3]
        color = tuple(max(0, min(255, round(value))) for value in
                      (y + 1.402 * (cr - 128), y - .344136 * (cb - 128) - .714136 * (cr - 128), y + 1.772 * (cb - 128)))
    else:
        raise ValueError("Unsupported TIFF photometric interpretation")
    if extra:
        alpha = scale(values[-len(extra)])
        if extra[0] == 1:
            color = tuple(min(255, value * 255 // alpha) if alpha else 0 for value in color)
        color += (alpha,)
    return bytes(color)


def _segments(reader, tags, width, height, samples):
    planar = _one(tags, 284, 1)
    if planar not in (1, 2):
        raise ValueError("Invalid TIFF planar configuration")
    if 324 in tags or 325 in tags:
        tile_width, tile_height = _one(tags, 322), _one(tags, 323)
        checked_size((tile_width, tile_height))
        offsets, lengths = tags.get(324, ()), tags.get(325, ())
    else:
        tile_width, tile_height = width, min(height, _one(tags, 278, height))
        checked_size((tile_width, tile_height))
        offsets, lengths = tags.get(273, ()), tags.get(279, ())
    columns, rows = (width + tile_width - 1) // tile_width, (height + tile_height - 1) // tile_height
    per_plane = columns * rows
    if (not isinstance(offsets, tuple) or not isinstance(lengths, tuple) or len(offsets) != len(lengths)
            or len(offsets) != per_plane * (samples if planar == 2 else 1) or len(offsets) > 1_000_000):
        raise ValueError("Invalid TIFF strip or tile directory")
    for index, (offset, length) in enumerate(zip(offsets, lengths)):
        plane, cell = divmod(index, per_plane)
        row, column = divmod(cell, columns)
        x, y = column * tile_width, row * tile_height
        stored_rows = tile_height if 324 in tags else min(tile_height, height - y)
        yield x, y, tile_width, stored_rows, plane if planar == 2 else None, reader.span(offset, length)


def _page(reader, tags):
    width, height = checked_size((_one(tags, 256), _one(tags, 257)))
    photo, samples = _one(tags, 262, 1), _one(tags, 277, 1)
    compression, fill, predictor = _one(tags, 259, 1), _one(tags, 266, 1), _one(tags, 317, 1)
    depths, formats, extra = tags.get(258, (1,)), tags.get(339, (1,)), tags.get(338, ())
    if (not 1 <= samples <= 8 or len(depths) not in (1, samples) or len(set(depths)) != 1
            or depths[0] not in (1, 2, 4, 8, 16, 32, 64) or len(formats) not in (1, samples)
            or len(set(formats)) != 1 or fill not in (1, 2)):
        raise ValueError("Unsupported TIFF sample depths or fill order")
    depth, sample_format = depths[0], formats[0]
    base = {0: 1, 1: 1, 2: 3, 3: 1, 5: 4, 6: 3}.get(photo)
    if base is None or samples != base + len(extra) or len(extra) > 1 or extra and extra[0] not in (0, 1, 2):
        raise ValueError("Unsupported TIFF color channels or extra samples")
    # An unspecified RGB padding channel is discarded, matching the source API.
    alpha = extra if extra and extra[0] in (1, 2) else ()
    palette = tags.get(320, ())
    if photo == 3 and (depth > 16 or len(palette) != 3 * (1 << depth)):
        raise ValueError("Invalid TIFF palette")
    if photo == 5 and _one(tags, 332, 1) != 1:
        raise ValueError("Only CMYK TIFF separated inks are supported")
    mode = ("L" if base == 1 and photo != 3 else "RGB") + ("A" if alpha else "")
    image, bands = Raster(mode, (width, height), bytearray(width * height * len(mode))), len(mode)
    # Planar storage keeps only byte output per component, never Python objects per image sample.
    planar = _one(tags, 284, 1) == 2
    if planar and (depth != 8 or predictor not in (1, 2) or photo == 6):
        raise ValueError("Planar TIFF currently requires 8-bit channels")
    planes = [bytearray(width * height) for _ in range(samples)] if planar else None
    for x, y, sw, sh, plane, compressed in _segments(reader, tags, width, height, samples):
        channels = 1 if plane is not None else samples
        expected = sh * ((sw * channels * depth + 7) // 8)
        if expected > MAX_IMAGE_PIXELS * 8:
            raise ValueError("TIFF tile exceeds the decoded byte limit")
        if compression in (6, 7):
            from standalone import raster_jpeg
            if plane is not None or photo not in (0, 1, 2, 5, 6) or depth != 8 or alpha:
                raise ValueError("Unsupported JPEG-in-TIFF sample layout")
            tables = tags.get(347, b"")
            if tables:
                if not tables.startswith(b"\xff\xd8") or not tables.endswith(b"\xff\xd9"):
                    raise ValueError("Invalid TIFF JPEG tables")
                compressed = tables[:-2] + (compressed[2:] if compressed.startswith(b"\xff\xd8") else compressed)
            jw, jh, jm, pixels = raster_jpeg.decode(compressed, max_pixels=MAX_IMAGE_PIXELS)
            if (jw, jh) != (sw, sh) or jm != mode:
                raise ValueError("JPEG-in-TIFF dimensions or channels disagree")
            for sy in range(min(sh, height - y)):
                count = min(sw, width - x) * bands
                image._pixels[((y + sy) * width + x) * bands:((y + sy) * width + x) * bands + count] = pixels[sy * sw * bands:sy * sw * bands + count]
            continue
        if photo == 6 and compression not in (2, 3, 4):
            horizontal, vertical = tags.get(530, (2, 2))
            if (depth != 8 or predictor != 1 or channels != 3 or horizontal not in (1, 2, 4)
                    or vertical not in (1, 2, 4) or 529 in tags or 532 in tags):
                raise ValueError("Unsupported TIFF YCbCr layout or colorimetry")
            cells_x, cells_y = (sw + horizontal - 1) // horizontal, (sh + vertical - 1) // vertical
            cell_size = horizontal * vertical + 2
            expected = cells_x * cells_y * cell_size
            if expected > MAX_IMAGE_PIXELS * 4:
                raise ValueError("TIFF YCbCr tile exceeds the decoded byte limit")
            raw = _decompress(compressed, compression, expected)
            if fill == 2:
                raw = raw.translate(_REVERSE)
            for cell_y in range(cells_y):
                for cell_x in range(cells_x):
                    start = (cell_y * cells_x + cell_x) * cell_size
                    cb, cr = raw[start + cell_size - 2:start + cell_size]
                    for dy in range(vertical):
                        for dx in range(horizontal):
                            sx, sy = cell_x * horizontal + dx, cell_y * vertical + dy
                            if sx < min(sw, width - x) and sy < min(sh, height - y):
                                target = ((y + sy) * width + x + sx) * 3
                                image._pixels[target:target + 3] = _pixel((raw[start + dy * horizontal + dx], cb, cr), 6, 8, 1, (), ())
            continue
        if compression in (2, 3, 4):
            from standalone.raster_fax import decode as fax_decode
            if channels != 1 or depth != 1 or predictor != 1:
                raise ValueError("CCITT TIFF requires one bilevel channel")
            raw = fax_decode(compressed, sw, sh, compression, _one(tags, 292, 0), _one(tags, 293, 0), fill)
        else:
            raw = _decompress(compressed, compression, expected)
            if fill == 2:
                raw = raw.translate(_REVERSE)
        for sy, values in enumerate(_samples(raw, sw, sh, channels, depth, reader.endian, predictor, sample_format)):
            if y + sy >= height:
                continue
            for sx in range(min(sw, width - x)):
                if plane is not None:
                    planes[plane][(y + sy) * width + x + sx] = values[sx]
                else:
                    target = ((y + sy) * width + x + sx) * bands
                    image._pixels[target:target + bands] = _pixel(values[sx * channels:(sx + 1) * channels], photo, depth, sample_format, palette, alpha)
    if planes is not None:
        for index in range(width * height):
            image._pixels[index * bands:(index + 1) * bands] = _pixel([p[index] for p in planes], photo, depth, sample_format, palette, alpha)
    if 34675 in tags:
        image.info["icc_profile"] = bytes(tags[34675])
    orientation = _one(tags, 274, 1)
    if not 1 <= orientation <= 8:
        raise ValueError("Invalid TIFF orientation")
    image = _orientation(image, orientation)
    image.format = "TIFF"
    return image


def decode(data):
    reader, frames, visited, total = _Reader(data), [], set(), 0
    offset = reader.offset
    while offset:
        if offset in visited or len(frames) >= MAX_FRAMES:
            raise ValueError("Cyclic TIFF directories or excessive page count")
        visited.add(offset)
        tags, offset = reader.directory(offset)
        width, height = checked_size((_one(tags, 256), _one(tags, 257)))
        total += width * height
        if total > MAX_IMAGE_PIXELS:
            raise ValueError("TIFF aggregate pixels exceed the validation limit")
        frames.append(_page(reader, tags))
    if not frames:
        raise ValueError("TIFF has no image directory")
    image = frames[0].copy()
    image._frames = frames if len(frames) > 1 else None
    return image


def encode(image, compression="raw"):
    modes = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}
    if image.mode not in modes:
        image = image.convert("RGBA")
    compressed = {"raw": 1, "tiff_adobe_deflate": 8, "deflate": 8}.get(compression)
    if compressed is None:
        raise ValueError("Unsupported TIFF output compression")
    samples, raw = modes[image.mode], image.tobytes()
    payload = zlib.compress(raw, 6) if compressed == 8 else raw
    tags = {256: (4, (image.width,)), 257: (4, (image.height,)), 258: (3, (8,) * samples),
            259: (3, (compressed,)), 262: (3, (1 if image.mode.startswith("L") else 2,)),
            273: (4, (0,)), 277: (3, (samples,)), 278: (4, (image.height,)), 279: (4, (len(payload),)), 284: (3, (1,))}
    if image.mode.endswith("A"):
        tags[338] = (3, (2,))
    end = 8 + 2 + len(tags) * 12 + 4
    extra, fields = bytearray(), []
    for tag, (kind, values) in sorted(tags.items()):
        packed = struct.pack("<" + _TYPES[kind][1] * len(values), *values)
        value = packed.ljust(4, b"\0") if len(packed) <= 4 else struct.pack("<I", end + len(extra))
        if len(packed) > 4:
            extra.extend(packed)
        fields.append((tag, struct.pack("<HHI", tag, kind, len(values)) + value))
    pixels_offset = end + len(extra)
    fields = [struct.pack("<HHII", 273, 4, 1, pixels_offset) if tag == 273 else value for tag, value in fields]
    return b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", len(fields)) + b"".join(fields) + b"\0" * 4 + extra + payload
