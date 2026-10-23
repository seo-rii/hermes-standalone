"""Bounded PNG/APNG decoding, including Adam7, palettes and 16-bit samples."""

from __future__ import annotations

import struct
import zlib

from standalone.raster import Raster, MAX_FRAMES, MAX_IMAGE_PIXELS, checked_size, new


SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
          (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def _chunks(data):
    offset, chunks = 8, []
    while offset < len(data):
        if len(data) - offset < 12:
            raise ValueError("Truncated PNG chunk")
        length = int.from_bytes(data[offset:offset + 4], "big")
        kind = data[offset + 4:offset + 8]
        end = offset + 8 + length
        if end + 4 > len(data):
            raise ValueError("Truncated PNG chunk payload")
        payload = data[offset + 8:end]
        if zlib.crc32(kind + payload) != int.from_bytes(data[end:end + 4], "big"):
            raise ValueError("PNG chunk checksum mismatch")
        if len(kind) != 4 or any(not (65 <= c <= 90 or 97 <= c <= 122) for c in kind):
            raise ValueError("Invalid PNG chunk name")
        if not kind[0] & 32 and kind not in (b"IHDR", b"PLTE", b"IDAT", b"IEND"):
            raise ValueError("Unknown critical PNG chunk")
        chunks.append((kind, payload))
        offset = end + 4
        if kind == b"IEND":
            if payload or offset != len(data):
                raise ValueError("Invalid PNG end marker")
            break
    if not chunks or chunks[0][0] != b"IHDR" or chunks[-1][0] != b"IEND":
        raise ValueError("Missing PNG header or end marker")
    for kind in (b"IHDR", b"PLTE", b"tRNS", b"acTL"):
        if sum(name == kind for name, _ in chunks) > 1:
            raise ValueError("Duplicate PNG control chunk")
    return chunks


def _header(payload):
    if len(payload) != 13:
        raise ValueError("Invalid PNG header size")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
    checked_size((width, height))
    depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
    if color not in depths or depth not in depths[color] or compression or filtering or interlace not in (0, 1):
        raise ValueError("Invalid PNG color or compression format")
    return width, height, depth, color, interlace


def _paeth(left, up, corner):
    prediction = left + up - corner
    distances = abs(prediction - left), abs(prediction - up), abs(prediction - corner)
    return (left, up, corner)[distances.index(min(distances))]


def _unfilter(row, previous, bpp, method):
    if not 0 <= method <= 4:
        raise ValueError("Invalid PNG row filter")
    result = bytearray(row)
    for index in range(len(result)):
        left = result[index - bpp] if index >= bpp else 0
        up = previous[index]
        corner = previous[index - bpp] if index >= bpp else 0
        prediction = (0, left, up, (left + up) // 2, _paeth(left, up, corner))[method]
        result[index] = (result[index] + prediction) & 255
    return result


def _samples(row, depth, count):
    if depth == 8:
        return row
    if depth == 16:
        return [int.from_bytes(row[index:index + 2], "big") for index in range(0, count * 2, 2)]
    mask = (1 << depth) - 1
    return [(row[index * depth // 8] >> (8 - depth - index * depth % 8)) & mask for index in range(count)]


def _decode_pixels(header, compressed, palette, transparency):
    width, height, depth, color, interlace = header
    channels = _CHANNELS[color]
    passes, expected = [], 0
    for x, y, dx, dy in _ADAM7 if interlace else ((0, 0, 1, 1),):
        pw, ph = max(0, (width - x + dx - 1) // dx), max(0, (height - y + dy - 1) // dy)
        if pw and ph:
            row_size = (pw * channels * depth + 7) // 8
            passes.append((x, y, dx, dy, pw, ph, row_size))
            expected += ph * (1 + row_size)
    inflater = zlib.decompressobj()
    raw = inflater.decompress(compressed, expected + 1)
    if len(raw) != expected or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
        raise ValueError("Invalid, truncated or oversized PNG pixel stream")
    if color == 3:
        if not palette or len(palette) % 3 or len(palette) > min(768, 3 * (1 << depth)):
            raise ValueError("Invalid PNG palette")
        if len(transparency) > len(palette) // 3:
            raise ValueError("Invalid PNG palette transparency")
    elif transparency and (color not in (0, 2) or len(transparency) != channels * 2):
        raise ValueError("Invalid PNG transparency key")
    mode = {0: "LA" if transparency else "L", 2: "RGBA" if transparency else "RGB",
            3: "RGBA" if transparency else "RGB", 4: "LA", 6: "RGBA"}[color]
    bands, maximum = len(mode), (1 << depth) - 1
    transparent = tuple(int.from_bytes(transparency[i:i + 2], "big") for i in range(0, len(transparency), 2)) if color in (0, 2) else ()
    pixels, offset = bytearray(width * height * bands), 0
    for x0, y0, dx, dy, pw, ph, row_size in passes:
        previous = bytearray(row_size)
        for py in range(ph):
            row = _unfilter(raw[offset + 1:offset + 1 + row_size], previous, max(1, (channels * depth + 7) // 8), raw[offset])
            previous, offset = row, offset + 1 + row_size
            samples = _samples(row, depth, pw * channels)
            for px in range(pw):
                values = tuple(samples[px * channels:(px + 1) * channels])
                if color == 3:
                    index = values[0]
                    if index >= len(palette) // 3:
                        raise ValueError("PNG palette index is out of range")
                    output = tuple(palette[index * 3:index * 3 + 3])
                    if transparency:
                        output += (transparency[index] if index < len(transparency) else 255,)
                else:
                    output = tuple(value >> 8 if depth == 16 else value * 255 // maximum for value in values)
                    if transparent:
                        output += (0 if values == transparent else 255,)
                start = ((y0 + py * dy) * width + x0 + px * dx) * bands
                pixels[start:start + bands] = bytes(output)
    return Raster(mode, (width, height), pixels, format="PNG")


def _animation(chunks, header, palette, transparency, declared):
    if len(declared) != 8:
        raise ValueError("Invalid APNG animation control")
    count, loops = struct.unpack(">II", declared)
    if not 1 <= count <= MAX_FRAMES:
        raise ValueError("APNG frame count exceeds the validation limit")
    controls, default_data, current, sequence = [], bytearray(), None, 0
    seen_idat = False
    for kind, payload in chunks:
        if kind == b"fcTL":
            if len(payload) != 26 or int.from_bytes(payload[:4], "big") != sequence:
                raise ValueError("Invalid APNG frame sequence")
            sequence += 1
            _, w, h, x, y, num, den, dispose, blend = struct.unpack(">IIIIIHHBB", payload)
            checked_size((w, h))
            if x + w > header[0] or y + h > header[1] or dispose > 2 or blend > 1:
                raise ValueError("Invalid APNG frame rectangle or operation")
            current = ((w, h, x, y, num, den, dispose, blend), bytearray(), not seen_idat)
            controls.append(current)
        elif kind == b"IDAT":
            seen_idat = True
            default_data.extend(payload)
            if current is not None:
                if len(controls) != 1 or not current[2]:
                    raise ValueError("Misplaced APNG default frame data")
                current[1].extend(payload)
        elif kind == b"fdAT":
            if current is None or current[2] or len(payload) < 4 or int.from_bytes(payload[:4], "big") != sequence:
                raise ValueError("Invalid APNG frame data sequence")
            sequence += 1
            current[1].extend(payload[4:])
    if len(controls) != count:
        raise ValueError("APNG frame count does not match the stream")
    separate_default = not controls[0][2]
    total = count + int(separate_default)
    if total > MAX_FRAMES or total * header[0] * header[1] > MAX_IMAGE_PIXELS:
        raise ValueError("APNG aggregate decoded pixels exceed the validation limit")
    canvas, frames = new("RGBA", header[:2]), []
    if separate_default:
        frames.append(_decode_pixels(header, default_data, palette, transparency).convert("RGBA"))
    elif controls[0][0][:4] != (*header[:2], 0, 0):
        raise ValueError("APNG default frame must cover the full canvas")
    for control, compressed, _ in controls:
        w, h, x, y, num, den, dispose, blend = control
        frame = _decode_pixels((w, h, *header[2:]), compressed, palette, transparency).convert("RGBA")
        previous = canvas.copy() if dispose == 2 else None
        if blend:
            canvas.alpha_composite(frame, (x, y))
        else:
            canvas.paste(frame, (x, y))
        output = canvas.copy()
        output.info.update(duration=1000 * num / (den or 100), loop=loops, disposal=dispose)
        frames.append(output)
        if dispose == 1:
            canvas.paste((0, 0, 0, 0), (x, y, x + w, y + h))
        elif dispose == 2:
            canvas = previous
    result = frames[0].copy()
    result.format, result._frames = "PNG", frames
    if separate_default:
        result.info["default_image"] = True
    return result


def decode(data):
    chunks = _chunks(data)
    header = _header(chunks[0][1])
    palette = next((value for kind, value in chunks if kind == b"PLTE"), b"")
    transparency = next((value for kind, value in chunks if kind == b"tRNS"), b"")
    first_data = next((i for i, (kind, _) in enumerate(chunks) if kind == b"IDAT"), None)
    if first_data is None or any(kind in (b"PLTE", b"tRNS", b"acTL") for kind, _ in chunks[first_data:]):
        raise ValueError("Missing or misplaced PNG control/data chunk")
    ended = False
    for kind, _ in chunks[first_data:]:
        if kind != b"IDAT":
            ended = True
        elif ended:
            raise ValueError("PNG data chunks must be consecutive")
    animation = next((value for kind, value in chunks if kind == b"acTL"), None)
    if animation is not None:
        return _animation(chunks, header, palette, transparency, animation)
    if any(kind in (b"fcTL", b"fdAT") for kind, _ in chunks):
        raise ValueError("APNG frames have no animation control")
    return _decode_pixels(header, b"".join(value for kind, value in chunks if kind == b"IDAT"), palette, transparency)


def encode(image, *, compress_level=6, optimize=False):
    if not 0 <= compress_level <= 9:
        raise ValueError("PNG compression level must be in 0..9")
    color = {"L": 0, "RGB": 2, "P": 3, "LA": 4, "RGBA": 6}[image.mode]
    output = bytearray(SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", *image.size, 8, color, 0, 0, 0)))
    if image.mode == "P":
        if image._palette is None:
            raise ValueError("Indexed PNG requires a palette")
        output.extend(chunk(b"PLTE", bytes(image._palette)))
        if "transparency" in image.info:
            index = image.info["transparency"]
            output.extend(chunk(b"tRNS", bytes([255] * index + [0])))
    compressor, rows = zlib.compressobj(9 if optimize else compress_level), bytearray()
    stride, bpp = image.width * image.bands, image.bands
    previous = bytes(stride)
    for offset in range(0, len(image._pixels), stride):
        row = image._pixels[offset:offset + stride]
        candidates = [bytes(row)]
        for method in (1, 2, 3, 4):
            filtered = bytearray(stride)
            for index, value in enumerate(row):
                left = row[index - bpp] if index >= bpp else 0
                up = previous[index]
                corner = previous[index - bpp] if index >= bpp else 0
                predictor = (left, up, (left + up) // 2, _paeth(left, up, corner))[method - 1]
                filtered[index] = (value - predictor) & 255
            candidates.append(bytes(filtered))
        selected = min(range(5), key=lambda index: sum(min(v, 256 - v) for v in candidates[index]))
        rows.extend(compressor.compress(bytes([selected]) + candidates[selected]))
        previous = row
    rows.extend(compressor.flush())
    output.extend(chunk(b"IDAT", bytes(rows)))
    output.extend(chunk(b"IEND", b""))
    return bytes(output)
