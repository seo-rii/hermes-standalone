"""Pure Python WebP lossless pixels and deterministic lossless encoding.

The VP8L implementation follows Google's WebP Lossless Bitstream Specification
and WebP Container Specification, including all four reversible transforms,
spatial Huffman groups, the color cache, and overlapping LZ77 references:
https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification
https://developers.google.com/speed/webp/docs/riff_container

The animation integer compositor follows libwebp rounding. Copyright 2014
Google Inc.; BSD-3-Clause terms and immutable source provenance are shipped in
licenses/libwebp-COPYING and webp-provenance.json.

Lossy VP8 keyframes use the adjacent pure Python decoder. Both kinds of frames
support animation and alpha. No external decoder or native extension is used.
"""

from __future__ import annotations

from array import array
from collections import Counter
import heapq
import struct


class WebPError(ValueError):
    """Malformed, unsupported, or resource-exceeding WebP data."""


DEFAULT_MAX_PIXELS = 40_000_000
_LENGTH_ORDER = (17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
# The format's normative two-dimensional distance numbering (section 5.2.2).
_DISTANCES = (
    (0, 1), (1, 0), (1, 1), (-1, 1), (0, 2), (2, 0), (1, 2), (-1, 2),
    (2, 1), (-2, 1), (2, 2), (-2, 2), (0, 3), (3, 0), (1, 3), (-1, 3),
    (3, 1), (-3, 1), (2, 3), (-2, 3), (3, 2), (-3, 2), (0, 4), (4, 0),
    (1, 4), (-1, 4), (4, 1), (-4, 1), (3, 3), (-3, 3), (2, 4), (-2, 4),
    (4, 2), (-4, 2), (0, 5), (3, 4), (-3, 4), (4, 3), (-4, 3), (5, 0),
    (1, 5), (-1, 5), (5, 1), (-5, 1), (2, 5), (-2, 5), (5, 2), (-5, 2),
    (4, 4), (-4, 4), (3, 5), (-3, 5), (5, 3), (-5, 3), (0, 6), (6, 0),
    (1, 6), (-1, 6), (6, 1), (-6, 1), (2, 6), (-2, 6), (6, 2), (-6, 2),
    (4, 5), (-4, 5), (5, 4), (-5, 4), (3, 6), (-3, 6), (6, 3), (-6, 3),
    (0, 7), (7, 0), (1, 7), (-1, 7), (5, 5), (-5, 5), (7, 1), (-7, 1),
    (4, 6), (-4, 6), (6, 4), (-6, 4), (2, 7), (-2, 7), (7, 2), (-7, 2),
    (3, 7), (-3, 7), (7, 3), (-7, 3), (5, 6), (-5, 6), (6, 5), (-6, 5),
    (8, 0), (4, 7), (-4, 7), (7, 4), (-7, 4), (8, 1), (8, 2), (6, 6),
    (-6, 6), (8, 3), (5, 7), (-5, 7), (7, 5), (-7, 5), (8, 4), (6, 7),
    (-6, 7), (7, 6), (-7, 6), (8, 5), (7, 7), (-7, 7), (8, 6), (8, 7),
)


def _dimensions(width, height, maximum):
    if (type(width) is not int or type(height) is not int
            or not 1 <= width <= 16384 or not 1 <= height <= 16384
            or type(maximum) is not int or maximum < 1 or width * height > maximum):
        raise WebPError("WebP dimensions exceed the permitted pixel limit")


class _Bits:
    def __init__(self, data):
        self.data = data
        self.position = 0

    def read(self, count):
        if count < 0 or self.position + count > len(self.data) * 8:
            raise WebPError("Truncated WebP bitstream")
        start, shift = divmod(self.position, 8)
        self.position += count
        needed = (shift + count + 7) // 8
        return (int.from_bytes(self.data[start:start + needed], "little") >> shift) & ((1 << count) - 1)

    def finish(self):
        # The image has an implicit end at its last decoded pixel. Libwebp
        # permits unused bytes after that point; RIFF still bounds the payload.
        if self.position > len(self.data) * 8:
            raise WebPError("Truncated WebP image")


class _Tree:
    def __init__(self, lengths):
        symbols = [(size, symbol) for symbol, size in enumerate(lengths) if size]
        if not symbols or any(size > 15 for size, _ in symbols):
            raise WebPError("Invalid WebP Huffman tree")
        self.single = symbols[0][1] if len(symbols) == 1 else None
        if self.single is not None:
            if symbols[0][0] != 1:
                raise WebPError("Invalid WebP single-symbol tree")
            self.tables, self.maximum = {}, 0
            return
        counts = Counter(size for size, _ in symbols)
        space = 1
        for size in range(1, 16):
            space = space * 2 - counts[size]
            if space < 0:
                raise WebPError("Oversubscribed WebP Huffman tree")
        if space:
            raise WebPError("Incomplete WebP Huffman tree")
        self.tables = {}
        code, previous = 0, 0
        for size, symbol in sorted(symbols):
            code <<= size - previous
            self.tables.setdefault(size, {})[code] = symbol
            code, previous = code + 1, size
        self.maximum = max(counts)

    def read(self, bits):
        if self.single is not None:
            return self.single
        code = 0
        for size in range(1, self.maximum + 1):
            code = (code << 1) | bits.read(1)
            symbol = self.tables.get(size, {}).get(code)
            if symbol is not None:
                return symbol
        raise WebPError("Invalid WebP Huffman symbol")


def _read_tree(bits, alphabet):
    lengths = [0] * alphabet
    if bits.read(1):
        count = bits.read(1) + 1
        first = bits.read(8 if bits.read(1) else 1)
        if first >= alphabet:
            raise WebPError("WebP symbol exceeds its alphabet")
        lengths[first] = 1
        if count == 2:
            second = bits.read(8)
            if second >= alphabet:
                raise WebPError("WebP symbol exceeds its alphabet")
            lengths[second] = 1
        return _Tree(lengths)
    code_lengths = [0] * 19
    for symbol in _LENGTH_ORDER[:bits.read(4) + 4]:
        code_lengths[symbol] = bits.read(3)
    tree = _Tree(code_lengths)
    maximum = alphabet
    if bits.read(1):
        maximum = 2 + bits.read(2 + 2 * bits.read(3))
        if maximum > alphabet:
            raise WebPError("WebP code length count exceeds its alphabet")
    position, previous = 0, 8
    for _ in range(maximum):
        if position == alphabet:
            break
        symbol = tree.read(bits)
        if symbol < 16:
            lengths[position] = symbol
            position += 1
            if symbol:
                previous = symbol
        else:
            count = bits.read((2, 3, 7)[symbol - 16]) + (3, 3, 11)[symbol - 16]
            if position + count > alphabet:
                raise WebPError("WebP code length repeat exceeds its alphabet")
            lengths[position:position + count] = [previous if symbol == 16 else 0] * count
            position += count
    return _Tree(lengths)


def _prefix(bits, symbol):
    if symbol < 4:
        return symbol + 1
    count = (symbol - 2) // 2
    return ((2 + (symbol & 1)) << count) + bits.read(count) + 1


def _image(bits, width, height, *, main=False):
    has_cache = bits.read(1)
    cache_bits = bits.read(4) if has_cache else 0
    if has_cache and not 1 <= cache_bits <= 11:
        raise WebPError("Invalid WebP color cache size")
    cache = array("I", [0]) * (1 << cache_bits) if cache_bits else None
    codes, prefix_bits, prefix_width = None, 0, 0
    group_count = 1
    if main and bits.read(1):
        prefix_bits = bits.read(3) + 2
        prefix_width = (width + (1 << prefix_bits) - 1) >> prefix_bits
        prefix_height = (height + (1 << prefix_bits) - 1) >> prefix_bits
        codes = _image(bits, prefix_width, prefix_height)
        codes = array("H", ((value >> 8) & 65535 for value in codes))
        group_count = max(codes) + 1
    used = set(codes) if codes is not None else {0}
    groups = {}
    # Unreferenced groups are still parsed and validated, without retaining
    # their tables. This avoids a 16-bit sparse group index expanding memory.
    for group in range(group_count):
        trees = tuple(_read_tree(bits, alphabet) for alphabet in
                      (280 + (len(cache) if cache is not None else 0), 256, 256, 256, 40))
        if group in used:
            groups[group] = trees
    result = array("I")
    total = width * height
    while len(result) < total:
        position = len(result)
        if codes is None:
            trees = groups[0]
        else:
            y, x = divmod(position, width)
            trees = groups[codes[(y >> prefix_bits) * prefix_width + (x >> prefix_bits)]]
        symbol = trees[0].read(bits)
        if symbol < 256:
            red, blue, alpha = (tree.read(bits) for tree in trees[1:4])
            result.append((alpha << 24) | (red << 16) | (symbol << 8) | blue)
        elif symbol < 280:
            length = _prefix(bits, symbol - 256)
            distance_code = _prefix(bits, trees[4].read(bits))
            if distance_code <= 120:
                dx, dy = _DISTANCES[distance_code - 1]
                distance = max(1, dx + dy * width)
            else:
                distance = distance_code - 120
            if distance > position or position + length > total:
                raise WebPError("WebP backward reference exceeds decoded pixels")
            for _ in range(length):
                result.append(result[len(result) - distance])
        else:
            if cache is None or symbol - 280 >= len(cache):
                raise WebPError("Invalid WebP color cache index")
            result.append(cache[symbol - 280])
        if cache is not None:
            for index in range(position, len(result)):
                color = result[index]
                slot = ((0x1e35a7bd * color) & 0xffffffff) >> (32 - cache_bits)
                cache[slot] = color
    return result


def _channels(value):
    return tuple((value >> shift) & 255 for shift in (0, 8, 16, 24))


def _pack(values):
    return sum((value & 255) << shift for value, shift in zip(values, (0, 8, 16, 24)))


def _add(left, right):
    return (((left & 0xff00ff) + (right & 0xff00ff)) & 0xff00ff
            | ((left & 0xff00ff00) + (right & 0xff00ff00)) & 0xff00ff00)


def _average(left, right):
    return (left & right) + (((left ^ right) & 0xfefefefe) >> 1)


def _predict(mode, left, top, top_left, top_right):
    if mode < 5:
        return (0xff000000, left, top, top_right, top_left)[mode]
    if mode == 5:
        return _average(_average(left, top_right), top)
    if mode == 6:
        return _average(left, top_left)
    if mode == 7:
        return _average(left, top)
    if mode == 8:
        return _average(top_left, top)
    if mode == 9:
        return _average(top, top_right)
    if mode == 10:
        return _average(_average(left, top_left), _average(top, top_right))
    if mode == 11:
        a, b, c = _channels(left), _channels(top), _channels(top_left)
        left_error = sum(abs(bv - cv) for bv, cv in zip(b, c))
        top_error = sum(abs(av - cv) for av, cv in zip(a, c))
        return left if left_error < top_error else top
    if mode == 12:
        return _pack(max(0, min(255, a + b - c)) for a, b, c in
                     zip(_channels(left), _channels(top), _channels(top_left)))
    if mode == 13:
        average = _channels(_average(left, top))
        # C integer division truncates toward zero, also for negative deltas.
        return _pack(max(0, min(255, a + (a - b) // 2 if a >= b else a - (b - a) // 2))
                     for a, b in zip(average, _channels(top_left)))
    raise WebPError("Invalid WebP spatial prediction mode")


def _inverse_predict(pixels, width, height, size_bits, modes):
    block_width = (width + (1 << size_bits) - 1) >> size_bits
    if any((value >> 8) & 255 > 13 for value in modes):
        raise WebPError("Invalid WebP spatial prediction mode")
    for index in range(len(pixels)):
        y, x = divmod(index, width)
        if not y:
            prediction = pixels[index - 1] if x else 0xff000000
        elif not x:
            prediction = pixels[index - width]
        else:
            mode = (modes[(y >> size_bits) * block_width + (x >> size_bits)] >> 8) & 255
            prediction = _predict(mode, pixels[index - 1], pixels[index - width],
                                  pixels[index - width - 1], pixels[index - width + 1])
        pixels[index] = _add(pixels[index], prediction)
    return pixels


def _signed(value):
    return value if value < 128 else value - 256


def _inverse_color(pixels, width, height, size_bits, colors):
    block_width = (width + (1 << size_bits) - 1) >> size_bits
    for index, value in enumerate(pixels):
        y, x = divmod(index, width)
        color = colors[(y >> size_bits) * block_width + (x >> size_bits)]
        blue, green, red, alpha = _channels(value)
        red = (red + ((_signed(color & 255) * _signed(green)) >> 5)) & 255
        blue += ((_signed((color >> 8) & 255) * _signed(green)) >> 5)
        blue += ((_signed((color >> 16) & 255) * _signed(red)) >> 5)
        pixels[index] = (alpha << 24) | (red << 16) | (green << 8) | (blue & 255)
    return pixels


def _lossless(data, maximum):
    if len(data) < 5 or data[0] != 0x2f:
        raise WebPError("Missing WebP lossless signature")
    bits = _Bits(data[1:])
    width, height = bits.read(14) + 1, bits.read(14) + 1
    _dimensions(width, height, maximum)
    bits.read(1)  # Alpha hint does not control decoding.
    if bits.read(3):
        raise WebPError("Unsupported WebP lossless version")
    transforms, seen = [], set()
    current_width = width
    while bits.read(1):
        kind = bits.read(2)
        if kind in seen:
            raise WebPError("Duplicate WebP lossless transform")
        seen.add(kind)
        original_width = current_width
        if kind in (0, 1):
            size_bits = bits.read(3) + 2
            sub_width = (current_width + (1 << size_bits) - 1) >> size_bits
            sub_height = (height + (1 << size_bits) - 1) >> size_bits
            transform = (size_bits, _image(bits, sub_width, sub_height))
        elif kind == 3:
            count = bits.read(8) + 1
            palette = _image(bits, count, 1)
            for index in range(1, count):
                palette[index] = _add(palette[index], palette[index - 1])
            packing = 3 if count <= 2 else 2 if count <= 4 else 1 if count <= 16 else 0
            transform = packing, palette
            current_width = (current_width + (1 << packing) - 1) >> packing
        else:
            transform = None
        transforms.append((kind, original_width, transform))
    pixels = _image(bits, current_width, height, main=True)
    for kind, target_width, transform in reversed(transforms):
        if kind == 0:
            pixels = _inverse_predict(pixels, current_width, height, *transform)
        elif kind == 1:
            pixels = _inverse_color(pixels, current_width, height, *transform)
        elif kind == 2:
            for index, value in enumerate(pixels):
                green = (value >> 8) & 255
                pixels[index] = _add(value, (green << 16) | green)
        else:
            packing, palette = transform
            output = array("I")
            per_byte, index_bits = 1 << packing, 8 >> packing
            for y in range(height):
                for x in range(target_width):
                    value = (pixels[y * current_width + x // per_byte] >> 8) & 255
                    entry = (value >> ((x % per_byte) * index_bits)) & ((1 << index_bits) - 1)
                    output.append(palette[entry] if entry < len(palette) else 0)
            pixels, current_width = output, target_width
    bits.finish()
    return width, height, pixels


def _chunks(data):
    offset = 0
    while offset < len(data):
        if offset + 8 > len(data):
            raise WebPError("Truncated WebP chunk header")
        kind = data[offset:offset + 4]
        length = int.from_bytes(data[offset + 4:offset + 8], "little")
        start, end = offset + 8, offset + 8 + length
        if end + (length & 1) > len(data):
            raise WebPError("Truncated WebP chunk")
        if length & 1 and data[end]:
            raise WebPError("Invalid WebP RIFF padding")
        yield kind, data[start:end]
        offset = end + (length & 1)


def _pixel_bytes(pixels):
    alpha = any(value >> 24 != 255 for value in pixels)
    output = bytearray(len(pixels) * (4 if alpha else 3))
    step = 4 if alpha else 3
    for index, value in enumerate(pixels):
        start = index * step
        output[start:start + step] = bytes(((value >> 16) & 255, (value >> 8) & 255, value & 255,
                                          value >> 24))[:step]
    return "RGBA" if alpha else "RGB", bytes(output)


def _alpha(payload, width, height, maximum):
    if not payload:
        raise WebPError("Missing WebP alpha header")
    compression, filtering = payload[0] & 3, (payload[0] >> 2) & 3
    if compression == 0:
        if len(payload) != 1 + width * height:
            raise WebPError("WebP alpha plane size differs from its pixels")
        pixels = bytearray(payload[1:])
    elif compression == 1:
        header = ((width - 1) | ((height - 1) << 14) | (1 << 28)).to_bytes(4, "little")
        _, _, colors = _lossless(b"\x2f" + header + payload[1:], maximum)
        pixels = bytearray((value >> 8) & 255 for value in colors)
    else:
        raise WebPError("Unsupported WebP alpha compression method")
    if filtering:
        for index in range(len(pixels)):
            row, column = divmod(index, width)
            if not row:
                prediction = pixels[index - 1] if column else 0
            elif not column:
                prediction = pixels[index - width]
            elif filtering == 1:
                prediction = pixels[index - 1]
            elif filtering == 2:
                prediction = pixels[index - width]
            else:
                prediction = max(0, min(255, pixels[index - 1] + pixels[index - width] - pixels[index - width - 1]))
            pixels[index] = (pixels[index] + prediction) & 255
    return pixels


def _lossy(payload, alpha, maximum):
    from standalone import raster_vp8
    try:
        width, height, _, pixels = raster_vp8.decode(payload, max_pixels=maximum)
    except raster_vp8.VP8Error as exc:
        raise WebPError(str(exc)) from exc
    alphas = _alpha(alpha, width, height, maximum) if alpha is not None else None
    values = array("I")
    for index in range(width * height):
        red, green, blue = pixels[index * 3:index * 3 + 3]
        opacity = alphas[index] if alphas is not None else 255
        values.append((opacity << 24) | (red << 16) | (green << 8) | blue)
    return width, height, values


def _blend(source, target):
    alpha = source >> 24
    if alpha == 255:
        return source
    if alpha == 0:
        return target
    # Match libwebp's integer animation compositor, avoiding cumulative
    # per-frame rounding differences from a floating-point implementation.
    target_weight = ((target >> 24) * (256 - alpha)) >> 8
    output_alpha = alpha + target_weight
    reciprocal = (1 << 24) // output_alpha
    channels = [((((source >> shift) & 255) * alpha
                  + ((target >> shift) & 255) * target_weight) * reciprocal) >> 24
                for shift in (0, 8, 16)]
    return _pack((*channels, output_alpha))


def _frame(payload, canvas_size, maximum):
    if len(payload) < 16:
        raise WebPError("Truncated WebP animation frame header")
    x = 2 * int.from_bytes(payload[0:3], "little")
    y = 2 * int.from_bytes(payload[3:6], "little")
    width = 1 + int.from_bytes(payload[6:9], "little")
    height = 1 + int.from_bytes(payload[9:12], "little")
    duration = int.from_bytes(payload[12:15], "little")
    flags = payload[15]
    if flags & 0xfc or x + width > canvas_size[0] or y + height > canvas_size[1]:
        raise WebPError("WebP animation frame exceeds its canvas")
    _dimensions(width, height, maximum)
    decoded, alpha = None, None
    for kind, data in _chunks(payload[16:]):
        if kind == b"VP8L":
            if decoded is not None or alpha is not None:
                raise WebPError("Multiple bitstreams in a WebP animation frame")
            decoded = _lossless(data, maximum)
        elif kind == b"VP8 ":
            if decoded is not None:
                raise WebPError("Multiple bitstreams in a WebP animation frame")
            decoded = _lossy(data, alpha, maximum)
        elif kind == b"ALPH":
            if alpha is not None or decoded is not None:
                raise WebPError("Duplicate or misplaced WebP alpha plane")
            alpha = data
        elif kind in (b"VP8X", b"ANIM", b"ANMF"):
            raise WebPError("Nested WebP animation container")
    if decoded is None or decoded[:2] != (width, height):
        raise WebPError("WebP animation frame dimensions differ from its pixels")
    return x, y, width, height, duration, flags, decoded[2]


def decode_frames(data: bytes, max_pixels: int = DEFAULT_MAX_PIXELS, max_frames: int = 100):
    """Return canvas width/height, composed (mode, pixels, duration) frames, info.

    Every frame is decoded. Both individual and aggregate displayed pixel counts
    are bounded before allocating a canvas or decoding the next frame. The
    animation background is transparent, as in libwebp's animation decoder; its
    declared background remains available in ``info`` for callers.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise WebPError("Missing WebP RIFF signature")
    if int.from_bytes(data[4:8], "little") + 8 != len(data):
        raise WebPError("WebP RIFF length does not match its data")
    if type(max_frames) is not int or max_frames < 1:
        raise WebPError("WebP frame limit must be a positive integer")
    decoded, canvas, buffer, alpha = None, None, None, None
    animated, animation_header, previous, previous_keyframe = False, False, None, False
    frames, info = [], {}
    for kind, payload in _chunks(data[12:]):
        if kind == b"VP8X":
            if canvas is not None or decoded is not None or len(payload) != 10:
                raise WebPError("Invalid WebP extended header")
            if payload[0] & 0xc1 or payload[1:4] != b"\x00\x00\x00":
                raise WebPError("Reserved WebP extended header bits are set")
            animated = bool(payload[0] & 2)
            canvas = (int.from_bytes(payload[4:7], "little") + 1,
                      int.from_bytes(payload[7:10], "little") + 1)
            _dimensions(*canvas, max_pixels)
        elif kind == b"VP8L":
            if decoded is not None or animated or alpha is not None:
                raise WebPError("Multiple WebP image chunks")
            decoded = _lossless(payload, max_pixels)
        elif kind == b"VP8 ":
            if decoded is not None or animated:
                raise WebPError("Multiple WebP image chunks")
            decoded = _lossy(payload, alpha, max_pixels)
        elif kind == b"ALPH":
            if alpha is not None or decoded is not None or animated or canvas is None:
                raise WebPError("Duplicate or misplaced WebP alpha plane")
            alpha = payload
        elif kind == b"ANIM":
            if not animated:
                continue
            if animation_header or frames or len(payload) != 6:
                raise WebPError("Invalid WebP animation header")
            background = int.from_bytes(payload[:4], "little")
            info.update(background=((background >> 16) & 255, (background >> 8) & 255,
                                    background & 255, background >> 24),
                        loop=int.from_bytes(payload[4:], "little"))
            animation_header = True
            buffer = array("I", [0]) * (canvas[0] * canvas[1])
        elif kind == b"ANMF":
            if not animated or not animation_header:
                raise WebPError("WebP animation frame precedes its headers")
            if len(frames) >= max_frames or (len(frames) + 1) * canvas[0] * canvas[1] > max_pixels:
                raise WebPError("WebP animation exceeds its frame or aggregate pixel limit")
            x, y, width, height, duration, flags, pixels = _frame(payload, canvas, max_pixels)
            keyframe = previous is None or ((width, height) == canvas
                        and (flags & 2 or all(value >> 24 == 255 for value in pixels)))
            if previous is not None and previous[4] & 1:
                keyframe = keyframe or previous[2:4] == canvas or previous_keyframe
            if keyframe:
                buffer = array("I", [0]) * (canvas[0] * canvas[1])
            elif previous is not None and previous[4] & 1:
                px, py, pw, ph, _ = previous
                for row in range(py, py + ph):
                    start = row * canvas[0] + px
                    buffer[start:start + pw] = array("I", [0]) * pw
            for row in range(height):
                start = (y + row) * canvas[0] + x
                source = pixels[row * width:(row + 1) * width]
                if flags & 2 or keyframe:
                    buffer[start:start + width] = source
                else:
                    for column, value in enumerate(source):
                        # Blending over a disposed rectangle is an exact copy,
                        # including invisible source RGB values. The native
                        # compositor likewise skips integer alpha arithmetic.
                        disposed = previous is not None and previous[4] & 1
                        if (disposed and previous[0] <= x + column < previous[0] + previous[2]
                                and previous[1] <= y + row < previous[1] + previous[3]):
                            buffer[start + column] = value
                        else:
                            buffer[start + column] = _blend(value, buffer[start + column])
            mode, output = _pixel_bytes(buffer)
            frames.append((mode, output, duration))
            previous = x, y, width, height, flags
            previous_keyframe = keyframe
        elif kind in (b"ICCP", b"EXIF", b"XMP "):
            info.setdefault({b"ICCP": "icc_profile", b"EXIF": "exif", b"XMP ": "xmp"}[kind], payload)
    if animated:
        if not frames:
            raise WebPError("WebP animation has no decoded frames")
        return *canvas, frames, info
    if decoded is None:
        raise WebPError("WebP has no decoded image")
    width, height, pixels = decoded
    if canvas is not None and canvas != (width, height):
        raise WebPError("WebP canvas and decoded dimensions differ")
    mode, output = _pixel_bytes(pixels)
    return width, height, [(mode, output, 0)], info


def decode(data: bytes, max_pixels: int = DEFAULT_MAX_PIXELS):
    """Decode the first image, validating every frame in an animation."""
    width, height, frames, _ = decode_frames(data, max_pixels=max_pixels)
    mode, pixels, _ = frames[0]
    return width, height, mode, pixels


class _Writer:
    def __init__(self):
        self.data = bytearray()
        self.value = self.count = 0

    def write(self, value, count):
        self.value |= value << self.count
        self.count += count
        while self.count >= 8:
            self.data.append(self.value & 255)
            self.value >>= 8
            self.count -= 8

    def finish(self):
        if self.count:
            self.data.append(self.value & 255)
            self.value = self.count = 0
        return bytes(self.data)


def _codes(lengths):
    symbols = sorted((size, symbol) for symbol, size in enumerate(lengths) if size)
    if len(symbols) == 1:
        return {symbols[0][1]: (0, 0)}
    result, code, previous = {}, 0, 0
    for size, symbol in symbols:
        code <<= size - previous
        reverse = int(f"{code:0{size}b}"[::-1], 2)
        result[symbol] = reverse, size
        code, previous = code + 1, size
    return result


def _lengths(counts, alphabet):
    # Stable Huffman tie-breaking makes byte output reproducible.
    leaves = sorted(counts)
    if len(leaves) == 1:
        result = [0] * alphabet
        result[leaves[0]] = 1
        return result
    queue = [(counts[symbol], symbol, symbol) for symbol in leaves]
    heapq.heapify(queue)
    serial = alphabet
    while len(queue) > 1:
        af, _, a = heapq.heappop(queue)
        bf, _, b = heapq.heappop(queue)
        heapq.heappush(queue, (af + bf, serial, (a, b)))
        serial += 1
    result = [0] * alphabet
    pending = [(queue[0][2], 0)]
    while pending:
        node, depth = pending.pop()
        if isinstance(node, int):
            result[node] = max(1, depth)
        else:
            pending.extend((child, depth + 1) for child in node)
    if max(result) > 15:
        # A complete balanced tree over used symbols bounds adversarially skewed
        # frequencies, while preserving length and distance symbols as well.
        depth = (len(leaves) - 1).bit_length()
        shallow = (1 << depth) - len(leaves)
        result = [0] * alphabet
        for index, symbol in enumerate(leaves):
            result[symbol] = depth - 1 if index < shallow else depth
    return result


def _write_tree(writer, lengths):
    symbols = [symbol for symbol, length in enumerate(lengths) if length]
    if len(symbols) <= 2 and max(symbols) < 256:
        writer.write(1, 1)
        writer.write(len(symbols) - 1, 1)
        writer.write(int(symbols[0] > 1), 1)
        writer.write(symbols[0], 8 if symbols[0] > 1 else 1)
        if len(symbols) == 2:
            writer.write(symbols[1], 8)
        return _codes(lengths)
    writer.write(0, 1)
    # Encode each length directly through a complete 16-symbol tree. These
    # tiny headers trade a little density for a simple, auditable encoder.
    code_lengths = [4] * 16 + [0] * 3
    writer.write(15, 4)  # All 19 length alphabet entries are transmitted.
    for symbol in _LENGTH_ORDER:
        writer.write(code_lengths[symbol], 3)
    writer.write(0, 1)  # Full alphabet; no shortened symbol count.
    length_codes = _codes(code_lengths)
    for length in lengths:
        value, count = length_codes[length]
        writer.write(value, count)
    return _codes(lengths)


def _prefix_code(value):
    if value <= 4:
        return value - 1, 0, 0
    count = (value - 1).bit_length() - 2
    high = (value - 1) >> count
    return count * 2 + high, value - 1 - (high << count), count


def _tokens(values, width):
    """Yield a bounded-memory, deterministic greedy LZ77 parse of pixels."""
    recent = array("i", [-1]) * 65536
    distances = {}
    for code, (dx, dy) in enumerate(_DISTANCES, 1):
        distances.setdefault(max(1, dx + dy * width), code)
    position, total = 0, len(values)
    while position < total:
        remaining = total - position
        if remaining < 3:
            yield 0, values[position], 0
            position += 1
            continue
        slot = ((values[position] * 0x1e35a7bd) ^ (values[position + 1] * 0x9e3779b1)
                ^ (values[position + 2] * 0x85ebca6b)) & 65535
        candidates = (position - 1, position - width, recent[slot])
        recent[slot] = position
        best_length, best_distance = 0, 0
        limit = min(4096, remaining)
        for candidate in candidates:
            distance = position - candidate
            if candidate < 0 or not 1 <= distance <= 1_000_000:
                continue
            if (values[candidate] != values[position] or values[candidate + 1] != values[position + 1]
                    or values[candidate + 2] != values[position + 2]):
                continue
            length = 3
            while length < limit and values[candidate + length] == values[position + length]:
                length += 1
            if length > best_length:
                best_length, best_distance = length, distance
        if best_length >= 3:
            yield 1, best_length, distances.get(best_distance, best_distance + 120)
            position += best_length
        else:
            yield 0, values[position], 0
            position += 1


def encode(width: int, height: int, mode: str, pixels: bytes, **options) -> bytes:
    """Encode exact L/LA/RGB/RGBA pixels as deterministic VP8L lossless WebP.

    The quality and method settings affect effort in native encoders; this
    encoder always uses deterministic Huffman/LZ77 codes and preserves invisible
    RGB values (the native ``exact=True`` behavior).
    """
    _dimensions(width, height, DEFAULT_MAX_PIXELS)
    channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}.get(mode)
    if channels is None or len(pixels) != width * height * channels:
        raise WebPError("WebP encoding needs correctly sized L/LA/RGB/RGBA pixels")
    if options.get("lossless", True) is not True:
        raise WebPError("Pure Python WebP encoding supports lossless pixels")
    if set(options) - {"lossless", "quality", "method", "exact", "optimize"}:
        raise WebPError("Unsupported WebP encoder option")
    quality, method = options.get("quality", 100), options.get("method", 6)
    if isinstance(quality, bool) or not isinstance(quality, (int, float)) or not 0 <= quality <= 100:
        raise WebPError("WebP quality must be from 0 through 100")
    if type(method) is not int or not 0 <= method <= 6:
        raise WebPError("WebP method must be an integer from 0 through 6")
    values = array("I")
    for offset in range(0, len(pixels), channels):
        if mode in ("L", "LA"):
            red = green = blue = pixels[offset]
            alpha = pixels[offset + 1] if mode == "LA" else 255
        else:
            red, green, blue = pixels[offset:offset + 3]
            alpha = pixels[offset + 3] if mode == "RGBA" else 255
        values.append((alpha << 24) | (red << 16) | (green << 8) | blue)
    writer = _Writer()
    writer.write(width - 1, 14)
    writer.write(height - 1, 14)
    writer.write(int(any(value >> 24 != 255 for value in values)), 1)
    writer.write(0, 3)
    writer.write(0, 1)  # No reversible transforms.
    writer.write(0, 1)  # No color cache.
    writer.write(0, 1)  # One entropy code group.
    shifts = (8, 16, 0, 24)
    frequencies = [Counter() for _ in range(5)]
    for kind, value, distance in _tokens(values, width):
        if kind:
            frequencies[0][256 + _prefix_code(value)[0]] += 1
            frequencies[4][_prefix_code(distance)[0]] += 1
        else:
            for shift, counts in zip(shifts, frequencies):
                counts[(value >> shift) & 255] += 1
    code_groups = []
    for counts, alphabet in zip(frequencies, (280, 256, 256, 256, 40)):
        code_groups.append(_write_tree(writer, _lengths(counts or Counter({0: 1}), alphabet)))
    for kind, value, distance in _tokens(values, width):
        if kind:
            symbol, extra, count = _prefix_code(value)
            encoded, size = code_groups[0][256 + symbol]
            writer.write(encoded, size)
            writer.write(extra, count)
            symbol, extra, count = _prefix_code(distance)
            encoded, size = code_groups[4][symbol]
            writer.write(encoded, size)
            writer.write(extra, count)
        else:
            for shift, codes in zip(shifts, code_groups):
                encoded, size = codes[(value >> shift) & 255]
                writer.write(encoded, size)
    payload = b"\x2f" + writer.finish()
    chunk = b"VP8L" + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) & 1 else b"")
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk
