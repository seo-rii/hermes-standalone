"""Deterministic resampling and compositing for the standalone raster boundary."""

from __future__ import annotations

from collections import Counter
import math

from standalone.raster import Raster, Resampling, new


def _kernel(value, kind):
    value = abs(value)
    if kind == Resampling.BOX:
        return float(value <= 0.5)
    if kind == Resampling.BILINEAR:
        return max(0.0, 1 - value)
    if kind == Resampling.HAMMING:
        return (math.sin(math.pi * value) / (math.pi * value) *
                (0.54 + 0.46 * math.cos(math.pi * value))) if 0 < value < 1 else float(value == 0)
    if kind == Resampling.BICUBIC:
        if value < 1:
            return (1.5 * value - 2.5) * value * value + 1
        return ((-0.5 * value + 2.5) * value - 4) * value + 2 if value < 2 else 0.0
    if kind == Resampling.LANCZOS:
        if value == 0:
            return 1.0
        return (math.sin(math.pi * value) * math.sin(math.pi * value / 3) /
                (math.pi * math.pi * value * value / 3)) if value < 3 else 0.0
    raise ValueError("Unknown resampling filter")


def _weights(source, target, kind):
    radius = {Resampling.BOX: 0.5, Resampling.BILINEAR: 1, Resampling.HAMMING: 1,
              Resampling.BICUBIC: 2, Resampling.LANCZOS: 3}.get(kind)
    if radius is None:
        raise ValueError("Unknown resampling filter")
    scale = source / target
    antialias = max(1.0, scale)
    result = []
    for index in range(target):
        center = (index + 0.5) * scale - 0.5
        low, high = max(0, math.ceil(center - radius * antialias)), min(source - 1, math.floor(center + radius * antialias))
        values = [(sample, _kernel((sample - center) / antialias, kind)) for sample in range(low, high + 1)]
        total = sum(weight for _, weight in values)
        result.append([(sample, weight / total) for sample, weight in values] if total else [(min(source - 1, round(center)), 1.0)])
    return result


def _byte(value):
    return max(0, min(255, int(value + 0.5)))


def resize(image, size, kind):
    if size == image.size:
        return image.copy()
    width, height = size
    bands, pixels = image.bands, image._pixels
    if kind == Resampling.NEAREST or image.mode == "P":
        result = bytearray(width * height * bands)
        for y in range(height):
            sy = min(image.height - 1, int((y + 0.5) * image.height / height))
            for x in range(width):
                sx = min(image.width - 1, int((x + 0.5) * image.width / width))
                dest, source = (y * width + x) * bands, (sy * image.width + sx) * bands
                result[dest:dest + bands] = pixels[source:source + bands]
        return Raster(image.mode, size, result, palette=image._palette, info=image.info)
    xs, ys = _weights(image.width, width, kind), _weights(image.height, height, kind)
    alpha = bands - 1 if image.mode in ("RGBA", "LA") else None
    # Filter premultiplied colors so a transparent pixel's hidden color never
    # bleeds into an opaque neighbor. Keep only the horizontal rows needed by
    # the current output row instead of an unbounded float image.
    rows, output = {}, bytearray(width * height * bands)
    for y, yw in enumerate(ys):
        needed = {row for row, _ in yw}
        rows = {row: values for row, values in rows.items() if row in needed}
        for sy in needed - rows.keys():
            row = []
            for xw in xs:
                for channel in range(bands):
                    value = 0.0
                    for sx, weight in xw:
                        start = (sy * image.width + sx) * bands
                        sample = pixels[start + channel]
                        if alpha is not None and channel != alpha:
                            sample *= pixels[start + alpha] / 255
                        value += sample * weight
                    row.append(value)
            rows[sy] = row
        for x in range(width):
            values = [sum(rows[sy][x * bands + channel] * weight for sy, weight in yw) for channel in range(bands)]
            if alpha is not None:
                raw_alpha = values[alpha]
                a = max(0.0, min(255.0, raw_alpha))
                values = [value * 255 / raw_alpha if raw_alpha > 0 else 0.0 for value in values[:alpha]] + [a]
            offset = (y * width + x) * bands
            output[offset:offset + bands] = bytes(map(_byte, values))
    return Raster(image.mode, size, output)


def paste(destination, image, box=None, mask=None):
    if box is None:
        box = (0, 0)
    if len(box) == 2:
        if not isinstance(image, Raster):
            raise ValueError("A solid-color paste requires a four-coordinate box")
        left, top = box
        right, bottom = left + image.width, top + image.height
    elif len(box) == 4:
        left, top, right, bottom = box
    else:
        raise ValueError("Invalid paste box")
    if right <= left or bottom <= top:
        return
    if not isinstance(image, Raster):
        image = new(destination.mode, (right - left, bottom - top), image)
    if image.size != (right - left, bottom - top):
        raise ValueError("Paste region and source dimensions differ")
    if image.mode != destination.mode:
        image = image.convert(destination.mode)
    if mask is not None:
        if mask.size != image.size or mask.mode not in ("L", "LA", "RGBA"):
            raise ValueError("Invalid paste mask")
        mask_pixels = mask._pixels[mask.bands - 1::mask.bands]
    bands = destination.bands
    for y in range(max(0, top), min(destination.height, bottom)):
        for x in range(max(0, left), min(destination.width, right)):
            target = (y * destination.width + x) * bands
            source_pixel = (y - top) * image.width + x - left
            source = source_pixel * bands
            a = mask_pixels[source_pixel] if mask is not None else 255
            for channel in range(bands):
                destination._pixels[target + channel] = (image._pixels[source + channel] * a +
                    destination._pixels[target + channel] * (255 - a) + 127) // 255


def alpha_composite(destination, image, dest=(0, 0), source=(0, 0)):
    if destination.mode != "RGBA" or image.mode != "RGBA":
        raise ValueError("Alpha compositing requires RGBA images")
    sx, sy = source[:2]
    right, bottom = source[2:] if len(source) == 4 else image.size
    if sx < 0 or sy < 0 or right < sx or bottom < sy:
        raise ValueError("Invalid compositing source box")
    dx, dy = dest
    for y in range(sy, min(bottom, image.height)):
        ty = dy + y - sy
        if not 0 <= ty < destination.height:
            continue
        for x in range(sx, min(right, image.width)):
            tx = dx + x - sx
            if not 0 <= tx < destination.width:
                continue
            src, dst = (y * image.width + x) * 4, (ty * destination.width + tx) * 4
            sa, da = image._pixels[src + 3], destination._pixels[dst + 3]
            combined = sa * 255 + da * (255 - sa)
            for channel in range(3):
                numerator = image._pixels[src + channel] * sa * 255 + destination._pixels[dst + channel] * da * (255 - sa)
                destination._pixels[dst + channel] = (numerator + combined // 2) // combined if combined else 0
            destination._pixels[dst + 3] = (combined + 127) // 255


def quantize(image, colors):
    if type(colors) is not int or not 1 <= colors <= 256:
        raise ValueError("Palette size must be in 1..256")
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    histogram = Counter(zip(rgb._pixels[0::3], rgb._pixels[1::3], rgb._pixels[2::3]))
    boxes = [list(sorted(histogram))]
    while len(boxes) < colors:
        choices = [(max(max(p[c] for p in box) - min(p[c] for p in box) for c in range(3)) *
                    sum(histogram[p] for p in box), index) for index, box in enumerate(boxes) if len(box) > 1]
        if not choices:
            break
        _, index = max(choices)
        box = boxes.pop(index)
        channel = max(range(3), key=lambda c: max(p[c] for p in box) - min(p[c] for p in box))
        box.sort(key=lambda p: (p[channel], p))
        half, subtotal, split = sum(histogram[p] for p in box) / 2, 0, 1
        for split, pixel in enumerate(box[:-1], 1):
            subtotal += histogram[pixel]
            if subtotal >= half:
                break
        boxes.extend((box[:split], box[split:]))
    palette, indices = [], {}
    for index, box in enumerate(boxes):
        total = sum(histogram[p] for p in box)
        palette.extend((sum(p[c] * histogram[p] for p in box) + total // 2) // total for c in range(3))
        indices.update((p, index) for p in box)
    pixels = bytes(indices[p] for p in zip(rgb._pixels[0::3], rgb._pixels[1::3], rgb._pixels[2::3]))
    return Raster("P", image.size, pixels, palette=palette + [0] * (768 - len(palette)))
