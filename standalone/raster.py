"""Byte-backed raster operations used by Hermes' vision and sprite paths.

The owning application imports this module explicitly. It is never registered
as PIL, and unknown codecs or image operations fail instead of accepting an
undecoded image as valid.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path


MAX_IMAGE_PIXELS = 100_000_000
MAX_FRAMES = 100


class UnidentifiedImageError(OSError):
    pass


class DecompressionBombError(ValueError):
    pass


class Resampling(IntEnum):
    NEAREST = 0
    LANCZOS = 1
    BILINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5


class Transpose(IntEnum):
    FLIP_LEFT_RIGHT = 0
    FLIP_TOP_BOTTOM = 1
    ROTATE_90 = 2
    ROTATE_180 = 3
    ROTATE_270 = 4


NEAREST, LANCZOS, BILINEAR, BICUBIC, BOX = (Resampling.NEAREST, Resampling.LANCZOS,
                                         Resampling.BILINEAR, Resampling.BICUBIC, Resampling.BOX)
MEDIANCUT = 0


def checked_size(size):
    width, height = size
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive integers")
    if width * height > MAX_IMAGE_PIXELS:
        raise DecompressionBombError("Image exceeds the decoded pixel limit")
    return width, height


def _pixel(value, bands):
    if isinstance(value, str):
        from standalone.raster_draw import color
        value = color(value, "RGBA" if bands in (2, 4) else "RGB")
        if bands in (1, 2):
            gray = (19595 * value[0] + 38470 * value[1] + 7471 * value[2] + 32768) >> 16
            value = (gray,) if bands == 1 else (gray, value[3])
    if isinstance(value, int):
        value = (value,) * bands
    if len(value) != bands or any(type(channel) is not int or not 0 <= channel <= 255 for channel in value):
        raise ValueError("Pixel does not match the image mode")
    return bytes(value)


class PixelAccess:
    def __init__(self, image):
        self.image = image

    def _offset(self, position):
        x, y = position
        if x < 0:
            x += self.image.width
        if y < 0:
            y += self.image.height
        if not 0 <= x < self.image.width or not 0 <= y < self.image.height:
            raise IndexError("Image coordinate out of range")
        return (y * self.image.width + x) * self.image.bands

    def __getitem__(self, position):
        start = self._offset(position)
        if self.image.bands == 1:
            return self.image._pixels[start]
        return tuple(self.image._pixels[start:start + self.image.bands])

    def __setitem__(self, position, value):
        start = self._offset(position)
        self.image._pixels[start:start + self.image.bands] = _pixel(value, self.image.bands)


class PixelData:
    """Read-only sequence over pixel bytes without materializing tuple copies."""
    def __init__(self, image):
        self.image = image

    def __len__(self):
        return self.image.width * self.image.height

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Pixel index out of range")
        start = index * self.image.bands
        return self.image._pixels[start] if self.image.bands == 1 else tuple(self.image._pixels[start:start + self.image.bands])


class Raster:
    def __init__(self, mode, size, pixels, *, format=None, info=None, palette=None):
        if mode not in ("L", "LA", "RGB", "RGBA", "P"):
            raise ValueError("Unsupported raster mode: " + str(mode))
        self.mode, self.size = mode, checked_size(size)
        self.width, self.height = self.size
        self.bands = 1 if mode == "P" else len(mode)
        self._pixels = bytearray(pixels)
        if len(self._pixels) != self.width * self.height * self.bands:
            raise ValueError("Pixel byte count does not match dimensions")
        self.format, self.info, self._palette = format, dict(info or {}), palette
        self._frames, self._frame_index = None, 0

    @property
    def n_frames(self):
        return len(self._frames) if self._frames else 1

    @property
    def is_animated(self):
        return self.n_frames > 1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        # Codecs fully decode while their source is open, so no source handle
        # survives this object. Copies remain independent after context exit.
        pass

    def verify(self):
        return None

    def load(self):
        return PixelAccess(self)

    def getpixel(self, position):
        return self.load()[position]

    def putpixel(self, position, value):
        self.load()[position] = value

    def getdata(self):
        return PixelData(self)

    def getbands(self):
        return ("P",) if self.mode == "P" else tuple(self.mode)

    def tobytes(self):
        return bytes(self._pixels)

    def copy(self):
        return Raster(self.mode, self.size, self._pixels, format=self.format, info=self.info,
                      palette=list(self._palette) if self._palette is not None else None)

    def convert(self, mode, **kwargs):
        if mode == self.mode:
            return self.copy()
        if mode == "P":
            return self.quantize(**kwargs)
        if mode not in ("L", "LA", "RGB", "RGBA"):
            raise ValueError("Unsupported raster mode: " + str(mode))
        result = bytearray()
        for pixel in self.getdata():
            if self.mode == "P":
                red, green, blue = self._palette[pixel * 3:pixel * 3 + 3]
                alpha = 0 if pixel == self.info.get("transparency") else 255
            elif self.mode in ("L", "LA"):
                red = green = blue = pixel if self.mode == "L" else pixel[0]
                alpha = 255 if self.mode == "L" else pixel[1]
            else:
                red, green, blue = pixel[:3]
                alpha = pixel[3] if self.mode == "RGBA" else 255
            luminance = (red * 19595 + green * 38470 + blue * 7471 + 32768) >> 16
            result.extend({"L": (luminance,), "LA": (luminance, alpha), "RGB": (red, green, blue),
                           "RGBA": (red, green, blue, alpha)}[mode])
        return Raster(mode, self.size, result)

    def crop(self, box):
        left, top, right, bottom = map(lambda value: int(round(value)), box)
        if right <= left or bottom <= top:
            raise ValueError("Crop box must have positive dimensions")
        result = new(self.mode, (right - left, bottom - top))
        result._palette = self._palette
        result.info = dict(self.info)
        for y in range(max(top, 0), min(bottom, self.height)):
            start, end = max(left, 0), min(right, self.width)
            if start >= end:
                continue
            source = (y * self.width + start) * self.bands
            target = ((y - top) * result.width + start - left) * self.bands
            result._pixels[target:target + (end - start) * self.bands] = self._pixels[source:source + (end - start) * self.bands]
        return result

    def resize(self, size, resample=BICUBIC, **kwargs):
        if kwargs:
            raise TypeError("Unsupported raster resize options: " + ", ".join(kwargs))
        from standalone.raster_ops import resize
        return resize(self, checked_size(size), resample)

    def thumbnail(self, size, resample=BICUBIC, **kwargs):
        ratio = min(size[0] / self.width, size[1] / self.height, 1)
        if ratio < 1:
            changed = self.resize((max(1, round(self.width * ratio)), max(1, round(self.height * ratio))), resample, **kwargs)
            self.__dict__.update(changed.__dict__)

    def split(self):
        return tuple(Raster("L", self.size, self._pixels[index::self.bands]) for index in range(self.bands))

    def getchannel(self, channel):
        index = self.mode.index(channel) if isinstance(channel, str) else channel
        return self.split()[index]

    def putalpha(self, alpha):
        if self.mode not in ("RGBA", "LA"):
            changed = self.convert("LA" if self.mode == "L" else "RGBA")
            self.__dict__.update(changed.__dict__)
        if isinstance(alpha, int):
            alpha = new("L", self.size, alpha)
        if alpha.mode != "L" or alpha.size != self.size:
            raise ValueError("Alpha mask must match the image size")
        self._pixels[self.bands - 1::self.bands] = alpha._pixels

    def point(self, transform):
        table = [transform(value) for value in range(256)] if callable(transform) else list(transform)
        if len(table) not in (256, 256 * self.bands):
            raise ValueError("Point lookup table has the wrong length")
        result = self.copy()
        for index, value in enumerate(self._pixels):
            result._pixels[index] = max(0, min(255, round(table[value + (index % self.bands * 256 if len(table) > 256 else 0)])))
        return result

    def getextrema(self):
        values = tuple((min(self._pixels[index::self.bands]), max(self._pixels[index::self.bands])) for index in range(self.bands))
        return values[0] if self.bands == 1 else values

    def histogram(self):
        output = [0] * (256 * self.bands)
        for index, value in enumerate(self._pixels):
            output[index % self.bands * 256 + value] += 1
        return output

    def getbbox(self, *, alpha_only=True):
        pixels = self.getchannel("A")._pixels if alpha_only and "A" in self.mode else self._pixels
        bands = 1 if alpha_only and "A" in self.mode else self.bands
        left, top, right, bottom = self.width, self.height, -1, -1
        for index in range(self.width * self.height):
            if any(pixels[index * bands:(index + 1) * bands]):
                y, x = divmod(index, self.width)
                left, top, right, bottom = min(left, x), min(top, y), max(right, x), max(bottom, y)
        return (left, top, right + 1, bottom + 1) if right >= 0 else None

    def paste(self, image, box=None, mask=None):
        from standalone.raster_ops import paste
        paste(self, image, box, mask)

    def alpha_composite(self, image, dest=(0, 0), source=(0, 0)):
        from standalone.raster_ops import alpha_composite
        alpha_composite(self, image, dest, source)

    def filter(self, operation):
        return operation.apply(self)

    def transpose(self, method):
        method = Transpose(method)
        if method in (Transpose.ROTATE_90, Transpose.ROTATE_270):
            result = new(self.mode, (self.height, self.width))
            result._palette, result.info = self._palette, dict(self.info)
            source, target = self.load(), result.load()
            for y in range(self.height):
                for x in range(self.width):
                    target[(y, self.width - x - 1) if method == Transpose.ROTATE_90 else (self.height - y - 1, x)] = source[x, y]
            return result
        result, source = self.copy(), self.load()
        target = result.load()
        for y in range(self.height):
            for x in range(self.width):
                target[x, y] = source[self.width - x - 1 if method != Transpose.FLIP_TOP_BOTTOM else x,
                                       self.height - y - 1 if method != Transpose.FLIP_LEFT_RIGHT else y]
        return result

    def quantize(self, colors=256, method=MEDIANCUT, **kwargs):
        if kwargs or method != MEDIANCUT:
            raise TypeError("Only deterministic median-cut quantization is supported")
        from standalone.raster_ops import quantize
        return quantize(self, colors)

    def getpalette(self):
        return list(self._palette) if self._palette is not None else None

    def seek(self, frame):
        if not 0 <= frame < self.n_frames:
            raise EOFError("No such image frame")
        if self._frames:
            changed = self._frames[frame]
            self.mode, self.bands, self.size = changed.mode, changed.bands, changed.size
            self.width, self.height = changed.size
            self._pixels, self.info = bytearray(changed._pixels), dict(changed.info)
        self._frame_index = frame

    def tell(self):
        return self._frame_index

    def save(self, destination, format=None, **options):
        from standalone.raster_codecs import encode
        kind = format or (Path(destination).suffix[1:] if isinstance(destination, (str, Path)) else self.format)
        data = encode(self, str(kind or "PNG").upper(), options)
        if hasattr(destination, "write"):
            destination.write(data)
        else:
            Path(destination).write_bytes(data)


Image = Raster


def new(mode, size, color=0):
    width, height = checked_size(size)
    bands = 1 if mode == "P" else len(mode)
    return Raster(mode, size, _pixel(color, bands) * (width * height))


def frombytes(mode, size, data):
    return Raster(mode, size, data)


def open(source, mode="r", formats=None):
    if mode != "r":
        raise ValueError("Raster images can only be opened for reading")
    from standalone.raster_codecs import decode
    if hasattr(source, "read"):
        data = source.read()
    else:
        data = Path(source).read_bytes()
    image = decode(data)
    if formats is not None and image.format not in formats:
        raise UnidentifiedImageError("Image format is not in the permitted set")
    return image


def composite(first, second, mask):
    output = second.copy()
    output.paste(first, mask=mask)
    return output


class ImageSequence:
    @staticmethod
    def Iterator(image):
        if image._frames:
            yield from image._frames
        else:
            yield image


class ImageChops:
    @staticmethod
    def darker(first, second):
        if first.size != second.size or first.mode != second.mode:
            raise ValueError("Channel images must match")
        return Raster(first.mode, first.size, bytes(map(min, first._pixels, second._pixels)))


class ImageFilter:
    class MinFilter:
        def __init__(self, size=3):
            if size <= 0 or size % 2 != 1:
                raise ValueError("Filter size must be positive and odd")
            self.size = size

        def apply(self, image):
            output = image.copy()
            radius = self.size // 2
            for y in range(image.height):
                for x in range(image.width):
                    for channel in range(image.bands):
                        output._pixels[(y * image.width + x) * image.bands + channel] = min(
                            image._pixels[(yy * image.width + xx) * image.bands + channel]
                            for yy in range(max(0, y - radius), min(image.height, y + radius + 1))
                            for xx in range(max(0, x - radius), min(image.width, x + radius + 1)))
            return output
