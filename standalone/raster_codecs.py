"""Image container dispatch; unknown formats are refused before validation."""

from standalone.raster import MAX_IMAGE_PIXELS, Raster, UnidentifiedImageError


def decode(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        from standalone import raster_png
        return raster_png.decode(data)
    if data.startswith((b"GIF87a", b"GIF89a")):
        from standalone import raster_gif
        return raster_gif.decode(data)
    if data.startswith(b"BM"):
        from standalone import raster_bmp
        return raster_bmp.decode(data)
    if data.startswith((b"II*\0", b"MM\0*", b"II+\0", b"MM\0+")):
        from standalone import raster_tiff
        return raster_tiff.decode(data)
    if data.startswith((b"\0\0\1\0", b"\0\0\2\0")):
        from standalone import raster_ico
        return raster_ico.decode(data)
    if data.startswith(b"\xff\xd8"):
        from standalone import raster_jpeg
        width, height, mode, pixels = raster_jpeg.decode(data, max_pixels=MAX_IMAGE_PIXELS)
        return Raster(mode, (width, height), pixels, format="JPEG")
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        from standalone import raster_webp
        from standalone.raster import MAX_FRAMES
        width, height, decoded, info = raster_webp.decode_frames(data, max_pixels=MAX_IMAGE_PIXELS, max_frames=MAX_FRAMES)
        frames = [Raster(mode, (width, height), pixels, format="WEBP", info={**info, "duration": duration})
                  for mode, pixels, duration in decoded]
        image = frames[0].copy()
        image._frames = frames if len(frames) > 1 else None
        return image
    raise UnidentifiedImageError("Unsupported or unrecognized image format")


def encode(image, kind, options):
    options = dict(options)
    if kind == "PNG":
        from standalone import raster_png
        allowed = {"compress_level", "optimize"}
        operation = lambda: raster_png.encode(image, **options)
    elif kind in ("JPEG", "JPG"):
        from standalone import raster_jpeg
        allowed = {"quality", "optimize"}
        options.pop("optimize", None)  # Deterministic baseline encoder has no separate optimizer.
        operation = lambda: raster_jpeg.encode(image.width, image.height, image.mode, image.tobytes(), **options)
    elif kind == "GIF":
        from standalone import raster_gif
        allowed = {"duration", "loop"}
        operation = lambda: raster_gif.encode(image, **options)
    elif kind == "BMP":
        from standalone import raster_bmp
        allowed = set()
        operation = lambda: raster_bmp.encode(image)
    elif kind == "WEBP":
        from standalone import raster_webp
        allowed = {"lossless", "quality", "method", "exact"}
        operation = lambda: raster_webp.encode(image.width, image.height, image.mode, image.tobytes(), **options)
    elif kind in ("TIFF", "TIF"):
        from standalone import raster_tiff
        allowed = {"compression"}
        operation = lambda: raster_tiff.encode(image, **options)
    elif kind == "ICO":
        from standalone import raster_ico
        allowed = {"sizes"}
        operation = lambda: raster_ico.encode(image, **options)
    else:
        raise ValueError("Unsupported image output format: " + kind)
    if set(options) - allowed:
        raise TypeError("Unsupported image encoder options: " + ", ".join(sorted(set(options) - allowed)))
    return operation()
