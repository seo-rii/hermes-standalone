"""Aspect-preserving image operations used by the browser's contact sheets."""

from standalone import raster


def contain(image, size, method=raster.Resampling.BICUBIC):
    width, height = raster.checked_size(size)
    ratio = min(width / image.width, height / image.height)
    target = max(1, round(image.width * ratio)), max(1, round(image.height * ratio))
    return image.copy() if target == image.size else image.resize(target, resample=method)


def cover(image, size, method=raster.Resampling.BICUBIC):
    width, height = raster.checked_size(size)
    ratio = max(width / image.width, height / image.height)
    target = max(1, round(image.width * ratio)), max(1, round(image.height * ratio))
    return image.copy() if target == image.size else image.resize(target, resample=method)


def fit(image, size, method=raster.Resampling.BICUBIC, bleed=0.0, centering=(0.5, 0.5)):
    if not 0 <= bleed < 0.5 or len(centering) != 2 or any(not 0 <= value <= 1 for value in centering):
        raise ValueError("Invalid image crop bleed or centering")
    width, height = raster.checked_size(size)
    if bleed:
        bx, by = round(image.width * bleed), round(image.height * bleed)
        image = image.crop((bx, by, image.width - bx, image.height - by))
    resized = cover(image, (width, height), method)
    left = round((resized.width - width) * centering[0])
    top = round((resized.height - height) * centering[1])
    return resized.crop((left, top, left + width, top + height))


def pad(image, size, method=raster.Resampling.BICUBIC, color=None, centering=(0.5, 0.5)):
    if len(centering) != 2 or any(not 0 <= value <= 1 for value in centering):
        raise ValueError("Invalid image padding centering")
    resized = contain(image, size, method)
    output = raster.new(image.mode, size, 0 if color is None else color)
    left = round((output.width - resized.width) * centering[0])
    top = round((output.height - resized.height) * centering[1])
    output.paste(resized, (left, top))
    return output
