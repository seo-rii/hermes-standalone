"""The application's explicit image backend for its selected distribution."""

from agent.sdk_boundary import bundled

if bundled():
    from standalone import raster as Image
    from standalone import raster_font as ImageFont
    from standalone import raster_imageops as ImageOps
    from standalone import raster_grab as ImageGrab
    from standalone.raster_draw import ImageDraw
    from standalone.raster import ImageChops, ImageFilter, ImageSequence
else:
    from PIL import Image, ImageChops, ImageFilter, ImageSequence, ImageDraw, ImageFont, ImageOps, ImageGrab


__all__ = ["Image", "ImageChops", "ImageFilter", "ImageSequence", "ImageDraw", "ImageFont", "ImageOps", "ImageGrab"]
