"""Bounded screenshots through Windows GDI and macOS CoreGraphics OS ABIs."""

from __future__ import annotations

import ctypes
import struct

from standalone.x11_capture import MAX_PIXELS


class CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class CGRect(ctypes.Structure):
    _fields_ = [("origin", CGPoint), ("size", CGSize)]


class WinRect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_int32), ("top", ctypes.c_int32),
                ("right", ctypes.c_int32), ("bottom", ctypes.c_int32)]


def _bind(library, name, arguments, result):
    function = getattr(library, name)
    function.argtypes, function.restype = arguments, result
    return function


def _size(width, height):
    if type(width) is not int or type(height) is not int or min(width, height) <= 0 or width * height > MAX_PIXELS:
        raise OSError("Native screenshot exceeds the decoded pixel limit")
    return width, height


def _bbox(value):
    if len(value) != 4 or any(type(item) is not int or not -(1 << 30) <= item < (1 << 30) for item in value):
        raise ValueError("Screenshot bbox requires four bounded integer coordinates")
    x, y, right, bottom = value
    width, height = _size(right - x, bottom - y)
    return x, y, width, height


def bitmap_info(width, height):
    _size(width, height)
    return struct.pack("<IiiHHIIiiII", 40, width, -height, 1, 32, 0, width * height * 4, 0, 0, 0, 0)


def dib_pixels(raw, width, height):
    _size(width, height)
    if len(raw) != width * height * 4:
        raise ValueError("Native DIB byte count disagrees with the screenshot size")
    pixels = bytearray(width * height * 3)
    pixels[0::3], pixels[1::3], pixels[2::3] = raw[2::4], raw[1::4], raw[0::4]
    return bytes(pixels)


def capture_windows(bbox=None, include_layered_windows=False, all_screens=False, window=None):
    user = ctypes.WinDLL("user32", use_last_error=True)
    gdi = ctypes.WinDLL("gdi32", use_last_error=True)
    ptr, integer, uint = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32
    _bind(user, "GetDC", [ptr], ptr)
    _bind(user, "GetWindowDC", [ptr], ptr)
    _bind(user, "ReleaseDC", [ptr, ptr], integer)
    _bind(user, "GetSystemMetrics", [integer], integer)
    _bind(user, "GetWindowRect", [ptr, ctypes.POINTER(WinRect)], integer)
    _bind(user, "PrintWindow", [ptr, ptr, uint], integer)
    _bind(gdi, "CreateCompatibleDC", [ptr], ptr)
    _bind(gdi, "CreateCompatibleBitmap", [ptr, integer, integer], ptr)
    _bind(gdi, "SelectObject", [ptr, ptr], ptr)
    _bind(gdi, "DeleteObject", [ptr], integer)
    _bind(gdi, "DeleteDC", [ptr], integer)
    _bind(gdi, "BitBlt", [ptr, integer, integer, integer, integer, ptr, integer, integer, uint], integer)
    _bind(gdi, "GetDIBits", [ptr, ptr, uint, uint, ptr, ptr, uint], integer)
    dpi = getattr(user, "SetThreadDpiAwarenessContext", None)
    previous_dpi = None
    if dpi is not None:
        dpi.argtypes, dpi.restype = [ptr], ptr
        previous_dpi = dpi(ptr(-3))
    dc = memory = bitmap = selected = None
    try:
        if window is not None:
            if type(window) is not int or not 0 < window < 1 << (ctypes.sizeof(ptr) * 8):
                raise ValueError("Screenshot window handle must be a positive integer")
            rectangle = WinRect()
            if not user.GetWindowRect(window, ctypes.byref(rectangle)):
                raise OSError("Windows could not read the screenshot window bounds")
            x, y, width, height = _bbox((0, 0, rectangle.right - rectangle.left, rectangle.bottom - rectangle.top))
        elif bbox is not None:
            x, y, width, height = _bbox(bbox)
        elif all_screens:
            x, y = user.GetSystemMetrics(76), user.GetSystemMetrics(77)
            width, height = _size(user.GetSystemMetrics(78), user.GetSystemMetrics(79))
        else:
            x, y, width, height = 0, 0, *_size(user.GetSystemMetrics(0), user.GetSystemMetrics(1))
        dc = user.GetWindowDC(window) if window is not None else user.GetDC(None)
        if not dc:
            raise OSError("Windows could not acquire the screenshot display context")
        memory = gdi.CreateCompatibleDC(dc)
        bitmap = gdi.CreateCompatibleBitmap(dc, width, height)
        if not memory or not bitmap:
            raise OSError("Windows could not allocate the screenshot bitmap")
        selected = gdi.SelectObject(memory, bitmap)
        if not selected or selected == ptr(-1).value:
            selected = None
            raise OSError("Windows could not select the screenshot bitmap")
        if window is not None:
            success = user.PrintWindow(window, memory, 2)
        else:
            success = gdi.BitBlt(memory, 0, 0, width, height, dc, x, y, 0x00CC0020 | (0x40000000 if include_layered_windows else 0))
        if not success:
            raise OSError("Windows could not copy the screenshot pixels")
        restored = gdi.SelectObject(memory, selected)
        if not restored or restored == ptr(-1).value:
            raise OSError("Windows could not release the selected screenshot bitmap")
        selected = None
        raw = ctypes.create_string_buffer(width * height * 4)
        info = ctypes.create_string_buffer(bitmap_info(width, height))
        if gdi.GetDIBits(dc, bitmap, 0, height, raw, info, 0) != height:
            raise OSError("Windows returned an incomplete screenshot bitmap")
        return width, height, dib_pixels(raw.raw, width, height)
    finally:
        if selected:
            gdi.SelectObject(memory, selected)
        if bitmap:
            gdi.DeleteObject(bitmap)
        if memory:
            gdi.DeleteDC(memory)
        if dc:
            user.ReleaseDC(window, dc)
        if previous_dpi:
            dpi(previous_dpi)


def coregraphics_pixels(raw, width, height, stride, bits, bitmap_info):
    _size(width, height)
    alpha, order = bitmap_info & 31, bitmap_info & 0x7000
    if (bitmap_info & 0x100 or bits not in (24, 32) or stride < width * (bits // 8)
            or stride * height > MAX_PIXELS * 4 or len(raw) != stride * height):
        raise ValueError("Unsupported or inconsistent CoreGraphics bitmap layout")
    if bits == 24 and alpha == 0 and order == 0:
        return b"".join(raw[y * stride:y * stride + width * 3] for y in range(height))
    if bits != 32 or alpha not in (1, 2, 3, 4, 5, 6) or order not in (0, 0x2000, 0x4000):
        raise ValueError("Unsupported CoreGraphics alpha or byte order")
    first, little = alpha in (2, 4, 6), order == 0x2000
    positions = (1, 2, 3) if first else (0, 1, 2)
    alpha_position = 0 if first else 3
    if little:
        positions = tuple(3 - index for index in positions)
        alpha_position = 3 - alpha_position
    pixels = bytearray(width * height * 3)
    for y in range(height):
        row = raw[y * stride:y * stride + width * 4]
        for channel, position in enumerate(positions):
            pixels[y * width * 3 + channel:(y + 1) * width * 3:3] = row[position::4]
        if alpha in (1, 2):
            for x in range(width):
                a = row[x * 4 + alpha_position]
                if a != 255:
                    start = (y * width + x) * 3
                    for c in range(3):
                        pixels[start + c] = min(255, (pixels[start + c] * 255 + a // 2) // a) if a else 0
    return bytes(pixels)


def capture_macos(bbox=None, all_screens=False, window=None):
    cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    ptr, uint, size = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t
    _bind(cg, "CGMainDisplayID", [], uint)
    _bind(cg, "CGDisplayBounds", [uint], CGRect)
    _bind(cg, "CGGetActiveDisplayList", [uint, ctypes.POINTER(uint), ctypes.POINTER(uint)], ctypes.c_int32)
    _bind(cg, "CGWindowListCreateImage", [CGRect, uint, uint, uint], ptr)
    for name in ("CGImageGetWidth", "CGImageGetHeight", "CGImageGetBytesPerRow", "CGImageGetBitsPerComponent", "CGImageGetBitsPerPixel"):
        _bind(cg, name, [ptr], size)
    _bind(cg, "CGImageGetBitmapInfo", [ptr], uint)
    _bind(cg, "CGImageGetDataProvider", [ptr], ptr)
    _bind(cg, "CGDataProviderCopyData", [ptr], ptr)
    _bind(cg, "CGImageRelease", [ptr], None)
    _bind(cf, "CFDataGetLength", [ptr], ctypes.c_ssize_t)
    _bind(cf, "CFDataGetBytePtr", [ptr], ptr)
    _bind(cf, "CFRelease", [ptr], None)
    preflight = getattr(cg, "CGPreflightScreenCaptureAccess", None)
    if preflight is not None:
        preflight.argtypes, preflight.restype = [], ctypes.c_bool
        if not preflight():
            raise OSError("macOS screen recording permission is required for screenshots")
    if bbox is not None:
        x, y, width, height = _bbox(bbox)
        rectangle = CGRect(CGPoint(x, y), CGSize(width, height))
    elif window is not None:
        if type(window) is not int or not 0 < window < 1 << 32:
            raise ValueError("Screenshot window ID must fit a positive 32-bit integer")
        rectangle = CGRect.in_dll(cg, "CGRectNull")
    elif all_screens:
        identifiers, count = (uint * 64)(), uint()
        if cg.CGGetActiveDisplayList(64, identifiers, ctypes.byref(count)) or not 0 < count.value <= 64:
            raise OSError("macOS could not enumerate active displays")
        bounds = [cg.CGDisplayBounds(identifiers[index]) for index in range(count.value)]
        x, y = min(r.origin.x for r in bounds), min(r.origin.y for r in bounds)
        right, bottom = max(r.origin.x + r.size.width for r in bounds), max(r.origin.y + r.size.height for r in bounds)
        _size(int(right - x), int(bottom - y))
        rectangle = CGRect(CGPoint(x, y), CGSize(right - x, bottom - y))
    else:
        rectangle = cg.CGDisplayBounds(cg.CGMainDisplayID())
        _size(int(rectangle.size.width), int(rectangle.size.height))
    image = cg.CGWindowListCreateImage(rectangle, 8 if window is not None else 1, window or 0, 0)
    if not image:
        raise OSError("macOS could not capture the selected screen or window")
    data = None
    try:
        width, height = _size(cg.CGImageGetWidth(image), cg.CGImageGetHeight(image))
        stride, bits = cg.CGImageGetBytesPerRow(image), cg.CGImageGetBitsPerPixel(image)
        if (cg.CGImageGetBitsPerComponent(image) != 8 or bits not in (24, 32)
                or not width * (bits // 8) <= stride <= width * 4 + 4096 or stride * height > MAX_PIXELS * 4):
            raise OSError("Unsupported or oversized macOS screenshot format")
        provider = cg.CGImageGetDataProvider(image)
        if not provider:
            raise OSError("macOS screenshot has no bitmap data provider")
        data = cg.CGDataProviderCopyData(provider)
        if not data or cf.CFDataGetLength(data) != stride * height:
            raise OSError("macOS returned an inconsistent screenshot byte count")
        address = cf.CFDataGetBytePtr(data)
        if not address:
            raise OSError("macOS screenshot data is inaccessible")
        raw = ctypes.string_at(address, stride * height)
        return width, height, coregraphics_pixels(raw, width, height, stride, bits, cg.CGImageGetBitmapInfo(image))
    finally:
        if data:
            cf.CFRelease(data)
        cg.CGImageRelease(image)
