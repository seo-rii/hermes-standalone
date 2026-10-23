"""Qualified screenshots through stdlib X11, Windows GDI or macOS OS APIs."""

from standalone import x11_capture
from standalone.raster import Raster


def grab(bbox=None, include_layered_windows=False, all_screens=False, xdisplay=None, window=None):
    from hermes_platform.host.facts import os_family

    family = os_family()
    if xdisplay is None and family == "win32":
        from standalone.raster_capture_native import capture_windows
        width, height, pixels = capture_windows(bbox, include_layered_windows, all_screens, window)
    elif xdisplay is None and family == "darwin":
        from standalone.raster_capture_native import capture_macos
        width, height, pixels = capture_macos(bbox, all_screens, window)
    else:
        if include_layered_windows or window is not None:
            raise ValueError("Window-specific X11 screenshots are not supported")
        width, height, pixels = x11_capture.capture(xdisplay, bbox)
    return Raster("RGB", (width, height), pixels)


def worker_script():
    from importlib.resources import files
    return files("standalone").joinpath("x11_capture.py").read_text(encoding="utf-8")


def from_worker(data):
    width, height, pixels = x11_capture.decode_wire(data)
    return Raster("RGB", (width, height), pixels)
