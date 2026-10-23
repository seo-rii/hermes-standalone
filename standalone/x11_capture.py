"""X11 GetImage over the standard-library socket protocol, without libX11.

This module is also sent as source to the existing sandbox Python interpreter;
its imports therefore intentionally use the standard library alone.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import socket
import struct
import sys
import zlib


MAX_PIXELS = 40_000_000
MAGIC = b"HX11"


def _read(stream, count):
    result = bytearray()
    while len(result) < count:
        part = stream.recv(min(65536, count - len(result)))
        if not part:
            raise OSError("X11 server closed a truncated response")
        result.extend(part)
    return bytes(result)


def _display(value):
    match = re.fullmatch(r"(?:(unix)/)?(\[[^\]]+\]|[^:]*):(\d+)(?:\.(\d+))?", value or "")
    if not match:
        raise OSError("An explicit valid X11 DISPLAY is required")
    transport, hostname, number, screen = match.groups()
    number, screen = int(number), int(screen or 0)
    if number > 59535 or screen > 255:
        raise OSError("X11 display or screen number exceeds the protocol range")
    local = transport == "unix" or hostname in ("", "unix")
    return hostname.strip("[]"), number, screen, local


def _authorization(hostname, number, local):
    path = Path(os.environ.get("XAUTHORITY") or Path.home() / ".Xauthority")
    try:
        with path.open("rb") as stream:
            data = stream.read(1_048_577)
    except FileNotFoundError:
        return b"", b""
    if len(data) > 1_048_576:
        raise OSError("Xauthority exceeds the validation limit")
    offset, candidates = 0, []
    while offset < len(data):
        if offset + 2 > len(data):
            raise OSError("Truncated Xauthority record")
        family = int.from_bytes(data[offset:offset + 2], "big")
        offset += 2
        fields = []
        for _ in range(4):
            if offset + 2 > len(data):
                raise OSError("Truncated Xauthority field length")
            length = int.from_bytes(data[offset:offset + 2], "big")
            offset += 2
            if length > len(data) - offset:
                raise OSError("Truncated Xauthority field")
            fields.append(data[offset:offset + length])
            offset += length
        address, display, name, cookie = fields
        if display not in (b"", str(number).encode()) or name != b"MIT-MAGIC-COOKIE-1" or len(cookie) != 16:
            continue
        matched = family == 65535
        if family == 256 and local:
            matched = address in (b"", socket.gethostname().encode(), b"localhost")
        elif family in (0, 6) and not local and len(address) == (4 if family == 0 else 16):
            try:
                matched = socket.inet_pton(socket.AF_INET if family == 0 else socket.AF_INET6, hostname) == address
            except OSError:
                matched = any(item[4][0] == socket.inet_ntop(socket.AF_INET if family == 0 else socket.AF_INET6, address)
                              for item in socket.getaddrinfo(hostname, None, socket.AF_INET if family == 0 else socket.AF_INET6))
        if matched:
            candidates.append((family != 65535, bool(display), name, cookie))
    return max(candidates)[2:] if candidates else (b"", b"")


def _setup(data, screen_number):
    if len(data) < 32:
        raise OSError("Truncated X11 setup information")
    vendor_length = struct.unpack_from("<H", data, 16)[0]
    screens, formats, byte_order = data[20:23]
    if byte_order not in (0, 1) or not 0 <= screen_number < screens:
        raise OSError("X11 screen or image byte order is invalid")
    offset = 32 + ((vendor_length + 3) & ~3)
    pixel_formats = {}
    for _ in range(formats):
        if offset + 8 > len(data):
            raise OSError("Truncated X11 pixmap format")
        depth, bits, pad = data[offset:offset + 3]
        if bits not in (8, 16, 24, 32) or pad not in (8, 16, 32):
            # Monochrome formats may coexist with the selected TrueColor root.
            pixel_formats[depth] = None
        else:
            pixel_formats[depth] = bits, pad
        offset += 8
    chosen = None
    for screen_index in range(screens):
        if offset + 40 > len(data):
            raise OSError("Truncated X11 screen")
        root = struct.unpack_from("<I", data, offset)[0]
        width, height = struct.unpack_from("<HH", data, offset + 20)
        root_visual = struct.unpack_from("<I", data, offset + 32)[0]
        depth, depths = data[offset + 38:offset + 40]
        offset += 40
        visuals = {}
        for _ in range(depths):
            if offset + 8 > len(data):
                raise OSError("Truncated X11 depth")
            count = struct.unpack_from("<H", data, offset + 2)[0]
            offset += 8
            if count * 24 > len(data) - offset:
                raise OSError("Truncated X11 visual information")
            for _ in range(count):
                identifier, kind = struct.unpack_from("<IB", data, offset)
                masks = struct.unpack_from("<III", data, offset + 8)
                visuals[identifier] = kind, masks
                offset += 24
        if screen_index == screen_number:
            if root_visual not in visuals or visuals[root_visual][0] != 4 or not pixel_formats.get(depth):
                raise OSError("X11 capture requires a TrueColor root visual")
            chosen = root, width, height, root_visual, depth, pixel_formats[depth], byte_order, visuals[root_visual][1]
    return chosen


def _pixels(raw, width, height, bits, pad, byte_order, masks):
    stride = ((width * bits + pad - 1) // pad) * (pad // 8)
    if len(raw) != stride * height:
        raise OSError("X11 pixel response size disagrees with its format")
    channels = []
    for mask in masks:
        if not mask or mask >> bits or any(mask & old for old in masks[:len(channels)]):
            raise OSError("Invalid X11 TrueColor channel masks")
        shift = (mask & -mask).bit_length() - 1
        maximum = mask >> shift
        if maximum & (maximum + 1):
            raise OSError("Noncontiguous X11 channel mask")
        channels.append((shift, maximum))
    result, size = bytearray(width * height * 3), bits // 8
    if bits in (24, 32) and masks == (0xFF0000, 0xFF00, 0xFF):
        positions = (2, 1, 0) if byte_order == 0 else ((0, 1, 2) if bits == 24 else (1, 2, 3))
        for y in range(height):
            row = raw[y * stride:y * stride + width * size]
            for channel, position in enumerate(positions):
                result[y * width * 3 + channel:(y + 1) * width * 3:3] = row[position::size]
    else:
        order = "little" if byte_order == 0 else "big"
        for y in range(height):
            for x in range(width):
                offset = y * stride + x * size
                value = int.from_bytes(raw[offset:offset + size], order)
                target = (y * width + x) * 3
                result[target:target + 3] = bytes(((value >> shift) & maximum) * 255 // maximum for shift, maximum in channels)
    return bytes(result)


def capture(display=None, bbox=None):
    hostname, number, screen, local = _display(display if display is not None else os.environ.get("DISPLAY"))
    name, cookie = _authorization(hostname, number, local)
    stream = socket.socket(socket.AF_UNIX if local else (socket.AF_INET6 if ":" in hostname else socket.AF_INET), socket.SOCK_STREAM)
    with stream:
        stream.settimeout(10)
        stream.connect(f"/tmp/.X11-unix/X{number}" if local else (hostname, 6000 + number))
        packet = struct.pack("<BBHHHHH", 108, 0, 11, 0, len(name), len(cookie), 0)
        stream.sendall(packet + name + b"\0" * (-len(name) % 4) + cookie + b"\0" * (-len(cookie) % 4))
        reply = _read(stream, 8)
        major, minor, length = struct.unpack_from("<HHH", reply, 2)
        setup = _read(stream, length * 4)
        if reply[0] != 1 or (major, minor) != (11, 0):
            raise OSError("X11 server rejected the display connection or authentication")
        root, width, height, visual, depth, (bits, pad), byte_order, masks = _setup(setup, screen)
        if bbox is None:
            x, y, right, bottom = 0, 0, width, height
        elif len(bbox) == 4 and all(type(value) is int for value in bbox):
            x, y, right, bottom = bbox
        else:
            raise ValueError("Screenshot bbox requires four integer coordinates")
        if not 0 <= x < right <= width or not 0 <= y < bottom <= height or x > 32767 or y > 32767:
            raise ValueError("Screenshot bbox is outside the X11 screen")
        width, height = right - x, bottom - y
        if width * height > MAX_PIXELS:
            raise OSError("X11 screenshot exceeds the decoded pixel limit")
        stream.sendall(struct.pack("<BBHIhhHHI", 73, 2, 5, root, x, y, width, height, 0xFFFFFFFF))
        reply = _read(stream, 32)
        if reply[0] != 1 or reply[1] != depth or struct.unpack_from("<H", reply, 2)[0] != 1:
            raise OSError("X11 GetImage failed")
        length, returned_visual = struct.unpack_from("<II", reply, 4)
        stride = ((width * bits + pad - 1) // pad) * (pad // 8)
        expected = stride * height
        if returned_visual != visual or length * 4 != (expected + 3) & ~3:
            raise OSError("X11 GetImage returned an invalid visual or pixel size")
        raw = _read(stream, length * 4)
        return width, height, _pixels(raw[:expected], width, height, bits, pad, byte_order, masks)


def decode_wire(data):
    if len(data) < 12 or data[:4] != MAGIC:
        raise ValueError("Invalid sandbox screenshot header")
    width, height = struct.unpack_from(">II", data, 4)
    if not width or not height or width * height > MAX_PIXELS or len(data) > width * height * 3 + 65548:
        raise ValueError("Sandbox screenshot exceeds the validated size")
    inflater = zlib.decompressobj()
    pixels = inflater.decompress(data[12:], width * height * 3 + 1)
    if len(pixels) != width * height * 3 or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
        raise ValueError("Truncated or oversized sandbox screenshot pixels")
    return width, height, pixels


def main():
    if len(sys.argv) != 2:
        raise ValueError("X11 screenshot worker requires its owning display")
    width, height, pixels = capture(sys.argv[1])
    sys.stdout.buffer.write(MAGIC + struct.pack(">II", width, height) + zlib.compress(pixels, 1))


if __name__ == "__main__":
    main()
