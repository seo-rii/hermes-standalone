"""Opus packet framing (RFC 6716), adapted from the Opus 1.3.1 parser.

Copyright (c) 2011 Xiph.Org Foundation and Skype Limited, Jean-Marc Valin
and Koen Vos. Original redistribution notices: licenses/opus*.txt.
"""
from dataclasses import dataclass


class OpusError(ValueError):
    pass


def samples_per_frame(toc, sample_rate=48000):
    if type(toc) is not int or not 0 <= toc <= 255 or sample_rate not in (8000,12000,16000,24000,48000):
        raise OpusError('Invalid Opus TOC or sample rate')
    if toc & 0x80:
        return (sample_rate << ((toc >> 3) & 3)) // 400
    if toc & 0x60 == 0x60:
        return sample_rate // (50 if toc & 8 else 100)
    configuration = (toc >> 3) & 3
    return sample_rate * 60 // 1000 if configuration == 3 else (sample_rate << configuration) // 100


@dataclass(frozen=True)
class Packet:
    toc: int
    frames: tuple
    payload_offset: int
    packet_offset: int
    padding: int

    @property
    def channels(self):
        return 2 if self.toc & 4 else 1

    @property
    def mode(self):
        return 'celt' if self.toc & 0x80 else 'hybrid' if self.toc & 0x60 == 0x60 else 'silk'

    @property
    def bandwidth(self):
        if self.toc & 0x80:
            return ('narrowband','wideband','superwideband','fullband')[(self.toc >> 5) & 3]
        if self.toc & 0x60 == 0x60:
            return 'fullband' if self.toc & 0x10 else 'superwideband'
        return ('narrowband','mediumband','wideband')[(self.toc >> 5) & 3]

    @property
    def duration_ms(self):
        return len(self.frames) * samples_per_frame(self.toc) / 48


def _parse_size(data, position, end):
    if position >= end:
        raise OpusError('Truncated Opus frame length')
    first = data[position]
    if first < 252:
        return first, position + 1
    if position + 1 >= end:
        raise OpusError('Truncated two-byte Opus frame length')
    return first + 4 * data[position + 1], position + 2


def _encode_size(length):
    if not 0 <= length <= 1275:
        raise OpusError('Opus frame exceeds 1275 bytes')
    if length < 252:
        return bytes((length,))
    first = 252 + (length & 3)
    return bytes((first,(length - first) >> 2))


def parse_packet(data, *, self_delimited=False):
    data = bytes(data)
    if not data:
        raise OpusError('Empty Opus packet')
    toc, position, end = data[0], 1, len(data)
    remaining, padding = end - position, 0
    sizes = []
    code = toc & 3
    cbr = False
    last_size = remaining
    if code == 0:
        count = 1
    elif code == 1:
        count, cbr = 2, True
        if not self_delimited:
            if remaining & 1:
                raise OpusError('Unequal two-frame CBR lengths')
            last_size = remaining >> 1
            sizes = [last_size]
    elif code == 2:
        count = 2
        first, updated = _parse_size(data,position,end)
        remaining -= updated - position
        position = updated
        if first > remaining:
            raise OpusError('Opus frame length exceeds packet')
        sizes = [first]
        last_size = remaining - first
    else:
        if remaining < 1:
            raise OpusError('Truncated Opus frame count')
        control = data[position]
        position += 1
        remaining -= 1
        count = control & 63
        if not count or count * samples_per_frame(toc) > 5760:
            raise OpusError('Opus packet exceeds 120 ms')
        if control & 64:
            while True:
                if remaining <= 0:
                    raise OpusError('Truncated Opus padding')
                value = data[position]
                position += 1
                remaining -= 1
                amount = 254 if value == 255 else value
                remaining -= amount
                padding += amount
                if remaining < 0:
                    raise OpusError('Opus padding exceeds packet')
                if value != 255:
                    break
        cbr = not control & 128
        if cbr:
            if not self_delimited:
                last_size = remaining // count
                if last_size * count != remaining:
                    raise OpusError('Unequal multi-frame CBR lengths')
                sizes = [last_size] * (count - 1)
        else:
            last_size = remaining
            for _ in range(count - 1):
                size, updated = _parse_size(data,position,position + remaining)
                consumed = updated - position
                remaining -= consumed
                position = updated
                if size > remaining:
                    raise OpusError('Opus VBR frame length exceeds packet')
                sizes.append(size)
                last_size -= consumed + size
            if last_size < 0:
                raise OpusError('Opus VBR frames exceed packet')
    if self_delimited:
        last_size, updated = _parse_size(data,position,position + remaining)
        consumed = updated - position
        remaining -= consumed
        position = updated
        if last_size > remaining:
            raise OpusError('Self-delimited Opus length exceeds packet')
        if cbr:
            if last_size * count > remaining:
                raise OpusError('Self-delimited Opus CBR lengths exceed packet')
            sizes = [last_size] * (count - 1)
        elif sum(sizes) + last_size > remaining:
            raise OpusError('Self-delimited Opus VBR lengths exceed packet')
    elif last_size > 1275:
        raise OpusError('Opus frame exceeds 1275 bytes')
    sizes.append(last_size)
    payload_offset = position
    frames = []
    for size in sizes:
        frames.append(data[position:position + size])
        position += size
    return Packet(toc,tuple(frames),payload_offset,position + padding,padding)


def pack_frames(toc, frames, *, padding=0, self_delimited=False):
    frames = tuple(bytes(frame) for frame in frames)
    if type(toc) is not int or not 0 <= toc <= 255 or not frames or len(frames) > 48:
        raise OpusError('Invalid Opus frame packing parameters')
    if type(padding) is not int or not 0 <= padding <= 65535:
        raise OpusError('Invalid Opus padding length')
    if len(frames) * samples_per_frame(toc) > 5760:
        raise OpusError('Opus packet exceeds 120 ms')
    sizes = [len(frame) for frame in frames]
    if max(sizes) > 1275:
        raise OpusError('Opus frame exceeds 1275 bytes')
    cbr = all(size == sizes[0] for size in sizes)
    if len(frames) == 1 and not padding:
        header = bytearray((toc & 252,))
    elif len(frames) == 2 and not padding:
        header = bytearray(((toc & 252) | (1 if cbr else 2),))
        if not cbr:
            header.extend(_encode_size(sizes[0]))
    else:
        header = bytearray(((toc & 252) | 3,len(frames) | (0 if cbr else 128) | (64 if padding else 0)))
        if padding:
            left = padding
            while left >= 254:
                header.append(255)
                left -= 254
            header.append(left)
        if not cbr:
            for size in sizes[:-1]:
                header.extend(_encode_size(size))
    if self_delimited:
        header.extend(_encode_size(sizes[-1]))
    return bytes(header) + b''.join(frames) + b'\0' * padding
