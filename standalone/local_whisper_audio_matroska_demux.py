"""Qualified stdlib Matroska/WebM EBML/audio packet reader, pinned FFmpeg8.1.2."""
from dataclasses import dataclass
import math
import struct
import zlib


class MatroskaError(ValueError):
    pass


@dataclass
class Element:
    code: int
    start: int
    end: int


@dataclass
class Track:
    number: int = 0
    kind: int = 0
    codec: str = ""
    private: bytes = b""
    rate: float = 8000.0
    channels: int = 1
    bits: int = 0
    codec_delay: int = 0
    default_duration: int = 0
    encoding: int = -1
    encoding_scope: int = 0
    encoding_settings: bytes = b""


@dataclass
class Packet:
    data: bytes
    timecode: int
    discard_padding: int


_LEVEL_ONE = {0x114D9B74, 0x1549A966, 0x1654AE6B, 0x1F43B675,
              0x1C53BB6B, 0x1941A469, 0x1043A770, 0x1254C367}


def vint(data, position, end, maximum=8, marker=False):
    if position >= end or not data[position]:
        raise MatroskaError("Invalid or truncated EBML number")
    width = 9 - data[position].bit_length()
    following = position + width
    if width > maximum or following > end:
        raise MatroskaError("Truncated or oversized EBML number")
    value = int.from_bytes(data[position:following], "big")
    if not marker:
        value ^= 1 << (7 * width)
    return value, following, width


def _header(data, position, end):
    code, position, _ = vint(data, position, end, 4, True)
    size, position, width = vint(data, position, end)
    following = None if size == (1 << (7 * width)) - 1 else position + size
    if following is not None and following > end:
        raise MatroskaError("Truncated EBML element")
    return code, position, following


def _cluster_end(data, position, end):
    while position < end:
        code, start, following = _header(data, position, end)
        if code in _LEVEL_ONE:
            return position
        if following is None:
            raise MatroskaError("Unsupported unknown-sized cluster child")
        position = following
    return end


def elements(data, start, end):
    position = start
    while position < end:
        code, payload, following = _header(data, position, end)
        if following is None:
            if code == 0x18538067:
                following = end
            elif code == 0x1F43B675:
                following = _cluster_end(data, payload, end)
            else:
                raise MatroskaError("Unsupported unknown-sized EBML element")
        yield Element(code, payload, following)
        position = following


def _uint(data, element, signed=False):
    if element.end - element.start > 8:
        raise MatroskaError("Oversized EBML integer")
    return int.from_bytes(data[element.start:element.end], "big", signed=signed)


def _float(data, element):
    raw = data[element.start:element.end]
    if len(raw) not in (4, 8):
        raise MatroskaError("Invalid EBML floating value")
    value = struct.unpack(">f" if len(raw) == 4 else ">d", raw)[0]
    if not math.isfinite(value) or value <= 0:
        raise MatroskaError("Invalid Matroska sampling frequency")
    return value


def _audio_fields(data, parent, track):
    output_rate = None
    for element in elements(data, parent.start, parent.end):
        if element.code == 0xB5:
            track.rate = _float(data, element)
        elif element.code == 0x78B5:
            output_rate = _float(data, element)
        elif element.code == 0x9F:
            track.channels = _uint(data, element)
        elif element.code == 0x6264:
            track.bits = _uint(data, element)
    if output_rate is not None:
        track.rate = output_rate


def _encoding(data, parent, track):
    entries = [item for item in elements(data, parent.start, parent.end) if item.code == 0x6240]
    if len(entries) != 1:
        return  # The pinned demuxer ignores combined encodings.
    track.encoding_scope, track.encoding = 1, 0
    kind = 0
    for item in elements(data, entries[0].start, entries[0].end):
        if item.code == 0x5032:
            track.encoding_scope = _uint(data, item)
        elif item.code == 0x5033:
            kind = _uint(data, item)
        elif item.code == 0x5034:
            for child in elements(data, item.start, item.end):
                if child.code == 0x4254:
                    track.encoding = _uint(data, child)
                elif child.code == 0x4255:
                    track.encoding_settings = data[child.start:child.end]
    if kind or track.encoding not in (0, 2, 3):
        # The fixed reference build does not compile bzip2 support.
        track.encoding_scope = 0


def _decode_payload(payload, track):
    if len(payload) >= 10000000:
        raise MatroskaError("Oversized Matroska content encoding input")
    if track.encoding == 3:
        return track.encoding_settings + payload
    if track.encoding == 0:
        decoder = zlib.decompressobj()
        try:
            result = decoder.decompress(payload, 30000000)
        except zlib.error as error:
            raise MatroskaError("Invalid Matroska zlib payload") from error
        if not decoder.eof:
            raise MatroskaError("Truncated or oversized Matroska zlib payload")
        return result
    if track.encoding == 2:
        from .local_whisper_audio_matroska_lzo import decode_lzo
        return decode_lzo(payload)
    raise MatroskaError("Invalid Matroska content encoding")


def _track(data, parent):
    track = Track()
    integer_fields = {0xD7: "number", 0x83: "kind", 0x56AA: "codec_delay", 0x23E383: "default_duration"}
    for element in elements(data, parent.start, parent.end):
        if element.code in integer_fields:
            setattr(track, integer_fields[element.code], _uint(data, element))
        elif element.code == 0x86:
            track.codec = data[element.start:element.end].decode("ascii")
        elif element.code == 0x63A2:
            track.private = data[element.start:element.end]
        elif element.code == 0xE1:
            _audio_fields(data, element, track)
        elif element.code == 0x6D80:
            _encoding(data, element, track)
    if not track.number:
        raise MatroskaError("Missing Matroska track number")
    if track.private and track.encoding_scope & 2:
        track.private = _decode_payload(track.private, track)
    return track


def _xiph_sizes(data, position, end, count):
    sizes = []
    for _ in range(count - 1):
        size = 0
        while True:
            if position >= end:
                raise MatroskaError("Truncated Xiph lace size")
            value = data[position]
            position += 1
            size += value
            if value != 255:
                break
        sizes.append(size)
    return sizes, position


def _ebml_sizes(data, position, end, count):
    first, position, _ = vint(data, position, end)
    sizes = [first]
    for _ in range(count - 2):
        value, position, width = vint(data, position, end)
        size = sizes[-1] + value - ((1 << (7 * width - 1)) - 1)
        if not 0 <= size <= 0x7FFFFFFF:
            raise MatroskaError("Invalid EBML lace size")
        sizes.append(size)
    return sizes, position


def _frames(data, position, end, lace):
    if not lace:
        return [data[position:end]]
    if position >= end:
        raise MatroskaError("Missing Matroska lace count")
    count = data[position] + 1
    position += 1
    if lace == 2:
        if (end - position) % count:
            raise MatroskaError("Invalid fixed Matroska lace size")
        sizes = [(end - position) // count] * count
    else:
        sizes, position = (_xiph_sizes if lace == 1 else _ebml_sizes)(data, position, end, count)
        sizes.append(end - position - sum(sizes))
    if min(sizes) < 0 or sum(sizes) != end - position:
        raise MatroskaError("Invalid Matroska lace payload")
    frames = []
    for size in sizes:
        frames.append(data[position:position + size])
        position += size
    return frames


def _block(data, parent, selected, timecode, discard_padding):
    number, position, _ = vint(data, parent.start, parent.end)
    if position + 3 > parent.end:
        raise MatroskaError("Truncated Matroska block")
    if number != selected.number:
        return []
    timestamp = timecode + int.from_bytes(data[position:position + 2], "big", signed=True)
    flags = data[position + 2]
    frames = _frames(data, position + 3, parent.end, (flags >> 1) & 3)
    if selected.encoding_scope & 1:
        frames = [_decode_payload(frame, selected) for frame in frames]
    # The pinned demuxer supplies DiscardPadding to each lace, in source order.
    return [Packet(frame, timestamp, discard_padding) for frame in frames if frame]


def _block_group(data, parent, selected, timecode):
    block, discard_padding = None, 0
    for element in elements(data, parent.start, parent.end):
        if element.code == 0xA1:
            block = element
        elif element.code == 0x75A2:
            discard_padding = _uint(data, element, signed=True)
    return [] if block is None else _block(data, block, selected, timecode, discard_padding)


def _packets(data, parent, selected):
    timecode = 0
    for element in elements(data, parent.start, parent.end):
        if element.code == 0xE7:
            timecode = _uint(data, element)
        elif element.code == 0xA3:
            yield from _block(data, element, selected, timecode, 0)
        elif element.code == 0xA0:
            yield from _block_group(data, element, selected, timecode)


def _segment(data):
    top = list(elements(data, 0, len(data)))
    if not top or top[0].code != 0x1A45DFA3:
        raise MatroskaError("Missing Matroska EBML header")
    for element in elements(data, top[0].start, top[0].end):
        if element.code == 0x4282 and data[element.start:element.end] not in (b"webm", b"matroska"):
            raise MatroskaError("Unsupported EBML document type")
    segment = next((element for element in top if element.code == 0x18538067), None)
    if segment is None:
        raise MatroskaError("Missing Matroska segment")
    return list(elements(data, segment.start, segment.end))


def demux(data):
    children = _segment(data)
    tracks, scale = [], 1000000
    for element in children:
        if element.code == 0x1654AE6B:
            tracks.extend(_track(data, child) for child in elements(data, element.start, element.end)
                          if child.code == 0xAE)
        elif element.code == 0x1549A966:
            for child in elements(data, element.start, element.end):
                if child.code == 0x2AD7B1:
                    scale = _uint(data, child)
    selected = next((track for track in tracks if track.kind == 2), None)
    if selected is None:
        raise MatroskaError("No Matroska audio stream")
    packets = [packet for element in children if element.code == 0x1F43B675
               for packet in _packets(data, element, selected)]
    return selected, packets, scale
