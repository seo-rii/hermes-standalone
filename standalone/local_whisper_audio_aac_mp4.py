"""Bounded ISO BMFF AAC sample tables, fragments and edit lists (FFmpeg LGPL)."""
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
import struct
from .local_whisper_audio_aac_bits import AacError, UnsupportedAac
from .local_whisper_audio_aac_config import parse_config


def _uint(data, position, width=4, *, signed=False):
    if position < 0 or position + width > len(data):
        raise AacError('Truncated MP4 field')
    return int.from_bytes(data[position:position + width], 'big', signed=signed)


def _time_version(data):
    version = _uint(data, 0, 1)
    if version > 1:
        raise AacError('Invalid MP4 time-box version')
    return version


@dataclass(frozen=True)
class Box:
    kind: bytes
    start: int
    payload: int
    end: int


def boxes(data, start=0, end=None):
    end = len(data) if end is None else end
    if not 0 <= start <= end <= len(data):
        raise AacError('Invalid MP4 box boundary')
    while start < end:
        if end - start < 8:
            raise AacError('Truncated MP4 box header')
        size, kind = _uint(data, start), bytes(data[start + 4:start + 8])
        header = 8
        if size == 1:
            size, header = _uint(data, start + 8, 8), 16
        elif size == 0:
            size = end - start
        if kind == b'uuid':
            header += 16
        if size < header or size > end - start:
            raise AacError('MP4 box exceeds its parent')
        yield Box(kind, start, start + header, start + size)
        start += size


def _children(data, parent):
    return list(boxes(data, parent.payload, parent.end))


def _one(children, kind, *, required=True):
    matches = [box for box in children if box.kind == kind]
    if len(matches) > 1 or (required and not matches):
        raise AacError('Missing or repeated MP4 ' + kind.decode('latin1') + ' box')
    return matches[0] if matches else None


def _payload(data, box):
    return memoryview(data)[box.payload:box.end]


def _descriptors(data, start=0, end=None):
    end = len(data) if end is None else end
    while start < end:
        kind, start = _uint(data, start, 1), start + 1
        size = 0
        for _ in range(4):
            byte, start = _uint(data, start, 1), start + 1
            size = (size << 7) | (byte & 127)
            if not byte & 128:
                break
        else:
            raise AacError('Invalid MP4 descriptor length')
        if size > end - start:
            raise AacError('MP4 descriptor exceeds its container')
        yield kind, start, start + size
        start += size


def _audio_specific_config(data):
    """ISO/IEC 14496 ES_Descriptor and DecoderConfigDescriptor nesting."""
    found = []
    for kind, start, end in _descriptors(data, 4):
        if kind == 3:
            flags = _uint(data, start + 2, 1)
            nested = start + 3 + (2 if flags & 128 else 0)
            if flags & 64:
                nested += 1 + _uint(data, nested, 1)
            nested += 2 if flags & 32 else 0
            for child, a, b in _descriptors(data, nested, end):
                if child != 4:
                    continue
                indication = _uint(data, a, 1)
                if indication not in (0x40, 0x66, 0x67, 0x68):
                    raise UnsupportedAac('MP4 elementary stream is not MPEG AAC')
                if b - a < 13:
                    raise AacError('Truncated MPEG decoder configuration')
                found.extend(bytes(data[c:d]) for tag, c, d in _descriptors(data, a + 13, b) if tag == 5)
        elif kind == 4:
            if end - start < 13:
                raise AacError('Truncated MPEG decoder configuration')
            found.extend(bytes(data[c:d]) for tag, c, d in _descriptors(data, start + 13, end) if tag == 5)
    if len(found) != 1:
        raise AacError('MP4 AAC needs one AudioSpecificConfig')
    return found[0]


def _sample_descriptions(data, stsd):
    payload = _payload(data, stsd)
    count = _uint(payload, 4)
    entries = list(boxes(payload, 8))
    if len(entries) != count or not entries:
        raise AacError('Invalid MP4 sample description count')
    configs = []
    for entry in entries:
        if entry.kind == b'enca':
            raise UnsupportedAac('Encrypted MP4 audio is not a clear AAC file')
        if entry.kind != b'mp4a':
            raise UnsupportedAac('MP4 audio sample entry is not AAC mp4a')
        content = _payload(payload, entry)
        version = _uint(content, 8, 2)
        if version not in (0, 1, 2):
            raise UnsupportedAac('Unsupported QuickTime audio sample entry version')
        children = list(boxes(content, (28, 44, 64)[version]))
        wave = _one(children, b'wave', required=False)
        if wave is not None:
            children += _children(content, wave)
        esds = _one(children, b'esds')
        configs.append(parse_config(_audio_specific_config(_payload(content, esds))))
    first = configs[0]
    if any(config != first for config in configs):
        raise UnsupportedAac('MP4 switches AAC decoder configuration within one track')
    return first, len(configs)


def _table(data, box, words):
    content = _payload(data, box)
    count = _uint(content, 4)
    if len(content) != 8 + count * words * 4:
        raise AacError('Invalid MP4 sample table size')
    return list(struct.iter_unpack('>' + 'I' * words, content[8:]))


def _sample_sizes(data, stbl):
    stsz, stz2 = _one(stbl, b'stsz', required=False), _one(stbl, b'stz2', required=False)
    if (stsz is None) == (stz2 is None):
        raise AacError('MP4 needs exactly one sample size table')
    content = _payload(data, stsz or stz2)
    fixed, count = _uint(content, 4), _uint(content, 8)
    if stsz is not None:
        if fixed:
            if len(content) != 12 or count * fixed > len(data):
                raise AacError('MP4 fixed sample sizes exceed file')
            return [fixed] * count
        if len(content) != 12 + count * 4:
            raise AacError('Invalid MP4 variable sample sizes')
        return list(struct.unpack('>' + str(count) + 'I', content[12:]))
    width = fixed & 255
    if width not in (4, 8, 16) or len(content) != 12 + (count * width + 7) // 8:
        raise AacError('Invalid MP4 compact sample sizes')
    if width == 4:
        return [((content[12 + i // 2] >> (0 if i & 1 else 4)) & 15) for i in range(count)]
    if width == 8:
        return list(content[12:])
    return list(struct.unpack('>' + str(count) + 'H', content[12:]))


@dataclass
class Packet:
    start: int
    size: int
    timestamp: int
    duration: int
    discard: bool = False


def _sample_packets(data, stbl, descriptions):
    sizes = _sample_sizes(data, stbl)
    chunks32, chunks64 = _one(stbl, b'stco', required=False), _one(stbl, b'co64', required=False)
    if chunks32 is not None and chunks64 is not None:
        raise AacError('MP4 has two chunk offset tables')
    if chunks32 is None and chunks64 is None:
        if sizes:
            raise AacError('Missing MP4 chunk offset table')
        return []
    content = _payload(data, chunks32 or chunks64)
    count, width = _uint(content, 4), 4 if chunks32 is not None else 8
    if len(content) != 8 + count * width:
        raise AacError('Invalid MP4 chunk offsets')
    offsets = [_uint(content, 8 + i * width, width) for i in range(count)]
    stsc = _table(data, _one(stbl, b'stsc'), 3)
    if sizes and (not stsc or stsc[0][0] != 1):
        raise AacError('MP4 sample-to-chunk table must begin at chunk one')
    if any(first < 1 or not samples or not 1 <= desc <= descriptions or
           (i and first <= stsc[i - 1][0]) for i, (first, samples, desc) in enumerate(stsc)):
        raise AacError('Invalid MP4 sample-to-chunk mapping')
    packets, index, section = [], 0, 0
    for chunk, start in enumerate(offsets, 1):
        while section + 1 < len(stsc) and stsc[section + 1][0] <= chunk:
            section += 1
        if not stsc:
            raise AacError('Empty MP4 sample-to-chunk mapping')
        count = stsc[section][1]
        if index + count > len(sizes):
            raise AacError('MP4 chunk exceeds sample count')
        for size in sizes[index:index + count]:
            packets.append(Packet(start, size, 0, 0))
            start += size
        index += count
    if index != len(sizes):
        raise AacError('MP4 sample table has unaddressed samples')
    timing = _table(data, _one(stbl, b'stts'), 2)
    if sum(count for count, _ in timing) != len(packets):
        raise AacError('MP4 timing/sample counts differ')
    index = timestamp = 0
    for count, duration in timing:
        if not duration and count:
            raise AacError('MP4 audio sample duration is zero')
        for _ in range(count):
            packets[index].timestamp, packets[index].duration = timestamp, duration
            index, timestamp = index + 1, timestamp + duration
    ctts = _one(stbl, b'ctts', required=False)
    if ctts is not None:
        content = _payload(data, ctts)
        rows = _table(data, ctts, 2)
        if sum(count for count, _ in rows) != len(packets):
            raise AacError('MP4 composition time/sample counts differ')
        index = 0
        for count, offset in rows:
            offset = offset - 2**32 if content[0] == 1 and offset & 2**31 else offset
            for _ in range(count):
                packets[index].timestamp += offset
                index += 1
    return packets


def _fragment_packets(data, roots, track_id, defaults, initial_time, descriptions):
    packets, next_time = [], initial_time
    for moof in (box for box in roots if box.kind == b'moof'):
        previous_end = moof.start
        for traf in (box for box in _children(data, moof) if box.kind == b'traf'):
            children = _children(data, traf)
            tfhd = _payload(data, _one(children, b'tfhd'))
            flags, identity, cursor = _uint(tfhd, 0) & 0xffffff, _uint(tfhd, 4), 8
            base = _uint(tfhd, cursor, 8) if flags & 1 else moof.start if flags & 0x20000 else previous_end
            cursor += 8 if flags & 1 else 0
            description, duration, size, sample_flags = defaults.get(identity, (1, 0, 0, 0))
            for bit, name in ((2, 'description'), (8, 'duration'), (16, 'size'), (32, 'sample_flags')):
                if flags & bit:
                    value = _uint(tfhd, cursor)
                    cursor += 4
                    if name == 'description': description = value
                    elif name == 'duration': duration = value
                    elif name == 'size': size = value
                    else: sample_flags = value
            tfdt = _one(children, b'tfdt', required=False)
            timestamp = next_time
            if tfdt is not None:
                content = _payload(data, tfdt)
                timestamp = _uint(content, 4, 8 if _time_version(content) else 4)
            position = base
            for run in (box for box in children if box.kind == b'trun'):
                content = _payload(data, run)
                run_flags, count, cursor = _uint(content, 0) & 0xffffff, _uint(content, 4), 8
                if count > len(data):
                    raise AacError('MP4 fragment sample count exceeds file')
                if run_flags & 1:
                    position, cursor = base + _uint(content, cursor, signed=True), cursor + 4
                if run_flags & 4:
                    cursor += 4
                for _ in range(count):
                    sample_duration, sample_size, composition = duration, size, 0
                    for bit in (0x100, 0x200, 0x400, 0x800):
                        if run_flags & bit:
                            value = _uint(content, cursor, signed=bit == 0x800 and content[0] == 1)
                            cursor += 4
                            if bit == 0x100: sample_duration = value
                            elif bit == 0x200: sample_size = value
                            elif bit == 0x800: composition = value
                    if identity == track_id:
                        if not sample_duration or not sample_size or not 1 <= description <= descriptions:
                            raise AacError('Missing MP4 fragment sample parameters')
                        packets.append(Packet(position, sample_size, timestamp + composition, sample_duration))
                    timestamp += sample_duration
                    position += sample_size
                if cursor != len(content):
                    raise AacError('Invalid MP4 track run field count')
            previous_end = position
            if identity == track_id:
                next_time = timestamp
    return packets


def _rescale(value, numerator, denominator):
    if value < 0:
        return -_rescale(-value, numerator, denominator)
    return (value * numerator + denominator // 2) // denominator


def _itunes_priming(data, entries, depth=0):
    if depth > 10:
        raise AacError('MP4 metadata box nesting exceeds source demuxer limit')
    found = None
    for entry in entries:
        if entry.kind in (b'udta', b'ilst'):
            value = _itunes_priming(data, _children(data, entry), depth + 1)
            if value is not None:
                found = value
        elif entry.kind == b'meta':
            children = list(boxes(data, entry.payload + 4, entry.end))
            handlers = [i for i, child in enumerate(children) if child.kind == b'hdlr']
            if not handlers:
                continue
            value = _itunes_priming(data, children[handlers[0]:], depth + 1)
            if value is not None:
                found = value
        elif entry.kind == b'----':
            values = {}
            for item in _children(data, entry):
                if item.kind in (b'mean', b'name', b'data'):
                    n = 8 if item.kind == b'data' else 4
                    content = bytes(_payload(data, item)[n:]).split(b'\0')[0]
                    values.setdefault(item.kind, content)
            if values.get(b'mean') and values.get(b'name') == b'iTunSMPB':
                fields = values.get(b'data', b'').split()
                if len(fields) >= 4:
                    try:
                        _, priming, _, _ = (int(value, 16) for value in fields[:4])
                    except ValueError:
                        continue
                    if 0 < priming < 16384:
                        found = priming, entry.start
    return found


def _selected_edits(packets, edits):
    for begin, end, start, duration, first in edits:
        for index in range(begin, end):
            packet = packets[index]
            outside = packet.timestamp < start or packet.timestamp >= start + duration
            overlap = packet.timestamp < start < packet.timestamp + packet.duration
            yield Packet(packet.start, packet.size, packet.timestamp, packet.duration,
                         outside and not (first and overlap))


def _edit_packets(data, track, packets, timescale, movie_scale, rate, fragmented):
    edts = _one(_children(data, track), b'edts', required=False)
    if edts is None:
        return packets, 0
    elst = _one(_children(data, edts), b'elst', required=False)
    if elst is None:
        return packets, 0
    content = _payload(data, elst)
    version, count = _uint(content, 0, 1), _uint(content, 4)
    width = 20 if version == 1 else 12
    if version > 1 or len(content) != 8 + count * width:
        raise AacError('Invalid MP4 edit list')
    if fragmented:
        # mov.c disables its advanced index rewrite for fragmented input.
        # A sole non-empty edit sets AAC start_pad, including the common
        # duration=0 edit in a delayed-moov initialization segment.
        entries = []
        for i in range(count):
            cursor, n = 8 + i * width, 8 if version else 4
            entries.append((_uint(content, cursor, n), _uint(content, cursor + n, n, signed=True)))
        if entries and entries[0][1] == -1:
            entries = entries[1:]
        start = entries[0][1] if len(entries) == 1 else 0
        return packets, _rescale(max(0, start), rate, timescale)
    edits, skip, first_audio = [], 0, True
    timestamps = [packet.timestamp for packet in packets]
    for i in range(count):
        cursor, length_width = 8 + i * width, 8 if version else 4
        duration = _rescale(_uint(content, cursor, length_width), timescale, movie_scale)
        start = _uint(content, cursor + length_width, length_width, signed=True)
        _uint(content, cursor + length_width * 2)  # mov.c records, but does not apply, the edit rate.
        if start == -1 or not duration:
            continue
        if start < 0:
            raise AacError('Invalid MP4 audio edit start')
        search = max(start - timescale, timestamps[0] if timestamps else 0)
        begin = max(0, bisect_right(timestamps, search) - 1)
        end = min(len(packets), max(begin + 1, bisect_left(timestamps, start + duration)))
        if first_audio:
            for j in range(begin, end):
                packet = packets[j]
                if packet.timestamp < start:
                    skip += min(packet.duration, start - packet.timestamp)
        edits.append((begin, end, start, duration, first_audio))
        first_audio = False
    return _selected_edits(packets, edits), _rescale(skip, rate, timescale)


def demux_mp4(data):
    roots = list(boxes(data))
    moov = _one(roots, b'moov')
    movie = _children(data, moov)
    mvhd = _payload(data, _one(movie, b'mvhd'))
    movie_scale = _uint(mvhd, 20 if _time_version(mvhd) else 12)
    if not movie_scale:
        raise AacError('MP4 movie timescale is zero')
    for track in (box for box in movie if box.kind == b'trak'):
        children = _children(data, track)
        mdia = _children(data, _one(children, b'mdia'))
        handler = _payload(data, _one(mdia, b'hdlr'))
        if bytes(handler[8:12]) != b'soun':
            continue
        tkhd = _payload(data, _one(children, b'tkhd'))
        track_id = _uint(tkhd, 20 if _time_version(tkhd) else 12)
        mdhd = _payload(data, _one(mdia, b'mdhd'))
        timescale = _uint(mdhd, 20 if _time_version(mdhd) else 12)
        if not timescale:
            raise AacError('MP4 audio timescale is zero')
        minf = _children(data, _one(mdia, b'minf'))
        stbl = _children(data, _one(minf, b'stbl'))
        config, descriptions = _sample_descriptions(data, _one(stbl, b'stsd'))
        packets = _sample_packets(data, stbl, descriptions)
        mvex = _one(movie, b'mvex', required=False)
        defaults = {}
        if mvex is not None:
            for trex in _children(data, mvex):
                if trex.kind == b'trex':
                    content = _payload(data, trex)
                    if len(content) != 24:
                        raise AacError('Invalid MP4 track extends defaults')
                    defaults[_uint(content, 4)] = tuple(_uint(content, i) for i in (8, 12, 16, 20))
        packets += _fragment_packets(data, roots, track_id, defaults,
                                     packets[-1].timestamp + packets[-1].duration if packets else 0, descriptions)
        if not packets:
            raise AacError('MP4 AAC track contains no samples')
        media = [(box.payload, box.end) for box in roots if box.kind == b'mdat']
        for packet in packets:
            if packet.size <= 0 or not any(start <= packet.start and packet.start + packet.size <= end for start, end in media):
                raise AacError('MP4 audio sample is outside media data')
        packets, skip = _edit_packets(data, track, packets, timescale, movie_scale, config.rate,
                                     any(box.kind == b'moof' for box in roots))
        priming = _itunes_priming(data, children)
        # Global custom metadata is attached to the last stream by mov.c.
        tracks = [box for box in movie if box.kind == b'trak']
        if track == tracks[-1]:
            priming = _itunes_priming(data, movie) or priming
        if priming is not None and (_one(children, b'edts', required=False) is None or priming[1] > track.end):
            skip = priming[0]
        return config, packets, skip
    raise AacError('MP4 contains no audio track')
