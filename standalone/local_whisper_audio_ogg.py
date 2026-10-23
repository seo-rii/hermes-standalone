"""Ogg Opus framing, mapping, pre-skip and end trimming (RFC 7845)."""
import struct

from standalone.opus import OpusError, pack_frames, parse_packet, samples_per_frame
from standalone.opus_decoder import OpusDecoder
from .local_whisper_audio_resample import f32


def _crc_table():
    table = []
    for value in range(256):
        value <<= 24
        for _ in range(8):
            value = (value << 1) ^ (0x04c11db7 if value & 0x80000000 else 0)
        table.append(value & 0xffffffff)
    return table


_CRC = _crc_table()


def _pages(data):
    position = 0
    while position < len(data):
        if position + 27 > len(data) or data[position:position + 4] != b'OggS':
            raise ValueError('Invalid or truncated Ogg page')
        if data[position + 4] != 0:
            raise ValueError('Unsupported Ogg version')
        flags = data[position + 5]
        granule, serial, sequence, checksum = struct.unpack_from('<QIII', data, position + 6)
        if granule != 0xffffffffffffffff and granule > (1 << 62):
            raise ValueError('Unsupported huge Opus granule position')
        count = data[position + 26]
        end = position + 27 + count
        if end > len(data):
            raise ValueError('Truncated Ogg lacing table')
        lacing = data[position + 27:end]
        body_end = end + sum(lacing)
        if body_end > len(data):
            raise ValueError('Truncated Ogg page body')
        page = data[position:body_end]
        value = 0
        for i, byte in enumerate(page):
            if 22 <= i < 26:
                byte = 0
            value = ((value << 8) ^ _CRC[((value >> 24) & 255) ^ byte]) & 0xffffffff
        if value != checksum:
            raise ValueError('Ogg page checksum mismatch')
        yield flags, granule, serial, sequence, lacing, data[end:body_end]
        position = body_end


_VORBIS_ORDER = {1: [0], 2: [0, 1], 3: [0, 2, 1], 4: [0, 1, 2, 3],
                 5: [0, 2, 1, 3, 4], 6: [0, 2, 1, 5, 3, 4],
                 7: [0, 2, 1, 6, 5, 3, 4], 8: [0, 2, 1, 7, 5, 6, 3, 4]}
_MASKS = {1: 4, 2: 3, 3: 7, 4: 0x33, 5: 0x37, 6: 0x3f, 7: 0x70f, 8: 0x63f}


class _OpusLink:
    def __init__(self, header):
        if len(header) < 19 or header[:8] != b'OpusHead' or header[8] & 0xf0:
            raise ValueError('Invalid Opus identification header')
        self.channels = header[9]
        self.skip, self.gain = struct.unpack_from('<H4xh', header, 10)
        family = header[18]
        if family == 0:
            if self.channels not in (1, 2):
                raise ValueError('Invalid Opus mapping family 0 channels')
            streams, coupled = 1, self.channels - 1
            mapping = list(range(self.channels))
        elif family in (1, 255):
            if len(header) < 21 + self.channels or not self.channels:
                raise ValueError('Truncated Opus channel mapping')
            if family == 1 and self.channels > 8:
                raise ValueError('Invalid Opus surround channel count')
            streams, coupled = header[19:21]
            mapping = list(header[21:21 + self.channels])
            if not streams or coupled > streams or streams + coupled > 255:
                raise ValueError('Invalid Opus stream mapping')
            if any(value != 255 and value >= streams + coupled for value in mapping):
                raise ValueError('Invalid Opus mapped channel')
        else:
            raise ValueError('Unsupported Opus projection channel mapping')
        self.mask = _MASKS.get(self.channels) if family in (0, 1) else None
        order = _VORBIS_ORDER[self.channels] if family == 1 else list(range(self.channels))
        self.mapping = [mapping[ch] for ch in order]
        self.decoders = [OpusDecoder(48000, 2 if i < coupled else 1) for i in range(streams)]
        self.output = [[] for _ in range(self.channels)]
        self.decoded = 0
        self.initial_offset = None
        self.final_granule = None
        self.have_tags = False

    def decode(self, packet):
        if not self.have_tags:
            if not packet.startswith(b'OpusTags') or len(packet) < 16:
                raise ValueError('Missing Opus comment header')
            vendor = int.from_bytes(packet[8:12], 'little')
            if 16 + vendor > len(packet):
                raise ValueError('Truncated Opus comment header')
            self.have_tags = True
            return 0
        if packet.startswith(b'OpusTags'):
            return 0
        decoded, offset, duration = [], 0, None
        for i, decoder in enumerate(self.decoders):
            parsed = parse_packet(packet[offset:], self_delimited=i + 1 < len(self.decoders))
            offset += parsed.packet_offset
            size = samples_per_frame(parsed.toc) * len(parsed.frames)
            if duration is not None and size != duration:
                raise ValueError('Opus multistream duration mismatch')
            duration = size
            pcm = decoder.decode_float(pack_frames(parsed.toc, parsed.frames), 5760)
            decoded.extend(pcm[ch::decoder.channels] for ch in range(decoder.channels))
        if offset != len(packet):
            raise ValueError('Trailing Opus multistream packet data')
        gain = f32(10 ** (self.gain / 5120.)) if self.gain else 1.
        for channel, mapped in zip(self.output, self.mapping):
            channel.extend([0.] * duration if mapped == 255 else
                           decoded[mapped] if gain == 1. else [f32(v * gain) for v in decoded[mapped]])
        self.decoded += duration
        return duration

    def finish(self):
        end = self.decoded
        if self.final_granule is not None:
            end = min(end, self.final_granule - (self.initial_offset or 0))
        if end < 0:
            raise ValueError('Invalid Opus end granule')
        return [channel[min(self.skip, end):end] for channel in self.output]


def decode_ogg_opus(data, max_seconds=3600):
    """Decode the first Ogg Opus audio stream, including compatible chains."""
    selected, link, pending, sequences, all_channels, mask = None, None, {}, {}, None, None
    for flags, granule, serial, sequence, lacing, body in _pages(data):
        previous = sequences.get(serial)
        if previous is not None and sequence != (previous + 1) & 0xffffffff:
            raise ValueError('Ogg page sequence discontinuity')
        sequences[serial] = sequence
        buffer = pending.setdefault(serial, bytearray())
        if bool(flags & 1) != bool(buffer):
            raise ValueError('Inconsistent Ogg continued packet')
        offset, page_duration = 0, 0
        for size in lacing:
            buffer.extend(body[offset:offset + size])
            offset += size
            if size == 255:
                continue
            packet = bytes(buffer)
            buffer.clear()
            if selected is None and packet.startswith(b'\x01vorbis'):
                raise ValueError('Ogg Vorbis audio decoding is not implemented')
            if selected is None and packet.startswith(b'OpusHead'):
                selected, link = serial, _OpusLink(packet)
                if all_channels is None:
                    all_channels, mask = [[] for _ in range(link.channels)], link.mask
                elif len(all_channels) != link.channels or mask != link.mask:
                    raise ValueError('Opus chained stream channel change')
            elif serial == selected:
                page_duration += link.decode(packet)
                if max_seconds is not None and link.decoded > (max_seconds * 48000 + link.skip + 5760):
                    raise ValueError('Whisper waveform exceeds the frontend one-hour bound')
        if serial == selected:
            if page_duration and link.initial_offset is None:
                link.initial_offset = max(0, granule - link.decoded) if not flags & 4 and granule != 0xffffffffffffffff else 0
            if flags & 4:
                if buffer or granule == 0xffffffffffffffff:
                    raise ValueError('Invalid Opus final page')
                link.final_granule = granule
                for dest, channel in zip(all_channels, link.finish()):
                    dest.extend(channel)
                selected, link = None, None
    if link is not None:
        if pending.get(selected):
            raise ValueError('Truncated Ogg Opus packet')
        for dest, channel in zip(all_channels, link.finish()):
            dest.extend(channel)
    if all_channels is None:
        raise ValueError('No Ogg Opus audio stream')
    if max_seconds is not None and len(all_channels[0]) > max_seconds * 48000:
        raise ValueError('Whisper waveform exceeds the frontend one-hour bound')
    return 48000, all_channels, 'flt', mask
