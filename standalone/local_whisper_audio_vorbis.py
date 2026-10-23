"""Stdlib Vorbis I/Ogg file decoding with original PCM/channel contracts."""
from standalone.local_whisper_audio_ogg import _pages
from .local_whisper_audio_vorbis_bits import Bits, VorbisError
from .local_whisper_audio_vorbis_codec import VorbisDecoder, VorbisParser


class _Link:
    def __init__(self, identification, previous_decoder=None, *, stop_on_invalid=False):
        self.headers = {1: identification}
        self.replacement_headers = {}
        self.previous_decoder = previous_decoder
        self.decoder = None
        self.parser = None
        self.output = None
        self.decoded = 0
        self.last_pts = 0
        self.final_pts = 0
        self.final_duration = 0
        self.have_audio = False
        self.stop_on_invalid, self.stopped = stop_on_invalid, False

    def header(self, packet):
        if not self.have_audio and packet and packet[0] > 5 and packet[0] & 1:
            return True  # Original ogg Vorbis header phase ignores unknown types.
        if len(packet) >= 7 and packet[1:7] == b'vorbis' and packet[0] in (1, 3, 5):
            kind = packet[0]
            if self.decoder is not None:
                self.replacement_headers[kind] = packet
                if all(k in self.replacement_headers for k in (1, 3, 5)):
                    self._configure(self.replacement_headers)
                    self.replacement_headers.clear()
                return True
            if kind == 3:
                if 1 not in self.headers or len(packet) < 16:
                    raise VorbisError('Invalid Vorbis comment header')
                vendor = int.from_bytes(packet[7:11], 'little')
                if vendor + 16 > len(packet):
                    raise VorbisError('Truncated Vorbis comment vendor')
            elif kind == 5:
                if 3 not in self.headers:
                    raise VorbisError('Missing Vorbis comment header')
                self.headers[kind] = packet
                self._configure(self.headers)
            self.headers[kind] = packet
            return True
        return False

    def _configure(self, headers):
        parser = VorbisParser(headers[1], headers[5])
        previous = self.decoder or self.previous_decoder
        if previous is not None:
            rate = parser.rate if 0 < parser.rate <= 0x7fffffff else previous.rate
            if (previous.rate, previous.channels, previous.mask) != (rate, parser.channels, parser.mask):
                raise VorbisError('Vorbis chained stream format changes')
        else:
            try:
                previous = VorbisDecoder(headers[1], headers[5])
            except VorbisError as exc:
                # avformat accepted the stream and duration setup. Native
                # avcodec returns INVALIDDATA here; original FW filters every
                # failed initialization frame. Negative signed sample rates
                # instead produce EINVAL, which the original does not filter.
                exc.invalid_frame = parser.rate <= 0x7fffffff
                raise
        # oggparsevorbis replaces its duration parser via NEW_EXTRADATA, while
        # the pinned native Vorbis decoder keeps its overlap/first-frame state.
        self.decoder = previous
        self.parser = parser
        self.parser_previous = parser.sizes[parser.modes[0][0]]
        if self.output is None:
            self.output = [[] for _ in range(self.decoder.channels)]

    def duration(self, packet, previous):
        bits = Bits(packet)
        if bits.read(1):
            return 0, previous
        mode = bits.index((len(self.parser.modes) - 1).bit_length(), len(self.parser.modes))
        flag = self.parser.modes[mode][0]
        if flag:
            previous = self.parser.sizes[bits.read(1)]
        current = self.parser.sizes[flag]
        return (previous + current) // 4, current

    def packet(self, packet, remaining, granule, eos, last):
        if self.decoder is None:
            raise VorbisError('Vorbis audio precedes its setup headers')
        self.have_audio = True
        if self.last_pts in (None, 0) and not eos and granule != 0xffffffffffffffff:
            total, previous = 0, self.parser.sizes[0]
            for value in remaining:
                duration, previous = self.duration(value, previous)
                total += duration
            self.last_pts = granule - total
            if not granule and total:
                self.last_pts = None
            self.final_pts = None
            self.parser_previous = self.parser.sizes[0]
        duration, self.parser_previous = self.duration(packet, self.parser_previous)
        discard = 0
        if eos:
            if self.last_pts is not None:
                self.final_pts, self.final_duration = self.last_pts, 0
            if last and self.final_pts is not None:
                skip = self.final_pts + self.final_duration + duration - granule
                if skip > 0:
                    discard = skip & 0xffffffff
                duration = granule - self.final_pts - self.final_duration
            self.final_duration += duration
        try:
            values = self.decoder.decode(packet)
        except VorbisError:
            if not self.stop_on_invalid:
                raise
            # PyAV's container.decode generator closes at INVALIDDATA. The
            # original FW filter catches it, then sees StopIteration; earlier
            # decoded frames survive and no later packet is returned.
            self.stopped = True
            return 0
        count = len(values[0])
        # Original discard_samples only applies padding fitting this frame.
        # It does not retrospectively truncate already returned Ogg pages.
        if 0 < discard <= count:
            values = [channel[:-discard] for channel in values]
            count -= discard
        for destination, channel in zip(self.output, values):
            destination.extend(channel)
        self.decoded += count
        self.last_pts = granule if last and granule != 0xffffffffffffffff else None
        return count

    def finish(self):
        if self.decoder is None:
            raise VorbisError('Missing Vorbis setup header')
        return self.output


def probe_vorbis(data):
    if data[:4] != b'OggS':
        return False
    pending = {}
    for flags, granule, serial, sequence, lacing, body in _pages(data):
        buffer = pending.setdefault(serial, bytearray())
        offset = 0
        for size in lacing:
            buffer.extend(body[offset:offset + size])
            offset += size
            if size < 255:
                packet = bytes(buffer)
                buffer.clear()
                if packet.startswith(b'\x01vorbis'):
                    return True
                if packet.startswith(b'OpusHead'):
                    return False
        if flags & 4:
            pending.pop(serial, None)
    return False


class _Packets:
    """Owned Ogg packet continuation and sequence state."""
    def __init__(self):
        self.pending, self.sequences = {}, {}

    def read(self, flags, serial, sequence, lacing, body):
        previous = self.sequences.get(serial)
        if previous is not None and sequence != (previous + 1) & 0xffffffff:
            raise VorbisError('Ogg Vorbis page sequence discontinuity')
        self.sequences[serial] = sequence
        buffer = self.pending.setdefault(serial, bytearray())
        if bool(flags & 1) != bool(buffer):
            raise VorbisError('Inconsistent Ogg Vorbis continued packet')
        offset, packets = 0, []
        for size in lacing:
            buffer.extend(body[offset:offset + size])
            offset += size
            if size == 255:
                continue
            packet = bytes(buffer)
            buffer.clear()
            packets.append(packet)
        return packets


class _VorbisFile:
    def __init__(self, max_frames, max_seconds, stop_on_invalid):
        self.max_frames, self.max_seconds = max_frames, max_seconds
        self.selected, self.link = None, None
        self.output, self.rate, self.mask, self.previous_decoder = None, None, None, None
        self.packets = _Packets()
        self.stop_on_invalid = stop_on_invalid

    def page(self, flags, granule, serial, sequence, lacing, body):
        packets = self.packets.read(flags, serial, sequence, lacing, body)
        for index, packet in enumerate(packets):
            if self.selected is None and packet.startswith(b'\x01vorbis'):
                self.selected = serial
                self.link = _Link(packet, self.previous_decoder, stop_on_invalid=self.stop_on_invalid)
                continue
            if serial != self.selected:
                continue
            header = self.link.header(packet)
            if not header:
                self.link.packet(packet, packets[index:], granule, bool(flags & 4), index + 1 == len(packets))
                if self.link.stopped:
                    return False
            if self.link.decoder is not None:
                self._check_output()
        if serial == self.selected and flags & 4:
            self._end_link(serial, granule)
        return True

    def _check_output(self):
        decoder = self.link.decoder
        if self.output is None:
            self.output = [[] for _ in range(decoder.channels)]
            self.rate, self.mask = decoder.rate, decoder.mask
        elif (self.rate, len(self.output), self.mask) != (decoder.rate, decoder.channels, decoder.mask):
            raise VorbisError('Vorbis chained stream format changes')
        budget = self.max_frames
        if self.max_seconds is not None:
            seconds_budget = self.rate * self.max_seconds
            budget = seconds_budget if budget is None else min(budget, seconds_budget)
        if budget is not None and len(self.output[0]) + self.link.decoded > budget + 8192:
            raise VorbisError('Vorbis decoded sample budget exceeded')

    def _end_link(self, serial, granule):
        if self.packets.pending.get(serial) or granule == 0xffffffffffffffff:
            raise VorbisError('Invalid Vorbis final page')
        for destination, channel in zip(self.output, self.link.finish()):
            destination.extend(channel)
        self.previous_decoder = self.link.decoder
        self.selected, self.link = None, None

    def finish(self):
        if self.link is not None:
            if not self.link.stopped and self.packets.pending.get(self.selected):
                raise VorbisError('Truncated Ogg Vorbis packet')
            for destination, channel in zip(self.output or [], self.link.finish()):
                destination.extend(channel)
        if self.output is None:
            raise VorbisError('No Ogg Vorbis audio stream')
        if ((self.max_frames is not None and len(self.output[0]) > self.max_frames)
                or (self.max_seconds is not None and len(self.output[0]) > self.rate * self.max_seconds)):
            raise VorbisError('Vorbis decoded sample budget exceeded')
        return self.rate, self.output, 'fltp', self.mask


def decode_vorbis(data, max_frames=None, max_seconds=3600, *, stop_on_invalid=False):
    """Return source-rate planar float32 in FFmpeg's native channel order."""
    state = _VorbisFile(max_frames, max_seconds, stop_on_invalid)
    for page in _pages(data):
        if not state.page(*page):
            break
    return state.finish()
