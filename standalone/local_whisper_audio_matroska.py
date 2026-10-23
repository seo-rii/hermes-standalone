"""Qualified Matroska file adapter for the qualified owned audio codecs."""
from .local_whisper_audio_matroska_demux import MatroskaError, _xiph_sizes, demux
from .local_whisper_audio_ogg import _OpusLink
from .local_whisper_audio_resample import convert_audio
from .local_whisper_audio_vorbis_codec import VorbisDecoder


def _samples(nanoseconds, rate):
    count = (abs(nanoseconds) * rate + 500000000) // 1000000000
    return -count if nanoseconds < 0 else count


class _OpusAudio:
    def __init__(self, track):
        self.link = _OpusLink(track.private)
        self.link.have_tags = True
        self.rate = 48000
        self.mask = self.link.mask
        self.channels = self.link.channels
        self.skip = self.link.skip

    def decode(self, packet):
        self.link.decode(packet)
        values = self.link.output
        self.link.output = [[] for _ in range(self.channels)]
        return values


class _VorbisFrames(VorbisDecoder):
    def _overlap(self, transformed, flag, size, previous):
        self.raw_frame = super()._overlap(transformed, flag, size, previous)
        return self.raw_frame


class _VorbisAudio:
    def __init__(self, track):
        private = track.private
        if not private or private[0] != 2:
            raise MatroskaError("Invalid Vorbis Matroska codec private header")
        sizes, position = _xiph_sizes(private, 1, len(private), 3)
        sizes.append(len(private) - position - sum(sizes))
        headers = []
        for size in sizes:
            if size < 0:
                raise MatroskaError("Invalid Vorbis Matroska private header lengths")
            headers.append(private[position:position + size])
            position += size
        self.decoder = _VorbisFrames(headers[0], headers[2])
        self.rate = self.decoder.rate
        self.channels = self.decoder.channels
        self.mask = self.decoder.mask
        self.skip = 0

    def decode(self, packet):
        first = self.decoder.first
        values = self.decoder.decode(packet)
        if first:
            values = [self.decoder.raw_frame[channel] for channel in self.decoder.order]
            self.skip = len(values[0])
        return values


class _FrameTrim:
    def __init__(self, decoder, track):
        self.decoder, self.remaining, self.first = decoder, 0, True
        self.initial = _samples(track.codec_delay, decoder.rate)

    def take(self, values, packet):
        count = len(values[0])
        padding = _samples(packet.discard_padding, self.decoder.rate)
        if self.first:
            self.remaining = self.decoder.skip
        if self.first and self.initial:
            self.remaining, padding = self.initial, 0
        elif packet.discard_padding:
            self.remaining = max(-padding, 0)
        removed = min(self.remaining, count)
        self.remaining -= removed
        self.first = False
        stop = count - padding if 0 < padding <= count - removed else count
        return [channel[removed:stop] for channel in values]


def decode_matroska(data, max_seconds=3600):
    track, packets, scale = demux(data)
    factories = {"A_OPUS": _OpusAudio, "A_VORBIS": _VorbisAudio}
    if track.codec not in factories:
        raise MatroskaError("Unqualified Matroska audio codec: " + track.codec)
    decoder = factories[track.codec](track)
    trim = _FrameTrim(decoder, track)
    output = [[] for _ in range(decoder.channels)]
    decoded = 0
    for packet in packets:
        try:
            values = decoder.decode(packet.data)
        except ValueError:
            # The original FW filter catches decoder INVALIDDATA. PyAV's
            # decoding generator then closes: only already yielded PCM survives.
            break
        values = trim.take(values, packet)
        for destination, channel in zip(output, values):
            destination.extend(channel)
        decoded += len(values[0])
        if max_seconds is not None and decoded > decoder.rate * max_seconds:
            raise MatroskaError("Whisper waveform exceeds the frontend one-hour bound")
    return decoder.rate, output, "flt", decoder.mask


def decode_audio(data, rate=16000, split=False):
    source_rate, values, fmt, mask = decode_matroska(data)
    return convert_audio(values, source_rate, rate, fmt, mask, split)
