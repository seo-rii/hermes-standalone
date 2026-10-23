"""Standard-library AAC-LC/SBR/PS ADTS/M4A decode; original planar float PCM."""
from .local_whisper_audio_aac_bits import AacError, UnsupportedAac
from .local_whisper_audio_aac_codec import AacDecoder
from .local_whisper_audio_aac_config import Config, RATES, defaults
from .local_whisper_audio_aac_mp4 import demux_mp4


def _skip_id3(data, position):
    while data[position:position + 3] == b'ID3':
        if len(data) - position < 10:
            raise AacError('Truncated ID3 tag before AAC')
        version, flags = data[position + 3], data[position + 5]
        size_bytes = data[position + 6:position + 10]
        if version not in (2, 3, 4) or any(value & 128 for value in size_bytes):
            raise AacError('Invalid ID3 tag before AAC')
        size = 0
        for value in size_bytes:
            size = size * 128 + value
        end = position + 10 + size + (10 if version == 4 and flags & 16 else 0)
        if end > len(data):
            raise AacError('ID3 tag exceeds AAC file')
        if version == 4 and flags & 16 and data[end - 10:end - 7] != b'3DI':
            raise AacError('Missing ID3 footer before AAC')
        position = end
    return position


def _adts_header(data, position):
    if len(data) - position < 7:
        raise AacError('Truncated ADTS frame header')
    head = int.from_bytes(data[position:position + 7], 'big')
    if head >> 44 != 0xfff or (head >> 41) & 3:
        raise AacError('Invalid ADTS synchronization or layer')
    profile, index = ((head >> 38) & 3) + 1, (head >> 34) & 15
    if index >= len(RATES):
        raise AacError('Reserved ADTS sampling frequency')
    channels, size = (head >> 30) & 7, (head >> 13) & 8191
    protected, blocks = not ((head >> 40) & 1), (head & 3) + 1
    header_size = 9 if protected else 7
    if size <= header_size or size > len(data) - position:
        raise AacError('ADTS frame exceeds file')
    if protected and blocks > 1:
        # The pinned FFmpeg parser only skips one CRC and explicitly reports
        # this protected multi-RDB feature as missing. Do not reinterpret CRC
        # positions as LC spectral data.
        raise UnsupportedAac('Protected ADTS with multiple raw data blocks is not supported')
    return Config(profile, index, channels), size, header_size, blocks


def _adts_frames(data):
    position = _skip_id3(data, 0)
    # The original ADTS demuxer can resynchronize past a leading non-audio
    # prefix. Require the whole remaining chain to be valid, not one accidental
    # AAC header embedded in another container.
    first = position
    known = (b'RIFF', b'RF64', b'BW64', b'OggS', b'fLaC', b'ADIF', b'\x1aE\xdf\xa3', b'\x30\x26\xb2\x75')
    if data[first:first + 4] in known or data[first + 4:first + 8] in (b'ftyp', b'styp', b'moov'):
        raise AacError('Input is another container, not an ADTS stream')
    while position + 7 <= len(data):
        position = data.find(b'\xff', position)
        if position < 0 or position + 7 > len(data):
            raise AacError('No complete AAC ADTS frame')
        if data[position] == 255 and data[position + 1] & 0xf6 == 0xf0:
            try:
                _adts_header(data, position)
                break
            except UnsupportedAac:
                raise
            except AacError:
                pass
        position += 1
    if position + 7 > len(data):
        raise AacError('No complete AAC ADTS frame')
    result = []
    while position < len(data):
        position = _skip_id3(data, position)
        if position == len(data):
            break
        if data[position:position + 3] == b'TAG' and len(data) - position == 128:
            break
        config, size, header, blocks = _adts_header(data, position)
        result.append((config, bytes(data[position + header:position + size]), blocks))
        position += size
    return result


def probe_aac(data):
    """Recognize a complete ADTS stream or an ISO BMFF audio container."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        return False
    data = bytes(data)
    try:
        position = _skip_id3(data, 0)
        if data[position + 4:position + 8] in (b'ftyp', b'styp', b'moov', b'free', b'wide'):
            return True
        return bool(_adts_frames(data))
    except AacError:
        return False


def decode_aac(data, *, max_frames=None, max_seconds=None):
    """Return (source rate, original planar float PCM, 'fltp', channel mask).

    max_frames bounds samples per channel, including the amount of raw decoder
    work. One second of pre-roll is allowed for MP4 edit-list decoder delay.
    HE-AAC SBR/PS uses the source-order float32 GENERIC transform profile.
    Other unported core object types raise UnsupportedAac.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise AacError('AAC input must be bytes')
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 0):
        raise AacError('AAC sample budget must be a nonnegative integer')
    if max_seconds is not None and (isinstance(max_seconds, bool) or not isinstance(max_seconds, int) or max_seconds < 0):
        raise AacError('AAC duration budget must be a nonnegative integer')
    data = bytes(data)
    position = _skip_id3(data, 0)
    if position:
        data = data[position:]
    mp4 = data[4:8] in (b'ftyp', b'styp', b'moov', b'mdat', b'free', b'wide')
    if mp4:
        config, packets, skip = demux_mp4(data)
        frames = ((config, bytes(data[packet.start:packet.start + packet.size]), 1, packet.discard) for packet in packets)
    else:
        frames = ((config, payload, blocks, False) for config, payload, blocks in _adts_frames(data))
        config, skip = None, 0
    requested_frames = max_frames
    decoder, result, rate, mask, raw_count, wire_format = None, [], None, None, 0, None
    for settings, payload, blocks, discard in frames:
        if decoder is None:
            decoder = AacDecoder(settings, adts=not mp4)
            config, rate = settings, settings.output_rate
            wire_format = settings.object_type, settings.sampling_index, settings.frame_length, settings.channel_config
            if max_seconds is not None:
                max_frames = rate * max_seconds if max_frames is None else min(max_frames, rate * max_seconds)
        elif (settings.object_type, settings.sampling_index, settings.frame_length) != wire_format[:3]:
            raise AacError('AAC format changes within one input')
        if settings.channel_config and settings.channel_config != wire_format[3]:
            raise AacError('AAC channel configuration changes within one input')
        if not mp4:
            if settings.channel_config:
                decoder._configure(defaults(settings.channel_config), settings.channel_config)
            else:
                decoder.config.channel_config = 0
        for _ in range(blocks):
            prospective = settings.frame_length * (2 if decoder.config.sbr == 1 and decoder.config.extension_rate > settings.rate else 1)
            if max_frames is not None and raw_count + prospective > max_frames + rate + 2 * prospective:
                raise AacError('AAC raw decoder work exceeds sample budget')
            pcm = decoder.decode(payload)
            new_rate = decoder.config.output_rate
            if not result:
                rate = new_rate
                if max_seconds is not None:
                    max_frames = rate * max_seconds if requested_frames is None else min(requested_frames, rate * max_seconds)
            elif rate != new_rate:
                raise AacError('AAC output sample rate changes within one input')
            raw_count += len(pcm[0])
            payload = payload[decoder.consumed:]
            if decoder.skip_samples:
                skip = max(skip, decoder.skip_samples)
                decoder.skip_samples = 0
            if not result:
                result, mask = [[] for _ in pcm], decoder.mask
            elif len(result) != len(pcm) or mask != decoder.mask:
                raise AacError('AAC output channel layout changes within one input')
            count = len(pcm[0])
            start = min(skip, count)
            skip -= start
            if not discard:
                if max_frames is not None and len(result[0]) + count - start > max_frames:
                    raise AacError('AAC decoded audio exceeds sample budget')
                for target, channel in zip(result, pcm):
                    target.extend(channel[start:])
        if payload and any(payload):
            raise AacError('Unexpected non-padding data after AAC raw blocks')
    if decoder is None or not result or not result[0]:
        raise AacError('AAC file contains no decoded audio samples')
    return rate, result, 'fltp', mask
