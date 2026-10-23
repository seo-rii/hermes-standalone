"""Qualified local Whisper audio decoding, without native libraries or tools."""
from array import array
import os

from .local_whisper_audio_resample import convert_audio
from .local_whisper_audio_wave import decode_wave
from .local_whisper_audio_vorbis_bits import VorbisError


def _decode_container(data):
    if data[:4] in (b'RIFF', b'RIFX', b'RF64', b'BW64', b'riff'):
        return decode_wave(data)
    if data[:4] == b'OggS':
        from .local_whisper_audio_vorbis import decode_vorbis, probe_vorbis
        if probe_vorbis(data):
            rate, channels, sample_format, mask = decode_vorbis(data, stop_on_invalid=True)
            return rate, channels, 'flt', mask
        from .local_whisper_audio_ogg import decode_ogg_opus
        return decode_ogg_opus(data)
    if data[:4] == b'\x1a\x45\xdf\xa3':
        from .local_whisper_audio_matroska import decode_matroska
        return decode_matroska(data)
    if data[:4] == b'fLaC':
        from .local_whisper_audio_flac import decode_flac
        rate, channels, bits, mask = decode_flac(data)
        width = 16 if bits <= 16 else 32
        channels = [[v << (width - bits) for v in channel] for channel in channels]
        if len(channels[0]) > rate * 3600:
            raise ValueError('Whisper waveform exceeds the frontend one-hour bound')
        return rate, channels, 's16' if width == 16 else 's32', mask
    from .local_whisper_audio_aac import decode_aac, probe_aac
    from .local_whisper_audio_mp3 import decode_mp3, probe_mp3
    if probe_aac(data):
        rate, channels, sample_format, mask = decode_aac(data, max_seconds=3600)
    elif probe_mp3(data):
        rate, channels, sample_format, mask = decode_mp3(data)
    else:
        raise ValueError('Unsupported audio container or codec')
    if len(channels[0]) > rate * 3600:
        raise ValueError('Whisper waveform exceeds the frontend one-hour bound')
    return rate, channels, 'flt', mask


def decode_audio(input_file, sampling_rate=16000, split_stereo=False):
    """Return normalized float32 PCM after original no-dither S16 resampling."""
    if isinstance(input_file, (str, os.PathLike)):
        with open(input_file, 'rb') as source:
            data = source.read()
    else:
        data = input_file.read()
    if not isinstance(data, bytes):
        raise TypeError('Audio input must be a binary file')
    while data[:3] == b'ID3':
        if len(data) < 10 or any(v & 128 for v in data[6:10]):
            raise ValueError('Invalid ID3 audio prefix')
        size = sum(v << (7 * (3 - i)) for i, v in enumerate(data[6:10]))
        size += 10 + (10 if data[3] == 4 and data[5] & 16 else 0)
        if size > len(data):
            raise ValueError('Truncated ID3 audio prefix')
        data = data[size:]
    try:
        rate, channels, sample_format, mask = _decode_container(data)
    except VorbisError as error:
        if not error.invalid_frame:
            raise
        # Original Faster-Whisper filters decoder INVALIDDATA after successful
        # Ogg demux/open, then returns empty float32 PCM when no frame survived.
        return (array('f'), array('f')) if split_stereo else array('f')
    pcm = convert_audio(channels, rate, sampling_rate, sample_format, mask, split_stereo)
    output = tuple(array('f', (v / 32768 for v in channel)) for channel in pcm)
    return output if split_stereo else output[0]
