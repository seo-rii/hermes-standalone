"""RIFF/RIFX/RF64 linear PCM and IEEE-float WAVE demuxing."""
import struct


def _chunks(data):
    if data[:16] == b'riff\x2e\x91\xcf\x11\xa5\xd6\x28\xdb\x04\xc1\0\0':
        if len(data) < 40 or data[24:40] != b'wave\xf3\xac\xd3\x11\x8c\xd1\0\xc0\x4f\x8e\xdb\x8a':
            raise ValueError('Invalid Wave64 header')
        position = 40
        while position + 24 <= len(data):
            name = data[position:position + 4]
            size = struct.unpack_from('<Q', data, position + 16)[0]
            if size < 24:
                raise ValueError('Invalid Wave64 chunk size')
            yield name, data[position + 24:min(position + size, len(data))]
            position += (size + 7) & ~7
        return
    if len(data) < 12 or data[:4] not in (b'RIFF', b'RIFX', b'RF64', b'BW64') or data[8:12] != b'WAVE':
        raise ValueError('Invalid WAVE header')
    endian = '>' if data[:4] == b'RIFX' else '<'
    position, rf_size, table = 12, None, {}
    while position + 8 <= len(data):
        name = data[position:position + 4]
        size = struct.unpack_from(endian + 'I', data, position + 4)[0]
        position += 8
        if size == 0xffffffff:
            if name == b'data':
                size = rf_size if rf_size is not None else len(data) - position
            elif name in table:
                size = table[name].pop(0)
            else:
                raise ValueError('Missing RF64 extended chunk size')
        if name == b'ds64' and size >= 28 and position + size <= len(data):
            rf_size = struct.unpack_from('<Q', data, position + 8)[0]
            entries = struct.unpack_from('<I', data, position + 24)[0]
            if 28 + 12 * entries > size:
                raise ValueError('Truncated RF64 size table')
            for i in range(entries):
                offset = position + 28 + i * 12
                table.setdefault(data[offset:offset + 4], []).append(struct.unpack_from('<Q', data, offset + 4)[0])
        if name == b'fmt ' and position + size > len(data):
            raise ValueError('Truncated WAVE format')
        yield name, data[position:min(position + size, len(data))]
        position += size + (size & 1)


def _law(byte, alaw):
    if alaw:
        value = byte ^ 0x55
        sample = (value & 15) << 4
        segment = (value >> 4) & 7
        sample += 8 if segment == 0 else 264
        if segment > 1:
            sample <<= segment - 1
        return sample if value & 128 else -sample
    value = byte ^ 255
    sample = ((value & 15) << 3) + 132
    sample <<= (value >> 4) & 7
    return 132 - sample if value & 128 else sample - 132


def decode_wave(data, max_seconds=3600):
    endian = '>' if data[:4] == b'RIFX' else '<'
    format_data, raw = None, []
    for name, body in _chunks(data):
        if name == b'fmt ':
            format_data = body
        elif name == b'data':
            raw.append(body)
    if format_data is None or len(format_data) < 16:
        raise ValueError('Missing WAVE format')
    kind, count, rate, _, block, bits = struct.unpack_from(endian + 'HHIIHH', format_data)
    mask = None
    if kind == 65534:
        if len(format_data) < 40 or struct.unpack_from('<H', format_data, 16)[0] < 22:
            raise ValueError('Truncated WAVE extensible format')
        valid, mask = struct.unpack_from('<HI', format_data, 18)
        guid = format_data[24:40]
        if guid[4:] != b'\0\0\x10\0\x80\0\0\xaa\0\x38\x9b\x71':
            raise ValueError('Unsupported WAVE subformat GUID')
        kind = int.from_bytes(guid[:4], 'little')
        if valid > bits:
            raise ValueError('Invalid WAVE valid bit depth')
    if not 1 <= count <= 64 or not 1 <= rate <= 2147483647:
        raise ValueError('Invalid WAVE audio format')
    if kind in (2, 17):
        from .local_whisper_audio_adpcm import decode_adpcm
        budget = None if max_seconds is None else rate * max_seconds
        output = decode_adpcm(b''.join(raw), kind, count, bits, block, max_frames=budget)
        return rate, output, 's16', mask
    if kind in (6, 7) and bits == 8:
        sample_format = 's16'
    elif kind == 1 and bits in (8, 16, 24, 32, 64):
        sample_format = 'u8' if bits == 8 else 's16' if bits == 16 else 's64' if bits == 64 else 's32'
    elif kind == 3 and bits in (32, 64):
        sample_format = 'flt' if bits == 32 else 'dbl'
    else:
        raise ValueError('Unsupported WAVE codec or bit depth')
    width = bits // 8
    if block != width * count:
        raise ValueError('Invalid WAVE block alignment')
    payload = b''.join(raw)
    frames = len(payload) // block
    if max_seconds is not None and frames > rate * max_seconds:
        raise ValueError('Whisper waveform exceeds the frontend one-hour bound')
    payload = payload[:frames * block]
    if kind in (6, 7):
        values = [_law(v, kind == 6) for v in payload]
    elif sample_format == 'u8':
        values = list(payload)
    elif sample_format in ('flt', 'dbl'):
        code = 'f' if bits == 32 else 'd'
        # Pinned wavdec uses RIFX byte order for header integers only. Its
        # WAVE codec table selects the little-endian PCM/float decoder.
        values = [v[0] for v in struct.iter_unpack('<' + code, payload)]
    else:
        byteorder = 'little'
        shift = 8 if bits == 24 else 0
        values = [int.from_bytes(payload[i:i + width], byteorder, signed=True) << shift for i in range(0, len(payload), width)]
    return rate, [values[ch::count] for ch in range(count)], sample_format, mask
