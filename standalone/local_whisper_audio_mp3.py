"""MPEG-1/2/2.5 Layer III decoding using only the Python standard library.

The entropy, IMDCT and polyphase synthesis algorithms follow pinned minimp3
(CC0). This module returns normalized planar floats without PCM16 quantization.
See local-whisper-mp3-provenance.json for source pins and native qualification.
"""

from types import SimpleNamespace

from .local_whisper_audio_mp3_tables import (
    SCF_LONG, SCF_SHORT, SCF_MIXED, SCF_PARTITIONS, SCFC_DECODE, MOD, PREAMP,
    POW43, TABS, TAB32, TAB33, TABINDEX, LINBITS, PAN, EXPFRAC,
)
from .local_whisper_audio_mp3_dsp import transform


class Mp3Error(ValueError):
    """Invalid or unsupported MPEG Layer III data."""


class _Bits:
    __slots__ = ('data', 'pos', 'limit', 'integer')

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.limit = len(data) * 8
        self.integer = int.from_bytes(data, 'big')

    def peek(self, count):
        shift = self.limit - self.pos - count
        if shift < 0:
            return (self.integer << -shift) & ((1 << count) - 1)
        return (self.integer >> shift) & ((1 << count) - 1)

    def get(self, count):
        value = self.peek(count)
        self.pos += count
        return value


def _header(data):
    if len(data) < 4 or data[0] != 255:
        raise Mp3Error('Missing MPEG audio frame header')
    version = (data[1] >> 3) & 3
    layer = (data[1] >> 1) & 3
    bitrate = data[2] >> 4
    frequency = (data[2] >> 2) & 3
    if data[1] & 0xe0 != 0xe0 or version == 1 or layer != 1 or bitrate == 15 or frequency == 3:
        raise Mp3Error('Invalid MPEG Layer III header')
    rate = (44100, 48000, 32000)[frequency] >> (0 if version == 3 else 1 if version == 2 else 2)
    kbps = ((0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320) if version == 3 else
            (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160))[bitrate]
    channels = 1 if data[3] >> 6 == 3 else 2
    samples = 1152 if version == 3 else 576
    size = samples * kbps * 125 // rate + ((data[2] >> 1) & 1) if kbps else 0
    sr_index = frequency + (((data[1] >> 3) & 1) + ((data[1] >> 4) & 1)) * 3
    return SimpleNamespace(rate=rate, channels=channels, size=size, samples=samples,
                           mpeg1=version == 3, crc=not(data[1] & 1),
                           ms=(data[3] & 0xe0) == 0x60, intensity=(data[3] >> 6) == 1 and bool(data[3] & 0x10),
                           ms_flag=(data[3] >> 6) == 1 and bool(data[3] & 0x20), sr_index=sr_index,
                           sfb_index=sr_index - bool(sr_index))


def _side_info(bits, header):
    count = header.channels
    scfsi = 0
    if header.mpeg1:
        count *= 2
        begin = bits.get(9)
        scfsi = bits.get(7 + count)
    else:
        begin = bits.get(8 + count) >> count
    result = []
    for _ in range(count):
        if header.channels == 1:
            scfsi <<= 4
        gr = SimpleNamespace(part_23_length=bits.get(12), big_values=bits.get(9),
                             global_gain=bits.get(8), scalefac_compress=bits.get(4 if header.mpeg1 else 9))
        if gr.big_values > 288:
            raise Mp3Error('Invalid Layer III big_values')
        gr.sfbtab = SCF_LONG[header.sfb_index]
        gr.n_long_sfb, gr.n_short_sfb = 22, 0
        gr.subblock_gain = [0] * 3
        if bits.get(1):
            gr.block_type = bits.get(2)
            if not gr.block_type:
                raise Mp3Error('Invalid switched Layer III block')
            gr.mixed_block_flag = bits.get(1)
            gr.region_count = [7, 255, 0]
            if gr.block_type == 2:
                scfsi &= 0x0f0f
                if not gr.mixed_block_flag:
                    gr.region_count[0] = 8
                    gr.sfbtab = SCF_SHORT[header.sfb_index]
                    gr.n_long_sfb, gr.n_short_sfb = 0, 39
                else:
                    gr.sfbtab = SCF_MIXED[header.sfb_index]
                    if header.rate == 8000:
                        # FFmpeg uses six long scalefactor bands (72 lines)
                        # before short band three at this MPEG-2.5 rate.
                        gr.sfbtab = SCF_LONG[header.sfb_index][:6] + SCF_SHORT[header.sfb_index][9:]
                    gr.n_long_sfb, gr.n_short_sfb = (8 if header.mpeg1 else 6), 30
            tables = bits.get(10) << 5
            gr.subblock_gain = [bits.get(3) for _ in range(3)]
        else:
            gr.block_type = gr.mixed_block_flag = 0
            tables = bits.get(15)
            gr.region_count = [bits.get(4), bits.get(3), 255]
        gr.table_select = [tables >> 10, (tables >> 5) & 31, tables & 31]
        if any(table in (4, 14) for table in gr.table_select):
            raise Mp3Error('Reserved Layer III Huffman table')
        gr.preflag = bits.get(1) if header.mpeg1 else gr.scalefac_compress >= 500
        gr.scalefac_scale, gr.count1_table = bits.get(1), bits.get(1)
        gr.scfsi = (scfsi >> 12) & 15
        scfsi <<= 4
        result.append(gr)
    if bits.pos > bits.limit or sum(gr.part_23_length for gr in result) + bits.pos > bits.limit + begin * 8:
        raise Mp3Error('Layer III main data exceeds frame and bit reservoir')
    return begin, result


def _ldexp_q2(value, exponent):
    while True:
        chunk = min(120, exponent)
        value *= EXPFRAC[chunk & 3] * ((1 << 30) >> (chunk >> 2))
        exponent -= chunk
        if exponent <= 0:
            return value


def _scalefactors(header, positions, bits, gr, channel):
    partitions = SCF_PARTITIONS[bool(gr.n_short_sfb) + (not gr.n_long_sfb)]
    shift, reuse = gr.scalefac_scale + 1, gr.scfsi
    if header.mpeg1:
        part = SCFC_DECODE[gr.scalefac_compress]
        sizes = [part >> 2] * 2 + [part & 3] * 2
    else:
        intensity = bool(header.intensity and channel)
        sfc = gr.scalefac_compress >> intensity
        k = int(intensity) * 12
        while sfc >= 0:
            product = 1
            sizes = [0] * 4
            for i in range(3, -1, -1):
                sizes[i] = sfc // product % MOD[k + i]
                product *= MOD[k + i]
            sfc -= product
            k += 4
        partitions = partitions[k:]
        reuse = -16
    integer = [0] * 40
    offset = 0
    for i in range(4):
        count = partitions[i]
        if not count:
            break
        if reuse & 8:
            integer[offset:offset + count] = positions[offset:offset + count]
        else:
            maximum = (1 << sizes[i]) - 1 if reuse < 0 else -1
            for j in range(offset, offset + count):
                value = bits.get(sizes[i])
                positions[j] = 255 if sizes[i] and value == maximum else value
                integer[j] = value
        offset += count
        reuse *= 2
    if gr.n_short_sfb:
        for i in range(gr.n_short_sfb):
            j = gr.n_long_sfb + i
            integer[j] = (integer[j] + (gr.subblock_gain[i % 3] << (3 - shift))) & 255
    elif gr.preflag:
        for i, value in enumerate(PREAMP, 11):
            integer[i] += value
    exponent = gr.global_gain - 214 - (2 if header.ms else 0)
    gain = _ldexp_q2(1 << 11, 44 - exponent)
    return [_ldexp_q2(gain, integer[i] << shift) for i in range(gr.n_long_sfb + gr.n_short_sfb)]


def _pow43(value):
    if value < 129:
        return POW43[16 + value]
    multiplier = 256
    if value < 1024:
        multiplier = 16
        value <<= 3
    sign = (2 * value) & 64
    fraction = ((value & 63) - sign) / ((value & ~63) + sign)
    return POW43[16 + ((value + sign) >> 6)] * (1 + fraction * (4 / 3 + fraction * (2 / 9))) * multiplier


def _huffman(bits, gr, scale, limit):
    output = [0.] * 576
    remaining, region, band, destination = gr.big_values, 0, 0, 0
    one = 0.
    while remaining > 0:
        table = gr.table_select[region]
        band_count = gr.region_count[region]
        region += 1
        base, linbits = TABINDEX[table], LINBITS[table]
        while True:
            pairs = gr.sfbtab[band] // 2
            one = scale[band]
            band += 1
            for _ in range(min(remaining, pairs)):
                width = 5
                leaf = TABS[base + bits.peek(width)]
                while leaf < 0:
                    bits.pos += width
                    width = leaf & 7
                    leaf = TABS[base + bits.peek(width) - (leaf >> 3)]
                bits.pos += leaf >> 8
                for _ in range(2):
                    value = leaf & 15
                    if value == 15 and linbits:
                        value += bits.get(linbits)
                        dequantized = _pow43(value)
                    else:
                        dequantized = POW43[16 + value]
                    output[destination] = (-dequantized if value and bits.get(1) else dequantized) * one
                    destination += 1
                    leaf >>= 4
            remaining -= pairs
            if remaining <= 0:
                break
            band_count -= 1
            if band_count < 0:
                break
    if bits.pos > limit:
        raise Mp3Error('Layer III Huffman pairs exceed granule')
    pairs_left = 1 - remaining
    count_table = TAB33 if gr.count1_table else TAB32
    while destination + 4 <= 576:
        leaf = count_table[bits.peek(4)]
        if not leaf & 8:
            leaf = count_table[(leaf >> 3) + (bits.peek(4 + (leaf & 3)) & ((1 << (leaf & 3)) - 1))]
        bits.pos += leaf & 7
        if bits.pos > limit:
            break
        for pair in range(2):
            pairs_left -= 1
            if not pairs_left:
                pairs_left = gr.sfbtab[band] // 2
                band += 1
                if not pairs_left:
                    bits.pos = limit
                    return output
                one = scale[band - 1]
            for j in range(2):
                if leaf & (128 >> (pair * 2 + j)):
                    output[destination + pair * 2 + j] = -one if bits.get(1) else one
        destination += 4
    bits.pos = limit
    return output


def _stereo(channels, positions, granules, header):
    left, right = channels
    if not header.intensity:
        if header.ms:
            for i in range(576):
                a, b = left[i], right[i]
                left[i], right[i] = a + b, a - b
        return
    gr = granules[0]
    bands = gr.n_long_sfb + gr.n_short_sfb
    top = [-1] * 3
    offset = 0
    for i in range(bands):
        if any(right[offset:offset + gr.sfbtab[i]]):
            top[i % 3] = i
        offset += gr.sfbtab[i]
    if gr.n_long_sfb:
        top = [max(top)] * 3
    blocks = 3 if gr.n_short_sfb else 1
    for i in range(blocks):
        last = bands - blocks + i
        previous = last - blocks
        positions[last] = (3 if header.mpeg1 else 0) if top[i] >= previous else positions[previous]
    offset = 0
    for i, count in enumerate(gr.sfbtab):
        if not count:
            break
        position = positions[i]
        if i > top[i % 3] and position < (7 if header.mpeg1 else 64):
            factor = 1.4142135381698608 if header.ms_flag else 1
            if header.mpeg1:
                kl, kr = PAN[2 * position:2 * position + 2]
            else:
                kl, kr = 1, _ldexp_q2(1, ((position + 1) >> 1) << (granules[1].scalefac_compress & 1))
                if position & 1:
                    kl, kr = kr, 1
            for j in range(offset, offset + count):
                left[j], right[j] = left[j] * kl * factor, left[j] * kr * factor
        elif header.ms_flag:
            for j in range(offset, offset + count):
                a, b = left[j], right[j]
                left[j], right[j] = a + b, a - b
        offset += count


class _Decoder:
    def __init__(self, channels):
        self.reservoir = b''
        self.overlap = [[0.] * 288 for _ in range(channels)]
        self.qmf = [0.] * 960

    def frame(self, data, header):
        bits = _Bits(data[4:])
        if header.crc:
            bits.get(16)
        begin, granules = _side_info(bits, header)
        main_data = self.reservoir[max(0, len(self.reservoir) - begin):] + bits.data[bits.pos // 8:]
        if len(self.reservoir) < begin:
            raise Mp3Error('Missing Layer III bit reservoir')
        bits = _Bits(main_data)
        positions = [[0] * 39 for _ in range(header.channels)]
        output = [[] for _ in range(header.channels)]
        for index in range(0, len(granules), header.channels):
            group = granules[index:index + header.channels]
            values = []
            for channel, gr in enumerate(group):
                limit = bits.pos + gr.part_23_length
                factors = _scalefactors(header, positions[channel], bits, gr, channel)
                if bits.pos > limit:
                    raise Mp3Error('Layer III scalefactors exceed granule')
                values.append(_huffman(bits, gr, factors, limit))
            if header.channels == 2:
                _stereo(values, positions[1], group, header)
            pcm = transform(values, group, header, self.overlap, self.qmf)
            for dest, source in zip(output, pcm):
                dest.extend(source)
        self.reservoir = main_data[(bits.pos + 7) // 8:][-511:]
        return output


def _tags(data):
    start, end = 0, len(data)
    while data[start:start + 3] == b'ID3':
        if end - start < 10 or any(value & 128 for value in data[start + 6:start + 10]):
            raise Mp3Error('Invalid ID3v2 header')
        if data[start + 3] not in (2, 3, 4) or data[start + 4] == 255:
            raise Mp3Error('Unsupported ID3v2 version')
        size = 0
        for value in data[start + 6:start + 10]:
            size = (size << 7) | value
        footer = bool(data[start + 5] & 16)
        if footer and data[start + 3] != 4:
            raise Mp3Error('Invalid ID3v2 footer flag')
        start += 10 + size + (10 if footer else 0)
        if start > end:
            raise Mp3Error('Truncated ID3v2 tag')
    if end - start >= 128 and data[end - 128:end - 125] == b'TAG':
        end -= 128
        if end - start >= 227 and data[end - 227:end - 223] == b'TAG+':
            end -= 227
    if end - start >= 32 and data[end - 32:end - 24] == b'APETAGEX':
        size = int.from_bytes(data[end - 20:end - 16], 'little')
        if not 32 <= size <= end - start:
            raise Mp3Error('Invalid APEv2 tag size')
        end -= size
        if end - start >= 32 and data[end - 32:end - 24] == b'APETAGEX':
            end -= 32
    return data[start:end]


def _compatible(first, second):
    return len(second) >= 4 and second[0] == 255 and second[2] >> 4 != 15 and (first[1] ^ second[1]) & 0xfe == 0 and (first[2] ^ second[2]) & 12 == 0 and bool(first[2] >> 4) == bool(second[2] >> 4)


def _frame_size(data, offset, header, free_size):
    if header.size:
        return header.size, free_size
    padding = (data[offset + 2] >> 1) & 1
    if free_size:
        return free_size + padding, free_size
    for distance in range(24, min(2304 + padding, len(data) - offset - 3)):
        following = data[offset + distance:offset + distance + 4]
        try:
            _header(following)
        except Mp3Error:
            continue
        if not _compatible(data[offset:offset + 4], following):
            continue
        size = distance - padding
        third = offset + distance + size + ((following[2] >> 1) & 1)
        if third == len(data) or (third + 4 <= len(data) and _compatible(following, data[third:third + 4])):
            return distance, size
    raise Mp3Error('Cannot determine free-format MPEG frame size')


def _reject_other_container(data):
    signatures = (b'RIFF', b'RIFX', b'RF64', b'fLaC', b'OggS', b'FORM',
                  b'caff', b'.snd', b'MThd', b'MAC ', b'FLV\x01', b'\x1aE\xdf\xa3',
                  b'ADIF', b'#!AMR')
    box_size = int.from_bytes(data[:4], 'big')
    mp4 = data[4:8] in (b'ftyp', b'moov', b'mdat', b'wide') and (box_size in (0, 1) or 8 <= box_size <= len(data))
    if data.startswith(signatures) or mp4:
        raise Mp3Error('Input is another audio or media container')
    if len(data) > 1 and data[0] == 255 and data[1] & 0xf6 == 0xf0:
        raise Mp3Error('AAC ADTS input is not MPEG Layer III')


def _find_start(data):
    _reject_other_container(data)
    position = 0
    while True:
        position = data.find(b'\xff', position)
        if position < 0:
            raise Mp3Error('No consecutive MPEG Layer III frames')
        try:
            header = _header(data[position:position + 4])
            size, _ = _frame_size(data, position, header, 0)
        except Mp3Error:
            position += 1
            continue
        following = position + size
        if following == len(data) and position == 0:
            return position, header
        if following + 4 <= len(data) and _compatible(data[position:position + 4], data[following:following + 4]):
            other = _header(data[following:following + 4])
            if header.channels == other.channels:
                return position, header
        position += 1


def _walk_headers(data, start, first):
    offset, free_size = start, 0
    while offset < len(data):
        header = _header(data[offset:offset + 4])
        if header.rate != first.rate or header.channels != first.channels:
            raise Mp3Error('MPEG sample rate or channel count changes within input')
        size, free_size = _frame_size(data, offset, header, free_size)
        if size < 4 or offset + size > len(data):
            raise Mp3Error('Truncated MPEG audio frame')
        yield offset, size, header
        offset += size


def probe_mp3(data):
    """Identify a complete Layer III frame chain, including tags/leading junk.

    Recognized other containers are rejected before scanning their payload.
    The full chain after the first frame must end at the trimmed file boundary.
    CRC/entropy verification is performed by decode_mp3, after identification.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        return False
    try:
        data = _tags(bytes(data))
        start, first = _find_start(data)
        return sum(1 for _ in _walk_headers(data, start, first)) > 0
    except Mp3Error:
        return False


def _crc(frame, header):
    if not header.crc:
        return
    side = (17 if header.channels == 1 else 32) if header.mpeg1 else (9 if header.channels == 1 else 17)
    if len(frame) < 6 + side:
        raise Mp3Error('Truncated protected MPEG frame')
    value = 65535
    for byte in frame[2:4] + frame[6:6 + side]:
        value ^= byte << 8
        for _ in range(8):
            value = ((value << 1) ^ (0x8005 if value & 32768 else 0)) & 65535
    if value != int.from_bytes(frame[4:6], 'big'):
        raise Mp3Error('MPEG audio CRC mismatch')


def _vbr_info(frame, header, file_size):
    side = (17 if header.channels == 1 else 32) if header.mpeg1 else (9 if header.channels == 1 else 17)
    offset = 4 + side
    # FFmpeg/LAME place the dummy Xing frame at this fixed offset even
    # when protection is set; also accept minimp3's CRC-adjusted layout.
    if header.crc and frame[offset:offset + 4] not in (b'Xing', b'Info'):
        offset += 2
    if frame[offset:offset + 4] not in (b'Xing', b'Info'):
        if frame[36:40] == b'VBRI' and len(frame) >= 54 and frame[40:42] == b'\x00\x01':
            return True, int.from_bytes(frame[50:54], 'big'), 0, 0
        return False, 0, 0, 0
    if len(frame) < offset + 8:
        raise Mp3Error('Truncated Xing header')
    flags = int.from_bytes(frame[offset + 4:offset + 8], 'big')
    offset += 8
    frames = size = 0
    for bit, width in ((1, 4), (2, 4), (4, 100), (8, 4)):
        if flags & bit:
            if offset + width > len(frame):
                raise Mp3Error('Truncated Xing fields')
            if bit == 1:
                frames = int.from_bytes(frame[offset:offset + width], 'big')
            if bit == 2:
                size = int.from_bytes(frame[offset:offset + width], 'big')
            offset += width
    if size and file_size > size and file_size - size > size >> 4:
        frames = 0
    delay = padding = 0
    if frame[offset:offset + 4] in (b'LAME', b'Lavf', b'Lavc'):
        if len(frame) < offset + 24:
            raise Mp3Error('Truncated LAME gapless fields')
        packed = int.from_bytes(frame[offset + 21:offset + 24], 'big')
        delay = (packed >> 12) + 529
        padding = (packed & 4095) - 529
    return bool(frames or size), frames, delay, padding


def decode_mp3(data, *, max_frames=None):
    """Return ``(rate, normalized_planar_float, 'fltp', channel_mask)``.

    ``max_frames`` bounds decoded output per channel at the source sample rate.
    ID3/APE tags and Xing/LAME encoder delay/padding are handled before returning.
    MPEG Layer I/II and changes of sample rate/channel count are errors.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError('MP3 input must be bytes-like')
    if max_frames is not None and (not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 0):
        raise ValueError('max_frames must be a nonnegative integer or None')
    data = _tags(bytes(data))
    if len(data) < 4:
        raise Mp3Error('No MPEG audio frames')
    start, first = _find_start(data)
    decoder = _Decoder(first.channels)
    output = [[] for _ in range(first.channels)]
    seen = start_skip = raw_samples = initial_delay = padding_allowance = 0
    expected = None
    for offset, size, header in _walk_headers(data, start, first):
        frame = data[offset:offset + size]
        _crc(frame, header)
        if not seen:
            tag, count, start_skip, padding = _vbr_info(frame, header, len(data) - 4)
            # Native packet discovery suppresses a Xing frame after junk,
            # but its gapless side data is unavailable after resync.
            if start:
                count = start_skip = padding = 0
            initial_delay, padding_allowance = start_skip, max(0, padding)
            if count:
                expected = count * header.samples - start_skip - max(0, padding)
                if expected < 0:
                    raise Mp3Error('Invalid MPEG gapless duration')
                if max_frames is not None and expected > max_frames:
                    raise Mp3Error('Decoded MP3 exceeds frame budget')
            seen += 1
            if tag:
                continue
        seen += 1
        raw_samples += header.samples
        if max_frames is not None and raw_samples > max_frames + initial_delay + padding_allowance:
            raise Mp3Error('Decoded MP3 exceeds frame budget')
        pcm = decoder.frame(frame, header)
        skipped = min(start_skip, header.samples)
        start_skip -= skipped
        available = header.samples - skipped
        if expected is not None:
            available = min(available, max(0, expected - len(output[0])))
        if max_frames is not None and len(output[0]) + available > max_frames:
            raise Mp3Error('Decoded MP3 exceeds frame budget')
        for dest, channel in zip(output, pcm):
            dest.extend(channel[skipped:skipped + available])
    if not output[0]:
        raise Mp3Error('No decodable MPEG audio samples')
    return first.rate, output, 'fltp', 4 if first.channels == 1 else 3
