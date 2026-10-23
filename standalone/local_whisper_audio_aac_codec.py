"""AAC-LC entropy and spectral decoder from pinned FFmpeg 8.1.2 (LGPL-2.1+)."""
import copy
import math
from types import SimpleNamespace
from .local_whisper_audio_aac_bits import Bits, AacError, UnsupportedAac
from .local_whisper_audio_aac_dsp import f32, windowing
from . import local_whisper_audio_aac_tables as table
from .local_whisper_audio_aac_config import defaults, channel_order, parse_pce


class _Vlc:
    def __init__(self, codes, widths):
        self.fast = [None] * 256
        self.long = [{} for _ in range(max(widths) + 1)]
        for symbol, (code, width) in enumerate(zip(codes, widths)):
            if width <= 8:
                start, count = code << (8 - width), 1 << (8 - width)
                self.fast[start:start + count] = [(width, symbol)] * count
            else:
                self.long[width][code] = symbol

    def decode(self, bits):
        found = self.fast[bits.peek(8)]
        if found is not None:
            width, symbol = found
            bits.skip(width)
            return symbol
        for width in range(9, len(self.long)):
            symbol = self.long[width].get(bits.peek(width))
            if symbol is not None:
                bits.skip(width)
                return symbol
        raise AacError('Invalid AAC Huffman code')


_SCALE = _Vlc(table.SCALEFACTOR_CODE, table.SCALEFACTOR_BITS)
_SPECTRAL = tuple(_Vlc(getattr(table, 'CODES' + str(i)), getattr(table, 'BITS' + str(i))) for i in range(1, 12))
_TNS = (table.TNS_TMP2_MAP_0_3, table.TNS_TMP2_MAP_0_4, table.TNS_TMP2_MAP_1_3, table.TNS_TMP2_MAP_1_4)


class _Channel:
    def __init__(self, frame_length):
        self.ics = SimpleNamespace(sequence=0, shape=0)
        self.saved = [0.] * (frame_length // 2)
        self.coefficients = [0.] * 1024
        self.bands = self.scales = self.tns = self.output = None


def _ics_info(bits, previous, config):
    if bits.get(1):
        raise AacError('Reserved AAC ICS bit is set')
    sequence, shape = bits.get(2), bits.get(1)
    groups = [1]
    if sequence == 2:
        maximum = bits.get(4)
        for _ in range(7):
            if bits.get(1):
                groups[-1] += 1
            else:
                groups.append(1)
        offsets = (table.SWB_120 if config.frame_length == 960 else table.SWB_128)[config.sampling_index]
        bands = table.NUM_SWB_128[config.sampling_index]
        tns_max = table.TNS_MAX_BANDS_128[config.sampling_index]
    else:
        maximum = bits.get(6)
        offsets = (table.SWB_960 if config.frame_length == 960 else table.SWB_1024)[config.sampling_index]
        bands = (table.NUM_SWB_960 if config.frame_length == 960 else table.NUM_SWB_1024)[config.sampling_index]
        tns_max = table.TNS_MAX_BANDS_1024[config.sampling_index]
        if bits.get(1):
            raise AacError('Prediction is not allowed in AAC-LC')
    if maximum > bands:
        raise AacError('AAC scalefactor band count exceeds sampling-rate limit')
    return SimpleNamespace(sequence=sequence, shape=shape, previous_sequence=previous.sequence,
                           previous_shape=previous.shape, groups=groups, maximum=maximum,
                           windows=8 if sequence == 2 else 1, offsets=offsets, num_swb=bands,
                           tns_max=tns_max)


def _band_types(bits, ics):
    width = 3 if ics.sequence == 2 else 5
    result = []
    for _ in ics.groups:
        index = 0
        while index < ics.maximum:
            band = bits.get(4)
            if band == 12:
                raise AacError('Reserved AAC spectral codebook')
            count = 0
            while True:
                piece = bits.get(width)
                count += piece
                if index + count > ics.maximum:
                    raise AacError('AAC section exceeds scalefactor bands')
                if piece != (1 << width) - 1:
                    break
            if not count:
                raise AacError('Empty AAC spectral section')
            result.extend([band] * count)
            index += count
    return result


def _scales(bits, bands, gain):
    offsets = [gain, gain - 90, 0]
    noise_first = True
    result = []
    for band in bands:
        if band == 0:
            result.append(0.)
        elif band in (14, 15):
            offsets[2] += _SCALE.decode(bits) - 60
            clipped = max(-155, min(100, offsets[2]))
            result.append(table.POW2SF[200 - clipped])
        elif band == 13:
            offsets[1] += bits.get(9) - 256 if noise_first else _SCALE.decode(bits) - 60
            noise_first = False
            clipped = max(-100, min(155, offsets[1]))
            result.append(-table.POW2SF[clipped + 200])
        else:
            offsets[0] += _SCALE.decode(bits) - 60
            if not 0 <= offsets[0] <= 255:
                raise AacError('AAC scalefactor outside 0..255')
            result.append(-table.POW2SF[offsets[0] + 100])
    return result


def _pulses(bits, ics):
    count, band = bits.get(2) + 1, bits.get(6)
    if band >= ics.num_swb:
        raise AacError('AAC pulse band outside spectrum')
    result, position = [], ics.offsets[band]
    for _ in range(count):
        position += bits.get(5)
        if position >= ics.offsets[ics.num_swb]:
            raise AacError('AAC pulse position outside spectrum')
        result.append((position, bits.get(4)))
    return result


def _tns(bits, ics):
    short = ics.sequence == 2
    filters = []
    for _ in range(ics.windows):
        count = bits.get(1 if short else 2)
        resolution = bits.get(1) if count else 0
        window = []
        for _ in range(count):
            length = bits.get(4 if short else 6)
            order = bits.get(3 if short else 5)
            if order > (7 if short else 12):
                raise AacError('AAC-LC TNS filter order exceeds profile limit')
            direction, coefficients = 0, []
            if order:
                direction, compressed = bits.get(1), bits.get(1)
                width = resolution + 3 - compressed
                mapping = _TNS[2 * compressed + resolution]
                coefficients = [mapping[bits.get(width)] for _ in range(order)]
            window.append((length, direction, coefficients))
        filters.append(window)
    return filters


def _digits(symbol, base, count, signed):
    values = [0] * count
    for index in range(count - 1, -1, -1):
        values[index] = symbol % base - signed
        symbol //= base
    return values


def _spectral_tuple(bits, band):
    symbol = _SPECTRAL[band - 1].decode(bits)
    if band <= 4:
        values = _digits(symbol, 3, 4, 1 if band <= 2 else 0)
    elif band <= 6:
        values = _digits(symbol, 9, 2, 4)
    else:
        values = _digits(symbol, 8 if band <= 8 else 13 if band <= 10 else 17, 2, 0)
    if band in (3, 4, 7, 8, 9, 10, 11):
        signs = [-1 if value and bits.get(1) else 1 for value in values]
        if band == 11:
            for index, value in enumerate(values):
                if value == 16:
                    width = 4
                    while bits.get(1):
                        width += 1
                        if width > 12:
                            raise AacError('AAC spectral escape overflows 13-bit magnitude')
                    values[index] = (1 << width) + bits.get(width)
        values = [value * sign for value, sign in zip(values, signs)]
    return values


def _spectrum(decoder, bits, channel, pulses):
    coefficients = [0.] * 1024
    ics, group_start = channel.ics, 0
    for group, length in enumerate(ics.groups):
        for band in range(ics.maximum):
            index = group * ics.maximum + band
            kind, scale = channel.bands[index], channel.scales[index]
            start, end = ics.offsets[band:band + 2]
            if kind == 0 or kind >= 14:
                continue
            for window in range(length):
                first, last = group_start + window * 128 + start, group_start + window * 128 + end
                if kind == 13:
                    noise = []
                    for _ in range(last - first):
                        decoder.random = (decoder.random * 1664525 + 1013904223) & 0xffffffff
                        value = decoder.random if decoder.random < 0x80000000 else decoder.random - 0x100000000
                        noise.append(f32(value))
                    energy = f32(sum(value * value for value in noise))
                    factor = f32(scale / f32(math.sqrt(energy)))
                    coefficients[first:last] = [f32(value * factor) for value in noise]
                else:
                    position = first
                    while position < last:
                        values = _spectral_tuple(bits, kind)
                        if position + len(values) > last:
                            raise AacError('AAC codebook dimension does not fit scalefactor band')
                        for value in values:
                            coefficients[position] = f32(table.POW43[abs(value)] * (-scale if value < 0 else scale))
                            position += 1
        group_start += length * 128
    for position, amplitude in pulses:
        band = next(i for i in range(ics.num_swb) if ics.offsets[i + 1] > position)
        if band >= len(channel.bands) or channel.bands[band] == 13 or not channel.scales[band]:
            continue
        scale = channel.scales[band]
        value = coefficients[position]
        integer = -amplitude
        if value:
            value = f32(value / scale)
            integer = f32(f32(value / f32(math.sqrt(f32(math.sqrt(abs(value)))))) + (amplitude if value > 0 else -amplitude))
        coefficients[position] = f32(f32(f32(math.cbrt(abs(integer))) * integer) * scale)
    channel.coefficients = coefficients


def _ics(decoder, bits, channel, common=False):
    gain = bits.get(8)
    if not common:
        channel.ics = _ics_info(bits, channel.ics, decoder.config)
    channel.bands = _band_types(bits, channel.ics)
    channel.scales = _scales(bits, channel.bands, gain)
    pulses = []
    if bits.get(1):
        if channel.ics.sequence == 2:
            raise AacError('AAC pulses are invalid in eight-short sequence')
        pulses = _pulses(bits, channel.ics)
    channel.tns = _tns(bits, channel.ics) if bits.get(1) else None
    if bits.get(1):
        raise UnsupportedAac('AAC SSR gain control is not an AAC-LC tool')
    _spectrum(decoder, bits, channel, pulses)


def _stereo(left, right, mask, present):
    ics, position = left.ics, 0
    for group, length in enumerate(ics.groups):
        for band in range(ics.maximum):
            index = group * ics.maximum + band
            first, last = ics.offsets[band:band + 2]
            if mask[index] and left.bands[index] < 13 and right.bands[index] < 13:
                for window in range(length):
                    for k in range(position + window * 128 + first, position + window * 128 + last):
                        a, b = left.coefficients[k], right.coefficients[k]
                        left.coefficients[k], right.coefficients[k] = f32(a + b), f32(a - b)
        position += length * 128
    _intensity_only(left, right, mask, present)


def _intensity_only(left, right, mask, present):
    ics, position = right.ics, 0
    for group, length in enumerate(ics.groups):
        for band in range(ics.maximum):
            index = group * ics.maximum + band
            if right.bands[index] not in (14, 15):
                continue
            factor = (-1 if right.bands[index] == 14 else 1) * right.scales[index]
            if present and mask[index]:
                factor = -factor
            first, last = ics.offsets[band:band + 2]
            for window in range(length):
                for k in range(position + window * 128 + first, position + window * 128 + last):
                    right.coefficients[k] = f32(left.coefficients[k] * factor)
        position += length * 128


def _cpe(decoder, bits, channels):
    common = bits.get(1)
    left, right = channels
    mask, present = [], 0
    if common:
        previous_shape = right.ics.shape
        left.ics = _ics_info(bits, left.ics, decoder.config)
        right.ics = copy.copy(left.ics)
        right.ics.previous_shape = previous_shape
        present = bits.get(2)
        if present == 3:
            raise AacError('Reserved AAC mid-side mask')
        count = len(left.ics.groups) * left.ics.maximum
        mask = [bits.get(1) for _ in range(count)] if present == 1 else [present == 2] * count
    _ics(decoder, bits, left, common)
    _ics(decoder, bits, right, common)
    if common:
        _stereo(left, right, mask, present)
    else:
        _intensity_only(left, right, (), 0)


def _apply_tns(channel):
    if channel.tns is None:
        return
    ics = channel.ics
    maximum = min(ics.tns_max, ics.maximum)
    if not maximum:
        return
    values = channel.coefficients
    for window, filters in enumerate(channel.tns):
        bottom = ics.num_swb
        for length, backwards, reflection in filters:
            top, bottom = bottom, max(0, bottom - length)
            if not reflection:
                continue
            lpc = []
            for i, value in enumerate(reflection):
                r = -value
                previous = lpc[:]
                lpc.append(r)
                for j in range((i + 1) // 2):
                    a, b = previous[j], previous[i - 1 - j]
                    lpc[j], lpc[i - 1 - j] = f32(a + f32(r * b)), f32(b + f32(r * a))
            start, end = ics.offsets[min(bottom, maximum)], ics.offsets[min(top, maximum)]
            count = end - start
            if count <= 0:
                continue
            increment = -1 if backwards else 1
            position = (end - 1 if backwards else start) + window * 128
            for m in range(count):
                for i in range(1, min(m, len(lpc)) + 1):
                    values[position] = f32(values[position] - f32(values[position - i * increment] * lpc[i - 1]))
                position += increment


def _data_stream(bits):
    align, count = bits.get(1), bits.get(8)
    if count == 255:
        count += bits.get(8)
    if align:
        bits.align()
    bits.skip(count * 8)


def _dynamic_range(bits):
    size, bands = 1, 1
    if bits.get(1):
        bits.skip(8)
        size += 1
    if bits.get(1):
        while True:
            bits.skip(7)
            size += 1
            if not bits.get(1):
                break
    if bits.get(1):
        bands += bits.get(4)
        bits.skip(4 + bands * 8)
        size += 1 + bands
    if bits.get(1):
        bits.skip(8)
        size += 1
    bits.skip(bands * 8)
    return size + bands


def _fill(decoder, bits, count):
    if count == 15:
        count += bits.get(8) - 1
    end = bits.pos + count * 8
    if end > bits.limit:
        raise AacError('Truncated AAC fill element')
    while count:
        extension = bits.get(4)
        if extension in (13, 14):
            decoder._read_sbr(bits, count, extension == 14)
            count = 0
            continue
        consumed = _dynamic_range(bits) if extension == 11 else count
        if not 0 < consumed <= count:
            raise AacError('Invalid AAC extension payload length')
        if extension == 0 and count * 8 - 4 >= 13 + 7 * 8:
            bits.skip(13)
            text = bytes(bits.get(8) for _ in range((count * 8 - 17) // 8))
            import re
            if re.match(rb'libfaac\s+[+-]?\d+\.[+-]?\d+', text):
                decoder.skip_samples = 1024
            bits.skip(end - bits.pos)
        elif extension != 11:
            bits.skip(count * 8 - 4)
        count -= consumed
    if bits.pos != end:
        raise AacError('AAC extension payload exceeds fill element')


def _cce(decoder, bits, channel):
    point = 2 * bits.get(1)
    targets, count = [], 0
    for _ in range(bits.get(3) + 1):
        kind, identity = bits.get(1), bits.get(4)
        selection = bits.get(2) if kind == 1 else 2
        targets.append((kind, identity, selection))
        count += 1 + (selection == 3)
    point += bool(bits.get(1) or (point >> 1))
    sign, scale = bits.get(1), f32(2 ** (1 / (8 >> bits.get(2))))
    _ics(decoder, bits, channel)
    gains = []
    for index in range(count):
        common, gain, cache = 1, 0, 1.
        if index:
            common = 1 if point == 3 else bits.get(1)
            gain = _SCALE.decode(bits) - 60 if common else 0
            cache = f32(scale ** -gain)
        if point == 3:
            gains.append([cache])
            continue
        values = [0.] * len(channel.bands)
        for band, kind in enumerate(channel.bands):
            if kind:
                if not common:
                    delta = _SCALE.decode(bits) - 60
                    if delta:
                        gain += delta
                        exponent, polarity = gain, 1
                        if sign:
                            polarity = 1 - 2 * (gain & 1)
                            exponent >>= 1
                        cache = f32(scale ** -exponent) * polarity
                values[band] = cache
        gains.append(values)
    channel.coupling = point, targets, gains


def _couple(target, source, gains, point):
    if point == 3:
        for i, value in enumerate(source.output[:len(target.output)]):
            target.output[i] = f32(target.output[i] + f32(gains[0] * value))
        return
    base, band = 0, 0
    ics = source.ics
    for length in ics.groups:
        for sfb in range(ics.maximum):
            if source.bands[band]:
                gain = gains[band]
                for group in range(length):
                    start = base + group * 128
                    for k in range(ics.offsets[sfb], ics.offsets[sfb + 1]):
                        i = start + k
                        target.coefficients[i] = f32(target.coefficients[i] + f32(gain * source.coefficients[i]))
            band += 1
        base += length * 128


def _channel_coupling(decoder, key, channels, stage, present):
    for identity in range(16):
        cce = (2, identity)
        if cce not in present:
            continue
        source = decoder.elements[cce][0]
        point, targets, gains = source.coupling
        if point != stage:
            continue
        index = 0
        for kind, tag, selection in targets:
            if (kind, tag) == key:
                if selection != 1:
                    _couple(channels[0], source, gains[index], point)
                    index += selection != 0
                if selection != 2:
                    if len(channels) < 2:
                        raise AacError('AAC coupling targets a missing channel')
                    _couple(channels[1], source, gains[index], point)
                    index += 1
            else:
                index += 1 + (selection == 3)


class AacDecoder:
    """Stateful raw AAC LC/SBR/PS decoder; demuxers own delay and padding."""
    def __init__(self, config, *, adts=False):
        self.adts_input = adts
        if config.object_type != 2:
            raise UnsupportedAac('AAC object type ' + str(config.object_type) + ' is not AAC-LC')
        if config.frame_length not in (960, 1024):
            raise AacError('Invalid AAC-LC frame length')
        self.config = config
        self.sbr_contexts, self.sbr_engines = {}, {}
        self.frame_count, self.previous_element = 0, None
        self.elements = {}
        self._configure(config.layout_map if getattr(config, "layout_map", ()) else defaults(config.channel_config) if config.channel_config else ())
        self.random = 0x1f2e3d4c
        self.skip_samples = 0

    def _read_sbr(self, bits, count, crc):
        key = self.previous_element
        config = self.config
        if key is None:
            raise AacError('SBR extension precedes an audio channel element')
        if config.sbr == -1 and self.frame_count and self.adts_input:
            raise AacError('AAC ADTS output format changes within one input')
        if config.frame_length == 960 or config.sbr == 0 or (config.sbr == -1 and self.frame_count):
            bits.skip(count * 8 - 4)
            return
        if config.ps == -1 and not self.frame_count:
            channels = sum(2 if k[0] == 1 else 1 for k in self.order)
            if channels == 1:
                config.ps = 1
        config.sbr = 1
        if not config.extension_rate:
            config.extension_rate = config.rate * 2
        self._sbr_context(key).read(bits, count, crc=crc, kind=key[0])

    def _sbr_context(self, key):
        from .local_whisper_audio_aac_sbr_bits import Sbr
        from .local_whisper_audio_aac_ps import Ps
        if key not in self.sbr_contexts:
            self.sbr_contexts[key] = Sbr(self.config.rate, ps=Ps() if self.config.ps == 1 else None)
        return self.sbr_contexts[key]

    def _apply_sbr(self, key, channels):
        from .local_whisper_audio_aac_sbr_dsp import Engine
        context = self._sbr_context(key)
        if key not in self.sbr_engines:
            self.sbr_engines[key] = Engine(context)
        pcm = self.sbr_engines[key].run([channel.output for channel in channels], kind=key[0],
            downsample=self.config.extension_rate < context.rate, ps_enabled=self.config.ps == 1)
        for channel, samples in zip(channels, pcm):
            channel.output = samples
        return pcm

    def _configure(self, mapping, channel_config=None):
        if channel_config is not None:
            self.config.channel_config = channel_config
        self.mapping = tuple(mapping)
        counts, self.key_map = [0] * 4, {}
        for kind, identity, _ in self.mapping:
            self.key_map[kind, identity] = kind, counts[kind]
            counts[kind] += 1
        order, self.mask = channel_order(self.mapping)
        self.order = tuple(self.key_map[key] for key in order)

    def _element_key(self, kind, identity, mapped):
        config = self.config.channel_config
        if not config:
            if (kind, identity) not in self.key_map:
                raise AacError('AAC element is absent from program configuration')
            return self.key_map[kind, identity]
        if not mapped and kind == 1 and config == 1:
            self._configure(defaults(2), 2)
            config = 2
        elif not mapped and kind == 0 and config == 2:
            self._configure(((0, 0, 1), (0, 1, 1)))
        if (config == 13 and mapped > 3) or (config == 14 and mapped > 2):
            if (kind, identity) in [(t, i) for t, i, _ in self.mapping]:
                return self.key_map[kind, identity]
        if mapped >= len(self.mapping):
            raise AacError('Too many AAC channel elements for configuration')
        expected, tag, _ = self.mapping[mapped]
        remapped_last = mapped == len(self.mapping) - 1 and config in (4, 6, 7, 11, 12, 13, 14) and kind in (0, 3) and expected in (0, 3)
        if kind != expected and not remapped_last:
            raise AacError('AAC element type does not match channel configuration')
        return self.key_map[expected, tag]

    def decode(self, data):
        bits = Bits(data)
        self.previous_element = None
        present, outputs = [], {}
        while True:
            kind = bits.get(3)
            if kind == 7:
                break
            identity = bits.get(4)
            if kind == 6:
                _fill(self, bits, identity)
                continue
            if kind == 4:
                _data_stream(bits)
                continue
            if kind == 5:
                self._configure(parse_pce(bits, self.config.sampling_index), 0)
                continue
            key = self._element_key(kind, identity, sum(key[0] != 2 for key in present))
            if key in present:
                raise AacError('Duplicate AAC channel element')
            channels = self.elements.get(key)
            if channels is None:
                channels = [_Channel(self.config.frame_length) for _ in range(2 if kind == 1 else 1)]
                self.elements[key] = channels
            if kind == 2:
                _cce(self, bits, channels[0])
            elif kind == 1:
                _cpe(self, bits, channels)
            else:
                _ics(self, bits, channels[0])
            self.previous_element = key
            present.append(key)
        bits.align()
        self.consumed = bits.pos // 8
        if not present:
            raise AacError('AAC frame contains no audio channel elements')
        if set(key for key in present if key[0] != 2) != set(self.order):
            raise AacError('AAC frame does not contain all configured audio channels')
        try:
            for key in sorted(present, key=lambda key: (-key[0], key[1])):
                channels = self.elements[key]
                if key[0] <= 1:
                    _channel_coupling(self, key, channels, 0, present)
                for channel in channels:
                    _apply_tns(channel)
                if key[0] <= 1:
                    _channel_coupling(self, key, channels, 1, present)
                if key[0] == 2 and channels[0].coupling[0] != 3:
                    continue
                for channel in channels:
                    ics = channel.ics
                    transform = windowing
                    if self.config.sbr == 1:
                        from .local_whisper_audio_aac_sbr_transform import windowing as transform
                    pcm = transform(channel.coefficients, channel.saved, ics.sequence, ics.previous_sequence,
                                    ics.shape, ics.previous_shape, self.config.frame_length)
                    channel.output = pcm
                if self.config.sbr == 1:
                    outputs[key] = self._apply_sbr(key, channels)
                if key[0] <= 2:
                    _channel_coupling(self, key, channels, 3, present)
        except OverflowError as error:
            raise AacError('AAC numeric range overflow') from error
        self.frame_count += 1
        if self.config.ps == 1:
            self.mask = 3
        return [pcm for key in self.order for pcm in outputs.get(key, [channel.output for channel in self.elements[key]])]
