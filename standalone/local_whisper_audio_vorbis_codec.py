"""Original Vorbis I codebooks, spectral reconstruction and overlap state."""
import struct

from standalone.local_whisper_audio_aac_dsp import f32
from standalone.local_whisper_audio_aac_sbr_transform import imdct, _multiply_window
from standalone.local_whisper_audio_ogg import _MASKS, _VORBIS_ORDER
from .local_whisper_audio_vorbis_bits import Bits, Codebook, VorbisError
from .local_whisper_audio_vorbis_floor import Floor0, Floor1
from .local_whisper_audio_vorbis_residue import Residue
from .local_whisper_audio_vorbis_tables import WINDOWS


class VorbisParser:
    """Original Ogg ID checks and reverse setup duration-parser boundary."""
    def __init__(self, identification, setup):
        if len(identification) != 30 or identification[:7] != b'\x01vorbis':
            raise VorbisError('Invalid Ogg Vorbis identification packet')
        short, long = identification[28] & 15, identification[28] >> 4
        if (int.from_bytes(identification[7:11], 'little') or identification[29] != 1
                or not 6 <= short <= long <= 13):
            raise VorbisError('Invalid Ogg Vorbis stream header')
        self.sizes = 1 << short, 1 << long
        self.channels, self.rate = identification[11], int.from_bytes(identification[12:16], 'little')
        self.mask = _MASKS.get(self.channels)
        if len(setup) < 7 or setup[:7] != b'\x05vorbis':
            raise VorbisError('Invalid Ogg Vorbis setup header')
        length, raw, position = 8 * len(setup), int.from_bytes(setup[::-1], 'big'), 0

        def read(count):
            nonlocal position
            position += count
            if position > length:
                raise VorbisError('Truncated Ogg Vorbis duration setup')
            return (raw >> (length - position)) & ((1 << count) - 1)

        framing = 0
        while length - position > 97:
            if read(1):
                framing = position
                break
        if not framing:
            raise VorbisError('Invalid Ogg Vorbis setup framing')
        count, last = 0, 0
        while length - position >= 97:
            if read(8) > 63 or read(16) or read(16):
                break
            read(1)
            count += 1
            if count > 64:
                break
            saved = position
            if read(6) + 1 == count:
                last = count
            position = saved
        if not 1 <= last <= 63:
            raise VorbisError('Invalid Ogg Vorbis duration modes')
        position = framing
        self.modes = [None] * last
        for i in range(last - 1, -1, -1):
            read(40)
            self.modes[i] = (read(1), 0)


class Mapping:
    def __init__(self, bits, channels, floors, residues):
        if bits.read(16):
            raise VorbisError('Only Vorbis I mapping0 is supported by the original decoder')
        submaps = bits.read(4) + 1 if bits.read(1) else 1
        self.coupling = []
        if bits.read(1):
            steps = bits.read(8) + 1
            if channels < 2:
                raise VorbisError('Vorbis coupling requires multiple channels')
            for _ in range(steps):
                self.coupling.append((bits.index((channels - 1).bit_length(), channels),
                                      bits.index((channels - 1).bit_length(), channels)))
        if bits.read(2):
            raise VorbisError('Nonzero Vorbis mapping reserved bits')
        self.mux = [bits.index(4, submaps) for _ in range(channels)] if submaps > 1 else [0] * channels
        self.floors, self.residues = [], []
        for _ in range(submaps):
            bits.read(8)
            self.floors.append(floors[bits.index(8, len(floors))])
            self.residues.append(residues[bits.index(8, len(residues))])


class VorbisDecoder:
    def __init__(self, identification, setup):
        if len(identification) < 30 or identification[:7] != b'\x01vorbis':
            raise VorbisError('Invalid Vorbis identification packet')
        version, self.channels, self.rate = struct.unpack_from('<IBI', identification, 7)
        short, long = identification[28] & 15, identification[28] >> 4
        if version or not self.channels or not 0 < self.rate <= 0x7fffffff:
            raise VorbisError('Invalid Vorbis stream parameters')
        if not 6 <= short <= long <= 13 or not identification[29] & 1:
            raise VorbisError('Invalid Vorbis block sizes or header framing')
        self.sizes = 1 << short, 1 << long
        self.mask = _MASKS.get(self.channels)
        self.order = _VORBIS_ORDER.get(self.channels, list(range(self.channels)))
        if setup[:7] != b'\x05vorbis':
            raise VorbisError('Missing Vorbis setup packet')
        bits = Bits(setup[7:])
        self.books = [Codebook(bits) for _ in range(bits.read(8) + 1)]
        for _ in range(bits.read(6) + 1):
            if bits.read(16):
                raise VorbisError('Nonzero Vorbis time transform')
        self.floors = []
        for _ in range(bits.read(6) + 1):
            kind = bits.read(16)
            if kind not in (0, 1):
                raise VorbisError('Invalid Vorbis floor type')
            self.floors.append((Floor1 if kind else Floor0)(bits, self.books, self.sizes))
        self.residues = [Residue(bits, self.books) for _ in range(bits.read(6) + 1)]
        self.mappings = [Mapping(bits, self.channels, self.floors, self.residues)
                         for _ in range(bits.read(6) + 1)]
        self.modes = []
        for _ in range(bits.read(6) + 1):
            block, window, transform = bits.read(1), bits.read(16), bits.read(16)
            self.modes.append((block, bits.index(8, len(self.mappings))))
        if not bits.read(1):
            raise VorbisError('Missing Vorbis setup framing bit')
        self.previous = -1
        self.first = True
        self.saved = [[0.] * (self.sizes[1] // 4) for _ in range(self.channels)]

    def decode(self, packet):
        bits = Bits(packet, padded=True)
        if bits.read(1):
            raise VorbisError('Non-audio Vorbis packet')
        mode = bits.index((len(self.modes) - 1).bit_length(), len(self.modes))
        flag, index = self.modes[mode]
        mapping = self.mappings[index]
        previous = self.previous
        size, length = self.sizes[flag], self.sizes[flag] // 2
        if flag:
            code = bits.read(2)
            if previous < 0:
                previous = code >> 1
        elif previous < 0:
            previous = 0
        transformed = self._spectrum(bits, mapping, length)
        result = self._overlap(transformed, flag, size, previous)
        self.previous = flag
        if self.first:
            self.first = False
            return [[] for _ in range(self.channels)]
        return [result[ch] for ch in self.order]

    def _spectrum(self, bits, mapping, length):
        floors, skipped = [], []
        for ch in range(self.channels):
            values, skip = mapping.floors[mapping.mux[ch]].decode(bits, length)
            floors.append(values)
            skipped.append(skip)
        for magnitude, angle in reversed(mapping.coupling):
            if not skipped[magnitude] or not skipped[angle]:
                skipped[magnitude] = skipped[angle] = False
        residue, mapped = self._residues(bits, mapping, length, skipped)
        self._inverse_coupling(mapping.coupling, residue, mapped, length)
        transformed = []
        for ch in range(self.channels):
            offset = mapped[ch] * length
            values = [f32(floors[ch][i] * residue[offset + i]) for i in range(length)]
            transformed.append(imdct(values, scale=-1.))
        return transformed

    def _residues(self, bits, mapping, length, skipped):
        residue = [0.] * (self.channels * length)
        mapped, used = [0] * self.channels, 0
        for index, decoder in enumerate(mapping.residues):
            channels = [ch for ch in range(self.channels) if mapping.mux[ch] == index]
            for i, ch in enumerate(channels):
                mapped[ch] = used + i
            if channels:
                decoder.decode(bits, residue, used * length, length, len(channels),
                               [skipped[ch] for ch in channels], self.channels - used)
            used += len(channels)
        if used != self.channels:
            raise VorbisError('Invalid Vorbis channel submaps')
        return residue, mapped

    @staticmethod
    def _inverse_coupling(coupling, residue, mapped, length):
        for magnitude, angle in reversed(coupling):
            mag, ang = mapped[magnitude] * length, mapped[angle] * length
            for i in range(length):
                m, a = residue[mag + i], residue[ang + i]
                if m > 0:
                    residue[ang + i] = f32(m - a) if a > 0 else m
                    if a <= 0:
                        residue[mag + i] = f32(m + a)
                else:
                    residue[ang + i] = f32(m + a) if a > 0 else m
                    if a <= 0:
                        residue[mag + i] = f32(m - a)

    def _overlap(self, transformed, flag, size, previous):
        retlen = (size + self.sizes[previous]) // 4
        result = []
        for ch, values in enumerate(transformed):
            saved = self.saved[ch]
            window = WINDOWS[self.sizes[flag & previous]]
            if flag == previous:
                output = _multiply_window(saved, values, window, size // 4)
            elif flag > previous:
                output = _multiply_window(saved, values, window, self.sizes[0] // 4)
                output += values[self.sizes[0] // 4:self.sizes[1] // 4]
            else:
                delta = (self.sizes[1] - self.sizes[0]) // 4
                output = saved[:delta] + _multiply_window(saved[delta:], values, window, self.sizes[0] // 4)
            saved[:size // 4] = values[size // 4:size // 2]
            result.append(output[:retlen])
        return result
