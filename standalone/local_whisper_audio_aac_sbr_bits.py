"""Stateful HE-AAC SBR header/grid/envelope syntax, from pinned FFmpeg LGPL."""
import copy
from standalone.local_whisper_audio_aac_bits import AacError
from standalone.local_whisper_audio_aac_codec import _Vlc
from .local_whisper_audio_aac_sbr_frequency import Frequency, Parameters
from .local_whisper_audio_aac_sbr_tables import HUFFMAN


_VLCS = tuple((_Vlc([row[1] for row in rows], [row[2] for row in rows]), tuple(row[0] for row in rows)) for rows in HUFFMAN)


def _huff(bits, codebook):
    vlc, symbols = _VLCS[codebook]
    return symbols[vlc.decode(bits)]


class Channel:
    def __init__(self):
        self.count = self.noise_count = self.frame_class = self.amp_res = 0
        self.freq_res = [0] * 9
        self.env_q = [[0] * 48 for _ in range(9)]
        self.noise_q = [[0] * 5 for _ in range(3)]
        self.time = [0] * 9
        self.noise_time = [0] * 3
        self.old_end = 0
        self.e_a = [0, -1]
        self.invf = [[0] * 5 for _ in range(2)]
        self.harmonic = [0] * 48
        self.harmonic_flag = 0
        self.noise_index = self.sine_index = 0

    def grid(self, bits, amplitude):
        old_count = self.count
        self.freq_res[0] = self.freq_res[old_count]
        self.old_end = self.time[old_count]
        self.amp_res = amplitude
        kind, pointer, trailing = bits.get(2), 0, 16
        if kind == 0:
            count = 1 << bits.get(2)
            if count > 5:
                raise AacError('Too many SBR FIXFIX envelopes')
            self.time[0], self.time[count] = 0, 16
            if count == 1:
                self.amp_res = 0
            step = (trailing + (count >> 1)) // count
            for i in range(1, count):
                self.time[i] = self.time[i - 1] + step
            resolution = bits.get(1)
            self.freq_res[1:count + 1] = [resolution] * count
        elif kind == 1:
            trailing += bits.get(2)
            rel = bits.get(2)
            count = rel + 1
            self.time[0], self.time[count] = 0, trailing
            for i in range(rel):
                self.time[count - 1 - i] = self.time[count - i] - 2 * bits.get(2) - 2
            pointer = bits.get(count.bit_length())
            for i in range(count):
                self.freq_res[count - i] = bits.get(1)
        elif kind == 2:
            self.time[0] = bits.get(2)
            rel = bits.get(2)
            count = rel + 1
            self.time[count] = trailing
            for i in range(rel):
                self.time[i + 1] = self.time[i] + 2 * bits.get(2) + 2
            pointer = bits.get(count.bit_length())
            self.freq_res[1:count + 1] = [bits.get(1) for _ in range(count)]
        else:
            self.time[0] = bits.get(2)
            trailing += bits.get(2)
            lead, trail = bits.get(2), bits.get(2)
            count = lead + trail + 1
            if count > 5:
                raise AacError('Too many SBR VARVAR envelopes')
            self.time[count] = trailing
            for i in range(lead):
                self.time[i + 1] = self.time[i] + 2 * bits.get(2) + 2
            for i in range(trail):
                self.time[count - 1 - i] = self.time[count - i] - 2 * bits.get(2) - 2
            pointer = bits.get(count.bit_length())
            self.freq_res[1:count + 1] = [bits.get(1) for _ in range(count)]
        if pointer > count + 1 or any(self.time[i - 1] >= self.time[i] for i in range(1, count + 1)):
            raise AacError('Invalid SBR envelope time borders or pointer')
        self.count, self.frame_class = count, kind
        self.noise_count = 1 + (count > 1)
        self.noise_time[0], self.noise_time[self.noise_count] = self.time[0], self.time[count]
        if count > 1:
            index = count >> 1 if kind == 0 else count - max(pointer - 1, 1) if kind & 1 else 1 if pointer == 0 else count - 1 if pointer == 1 else pointer - 1
            self.noise_time[1] = self.time[index]
        self.e_a = [-int(self.e_a[1] != old_count), -1]
        if kind & 1 and pointer:
            self.e_a[1] = count + 1 - pointer
        elif kind == 2 and pointer > 1:
            self.e_a[1] = pointer - 1

    def copy_grid(self, source):
        self.freq_res[0] = self.freq_res[self.count]
        self.old_end = self.time[self.count]
        self.e_a[0] = -int(self.e_a[1] != self.count)
        self.freq_res[1:] = source.freq_res[1:]
        self.time[:], self.noise_time[:] = source.time, source.noise_time
        self.count, self.noise_count = source.count, source.noise_count
        self.amp_res, self.frame_class = source.amp_res, source.frame_class
        self.e_a[1] = source.e_a[1]

    def dtdf(self, bits):
        self.delta_env = [bits.get(1) for _ in range(self.count)]
        self.delta_noise = [bits.get(1) for _ in range(self.noise_count)]

    def read_inverse(self, bits, count):
        self.invf[1][:] = self.invf[0]
        self.invf[0][:count] = [bits.get(2) for _ in range(count)]

    def read_envelope(self, bits, freq, balanced):
        delta = 2 if balanced else 1
        if balanced:
            width, time_huff, freq_huff = (5, 6, 7) if self.amp_res else (6, 2, 3)
        else:
            width, time_huff, freq_huff = (6, 4, 5) if self.amp_res else (7, 0, 1)
        band_counts, odd = (len(freq.low) - 1, len(freq.high) - 1), (len(freq.high) - 1) & 1
        for i in range(self.count):
            resolution = self.freq_res[i + 1]
            count = band_counts[resolution]
            if self.delta_env[i]:
                for j in range(count):
                    if resolution == self.freq_res[i]:
                        k = j
                    elif resolution:
                        k = (j + odd) >> 1
                    else:
                        k = 2 * j - odd if j else 0
                    value = (self.env_q[i][k] + delta * _huff(bits, time_huff)) & 255
                    if value > 127:
                        raise AacError('Invalid SBR envelope delta')
                    self.env_q[i + 1][j] = value
            else:
                self.env_q[i + 1][0] = delta * bits.get(width)
                for j in range(1, count):
                    value = (self.env_q[i + 1][j - 1] + delta * _huff(bits, freq_huff)) & 255
                    if value > 127:
                        raise AacError('Invalid SBR envelope frequency value')
                    self.env_q[i + 1][j] = value
        self.env_q[0][:] = self.env_q[self.count]

    def read_noise(self, bits, count, balanced):
        delta, time_huff, freq_huff = (2, 9, 7) if balanced else (1, 8, 5)
        for i in range(self.noise_count):
            if self.delta_noise[i]:
                for j in range(count):
                    value = (self.noise_q[i][j] + delta * _huff(bits, time_huff)) & 255
                    if value > 30:
                        raise AacError('Invalid SBR noise delta')
                    self.noise_q[i + 1][j] = value
            else:
                self.noise_q[i + 1][0] = delta * bits.get(5)
                for j in range(1, count):
                    value = (self.noise_q[i + 1][j - 1] + delta * _huff(bits, freq_huff)) & 255
                    if value > 30:
                        raise AacError('Invalid SBR noise frequency value')
                    self.noise_q[i + 1][j] = value
        self.noise_q[0][:] = self.noise_q[self.noise_count]


class Sbr:
    def __init__(self, core_rate, *, ps=None):
        self.rate, self.ps = core_rate * 2, ps
        self.channels = [Channel(), Channel()]
        self.frequency = self.parameters = None
        self.started = self.ready = self.reset = self.coupling = False
        self.amplitude = 0
        self.limiter, self.limiter_gain, self.interpolate, self.smooth = 2, 2, 1, 1
        self.previous_kx, self.previous_m = 0, 0
        self.kx, self.m = 32, 0
        self.pushed = False

    def turnoff(self):
        self.started = self.ready = False
        self.kx, self.m = 32, 0
        self.parameters = None
        for channel in self.channels:
            channel.e_a[1] = -1

    def header(self, bits):
        self.started, self.ready = True, False
        self.amplitude = bits.get(1)
        start, stop, crossover = bits.get(4), bits.get(4), bits.get(3)
        bits.skip(2)
        extra1, extra2 = bits.get(1), bits.get(1)
        scale, alter, noise = (bits.get(2), bits.get(1), bits.get(2)) if extra1 else (2, 1, 2)
        parameters = Parameters(start, stop, crossover, scale, alter, noise)
        old_limiter = self.limiter
        self.limiter, self.limiter_gain, self.interpolate, self.smooth = (bits.get(2), bits.get(2), bits.get(1), bits.get(1)) if extra2 else (2, 2, 1, 1)
        if parameters != self.parameters:
            self.reset, self.parameters = True, parameters
            self.frequency = Frequency(self.rate, parameters, self.limiter)
            self.kx, self.m = self.frequency.kx, self.frequency.m
            for channel in self.channels:
                channel.noise_index = 0
        elif old_limiter != self.limiter:
            self.frequency.limits = self.frequency._limits(self.limiter)

    def data(self, bits, kind):
        self.ready = True
        a, b = self.channels
        freq = self.frequency
        n_q, n_high = len(freq.noise) - 1, len(freq.high) - 1
        if bits.get(1):
            bits.skip(8 if kind == 1 else 4)
        if kind == 1:
            self.coupling = bool(bits.get(1))
            a.grid(bits, self.amplitude)
            b.copy_grid(a) if self.coupling else b.grid(bits, self.amplitude)
            a.dtdf(bits)
            b.dtdf(bits)
            a.read_inverse(bits, n_q)
            if self.coupling:
                b.invf[1][:], b.invf[0][:] = b.invf[0][:], a.invf[0][:]
                a.read_envelope(bits, freq, False)
                a.read_noise(bits, n_q, False)
                b.read_envelope(bits, freq, True)
                b.read_noise(bits, n_q, True)
            else:
                b.read_inverse(bits, n_q)
                a.read_envelope(bits, freq, False)
                b.read_envelope(bits, freq, False)
                a.read_noise(bits, n_q, False)
                b.read_noise(bits, n_q, False)
            active = (a, b)
        elif kind in (0, 2):
            a.grid(bits, self.amplitude)
            a.dtdf(bits)
            a.read_inverse(bits, n_q)
            a.read_envelope(bits, freq, False)
            a.read_noise(bits, n_q, False)
            active = (a,)
        else:
            raise AacError('SBR is not valid on an AAC LFE element')
        for channel in active:
            channel.harmonic_flag = bits.get(1)
            if channel.harmonic_flag:
                channel.harmonic[:n_high] = [bits.get(1) for _ in range(n_high)]
        if bits.get(1):
            count = bits.get(4)
            if count == 15:
                count += bits.get(8)
            end = bits.pos + count * 8
            while end - bits.pos > 7:
                kind = bits.get(2)
                if kind == 2:
                    if self.ps is None:
                        bits.skip(end - bits.pos)
                    else:
                        self.ps.read(bits, end - bits.pos)
                else:
                    bits.skip(end - bits.pos)
            bits.skip(end - bits.pos)

    def read(self, host, count, *, crc=False, kind=0):
        end = host.pos + count * 8 - 4
        bits = copy.copy(host)
        host.skip(count * 8 - 4)
        self.reset = False
        if crc:
            bits.skip(10)
        self.previous_kx, self.previous_m = self.kx, self.m
        self.pushed = True
        if bits.get(1):
            self.header(bits)
        if self.started:
            self.data(bits, kind)
        if bits.pos > end:
            raise AacError('SBR payload crosses the fill element')
        self.consumed = bits.pos
