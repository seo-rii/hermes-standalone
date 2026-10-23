"""MPEG-4 Parametric Stereo syntax, from pinned FFmpeg 8.1.2 (LGPL)."""
import copy
from standalone.local_whisper_audio_aac_bits import AacError
from standalone.local_whisper_audio_aac_codec import _Vlc
from .local_whisper_audio_aac_ps_tables import HUFFMAN

_VLCS = tuple((_Vlc([r[1] for r in rows], [r[2] for r in rows]), tuple(r[0] for r in rows)) for rows in HUFFMAN)


class PsBits:
    def __init__(self):
        for name in ('start', 'enable_iid', 'iid_quant', 'nr_iid_par', 'nr_ipdopd_par',
                     'enable_icc', 'icc_mode', 'nr_icc_par', 'enable_ext', 'frame_class',
                     'num_env_old', 'num_env', 'enable_ipdopd', 'is34bands', 'is34bands_old'):
            setattr(self, name, 0)
        self.border_position = [0] * 6
        for name in ('iid_par', 'icc_par', 'ipd_par', 'opd_par'):
            setattr(self, name, [[0] * 34 for _ in range(5)])

    @property
    def started(self):
        return bool(self.start)

    def _parameters(self, bits, name, book, envelope, dt):
        par = getattr(self, name + '_par')
        count = getattr(self, 'nr_' + ('ipdopd' if name in ('ipd', 'opd') else name) + '_par')
        vlc, symbols = _VLCS[book]
        previous = max(envelope - 1 if envelope else self.num_env_old - 1, 0)
        value = 0
        for band in range(count):
            delta = symbols[vlc.decode(bits)]
            value = par[previous][band] + delta if dt else value + delta
            if name in ('ipd', 'opd'):
                value &= 7
            par[envelope][band] = (value + 128) % 256 - 128
            actual = par[envelope][band]
            if name == 'iid' and abs(actual) > 7 + 8 * self.iid_quant:
                raise AacError('Invalid PS intensity difference')
            if name == 'icc' and not 0 <= actual <= 7:
                raise AacError('Invalid PS coherence')

    def _extension(self, bits, kind):
        if kind:
            return
        self.enable_ipdopd = bits.get(1)
        if self.enable_ipdopd:
            for e in range(self.num_env):
                dt = bits.get(1)
                self._parameters(bits, 'ipd', 6 + dt, e, dt)
                dt = bits.get(1)
                self._parameters(bits, 'opd', 8 + dt, e, dt)
        bits.skip(1)

    def read(self, host, bits_left):
        bits = copy.copy(host)
        start = bits.pos
        header = bits.get(1)
        if header:
            self.enable_iid = bits.get(1)
            if self.enable_iid:
                mode = bits.get(3)
                if mode > 5:
                    raise AacError('Reserved PS intensity mode')
                self.nr_iid_par = (10, 20, 34)[mode % 3]
                self.iid_quant = int(mode > 2)
                self.nr_ipdopd_par = (5, 11, 17)[mode % 3]
            self.enable_icc = bits.get(1)
            if self.enable_icc:
                self.icc_mode = bits.get(3)
                if self.icc_mode > 5:
                    raise AacError('Reserved PS coherence mode')
                self.nr_icc_par = (10, 20, 34)[self.icc_mode % 3]
            self.enable_ext = bits.get(1)
        self.frame_class = bits.get(1)
        self.num_env_old = self.num_env
        self.num_env = ((0, 1, 2, 4), (1, 2, 3, 4))[self.frame_class][bits.get(2)]
        self.border_position[0] = -1
        for e in range(1, self.num_env + 1):
            self.border_position[e] = bits.get(5) if self.frame_class else (e * 32 >> (self.num_env.bit_length() - 1)) - 1
            if self.border_position[e] < self.border_position[e - 1]:
                raise AacError('Nonmonotone PS envelope borders')
        for name in ('iid', 'icc'):
            if getattr(self, 'enable_' + name):
                for e in range(self.num_env):
                    dt = bits.get(1)
                    book = (2, 0, 3, 1)[2 * dt + self.iid_quant] if name == 'iid' else 4 + dt
                    self._parameters(bits, name, book, e, dt)
            else:
                getattr(self, name + '_par')[:] = [[0] * 34 for _ in range(5)]
        if self.enable_ext:
            count = bits.get(4)
            if count == 15:
                count += bits.get(8)
            end = bits.pos + count * 8
            while end - bits.pos > 7:
                self._extension(bits, bits.get(2))
            if bits.pos > end:
                raise AacError('PS extension overflow')
            bits.skip(end - bits.pos)
        if not self.num_env or self.border_position[self.num_env] < 31:
            source = self.num_env - 1 if self.num_env else self.num_env_old - 1
            if source >= 0 and source != self.num_env:
                for name in ('iid', 'icc', 'ipd', 'opd'):
                    enabled = getattr(self, 'enable_' + ('ipdopd' if name in ('ipd', 'opd') else name))
                    if enabled:
                        par = getattr(self, name + '_par')
                        par[self.num_env][:] = par[source]
            if self.enable_iid and any(abs(x) > 7 + 8 * self.iid_quant for x in self.iid_par[self.num_env][:self.nr_iid_par]):
                raise AacError('Invalid repeated PS intensity envelope')
            if self.enable_icc and any(not 0 <= x <= 7 for x in self.icc_par[self.num_env][:self.nr_iid_par]):
                raise AacError('Invalid repeated PS coherence envelope')
            self.num_env += 1
            self.border_position[self.num_env] = 31
        self.is34bands_old = self.is34bands
        if self.enable_iid or self.enable_icc:
            self.is34bands = int((self.enable_iid and self.nr_iid_par == 34) or (self.enable_icc and self.nr_icc_par == 34))
        if not self.enable_ipdopd:
            self.ipd_par[:] = [[0] * 34 for _ in range(5)]
            self.opd_par[:] = [[0] * 34 for _ in range(5)]
        if header:
            self.start = 1
        count = bits.pos - start
        if count > bits_left:
            raise AacError('PS payload crosses SBR extension')
        host.skip(count)
        return count
