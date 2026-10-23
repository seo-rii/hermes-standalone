"""Stateful received-frame CELT decoder adapted from scalar Opus 1.3.1.

Copyright (c) 2007–2011 Xiph.Org Foundation, CSIRO, Jean-Marc Valin
and Gregory Maxwell. Original notices are retained in licenses/opus*.txt.
"""
from .opus_range import OpusRangeDecoder, OpusRangeError
from .opus_celt_alloc import allocation, init_caps
from .opus_celt_energy import coarse, tf_decode, fine_energy, final_energy
from .opus_celt_bands import decode_bands
from .opus_celt_dsp import anti_collapse, synthesis, comb_filter
from .opus_tables import EBANDS, PREEMPH
from .opus_celt_plc import conceal


class CeltDecoder:
    def __init__(self, channels=2, sample_rate=48000):
        if channels not in (1,2) or sample_rate not in (8000,12000,16000,24000,48000):
            raise ValueError('Unsupported Opus CELT output format')
        self.channels, self.sample_rate = channels, sample_rate
        self.downsample = 48000 // sample_rate
        self.stream_channels, self.start, self.end = channels, 0, 21
        self.disable_inv = channels == 1
        self.reset()

    def reset(self):
        self.histories = [[0.] * 2168 for _ in range(self.channels)]
        self.old_energy = [0.] * 42
        self.old_log_energy = [-28.] * 42
        self.old_log_energy2 = [-28.] * 42
        self.background_energy = [0.] * 42
        self.preemphasis_memory = [0.] * self.channels
        self.rng = self.error = self.loss_count = 0
        self.skip_plc = True
        self.last_pitch_index = 0
        self.lpc = [[0.] * 24 for _ in range(self.channels)]
        self.postfilter_period = self.postfilter_period_old = 0
        self.postfilter_gain = self.postfilter_gain_old = 0.
        self.postfilter_tapset = self.postfilter_tapset_old = 0

    def _deemphasis(self, size):
        output_size = size // self.downsample
        result = [0.] * (output_size * self.channels)
        for channel in range(self.channels):
            memory = self.preemphasis_memory[channel]
            history = self.histories[channel]
            start = 2048 - size
            for i in range(size):
                sample = history[start + i] + 1e-30 + memory
                memory = PREEMPH[0] * sample
                if i % self.downsample == 0:
                    result[(i // self.downsample) * self.channels + channel] = sample / 32768.
            self.preemphasis_memory[channel] = memory
        return result

    def decode_float(self, data, frame_size=960, *, stream_channels=None, start=None, end=None, range_decoder=None):
        """Decode a frame payload; frame_size is samples per output channel.

        Opus packet framing/TOC is handled separately. A hybrid caller supplies
        the range decoder already advanced past its SILK component.
        """
        size = frame_size * self.downsample
        if size not in (120,240,480,960):
            raise ValueError('Invalid Opus CELT frame duration')
        if stream_channels is None:
            stream_channels = self.stream_channels
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        if stream_channels not in (1,2) or start not in (0,17) or not start < end <= 21:
            raise ValueError('Invalid Opus CELT band/channel configuration')
        self.stream_channels, self.start, self.end = stream_channels, start, end
        if data is None or len(data) <= 1:
            conceal(self, size, (size // 120).bit_length() - 1, start, end)
            return self._deemphasis(size)
        if len(data) > 1275:
            raise OpusRangeError('Opus CELT frame exceeds 1275 bytes')
        lm = (size // 120).bit_length() - 1
        multiplier = 1 << lm
        decoder = OpusRangeDecoder(data) if range_decoder is None else range_decoder
        self.skip_plc = self.loss_count != 0
        if stream_channels == 1:
            for band in range(21):
                self.old_energy[band] = max(self.old_energy[band], self.old_energy[21 + band])
        total_bits, tell = len(data) * 8, decoder.tell()
        silence = 1 if tell >= total_bits else decoder.bit_logp(15) if tell == 1 else 0
        if silence:
            tell = total_bits
            decoder.nbits_total += tell - decoder.tell()
        postfilter_gain, postfilter_pitch, postfilter_tapset = 0., 0, 0
        if start == 0 and tell + 16 <= total_bits:
            if decoder.bit_logp(1):
                octave = decoder.uint(6)
                postfilter_pitch = (16 << octave) + decoder.bits(4 + octave) - 1
                quantized_gain = decoder.bits(3)
                if decoder.tell() + 2 <= total_bits:
                    postfilter_tapset = decoder.icdf((2,1,0),2)
                postfilter_gain = .09375 * (quantized_gain + 1)
            tell = decoder.tell()
        if lm > 0 and tell + 3 <= total_bits:
            transient = decoder.bit_logp(3)
            tell = decoder.tell()
        else:
            transient = 0
        intra = decoder.bit_logp(3) if tell + 3 <= total_bits else 0
        coarse(start, end, self.old_energy, intra, decoder, stream_channels, lm)
        tf = tf_decode(start, end, transient, lm, decoder)
        spread = decoder.icdf((25,23,2,0),5) if decoder.tell() + 4 <= total_bits else 2
        cap = init_caps(lm, stream_channels)
        offsets = [0] * 21
        dynalloc_logp = 6
        total_bits <<= 3
        tell = decoder.tell_frac()
        for band in range(start, end):
            width = stream_channels * (EBANDS[band + 1] - EBANDS[band]) << lm
            quantum = min(width << 3, max(48, width))
            probability = dynalloc_logp
            boost = 0
            while tell + (probability << 3) < total_bits and boost < cap[band]:
                flag = decoder.bit_logp(probability)
                tell = decoder.tell_frac()
                if not flag:
                    break
                boost += quantum
                total_bits -= quantum
                probability = 1
            offsets[band] = boost
            if boost > 0:
                dynalloc_logp = max(2, dynalloc_logp - 1)
        trim = decoder.icdf((126,124,119,109,87,41,19,9,4,2,0),7) if tell + 48 <= total_bits else 5
        bits = len(data) * 64 - decoder.tell_frac() - 1
        collapse_reserve = 8 if transient and lm >= 2 and bits >= (lm + 2) << 3 else 0
        bits -= collapse_reserve
        coded, intensity, dual, balance, pulses, fine, priority = allocation(start, end, offsets, cap, trim, bits, stream_channels, lm, decoder)
        fine_energy(start, end, self.old_energy, fine, decoder, stream_channels)
        for history in self.histories:
            history[:2048 - size + 60] = history[size:2108]
        spectra, masks, self.rng = decode_bands(start, end, stream_channels, pulses, multiplier if transient else 0, spread, dual, intensity, tf, len(data) * 64 - collapse_reserve, balance, decoder, lm, coded, self.rng, self.disable_inv)
        collapse_on = decoder.bits(1) if collapse_reserve else 0
        final_energy(start, end, self.old_energy, fine, priority, len(data) * 8 - decoder.tell(), decoder, stream_channels)
        if collapse_on:
            anti_collapse(spectra, masks, lm, start, end, self.old_energy, self.old_log_energy, self.old_log_energy2, pulses, self.rng)
        if silence:
            self.old_energy[:stream_channels * 21] = [-28.] * (stream_channels * 21)
        synthesis(spectra, self.histories, self.old_energy, start, end, self.channels, transient, lm, self.downsample, silence)
        self.postfilter_period = max(self.postfilter_period, 15)
        self.postfilter_period_old = max(self.postfilter_period_old, 15)
        for history in self.histories:
            comb_filter(history, 2048 - size, self.postfilter_period_old, self.postfilter_period, 120, self.postfilter_gain_old, self.postfilter_gain, self.postfilter_tapset_old, self.postfilter_tapset)
            if lm != 0:
                comb_filter(history, 2048 - size + 120, self.postfilter_period, postfilter_pitch, size - 120, self.postfilter_gain, postfilter_gain, self.postfilter_tapset, postfilter_tapset)
        self.postfilter_period_old, self.postfilter_period = self.postfilter_period, postfilter_pitch
        self.postfilter_gain_old, self.postfilter_gain = self.postfilter_gain, postfilter_gain
        self.postfilter_tapset_old, self.postfilter_tapset = self.postfilter_tapset, postfilter_tapset
        if lm != 0:
            self.postfilter_period_old, self.postfilter_gain_old, self.postfilter_tapset_old = self.postfilter_period, self.postfilter_gain, self.postfilter_tapset
        if stream_channels == 1:
            self.old_energy[21:] = self.old_energy[:21]
        if not transient:
            self.old_log_energy2[:] = self.old_log_energy
            self.old_log_energy[:] = self.old_energy
            increase = multiplier * .001 if self.loss_count < 10 else 1.
            self.background_energy[:] = [min(background + increase, energy) for background, energy in zip(self.background_energy, self.old_energy)]
        else:
            self.old_log_energy[:] = [min(previous, energy) for previous, energy in zip(self.old_log_energy, self.old_energy)]
        for channel in range(2):
            for band in list(range(start)) + list(range(end,21)):
                index = channel * 21 + band
                self.old_energy[index] = 0.
                self.old_log_energy[index] = self.old_log_energy2[index] = -28.
        self.rng, self.error, self.loss_count = decoder.rng, self.error | int(bool(decoder.error)), 0
        result = self._deemphasis(size)
        if decoder.tell() > 8 * len(data):
            raise OpusRangeError('Opus CELT frame exceeded its entropy bit budget')
        return result

    def decode(self, data, frame_size=960, **options):
        return [max(-32768, min(32767, round(value * 32768.))) for value in self.decode_float(data, frame_size, **options)]
