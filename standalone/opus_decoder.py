"""Complete Opus packet decoding and Discord PCM boundary, from Opus 1.3.1.

Copyright (c) 2010 Xiph.Org Foundation, Skype Limited, Jean-Marc Valin
and Koen Vos. Redistribution notices are retained in licenses/opus*.txt.
The algorithm uses the Python SILK/CELT implementations and shared entropy
state; no native codec, executable or network service is used.
"""
import math
import struct

from .opus import OpusError, parse_packet, samples_per_frame
from .opus_celt import CeltDecoder
from .opus_celt_plc import f32
from .opus_range import OpusRangeDecoder
from .opus_silk import SilkDecoder
from .opus_tables import WINDOW


_ERRORS = {-1: 'invalid argument', -2: 'buffer too small',
           -3: 'internal error', -4: 'corrupted stream'}
_BANDWIDTHS = dict(zip(('narrowband','mediumband','wideband','superwideband','fullband'),range(1101,1106)))
_END_BANDS = {1101:13,1102:17,1103:17,1104:19,1105:21}


class DecodeError(OpusError):
    def __init__(self, code, message=None):
        self.code = code
        super().__init__(message or _ERRORS.get(code,'Opus decoding error'))


def soft_clip(pcm, channels, memory):
    """Apply the original stateful float32 Opus PCM soft-clipping curve."""
    size = len(pcm) // channels
    if not size:
        return pcm
    pcm[:] = [f32(max(-2.,min(2.,value))) for value in pcm]
    for channel in range(channels):
        a = memory[channel]
        for i in range(size):
            index = i * channels + channel
            value = pcm[index]
            if value * a >= 0:
                break
            pcm[index] = f32(value + f32(f32(a * value) * value))
        current = 0
        first = pcm[channel]
        while True:
            index = current
            while index < size and -1. <= pcm[index * channels + channel] <= 1.:
                index += 1
            if index == size:
                a = 0.
                break
            peak = start = end = index
            value = pcm[index * channels + channel]
            maximum = abs(value)
            while start > 0 and value * pcm[(start - 1) * channels + channel] >= 0:
                start -= 1
            while end < size and value * pcm[end * channels + channel] >= 0:
                if abs(pcm[end * channels + channel]) > maximum:
                    maximum = abs(pcm[end * channels + channel])
                    peak = end
                end += 1
            special = start == 0 and value * pcm[channel] >= 0
            a = f32(f32(maximum - 1.) / f32(maximum * maximum))
            a = f32(a + f32(a * f32(2.4e-7)))
            if value > 0:
                a = -a
            for i in range(start,end):
                position = i * channels + channel
                sample = pcm[position]
                pcm[position] = f32(sample + f32(f32(a * sample) * sample))
            if special and peak >= 2:
                offset = f32(first - pcm[channel])
                delta = f32(offset / peak)
                for i in range(current,peak):
                    position = i * channels + channel
                    offset = f32(offset - delta)
                    pcm[position] = max(-1.,min(1.,f32(pcm[position] + offset)))
            current = end
            if current == size:
                break
        memory[channel] = a
    return pcm


class OpusDecoder:
    """Stateful SILK, CELT and hybrid decoder at any supported API rate.

    ``decode`` returns interleaved signed PCM samples and ``decode_float``
    returns normalized floats. ``frame_size`` is the output capacity per
    channel, or the requested duration for PLC/FEC, measured at sample_rate.
    """
    def __init__(self, sample_rate=48000, channels=2):
        if type(sample_rate) is not int or type(channels) is not int or sample_rate not in (8000,12000,16000,24000,48000) or channels not in (1,2):
            raise DecodeError(-1,'Unsupported Opus output format')
        self.sample_rate, self.channels = sample_rate, channels
        self.celt = CeltDecoder(channels,sample_rate)
        self.silk = SilkDecoder()
        self.gain_q8 = 0
        self._silk_rate = self._silk_channels = self._silk_payload_ms = self._silk_pitch = 0
        self.reset()

    def reset(self):
        self.celt.reset()
        self.silk.reset()
        self.stream_channels = self.channels
        self.bandwidth = self.mode = self.prev_mode = 0
        self.frame_size = self.sample_rate // 400
        self.prev_redundancy = False
        self.last_packet_duration = self.final_range = 0
        self.softclip_memory = [0.,0.]

    @property
    def pitch(self):
        return self.celt.postfilter_period if self.prev_mode == 'celt' else self._silk_pitch

    @property
    def phase_inversion_disabled(self):
        return bool(self.celt.disable_inv)

    def set_phase_inversion_disabled(self, value):
        if value not in (0,1):
            raise DecodeError(-1)
        self.celt.disable_inv = bool(value)

    def set_gain(self, gain_q8):
        if type(gain_q8) is not int or not -32768 <= gain_q8 <= 32767:
            raise DecodeError(-1,'Opus gain requires a signed 16-bit Q8 dB value')
        self.gain_q8 = gain_q8
        return 0

    def _fade(self, first, second, count):
        step = 48000 // self.sample_rate
        result = []
        for i in range(count):
            weight = f32(WINDOW[i * step] * WINDOW[i * step])
            for channel in range(self.channels):
                index = i * self.channels + channel
                result.append(f32(f32(weight * second[index]) + f32(f32(1. - weight) * first[index])))
        return result

    def _frame(self, payload, frame_size, fec=False):
        f2, f5, f10, f20 = (self.sample_rate // divisor for divisor in (400,200,100,50))
        if frame_size < f2:
            raise DecodeError(-2)
        frame_size = min(frame_size,self.sample_rate * 3 // 25)
        length = len(payload) if payload is not None else 0
        if length <= 1:
            payload = None
            frame_size = min(frame_size,self.frame_size)
        if payload is not None:
            size, mode, bandwidth = self.frame_size,self.mode,self.bandwidth
            decoder = OpusRangeDecoder(payload)
        else:
            size, mode, bandwidth = frame_size,self.prev_mode,0
            decoder = None
            if not mode:
                return [0.] * (size * self.channels)
            if size > f20:
                result = []
                while len(result) < size * self.channels:
                    result.extend(self._frame(None,min(f20,size - len(result) // self.channels)))
                return result
            if size < f20:
                if size > f10:
                    size = f10
                elif mode != 'silk' and f5 < size < f10:
                    size = f5
        transition = payload is not None and self.prev_mode and (
            mode == 'celt' and self.prev_mode != 'celt' and not self.prev_redundancy
            or mode != 'celt' and self.prev_mode == 'celt')
        transition_pcm = None
        if transition and mode == 'celt':
            transition_pcm = self._frame(None,min(f5,size))
        if size > frame_size:
            raise DecodeError(-1)
        silk_pcm = None
        if mode != 'celt':
            if self.prev_mode == 'celt':
                self.silk.reset()
            self._silk_payload_ms = max(10,1000 * size // self.sample_rate)
            if payload is not None:
                self._silk_channels = self.stream_channels
                self._silk_rate = {1101:8000,1102:12000,1103:16000}.get(bandwidth,16000) if mode == 'silk' else 16000
            try:
                silk_pcm = self.silk.decode(decoder,internal_rate=self._silk_rate,api_rate=self.sample_rate,
                    duration_ms=self._silk_payload_ms,channels=self._silk_channels,out_channels=self.channels,
                    lost=payload is None,fec=fec and payload is not None)
            except ValueError as error:
                raise DecodeError(-3,'Opus SILK decode failed') from error
            self._silk_pitch = self.silk.prev_pitch_lag
        redundancy = celt_to_silk = False
        redundant_bytes = redundant_rng = 0
        if not fec and mode != 'celt' and payload is not None and decoder.tell() + 17 + 20 * (self.mode == 'hybrid') <= 8 * length:
            redundancy = bool(decoder.bit_logp(12)) if mode == 'hybrid' else True
            if redundancy:
                celt_to_silk = bool(decoder.bit_logp(1))
                redundant_bytes = decoder.uint(256) + 2 if mode == 'hybrid' else length - ((decoder.tell() + 7) >> 3)
                length -= redundant_bytes
                if length * 8 < decoder.tell():
                    length = redundant_bytes = 0
                    redundancy = False
                decoder.storage -= redundant_bytes
        if redundancy:
            transition = False
        if transition and mode != 'celt':
            transition_pcm = self._frame(None,min(f5,size))
        if bandwidth:
            self.celt.end = _END_BANDS[bandwidth]
        self.celt.stream_channels = self.stream_channels
        redundant_pcm = None
        try:
            if redundancy and celt_to_silk:
                redundant_pcm = self.celt.decode_float(payload[length:length + redundant_bytes],f5,start=0)
                redundant_rng = self.celt.rng
            self.celt.start = 0 if mode == 'celt' else 17
            if mode != 'silk':
                if mode != self.prev_mode and self.prev_mode and not self.prev_redundancy:
                    self.celt.reset()
                pcm = self.celt.decode_float(None if fec or payload is None else payload[:length],min(f20,size),range_decoder=decoder)
            else:
                pcm = [0.] * (size * self.channels)
                if self.prev_mode == 'hybrid' and not (redundancy and celt_to_silk and self.prev_redundancy):
                    pcm[:f2 * self.channels] = self.celt.decode_float(b'\xff\xff',f2,start=0)
            if mode != 'celt':
                pcm = [f32(value + silk_pcm[i] / 32768.) for i,value in enumerate(pcm)]
            if redundancy and not celt_to_silk:
                self.celt.reset()
                redundant_pcm = self.celt.decode_float(payload[length:length + redundant_bytes],f5,start=0)
                redundant_rng = self.celt.rng
                offset = (size - f2) * self.channels
                pcm[offset:] = self._fade(pcm[offset:],redundant_pcm[f2 * self.channels:],f2)
            if redundancy and celt_to_silk:
                midpoint = f2 * self.channels
                pcm[:midpoint] = redundant_pcm[:midpoint]
                pcm[midpoint:2 * midpoint] = self._fade(redundant_pcm[midpoint:],pcm[midpoint:],f2)
        except ValueError as error:
            raise DecodeError(-4,'Opus CELT decode failed') from error
        if transition:
            midpoint = f2 * self.channels
            if size >= f5:
                pcm[:midpoint] = transition_pcm[:midpoint]
                pcm[midpoint:2 * midpoint] = self._fade(transition_pcm[midpoint:],pcm[midpoint:],f2)
            else:
                pcm[:midpoint] = self._fade(transition_pcm,pcm,f2)
        if self.gain_q8:
            exponent = f32(f32(6.48814081e-4) * self.gain_q8)
            gain = f32(math.exp(.6931471805599453094 * exponent))
            pcm = [f32(value * gain) for value in pcm]
        self.final_range = 0 if length <= 1 else decoder.rng ^ redundant_rng
        self.prev_mode = mode
        self.prev_redundancy = bool(redundancy and not celt_to_silk)
        return pcm

    def _native(self, data, frame_size, fec, clip):
        if type(frame_size) is not int or frame_size <= 0 or fec not in (False,True):
            raise DecodeError(-1)
        data = None if data is None else bytes(data)
        if (fec or not data) and frame_size % (self.sample_rate // 400):
            raise DecodeError(-1)
        if not data:
            result = []
            while len(result) < frame_size * self.channels:
                result.extend(self._frame(None,frame_size - len(result) // self.channels))
            self.last_packet_duration = len(result) // self.channels
            return result
        try:
            packet = parse_packet(data)
        except OpusError as error:
            raise DecodeError(-4) from error
        packet_size = samples_per_frame(packet.toc,self.sample_rate)
        if fec:
            if frame_size < packet_size or packet.mode == 'celt' or self.mode == 'celt':
                return self._native(None,frame_size,False,clip)
            previous_duration = self.last_packet_duration
            result = []
            if frame_size > packet_size:
                try:
                    result = self._native(None,frame_size - packet_size,False,clip)
                except DecodeError:
                    self.last_packet_duration = previous_duration
                    raise
            self.mode,self.bandwidth,self.frame_size,self.stream_channels = packet.mode,_BANDWIDTHS[packet.bandwidth],packet_size,packet.channels
            result.extend(self._frame(packet.frames[0],packet_size,True))
            self.last_packet_duration = frame_size
            return result
        if len(packet.frames) * packet_size > frame_size:
            raise DecodeError(-2)
        self.mode,self.bandwidth,self.frame_size,self.stream_channels = packet.mode,_BANDWIDTHS[packet.bandwidth],packet_size,packet.channels
        result = []
        for payload in packet.frames:
            result.extend(self._frame(payload,frame_size - len(result) // self.channels))
        self.last_packet_duration = len(result) // self.channels
        if clip:
            soft_clip(result,self.channels,self.softclip_memory)
        else:
            self.softclip_memory[:] = [0.,0.]
        return result

    def decode_float(self, data, frame_size=5760, *, fec=False):
        return self._native(data,frame_size,fec,False)

    def decode(self, data, frame_size=5760, *, fec=False):
        return [max(-32768,min(32767,round(value * 32768.))) for value in self._native(data,frame_size,fec,True)]

    def decode_pcm16(self, data, frame_size=5760, *, fec=False):
        pcm = OpusDecoder.decode(self,data,frame_size,fec=fec)
        return struct.pack('<' + str(len(pcm)) + 'h',*pcm)


class Decoder(OpusDecoder):
    """discord.py's 48 kHz stereo PCM16 decoder interface."""
    SAMPLING_RATE = 48000
    CHANNELS = 2
    FRAME_LENGTH = 20
    SAMPLE_SIZE = 4
    SAMPLES_PER_FRAME = 960
    FRAME_SIZE = 3840

    def __init__(self):
        super().__init__(self.SAMPLING_RATE,self.CHANNELS)

    @staticmethod
    def get_opus_version():
        return 'libopus 1.3.1 (Python standard-library port)'

    @staticmethod
    def packet_get_nb_frames(data):
        if not data:
            raise DecodeError(-1)
        code = data[0] & 3
        if code == 3:
            if len(data) < 2:
                raise DecodeError(-4)
            return data[1] & 63
        return 1 if code == 0 else 2

    @staticmethod
    def packet_get_nb_channels(data):
        if not data:
            raise DecodeError(-1)
        return 2 if data[0] & 4 else 1

    @classmethod
    def packet_get_samples_per_frame(cls,data):
        if not data:
            raise DecodeError(-1)
        return samples_per_frame(data[0],cls.SAMPLING_RATE)

    def _get_last_packet_duration(self):
        return self.last_packet_duration

    def _set_gain(self,adjustment):
        return OpusDecoder.set_gain(self,adjustment)

    def set_gain(self,dB):
        return self._set_gain(max(-32768,min(32767,round(dB * 256))))

    def set_volume(self,mult):
        return self.set_gain(20 * math.log10(mult))

    def decode(self,data,*,fec=False):
        if data is None and fec:
            raise TypeError('Invalid arguments: FEC cannot be used with null data')
        size = self.last_packet_duration or self.SAMPLES_PER_FRAME if data is None else self.packet_get_nb_frames(data) * self.packet_get_samples_per_frame(data)
        return self.decode_pcm16(data,size,fec=fec)
