"""Pure Python Skype SILK v3 decoder and standard/Tencent file framing.

No native library or external decoder is loaded. The fixed-point SDK algorithm
and its tables are attributed in silk-provenance.json.
"""

from pathlib import Path
import os
import struct
import tempfile
from types import SimpleNamespace
import wave

from standalone.silk_entropy import RangeDecoder, SilkError
from standalone.silk_params import control, parameters, set_frequency


def new_state():
    state = SimpleNamespace(
        fs_kHz=0, frame_length=0, subfr_length=0, LPC_order=0,
        sLTP_Q16=[0] * 960, sLPC_Q14=[0] * 136, exc_Q10=[0] * 480,
        res_Q10=[0] * 480, outBuf=[0] * 960, prevNLSF_Q15=[0] * 16,
        HPState=[0, 0], prev_inv_gain_Q16=65536, typeOffsetPrev=0,
        prev_API_sampleRate=0, moreInternalDecoderFrames=0, nFramesDecoded=0,
        nFramesInPacket=0, nBytesLeft=0, FrameTermination=0, vadFlag=0,
        no_FEC_counter=0, inband_FEC_offset=0, lossCnt=0,
        sPLC=SimpleNamespace(pitchL_Q8=0, LTPCoef_Q14=[0] * 5, prevLPC_Q12=[0] * 16,
                             last_frame_lost=0, rand_seed=0, randScale_Q14=0,
                             conc_energy=0, conc_energy_shift=0, prevLTP_scale_Q14=0,
                             prevGain_Q16=[0] * 4, fs_kHz=0),
        sCNG=SimpleNamespace(CNG_exc_buf_Q10=[0] * 480, CNG_smth_NLSF_Q15=[0] * 16,
                             CNG_synth_state=[0] * 16, CNG_smth_Gain_Q16=0,
                             rand_seed=3176576, fs_kHz=0),
    )
    set_frequency(state, 24)
    state.sCNG.CNG_smth_NLSF_Q15[:] = [(32767 // 17) * (i + 1) for i in range(16)]
    state.sPLC.pitchL_Q8 = state.frame_length >> 1
    return state


class Decoder:
    def __init__(self, rate=24000):
        if type(rate) is not int or not 8000 <= rate <= 48000:
            raise SilkError("SILK output sampling rate must be between 8000 and 48000 Hz")
        self.rate, self.state = rate, new_state()
        self.resampler, self.resampler_input_rate = None, None

    def decode(self, packet):
        from standalone.silk_resample import Resampler
        from standalone.silk_synthesis import synthesis

        state = self.state
        state.nFramesDecoded = 0
        decoder = RangeDecoder(packet) if packet is not None else None
        pcm = []
        while True:
            old_rate = state.fs_kHz
            if decoder is not None:
                current, excitation = parameters(state, decoder)
                state.nFramesDecoded += 1
                more = state.nBytesLeft > 0 and state.FrameTermination == 1 and state.nFramesDecoded < 5
                state.moreInternalDecoderFrames = int(more)
                if not more:
                    state.nFramesInPacket = state.nFramesDecoded
                    if state.vadFlag:
                        if state.FrameTermination == 0:
                            state.no_FEC_counter += 1
                            if state.no_FEC_counter > 10:
                                state.inband_FEC_offset = 0
                        elif state.FrameTermination in (2, 3):
                            state.inband_FEC_offset = state.FrameTermination - 1
                            state.no_FEC_counter = 0
            else:
                current, excitation, more = control(), None, False
            samples = synthesis(state, current, excitation, lost=decoder is None)
            input_rate = state.fs_kHz * 1000
            if input_rate != self.rate:
                if (self.resampler is None or self.resampler_input_rate != input_rate
                        or old_rate != state.fs_kHz or state.prev_API_sampleRate != self.rate):
                    self.resampler = Resampler(input_rate, self.rate)
                    self.resampler_input_rate = input_rate
                samples = self.resampler.process(samples)
                # The SDK returns this many samples from its possibly rounded
                # internal resampler output for arbitrary API sampling rates.
                samples = samples[:state.frame_length * self.rate // input_rate]
            state.prev_API_sampleRate = self.rate
            pcm.extend(samples)
            if not more:
                break
        return struct.pack("<" + "h" * len(pcm), *pcm)


def silk_to_wav(source, destination, rate=24000):
    """Decode bounded standard or Tencent framing to mono little-endian PCM."""
    source, destination = Path(source), Path(destination)
    decoder = Decoder(rate)
    if source.resolve() == destination.resolve():
        raise SilkError("SILK source and WAV destination must differ")
    if source.stat().st_size > 32 * 1024 * 1024:
        raise SilkError("SILK source exceeds 32 MiB")
    output, packets = 0, 0
    fd, temporary = tempfile.mkstemp(prefix=".hermes-silk-", suffix=".wav", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output_stream, source.open("rb") as stream, wave.open(output_stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            first = stream.read(1)
            tencent = first == b"\x02"
            header = stream.read(9) if tencent else first + stream.read(8)
            if header != b"#!SILK_V3":
                raise SilkError("Invalid SILK v3 file header")
            while True:
                raw = stream.read(2)
                if raw == b"":
                    break
                if len(raw) != 2:
                    raise SilkError("Truncated SILK packet length")
                size = struct.unpack("<h", raw)[0]
                if size == -1:
                    if stream.read(1):
                        raise SilkError("Invalid SILK terminator or trailing data")
                    break
                if not 0 < size <= 1024:
                    raise SilkError("Invalid SILK packet size")
                packet = stream.read(size)
                if len(packet) != size:
                    raise SilkError("Truncated SILK packet")
                pcm = decoder.decode(packet)
                output += len(pcm)
                packets += 1
                if output > 64 * 1024 * 1024:
                    raise SilkError("Decoded SILK audio exceeds 64 MiB")
                wav.writeframesraw(pcm)
            if not packets:
                raise SilkError("SILK file contains no audio packets")
        os.replace(temporary, destination)
        return output // 2
    finally:
        Path(temporary).unlink(missing_ok=True)
