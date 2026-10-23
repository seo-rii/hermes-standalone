"""Opus 1.3.1 SILK Q14 synthesis, PLC and comfort noise.

Port of the BSD Skype/IETF reference sources. The complete notices and
disclaimer are retained in standalone/licenses/opus.txt. No native loader.
"""

from types import SimpleNamespace

from standalone.silk_fixed import (cdiv, clz32, div32_varq, i16, i32, inverse32_varq,
    lshift_sat32, rshift_round, sat16, sat32, smlawb, smulwb, smulww, u32)
from standalone.silk_lpc import bwexpander
from standalone.silk_synthesis import sqrt_approx
from standalone.opus_silk_lpc import inverse_gain, nlsf2a

_QUANTIZATION_OFFSETS_Q10 = ((100, 240), (32, 100))


def initialize(state):
    """Add DSP state matching init_decoder followed by decoder_set_fs."""
    state.prev_gain_Q16 = 65536
    state.exc_Q14 = [0] * 320
    state.sLPC_Q14_buf = [0] * 16
    state.outBuf = [0] * 480
    state.lagPrev = 100
    state.lossCnt = 0
    state.prevSignalType = 0
    state.arch = 0
    state.sPLC = SimpleNamespace(pitchL_Q8=0, LTPCoef_Q14=[0]*5, prevLPC_Q12=[0]*16,
        last_frame_lost=0, rand_seed=0, randScale_Q14=0, conc_energy=0,
        conc_energy_shift=0, prevLTP_scale_Q14=0, prevGain_Q16=[65536]*2,
        fs_kHz=0, nb_subfr=2, subfr_length=20)
    state.sCNG = SimpleNamespace(CNG_exc_buf_Q14=[0]*320, CNG_smth_NLSF_Q15=[0]*16,
        CNG_synth_state=[0]*16, CNG_smth_Gain_Q16=0, rand_seed=3176576, fs_kHz=0)
    return state


def _dimensions(state):
    if (state.fs_kHz not in (8, 12, 16) or state.nb_subfr not in (2, 4)
        or state.subfr_length != 5*state.fs_kHz or state.frame_length != state.subfr_length*state.nb_subfr
        or state.ltp_mem_length != 20*state.fs_kHz or state.LPC_order != (16 if state.fs_kHz == 16 else 10)):
        raise ValueError("Invalid Opus SILK synthesis dimensions")
    return state.frame_length, state.subfr_length, state.LPC_order


def sum_sqr_shift(samples):
    """Opus' two-pass, two-bit-headroom energy estimator (not legacy SILK)."""
    length = len(samples)
    if not length:
        return 0, 0
    shift = length.bit_length() - 1
    energy = length
    for index in range(0, length, 2):
        pair = samples[index] ** 2
        if index + 1 < length:
            pair += samples[index + 1] ** 2
        energy = i32(u32(energy) + (u32(pair) >> shift))
    shift = max(0, shift + 3 - clz32(energy))
    energy = 0
    for index in range(0, length, 2):
        pair = samples[index] ** 2
        if index + 1 < length:
            pair += samples[index + 1] ** 2
        energy = i32(u32(energy) + (u32(pair) >> shift))
    return energy, shift


def lpc_analysis_filter(samples, coefficients):
    order = len(coefficients)
    if not 6 <= order <= 16 or order % 2 or len(samples) < order:
        raise ValueError("Invalid Opus LPC analysis dimensions")
    output = [0] * len(samples)
    for index in range(order, len(samples)):
        prediction = 0
        for tap, coefficient in enumerate(coefficients):
            prediction = i32(prediction + i16(samples[index-tap-1]) * i16(coefficient))
        residual = i32((i16(samples[index]) << 12) - prediction)
        output[index] = sat16(rshift_round(residual, 12))
    return output


def _short_term(residual, history, coefficients):
    order = len(coefficients)
    for index, sample in enumerate(residual):
        prediction = order >> 1
        for tap, coefficient in enumerate(coefficients):
            prediction = smlawb(prediction, history[16+index-tap-1], coefficient)
        history[16+index] = sat32(sample + lshift_sat32(prediction, 4))


def decode_core(state, control, pulses):
    length, subframe, order = _dimensions(state)
    if len(pulses) < length or state.prev_gain_Q16 <= 0:
        raise ValueError("Invalid Opus SILK excitation or previous gain")
    offset = _QUANTIZATION_OFFSETS_Q10[state.indices.signalType >> 1][state.indices.quantOffsetType]
    interpolation = state.indices.NLSFInterpCoef_Q2 < 4
    seed = state.indices.Seed
    for index in range(length):
        pulse = i16(pulses[index])
        seed = i32(907633515 + seed * 196314165)
        value = pulse << 14
        value += -1280 if value > 0 else 1280 if value < 0 else 0
        value = i32(value + (offset << 4))
        state.exc_Q14[index] = i32(-value) if seed < 0 else value
        seed = i32(seed + pulse)
    ltp = [0] * (state.ltp_mem_length + length)
    history = state.sLPC_Q14_buf[:] + [0] * subframe
    output, ltp_index = [], state.ltp_mem_length
    for block in range(state.nb_subfr):
        start = block * subframe
        coefficients = control.PredCoef_Q12[block >> 1][:order]
        taps_start = block * 5
        signal_type = state.indices.signalType
        gain = control.Gains_Q16[block]
        if gain <= 0:
            raise ValueError("Invalid Opus SILK subframe gain")
        gain_q10 = gain >> 6
        inverse = inverse32_varq(gain, 47)
        adjustment = div32_varq(state.prev_gain_Q16, gain, 16) if gain != state.prev_gain_Q16 else 65536
        if adjustment != 65536:
            for index in range(16):
                history[index] = smulww(adjustment, history[index])
        state.prev_gain_Q16 = gain
        if state.lossCnt and state.prevSignalType == 2 and signal_type != 2 and block < 2:
            control.LTPCoef_Q14[taps_start:taps_start+5] = [0, 0, 4096, 0, 0]
            control.pitchL[block] = state.lagPrev
            signal_type = 2
        taps = control.LTPCoef_Q14[taps_start:taps_start+5]
        if signal_type == 2:
            lag = control.pitchL[block]
            whiten_start = state.ltp_mem_length-lag-order-2
            if whiten_start <= 0:
                raise ValueError("Opus SILK pitch lag exceeds the synthesis history")
            if block == 0 or (block == 2 and interpolation):
                if block == 2:
                    state.outBuf[state.ltp_mem_length:state.ltp_mem_length+2*subframe] = output[:2*subframe]
                source = state.outBuf[whiten_start+block*subframe:state.ltp_mem_length+block*subframe]
                whitened = lpc_analysis_filter(source, coefficients)
                if block == 0:
                    inverse = i32(smulwb(inverse, control.LTP_scale_Q14) << 2)
                for index in range(lag + 2):
                    ltp[ltp_index-index-1] = smulwb(inverse, whitened[state.ltp_mem_length-index-1-whiten_start])
            elif adjustment != 65536:
                for index in range(lag + 2):
                    position = ltp_index-index-1
                    ltp[position] = smulww(adjustment, ltp[position])
        residual = []
        for index in range(subframe):
            value = state.exc_Q14[start+index]
            if signal_type == 2:
                pointer = ltp_index-lag+2
                prediction = 2
                for tap, coefficient in enumerate(taps):
                    prediction = smlawb(prediction, ltp[pointer-tap], coefficient)
                value = i32(value + (prediction << 1))
                ltp[ltp_index] = i32(value << 1)
                ltp_index += 1
            residual.append(value)
        _short_term(residual, history, coefficients)
        output.extend(sat16(rshift_round(smulww(history[16+index], gain_q10), 8)) for index in range(subframe))
        history[:16] = history[subframe:subframe+16]
    state.sLPC_Q14_buf[:] = history[:16]
    return output


def _plc_reset(state):
    previous = state.sPLC
    previous.pitchL_Q8 = state.frame_length << 7
    previous.prevGain_Q16[:] = [65536, 65536]
    previous.subfr_length = 20
    previous.nb_subfr = 2


def _plc_update(state, control):
    previous, gain = state.sPLC, 0
    state.prevSignalType = state.indices.signalType
    if state.indices.signalType == 2:
        for back in range(state.nb_subfr):
            if back * state.subfr_length >= control.pitchL[state.nb_subfr-1]:
                break
            block = state.nb_subfr-1-back
            candidate = sum(control.LTPCoef_Q14[block*5:block*5+5])
            if candidate > gain:
                gain = candidate
                previous.pitchL_Q8 = control.pitchL[block] << 8
        previous.LTPCoef_Q14[:] = [0, 0, i16(gain), 0, 0]
        if gain < 11469:
            scale = cdiv(11469 << 10, max(gain, 1))
            previous.LTPCoef_Q14[:] = [i16(i32(value*i16(scale)) >> 10) for value in previous.LTPCoef_Q14]
        elif gain > 15565:
            scale = cdiv(15565 << 14, max(gain, 1))
            previous.LTPCoef_Q14[:] = [i16(i32(value*i16(scale)) >> 14) for value in previous.LTPCoef_Q14]
    else:
        previous.pitchL_Q8 = (state.fs_kHz * 18) << 8
        previous.LTPCoef_Q14[:] = [0] * 5
    previous.prevLPC_Q12[:state.LPC_order] = control.PredCoef_Q12[1][:state.LPC_order]
    previous.prevLTP_scale_Q14 = i16(control.LTP_scale_Q14)
    previous.prevGain_Q16[:] = control.Gains_Q16[state.nb_subfr-2:state.nb_subfr]
    previous.subfr_length, previous.nb_subfr = state.subfr_length, state.nb_subfr


def _plc_conceal(state, control):
    length, subframe, order = _dimensions(state)
    previous = state.sPLC
    gains = [value >> 6 for value in previous.prevGain_Q16]
    if state.first_frame_after_reset:
        previous.prevLPC_Q12[:] = [0] * 16
    energies = []
    for block in range(2):
        start = (block+state.nb_subfr-2)*subframe
        samples = [sat16(smulww(state.exc_Q14[start+index], gains[block]) >> 8) for index in range(subframe)]
        energies.append(sum_sqr_shift(samples))
    (first, shift1), (second, shift2) = energies
    start = (previous.nb_subfr-1 if first >> shift2 < second >> shift1 else previous.nb_subfr)*previous.subfr_length-128
    random_start = max(0, start)
    random_scale = previous.randScale_Q14
    harmonic_gain = (32440, 31130)[min(1, state.lossCnt)]
    noise_gain = ((31130, 26214) if state.prevSignalType == 2 else (32440, 29491))[min(1, state.lossCnt)]
    coefficients = previous.prevLPC_Q12[:order]
    bwexpander(coefficients, 64881)
    previous.prevLPC_Q12[:order] = coefficients
    if state.lossCnt == 0:
        random_scale = 16384
        if state.prevSignalType == 2:
            for value in previous.LTPCoef_Q14:
                random_scale = i16(random_scale-value)
            random_scale = max(3277, random_scale)
            random_scale = i16(i32(random_scale*previous.prevLTP_scale_Q14) >> 14)
        else:
            down = max(1 << 22, min(1 << 27, inverse_gain(coefficients))) << 3
            noise_gain = smulwb(down, noise_gain) >> 14
    seed = previous.rand_seed
    lag = rshift_round(previous.pitchL_Q8, 8)
    index = state.ltp_mem_length-lag-order-2
    if index <= 0 or previous.prevGain_Q16[1] <= 0:
        raise ValueError("Invalid Opus PLC pitch or previous gain")
    whitened = lpc_analysis_filter(state.outBuf[index:state.ltp_mem_length], coefficients)
    inverse = min(inverse32_varq(previous.prevGain_Q16[1], 46), 0x3FFFFFFF)
    ltp = [0] * (state.ltp_mem_length+length)
    for position in range(index+order, state.ltp_mem_length):
        ltp[position] = smulwb(inverse, whitened[position-index])
    ltp_index = state.ltp_mem_length
    for block in range(state.nb_subfr):
        for index in range(subframe):
            pointer = ltp_index-lag+2
            prediction = 2
            for tap, coefficient in enumerate(previous.LTPCoef_Q14):
                prediction = smlawb(prediction, ltp[pointer-tap], coefficient)
            seed = i32(907633515 + seed*196314165)
            random_index = (seed >> 25) & 127
            ltp[ltp_index] = i32(smlawb(prediction, state.exc_Q14[random_start+random_index], random_scale) << 2)
            ltp_index += 1
        previous.LTPCoef_Q14[:] = [i16(i32(harmonic_gain*value) >> 15) for value in previous.LTPCoef_Q14]
        if state.indices.signalType != 0:
            random_scale = i16(i32(random_scale*noise_gain) >> 15)
        previous.pitchL_Q8 = min(smlawb(previous.pitchL_Q8, previous.pitchL_Q8, 655), (18*state.fs_kHz) << 8)
        lag = rshift_round(previous.pitchL_Q8, 8)
    history = state.sLPC_Q14_buf[:] + ltp[state.ltp_mem_length:state.ltp_mem_length+length]
    residual = history[16:]
    _short_term(residual, history, coefficients)
    output = [sat16(rshift_round(smulww(history[16+index], gains[1]), 8)) for index in range(length)]
    state.sLPC_Q14_buf[:] = history[length:length+16]
    previous.rand_seed, previous.randScale_Q14 = seed, random_scale
    control.pitchL[:] = [lag] * 4
    return output


def plc(state, control, lost):
    _dimensions(state)
    if state.fs_kHz != state.sPLC.fs_kHz:
        _plc_reset(state)
        state.sPLC.fs_kHz = state.fs_kHz
    if lost:
        output = _plc_conceal(state, control)
        state.lossCnt += 1
        return output
    _plc_update(state, control)
    return None


def glue_frames(state, pcm):
    previous = state.sPLC
    if state.lossCnt:
        previous.conc_energy, previous.conc_energy_shift = sum_sqr_shift(pcm)
        previous.last_frame_lost = 1
    else:
        if previous.last_frame_lost:
            energy, shift = sum_sqr_shift(pcm)
            if shift > previous.conc_energy_shift:
                previous.conc_energy >>= shift-previous.conc_energy_shift
            elif shift < previous.conc_energy_shift:
                energy >>= previous.conc_energy_shift-shift
            if energy > previous.conc_energy:
                zeros = clz32(previous.conc_energy)-1
                previous.conc_energy = i32(previous.conc_energy << zeros)
                energy >>= max(24-zeros, 0)
                fraction = cdiv(previous.conc_energy, max(energy, 1))
                gain = sqrt_approx(fraction) << 4
                slope = cdiv(65536-gain, len(pcm)) << 2
                for index in range(len(pcm)):
                    pcm[index] = i16(smulwb(gain, pcm[index]))
                    gain = i32(gain+slope)
                    if gain > 65536:
                        break
        previous.last_frame_lost = 0
    return pcm


def _cng_reset(state):
    previous = state.sCNG
    step = 32767 // (state.LPC_order+1)
    previous.CNG_smth_NLSF_Q15[:state.LPC_order] = [step*(index+1) for index in range(state.LPC_order)]
    previous.CNG_smth_Gain_Q16 = 0
    previous.rand_seed = 3176576


def cng(state, control, pcm):
    length, subframe, order = _dimensions(state)
    if len(pcm) != length:
        raise ValueError("Opus comfort noise requires one decoder frame")
    previous = state.sCNG
    if state.fs_kHz != previous.fs_kHz:
        _cng_reset(state)
        previous.fs_kHz = state.fs_kHz
    if state.lossCnt == 0 and state.prevSignalType == 0:
        for index in range(order):
            previous.CNG_smth_NLSF_Q15[index] = i16(previous.CNG_smth_NLSF_Q15[index]+smulwb(state.prevNLSF_Q15[index]-previous.CNG_smth_NLSF_Q15[index], 16348))
        maximum, block = 0, 0
        for index in range(state.nb_subfr):
            if control.Gains_Q16[index] > maximum:
                maximum, block = control.Gains_Q16[index], index
        previous.CNG_exc_buf_Q14[subframe:state.nb_subfr*subframe] = previous.CNG_exc_buf_Q14[:(state.nb_subfr-1)*subframe]
        previous.CNG_exc_buf_Q14[:subframe] = state.exc_Q14[block*subframe:(block+1)*subframe]
        for gain in control.Gains_Q16[:state.nb_subfr]:
            previous.CNG_smth_Gain_Q16 = smlawb(previous.CNG_smth_Gain_Q16, i32(gain-previous.CNG_smth_Gain_Q16), 4634)
    if state.lossCnt:
        gain = smulww(state.sPLC.randScale_Q14, state.sPLC.prevGain_Q16[1])
        smoothed = previous.CNG_smth_Gain_Q16
        if gain >= (1 << 21) or smoothed > (1 << 23):
            difference = i32((smoothed >> 16)**2-i32((gain >> 16)**2 << 5))
            gain = i32(sqrt_approx(difference) << 16)
        else:
            difference = i32(smulww(smoothed, smoothed)-i32(smulww(gain, gain) << 5))
            gain = i32(sqrt_approx(difference) << 8)
        gain_q10 = gain >> 6
        mask, seed = 255, previous.rand_seed
        while mask > length:
            mask >>= 1
        residual = []
        for _ in range(length):
            seed = i32(907633515+seed*196314165)
            residual.append(previous.CNG_exc_buf_Q14[(seed >> 24) & mask])
        previous.rand_seed = seed
        coefficients = nlsf2a(previous.CNG_smth_NLSF_Q15[:order])
        history = previous.CNG_synth_state[:] + [0]*length
        _short_term(residual, history, coefficients)
        for index in range(length):
            pcm[index] = sat16(pcm[index]+sat16(rshift_round(smulww(history[16+index], gain_q10), 8)))
        previous.CNG_synth_state[:] = history[length:length+16]
    else:
        previous.CNG_synth_state[:order] = [0]*order
    return pcm
