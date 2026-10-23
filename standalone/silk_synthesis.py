"""Fixed-point Skype SILK synthesis, packet-loss concealment and comfort noise.

Derived from the 2006–2012 Skype SILK SDK C implementation. The original
copyright, redistribution terms and disclaimer are in licenses/skype-silk-sdk.txt.
All arithmetic uses explicit SDK widths; this module contains no native loader.
"""

from __future__ import annotations

from types import SimpleNamespace

from standalone.silk_fixed import (cdiv, clz32, div32_varq, i16, i32,
                                  inverse32_varq, lshift_sat32, rshift_round,
                                  sat16, sat32, smlawb, smulwb, smulww, u32)
from standalone.silk_tables import SKP_Silk_Quantization_Offsets_Q10


def initialize(state):
    """Initialize DSP-owned state after the caller sets frame_length/LPC_order.

    As in SKP_Silk_init_decoder, this is a whole-decoder initialization; a
    sampling-rate change must reset only the fields in decoder_set_fs instead.
    """
    for name, size in (("sLTP_Q16", 960), ("sLPC_Q14", 136), ("exc_Q10", 480),
                       ("res_Q10", 480), ("outBuf", 960), ("HPState", 2)):
        setattr(state, name, [0] * size)
    state.prev_inv_gain_Q16 = 65536
    state.lossCnt = 0
    state.sPLC = SimpleNamespace(pitchL_Q8=state.frame_length >> 1, LTPCoef_Q14=[0] * 5,
        prevLPC_Q12=[0] * 16, last_frame_lost=0, rand_seed=0, randScale_Q14=0,
        conc_energy=0, conc_energy_shift=0, prevLTP_scale_Q14=0, prevGain_Q16=[0] * 4, fs_kHz=0)
    step = 32767 // (state.LPC_order + 1)
    state.sCNG = SimpleNamespace(CNG_exc_buf_Q10=[0] * 480,
        CNG_smth_NLSF_Q15=[step * (index + 1) if index < state.LPC_order else 0 for index in range(16)],
        CNG_synth_state=[0] * 16, CNG_smth_Gain_Q16=0, rand_seed=3176576, fs_kHz=0)
    return state


def sqrt_approx(value):
    if value <= 0:
        return 0
    zeros = clz32(value)
    rotation = (24 - zeros) & 31
    fraction = ((u32(value) >> rotation) | u32(value << (32 - rotation))) & 127
    output = (32768 if zeros & 1 else 46214) >> (zeros >> 1)
    return smlawb(output, output, 213 * fraction)


def sum_sqr_shift(samples, *, unaligned=False):
    """Return the SDK's pairwise, overflow-scaled energy and right shift.

    Decoder frame/subframe buffers are 32-bit aligned. The optional alignment
    flag also models the SDK helper's 16-bit-offset input boundary.
    """
    if not samples:
        return 0, 0
    energy, index = (samples[0] * samples[0], 1) if unaligned else (0, 0)
    shift, end = 0, len(samples) - 1
    while index < end:
        energy = i32(energy + samples[index] ** 2 + samples[index + 1] ** 2)
        index += 2
        if energy < 0:
            energy, shift = u32(energy) >> 2, 2
            break
    while index < end:
        pair = u32(samples[index] ** 2 + samples[index + 1] ** 2)
        energy = i32(energy + (pair >> shift))
        index += 2
        if energy < 0:
            energy, shift = u32(energy) >> 2, shift + 2
    if index == end:
        energy = i32(energy + (samples[index] ** 2 >> shift))
    if energy & 0xC0000000:
        energy, shift = u32(energy) >> 2, shift + 2
    return energy, shift


def ma_prediction(samples, coefficients, history=None):
    order = len(coefficients)
    history = [0] * order if history is None else history
    output = []
    for sample in samples:
        residual = rshift_round(i32((sample << 12) - history[0]), 12)
        for index in range(order - 1):
            history[index] = i32(history[index + 1] + sample * coefficients[index])
        history[order - 1] = i32(sample * coefficients[order - 1])
        output.append(sat16(residual))
    return output


def _short_term(residual, history, coefficients):
    output = []
    for index, sample in enumerate(residual):
        predicted = 0
        for tap, coefficient in enumerate(coefficients):
            predicted = smlawb(predicted, history[16 + index - tap - 1], coefficient)
        value = i32(sample + predicted)
        output.append(value)
        history[16 + index] = i32(value << 4)
    return output


def lpc_synthesis(samples, coefficients, gain_q26, history):
    order, output = len(coefficients), []
    for sample in samples:
        predicted = 0
        for index, coefficient in enumerate(coefficients):
            predicted = smlawb(predicted, history[order - index - 1], coefficient)
        value = sat32(predicted + smulwb(gain_q26, sample))
        output.append(sat16(rshift_round(value, 10)))
        history[:order - 1] = history[1:order]
        history[order - 1] = lshift_sat32(value, 4)
    return output


def biquad(samples, numerator, denominator, history):
    first, second = history
    output = []
    for sample in samples:
        value = i32(first + sample * numerator[0])
        first = i32(second + sample * numerator[1] + (smulwb(value, -denominator[0]) << 3))
        second = i32((smulwb(value, -denominator[1]) << 3) + sample * numerator[2])
        output.append(sat16(rshift_round(value, 13) + 1))
    history[:] = [first, second]
    return output


def decode_core(state, control, pulses):
    length, subframe, order = state.frame_length, state.subfr_length, state.LPC_order
    if len(pulses) != length or length not in (160, 240, 320, 480) or order not in (10, 16):
        raise ValueError("Invalid SILK synthesis frame dimensions")
    if state.prev_inv_gain_Q16 <= 0:
        raise ValueError("Invalid SILK previous inverse gain")
    offset = SKP_Silk_Quantization_Offsets_Q10[control.sigtype][control.QuantOffsetType]
    seed, interpolation = control.Seed, int(control.NLSFInterpCoef_Q2 < 4)
    for index, pulse in enumerate(pulses):
        seed = i32(907633515 + seed * 196314165)
        dither = seed >> 31
        state.exc_Q10[index] = i32((((pulse << 10) + offset) ^ dither) - dither)
        seed = i32(seed + pulse)
    ltp_index = length
    for block in range(4):
        start, signal_type = block * subframe, control.sigtype
        coefficients = control.PredCoef_Q12[block >> 1][:order]
        taps_start, gain = block * 5, control.Gains_Q16[block]
        inverse = min(inverse32_varq(max(gain, 1), 32), 32767)
        adjustment = div32_varq(inverse, state.prev_inv_gain_Q16, 16) if inverse != state.prev_inv_gain_Q16 else 65536
        if state.lossCnt and state.prev_sigtype == 0 and control.sigtype == 1 and block < 2:
            control.LTPCoef_Q14[taps_start:taps_start + 5] = [0, 0, 4096, 0, 0]
            signal_type = 0
            control.pitchL[block] = state.lagPrev
        taps = control.LTPCoef_Q14[taps_start:taps_start + 5]
        if signal_type == 0:
            lag = control.pitchL[block]
            if not 0 < lag <= length - order - 2:
                raise ValueError("SILK pitch lag exceeds its synthesis history")
            if block & (3 - (interpolation << 1)) == 0:
                whiten_start = length - lag - order - 2
                whitened = ma_prediction(state.outBuf[whiten_start + start:length + start], coefficients)
                inverse_q32 = i32(inverse << 16)
                if block == 0:
                    inverse_q32 = i32(smulwb(inverse_q32, control.LTP_scale_Q14) << 2)
                for index in range(lag + 2):
                    state.sLTP_Q16[ltp_index - index - 1] = smulwb(inverse_q32, whitened[length - index - 1 - whiten_start])
            elif adjustment != 65536:
                for index in range(lag + 2):
                    position = ltp_index - index - 1
                    state.sLTP_Q16[position] = smulww(adjustment, state.sLTP_Q16[position])
        for index in range(16):
            state.sLPC_Q14[index] = smulww(adjustment, state.sLPC_Q14[index])
        state.prev_inv_gain_Q16 = inverse
        residual = []
        for index in range(subframe):
            value = state.exc_Q10[start + index]
            if signal_type == 0:
                predicted = 0
                for tap in range(5):
                    predicted = smlawb(predicted, state.sLTP_Q16[ltp_index - lag + 2 - tap], taps[tap])
                value = i32(value + rshift_round(predicted, 4))
                state.sLTP_Q16[ltp_index] = i32(value << 6)
                ltp_index += 1
            residual.append(value)
        state.res_Q10[start:start + subframe] = residual
        vector = _short_term(residual, state.sLPC_Q14, coefficients)
        state.outBuf[length + start:length + start + subframe] = [sat16(rshift_round(smulww(value, gain), 10)) for value in vector]
        state.sLPC_Q14[:16] = state.sLPC_Q14[subframe:subframe + 16]
    return state.outBuf[length:length * 2]


def _plc_update(state, control):
    plc, gain = state.sPLC, 0
    state.prev_sigtype = control.sigtype
    if control.sigtype == 0:
        block = 0
        while block * state.subfr_length < control.pitchL[3]:
            index = 3 - block
            candidate = sum(control.LTPCoef_Q14[index * 5:index * 5 + 5])
            if candidate > gain:
                gain = candidate
                plc.LTPCoef_Q14[:] = control.LTPCoef_Q14[index * 5:index * 5 + 5]
                plc.pitchL_Q8 = control.pitchL[index] << 8
            block += 1
        plc.LTPCoef_Q14[:] = [0, 0, i16(gain), 0, 0]
        if gain < 11469:
            scale = cdiv(11469 << 10, max(gain, 1))
            plc.LTPCoef_Q14[:] = [i16(i16(value) * i16(scale) >> 10) for value in plc.LTPCoef_Q14]
        elif gain > 15565:
            scale = cdiv(15565 << 14, max(gain, 1))
            plc.LTPCoef_Q14[:] = [i16(i16(value) * i16(scale) >> 14) for value in plc.LTPCoef_Q14]
    else:
        plc.pitchL_Q8 = state.fs_kHz * 18 << 8
        plc.LTPCoef_Q14[:] = [0] * 5
    plc.prevLPC_Q12[:state.LPC_order] = control.PredCoef_Q12[1][:state.LPC_order]
    plc.prevLTP_scale_Q14 = i16(control.LTP_scale_Q14)
    plc.prevGain_Q16[:] = control.Gains_Q16


def _plc_conceal(state, control):
    from standalone.silk_lpc import bwexpander, lpc_inverse_pred_gain
    plc, length, subframe = state.sPLC, state.frame_length, state.subfr_length
    state.sLTP_Q16[:length] = state.sLTP_Q16[length:length * 2]
    coefficients = plc.prevLPC_Q12[:state.LPC_order]
    bwexpander(coefficients, 64880)
    plc.prevLPC_Q12[:state.LPC_order] = coefficients
    scaled = [i16(smulww(state.exc_Q10[index], plc.prevGain_Q16[index // subframe]) >> 10)
              for index in range(2 * subframe, length)]
    energy1, shift1 = sum_sqr_shift(scaled[:subframe])
    energy2, shift2 = sum_sqr_shift(scaled[subframe:])
    noise_start = max(0, (3 * subframe if energy1 >> shift2 < energy2 >> shift1 else length) - 128)
    scale, attenuation = plc.randScale_Q14, min(1, state.lossCnt)
    harmonic = (32440, 31130)[attenuation]
    random_gain = ((31130, 26214) if state.prev_sigtype == 0 else (32440, 29491))[attenuation]
    if state.lossCnt == 0:
        scale = 16384
        if state.prev_sigtype == 0:
            for coefficient in plc.LTPCoef_Q14:
                scale = i16(scale - coefficient)
            scale = i16(i16(max(3277, scale)) * i16(plc.prevLTP_scale_Q14) >> 14)
        else:
            _unstable, inverse = lpc_inverse_pred_gain(coefficients)
            reduction = max(1 << 22, min(1 << 27, inverse)) << 3
            random_gain = smulwb(reduction, random_gain) >> 14
    seed, lag, ltp_index, residual = plc.rand_seed, rshift_round(plc.pitchL_Q8, 8), length, []
    for _block in range(4):
        for _index in range(subframe):
            seed = i32(907633515 + seed * 196314165)
            random_index = (seed >> 25) & 127
            predicted = 0
            for tap in range(5):
                position = ltp_index - lag + 2 - tap
                if not 0 <= position < len(state.sLTP_Q16):
                    raise ValueError("SILK concealment pitch exceeds its synthesis history")
                predicted = smlawb(predicted, state.sLTP_Q16[position], plc.LTPCoef_Q14[tap])
            value = i32((smulwb(state.exc_Q10[noise_start + random_index], scale) << 2) + rshift_round(predicted, 4))
            state.sLTP_Q16[ltp_index] = i32(value << 6)
            ltp_index += 1
            residual.append(value)
        plc.LTPCoef_Q14[:] = [i16(i16(harmonic) * i16(value) >> 15) for value in plc.LTPCoef_Q14]
        scale = i16(i16(scale) * i16(random_gain) >> 15)
        plc.pitchL_Q8 = min(i32(plc.pitchL_Q8 + smulwb(plc.pitchL_Q8, 655)), state.fs_kHz * 18 << 8)
        lag = rshift_round(plc.pitchL_Q8, 8)
    vector = []
    for start in range(0, length, subframe):
        vector.extend(_short_term(residual[start:start + subframe], state.sLPC_Q14, coefficients))
        state.sLPC_Q14[:16] = state.sLPC_Q14[subframe:subframe + 16]
    plc.rand_seed, plc.randScale_Q14 = seed, scale
    control.pitchL[:] = [lag] * 4
    return [sat16(rshift_round(smulww(value, plc.prevGain_Q16[3]), 10)) for value in vector]


def plc(state, control, signal, lost=False):
    if state.fs_kHz != state.sPLC.fs_kHz:
        state.sPLC.pitchL_Q8 = state.frame_length >> 1
        state.sPLC.fs_kHz = state.fs_kHz
    if lost:
        signal[:] = _plc_conceal(state, control)
        state.lossCnt += 1
    else:
        _plc_update(state, control)
    return signal


def plc_glue_frames(state, signal):
    plc = state.sPLC
    if state.lossCnt:
        plc.conc_energy, plc.conc_energy_shift = sum_sqr_shift(signal)
        plc.last_frame_lost = 1
    else:
        if plc.last_frame_lost:
            energy, shift = sum_sqr_shift(signal)
            if shift > plc.conc_energy_shift:
                plc.conc_energy >>= shift - plc.conc_energy_shift
            elif shift < plc.conc_energy_shift:
                energy >>= plc.conc_energy_shift - shift
            if energy > plc.conc_energy:
                zeros = clz32(plc.conc_energy) - 1
                plc.conc_energy = i32(plc.conc_energy << zeros)
                energy >>= max(24 - zeros, 0)
                gain = sqrt_approx(cdiv(plc.conc_energy, max(energy, 1)))
                slope = cdiv(4096 - gain, len(signal))
                for index, sample in enumerate(signal):
                    signal[index] = i16(i32(gain * sample) >> 12)
                    gain = min(gain + slope, 4096)
        plc.last_frame_lost = 0
    return signal


def cng(state, control, signal):
    noise, order = state.sCNG, state.LPC_order
    if state.fs_kHz != noise.fs_kHz:
        step = 32767 // (order + 1)
        noise.CNG_smth_NLSF_Q15[:order] = [step * (index + 1) for index in range(order)]
        noise.CNG_smth_Gain_Q16, noise.rand_seed, noise.fs_kHz = 0, 3176576, state.fs_kHz
    if not state.lossCnt and not state.vadFlag:
        for index in range(order):
            difference = state.prevNLSF_Q15[index] - noise.CNG_smth_NLSF_Q15[index]
            noise.CNG_smth_NLSF_Q15[index] = i32(noise.CNG_smth_NLSF_Q15[index] + smulwb(difference, 16348))
        block = 0
        maximum = 0
        for index, gain in enumerate(control.Gains_Q16):
            if gain > maximum:
                maximum, block = gain, index
        size = state.subfr_length
        noise.CNG_exc_buf_Q10[size:size * 4] = noise.CNG_exc_buf_Q10[:size * 3]
        noise.CNG_exc_buf_Q10[:size] = state.exc_Q10[block * size:(block + 1) * size]
        for gain in control.Gains_Q16:
            noise.CNG_smth_Gain_Q16 = i32(noise.CNG_smth_Gain_Q16 + smulwb(gain - noise.CNG_smth_Gain_Q16, 4634))
    if state.lossCnt:
        from standalone.silk_lpc import nlsf2a_stable
        mask = 255
        while mask > len(signal):
            mask >>= 1
        excitation = []
        for _index in range(len(signal)):
            noise.rand_seed = i32(907633515 + noise.rand_seed * 196314165)
            position = (noise.rand_seed >> 24) & mask
            excitation.append(sat16(rshift_round(smulww(noise.CNG_exc_buf_Q10[position], noise.CNG_smth_Gain_Q16), 10)))
        coefficients = nlsf2a_stable(noise.CNG_smth_NLSF_Q15[:order])
        generated = lpc_synthesis(excitation, coefficients, 1 << 26, noise.CNG_synth_state)
        signal[:] = [sat16(sample + extra) for sample, extra in zip(signal, generated)]
    else:
        noise.CNG_synth_state[:order] = [0] * order
    return signal


def synthesis(state, control, pulses=None, *, lost=False):
    """Decode one 20 ms internal-rate frame, including the SDK output HP filter."""
    if lost:
        signal = [0] * state.frame_length
        plc(state, control, signal, lost=True)
    else:
        signal = decode_core(state, control, pulses)
        plc(state, control, signal)
        state.lossCnt, state.prev_sigtype, state.first_frame_after_reset = 0, control.sigtype, 0
    state.outBuf[:state.frame_length] = signal
    plc_glue_frames(state, signal)
    cng(state, control, signal)
    output = biquad(signal, state.HP_B, state.HP_A, state.HPState)
    state.lagPrev = control.pitchL[3]
    return output
