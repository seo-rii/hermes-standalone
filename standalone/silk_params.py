"""SILK frame parameter decoding, with the original SDK's state transitions."""

from types import SimpleNamespace

from standalone import silk_tables as T
from standalone.silk_entropy import SilkError, pulses
from standalone.silk_fixed import log2lin, smulwb


def control():
    return SimpleNamespace(pitchL=[0] * 4, Gains_Q16=[0] * 4, Seed=0,
                           PredCoef_Q12=[[0] * 16, [0] * 16], LTPCoef_Q14=[0] * 20,
                           LTP_scale_Q14=0, PERIndex=0, RateLevelIndex=0,
                           QuantOffsetType=0, sigtype=0, NLSFInterpCoef_Q2=0)


def set_frequency(state, frequency):
    if frequency not in (8, 12, 16, 24):
        raise SilkError("Unsupported SILK internal sampling rate")
    if state.fs_kHz != frequency:
        state.fs_kHz = frequency
        state.frame_length, state.subfr_length = frequency * 20, frequency * 5
        state.LPC_order = 10 if frequency == 8 else 16
        state.sLPC_Q14[:16] = [0] * 16
        state.outBuf[:480] = [0] * 480
        state.prevNLSF_Q15[:] = [0] * 16
        state.lagPrev, state.LastGainIndex, state.prev_sigtype = 100, 1, 0
        state.first_frame_after_reset = 1
        state.HP_A = getattr(T, "SKP_Silk_Dec_A_HP_" + str(frequency))
        state.HP_B = getattr(T, "SKP_Silk_Dec_B_HP_" + str(frequency))


def parameters(state, decoder):
    from standalone.silk_lpc import bwexpander, nlsf2a_stable, nlsf_stabilize

    current = control()
    first = state.nFramesDecoded == 0
    if first:
        index = decoder.symbol(T.SKP_Silk_SamplingRates_CDF, T.SKP_Silk_SamplingRates_offset)
        if index not in range(4):
            raise SilkError("Invalid SILK internal sampling rate")
        set_frequency(state, T.SKP_Silk_SamplingRates_table[index])
        index = decoder.symbol(T.SKP_Silk_type_offset_CDF, T.SKP_Silk_type_offset_CDF_offset)
    else:
        index = decoder.symbol(T.SKP_Silk_type_offset_joint_CDF[state.typeOffsetPrev],
                               T.SKP_Silk_type_offset_CDF_offset)
    current.sigtype, current.QuantOffsetType = index >> 1, index & 1
    state.typeOffsetPrev = index
    if first:
        gain = decoder.symbol(T.SKP_Silk_gain_CDF[current.sigtype], T.SKP_Silk_gain_CDF_offset)
    else:
        gain = decoder.symbol(T.SKP_Silk_delta_gain_CDF, T.SKP_Silk_delta_gain_CDF_offset)
    gains = [gain] + [decoder.symbol(T.SKP_Silk_delta_gain_CDF, T.SKP_Silk_delta_gain_CDF_offset)
                      for _ in range(3)]
    for k, gain in enumerate(gains):
        if k == 0 and first:
            state.LastGainIndex = gain
        else:
            state.LastGainIndex += gain - 4
        inverse_scale = (65536 * ((86 - 6) * 128 // 6)) // 63
        current.Gains_Q16[k] = log2lin(min(smulwb(inverse_scale, state.LastGainIndex) + 2176, 3967))
    order = state.LPC_order
    name = "CB" + str(current.sigtype) + "_" + str(order)
    stages = getattr(T, "SKP_Silk_NLSF_" + name + "_Stage_info")
    cdf = getattr(T, "SKP_Silk_NLSF_MSVQ_" + name + "_CDF")
    starts = getattr(T, "SKP_Silk_NLSF_MSVQ_" + name + "_CDF_start_ptr_offsets")
    middles = getattr(T, "SKP_Silk_NLSF_MSVQ_" + name + "_CDF_middle_idx")
    vectors = getattr(T, "SKP_Silk_NLSF_MSVQ_" + name + "_Q15")
    nlsf = [0] * order
    for (size, offset, _), start, middle in zip(stages, starts, middles):
        index = decoder.symbol(cdf[start:start + size + 1], middle)
        if not 0 <= index < size:
            raise SilkError("Invalid SILK NLSF vector index")
        offset += index * order
        for i in range(order):
            nlsf[i] += vectors[offset + i]
    nlsf_stabilize(nlsf, getattr(T, "SKP_Silk_NLSF_MSVQ_" + name + "_ndelta_min_Q15"))
    current.NLSFInterpCoef_Q2 = decoder.symbol(T.SKP_Silk_NLSF_interpolation_factor_CDF,
                                             T.SKP_Silk_NLSF_interpolation_factor_offset)
    if state.first_frame_after_reset:
        current.NLSFInterpCoef_Q2 = 4
    current.PredCoef_Q12[1][:order] = nlsf2a_stable(nlsf)
    if current.NLSFInterpCoef_Q2 < 4:
        previous = [state.prevNLSF_Q15[i] + ((current.NLSFInterpCoef_Q2 *
                    (nlsf[i] - state.prevNLSF_Q15[i])) >> 2) for i in range(order)]
        current.PredCoef_Q12[0][:order] = nlsf2a_stable(previous)
    else:
        current.PredCoef_Q12[0][:] = current.PredCoef_Q12[1]
    state.prevNLSF_Q15[:order] = nlsf
    if state.lossCnt:
        for coefficients in current.PredCoef_Q12:
            part = coefficients[:order]
            bwexpander(part, 63570)  # BWE_AFTER_LOSS_Q16
            coefficients[:order] = part
    if current.sigtype == 0:
        band = {8: "NB", 12: "MB", 16: "WB", 24: "SWB"}[state.fs_kHz]
        lag = decoder.symbol(getattr(T, "SKP_Silk_pitch_lag_" + band + "_CDF"),
                             getattr(T, "SKP_Silk_pitch_lag_" + band + "_CDF_offset"))
        table = "SKP_Silk_pitch_contour_" + ("NB_" if state.fs_kHz == 8 else "") + "CDF"
        contour = decoder.symbol(getattr(T, table), getattr(T, table + "_offset"))
        lags = T.SKP_Silk_CB_lags_stage2 if state.fs_kHz == 8 else T.SKP_Silk_CB_lags_stage3
        current.pitchL = [2 * state.fs_kHz + lag + row[contour] for row in lags]
        current.PERIndex = decoder.symbol(T.SKP_Silk_LTP_per_index_CDF, T.SKP_Silk_LTP_per_index_CDF_offset)
        cdf = getattr(T, "SKP_Silk_LTP_gain_CDF_" + str(current.PERIndex))
        codebook = getattr(T, "SKP_Silk_LTP_gain_vq_" + str(current.PERIndex) + "_Q14")
        middle = T.SKP_Silk_LTP_gain_CDF_offsets[current.PERIndex]
        for k in range(4):
            index = decoder.symbol(cdf, middle)
            current.LTPCoef_Q14[k * 5:k * 5 + 5] = codebook[index]
        index = decoder.symbol(T.SKP_Silk_LTPscale_CDF, T.SKP_Silk_LTPscale_offset)
        current.LTP_scale_Q14 = T.SKP_Silk_LTPScales_table_Q14[index]
    current.Seed = decoder.symbol(T.SKP_Silk_Seed_CDF, T.SKP_Silk_Seed_offset)
    excitation, current.RateLevelIndex = pulses(decoder, state.frame_length, current.sigtype,
                                               current.QuantOffsetType)
    state.vadFlag = decoder.symbol(T.SKP_Silk_vadflag_CDF, T.SKP_Silk_vadflag_offset)
    state.FrameTermination = decoder.symbol(T.SKP_Silk_FrameTermination_CDF, T.SKP_Silk_FrameTermination_offset)
    state.nBytesLeft = len(decoder.packet) - decoder.length()[1]
    if state.nBytesLeft < 0:
        raise SilkError("SILK parameters exceed packet size")
    if not state.nBytesLeft:
        decoder.finish()
    return current, excitation
