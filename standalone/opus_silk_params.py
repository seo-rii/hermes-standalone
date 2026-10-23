"""Opus SILK side information, pulse decoding and parameter reconstruction.

Adapted from BSD Opus 1.3.1, Copyright (c) 2006-2011 Skype Limited.
Original notices and disclaimer are retained in licenses/opus.txt.
"""
from types import SimpleNamespace

from standalone.silk_fixed import i16, log2lin, smulwb
from standalone.silk_lpc import bwexpander
from standalone import opus_silk_tables as T
from standalone.opus_silk_lpc import decode_nlsf, nlsf2a, unpack


def initialize(frequency, subframes):
    if frequency not in (8, 12, 16) or subframes not in (2, 4):
        raise ValueError('Unsupported Opus SILK frame dimensions')
    return SimpleNamespace(fs_kHz=frequency, nb_subfr=subframes,
        frame_length=frequency*5*subframes, subfr_length=frequency*5,
        LPC_order=16 if frequency==16 else 10, ltp_mem_length=frequency*20,
        VAD_flags=[0]*3, ec_prevSignalType=0, ec_prevLagIndex=0,
        LastGainIndex=10, prevNLSF_Q15=[0]*16,
        first_frame_after_reset=1, lossCnt=0,
        indices=SimpleNamespace(GainsIndices=[0]*4, NLSFIndices=[0]*17,
            LTPIndex=[0]*4, lagIndex=0, contourIndex=0, PERIndex=0,
            LTP_scaleIndex=0, NLSFInterpCoef_Q2=0, Seed=0,
            signalType=0, quantOffsetType=0))


def decode_indices(state, decoder, frame=0, lbrr=False, conditional=0):
    indices, order = state.indices, state.LPC_order
    index = decoder.icdf(T.silk_type_offset_VAD_iCDF)+2 if lbrr or state.VAD_flags[frame] else decoder.icdf(T.silk_type_offset_no_VAD_iCDF)
    indices.signalType, indices.quantOffsetType = index>>1, index&1
    indices.GainsIndices[0] = decoder.icdf(T.silk_delta_gain_iCDF) if conditional==2 else (decoder.icdf(T.silk_gain_iCDF[indices.signalType])<<3)+decoder.icdf(T.silk_uniform8_iCDF)
    for k in range(1,state.nb_subfr):indices.GainsIndices[k]=decoder.icdf(T.silk_delta_gain_iCDF)
    band = 'WB' if order==16 else 'NB_MB'
    first = getattr(T,'silk_NLSF_CB1_iCDF_'+band)
    indices.NLSFIndices[0]=decoder.icdf(first[(indices.signalType>>1)*32:(indices.signalType>>1)*32+32])
    offsets, _ = unpack(order,indices.NLSFIndices[0])
    residual = getattr(T,'silk_NLSF_CB2_iCDF_'+band)
    for k,offset in enumerate(offsets):
        index=decoder.icdf(residual[offset:offset+9])
        if index==0:index-=decoder.icdf(T.silk_NLSF_EXT_iCDF)
        elif index==8:index+=decoder.icdf(T.silk_NLSF_EXT_iCDF)
        indices.NLSFIndices[k+1]=index-4
    indices.NLSFInterpCoef_Q2=decoder.icdf(T.silk_NLSF_interpolation_factor_iCDF) if state.nb_subfr==4 else 4
    if indices.signalType==2:
        absolute=True
        if conditional==2 and state.ec_prevSignalType==2:
            delta=decoder.icdf(T.silk_pitch_delta_iCDF)
            if delta:
                indices.lagIndex=i16(state.ec_prevLagIndex+delta-9)
                absolute=False
        if absolute:
            indices.lagIndex=decoder.icdf(T.silk_pitch_lag_iCDF)*(state.fs_kHz>>1)+decoder.icdf(getattr(T,'silk_uniform'+str(state.fs_kHz//2)+'_iCDF'))
        state.ec_prevLagIndex=indices.lagIndex
        contour='silk_pitch_contour_'+('10_ms_' if state.nb_subfr==2 else '')+('NB_' if state.fs_kHz==8 else '')+'iCDF'
        indices.contourIndex=decoder.icdf(getattr(T,contour))
        indices.PERIndex=decoder.icdf(T.silk_LTP_per_index_iCDF)
        for k in range(state.nb_subfr):
            indices.LTPIndex[k]=decoder.icdf(getattr(T,'silk_LTP_gain_iCDF_'+str(indices.PERIndex)))
        indices.LTP_scaleIndex=decoder.icdf(T.silk_LTPscale_iCDF) if conditional==0 else 0
    state.ec_prevSignalType=indices.signalType
    indices.Seed=decoder.icdf(T.silk_uniform4_iCDF)
    return indices


def decode_pulses(decoder, signal_type, quantizer, length):
    if signal_type not in (0,1,2) or quantizer not in (0,1) or length not in (80,120,160,240,320):
        raise ValueError('Invalid Opus SILK excitation dimensions')
    level=decoder.icdf(T.silk_rate_levels_iCDF[signal_type>>1])
    sums, shifts=[],[]
    for _ in range((length+15)//16):
        shift=0;total=decoder.icdf(T.silk_pulses_per_block_iCDF[level])
        while total==17:
            shift+=1
            if shift>10:raise ValueError('Opus SILK pulse shift overflow')
            total=decoder.icdf(T.silk_pulses_per_block_iCDF[9][int(shift==10):])
        sums.append(total);shifts.append(shift)
    def shell(total,level):
        if level<0:return [total]
        if not total:return [0]*(2<<level)
        start=T.silk_shell_code_table_offsets[total]
        first=decoder.icdf(getattr(T,'silk_shell_code_table'+str(level))[start:start+total+1])
        return shell(first,level-1)+shell(total-first,level-1)
    pulses=[]
    for total in sums:pulses.extend(shell(total,3))
    for k,shift in enumerate(shifts):
        if shift:
            for j in range(16):
                value=pulses[k*16+j]
                for _ in range(shift):value=(value<<1)+decoder.icdf(T.silk_lsb_iCDF)
                pulses[k*16+j]=i16(value)
            sums[k]|=shift<<5
    signs=T.silk_sign_iCDF[7*(quantizer+(signal_type<<1)):]
    for k,total in enumerate(sums):
        if total>0:
            table=(signs[min(total&31,6)],0)
            for j in range(16):
                index=k*16+j
                if pulses[index]>0:pulses[index]*=(decoder.icdf(table)<<1)-1
    return pulses


def decode_parameters(state,conditional=0):
    indices,order=state.indices,state.LPC_order
    result=SimpleNamespace(Gains_Q16=[0]*4,PredCoef_Q12=[[0]*16,[0]*16],
        pitchL=[0]*4,LTPCoef_Q14=[0]*20,LTP_scale_Q14=0)
    for k in range(state.nb_subfr):
        index=indices.GainsIndices[k]
        if k==0 and conditional!=2:state.LastGainIndex=max(index,state.LastGainIndex-16)
        else:
            index-=4
            threshold=8+state.LastGainIndex
            state.LastGainIndex+=2*index-threshold if index>threshold else index
        state.LastGainIndex=max(0,min(63,state.LastGainIndex))
        scale=(65536*(((88-2)*128)//6))//63
        result.Gains_Q16[k]=log2lin(min(smulwb(scale,state.LastGainIndex)+(2*128)//6+16*128,3967))
    nlsf=decode_nlsf(indices.NLSFIndices[:order+1])
    result.PredCoef_Q12[1][:order]=nlsf2a(nlsf)
    if state.first_frame_after_reset:indices.NLSFInterpCoef_Q2=4
    if indices.NLSFInterpCoef_Q2<4:
        interpolated=[state.prevNLSF_Q15[k]+(indices.NLSFInterpCoef_Q2*(nlsf[k]-state.prevNLSF_Q15[k])>>2) for k in range(order)]
        result.PredCoef_Q12[0][:order]=nlsf2a(interpolated)
    else:result.PredCoef_Q12[0][:]=result.PredCoef_Q12[1]
    state.prevNLSF_Q15[:order]=nlsf
    if state.lossCnt:
        for values in result.PredCoef_Q12:
            coefficients=values[:order];bwexpander(coefficients,63570);values[:order]=coefficients
    if indices.signalType==2:
        name='silk_CB_lags_stage'+('2' if state.fs_kHz==8 else '3')+('_10_ms' if state.nb_subfr==2 else '')
        lags=getattr(T,name)
        for k in range(state.nb_subfr):
            result.pitchL[k]=max(2*state.fs_kHz,min(18*state.fs_kHz,2*state.fs_kHz+indices.lagIndex+lags[k][indices.contourIndex]))
            result.LTPCoef_Q14[k*5:k*5+5]=[value<<7 for value in getattr(T,'silk_LTP_gain_vq_'+str(indices.PERIndex))[indices.LTPIndex[k]]]
        result.LTP_scale_Q14=T.silk_LTPScales_table_Q14[indices.LTP_scaleIndex]
    else:indices.PERIndex=0
    return result
