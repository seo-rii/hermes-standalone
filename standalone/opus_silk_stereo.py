"""Original Opus SILK adaptive stereo predictors and delayed reconstruction.

Adapted from BSD Opus 1.3.1 stereo_decode_pred.c and stereo_MS_to_LR.c.
Copyright (c) 2006-2011 Skype Limited; complete notice: licenses/opus.txt.
"""
from types import SimpleNamespace
from standalone.silk_fixed import i16, i32, rshift_round, sat16, smlawb, smulwb
from standalone import opus_silk_tables as T


def initialize():
    return SimpleNamespace(pred_prev_Q13=[0,0],sMid=[0,0],sSide=[0,0])


def decode_pred(decoder):
    index=decoder.icdf(T.silk_stereo_pred_joint_iCDF)
    selectors=(index//5,index%5);result=[]
    for selected in selectors:
        coarse=decoder.icdf(T.silk_uniform3_iCDF)+3*selected
        fine=decoder.icdf(T.silk_uniform5_iCDF)
        low=T.silk_stereo_pred_quant_Q13[coarse]
        step=smulwb(T.silk_stereo_pred_quant_Q13[coarse+1]-low,6554)
        result.append(i32(low+i16(step)*i16(2*fine+1)))
    result[0]=i32(result[0]-result[1])
    return result


def mid_only(decoder):
    return decoder.icdf(T.silk_stereo_only_code_mid_iCDF)


def ms_to_lr(state,mid,side,prediction,frequency,length):
    if frequency not in (8,12,16) or length not in (frequency*10,frequency*20) or len(mid)!=length+2 or len(side)!=length+2 or len(prediction)!=2:
        raise ValueError('Invalid Opus SILK stereo dimensions')
    mid[:2]=state.sMid;side[:2]=state.sSide
    state.sMid[:]=mid[length:length+2];state.sSide[:]=side[length:length+2]
    first,second=state.pred_prev_Q13;interpolation=frequency*8;denominator=65536//interpolation
    delta0=rshift_round(i32(i16(prediction[0]-first)*i16(denominator)),16)
    delta1=rshift_round(i32(i16(prediction[1]-second)*i16(denominator)),16)
    for k in range(length):
        if k<interpolation:first=i32(first+delta0);second=i32(second+delta1)
        else:first,second=prediction
        value=i32((mid[k]+mid[k+2]+(mid[k+1]<<1))<<9)
        value=smlawb(side[k+1]<<8,value,first)
        value=smlawb(value,mid[k+1]<<11,second)
        side[k+1]=sat16(rshift_round(value,8))
    state.pred_prev_Q13[:]=prediction
    for k in range(1,length+1):
        left,right=mid[k]+side[k],mid[k]-side[k]
        mid[k],side[k]=sat16(left),sat16(right)
    return mid,side
