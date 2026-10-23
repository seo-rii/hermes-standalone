"""Opus SILK NLSF prediction, dequantization and stable LPC conversion.

Adapted from BSD Opus 1.3.1, Copyright (c) 2006-2011 Skype Limited.
The complete notice and original disclaimer are in licenses/opus.txt.
"""
from standalone.silk_fixed import (cdiv,clz32,i16,i32,inverse32_varq,rshift_round,
                                   sat16,sat32,smlawb,smmul,smulww)
from standalone.opus_silk_tables import silk_LSFCosTab_FIX_Q12
from standalone import opus_silk_tables as T


def nlsf_stabilize(values,deltas):
    order=len(values)
    for _ in range(20):
        differences=[values[0]-deltas[0]]+[values[k]-values[k-1]-deltas[k] for k in range(1,order)]+[32768-values[-1]-deltas[-1]]
        index=min(range(order+1),key=differences.__getitem__)
        if differences[index]>=0:return values
        if index==0:values[0]=deltas[0]
        elif index==order:values[-1]=32768-deltas[-1]
        else:
            half=deltas[index]>>1
            minimum=sum(deltas[:index])+half
            maximum=32768-sum(deltas[index+1:])-half
            center=max(minimum,min(maximum,rshift_round(values[index-1]+values[index],1)))
            values[index-1]=center-half;values[index]=values[index-1]+deltas[index]
    values[:]=sorted(values)
    values[0]=max(values[0],deltas[0])
    for k in range(1,order):values[k]=max(values[k],sat16(values[k-1]+deltas[k]))
    values[-1]=min(values[-1],32768-deltas[-1])
    for k in range(order-2,-1,-1):values[k]=min(values[k],values[k+1]-deltas[k+1])
    return values


def bwexpander32(values,chirp):
    difference=i32(chirp-65536)
    for index in range(len(values)-1):
        values[index]=smulww(chirp,values[index])
        chirp=i32(chirp+rshift_round(i32(chirp*difference),16))
    values[-1]=smulww(chirp,values[-1])
    return values


def inverse_gain(coefficients):
    if not 1<=len(coefficients)<=16 or any(type(v) is not int or not -32768<=v<=32767 for v in coefficients):
        raise ValueError('Invalid Opus LPC coefficients')
    if sum(coefficients)>=4096:return 0
    current=[i32(value<<12) for value in coefficients]
    gain=1<<30
    for k in range(len(current)-1,-1,-1):
        if not -16773022<=current[k]<=16773022:return 0
        reflection=i32(-(current[k]<<7))
        multiplier=i32((1<<30)-smmul(reflection,reflection))
        gain=i32(smmul(gain,multiplier)<<2)
        if gain<107374:return 0
        if not k:break
        precision=32-clz32(abs(multiplier))
        reciprocal=inverse32_varq(multiplier,precision+30)
        for index in range((k+1)>>1):
            first,second=current[index],current[k-index-1]
            left=rshift_round(sat32(first-i32(rshift_round(second*reflection,31)))*reciprocal,precision)
            right=rshift_round(sat32(second-i32(rshift_round(first*reflection,31)))*reciprocal,precision)
            if not -(1<<31)<=left<(1<<31) or not -(1<<31)<=right<(1<<31):return 0
            current[index],current[k-index-1]=left,right
    return gain


def fit(values,output_q=12,input_q=17):
    shift=input_q-output_q
    if not 0<shift<31 or not 1<=len(values)<=16:raise ValueError('Invalid Opus LPC fit dimensions')
    for _ in range(10):
        maximum,index=0,0
        for k,value in enumerate(values):
            if abs(value)>maximum:maximum,index=abs(value),k
        maximum=rshift_round(maximum,shift)
        if maximum<=32767:break
        maximum=min(maximum,163838)
        chirp=65470-cdiv((maximum-32767)<<14,i32(maximum*(index+1))>>2)
        bwexpander32(values,chirp)
    else:
        result=[sat16(rshift_round(value,shift)) for value in values]
        values[:]=[i32(value<<shift) for value in result]
        return result
    return [i16(rshift_round(value,shift)) for value in values]


def _polynomial(cosines):
    result=[65536,-cosines[0]]+[0]*(len(cosines)-1)
    for k in range(1,len(cosines)):
        cosine=cosines[k]
        result[k+1]=i32((result[k-1]<<1)-i32(rshift_round(cosine*result[k],16)))
        for n in range(k,1,-1):
            result[n]=i32(result[n]+result[n-2]-i32(rshift_round(cosine*result[n-1],16)))
        result[1]=i32(result[1]-cosine)
    return result


def nlsf2a(values):
    order=len(values)
    if order not in (10,16) or any(type(v) is not int or not 0<=v<32768 for v in values):
        raise ValueError('Invalid Opus NLSF vector')
    ordering=(0,15,8,7,4,11,12,3,2,13,10,5,6,9,14,1) if order==16 else (0,9,6,3,4,5,8,1,2,7)
    cosines=[0]*order
    for k,value in enumerate(values):
        integer,fraction=value>>8,value&255
        start=silk_LSFCosTab_FIX_Q12[integer];delta=silk_LSFCosTab_FIX_Q12[integer+1]-start
        cosines[ordering[k]]=rshift_round((start<<8)+delta*fraction,4)
    even,odd=_polynomial(cosines[::2]),_polynomial(cosines[1::2])
    expanded=[0]*order
    for k in range(order//2):
        first,second=i32(even[k+1]+even[k]),i32(odd[k+1]-odd[k])
        expanded[k],expanded[order-k-1]=i32(-second-first),i32(second-first)
    result=fit(expanded)
    for index in range(16):
        if inverse_gain(result):break
        bwexpander32(expanded,65536-(2<<index))
        result=[i16(rshift_round(value,5)) for value in expanded]
    return result


def unpack(order,index):
    if order not in (10,16) or type(index) is not int or not 0<=index<32:raise ValueError('Invalid Opus NLSF codebook index')
    band='WB' if order==16 else 'NB_MB'
    selectors=getattr(T,'silk_NLSF_CB2_SELECT_'+band)
    predictors=getattr(T,'silk_NLSF_PRED_'+band+'_Q8')
    offsets,prediction=[],[]
    for k in range(0,order,2):
        entry=selectors[index*order//2+k//2]
        offsets.extend((((entry>>1)&7)*9,((entry>>5)&7)*9))
        prediction.extend((predictors[k+(entry&1)*(order-1)],predictors[k+((entry>>4)&1)*(order-1)+1]))
    return offsets,prediction


def decode_nlsf(indices):
    order=len(indices)-1
    offsets,prediction=unpack(order,indices[0])
    if any(type(value) is not int or not -14<=value<=14 for value in indices[1:]):raise ValueError('Invalid Opus NLSF residual index')
    previous=0;residual=[0]*order
    for k in range(order-1,-1,-1):
        predicted=i32(previous*prediction[k])>>8
        previous=indices[k+1]<<10
        previous=previous-102 if previous>0 else previous+102 if previous<0 else previous
        previous=smlawb(predicted,previous,9830 if order==16 else 11796)
        residual[k]=i16(previous)
    band='WB' if order==16 else 'NB_MB'
    vectors=getattr(T,'silk_NLSF_CB1_'+band+'_Q8')
    weights=T.silk_NLSF_CB1_WB_Wght_Q9 if order==16 else T.silk_NLSF_CB1_Wght_Q9
    start=indices[0]*order
    result=[max(0,min(32767,cdiv(residual[k]<<14,weights[start+k])+(vectors[start+k]<<7))) for k in range(order)]
    nlsf_stabilize(result,getattr(T,'silk_NLSF_DELTA_MIN_'+band+'_Q15'))
    return result
