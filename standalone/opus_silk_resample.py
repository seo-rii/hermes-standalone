"""Delay-compensated Opus SILK resampling with explicit integer state.

Adapted from BSD Opus 1.3.1 resampler*.c, Copyright (c) 2006-2011 Skype Limited.
Original redistribution terms and disclaimer: licenses/opus.txt.
"""
from standalone.silk_fixed import i32, rshift_round, sat16, smlawb, smulwb, smulww
from standalone import opus_silk_tables as T

_RATES=(8000,12000,16000,24000,48000)
_DEC=((4,0,2,0,0),(0,9,4,7,4),(0,3,12,7,7))
_ENC=((6,0,3),(0,7,3),(0,1,10),(0,2,6),(18,10,12))


class Resampler:
    def __init__(self,in_rate,out_rate,*,encoder=False):
        if type(in_rate) is not int or type(out_rate) is not int or in_rate not in _RATES or out_rate not in _RATES:
            raise ValueError('Unsupported Opus SILK resampling rate')
        if (out_rate if encoder else in_rate) not in _RATES[:3]:
            raise ValueError('Unsupported Opus SILK internal sampling rate')
        self.in_rate,self.out_rate=in_rate,out_rate
        self.Fs_in_kHz,self.Fs_out_kHz=in_rate//1000,out_rate//1000
        self.inputDelay=(_ENC if encoder else _DEC)[_RATES.index(in_rate)][_RATES.index(out_rate)]
        self.batchSize=self.Fs_in_kHz*10
        self.sIIR=[0]*6;self.sFIR=[0]*36;self.delayBuf=[0]*48
        self.FIR_Order=self.FIR_Fracs=0;self.Coefs=()
        up=0;self.resampler_function=0
        if out_rate>in_rate:
            if out_rate==in_rate*2:self.resampler_function=1
            else:self.resampler_function=2;up=1
        elif out_rate<in_rate:
            self.resampler_function=3
            for numerator,denominator,fracs,order,name in ((3,4,3,18,'3_4'),(2,3,2,18,'2_3'),(1,2,1,24,'1_2'),(1,3,1,36,'1_3'),(1,4,1,36,'1_4'),(1,6,1,36,'1_6')):
                if out_rate*denominator==in_rate*numerator:
                    self.FIR_Order,self.FIR_Fracs=order,fracs
                    self.Coefs=getattr(T,'silk_Resampler_'+name+'_COEFS');break
            else:raise ValueError('Unsupported Opus SILK resampling ratio')
        self.invRatio_Q16=((in_rate<<(14+up))//out_rate)<<2
        while smulww(self.invRatio_Q16,out_rate)<in_rate<<up:self.invRatio_Q16+=1

    def _up2(self,samples):
        result=[]
        for sample in samples:
            for half,coefficients in enumerate((T.silk_resampler_up2_hq_0,T.silk_resampler_up2_hq_1)):
                value=sample<<10
                for k,coefficient in enumerate(coefficients):
                    index=half*3+k;difference=i32(value-self.sIIR[index])
                    filtered=smlawb(difference,difference,coefficient) if k==2 else smulwb(difference,coefficient)
                    output=i32(self.sIIR[index]+filtered);self.sIIR[index]=i32(value+filtered);value=output
                result.append(sat16(rshift_round(value,10)))
        return result

    def _filter(self,samples):
        result=[];method=self.resampler_function
        if method==0:return list(samples)
        if method==1:return self._up2(samples)
        for offset in range(0,len(samples),self.batchSize):
            block=samples[offset:offset+self.batchSize]
            if method==2:
                filtered=self._up2(block);order=8
                buffer=self.sFIR[:order]+filtered
                for position in range(0,len(filtered)<<16,self.invRatio_Q16):
                    index,phase=position>>16,((position&65535)*12)>>16
                    coefficients=(*T.silk_resampler_frac_FIR_12[phase],*reversed(T.silk_resampler_frac_FIR_12[11-phase]))
                    value=0
                    for k,coefficient in enumerate(coefficients):value=i32(value+buffer[index+k]*coefficient)
                    result.append(sat16(rshift_round(value,15)))
            else:
                order=self.FIR_Order;filtered=[]
                for sample in block:
                    value=i32(self.sIIR[0]+(sample<<8));filtered.append(value);value=i32(value<<2)
                    self.sIIR[0]=smlawb(self.sIIR[1],value,self.Coefs[0]);self.sIIR[1]=smulwb(value,self.Coefs[1])
                buffer=self.sFIR[:order]+filtered;coefficients=self.Coefs[2:];half=order//2
                for position in range(0,len(filtered)<<16,self.invRatio_Q16):
                    index,value=position>>16,0
                    if order==18:
                        phase=((position&65535)*self.FIR_Fracs)>>16;reverse=self.FIR_Fracs-1-phase
                        for k in range(half):value=smlawb(value,buffer[index+k],coefficients[half*phase+k])
                        for k in range(half):value=smlawb(value,buffer[index+order-1-k],coefficients[half*reverse+k])
                    else:
                        for k in range(half):value=smlawb(value,i32(buffer[index+k]+buffer[index+order-1-k]),coefficients[k])
                    result.append(sat16(rshift_round(value,6)))
            self.sFIR[:order]=buffer[len(filtered):len(filtered)+order]
            if method==3 and len(samples)-offset-len(block)<=1:break
        return result

    def process(self,samples):
        if not isinstance(samples,list) or not self.Fs_in_kHz<=len(samples)<=self.in_rate:
            raise ValueError('Opus SILK resampling requires 1 ms to 1 s of PCM16')
        if any(type(value) is not int or not -32768<=value<=32767 for value in samples):
            raise ValueError('Opus SILK resampling requires signed PCM16 integers')
        count=self.Fs_in_kHz-self.inputDelay
        self.delayBuf[self.inputDelay:self.Fs_in_kHz]=samples[:count]
        result=self._filter(self.delayBuf[:self.Fs_in_kHz])
        result+=self._filter(samples[count:len(samples)-self.inputDelay])
        if self.inputDelay:self.delayBuf[:self.inputDelay]=samples[-self.inputDelay:]
        return result
