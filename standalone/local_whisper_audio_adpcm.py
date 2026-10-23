"""WAVE IMA2/3/4/5-bit and Microsoft ADPCM from FFmpeg8.1.2 (LGPL2.1+)."""
import struct

_STEPS=(7,8,9,10,11,12,13,14,16,17,19,21,23,25,28,31,34,37,41,45,50,55,
        60,66,73,80,88,97,107,118,130,143,157,173,190,209,230,253,279,307,
        337,371,408,449,494,544,598,658,724,796,876,963,1060,1166,1282,1411,
        1552,1707,1878,2066,2272,2499,2749,3024,3327,3660,4026,4428,4871,
        5358,5894,6484,7132,7845,8630,9493,10442,11487,12635,13899,15289,
        16818,18500,20350,22385,24623,27086,29794,32767)
_INDEX={2:(-1,2)*2,3:(-1,-1,1,2)*2,
        4:(-1,-1,-1,-1,2,4,6,8)*2,
        5:(-1,)*8+(1,2,4,6,8,10,13,16)+(-1,)*8+(1,2,4,6,8,10,13,16)}
_BLOCK_BYTES={2:4,3:12,4:4,5:20}
_BLOCK_SAMPLES={2:16,3:32,4:8,5:32}
_ADAPT=(230,230,230,230,307,409,512,614,768,614,512,409,307,230,230,230)
_COEFF1=(64,128,0,48,60,115,98)
_COEFF2=(0,-64,0,16,0,-52,-58)


def _ima_sample(state,nibble,bits):
    predictor,index=state
    step=_STEPS[index]
    shift=bits-1
    difference=step>>shift
    for i in range(shift):
        if nibble&(1<<i):
            difference+=step>>(shift-1-i)
    predictor+=-difference if nibble&(1<<shift)else difference
    predictor=max(-32768,min(32767,predictor))
    state[:]=predictor,max(0,min(88,index+_INDEX[bits][nibble]))
    return predictor


def _ima_block(data,channels,bits):
    if len(data)<4*channels:
        raise ValueError('Truncated IMA WAVE block')
    states=[];output=[]
    for channel in range(channels):
        predictor,index=struct.unpack_from('<hB',data,channel*4)
        if index>88:
            raise ValueError('Invalid IMA WAVE step index')
        states.append([predictor,index]);output.append([predictor])
    block_size=_BLOCK_BYTES[bits]
    groups=(len(data)-4*channels)//(block_size*channels)
    for group in range(groups):
        for channel in range(channels):
            base=4*channels+block_size*group*channels+channel*4
            encoded=bytes(data[base+(j%4)+(j//4)*(channels*4)]for j in range(block_size))
            word=int.from_bytes(encoded,'little')
            for _ in range(_BLOCK_SAMPLES[bits]):
                nibble=word&((1<<bits)-1);word>>=bits
                output[channel].append(_ima_sample(states[channel],nibble,bits))
    return output


def _ms_sample(state,nibble):
    coefficient,delta,previous,older=state
    numerator=previous*_COEFF1[coefficient]+older*_COEFF2[coefficient]
    prediction=abs(numerator)//64*(1 if numerator>=0 else-1)
    prediction+=(nibble if nibble<8 else nibble-16)*delta
    sample=max(-32768,min(32767,prediction))
    delta=max(16,min(2147483647//768,(_ADAPT[nibble]*delta)>>8))
    state[:]=coefficient,delta,sample,previous
    return sample


def _ms_mono_blocks(data,channels):
    output=[]
    samples=(len(data)-6*channels)*2//channels
    if samples<2:
        raise ValueError('Truncated Microsoft ADPCM block')
    size=7+(samples-2)//2
    for channel in range(channels):
        offset=channel*size
        coefficient=data[offset]
        if coefficient>6:
            raise ValueError('Invalid Microsoft ADPCM predictor')
        delta,previous,older=struct.unpack_from('<hhh',data,offset+1)
        state=[coefficient,delta,previous,older];pcm=[older,previous]
        for value in data[offset+7:offset+size]:
            pcm.append(_ms_sample(state,value>>4));pcm.append(_ms_sample(state,value&15))
        output.append(pcm)
    return output


def _ms_block(data,channels):
    if channels>2:
        return _ms_mono_blocks(data,channels)
    if len(data)<7*channels:
        raise ValueError('Truncated Microsoft ADPCM block')
    coefficients=data[:channels]
    if any(value>6 for value in coefficients):
        raise ValueError('Invalid Microsoft ADPCM predictor')
    values=struct.unpack_from('<'+'h'*(3*channels),data,channels)
    states=[[coefficients[ch],values[ch],values[channels+ch],values[2*channels+ch]]for ch in range(channels)]
    output=[[state[3],state[2]]for state in states]
    for value in data[7*channels:]:
        output[0].append(_ms_sample(states[0],value>>4))
        channel=channels-1
        output[channel].append(_ms_sample(states[channel],value&15))
    return output


def decode_adpcm(data,kind,channels,bits,block_align,*,max_frames=None):
    """Decode valid WAVE blocks; damaged compressed block headers fail explicitly."""
    if kind not in (2,17) or type(block_align)is not int or block_align<1:
        raise ValueError('Invalid WAVE ADPCM format')
    if not 1<=channels<=(6 if kind==2 else 2) or (bits!=4 if kind==2 else bits not in _INDEX):
        raise ValueError('Unsupported WAVE ADPCM channel or bit depth')
    if max_frames is not None and(type(max_frames)is not int or max_frames<0):
        raise ValueError('Invalid ADPCM decoded frame budget')
    output=[[]for _ in range(channels)]
    for offset in range(0,len(data),block_align):
        block=data[offset:offset+block_align]
        decoded=_ms_block(block,channels)if kind==2 else _ima_block(block,channels,bits)
        if max_frames is not None and len(output[0])+len(decoded[0])>max_frames:
            raise ValueError('Whisper waveform exceeds the frontend one-hour bound')
        for dest,source in zip(output,decoded):dest.extend(source)
    return output
