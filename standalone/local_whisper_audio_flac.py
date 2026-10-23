"""FLAC file decoding with exact signed PCM integers and no native codec.

Format: RFC 9639. Algorithms cross-checked against FFmpeg 8.1.2 FLAC
decoding (Alex Beregszaszi, Justin Ruggles and FFmpeg contributors).
The original LGPL notices and source identities are retained in
licenses/local-whisper-flac-notices.txt and local-whisper-flac-provenance.json.
"""
import hashlib


class FlacError(ValueError):
    pass


_RATES = (0,88200,176400,192000,8000,16000,22050,24000,32000,44100,48000,96000)
_DEPTHS = (0,8,12,0,16,20,24,32)
_MASKS = (0,4,3,7,0x33,0x37,0x3f,0x70f,0x63f)
_FIXED = ((),(1,),(2,-1),(3,-3,1),(4,-6,4,-1))


def _crc_table(width,polynomial):
    table=[]
    for byte in range(256):
        value=byte << (width-8)
        for _ in range(8):
            value=(value<<1) ^ (polynomial if value & (1<<(width-1)) else 0)
        table.append(value & ((1<<width)-1))
    return tuple(table)


_CRC8=_crc_table(8,0x07)
_CRC16=_crc_table(16,0x8005)


def _crc(data,table,width):
    value=0
    mask=(1<<width)-1
    for byte in data:
        value=((value<<8)&mask)^table[(value>>(width-8))^byte]
    return value


class _Bits:
    __slots__=('data','position','end')

    def __init__(self,data,position=0):
        self.data=data
        self.position=position
        self.end=len(data)*8

    def read(self,count):
        end=self.position+count
        if count<0 or end>self.end:
            raise FlacError('Truncated FLAC bitstream')
        if count==0:
            return 0
        start=self.position>>3
        value=int.from_bytes(self.data[start:(end+7)>>3],'big')
        self.position=end
        return (value >> ((-end)&7)) & ((1<<count)-1)

    def signed(self,count):
        value=self.read(count)
        return value-(1<<count) if count and value & (1<<(count-1)) else value

    def unary(self):
        count=0
        while self.position<self.end:
            width=8-(self.position&7)
            value=self.data[self.position>>3] & ((1<<width)-1)
            if value:
                zeros=width-value.bit_length()
                self.position+=zeros+1
                return count+zeros
            count+=width
            self.position+=width
        raise FlacError('Truncated FLAC unary code')


def _coded_number(bits,variable):
    first=bits.read(8)
    if first<128:
        return first
    count=8-first.__xor__(255).bit_length()
    if not 2<=count<=7 or count==7 and not variable:
        raise FlacError('Invalid FLAC coded frame/sample number')
    value=first & ((1<<(7-count))-1)
    for _ in range(count-1):
        byte=bits.read(8)
        if byte & 0xc0 != 0x80:
            raise FlacError('Invalid FLAC coded number continuation')
        value=(value<<6) | (byte&63)
    minimum=1 << (7 if count==2 else 5*count-4)
    if value<minimum or value>=(1<<(36 if variable else 31)):
        raise FlacError('Noncanonical or excessive FLAC coded number')
    return value


def _streaminfo(data):
    if len(data)!=34:
        raise FlacError('FLAC STREAMINFO must contain 34 bytes')
    minimum=int.from_bytes(data[:2],'big')
    maximum=int.from_bytes(data[2:4],'big')
    min_frame=int.from_bytes(data[4:7],'big')
    max_frame=int.from_bytes(data[7:10],'big')
    packed=int.from_bytes(data[10:18],'big')
    rate=packed>>44
    channels=((packed>>41)&7)+1
    depth=((packed>>36)&31)+1
    total=packed & ((1<<36)-1)
    if minimum<16 or maximum<minimum or depth<4:
        raise FlacError('Invalid FLAC STREAMINFO block/depth bounds')
    if min_frame and max_frame and min_frame>max_frame:
        raise FlacError('Invalid FLAC STREAMINFO frame bounds')
    return dict(minimum=minimum,maximum=maximum,min_frame=min_frame,max_frame=max_frame,
                rate=rate,channels=channels,depth=depth,total=total,md5=data[18:34])


def _channel_mask(data,channels):
    position=0
    def item():
        nonlocal position
        if len(data)-position<4:
            raise FlacError('Truncated FLAC Vorbis comment')
        length=int.from_bytes(data[position:position+4],'little')
        position+=4
        if length>len(data)-position:
            raise FlacError('Truncated FLAC Vorbis comment field')
        value=data[position:position+length]
        position+=length
        return value
    item()
    if len(data)-position<4:
        raise FlacError('Missing FLAC Vorbis comment count')
    count=int.from_bytes(data[position:position+4],'little')
    position+=4
    if count>(len(data)-position)//4:
        raise FlacError('Excessive FLAC Vorbis comment count')
    mask=None
    for _ in range(count):
        field=item()
        name,separator,value=field.partition(b'=')
        if separator and name.upper()==b'WAVEFORMATEXTENSIBLE_CHANNEL_MASK':
            try:
                candidate=int(value,16) if value[:2].lower()==b'0x' else -1
            except ValueError:
                candidate=-1
            if 0<candidate<1<<18 and candidate.bit_count()==channels:
                mask=candidate
    return mask


def _metadata(data):
    if not data.startswith(b'fLaC'):
        raise FlacError('Missing FLAC signature')
    position=4
    info=mask=None
    while True:
        if len(data)-position<4:
            raise FlacError('Truncated FLAC metadata header')
        control=data[position]
        kind=control & 127
        length=int.from_bytes(data[position+1:position+4],'big')
        position+=4
        if length>len(data)-position:
            raise FlacError('Truncated FLAC metadata block')
        payload=data[position:position+length]
        position+=length
        if info is None:
            if kind!=0:
                raise FlacError('FLAC STREAMINFO must be the first metadata block')
            info=_streaminfo(payload)
        elif kind==0 or kind==127:
            raise FlacError('Duplicate or forbidden FLAC metadata block')
        elif kind==4:
            candidate=_channel_mask(payload,info['channels'])
            if candidate is not None:
                mask=candidate
        if control & 128:
            return position,info,mask or _MASKS[info['channels']]


def _header(bits,info):
    start=bits.position>>3
    if bits.read(15)!=0x7ffc:
        raise FlacError('Invalid FLAC frame synchronization/reserved bit')
    variable=bits.read(1)
    block_code=bits.read(4)
    rate_code=bits.read(4)
    assignment=bits.read(4)
    depth_code=bits.read(3)
    if bits.read(1) or not block_code or rate_code==15 or assignment>10 or depth_code==3:
        raise FlacError('Invalid FLAC frame header field')
    _coded_number(bits,variable)
    if block_code==1:
        size=192
    elif block_code<6:
        size=144<<block_code
    elif block_code in (6,7):
        size=bits.read(8 if block_code==6 else 16)+1
    else:
        size=1<<block_code
    if rate_code<12:
        rate=_RATES[rate_code] or info['rate']
    elif rate_code==12:
        rate=bits.read(8)*1000
    else:
        rate=bits.read(16)*(10 if rate_code==14 else 1)
    depth=_DEPTHS[depth_code] or info['depth']
    channels=assignment+1 if assignment<8 else 2
    bits.read(8)
    if _crc(memoryview(bits.data)[start:bits.position>>3],_CRC8,8):
        raise FlacError('FLAC frame header CRC mismatch')
    if size>65535 or size>info['maximum'] or channels!=info['channels'] or depth!=info['depth']:
        raise FlacError('FLAC frame disagrees with STREAMINFO')
    if rate<=0 or info['rate'] and rate!=info['rate']:
        raise FlacError('Invalid or changing FLAC sampling rate')
    return size,assignment,rate,depth


def _residual(bits,size,order):
    method=bits.read(2)
    partition_order=bits.read(4)
    partitions=1<<partition_order
    per_partition=size>>partition_order
    if method>1 or per_partition*partitions!=size or per_partition<=order:
        raise FlacError('Invalid FLAC residual method or partition order')
    parameter_bits=4+method
    escape=(1<<parameter_bits)-1
    result=[]
    for partition in range(partitions):
        count=per_partition-(order if partition==0 else 0)
        parameter=bits.read(parameter_bits)
        if parameter==escape:
            width=bits.read(5)
            result.extend(bits.signed(width) for _ in range(count))
        else:
            for _ in range(count):
                folded=(bits.unary()<<parameter) | bits.read(parameter)
                value=(folded>>1) ^ -(folded&1)
                if value<=-(1<<31) or value>=(1<<31):
                    raise FlacError('FLAC residual exceeds its signed integer range')
                result.append(value)
    return result


def _subframe(bits,size,depth):
    if bits.read(1):
        raise FlacError('Nonzero FLAC subframe padding bit')
    kind=bits.read(6)
    wasted=bits.unary()+1 if bits.read(1) else 0
    depth-=wasted
    if depth<1:
        raise FlacError('FLAC wasted bits exhaust the subframe bit depth')
    if kind==0:
        values=[bits.signed(depth)]*size
    elif kind==1:
        values=[bits.signed(depth) for _ in range(size)]
    elif 8<=kind<=12 or kind>=32:
        order=kind-8 if kind<32 else kind-31
        if order>=size:
            raise FlacError('FLAC predictor order exceeds its block size')
        values=[bits.signed(depth) for _ in range(order)]
        shift=0
        if kind>=32:
            precision=bits.read(4)+1
            shift=bits.signed(5)
            if precision==16 or shift<0:
                raise FlacError('Forbidden FLAC LPC precision or negative shift')
            coefficients=[bits.signed(precision) for _ in range(order)]
        else:
            coefficients=_FIXED[order]
        for residual in _residual(bits,size,order):
            prediction=sum(coefficient*values[-i-1] for i,coefficient in enumerate(coefficients))
            values.append(residual+(prediction>>shift))
    else:
        raise FlacError('Reserved FLAC subframe type')
    return [value<<wasted for value in values] if wasted else values


def decode_flac(data,*,max_frames=None):
    """Return ``(sample_rate, planar_signed_pcm, bits_per_sample, mask)``.

    PCM values retain their original bit depth, before FFmpeg's S16/S32
    output alignment. ``max_frames`` bounds decoded samples per channel.
    Header CRC8, frame CRC16, known sample counts and nonzero MD5 are checked.
    """
    if max_frames is not None and (type(max_frames) is not int or max_frames<0):
        raise ValueError('FLAC frame budget must be a nonnegative integer')
    if not isinstance(data,(bytes,bytearray,memoryview)):
        raise TypeError('FLAC input must be bytes')
    data=bytes(data)
    position,info,mask=_metadata(data)
    if max_frames is not None and info['total']>max_frames:
        raise FlacError('FLAC decoded frame budget exceeded')
    output=[[] for _ in range(info['channels'])]
    bits=_Bits(data,position*8)
    digest=hashlib.md5(usedforsecurity=False)
    count=0
    previous_size=None
    rate=info['rate']
    while bits.position<bits.end:
        start=bits.position>>3
        size,assignment,rate,depth=_header(bits,info)
        if max_frames is not None and count+size>max_frames:
            raise FlacError('FLAC decoded frame budget exceeded')
        if previous_size is not None and previous_size<info['minimum']:
            raise FlacError('Short FLAC block appears before the final frame')
        decoded=[]
        for channel in range(info['channels']):
            side=(assignment==9 and channel==0) or (assignment in (8,10) and channel==1)
            decoded.append(_subframe(bits,size,depth+side))
        if bits.read((-bits.position)&7):
            raise FlacError('Nonzero FLAC frame byte-alignment padding')
        bits.read(16)
        end=bits.position>>3
        if _crc(memoryview(data)[start:end],_CRC16,16):
            raise FlacError('FLAC frame CRC mismatch')
        if info['max_frame'] and end-start>info['max_frame']:
            raise FlacError('FLAC frame exceeds its declared byte bound')
        if assignment==8:
            decoded[1]=[left-side for left,side in zip(*decoded)]
        elif assignment==9:
            decoded[0]=[side+right for side,right in zip(*decoded)]
        elif assignment==10:
            mid,side=decoded
            decoded=[[middle+((difference+1)>>1) for middle,difference in zip(mid,side)],
                     [middle-(difference>>1) for middle,difference in zip(mid,side)]]
        packed=bytearray()
        bound=1<<(depth-1)
        for sample in zip(*decoded):
            for value in sample:
                if not -bound<=value<bound:
                    raise FlacError('Decoded FLAC sample exceeds its bit depth')
                packed.extend(value.to_bytes((depth+7)//8,'little',signed=True))
        digest.update(packed)
        for destination,channel in zip(output,decoded):
            destination.extend(channel)
        count+=size
        previous_size=size
        if info['total'] and count>info['total']:
            raise FlacError('FLAC frame samples exceed the declared total')
        if not info['rate']:
            info['rate']=rate
    if info['total'] and count!=info['total']:
        raise FlacError('Truncated FLAC sample count')
    if any(info['md5']) and digest.digest()!=info['md5']:
        raise FlacError('FLAC decoded PCM MD5 mismatch')
    return rate,output,info['depth'],mask
