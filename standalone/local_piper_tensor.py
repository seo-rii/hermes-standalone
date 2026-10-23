"""Owned dense tensors for the reviewed Piper VITS ONNX graph."""
from array import array
from dataclasses import dataclass
from itertools import product
import math
import operator
import struct

_FLOAT = struct.Struct('<f')
dot = getattr(math, 'sumprod', lambda a, b: sum(x*y for x,y in zip(a,b,strict=True)))


def f32(value):
    try:
        return _FLOAT.unpack(_FLOAT.pack(value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def strides(shape):
    result = [1]*len(shape)
    for i in range(len(shape)-2,-1,-1):result[i] = result[i+1]*shape[i+1]
    return tuple(result)


def coordinates(shape):
    return product(*(range(n) for n in shape))


@dataclass(frozen=True)
class Tensor:
    shape: tuple
    values: object
    dtype: int = 1

    def __post_init__(self):
        if any(not isinstance(v,int) or v<0 for v in self.shape) or math.prod(self.shape)!=len(self.values):
            raise ValueError(('Invalid tensor dimensions',self.shape,len(self.values)))

    @classmethod
    def make(cls, shape, values, dtype=1):
        shape=tuple(shape)
        if dtype==1:values=array('f',values)
        elif dtype in (6,7):
            width=32 if dtype==6 else 64;sign=1<<(width-1);mod=1<<width
            values=tuple((int(v)+sign)%mod-sign for v in values)
        elif dtype==9:values=tuple(bool(v) for v in values)
        else:raise ValueError(('Unsupported Piper tensor type',dtype))
        return cls(shape,values,dtype)

    @property
    def scalar(self):
        if len(self.values)!=1:raise ValueError('Tensor must contain one scalar')
        return self.values[0]

    def reshape(self, shape, allowzero=False):
        shape=[self.shape[i] if n==0 and not allowzero else n for i,n in enumerate(shape)]
        if shape.count(-1)>1:raise ValueError('Multiple inferred tensor dimensions')
        if -1 in shape:
            index=shape.index(-1);known=math.prod(n for n in shape if n!=-1)
            if not known or len(self.values)%known:raise ValueError('Invalid inferred tensor dimension')
            shape[index]=len(self.values)//known
        return Tensor(tuple(shape),self.values,self.dtype)


def broadcast_shape(*shapes):
    count=max(map(len,shapes),default=0);result=[1]*count
    for shape in shapes:
        for i,n in enumerate(shape,count-len(shape)):
            if result[i]==1:result[i]=n
            elif n not in (1,result[i]):raise ValueError(('Incompatible tensor broadcast',shapes))
    return tuple(result)


def broadcast_strides(shape, output):
    source=strides(shape);offset=len(output)-len(shape)
    return (0,)*offset+tuple(step if size!=1 else 0 for size,step in zip(shape,source))


def binary(a,b,operation,dtype=None):
    dtype=a.dtype if dtype is None else dtype
    shape=broadcast_shape(a.shape,b.shape)
    if a.shape==b.shape:
        values=(operation(x,y)for x,y in zip(a.values,b.values,strict=True))
    elif len(a.values)==1:
        values=(operation(a.scalar,y)for y in b.values)
    elif len(b.values)==1:
        values=(operation(x,b.scalar)for x in a.values)
    else:
        sa,sb=broadcast_strides(a.shape,shape),broadcast_strides(b.shape,shape)
        values=(operation(a.values[sum(c*s for c,s in zip(coord,sa))],b.values[sum(c*s for c,s in zip(coord,sb))])for coord in coordinates(shape))
    return Tensor.make(shape,values,dtype)


def unary(a,operation,dtype=None):
    return Tensor.make(a.shape,(operation(v)for v in a.values),a.dtype if dtype is None else dtype)


def transpose(a, permutation=None):
    permutation=tuple(reversed(range(len(a.shape))))if permutation is None else tuple(permutation)
    if sorted(permutation)!=list(range(len(a.shape))):raise ValueError('Invalid tensor permutation')
    shape=tuple(a.shape[i]for i in permutation);source=strides(a.shape)
    steps=tuple(source[i]for i in permutation)
    return Tensor.make(shape,(a.values[sum(c*s for c,s in zip(coord,steps))]for coord in coordinates(shape)),a.dtype)


def gather(a,indices,axis=0):
    axis%=len(a.shape);outer=math.prod(a.shape[:axis]);width=math.prod(a.shape[axis+1:]);size=a.shape[axis]
    output=[]
    for i in range(outer):
        for value in indices.values:
            value=value+size if value<0 else value
            if not 0<=value<size:raise ValueError('Tensor Gather index out of range')
            start=(i*size+value)*width;output.extend(a.values[start:start+width])
    return Tensor.make(a.shape[:axis]+indices.shape+a.shape[axis+1:],output,a.dtype)


def gather_nd(a,indices):
    rank=indices.shape[-1];width=math.prod(a.shape[rank:]);steps=strides(a.shape)
    output=[]
    for i in range(0,len(indices.values),rank):
        offset=0
        for axis,value in enumerate(indices.values[i:i+rank]):
            if value<0:value+=a.shape[axis]
            if not 0<=value<a.shape[axis]:raise ValueError('GatherND index out of range')
            offset+=value*steps[axis]
        output.extend(a.values[offset:offset+width])
    return Tensor.make(indices.shape[:-1]+a.shape[rank:],output,a.dtype)


def gather_elements(a,indices,axis=0):
    axis%=len(a.shape);steps=strides(a.shape);output=[]
    for coord,value in zip(coordinates(indices.shape),indices.values):
        if value<0:value+=a.shape[axis]
        if not 0<=value<a.shape[axis]:raise ValueError('GatherElements index out of range')
        output.append(a.values[sum((value if i==axis else c)*s for i,(c,s)in enumerate(zip(coord,steps)))])
    return Tensor.make(indices.shape,output,a.dtype)


def scatter_nd(a,indices,updates):
    rank=indices.shape[-1];width=math.prod(a.shape[rank:]);steps=strides(a.shape);output=list(a.values)
    if len(updates.values)!=(len(indices.values)//rank)*width:raise ValueError('ScatterND update dimensions')
    for i in range(0,len(indices.values),rank):
        offset=0
        for axis,value in enumerate(indices.values[i:i+rank]):
            if value<0:value+=a.shape[axis]
            if not 0<=value<a.shape[axis]:raise ValueError('ScatterND index out of range')
            offset+=value*steps[axis]
        start=(i//rank)*width;output[offset:offset+width]=updates.values[start:start+width]
    return Tensor.make(a.shape,output,a.dtype)


def concat(inputs,axis):
    rank=len(inputs[0].shape);axis%=rank;shape=list(inputs[0].shape);shape[axis]=sum(a.shape[axis]for a in inputs)
    outer=math.prod(shape[:axis]);width=math.prod(shape[axis+1:]);output=[]
    for index in range(outer):
        for a in inputs:
            count=a.shape[axis]*width;output.extend(a.values[index*count:(index+1)*count])
    return Tensor.make(shape,output,inputs[0].dtype)


def slice_tensor(a,starts,ends,axes=None,steps=None):
    axes=list(range(len(starts)))if axes is None else axes;steps=[1]*len(starts)if steps is None else steps
    ranges=[range(n)for n in a.shape]
    for start,end,axis,step in zip(starts,ends,axes,steps,strict=True):
        axis%=len(a.shape)
        if not step:raise ValueError('Zero tensor slice step')
        ranges[axis]=range(*slice(start,end,step).indices(a.shape[axis]))
    shape=tuple(len(r)for r in ranges);source=strides(a.shape)
    return Tensor.make(shape,(a.values[sum(c*s for c,s in zip(coord,source))]for coord in product(*ranges)),a.dtype)


def pad(a,pads,value=0):
    rank=len(a.shape)
    if len(pads)!=rank*2:raise ValueError('Piper Pad dimensions')
    starts=[max(0,-p)for p in pads[:rank]];ends=[n+min(0,p)for n,p in zip(a.shape,pads[rank:])]
    if any(starts)or any(p<0 for p in pads[rank:]):a=slice_tensor(a,starts,ends)
    pads=[max(0,p)for p in pads];shape=tuple(n+p+q for n,p,q in zip(a.shape,pads[:rank],pads[rank:]));source=strides(a.shape)
    output=[]
    for coord in coordinates(shape):
        old=tuple(c-p for c,p in zip(coord,pads))
        output.append(a.values[sum(c*s for c,s in zip(old,source))]if all(0<=c<n for c,n in zip(old,a.shape))else value)
    return Tensor.make(shape,output,a.dtype)


def reduce(a,axes=None,keepdims=True,kind='sum'):
    rank=len(a.shape);axes=set(range(rank))if axes is None else {axis%rank for axis in axes}
    retained=tuple(i for i in range(rank)if i not in axes);reduced=tuple(i for i in range(rank)if i in axes)
    outer=tuple(a.shape[i]for i in retained);inner=tuple(a.shape[i]for i in reduced);source=strides(a.shape)
    offsets=tuple(sum(c*source[i]for c,i in zip(coord,reduced))for coord in coordinates(inner));output=[]
    for coord in coordinates(outer):
        base=sum(c*source[i]for c,i in zip(coord,retained));values=[a.values[base+offset]for offset in offsets]
        value=max(values)if kind=='max'else sum(values)
        if kind=='mean':value/=len(values)
        output.append(value)
    shape=tuple(1 if i in axes else n for i,n in enumerate(a.shape))if keepdims else outer
    return Tensor.make(shape,output,a.dtype)


def softmax(a,axis=-1):
    axis%=len(a.shape);size=a.shape[axis];width=math.prod(a.shape[axis+1:]);outer=math.prod(a.shape[:axis]);output=[0.]*len(a.values)
    for i in range(outer):
        for j in range(width):
            indices=[(i*size+k)*width+j for k in range(size)];peak=max(a.values[k]for k in indices)
            values=[f32(math.exp(f32(a.values[k]-peak)))for k in indices];total=f32(sum(values))
            for k,v in zip(indices,values):output[k]=f32(v/total)
    return Tensor.make(a.shape,output,a.dtype)


def cumsum(a,axis,exclusive=False,reverse=False):
    axis%=len(a.shape);size=a.shape[axis];width=math.prod(a.shape[axis+1:]);outer=math.prod(a.shape[:axis]);output=[0]*len(a.values)
    for i in range(outer):
        for j in range(width):
            value=0
            for k in (reversed(range(size))if reverse else range(size)):
                index=(i*size+k)*width+j;new=value+a.values[index]
                if a.dtype==1:new=f32(new)
                output[index]=value if exclusive else new;value=new
    return Tensor.make(a.shape,output,a.dtype)


def where(condition,a,b):
    shape=broadcast_shape(condition.shape,a.shape,b.shape);steps=[broadcast_strides(v.shape,shape)for v in (condition,a,b)];out=[]
    for coord in coordinates(shape):
        ci,ai,bi=(sum(c*s for c,s in zip(coord,ss))for ss in steps)
        out.append(a.values[ai]if condition.values[ci]else b.values[bi])
    return Tensor.make(shape,out,a.dtype)


def matmul(a,b):
    av,bv=len(a.shape)==1,len(b.shape)==1
    ashape=(1,)+a.shape if av else a.shape;bshape=b.shape+(1,)if bv else b.shape
    if ashape[-1]!=bshape[-2]:raise ValueError('Piper MatMul dimensions')
    batch=broadcast_shape(ashape[:-2],bshape[:-2]);m,k,n=ashape[-2],ashape[-1],bshape[-1]
    astep=tuple(s*m*k for s in broadcast_strides(ashape[:-2],batch));bstep=tuple(s*k*n for s in broadcast_strides(bshape[:-2],batch))
    out=[]
    for coord in coordinates(batch):
        ai=sum(c*s for c,s in zip(coord,astep));bi=sum(c*s for c,s in zip(coord,bstep))
        columns=tuple(tuple(b.values[bi+j+z*n]for z in range(k))for j in range(n))
        for i in range(m):
            row=a.values[ai+i*k:ai+(i+1)*k]
            out.extend(f32(dot(row,column))for column in columns)
    shape=batch+(m,n)
    if av:shape=shape[:-2]+shape[-1:]
    if bv:shape=shape[:-1]
    return Tensor.make(shape,out,a.dtype)


def conv(a,weights,bias=None,*,pads=(0,0),strides=(1,),dilations=(1,),group=1):
    if len(a.shape)!=3 or len(weights.shape)!=3:raise ValueError('Piper convolution requires rank3')
    batch,channels,length=a.shape;count,pergroup,kernel=weights.shape
    if channels!=pergroup*group or count%group:raise ValueError('Piper convolution group dimensions')
    dilation,stride=dilations[0],strides[0];outlen=(length+sum(pads)-dilation*(kernel-1)-1)//stride+1
    if outlen<0:raise ValueError('Piper convolution negative output length')
    kernels=tuple(weights.values[i*pergroup*kernel:(i+1)*pergroup*kernel]for i in range(count));out=[]
    for sample in range(batch):
        for g in range(group):
            inputs=tuple(tuple(a.values[(sample*channels+g*pergroup+c)*length+index]if 0<=index<length else 0.
                               for c in range(pergroup)for index in (position*stride-pads[0]+k*dilation for k in range(kernel)))
                         for position in range(outlen))
            for i in range(count//group):
                index=g*(count//group)+i;offset=0 if bias is None else bias.values[index]
                out.extend(f32(dot(source,kernels[index])+offset)for source in inputs)
    return Tensor.make((batch,count,outlen),out)


def conv_transpose(a,weights,bias=None,*,pads=(0,0),strides=(1,),dilations=(1,),group=1,output_padding=(0,)):
    batch,channels,length=a.shape;cin,pergroup,kernel=weights.shape
    if cin!=channels or channels%group:raise ValueError('Piper transpose convolution groups')
    dilation,stride=dilations[0],strides[0];count=pergroup*group
    outlen=(length-1)*stride+output_padding[0]+dilation*(kernel-1)+1-sum(pads)
    output=[];cpg=channels//group
    for sample in range(batch):
        for index in range(count):
            g=index//pergroup;co=index%pergroup;biasval=0. if bias is None else bias.values[index]
            for position in range(outlen):
                source=[];kernels=[]
                for k in range(kernel):
                    value=position+pads[0]-k*dilation
                    if value%stride or not 0<=value//stride<length:continue
                    value//=stride
                    for c in range(g*cpg,(g+1)*cpg):
                        source.append(a.values[(sample*channels+c)*length+value]);kernels.append(weights.values[(c*pergroup+co)*kernel+k])
                output.append(f32(dot(source,kernels)+biasval))
    return Tensor.make((batch,count,outlen),output)
