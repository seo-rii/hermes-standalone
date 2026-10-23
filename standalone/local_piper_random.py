"""Pinned ORT Linux libstdc++ normal generator, implemented with integers/f32."""
import math
from .local_piper_tensor import f32


class NormalGenerator:
    def __init__(self,seed):
        self.state=(int(seed)&0xffffffff)%2147483647 or 1

    def uniform(self):
        self.state=self.state*16807%2147483647
        value=f32(f32(self.state-1)/2147483648.)
        return min(value,0.9999999403953552)

    def generate(self,count,mean=0.,scale=1.):
        # ORT constructs a new normal_distribution on every tensor invocation.
        # Its saved variate therefore ends at that tensor boundary, while the
        # underlying minstd_rand0 state is retained by the kernel/session.
        out=[];saved=None
        for _ in range(count):
            if saved is not None:
                value=saved;saved=None
            else:
                while True:
                    x=f32(f32(2*self.uniform())-1);y=f32(f32(2*self.uniform())-1)
                    radius=f32(f32(x*x)+f32(y*y))
                    if 0<radius<=1:break
                factor=f32(math.sqrt(f32(f32(-2*f32(math.log(radius)))/radius)))
                saved=f32(x*factor);value=f32(y*factor)
            out.append(f32(f32(value*scale)+mean))
        return out
