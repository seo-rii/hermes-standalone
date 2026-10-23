"""Exact uint8 integer products and opset-12 G2PW quantization."""
from array import array
import math
from . import local_piper_tensor as t
from .local_piper_tensor import Tensor, f32

def uint8(shape, values):
    return Tensor(tuple(shape), array('B', values), 2)

def dynamic_quantize(a):
    low = min(0.0, min(a.values))
    high = max(0.0, max(a.values))
    scale = f32(f32(high - low) / 255) if high != low else 1.0
    zero = min(255, max(0, round(f32(-low / scale))))
    inverse = f32(1 / scale)
    values = (min(255, max(0, round(f32(v * inverse)) + zero)) for v in a.values)
    return (uint8(a.shape, values), Tensor.make((), [scale]), uint8((), [zero]))

def dequantize(a, scale, zero=None):
    if len(scale.values) != 1 or (zero is not None and len(zero.values) != 1):
        raise ValueError('Opset-12 dequantization requires scalar scale and zero point')
    point = zero.scalar if zero is not None else 0
    return Tensor.make(a.shape, ((int(v) - point) * scale.scalar for v in a.values))

def bitplanes(values):
    raw = bytes(values)
    text = format(int.from_bytes(raw, 'big'), '0' + str(len(raw) * 8) + 'b')
    return tuple((int(text[7 - i::8], 2) for i in range(8)))

def weighted_count(plane, columns):
    a, b, c, d, e, f, g, h = columns
    return (plane & a).bit_count() + ((plane & b).bit_count() << 1) + ((plane & c).bit_count() << 2) + ((plane & d).bit_count() << 3) + ((plane & e).bit_count() << 4) + ((plane & f).bit_count() << 5) + ((plane & g).bit_count() << 6) + ((plane & h).bit_count() << 7)

class IntegerProducts:

    def __init__(self):
        self.columns = {}

    def _columns(self, b, offset, k, n):
        key = (id(b), offset, k, n)
        value = self.columns.get(key)
        if value is None:
            data = b.values[offset:offset + k * n]
            columns = [data[j::n] for j in range(n)]
            value = (b, tuple((bitplanes(c) for c in columns)), tuple((sum(c) for c in columns)))
            self.columns[key] = value
        return value[1:]

    def _matrix(self, a, b, apos, bpos, m, k, n, azero, bzero, rows):
        planes, sums = self._columns(b, bpos, k, n)
        out = []
        for row in range(m):
            source = bytes(a.values[apos + row * k:apos + (row + 1) * k])
            key = (bpos, source)
            output = rows.get(key)
            if output is None:
                bits = bitplanes(source)
                row_sum = sum(source)
                correction = k * azero - row_sum
                output = [sum((weighted_count(p, col) << i for i, p in enumerate(bits))) - azero * col_sum + point * correction for col, col_sum, point in zip(planes, sums, bzero)]
                rows[key] = output
            out.extend(output)
        return out

    def matmul(self, a, b, azero=None, bzero=None):
        if a.dtype != 2 or b.dtype != 2:
            raise ValueError('Qualified G2PW integer products require uint8 inputs')
        if not a.shape or not b.shape:
            raise ValueError('MatMulInteger inputs require rank at least one')
        if azero is not None and len(azero.values) != 1:
            raise ValueError('Original CPU MatMulInteger requires scalar input zero point')
        avector, bvector = (len(a.shape) == 1, len(b.shape) == 1)
        ashape = (1,) + a.shape if avector else a.shape
        bshape = b.shape + (1,) if bvector else b.shape
        m, k = ashape[-2:]
        if bshape[-2] != k:
            raise ValueError('MatMulInteger inner dimension mismatch')
        n = bshape[-1]
        batch = t.broadcast_shape(ashape[:-2], bshape[:-2])
        sa = t.broadcast_strides(ashape[:-2], batch)
        sb = t.broadcast_strides(bshape[:-2], batch)
        azero = azero.scalar if azero is not None else 0
        points = [0] * n if bzero is None else list(bzero.values) * (n if len(bzero.values) == 1 else 1)
        if len(points) != n:
            raise ValueError('Original CPU MatMulInteger requires scalar or column zero point')
        output, rows = ([], {})
        for coord in t.coordinates(batch):
            apos = sum((c * s for c, s in zip(coord, sa))) * m * k
            bpos = sum((c * s for c, s in zip(coord, sb))) * k * n
            output.extend(self._matrix(a, b, apos, bpos, m, k, n, azero, points, rows))
        shape = batch + (() if avector else (m,)) + (() if bvector else (n,))
        return Tensor.make(shape, output, 6)
