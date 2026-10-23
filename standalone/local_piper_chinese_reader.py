"""Bounded protobuf reader for the actual G2PW opset-12 uint8 graph."""
from array import array
import math
import struct
import sys
from .local_piper_model import message, integers, input_spec
from .local_piper_tensor import Tensor

def text(data):
    return bytes(data).decode()

def tensor(data):
    d = message(data)
    shape = tuple(integers(d.get(1, []), False))
    dtype = d[2][0][1]
    if 13 in d or d.get(14, [(0, 0)])[0][1]:
        raise ValueError('G2PW does not load external tensors')
    if len(shape) > 16 or math.prod(shape) > 200000000:
        raise ValueError('G2PW tensor dimension bound exceeded')
    name = text(d.get(8, [(2, b'')])[0][1])
    if 9 in d:
        raw = d[9][0][1]
        formats = {1: 'f', 2: 'B', 6: 'i', 7: 'q', 9: 'b'}
        if dtype not in formats:
            raise ValueError(('Unsupported G2PW tensor storage', dtype))
        if dtype == 2:
            return (name, Tensor(shape, raw, dtype))
        values = array(formats[dtype])
        values.frombytes(raw)
        if sys.byteorder != 'little':
            values.byteswap()
        return (name, Tensor(shape, values, dtype))
    if dtype == 1:
        raw = b''.join((bytes(v) for w, v in d.get(4, [])))
        value = Tensor.make(shape, (v[0] for v in struct.iter_unpack('<f', raw)))
    elif dtype in (6, 7, 9):
        value = Tensor.make(shape, integers(d.get(7 if dtype == 7 else 5, [])), dtype)
    elif dtype == 2:
        value = Tensor(shape, array('B', integers(d.get(5, []))), 2)
    else:
        raise ValueError(('Unsupported G2PW tensor type', dtype))
    return (name, value)

def attribute(data):
    d = message(data)
    name = text(d[1][0][1])
    kind = d[20][0][1]
    if kind == 1:
        value = struct.unpack('<f', d[2][0][1])[0]
    elif kind == 2:
        value = integers(d[3])[0]
    elif kind == 3:
        value = text(d[4][0][1])
    elif kind == 4:
        value = tensor(d[5][0][1])[1]
    elif kind == 6:
        value = [v[0] for v in struct.iter_unpack('<f', b''.join((bytes(v) for w, v in d[7])))]
    elif kind == 7:
        value = integers(d[8])
    else:
        raise ValueError(('Unsupported G2PW attribute', name, kind))
    return (name, value)

def nodes(graph):
    result = []
    for wire, value in graph.get(1, []):
        d = message(value)
        if text(d.get(7, [(2, b'')])[0][1]) not in ('', 'ai.onnx'):
            raise ValueError('G2PW requires standard ONNX operator domains')
        result.append(dict(name=text(d.get(3, [(2, b'')])[0][1]), op=text(d[4][0][1]), inputs=[text(v) for w, v in d.get(1, [])], outputs=[text(v) for w, v in d.get(2, [])], attrs=dict((attribute(v) for w, v in d.get(5, [])))))
    return result

def graph(raw):
    model = message(memoryview(raw))
    if len(model.get(7, [])) != 1:
        raise ValueError('G2PW requires one ONNX graph')
    if not model.get(8):
        raise ValueError('G2PW requires a declared operator schema')
    for wire, value in model[8]:
        item = message(value)
        if text(item.get(1, [(2, b'')])[0][1]) not in ('', 'ai.onnx') or item[2][0][1] != 12:
            raise ValueError('Qualified G2PW graph requires ONNX opset 12')
    return message(model[7][0][1])

def io(graph):
    specs = dict((input_spec(bytes(v)) for w, v in graph.get(11, [])))
    outputs = [text(message(v)[1][0][1]) for w, v in graph.get(12, [])]
    return (specs, outputs)
