"""Strict data reader and execution of the reviewed Piper VITS graph."""
from array import array
from collections import Counter, defaultdict
import math
import operator
from pathlib import Path
import struct
import sys
import threading
import time
from . import local_piper_tensor as t
from .local_piper_tensor import Tensor
from .local_piper_random import NormalGenerator

def varint(data, position):
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError('Truncated ONNX integer')
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return (value, position)
    raise ValueError('ONNX integer exceeds64bits')

def fields(data):
    position = 0
    while position < len(data):
        tag, position = varint(data, position)
        number, wire = (tag >> 3, tag & 7)
        if not number:
            raise ValueError('Invalid ONNX field')
        if wire == 0:
            value, position = varint(data, position)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, position = varint(data, position)
            else:
                size = 8 if wire == 1 else 4
            if size > len(data) - position:
                raise ValueError('Truncated ONNX data')
            value = data[position:position + size]
            position += size
        else:
            raise ValueError('Unsupported ONNX protobuf field')
        yield (number, wire, value)

def message(data):
    result = defaultdict(list)
    for number, wire, value in fields(data):
        result[number].append((wire, value))
    return result

def integers(values, signed=True):
    result = []
    for wire, value in values:
        if wire == 0:
            result.append(value)
        elif wire == 2:
            pos = 0
            while pos < len(value):
                number, pos = varint(value, pos)
                result.append(number)
        else:
            raise ValueError('Invalid ONNX integer field')
    return [v - (1 << 64) if signed and v & 1 << 63 else v for v in result]

def read_tensor(data):
    d = message(data)
    shape = tuple(integers(d.get(1, []), False))
    dtype = d[2][0][1]
    if 13 in d or d.get(14, [(0, 0)])[0][1]:
        raise ValueError('External ONNX tensors require an explicit data resolver')
    if len(shape) > 16 or math.prod(shape) > 200000000:
        raise ValueError('Piper model tensor exceeds dimensions bound')
    name = d.get(8, [(2, b'')])[0][1].decode()
    if 9 in d:
        raw = d[9][0][1]
        formats = {1: 'f', 6: 'i', 7: 'q', 9: 'b', 11: 'd'}
        if dtype not in formats:
            raise ValueError(('Unsupported ONNX tensor storage', dtype))
        values = array(formats[dtype])
        values.frombytes(raw)
        if sys.byteorder != 'little':
            values.byteswap()
        if dtype not in (1, 6, 7, 9):
            raise ValueError('Unsupported Piper tensor arithmetic')
        value = Tensor(shape, values, dtype)
    elif dtype == 1:
        raw = b''.join(v for w, v in d.get(4, []))
        value = Tensor.make(shape, (x[0] for x in struct.iter_unpack('<f', raw)))
    elif dtype in (6, 7, 9):
        value = Tensor.make(shape, integers(d.get(7 if dtype == 7 else 5, [])), dtype)
    else:
        raise ValueError(('Unsupported ONNX tensor dtype', dtype))
    return (name, value)

def read_attribute(data):
    d = message(data)
    name = d[1][0][1].decode()
    kind = d[20][0][1]
    if kind == 1:
        value = struct.unpack('<f', d[2][0][1])[0]
    elif kind == 2:
        value = integers(d[3])[0]
    elif kind == 3:
        value = d[4][0][1].decode()
    elif kind == 4:
        value = read_tensor(d[5][0][1])[1]
    elif kind == 6:
        value = [x[0] for x in struct.iter_unpack('<f', b''.join((v for w, v in d[7])))]
    elif kind == 7:
        value = integers(d[8])
    else:
        raise ValueError(('Unsupported ONNX attribute', name, kind))
    return (name, value)


def input_spec(data):
    info=message(data);name=info[1][0][1].decode()
    kind=message(info[2][0][1]);tensor=message(kind[1][0][1]);dtype=tensor[1][0][1]
    shape=None
    if 2 in tensor:
        dimensions=message(tensor[2][0][1]);shape=[]
        for wire,raw in dimensions.get(1,[]):
            dimension=message(raw);shape.append(dimension[1][0][1]if 1 in dimension else None)
        shape=tuple(shape)
    return name,(dtype,shape)

class PiperModel:
    _handlers = {'Add': '_arithmetic', 'And': '_arithmetic', 'Cast': '_arithmetic', 'Ceil': '_arithmetic', 'Clip': '_reductions', 'Concat': '_slicing', 'Constant': '_arithmetic', 'ConstantOfShape': '_reductions', 'Conv': '_neural', 'ConvTranspose': '_neural', 'CumSum': '_reductions', 'Div': '_arithmetic', 'Equal': '_arithmetic', 'Erf': '_arithmetic', 'Exp': '_arithmetic', 'Expand': '_slicing', 'Gather': '_indices', 'GatherElements': '_indices', 'GatherND': '_indices', 'GreaterOrEqual': '_arithmetic', 'Identity': '_arithmetic', 'LeakyRelu': '_arithmetic', 'Less': '_arithmetic', 'LessOrEqual': '_arithmetic', 'MatMul': '_neural', 'Mul': '_arithmetic', 'Neg': '_arithmetic', 'NonZero': '_indices', 'Not': '_arithmetic', 'Pad': '_slicing', 'Pow': '_arithmetic', 'RandomNormalLike': '_neural', 'Range': '_indices', 'ReduceMax': '_reductions', 'ReduceMean': '_reductions', 'ReduceSum': '_reductions', 'Relu': '_arithmetic', 'Reshape': '_axes', 'ScatterND': '_indices', 'Shape': '_indices', 'Sigmoid': '_arithmetic', 'Slice': '_slicing', 'Softmax': '_reductions', 'Softplus': '_arithmetic', 'Split': '_slicing', 'Sqrt': '_arithmetic', 'Squeeze': '_axes', 'Sub': '_arithmetic', 'Tanh': '_arithmetic', 'Transpose': '_axes', 'Unsqueeze': '_axes', 'Where': '_indices'}

    def __init__(self, path):
        raw = Path(path).read_bytes()
        model = message(raw)
        if len(model.get(7, [])) != 1:
            raise ValueError('Piper model requires a single graph')
        opsets = []
        for wire, value in model.get(8, []):
            item = message(value)
            domain = item.get(1, [(2, b'')])[0][1].decode()
            version = item[2][0][1]
            if domain not in ('', 'ai.onnx') or not 13 <= version <= 17:
                raise ValueError('Unsupported Piper ONNX operator schema')
            opsets.append(version)
        if not opsets:
            raise ValueError('Piper model does not declare an ONNX operator schema')
        graph = message(model[7][0][1])
        self.weights = {}
        self.nodes = []
        for wire, value in graph.get(5, []):
            name, tensor = read_tensor(value)
            if name in self.weights:
                raise ValueError('Duplicate ONNX initializer')
            self.weights[name] = tensor
        for wire, value in graph.get(1, []):
            d = message(value)
            attrs = dict((read_attribute(v) for w, v in d.get(5, [])))
            domain = d.get(7, [(2, b'')])[0][1].decode()
            if domain not in ('', 'ai.onnx'):
                raise ValueError('Piper model requires standard ONNX operators')
            self.nodes.append(dict(name=d.get(3, [(2, b'')])[0][1].decode(), op=d[4][0][1].decode(), inputs=[v.decode() for w, v in d.get(1, [])], outputs=[v.decode() for w, v in d.get(2, [])], attrs=attrs))
        supported = set(self._handlers)
        for node in self.nodes:
            if node['op'] not in supported:
                raise ValueError('Unsupported Piper ONNX operator ' + node['op'])
        self.input_specs=dict(input_spec(value)for wire,value in graph.get(11,[]))
        self.inputs = list(self.input_specs)
        self.outputs = [message(value)[1][0][1].decode() for wire, value in graph.get(12, [])]
        self.references = Counter((v for n in self.nodes for v in n['inputs'] if v))
        self._random = {}
        self.seed = time.time_ns() & 4294967295
        self._lock = threading.RLock()

    def run(self, inputs, outputs=None, *, stop=None, trace=None):
        with self._lock:
            return self._run(inputs, outputs, stop=stop, trace=trace)

    def _run(self, inputs, outputs=None, *, stop=None, trace=None):
        unknown=set(inputs)-set(self.inputs)
        if unknown:raise ValueError('Unexpected Piper model input '+', '.join(sorted(unknown)))
        for name,(dtype,shape)in self.input_specs.items():
            value=inputs.get(name,self.weights.get(name))
            if value is None:raise ValueError('Missing Piper model input '+name)
            if not isinstance(value,Tensor)or value.dtype!=dtype:
                raise ValueError('Wrong Piper tensor input type '+name)
            if shape is not None and (len(shape)!=len(value.shape)or any(a is not None and a!=b for a,b in zip(shape,value.shape))):
                raise ValueError('Wrong Piper tensor input shape '+name)
        values = dict(self.weights)
        values.update(inputs)
        outputs = self.outputs if outputs is None else outputs
        remaining = self.references.copy()
        retained = set(outputs)
        for index, node in enumerate(self.nodes):
            if stop is not None and index >= stop:
                break
            arguments = [values[name] if name else None for name in node['inputs']]
            try:
                result = self.execute(index, node, arguments)
            except Exception as exc:
                raise RuntimeError((index, node['name'], node['op'], [(a.shape, a.dtype) if a is not None else None for a in arguments], node['attrs'])) from exc
            if not isinstance(result, (tuple, list)):
                result = [result]
            if len(result) != len(node['outputs']):
                raise ValueError('Piper graph operator output count')
            for name, value in zip(node['outputs'], result):
                values[name] = value
            if trace:
                trace(index, node, result)
            for name in node['inputs']:
                if not name:
                    continue
                remaining[name] -= 1
                if not remaining[name] and name not in self.weights and (name not in retained):
                    values.pop(name, None)
        return [values[name] for name in outputs]

    def execute(self, index, node, args):
        handler = self._handlers.get(node['op'])
        if handler is None:
            raise ValueError(('Unsupported Piper ONNX operator', node['op']))
        return getattr(self, handler)(index, node, args)

    def _arithmetic(self, index, node, args):
        op = node['op']
        attr = node['attrs']
        a = args[0] if args else None
        if op == 'Identity':
            return a
        if op == 'Constant':
            return attr['value']
        if op in ('Add', 'Sub', 'Mul', 'Div', 'Pow', 'Less', 'Equal', 'GreaterOrEqual', 'LessOrEqual', 'And'):
            functions = {'Add': operator.add, 'Sub': operator.sub, 'Mul': operator.mul, 'Div': operator.truediv if a.dtype == 1 else lambda x, y: math.trunc(x / y), 'Pow': operator.pow, 'Less': operator.lt, 'Equal': operator.eq, 'GreaterOrEqual': operator.ge, 'LessOrEqual': operator.le, 'And': operator.and_}
            dtype = 9 if op in ('Less', 'Equal', 'GreaterOrEqual', 'LessOrEqual', 'And') else a.dtype
            return t.binary(a, args[1], functions[op], dtype)
        if op in ('Sqrt', 'Relu', 'Erf', 'Softplus', 'Neg', 'Exp', 'Ceil', 'Tanh', 'Sigmoid', 'LeakyRelu', 'Not'):
            functions = {'Sqrt': math.sqrt, 'Relu': lambda x: max(x, 0.0), 'Erf': math.erf, 'Softplus': lambda x: x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x)), 'Neg': operator.neg, 'Exp': lambda x: math.exp(x) if x < 89 else math.inf, 'Ceil': math.ceil, 'Tanh': math.tanh, 'Sigmoid': lambda x: 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x)), 'LeakyRelu': lambda x: x if x >= 0 else t.f32(x * attr.get('alpha', 0.01)), 'Not': operator.not_}
            return t.unary(a, functions[op])
        if op == 'Cast':
            return Tensor.make(a.shape, a.values, attr['to'])
        raise ValueError(('Unsupported Piper ONNX operator', op))

    def _indices(self, index, node, args):
        op = node['op']
        attr = node['attrs']
        a = args[0] if args else None
        if op == 'Shape':
            start = attr.get('start', 0)
            end = attr.get('end', len(a.shape))
            return Tensor.make((len(a.shape[start:end]),), a.shape[start:end], 7)
        if op == 'Gather':
            return t.gather(a, args[1], attr.get('axis', 0))
        if op == 'GatherND':
            return t.gather_nd(a, args[1])
        if op == 'GatherElements':
            return t.gather_elements(a, args[1], attr.get('axis', 0))
        if op == 'ScatterND':
            return t.scatter_nd(a, args[1], args[2])
        if op == 'NonZero':
            coords = [coord for coord, v in zip(t.coordinates(a.shape), a.values) if v]
            return Tensor.make((len(a.shape), len(coords)), (coord[i] for i in range(len(a.shape)) for coord in coords), 7)
        if op == 'Range':
            start, limit, step = (v.scalar for v in args)
            count = max(0, math.ceil((limit - start) / step))
            return Tensor.make((count,), (start + step * i for i in range(count)), a.dtype)
        if op == 'Where':
            return t.where(a, args[1], args[2])
        raise ValueError(('Unsupported Piper ONNX operator', op))

    def _axes(self, index, node, args):
        op = node['op']
        attr = node['attrs']
        a = args[0] if args else None
        if op == 'Transpose':
            return t.transpose(a, attr.get('perm'))
        if op == 'Reshape':
            return a.reshape(args[1].values, attr.get('allowzero', False))
        if op == 'Unsqueeze':
            axes = args[1].values if len(args) > 1 else attr['axes']
            rank = len(a.shape) + len(axes)
            axes = {axis % rank for axis in axes}
            source = iter(a.shape)
            return Tensor(tuple((1 if i in axes else next(source) for i in range(rank))), a.values, a.dtype)
        if op == 'Squeeze':
            axes = args[1].values if len(args) > 1 else attr.get('axes')
            axes = {i for i, n in enumerate(a.shape) if n == 1} if axes is None else {axis % len(a.shape) for axis in axes}
            if any((a.shape[i] != 1 for i in axes)):
                raise ValueError('Piper Squeeze requires unit dimensions')
            return Tensor(tuple((n for i, n in enumerate(a.shape) if i not in axes)), a.values, a.dtype)
        raise ValueError(('Unsupported Piper ONNX operator', op))

    def _slicing(self, index, node, args):
        op = node['op']
        attr = node['attrs']
        a = args[0] if args else None
        if op == 'Concat':
            return t.concat(args, attr['axis'])
        if op == 'Split':
            axis = attr.get('axis', 0) % len(a.shape)
            count = len(node['outputs'])
            sizes = args[1].values if len(args) > 1 else attr.get('split')
            if sizes is None:
                if a.shape[axis] % count:
                    raise ValueError('Unequal Piper Split')
                sizes = [a.shape[axis] // count] * count
            result = []
            start = 0
            for size in sizes:
                result.append(t.slice_tensor(a, [start], [start + size], [axis]))
                start += size
            if start != a.shape[axis]:
                raise ValueError('Piper Split must cover its input')
            return result
        if op == 'Slice':
            return t.slice_tensor(a, args[1].values, args[2].values, args[3].values if len(args) > 3 and args[3] is not None else None, args[4].values if len(args) > 4 and args[4] is not None else None)
        if op == 'Pad':
            if attr.get('mode', 'constant') != 'constant':
                raise ValueError('Unsupported Piper Pad mode')
            return t.pad(a, args[1].values, args[2].scalar if len(args) > 2 and args[2] is not None else 0)
        if op == 'Expand':
            shape = t.broadcast_shape(a.shape, tuple(args[1].values))
            steps = t.broadcast_strides(a.shape, shape)
            return Tensor.make(shape, (a.values[sum((c * s for c, s in zip(coord, steps)))] for coord in t.coordinates(shape)), a.dtype)
        raise ValueError(('Unsupported Piper ONNX operator', op))

    def _reductions(self, index, node, args):
        op = node['op']
        attr = node['attrs']
        a = args[0] if args else None
        if op in ('ReduceMean', 'ReduceSum', 'ReduceMax'):
            axes = args[1].values if len(args) > 1 else attr.get('axes')
            if axes is not None and (not len(axes)):
                if attr.get('noop_with_empty_axes', False):
                    return a
                axes = None
            return t.reduce(a, axes, attr.get('keepdims', 1), {'ReduceMean': 'mean', 'ReduceSum': 'sum', 'ReduceMax': 'max'}[op])
        if op == 'Softmax':
            return t.softmax(a, attr.get('axis', -1))
        if op == 'CumSum':
            return t.cumsum(a, args[1].scalar, attr.get('exclusive', False), attr.get('reverse', False))
        if op == 'ConstantOfShape':
            value = attr.get('value', Tensor.make((1,), [0.0]))
            shape = tuple(a.values)
            return Tensor.make(shape, (value.scalar for _ in range(math.prod(shape))), value.dtype)
        if op == 'Clip':
            low = args[1].scalar if len(args) > 1 and args[1] is not None else attr.get('min', -math.inf)
            high = args[2].scalar if len(args) > 2 and args[2] is not None else attr.get('max', math.inf)
            return t.unary(a, lambda x: max(low, min(high, x)))
        raise ValueError(('Unsupported Piper ONNX operator', op))

    def _neural(self, index, node, args):
        op = node['op']
        attr = node['attrs']
        a = args[0] if args else None
        if op == 'MatMul':
            return t.matmul(a, args[1])
        if op in ('Conv', 'ConvTranspose'):
            options = dict(pads=attr.get('pads', (0, 0)), strides=attr.get('strides', (1,)), dilations=attr.get('dilations', (1,)), group=attr.get('group', 1))
            if op == 'ConvTranspose':
                options['output_padding'] = attr.get('output_padding', (0,))
            return (t.conv if op == 'Conv' else t.conv_transpose)(a, args[1], args[2] if len(args) > 2 else None, **options)
        if op == 'RandomNormalLike':
            generator = self._random.setdefault(index, NormalGenerator(attr.get('seed', self.seed + index)))
            return Tensor.make(a.shape, generator.generate(len(a.values), attr.get('mean', 0.0), attr.get('scale', 1.0)))
        raise ValueError(('Unsupported Piper ONNX operator', op))
