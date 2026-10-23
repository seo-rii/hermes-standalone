"""Private stdlib execution of the pinned original G2PW quantized BERT graph."""
from collections import Counter
import math
from pathlib import Path
import threading
from .local_piper_model import PiperModel
from .local_piper_recurrent import ArabicModel
from . import local_piper_tensor as t
from .local_piper_tensor import Tensor
from . import local_piper_chinese_reader as reader
from .local_piper_chinese_quantized import IntegerProducts, uint8, dynamic_quantize, dequantize

def gather_uint8(a, indices, axis=0):
    axis %= len(a.shape)
    outer, width = (math.prod(a.shape[:axis]), math.prod(a.shape[axis + 1:]))
    size = a.shape[axis]
    output = bytearray()
    for i in range(outer):
        for value in indices.values:
            value = value + size if value < 0 else value
            if not 0 <= value < size:
                raise ValueError('G2PW Gather index out of range')
            start = (i * size + value) * width
            output.extend(a.values[start:start + width])
    return Tensor(a.shape[:axis] + indices.shape + a.shape[axis + 1:], output, 2)

class G2PWModel(PiperModel):
    _handlers = {**PiperModel._handlers, **{op: '_quantized' for op in ('ArgMax', 'Flatten', 'DequantizeLinear', 'DynamicQuantizeLinear', 'MatMulInteger')}}

    def __init__(self, path):
        data = Path(path).read_bytes()
        if len(data) > 300000000:
            raise ValueError('G2PW graph exceeds model size bound')
        graph = reader.graph(data)
        self.weights = {}
        for wire, value in graph.get(5, []):
            name, tensor = reader.tensor(value)
            if name in self.weights:
                raise ValueError('Duplicate G2PW initializer')
            self.weights[name] = tensor
        self.nodes = reader.nodes(graph)
        for node in self.nodes:
            if node['op'] not in self._handlers:
                raise ValueError('Unsupported G2PW ONNX operator ' + node['op'])
        self.input_specs, self.outputs = reader.io(graph)
        self.inputs = list(self.input_specs)
        self.references = Counter((v for node in self.nodes for v in node['inputs'] if v))
        self._random = {}
        self.seed = 0
        self._lock = threading.RLock()
        self._products = IntegerProducts()

    def _quantized(self, index, node, args):
        op, attrs = (node['op'], node['attrs'])
        if op == 'DynamicQuantizeLinear':
            return dynamic_quantize(args[0])
        if op == 'DequantizeLinear':
            return dequantize(*args)
        if op == 'MatMulInteger':
            return self._products.matmul(*args)
        if op == 'ArgMax':
            return ArabicModel._argmax(args[0], attrs)
        if op == 'Flatten':
            a = args[0]
            axis = attrs.get('axis', 1)
            if not -len(a.shape) <= axis <= len(a.shape):
                raise ValueError('Flatten axis out of bounds')
            if axis < 0:
                axis += len(a.shape)
            return a.reshape((math.prod(a.shape[:axis]), math.prod(a.shape[axis:])), True)
        raise ValueError('Unsupported G2PW quantized operator ' + op)

    def _arithmetic(self, index, node, args):
        if node['op'] == 'Cast' and node['attrs']['to'] == 2:
            return uint8(args[0].shape, (int(v) & 255 for v in args[0].values))
        return super()._arithmetic(index, node, args)

    def _indices(self, index, node, args):
        if node['op'] == 'Gather' and args[0].dtype == 2:
            return gather_uint8(args[0], args[1], node['attrs'].get('axis', 0))
        return super()._indices(index, node, args)

    def _reductions(self, index, node, args):
        if node['op'] == 'Softmax':
            a = args[0]
            axis = node['attrs'].get('axis', 1) % len(a.shape)
            flat = a.reshape((math.prod(a.shape[:axis]), math.prod(a.shape[axis:])), True)
            return t.softmax(flat, 1).reshape(a.shape, True)
        return super()._reductions(index, node, args)
