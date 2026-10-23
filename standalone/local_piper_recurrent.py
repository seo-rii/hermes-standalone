"""Piper libtashkeel GRU/attention graph inference using owned stdlib tensors."""
from array import array
import math
from standalone.local_piper_model import PiperModel
from standalone import local_piper_tensor as t
from standalone.local_piper_tensor import Tensor, f32, dot

def sigmoid(x):
    return f32(1 / (1 + math.exp(-x))) if x >= 0 else f32(math.exp(x) / (1 + math.exp(x)))

def gru(args, attrs):
    x, w, r = args[:3]
    bias = args[3] if len(args) > 3 else None
    lengths = args[4] if len(args) > 4 else None
    initial = args[5] if len(args) > 5 else None
    if attrs.get('layout', 0) != 0:
        raise ValueError('Qualified GRU requires time-major layout')
    if any((key in attrs for key in ('activations', 'activation_alpha', 'activation_beta'))):
        raise ValueError('Qualified GRU requires standard sigmoid/tanh activations')
    steps, batch, width = x.shape
    directions, gates, input_width = w.shape
    hidden = gates // 3
    if input_width != width or r.shape != (directions, hidden * 3, hidden) or attrs.get('hidden_size', hidden) != hidden:
        raise ValueError('Invalid GRU matrix dimensions')
    direction = attrs.get('direction', 'forward')
    if directions != (2 if direction == 'bidirectional' else 1):
        raise ValueError('Invalid GRU direction count')
    if direction not in ('forward', 'reverse', 'bidirectional'):
        raise ValueError('Invalid GRU direction')
    lens = [steps] * batch if lengths is None else list(lengths.values)
    if len(lens) != batch or any((not 0 <= v <= steps for v in lens)):
        raise ValueError('Invalid GRU sequence lengths')
    lbr = attrs.get('linear_before_reset', 0)
    clip = attrs.get('clip', math.inf)
    output = [0.0] * (steps * directions * batch * hidden)
    final = [0.0] * (directions * batch * hidden)
    for d in range(directions):
        kernels = tuple((w.values[(d * 3 * hidden + i) * width:(d * 3 * hidden + i + 1) * width] for i in range(3 * hidden)))
        recurrent = tuple((r.values[(d * 3 * hidden + i) * hidden:(d * 3 * hidden + i + 1) * hidden] for i in range(3 * hidden)))
        biases = [0.0] * (6 * hidden) if bias is None else bias.values[d * 6 * hidden:(d + 1) * 6 * hidden]
        for b in range(batch):
            state = [0.0] * hidden if initial is None else list(initial.values[(d * batch + b) * hidden:(d * batch + b + 1) * hidden])
            positions = reversed(range(lens[b])) if direction == 'reverse' or d == 1 else range(lens[b])
            for pos in positions:
                current = x.values[(pos * batch + b) * width:(pos * batch + b + 1) * width]
                projected = [f32(f32(dot(current, k)) + v) for k, v in zip(kernels, biases[:3 * hidden])]
                gates_ = [f32(f32(projected[i] + f32(dot(state, recurrent[i]))) + biases[3 * hidden + i]) for i in range(2 * hidden)]
                update = [sigmoid(max(-clip, min(clip, v))) for v in gates_[:hidden]]
                reset = [sigmoid(max(-clip, min(clip, v))) for v in gates_[hidden:]]
                candidate = []
                if lbr:
                    for i in range(hidden):
                        rec = f32(f32(dot(state, recurrent[2 * hidden + i])) + biases[5 * hidden + i])
                        value = f32(projected[2 * hidden + i] + f32(reset[i] * rec))
                        candidate.append(f32(math.tanh(max(-clip, min(clip, value)))))
                else:
                    reset_state = [f32(v * s) for v, s in zip(reset, state)]
                    for i in range(hidden):
                        value = f32(f32(projected[2 * hidden + i] + f32(dot(reset_state, recurrent[2 * hidden + i]))) + biases[5 * hidden + i])
                        candidate.append(f32(math.tanh(max(-clip, min(clip, value)))))
                state = [f32(f32(f32(1 - z) * c) + f32(z * h)) for z, c, h in zip(update, candidate, state)]
                start = ((pos * directions + d) * batch + b) * hidden
                output[start:start + hidden] = state
            if not lens[b]:
                state = [0.0] * hidden
            start = (d * batch + b) * hidden
            final[start:start + hidden] = state
    return (Tensor.make((steps, directions, batch, hidden), output), Tensor.make((directions, batch, hidden), final))

def topk(a, k, axis=-1, largest=True):
    axis %= len(a.shape)
    size = a.shape[axis]
    width = math.prod(a.shape[axis + 1:])
    outer = math.prod(a.shape[:axis])
    shape = list(a.shape)
    shape[axis] = k
    if not 0 <= k <= size:
        raise ValueError('TopK count out of range')
    values = [0] * (outer * k * width)
    indices = [0] * len(values)
    for i in range(outer):
        for j in range(width):
            ranked = sorted(range(size), key=lambda n: (-a.values[(i * size + n) * width + j] if largest else a.values[(i * size + n) * width + j], n))[:k]
            for out, index in enumerate(ranked):
                target = (i * k + out) * width + j
                values[target] = a.values[(i * size + index) * width + j]
                indices[target] = index
    return (Tensor.make(shape, values, a.dtype), Tensor.make(shape, indices, 7))

def scatter_elements(a, indices, updates, axis=0):
    axis %= len(a.shape)
    strides = t.strides(a.shape)
    out = list(a.values)
    for coord, index, value in zip(t.coordinates(indices.shape), indices.values, updates.values, strict=True):
        if index < 0:
            index += a.shape[axis]
        if not 0 <= index < a.shape[axis]:
            raise ValueError('ScatterElements index out of range')
        position = sum(((index if i == axis else c) * s for i, (c, s) in enumerate(zip(coord, strides))))
        out[position] = value
    return Tensor.make(a.shape, out, a.dtype)

def reduction_product(a, axes=None, keepdims=True):
    rank = len(a.shape)
    axes = set(range(rank)) if axes is None or not len(axes) else {v % rank for v in axes}
    retained = [i for i in range(rank) if i not in axes]
    ordered = retained + [i for i in range(rank) if i in axes]
    shape = tuple((1 if i in axes else n for i, n in enumerate(a.shape))) if keepdims else tuple((a.shape[i] for i in retained))
    moved = t.transpose(a, ordered)
    width = math.prod((a.shape[i] for i in axes))
    output = []
    for start in range(0, len(moved.values), width):
        value = 1
        for item in moved.values[start:start + width]:
            value = f32(value * item) if a.dtype == 1 else value * item
        output.append(value)
    return Tensor.make(shape, output, a.dtype)

class ArabicModel(PiperModel):
    _handlers = {**PiperModel._handlers, **{op: '_arabic' for op in ('Einsum', 'Sin', 'Cos', 'Dropout', 'TopK', 'ReduceProd', 'ScatterElements', 'GRU', 'BatchNormalization', 'ArgMax')}}

    def _arithmetic(self, index, node, args):
        if node['op'] == 'Cast' and node['attrs']['to'] == 2:
            return Tensor(args[0].shape, array('B', (int(v) & 255 for v in args[0].values)), 2)
        return super()._arithmetic(index, node, args)

    def _arabic(self, index, node, args):
        op = node['op']
        attrs = node['attrs']
        a = args[0]
        if op == 'GRU':
            return gru(args, attrs)[:len(node['outputs'])]
        if op in ('Sin', 'Cos'):
            return t.unary(a, math.sin if op == 'Sin' else math.cos)
        if op == 'Einsum':
            if ''.join(attrs['equation'].split()) != 'i,j->ij' or any((len(v.shape) != 1 for v in args)):
                raise ValueError('Unsupported Piper Einsum equation')
            return Tensor.make((a.shape[0], args[1].shape[0]), (x * y for x in a.values for y in args[1].values))
        if op == 'Dropout':
            if len(args) > 2 and args[2] is not None and args[2].scalar:
                raise ValueError('Piper model inference does not enable training dropout')
            return [a, Tensor.make(a.shape, (True for _ in a.values), 9)][:len(node['outputs'])]
        if op == 'TopK':
            if not attrs.get('sorted', 1):
                raise ValueError('Qualified Piper TopK requires sorted output')
            return topk(a, args[1].scalar, attrs.get('axis', -1), attrs.get('largest', 1))
        if op == 'ScatterElements':
            return scatter_elements(a, args[1], args[2], attrs.get('axis', 0))
        if op == 'ReduceProd':
            return reduction_product(a, args[1].values if len(args) > 1 else attrs.get('axes'), attrs.get('keepdims', 1))
        if op == 'BatchNormalization':
            return self._batch_normalization(args, attrs)
        if op == 'ArgMax':
            return self._argmax(a, attrs)
        raise ValueError('Unsupported Piper Arabic operator ' + op)

    @staticmethod
    def _batch_normalization(args, attrs):
        if attrs.get('training_mode', 0):
            raise ValueError('Qualified Piper normalization is inference-only')
        a, scale, bias, mean, var = args
        channels = a.shape[1]
        width = math.prod(a.shape[2:])
        output = []
        alpha = [f32(s / math.sqrt(f32(v + attrs.get('epsilon', 1e-05)))) for s, v in zip(scale.values, var.values)]
        beta = [f32(b - f32(m * s)) for b, m, s in zip(bias.values, mean.values, alpha)]
        for i, x in enumerate(a.values):
            channel = i // width % channels
            output.append(f32(f32(x * alpha[channel]) + beta[channel]))
        return Tensor.make(a.shape, output)

    @staticmethod
    def _argmax(a, attrs):
        axis = attrs.get('axis', 0) % len(a.shape)
        size = a.shape[axis]
        width = math.prod(a.shape[axis + 1:])
        outer = math.prod(a.shape[:axis])
        out = []
        for i in range(outer):
            for j in range(width):
                values = [a.values[(i * size + k) * width + j] for k in range(size)]
                maximum = max(values)
                out.append(max((k for k, v in enumerate(values) if v == maximum)) if attrs.get('select_last_index', 0) else values.index(maximum))
        shape = a.shape[:axis] + ((1,) if attrs.get('keepdims', 1) else ()) + a.shape[axis + 1:]
        return Tensor.make(shape, out, 7)
