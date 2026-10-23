"""Reviewed Nakdimon two bidirectional400-cell LSTM model, stdlib only."""
from array import array
import hashlib
import math
from pathlib import Path
from standalone.local_piper_model import message, read_tensor
from standalone.local_piper_tensor import f32, dot
MODEL_SHA256 = '9ff491dcc7d66392019d427a98b97d5de10c0d721628ae740858174ae22b190e'
MODEL_BYTES = 21312753

def sigmoid(value):
    return f32(1 / (1 + math.exp(-value))) if value >= 0 else f32(math.exp(value) / (1 + math.exp(value)))

def columns(tensor):
    height, width = tensor.shape
    return tuple((array('f', tensor.values[index::width]) for index in range(width)))

class NakdimonModel:

    def __init__(self, path):
        raw = Path(path).read_bytes()
        if len(raw) != MODEL_BYTES or hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
            raise ValueError('Nakdimon requires the reviewed Piper1.8.0 model data')
        graph = message(message(raw)[7][0][1])
        weights = dict((read_tensor(value) for wire, value in graph[5]))
        self.layers = []
        self.embedding = weights['model/embedding/embedding_lookup/7542:0']
        for layer in ('bidirectional', 'bidirectional_1'):
            directions = []
            for direction, index in (('forward', 3 if layer == 'bidirectional' else 27), ('backward', 15 if layer == 'bidirectional' else 39)):
                suffix = '' if layer == 'bidirectional' else '_1'
                stem = f'Func/model/{layer}/{direction}_lstm{suffix}/PartitionedCall/input/'
                kernel = weights[stem + '_' + str(index) + ':0']
                recurrent = weights[stem + '_' + str(index + 1) + ':0']
                bias = weights[stem + '_' + str(index + 2) + ':0']
                if kernel.shape != (400, 1600) or recurrent.shape != (400, 1600) or bias.shape != (1600,):
                    raise ValueError('Invalid reviewed Nakdimon LSTM dimensions')
                directions.append((columns(kernel), columns(recurrent), tuple(bias.values)))
            self.layers.append(directions)
        self.dense = (columns(weights['model/dense/Tensordot/ReadVariableOp:0']), tuple(weights['model/dense/BiasAdd/ReadVariableOp:0'].values))
        self.heads = [(columns(weights['model/' + name + '/Tensordot/ReadVariableOp:0']), tuple(weights['model/' + name + '/BiasAdd/ReadVariableOp:0'].values)) for name in ('N', 'D', 'S')]

    @staticmethod
    def recurrent(values, mask, parameters, reverse=False):
        kernel, recurrent, bias = parameters
        size = len(bias) // 4
        hidden = (0.0,) * size
        cell = (0.0,) * size
        out = [None] * len(values)
        for position in reversed(range(len(values))) if reverse else range(len(values)):
            if not mask[position]:
                out[position] = (0.0,) * size
                continue
            current = values[position]
            gates = [f32(f32(f32(dot(current, k)) + f32(dot(hidden, r))) + b) for k, r, b in zip(kernel, recurrent, bias)]
            inputs = [sigmoid(v) for v in gates[:size]]
            forget = [sigmoid(v) for v in gates[size:2 * size]]
            candidate = [f32(math.tanh(v)) for v in gates[2 * size:3 * size]]
            output = [sigmoid(v) for v in gates[3 * size:]]
            cell = tuple((f32(f32(f * c) + f32(i * g)) for f, c, i, g in zip(forget, cell, inputs, candidate)))
            hidden = tuple((f32(o * f32(math.tanh(c))) for o, c in zip(output, cell)))
            out[position] = hidden
        return out

    def run(self, ids, trace=None):
        if not ids:
            return ([], [], [])
        if any((not 0 <= value < 43 for value in ids)):
            raise ValueError('Nakdimon character ID out of range')
        values = [tuple(self.embedding.values[i * 400:(i + 1) * 400]) for i in ids]
        mask = [value != 0 for value in ids]
        for layer, directions in enumerate(self.layers):
            forward = self.recurrent(values, mask, directions[0])
            backward = self.recurrent(values, mask, directions[1], True)
            values = [tuple((f32(f + b) for f, b in zip(a, z))) for a, z in zip(forward, backward)]
            if trace:
                trace(layer, values)
        columns_, bias = self.dense
        values = [tuple((f32(f32(dot(row, column)) + offset) for column, offset in zip(columns_, bias))) for row in values]
        if trace:
            trace(2, values)
        return [[tuple((f32(f32(dot(row, column)) + offset) for column, offset in zip(cols, bias))) for row in values] for cols, bias in self.heads]
