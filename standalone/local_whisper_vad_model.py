"""Strict reader and stdlib inference for faster-whisper's Silero VAD v6 data.

Only the reviewed 16 kHz ONNX asset is accepted. The graph is data, and no
framework, executable tensor operation or native extension is loaded.
"""

from array import array
import hashlib
import math
from pathlib import Path
import struct
import sys

MODEL_SHA256 = '4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2'
MODEL_BYTES = 1245151
_FLOAT = struct.Struct('<f')
_dot = getattr(math, 'sumprod', lambda a, b: sum(x*y for x, y in zip(a, b, strict=True)))


def f32(value):
    return _FLOAT.unpack(_FLOAT.pack(value))[0]


def _varint(data, position):
    value = 0
    for shift in range(0, 70, 7):
        if position == len(data):
            raise ValueError('Truncated Silero model varint')
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, position
    raise ValueError('Silero model varint exceeds 64 bits')


def _fields(data):
    position = 0
    while position < len(data):
        tag, position = _varint(data, position)
        number, wire = tag >> 3, tag & 7
        if not number:
            raise ValueError('Invalid Silero model field')
        if wire == 0:
            value, position = _varint(data, position)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, position = _varint(data, position)
            else:
                size = 8 if wire == 1 else 4
            if size > len(data)-position:
                raise ValueError('Truncated Silero model field')
            value = data[position:position+size]
            position += size
        else:
            raise ValueError('Unsupported Silero model wire type')
        yield number, wire, value


def _weights(data):
    if len(data) != MODEL_BYTES or hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError('Silero VAD requires the reviewed faster-whisper 1.2.1 model')
    graph = [value for number, wire, value in _fields(data) if number == 7 and wire == 2]
    if len(graph) != 1:
        raise ValueError('Invalid reviewed Silero graph')
    tensors = {}
    for number, wire, value in _fields(graph[0]):
        if number != 5:
            continue
        fields = list(_fields(value))
        name = next(v.decode() for n, w, v in fields if n == 8)
        kind = next(v for n, w, v in fields if n == 2)
        if kind != 1:
            continue
        dimensions = []
        for n, w, v in fields:
            if n != 1:
                continue
            if w == 0:
                dimensions.append(v)
            else:
                offset = 0
                while offset < len(v):
                    dimension, offset = _varint(v, offset)
                    dimensions.append(dimension)
        raw = next(v for n, w, v in fields if n == 9)
        if len(raw) != math.prod(dimensions)*4 or name in tensors:
            raise ValueError('Invalid reviewed Silero tensor')
        values = array('f')
        values.frombytes(raw)
        if sys.byteorder != 'little':
            values.byteswap()
        tensors[name] = tuple(dimensions), values
    return tensors


def _sigmoid(value):
    if value >= 0:
        return f32(1.0/(1.0+math.exp(-value)))
    exponential = math.exp(value)
    return f32(exponential/(1.0+exponential))


class SileroVADModel:
    def __init__(self, path=None):
        path = Path(path) if path is not None else Path(__file__).with_name('assets')/'silero_vad_v6.onnx'
        with path.open('rb') as source:
            data = source.read(MODEL_BYTES+1)
        tensors = _weights(data)
        self._layers = []
        for index, shape in enumerate(((128, 129, 3), (64, 128, 3), (64, 64, 3), (128, 64, 3))):
            name = f'encoder.conv_layers.{index}'
            if tensors[name+'.weight'][0] != shape:
                raise ValueError('Invalid reviewed Silero convolution dimensions')
            weights = tensors[name+'.weight'][1]
            width = math.prod(shape[1:])
            kernels = tuple(weights[i*width:(i+1)*width] for i in range(shape[0]))
            self._layers.append((kernels, tensors[name+'.bias'][1], (1, 2, 2, 1)[index]))
        basis = tensors['encoder.feature_extractor.forward_basis_buffer'][1]
        self._basis = tuple(basis[i*256:(i+1)*256] for i in range(258))
        self._input = tuple(tensors['onnx::LSTM_209'][1][i*128:(i+1)*128] for i in range(512))
        self._recurrent = tuple(tensors['onnx::LSTM_210'][1][i*128:(i+1)*128] for i in range(512))
        bias = tensors['onnx::LSTM_211'][1]
        self._bias = tuple(f32(bias[i]+bias[i+512]) for i in range(512))
        self._output = tensors['decoder.conv1d.weight'][1]
        self._output_bias = tensors['decoder.conv1d.bias'][1][0]

    def encode(self, samples):
        if len(samples) != 576:
            raise ValueError('Silero VAD encoder requires 64 context + 512 samples')
        samples = tuple(f32(value) for value in samples)
        padded = samples[128:0:-1]+samples+samples[-2:-130:-1]
        frames = tuple(padded[128+offset:384+offset] for offset in range(0, 512, 128))
        transformed = tuple(tuple(f32(_dot(frame, kernel)) for frame in frames) for kernel in self._basis)
        magnitude = tuple(tuple(f32(math.sqrt(f32(f32(real*real)+f32(imag*imag))))
                                for real, imag in zip(transformed[i], transformed[i+129], strict=True))
                          for i in range(129))
        values = magnitude
        for kernels, bias, stride in self._layers:
            length = len(values[0])
            positions = tuple(tuple(channel[index] if 0 <= index < length else 0.0
                                    for channel in values for index in (position-1, position, position+1))
                              for position in range(0, length, stride))
            values = tuple(tuple(max(0.0, f32(_dot(position, kernel)+offset)) for position in positions)
                           for kernel, offset in zip(kernels, bias, strict=True))
        return tuple(channel[0] for channel in values)

    def step(self, samples, hidden=None, cell=None):
        hidden = (0.0,)*128 if hidden is None else tuple(hidden)
        cell = (0.0,)*128 if cell is None else tuple(cell)
        if len(hidden) != 128 or len(cell) != 128:
            raise ValueError('Silero VAD recurrent state requires 128 components')
        encoded = self.encode(samples)
        gates = tuple(f32(f32(_dot(encoded, first)+_dot(hidden, second))+offset)
                      for first, second, offset in zip(self._input, self._recurrent, self._bias, strict=True))
        inputs = tuple(_sigmoid(value) for value in gates[:128])
        outputs = tuple(_sigmoid(value) for value in gates[128:256])
        forget = tuple(_sigmoid(value) for value in gates[256:384])
        candidates = tuple(f32(math.tanh(value)) for value in gates[384:])
        cell = tuple(f32(f32(f*c)+f32(i*g)) for f, c, i, g in zip(forget, cell, inputs, candidates, strict=True))
        hidden = tuple(f32(o*f32(math.tanh(c))) for o, c in zip(outputs, cell, strict=True))
        prediction = _sigmoid(f32(_dot(tuple(max(0.0, h) for h in hidden), self._output)+self._output_bias))
        return prediction, hidden, cell

    def __call__(self, audio, num_samples=512, context_size_samples=64):
        if num_samples != 512 or context_size_samples != 64 or len(audio) % num_samples:
            raise ValueError('Reviewed Silero VAD requires complete 512-sample windows and 64-sample context')
        hidden = cell = None
        context = (0.0,)*context_size_samples
        probabilities = []
        for position in range(0, len(audio), num_samples):
            chunk = tuple(audio[position:position+num_samples])
            # Original NumPy context is a view: context[-1] = 0 also clears
            # the final input window's tail before concatenation.
            if position+num_samples == len(audio):
                chunk = chunk[:-context_size_samples]+(0.0,)*context_size_samples
            prediction, hidden, cell = self.step(context+chunk, hidden, cell)
            probabilities.append(prediction)
            context = chunk[-context_size_samples:]
        return probabilities
