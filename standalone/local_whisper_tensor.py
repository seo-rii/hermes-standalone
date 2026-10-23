"""Stdlib Whisper convolution, normalization, attention and quantized tensor operations."""
from array import array
from dataclasses import dataclass
import math
import re
import struct

from . import local_whisper_float

_FLOAT = struct.Struct("<f")
_dot = getattr(math, "sumprod", lambda a, b: sum(x * y for x, y in zip(a, b, strict=True)))


def f32(value):
    try:
        return _FLOAT.unpack(_FLOAT.pack(value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def row(values):
    return tuple(array("f", values))


def gelu(value):
    erf = f32(math.erf(f32(value * 0.7071067690849304)))
    return f32(f32(value * f32(1.0 + erf)) * 0.5)


def softmax(values):
    peak = max(values)
    exponentials = row(math.exp(f32(value - peak)) for value in values)
    total = f32(sum(exponentials))
    return row(value / total for value in exponentials)


@dataclass(frozen=True)
class EncodedAudio:
    rows: tuple[tuple[float, ...], ...]

    @property
    def shape(self):
        return 1, len(self.rows), len(self.rows[0]) if self.rows else 0


class WhisperMath:
    def __init__(self, weights, config=None, compute_type="float32", *, int8_saturating_pairs=False):
        self.weights, self.config = weights, config or {}
        if compute_type not in ("float32", "int8", "int8_float32", "auto"):
            raise ValueError("Whisper tensor compute type must be float32 or int8_float32")
        self.compute_type = "float32" if compute_type == "float32" else "int8_float32"
        self.int8_saturating_pairs = bool(int8_saturating_pairs)
        self._matrices = {}
        self._quantized = {}
        self._cached_cells = 0
        self._half_values = {}

    def attribute(self, name, default):
        weight = self.weights.get(name)
        if weight is None:
            return default
        if weight.size != 1:
            raise ValueError("Whisper model attribute must be a scalar")
        return weight.read()[0]

    def layer_count(self, scope):
        indices = {int(match.group(1)) for name in self.weights.tensors
                   if (match := re.match(re.escape(scope) + r"/layer_(\d+)/", name))}
        if not indices or len(indices) > 64 or indices != set(range(len(indices))):
            raise ValueError("Whisper model requires a contiguous layer stack")
        return len(indices)

    def activation(self, value):
        return local_whisper_float.gelu(value) if self.int8_saturating_pairs else gelu(value)

    def matrix(self, name):
        weight = self.weights[name]
        if len(weight.shape) not in (2, 3) or not all(weight.shape):
            raise ValueError("Invalid Whisper matrix/kernel shape")
        if name in self._matrices:
            return self._matrices[name]
        width = math.prod(weight.shape[1:])
        compact = weight.size > 2_000_000
        if weight.dtype == "float16":
            # Half weights have only 65,536 possible bit patterns. Shared
            # Python float objects retain exact values while tuple iteration
            # avoids allocating a new float for every sumprod multiplication.
            rows = []
            for index in range(weight.shape[0]):
                values = []
                for value in weight.floats(index * width, width):
                    key = value if value else (b"-0" if math.copysign(1.0, value) < 0 else b"+0")
                    values.append(self._half_values.setdefault(key, value))
                rows.append(tuple(values))
            rows = tuple(rows)
            compact = True
        else:
            rows = tuple(array("f", weight.floats(index * width, width)) if compact else weight.floats(index * width, width)
                         for index in range(weight.shape[0]))
        if self._cached_cells + weight.size <= (64_000_000 if compact else 8_000_000):
            self._matrices[name] = rows
            self._cached_cells += weight.size
        return rows

    def dense(self, inputs, scope, *, residual=None, activation=False):
        if self.compute_type == "float32":
            kernels = self.matrix(scope + "/weight")
            scales = None
        else:
            kernels, scales, saturation = self.quantized_matrix(scope + "/weight")
        bias = self.weights.get(scope + "/bias")
        bias = bias.read() if bias is not None else (0.0,) * len(kernels)
        if len(bias) != len(kernels) or any(len(source) != len(kernels[0]) for source in inputs):
            raise ValueError("Invalid Whisper dense input/bias shape")
        if residual is not None and (len(residual) != len(inputs) or any(len(value) != len(kernels) for value in residual)):
            raise ValueError("Invalid Whisper residual shape")
        outputs = []
        for position, source in enumerate(inputs):
            if scales is not None:
                source, input_scale = self.quantize(source, shift=True)
                inverse = f32(1.0 / input_scale)
                correction = 128 * sum(source)
            values = []
            for index, kernel in enumerate(kernels):
                product = _dot(source, kernel)
                if scales is not None:
                    product -= correction
                    if saturation[index]:
                        for column, first, second in saturation[index]:
                            pair = (source[column] + 128) * first + (source[column + 1] + 128) * second
                            product += max(-32768, min(32767, pair)) - pair
                    product = f32(f32(float(product)) * inverse)
                    product = f32(product / scales[index])
                value = f32(product + bias[index])
                if residual is not None:
                    value = f32(value + residual[position][index])
                values.append(self.activation(value) if activation else value)
            outputs.append(tuple(values))
        return tuple(outputs)

    def quantize(self, values, *, shift=False):
        maximum = max(map(abs, values))
        scale = f32(127.0 / maximum) if maximum else 1.0
        converted = []
        for value in values:
            if shift and self.int8_saturating_pairs:
                value = local_whisper_float.fma32(value, scale, 128.0)
            else:
                value = f32(value * scale)
                if shift:
                    value = f32(value + 128.0)
            value = round(value) if self.weights.binary_version >= 5 else int(value)
            converted.append(max(-128, min(127, value - 128 if shift else value)))
        return tuple(converted), scale

    def quantized_matrix(self, name):
        name = self.weights[name].name
        if name in self._quantized:
            return self._quantized[name]
        weight = self.weights[name]
        if len(weight.shape) != 2:
            raise ValueError("Whisper quantized dense weights must be matrices")
        width = weight.shape[1]
        if weight.dtype == "int8":
            kernels = tuple(bytes(value + 128 for value in weight.row(index)) for index in range(weight.shape[0]))
            scales = self.weights[name + "_scale"].read()
        else:
            kernels, scales = [], []
            for index in range(weight.shape[0]):
                quantized, scale = self.quantize(weight.floats(index * width, width))
                kernels.append(bytes(item + 128 for item in quantized))
                scales.append(scale)
            kernels, scales = tuple(kernels), tuple(scales)
        if len(scales) != len(kernels):
            raise ValueError("Whisper quantized dense weights have invalid per-row scales")
        saturation = tuple(tuple((index, kernel[index] - 128, kernel[index + 1] - 128)
                                 for index in range(0, len(kernel) - 1, 2)
                                 if (kernel[index] - 128) * (kernel[index + 1] - 128) > 0
                                 and abs(kernel[index] - 128) + abs(kernel[index + 1] - 128) > 128)
                           if self.int8_saturating_pairs else () for kernel in kernels)
        result = kernels, scales, saturation
        if self._cached_cells + weight.size <= 64_000_000:
            self._quantized[name] = result
            self._cached_cells += weight.size
        return result

    def embeddings(self, tokens, scope):
        weight = self.weights[scope + "/weight"]
        if len(weight.shape) != 2:
            raise ValueError("Whisper embeddings must be a matrix")
        width = weight.shape[1]
        if any(not 0 <= token < weight.shape[0] for token in tokens):
            raise ValueError("Whisper token is outside its embedding vocabulary")
        if self.compute_type == "float32":
            return tuple(weight.floats(token * width, width) for token in tokens)
        kernels, scales, _ = self.quantized_matrix(scope + "/weight")
        return tuple(row((value - 128) * f32(1.0 / scales[token]) for value in kernels[token]) for token in tokens)

    def normalize(self, inputs, scope):
        gamma = self.weights[scope + "/gamma"].read()
        beta = self.weights[scope + "/beta"].read()
        configured = self.config.get("layer_norm_epsilon")
        epsilon = f32(1e-5 if configured is None else float(configured))
        if epsilon < 0 or not math.isfinite(epsilon) or len(gamma) != len(beta):
            raise ValueError("Invalid Whisper layer normalization parameters")
        outputs = []
        width = len(gamma)
        for source in inputs:
            if len(source) != width or not width:
                raise ValueError("Invalid Whisper layer normalization shape")
            if self.int8_saturating_pairs:
                outputs.append(local_whisper_float.normalize(source, gamma, beta, epsilon))
                continue
            total, squares = 0.0, 0.0
            for value in source:
                total = f32(total + value)
                squares = f32(squares + f32(value * value))
            mean = f32(total / width)
            variance = max(f32(f32(squares / width) - f32(mean * mean)), 0.0)
            denominator = f32(math.sqrt(f32(variance + epsilon)))
            inverse = f32(1.0 / denominator) if denominator else math.inf
            outputs.append(tuple(f32(f32(f32(value - mean) * inverse) * gain + offset)
                                 for value, gain, offset in zip(source, gamma, beta, strict=True)))
        return tuple(outputs)

    def conv(self, channels, scope, stride):
        kernels = self.matrix(scope + "/weight")
        shape = self.weights[scope + "/weight"].shape
        if len(shape) != 3 or shape[1] != len(channels) or shape[2] != 3:
            raise ValueError("Whisper convolution requires [outputs, channels, 3] kernels")
        bias = self.weights.get(scope + "/bias")
        bias = bias.read() if bias is not None else (0.0,) * len(kernels)
        length = len(channels[0])
        if not length or any(len(channel) != length for channel in channels) or len(bias) != len(kernels):
            raise ValueError("Invalid Whisper convolution input/bias shape")
        sources = tuple(tuple(channel[position + delta] if 0 <= position + delta < length else 0.0
                              for channel in channels for delta in (-1, 0, 1))
                        for position in range(0, length, stride))
        return tuple(tuple(self.activation(f32(_dot(source, kernel) + offset)) for source in sources)
                     for kernel, offset in zip(kernels, bias, strict=True))

    def attention(self, inputs, scope, heads, *, memory=None, cache=None, causal=False, return_scores=False):
        fused = self.dense(self.normalize(inputs, scope + "/layer_norm"), scope + "/linear_0")
        width = len(inputs[0])
        if not 1 <= heads <= width or width % heads:
            raise ValueError("Whisper attention heads must divide its channel count")
        depth = width // heads
        if memory is None:
            if any(len(value) != width * 3 for value in fused):
                raise ValueError("Whisper self-attention requires fused QKV")
            queries = tuple(value[:width] for value in fused)
            keys = tuple(value[width:width * 2] for value in fused)
            values = tuple(value[width * 2:] for value in fused)
            output_layer = scope + "/linear_1"
            if cache is not None:
                keys, values = cache.get("keys", ()) + keys, cache.get("values", ()) + values
                cache["keys"], cache["values"] = keys, values
        else:
            queries = fused
            if cache is not None and "keys" in cache:
                keys, values = cache["keys"], cache["values"]
            else:
                projected = self.dense(memory, scope + "/linear_1")
                if any(len(value) != width * 2 for value in projected):
                    raise ValueError("Whisper cross-attention requires fused KV")
                keys, values = tuple(value[:width] for value in projected), tuple(value[width:] for value in projected)
                if cache is not None:
                    cache["keys"], cache["values"] = keys, values
            output_layer = scope + "/linear_2"
        scale = f32(self.attribute(scope + "/queries_scale", 1.0 / math.sqrt(depth)))
        outputs = [[] for _ in inputs]
        attention = [[] for _ in inputs] if return_scores else []
        offset = len(keys) - len(queries)
        for head in range(heads):
            start, end = head * depth, (head + 1) * depth
            head_keys = tuple(value[start:end] for value in keys)
            columns = tuple(tuple(value[index] for value in values) for index in range(start, end))
            for position, query in enumerate(queries):
                count = offset + position + 1 if causal else len(keys)
                query = query[start:end]
                logits = row(_dot(query, key) * scale for key in head_keys[:count])
                probabilities = local_whisper_float.softmax(logits) if self.int8_saturating_pairs else softmax(logits)
                outputs[position].extend(f32(_dot(probabilities, column[:count])) for column in columns)
                if return_scores:
                    attention[position].append(logits)
        output = self.dense(tuple(tuple(value) for value in outputs), output_layer, residual=inputs)
        return output, tuple(tuple(value) for value in attention)

    def feedforward(self, inputs, scope):
        hidden = self.dense(self.normalize(inputs, scope + "/layer_norm"), scope + "/linear_0", activation=True)
        return self.dense(hidden, scope + "/linear_1", residual=inputs)

    def close(self):
        self._matrices.clear()
        self._half_values.clear()
        self._quantized.clear()
        self._cached_cells = 0
