"""Bounded float32 TFLite inference for Hermes's existing wake-word models.

The FlatBuffer layout follows TensorFlow v2.19.0's Apache-2.0 schema; this is
an explicit interpreter for the operators used by the pinned openWakeWord
models, not an import replacement for TensorFlow Lite, NumPy or an SDK.
Weights and model graphs are read unchanged. Dot products use stdlib
``math.sumprod`` where available, with float32 rounding at tensor boundaries.
Native delegate/FMA accumulation order can differ; validated score tolerances
are recorded separately from the fixed-point SILK codec's exact-byte claims.
"""

from array import array
import itertools
import math
from pathlib import Path
import struct
import sys


MAX_MODEL = 32 * 1024 * 1024
MAX_ELEMENTS = 2 * 1024 * 1024
_SUPPORTED = {0, 3, 9, 14, 17, 18, 22, 34, 39, 40, 41, 43, 55, 57, 70, 73, 76, 82, 98, 126}
_OPTION_TAGS = {0: 11, 3: 1, 9: 8, 17: 5, 18: 21, 22: 17, 34: 22,
                39: 26, 40: 27, 41: 28, 43: 30, 55: 39, 57: 39, 70: 52,
                82: 27, 98: 75, 126: 101}
_sumprod = getattr(math, "sumprod", lambda a, b: sum(x * y for x, y in zip(a, b, strict=True)))


def _size(shape):
    if len(shape) > 8 or any(type(value) is not int or value < 0 for value in shape):
        raise ValueError("Invalid wake tensor dimensions")
    result = math.prod(shape)
    if result > MAX_ELEMENTS:
        raise ValueError("Wake tensor exceeds the element limit")
    return result


def _strides(shape):
    result, stride = [], 1
    for value in reversed(shape):
        result.append(stride)
        stride *= value
    return tuple(reversed(result))


class _FlatBuffer:
    def __init__(self, data):
        if not isinstance(data, bytes) or not 8 <= len(data) <= MAX_MODEL or data[4:8] != b"TFL3":
            raise ValueError("Invalid or oversized TFLite model")
        self.data = data

    def number(self, kind, position):
        size = struct.calcsize("<" + kind)
        if not 0 <= position <= len(self.data) - size:
            raise ValueError("TFLite offset is outside its model")
        return struct.unpack_from("<" + kind, self.data, position)[0]

    def field(self, table, index):
        vtable = table - self.number("i", table)
        extent, object_size = self.number("H", vtable), self.number("H", vtable + 2)
        if (extent < 4 or extent % 2 or object_size < 4 or vtable + extent > len(self.data)
                or not 0 <= table <= len(self.data) - object_size):
            raise ValueError("Invalid TFLite table extent")
        position = vtable + 4 + 2 * index
        if position >= vtable + extent:
            return None
        offset = self.number("H", position)
        if offset and not 4 <= offset < object_size:
            raise ValueError("Invalid TFLite table field")
        return table + offset if offset else None

    def value(self, table, index, kind="I", default=0):
        position = self.field(table, index)
        return default if position is None else self.number(kind, position)

    def indirect(self, position):
        return position + self.number("I", position)

    def table(self, table, index):
        position = self.field(table, index)
        return None if position is None else self.indirect(position)

    def vector(self, table, index, kind="i", *, tables=False):
        position = self.table(table, index)
        if position is None:
            return ()
        count, stride = self.number("I", position), struct.calcsize("<" + kind)
        if count > MAX_ELEMENTS or position + 4 + count * stride > len(self.data):
            raise ValueError("Invalid TFLite vector extent")
        start = position + 4
        return tuple(self.indirect(start + item * stride) if tables else self.number(kind, start + item * stride)
                     for item in range(count))

    def bytes(self, table, index):
        position = self.table(table, index)
        if position is None:
            return b""
        length = self.number("I", position)
        if length > MAX_MODEL or position + 4 + length > len(self.data):
            raise ValueError("Invalid TFLite byte vector")
        return self.data[position + 4:position + 4 + length]


class Tensor:
    def __init__(self, shape, values, kind=0):
        self.shape, self.kind = tuple(shape), kind
        expected = _size(self.shape)
        values = tuple(itertools.islice(values, expected + 1))
        if len(values) != expected or kind not in (0, 2):
            raise ValueError("Wake tensor shape/type does not match its data")
        if kind == 0:
            self.values = tuple(array("f", values))
        else:
            if any(type(value) is not int or not -2**31 <= value < 2**31 for value in values):
                raise ValueError("Invalid wake integer tensor")
            self.values = values

    @classmethod
    def _from_parts(cls, shape, prefix, tail):
        """Reuse already rounded immutable values; round only newly computed rows."""
        count = _size(shape)
        if len(prefix) > count:
            raise ValueError("Wake tensor prefix exceeds its shape")
        result = cls((count - len(prefix),), tail)
        result.shape, result.values = tuple(shape), prefix + result.values
        return result


def _activation(value, kind):
    if kind == 0:
        return value
    if kind == 1:
        return max(0.0, value)
    if kind == 2:
        return max(-1.0, min(1.0, value))
    if kind == 3:
        return max(0.0, min(6.0, value))
    if kind == 4:
        return math.tanh(value)
    raise ValueError("Unsupported TFLite fused activation")


def _binary(left, right, operation, activation=0, prefix=()):
    rank = max(len(left.shape), len(right.shape))
    ashape, bshape = (1,) * (rank - len(left.shape)) + left.shape, (1,) * (rank - len(right.shape)) + right.shape
    if any(a != b and a != 1 and b != 1 for a, b in zip(ashape, bshape)):
        raise ValueError("Incompatible wake tensor broadcasting")
    shape = tuple(max(a, b) for a, b in zip(ashape, bshape))
    size = _size(shape)
    if shape != left.shape:
        prefix = ()
    start = len(prefix)
    a, b = left.values, right.values
    if len(a) == size and len(b) == size:
        values = (operation(x, y) for x, y in zip(a[start:], b[start:]))
    elif len(a) == 1:
        values = (operation(a[0], value) for value in itertools.islice(itertools.cycle(b), start, size))
    elif len(b) == 1:
        values = (operation(value, b[0]) for value in itertools.islice(itertools.cycle(a), start, size))
    elif ashape[-1] == bshape[-1] and all(value == 1 for value in bshape[:-1]):
        values = (operation(a[index], b[index % len(b)]) for index in range(start, size))
    else:
        out_strides, a_strides, b_strides = _strides(shape), _strides(ashape), _strides(bshape)
        def coordinates():
            for index in range(start, size):
                aindex = bindex = 0
                for axis in range(rank):
                    coordinate = (index // out_strides[axis]) % shape[axis]
                    if ashape[axis] != 1:
                        aindex += coordinate * a_strides[axis]
                    if bshape[axis] != 1:
                        bindex += coordinate * b_strides[axis]
                yield operation(a[aindex], b[bindex])
        values = coordinates()
    if activation:
        values = (_activation(value, activation) for value in values)
    return Tensor._from_parts(shape, prefix, values)


def _padding(size, kernel, stride, mode):
    if stride <= 0 or kernel <= 0 or mode not in (0, 1):
        raise ValueError("Invalid wake convolution/pooling options")
    if mode == 1:
        output = max(0, (size - kernel) // stride + 1)
        return output, 0
    output = (size + stride - 1) // stride
    return output, max(0, (output - 1) * stride + kernel - size) // 2


class TFLiteModel:
    """A validated single-subgraph float model with no implicit I/O or downloads."""

    def __init__(self, data):
        self.reader = reader = _FlatBuffer(data)
        model = reader.number("I", 0)
        if reader.value(model, 0) != 3:
            raise ValueError("Wake TFLite model must use schema version 3")
        graphs = reader.vector(model, 2, tables=True)
        if len(graphs) != 1:
            raise ValueError("Wake inference requires one complete TFLite subgraph")
        graph, self.constants, self.shapes = graphs[0], {}, []
        self.inputs, self.outputs = reader.vector(graph, 1), reader.vector(graph, 2)
        buffers = reader.vector(model, 4, tables=True)
        tensors = reader.vector(graph, 0, tables=True)
        if not 1 <= len(tensors) <= 4096 or len(self.inputs) != 1 or len(self.outputs) != 1:
            raise ValueError("Invalid wake TFLite graph size or boundary")
        if any(not 0 <= index < len(tensors) for index in (*self.inputs, *self.outputs)):
            raise ValueError("Invalid wake TFLite boundary tensor")
        for index, table in enumerate(tensors):
            shape, kind, buffer = reader.vector(table, 0), reader.value(table, 1, "b"), reader.value(table, 2)
            if (kind not in (0, 2) or not 0 <= buffer < len(buffers) or reader.value(table, 5, "?")
                    or reader.table(table, 6) is not None):
                raise ValueError("Wake model requires fixed float32/int32 tensors")
            _size(shape)
            self.shapes.append(tuple(shape))
            content = reader.bytes(buffers[buffer], 0)
            if content:
                count = _size(shape)
                if len(content) != 4 * count:
                    raise ValueError("Invalid wake model constant size")
                self.constants[index] = Tensor(shape, struct.unpack("<" + ("f" if kind == 0 else "i") * count, content), kind)
        opcodes = []
        for table in reader.vector(model, 1, tables=True):
            code = max(reader.value(table, 0, "b"), reader.value(table, 3, "i"))
            if code not in _SUPPORTED or reader.value(table, 2, "i", 1) != 1 or reader.bytes(table, 1):
                raise ValueError(f"Unsupported wake TFLite operator/version: {code}")
            opcodes.append(code)
        self.operators = []
        available = set(self.constants) | set(self.inputs)
        for table in reader.vector(graph, 3, tables=True):
            opcode, inputs, outputs = reader.value(table, 0), reader.vector(table, 1), reader.vector(table, 2)
            if not 0 <= opcode < len(opcodes) or len(outputs) != 1 or len(inputs) > 8:
                raise ValueError("Invalid wake TFLite operator boundary")
            if any(value != -1 and value not in available for value in inputs):
                raise ValueError("Wake TFLite graph is not topologically complete")
            if any(not 0 <= value < len(tensors) or value in available for value in outputs):
                raise ValueError("Invalid wake TFLite tensor producer")
            tag, options = reader.value(table, 3, "B"), reader.table(table, 4)
            if tag not in (0, _OPTION_TAGS.get(opcodes[opcode], 0)) or bool(tag) != (options is not None):
                raise ValueError("Wake TFLite option type does not match its operator")
            available.update(outputs)
            self.operators.append((opcodes[opcode], inputs, outputs[0], options))
        if len(self.operators) > 4096 or any(value not in available for value in self.outputs):
            raise ValueError("Incomplete or oversized wake TFLite graph")
        self._weights, self._last_input, self._last_output = {}, None, None
        self._previous_nodes = {}

    @classmethod
    def from_file(cls, path):
        with Path(path).open("rb") as stream:
            data = stream.read(MAX_MODEL + 1)
        return cls(data)

    @property
    def input_shape(self):
        return self.shapes[self.inputs[0]]

    def invoke(self, values, shape=None, *, temporal_shift=None):
        source = Tensor(self.input_shape if shape is None else shape, values)
        if any(not math.isfinite(value) for value in source.values):
            raise ValueError("Wake model input must contain finite float32 samples")
        identity = source.shape, source.values
        if identity == self._last_input:
            return self._last_output
        tensors = dict(self.constants)
        tensors[self.inputs[0]] = source
        shift = None
        if (type(temporal_shift) is int and len(source.shape) == 4 and source.shape[0] == 1
                and 0 < temporal_shift < source.shape[1] and self._last_input is not None
                and source.shape == self._last_input[0]):
            stride = source.shape[2] * source.shape[3]
            if self._last_input[1][temporal_shift * stride:] == source.values[:-temporal_shift * stride]:
                shift = temporal_shift
        shifts, previous = {self.inputs[0]: shift}, {}
        for index, (code, inputs, output, options) in enumerate(self.operators):
            arguments = [None if value == -1 else tensors[value] for value in inputs]
            delta = shifts.get(inputs[0])
            if (code in (0, 18, 41, 55, 57) and inputs[1] not in self.constants
                    and shifts.get(inputs[1]) != delta):
                delta = None
            tensors[output] = result = self._execute(index, code, arguments, options, delta)
            previous[index] = result
            if delta is not None and len(result.shape) == 4:
                if (code in (0, 18, 41, 55, 57, 14, 73, 76, 98) and result.shape == arguments[0].shape
                        and (len(inputs) == 1 or inputs[1] in self.constants or shifts.get(inputs[1]) == delta)):
                    shifts[output] = delta
                elif code == 34 and arguments[1].values[2:4] == (0, 0):
                    shifts[output] = delta
                elif code in (3, 17):
                    stride = self._option(options, 2)
                    kernel = ((arguments[1].shape[1] - 1) * self._option(options, 5, default=1) + 1
                              if code == 3 else self._option(options, 4))
                    if stride > 0 and delta % stride == 0:
                        _, top = _padding(arguments[0].shape[1], kernel, stride, self._option(options, 0, "b"))
                        reusable = max(0, (arguments[0].shape[1] - delta - kernel) // stride + 1)
                        if top == 0 and reusable >= result.shape[1] - delta // stride:
                            shifts[output] = delta // stride
        result = tensors[self.outputs[0]]
        self._last_input, self._last_output = identity, result
        self._previous_nodes = previous
        return result

    def _option(self, options, index, kind="i", default=0):
        return default if options is None else self.reader.value(options, index, kind, default)

    def _execute(self, index, code, inputs, options, temporal_shift=None):
        first = inputs[0]
        prefix = ()
        previous = self._previous_nodes.get(index)
        if (temporal_shift is not None and len(first.shape) == 4 and first.shape[0] == 1
                and previous is not None and previous.shape == first.shape):
            row_size = first.shape[2] * first.shape[3]
            prefix = previous.values[temporal_shift * row_size:]
        if code in (0, 18, 41, 55, 57):
            operation = {0: lambda a, b: a + b, 18: lambda a, b: a * b,
                         41: lambda a, b: a - b, 55: max, 57: min}[code]
            return _binary(first, inputs[1], operation, self._option(options, 0, "b") if code in (0, 18, 41) else 0, prefix)
        if code in (14, 73, 76, 98):
            alpha = self._option(options, 0, "f", 0.2)
            operation = {
                14: lambda value: 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value)),
                73: lambda value: math.log(value) if value > 0 else -math.inf if value == 0 else math.nan,
                76: lambda value: 1 / math.sqrt(value) if value > 0 else math.inf if value == 0 else math.nan,
                98: lambda value: value if value >= 0 else alpha * value,
            }[code]
            return Tensor._from_parts(first.shape, prefix, map(operation, first.values[len(prefix):]))
        if code in (22, 43, 70):
            if code == 22:
                shape = list(inputs[1].values) if len(inputs) > 1 else list(self.reader.vector(options, 0))
                if shape.count(-1) > 1 or any(value < -1 for value in shape):
                    raise ValueError("Invalid wake reshape")
                if -1 in shape:
                    product = math.prod(value for value in shape if value != -1)
                    if not product or len(first.values) % product:
                        raise ValueError("Wake reshape does not preserve elements")
                    shape[shape.index(-1)] = len(first.values) // product
            elif code == 70:
                axis = inputs[1].values[0]
                axis = axis + len(first.shape) + 1 if axis < 0 else axis
                if not 0 <= axis <= len(first.shape):
                    raise ValueError("Invalid wake expand axis")
                shape = [*first.shape[:axis], 1, *first.shape[axis:]]
            else:
                axes = self.reader.vector(options, 0) if options is not None else ()
                selected = {axis % len(first.shape) for axis in axes} if axes else {i for i, size in enumerate(first.shape) if size == 1}
                if any(first.shape[axis] != 1 for axis in selected):
                    raise ValueError("Wake squeeze axis is not a singleton")
                shape = [size for axis, size in enumerate(first.shape) if axis not in selected]
            return Tensor(shape, first.values, first.kind)
        if code == 39:
            axes = inputs[1].values
            if sorted(axes) != list(range(len(first.shape))):
                raise ValueError("Invalid wake transpose permutation")
            shape, old_strides = tuple(first.shape[axis] for axis in axes), _strides(first.shape)
            values = (first.values[sum(coordinate * old_strides[axis] for axis, coordinate in zip(axes, coords))]
                      for coords in itertools.product(*(range(size) for size in shape)))
            return Tensor(shape, values, first.kind)
        if code in (40, 82):
            return self._reduce(first, inputs[1], code, bool(self._option(options, 0, "?")))
        if code == 9:
            return self._dense(index, inputs, options)
        if code == 3:
            return self._conv(index, inputs, options, temporal_shift)
        if code == 17:
            return self._pool(index, first, options, temporal_shift)
        if code == 34:
            return self._pad(first, inputs[1])
        if code == 126:
            return self._matmul(first, inputs[1], options)
        raise ValueError(f"Wake operator was not admitted: {code}")

    def _dense(self, index, inputs, options):
        first, weights, bias = inputs
        if len(weights.shape) != 2 or self._option(options, 1, "b"):
            raise ValueError("Wake fully-connected weights must use dense rows")
        rows, columns = weights.shape
        if not columns or len(first.values) % columns or bias is not None and len(bias.values) != rows:
            raise ValueError("Invalid wake fully-connected tensor shapes")
        if index not in self._weights:
            self._weights[index] = tuple(weights.values[offset:offset + columns]
                                         for offset in range(0, len(weights.values), columns))
        kernels = self._weights[index]
        activation = self._option(options, 0, "b")
        values = []
        for offset in range(0, len(first.values), columns):
            source = first.values[offset:offset + columns]
            for row in range(rows):
                values.append(_activation(_sumprod(source, kernels[row]) + (bias.values[row] if bias else 0), activation))
        shape = (*first.shape[:-1], rows) if self._option(options, 2, "?") else (len(first.values) // columns, rows)
        return Tensor(shape, values)

    def _reused_prefix(self, index, first, output_shape, kernel, stride, top, shift):
        previous = self._previous_nodes.get(index)
        if (shift is None or shift % stride or first.shape[0] != 1 or top != 0
                or previous is None or previous.shape != output_shape):
            return (), 0
        rows = max(0, min(output_shape[1], (first.shape[1] - shift - kernel) // stride + 1))
        row_size = output_shape[2] * output_shape[3]
        offset = shift // stride * row_size
        return previous.values[offset:offset + rows * row_size], rows

    def _conv(self, index, inputs, options, temporal_shift):
        first, weights, bias = inputs
        if len(first.shape) != 4 or len(weights.shape) != 4:
            raise ValueError("Wake convolution requires NHWC/OHWI tensors")
        batches, height, width, channels = first.shape
        outputs, kh, kw, depth = weights.shape
        if channels != depth or bias is not None and len(bias.values) != outputs:
            raise ValueError("Invalid wake convolution channels")
        sw, sh = self._option(options, 1), self._option(options, 2)
        dw, dh = self._option(options, 4, default=1), self._option(options, 5, default=1)
        if dw < 1 or dh < 1:
            raise ValueError("Invalid wake convolution dilation")
        oh, top = _padding(height, (kh - 1) * dh + 1, sh, self._option(options, 0, "b"))
        ow, left = _padding(width, (kw - 1) * dw + 1, sw, self._option(options, 0, "b"))
        _size((batches, oh, ow, outputs))
        length = kh * kw * channels
        if index not in self._weights:
            self._weights[index] = tuple(weights.values[row * length:(row + 1) * length] for row in range(outputs))
        kernels = self._weights[index]
        shape = (batches, oh, ow, outputs)
        prefix, start_row = self._reused_prefix(index, first, shape, (kh - 1) * dh + 1, sh, top, temporal_shift)
        activation, values = self._option(options, 3, "b"), []
        zeros = (0.0,) * channels
        for batch in range(batches):
            for y in range(start_row, oh):
                for x in range(ow):
                    patch = []
                    for ky in range(kh):
                        iy = y * sh + ky * dh - top
                        for kx in range(kw):
                            ix = x * sw + kx * dw - left
                            if 0 <= iy < height and 0 <= ix < width:
                                offset = ((batch * height + iy) * width + ix) * channels
                                patch.extend(first.values[offset:offset + channels])
                            else:
                                patch.extend(zeros)
                    source = tuple(patch)
                    for row in range(outputs):
                        values.append(_activation(_sumprod(source, kernels[row]) + (bias.values[row] if bias else 0), activation))
        return Tensor._from_parts(shape, prefix, values)

    def _pool(self, index, first, options, temporal_shift):
        if len(first.shape) != 4:
            raise ValueError("Wake pooling requires NHWC tensors")
        batches, height, width, channels = first.shape
        sw, sh, kw, kh = (self._option(options, field) for field in (1, 2, 3, 4))
        oh, top = _padding(height, kh, sh, self._option(options, 0, "b"))
        ow, left = _padding(width, kw, sw, self._option(options, 0, "b"))
        shape = (batches, oh, ow, channels)
        prefix, start_row = self._reused_prefix(index, first, shape, kh, sh, top, temporal_shift)
        activation, values = self._option(options, 5, "b"), []
        for batch in range(batches):
            for y in range(start_row, oh):
                for x in range(ow):
                    maxima = [-math.inf] * channels
                    for ky in range(kh):
                        for kx in range(kw):
                            iy, ix = y * sh + ky - top, x * sw + kx - left
                            if 0 <= iy < height and 0 <= ix < width:
                                offset = ((batch * height + iy) * width + ix) * channels
                                maxima = list(map(max, maxima, first.values[offset:offset + channels]))
                    values.extend(_activation(value, activation) for value in maxima)
        return Tensor._from_parts(shape, prefix, values)

    def _pad(self, first, padding):
        if padding.shape != (len(first.shape), 2) or any(value < 0 for value in padding.values):
            raise ValueError("Invalid wake tensor padding")
        before, after = padding.values[::2], padding.values[1::2]
        shape = tuple(size + low + high for size, low, high in zip(first.shape, before, after))
        values, strides = [0.0] * _size(shape), _strides(shape)
        for index, coordinates in enumerate(itertools.product(*(range(size) for size in first.shape))):
            target = sum((coordinate + low) * stride for coordinate, low, stride in zip(coordinates, before, strides))
            values[target] = first.values[index]
        return Tensor(shape, values, first.kind)

    def _reduce(self, first, axes, code, keep):
        rank = len(first.shape)
        if any(not -rank <= axis < rank for axis in axes.values):
            raise ValueError("Invalid wake reduction axis")
        selected = {axis % rank for axis in axes.values}
        shape = tuple(1 if axis in selected else size for axis, size in enumerate(first.shape) if keep or axis not in selected)
        count, strides = _size(shape), _strides(shape)
        values, totals = ([0.0] if code == 40 else [-math.inf]) * count, [0] * count
        for value, coordinates in zip(first.values, itertools.product(*(range(size) for size in first.shape))):
            target_coords = tuple(0 if axis in selected else coordinate for axis, coordinate in enumerate(coordinates) if keep or axis not in selected)
            target = sum(coordinate * stride for coordinate, stride in zip(target_coords, strides))
            values[target] = values[target] + value if code == 40 else max(values[target], value)
            totals[target] += 1
        if code == 40:
            values = [value / total if total else math.nan for value, total in zip(values, totals)]
        return Tensor(shape, values)

    def _matmul(self, first, second, options):
        if len(first.shape) < 2 or len(second.shape) < 2:
            raise ValueError("Wake batch matrix multiplication requires matrix dimensions")
        ax, ay = bool(self._option(options, 0, "?")), bool(self._option(options, 1, "?"))
        ar, ac = first.shape[-2:]
        br, bc = second.shape[-2:]
        rows, shared, bshared, columns = (ac, ar, bc, br) if ax and ay else (ac, ar, br, bc) if ax else (ar, ac, bc, br) if ay else (ar, ac, br, bc)
        if shared != bshared:
            raise ValueError("Wake matrix inner dimensions differ")
        batches_a, batches_b = math.prod(first.shape[:-2]), math.prod(second.shape[:-2])
        if batches_a != batches_b and min(batches_a, batches_b) != 1:
            raise ValueError("Unsupported wake batch matrix broadcasting")
        shape = (*((first if batches_a >= batches_b else second).shape[:-2]), rows, columns)
        _size(shape)
        values = []
        for batch in range(max(batches_a, batches_b)):
            aoffset, boffset = (batch % batches_a) * ar * ac, (batch % batches_b) * br * bc
            bcolumns = [tuple(second.values[boffset + (column * bc + k if ay else k * bc + column)] for k in range(shared))
                        for column in range(columns)]
            for row in range(rows):
                source = tuple(first.values[aoffset + (k * ac + row if ax else row * ac + k)] for k in range(shared))
                values.extend(_sumprod(source, column) for column in bcolumns)
        return Tensor(shape, values)
