"""Bounded, lazy reader for the pinned CTranslate2 Whisper model format.

This is a model-data reader, not a CTranslate2 module or inference substitute.
The v2-v6 layout and type IDs follow CTranslate2 4.8.1 model_spec.py/model.cc
(MIT). Quantized weights retain their stored integers and scales; dequantizing
them does not claim equivalence to an int8 activation inference backend.
"""

from array import array
from dataclasses import dataclass
import math
import mmap
from pathlib import Path
import struct


_TYPES = (("float32", "f", 4), ("int8", "b", 1), ("int16", "h", 2),
          ("int32", "i", 4), ("float16", "e", 2), ("bfloat16", "H", 2))
_MAX_BYTES = 16 * 1024**3
_MAX_READ = 2**22


@dataclass(frozen=True)
class Weight:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    byte_size: int
    _model: object
    _type_id: int

    @property
    def size(self):
        return math.prod(self.shape)

    def read(self, start=0, count=None):
        """Decode a bounded contiguous run; callers need not expand a model."""
        if count is None:
            count = self.size - start
        if (isinstance(start, bool) or not isinstance(start, int)
                or isinstance(count, bool) or not isinstance(count, int)
                or start < 0 or count < 0 or start + count > self.size or count > _MAX_READ):
            raise ValueError("Whisper weight read exceeds the tensor or read bound")
        self._model._require_open()
        _, code, width = _TYPES[self._type_id]
        result = struct.unpack_from("<" + str(count) + code, self._model._storage, self.offset + start * width)
        if self.dtype == "bfloat16":
            return tuple(struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in result)
        return result

    def row(self, index):
        if not self.shape or not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < self.shape[0]:
            raise ValueError("Whisper weight row is outside its first dimension")
        width = math.prod(self.shape[1:])
        return self.read(index * width, width)

    def floats(self, start=0, count=None):
        """Read f32 values, applying the stored per-row/scalar weight scale."""
        values = self.read(start, count)
        if self.dtype not in ("int8", "int16"):
            return tuple(float(value) for value in values)
        scale = self._model.get(self.name + "_scale")
        if scale is None:
            raise ValueError("Quantized Whisper weights have no associated scale")
        scales = scale.read()
        scale_count = self.shape[0] if self.dtype == "int8" and self.shape else 1
        if (scale.dtype not in ("float32", "float16", "bfloat16")
                or len(scales) != scale_count
                or any(not math.isfinite(value) or value <= 0 for value in scales)):
            raise ValueError("Invalid Whisper weight quantization scales")
        row_size = math.prod(self.shape[1:]) if self.shape else 1
        inverse = tuple(array("f", (1.0 / value for value in scales)))
        return tuple(array("f", (value * inverse[0 if len(scales) == 1 else (start + index) // row_size]
                                  for index, value in enumerate(values))))


class ModelWeights:
    """Read one model.bin without loading or executing a native extension."""

    def __init__(self, model, *, max_bytes=_MAX_BYTES):
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or not 1 <= max_bytes <= _MAX_BYTES:
            raise ValueError("Invalid Whisper model byte bound")
        self._closed = False
        self._storage = b""
        self.tensors = {}
        if isinstance(model, (str, Path)):
            with open(model, "rb") as source:
                source.seek(0, 2)
                length = source.tell()
                if length == 0 or length > max_bytes:
                    raise ValueError("Whisper model exceeds its file bound")
                self._storage = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        elif isinstance(model, bytes) and 0 < len(model) <= max_bytes:
            self._storage = model
        else:
            raise ValueError("Whisper model must be bounded bytes or a local model.bin path")
        try:
            self._parse()
        except BaseException:
            self.close()
            raise

    def _require_open(self):
        if self._closed:
            raise ValueError("Whisper model weights are closed")

    def _take(self, code):
        width = struct.calcsize("<" + code)
        if self._cursor + width > len(self._storage):
            raise ValueError("Truncated CTranslate2 model header")
        value = struct.unpack_from("<" + code, self._storage, self._cursor)[0]
        self._cursor += width
        return value

    def _string(self):
        length = self._take("H")
        if length < 2 or self._cursor + length > len(self._storage):
            raise ValueError("Invalid CTranslate2 model string extent")
        value = self._storage[self._cursor:self._cursor + length]
        self._cursor += length
        if value[-1:] != b"\0" or b"\0" in value[:-1]:
            raise ValueError("Invalid CTranslate2 model string terminator")
        try:
            return value[:-1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Invalid CTranslate2 model string encoding") from exc

    def _parse(self):
        self._cursor = 0
        self.binary_version = self._take("I")
        if not 2 <= self.binary_version <= 6:
            raise ValueError("Whisper model requires CTranslate2 binary version 2 through 6")
        self.specification, self.revision = self._string(), self._take("I")
        if self.specification != "WhisperSpec" or not 1 <= self.revision <= 3:
            raise ValueError("Expected a supported CTranslate2 Whisper specification")
        count = self._take("I")
        if count > 65536:
            raise ValueError("Whisper model exceeds its tensor count bound")
        for _ in range(count):
            name = self._string()
            if name in self.tensors:
                raise ValueError("Duplicate CTranslate2 tensor name")
            rank = self._take("B")
            if rank > 8:
                raise ValueError("Whisper weight exceeds its rank bound")
            shape = tuple(self._take("I") for _ in range(rank))
            size = math.prod(shape)
            type_id = self._take("B")
            if self.binary_version < 4:
                try:
                    type_id = {1: 1, 2: 2, 4: 0}[type_id]
                except KeyError as exc:
                    raise ValueError("Invalid legacy CTranslate2 item size") from exc
            if type_id >= len(_TYPES) or size > 2**31:
                raise ValueError("Invalid CTranslate2 weight type or element count")
            dtype, _, width = _TYPES[type_id]
            byte_size = self._take("I")
            if self.binary_version < 4:
                byte_size *= width
            if byte_size != size * width or self._cursor + byte_size > len(self._storage):
                raise ValueError("Invalid CTranslate2 weight payload extent")
            self.tensors[name] = Weight(name, shape, dtype, self._cursor, byte_size, self, type_id)
            self._cursor += byte_size
        self.aliases = {}
        count = self._take("I") if self.binary_version >= 3 else 0
        if count > 65536:
            raise ValueError("Whisper model exceeds its alias count bound")
        for _ in range(count):
            alias, target = self._string(), self._string()
            if alias in self.tensors or alias in self.aliases:
                raise ValueError("Duplicate CTranslate2 weight alias")
            # Native Model::register_variable_alias ignores missing targets;
            # optional quantization scale/zero aliases follow the same rule.
            self.aliases[alias] = target
            for suffix in ("", "_scale", "_zero"):
                value = self.tensors.get(target + suffix)
                if value is not None:
                    self.tensors[alias + suffix] = value
        if self._cursor != len(self._storage):
            raise ValueError("Trailing bytes after the CTranslate2 Whisper model")

    def __contains__(self, name):
        self._require_open()
        return name in self.tensors

    def __getitem__(self, name):
        self._require_open()
        return self.tensors[name]

    def get(self, name, default=None):
        self._require_open()
        return self.tensors.get(name, default)

    def close(self):
        if not self._closed:
            self._closed = True
            if isinstance(self._storage, mmap.mmap):
                self._storage.close()
            self._storage = b""

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()
