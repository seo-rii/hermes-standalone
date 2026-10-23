"""Losslessly transpose the byte embedding of the reviewed G2PW model.

The builder pins the complete original model's hash. These helpers are also
inlined into the bootstrap, which restores original bytes before resource hash
verification. They do not parse or execute an ONNX model.
"""

_model_g2pw_size = 159287362
_model_embedding_start = 598043
_model_embedding_rows = 21128
_model_embedding_columns = 768
_model_max_tensor_bytes = 32 << 20


def _model_transpose(data, rows, columns, *, inverse=False):
    """Transpose a bounded row-major byte matrix without normalizing values."""
    size = rows * columns
    if (rows <= 0 or columns <= 0 or columns > 4096
            or size > _model_max_tensor_bytes or len(data) != size):
        raise ValueError('Invalid packed model matrix dimensions or size')
    if not inverse:
        return b''.join(data[column::columns] for column in range(columns))
    output = bytearray(size)
    for column in range(columns):
        output[column::columns] = data[column * rows:(column + 1) * rows]
    return bytes(output)


def transform_g2pw_embedding(data, *, inverse=False):
    """Permute only the pinned 21128 by 768 uint8 tensor; preserve file length."""
    end = _model_embedding_start + _model_embedding_rows * _model_embedding_columns
    if len(data) != _model_g2pw_size or end > len(data):
        raise ValueError('Expected reviewed G2PW model size and embedding span')
    packed = _model_transpose(data[_model_embedding_start:end],
                              _model_embedding_rows, _model_embedding_columns,
                              inverse=inverse)
    view = memoryview(data)
    return b''.join((view[:_model_embedding_start], packed, view[end:]))
