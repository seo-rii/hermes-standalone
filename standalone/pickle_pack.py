"""Losslessly delta-code memo operands in one bounded protocol-3 pickle.

The builder pins the resource identity; the bootstrap verifies restored original
bytes before publishing them. Parsing here checks opcode framing only and never
loads a pickle, resolves a GLOBAL or executes a callable.
"""

_pickle_max_bytes = 32 << 20


def transform_pickle_memo(data, *, inverse=False):
    """Preserve every byte except reversible LONG_BINPUT uint32 operands."""
    if not 3 <= len(data) <= _pickle_max_bytes:
        raise ValueError('Invalid packed pickle resource size')
    if data[:2] != b'\x80\x03':
        raise ValueError('Expected a protocol-3 pickle resource')
    # A warm validated cache never needs to import the pickle tooling.
    import pickletools

    output = bytearray(data)
    previous = 0
    stopped = False
    try:
        for opcode, argument, position in pickletools.genops(data):
            name = opcode.name
            if opcode.proto > 3 or (name == 'PROTO' and position != 0):
                raise ValueError('Unexpected packed pickle protocol framing')
            if name == 'STOP':
                if position + 1 != len(data):
                    raise ValueError('Trailing data after packed pickle STOP')
                stopped = True
            elif name == 'LONG_BINPUT':
                value = ((previous + argument) if inverse else (argument - previous)) & 0xffffffff
                output[position + 1:position + 5] = value.to_bytes(4, 'little')
                previous = value if inverse else argument
    except (ValueError, OverflowError) as error:
        raise ValueError('Invalid protocol-3 packed pickle framing') from error
    if not stopped:
        raise ValueError('Missing packed pickle STOP')
    return bytes(output)
