"""Original OpenJTalk text2mecab byte conversion and bounded output contract."""
from .normalization_tables import CONVERSIONS

_GROUPS = {}
for _before, _after in CONVERSIONS:
    _encoded = _before.encode('utf-8')
    _GROUPS.setdefault(_encoded[0], []).append((_encoded, _after.encode('utf-8')))
del _before, _after, _encoded


def _variation_length(data, cursor):
    if data[cursor:cursor + 2] == b'\xef\xb8' and cursor + 2 < len(data) and 0x80 <= data[cursor + 2] <= 0x8f:
        return 3
    if data[cursor:cursor + 2] != b'\xf3\xa0' or cursor + 3 >= len(data):
        return 0
    third, fourth = data[cursor + 2:cursor + 4]
    if 0x84 <= third <= 0x86 and 0x80 <= fourth <= 0xbf or third == 0x87 and 0x80 <= fourth <= 0xaf:
        return 4
    return 0


def text_to_mecab(data, size=8192):
    """Return original result code and C-string bytes, without a native call.

    Input and output sizes are bytes. Callers with Python text strip embedded
    NULs before this low-level boundary, as the original Cython wrapper does.
    A truncated multibyte input has no defined source read; reject it instead.
    """
    if data is None or size <= 0:
        return 1, b''
    data = bytes(data).split(b'\0', 1)[0]
    output = bytearray()
    cursor = 0
    while cursor < len(data):
        value = data[cursor]
        for before, after in _GROUPS.get(value, ()):
            if data.startswith(before, cursor):
                cursor += len(before)
                break
        else:
            if value < 128:
                after = '　'.encode() if value in (9, 10, 13) else b''
                cursor += 1
            else:
                skip = _variation_length(data, cursor) if output else 0
                if skip:
                    cursor += skip
                    continue
                width = 2 if 0xc0 <= value <= 0xdf else 3 if 0xe0 <= value <= 0xef else 4 if 0xf0 <= value <= 0xf7 else 0
                if width > len(data) - cursor:
                    raise ValueError('Truncated OpenJTalk UTF-8 byte sequence')
                after = data[cursor:cursor + width]
                cursor += max(width, 1)
        if len(output) + len(after) >= size:
            return 2, b''
        output.extend(after)
    return 0, bytes(output)


def normalize_text(text, size=8192):
    """Apply the original OpenJTalk Python-input normalization boundary."""
    if isinstance(text, str):
        text = text.encode('utf-8')
    elif not isinstance(text, (bytes, bytearray)):
        raise TypeError('Input text must be str, bytes, or bytearray')
    status, result = text_to_mecab(bytes(text).replace(b'\0', b''), size)
    if status == 2:
        raise RuntimeError('Input text is too long')
    if status:
        raise RuntimeError('Invalid arguments for text2mecab')
    return result
