"""Bounded RFC 9420 TLS presentation syntax used by Discord's MLS suite."""

MAX_BYTES = 16 * 1024 * 1024


def uint(value, size):
    if type(value) is not int or not 0 <= value < 1 << (size * 8):
        raise ValueError("MLS integer outside its wire range")
    return value.to_bytes(size, "big")


def varint(value):
    if type(value) is not int or not 0 <= value < 1 << 30:
        raise ValueError("MLS variable integer outside its wire range")
    if value < 64:
        return uint(value, 1)
    if value < 16384:
        return uint(value | 0x4000, 2)
    return uint(value | 0x80000000, 4)


def opaque(data):
    data = bytes(data)
    if len(data) > MAX_BYTES:
        raise ValueError("MLS vector exceeds the message bound")
    return varint(len(data)) + data


class Reader:
    def __init__(self, data):
        self.data = memoryview(data)
        if len(self.data) > MAX_BYTES:
            raise ValueError("MLS message exceeds the message bound")
        self.offset = 0

    def __bool__(self):
        return self.offset < len(self.data)

    def take(self, size):
        if type(size) is not int or size < 0 or size > len(self.data) - self.offset:
            raise ValueError("Truncated MLS message")
        start = self.offset
        self.offset += size
        return bytes(self.data[start:self.offset])

    def uint(self, size):
        return int.from_bytes(self.take(size), "big")

    def varint(self):
        first = self.uint(1)
        prefix = first >> 6
        if prefix == 3:
            raise ValueError("MLS variable integers cannot use an eight-byte encoding")
        size = 1 << prefix
        value = int.from_bytes(bytes([first & 63]) + self.take(size - 1), "big")
        if (size == 2 and value < 64) or (size == 4 and value < 16384):
            raise ValueError("Non-minimal MLS variable integer")
        return value

    def opaque(self):
        return self.take(self.varint())

    def vector(self):
        return Reader(self.opaque())

    def optional(self, decode):
        present = self.uint(1)
        if present not in (0, 1):
            raise ValueError("Invalid MLS optional discriminant")
        return decode(self) if present else None

    def end(self):
        if self:
            raise ValueError("Trailing bytes after MLS message")
