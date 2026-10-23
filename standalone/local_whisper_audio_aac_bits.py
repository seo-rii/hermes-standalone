"""Strict MPEG-4 AAC bit reader and error boundary."""


class AacError(ValueError):
    """Malformed AAC data or invalid decoder arguments."""


class UnsupportedAac(AacError):
    """The actual AAC profile/tool has not been ported."""


class Bits:
    __slots__ = ('data', 'integer', 'pos', 'limit')

    def __init__(self, data):
        self.data = data
        self.integer = int.from_bytes(data, 'big')
        self.pos, self.limit = 0, len(data) * 8

    def peek(self, count):
        shift = self.limit - self.pos - count
        if shift < 0:
            return (self.integer << -shift) & ((1 << count) - 1)
        return (self.integer >> shift) & ((1 << count) - 1)

    def skip(self, count):
        if count < 0 or self.pos + count > self.limit:
            raise AacError('Truncated AAC bitstream')
        self.pos += count

    def get(self, count):
        value = self.peek(count)
        self.skip(count)
        return value

    def align(self):
        self.skip(-self.pos % 8)
