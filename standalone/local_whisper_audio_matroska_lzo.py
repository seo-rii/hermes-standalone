"""Owned LZO1X decoding in the pinned FFmpeg source operation order."""


class LzoError(ValueError):
    pass


class _Lzo:
    def __init__(self, data, limit):
        self.data, self.limit = data, limit
        self.position, self.state = 0, 0
        self.output = bytearray()

    def byte(self):
        if self.position >= len(self.data):
            raise LzoError("Truncated LZO1X input")
        value = self.data[self.position]
        self.position += 1
        return value

    def length(self, value, mask):
        count = value & mask
        if not count:
            value = self.byte()
            while not value:
                count += 255
                if count >= 0x7FFFFFFF - 1000:
                    raise LzoError("Oversized LZO1X count")
                value = self.byte()
            count += mask + value
        return count

    def literal(self, count):
        following = self.position + count
        if following > len(self.data) or len(self.output) + count > self.limit:
            raise LzoError("Truncated or oversized LZO1X literal")
        self.output.extend(self.data[self.position:following])
        self.position = following

    def back(self, distance, count):
        if distance > len(self.output) or len(self.output) + count > self.limit:
            raise LzoError("Invalid or oversized LZO1X back reference")
        pattern = self.output[-distance:]
        self.output.extend((pattern * ((count + distance - 1) // distance))[:count])

    def step(self, value):
        if value > 15:
            if value > 63:
                count = (value >> 5) - 1
                distance = (self.byte() << 3) + ((value >> 2) & 7) + 1
            elif value > 31:
                count = self.length(value, 31)
                value = self.byte()
                distance = (self.byte() << 6) + (value >> 2) + 1
            else:
                count = self.length(value, 7)
                distance = (1 << 14) + ((value & 8) << 11)
                value = self.byte()
                distance += (self.byte() << 6) + (value >> 2)
                if distance == 1 << 14:
                    if count != 1:
                        raise LzoError("Invalid LZO1X end marker")
                    return None
        elif not self.state:
            count = self.length(value, 15)
            self.literal(count + 3)
            value = self.byte()
            if value > 15:
                return value
            count = 1
            distance = (1 << 11) + (self.byte() << 2) + (value >> 2) + 1
        else:
            count = 0
            distance = (self.byte() << 2) + (value >> 2) + 1
        self.back(distance, count + 2)
        self.state = value & 3
        self.literal(self.state)
        return self.byte()


def decode_lzo(data):
    if not data or len(data) >= 10000000:
        raise LzoError("Invalid Matroska LZO1X input size")
    limit = len(data)
    while limit < 10000000:
        limit *= 3
    decoder = _Lzo(data, limit)
    value = decoder.byte()
    if value > 17:
        decoder.literal(value - 17)
        value = decoder.byte()
        if value < 16:
            raise LzoError("Invalid LZO1X initial match")
    while value is not None:
        value = decoder.step(value)
    return bytes(decoder.output)
