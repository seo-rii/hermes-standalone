"""Vorbis I floor0/1 reconstruction, pinned FFmpeg8.1.2 LGPL algorithm."""
import math

from standalone.local_whisper_audio_aac_dsp import f32
from .local_whisper_audio_vorbis_bits import VorbisError
from .local_whisper_audio_vorbis_tables import INVERSE_DB


def f(value):
    try:
        return f32(value)
    except OverflowError:
        # C assignment to float overflows to signed infinity. Python's
        # struct.pack raises instead; floor0 can expose this for singular LSPs.
        return math.copysign(math.inf, value)


def _clip16(value):
    return min(65535, max(0, value))


def _render_line(x0, y0, x1, y1, values):
    dy, adx = y1 - y0, x1 - x0
    ady, sy = abs(dy), -1 if dy < 0 else 1
    values[x0] = INVERSE_DB[min(255, max(0, y0))]
    if ady * 2 <= adx:
        err, x, y = -adx, x0 - x1 + 1, y0
        while True:
            x += 1
            if x >= 0:
                break
            err += ady
            if err >= 0:
                err += ady - adx
                y += sy
                values[x + x1 - 1] = INVERSE_DB[min(255, max(0, y))]
                x += 1
            values[x + x1 - 1] = INVERSE_DB[min(255, max(0, y))]
        if x <= 0:
            if err + ady >= 0:
                y += sy
            values[x + x1 - 1] = INVERSE_DB[min(255, max(0, y))]
    else:
        base = abs(dy) // adx * sy
        ady -= abs(base) * adx
        y, err = y0, -adx
        for x in range(x0 + 1, x1):
            y += base
            err += ady
            if err >= 0:
                err -= adx
                y += sy
            values[x] = INVERSE_DB[min(255, max(0, y))]


class Floor1:
    def __init__(self, bits, books, block_sizes):
        self.books = books
        self.partitions = [bits.read(4) for _ in range(bits.read(5))]
        self.classes = []
        for _ in range(max(self.partitions, default=-1) + 1):
            dimensions, subclasses = bits.read(3) + 1, bits.read(2)
            master = bits.index(8, len(books)) if subclasses else 0
            subbooks = []
            for _ in range(1 << subclasses):
                book = bits.read(8) - 1
                if book >= len(books):
                    raise VorbisError('Invalid Vorbis floor1 subclass book')
                subbooks.append(book)
            self.classes.append((dimensions, subclasses, master, subbooks))
        self.multiplier = bits.read(2) + 1
        range_bits = bits.read(4)
        if (not range_bits and self.partitions) or 1 << range_bits > block_sizes[1] // 2:
            raise VorbisError('Invalid Vorbis floor1 coordinate range')
        self.x = [0, 1 << range_bits]
        for partition in self.partitions:
            self.x.extend(bits.read(range_bits) for _ in range(self.classes[partition][0]))
        if len(set(self.x)) != len(self.x):
            raise VorbisError('Duplicate Vorbis floor1 coordinate')
        self.neighbors = []
        for i in range(2, len(self.x)):
            low = max((j for j in range(i) if self.x[j] < self.x[i]), key=self.x.__getitem__)
            high = min((j for j in range(i) if self.x[j] > self.x[i]), key=self.x.__getitem__)
            self.neighbors.append((low, high))
        self.order = sorted(range(len(self.x)), key=self.x.__getitem__)

    def decode(self, bits, length):
        if not bits.read(1):
            return [0.] * length, True
        limit = (256, 128, 86, 64)[self.multiplier - 1]
        values = [bits.read((limit - 1).bit_length()) for _ in range(2)]
        for partition in self.partitions:
            dimensions, subclasses, master, subbooks = self.classes[partition]
            cval = self.books[master].read(bits) if subclasses else 0
            for _ in range(dimensions):
                book = subbooks[cval & ((1 << subclasses) - 1)]
                cval >>= subclasses
                values.append(self.books[book].read(bits) if book >= 0 else 0)
        final, flags = values[:2] + [0] * (len(values) - 2), [True, True] + [False] * (len(values) - 2)
        for i, (low, high) in enumerate(self.neighbors, 2):
            dy, adx = final[high] - final[low], self.x[high] - self.x[low]
            off = abs(dy) * (self.x[i] - self.x[low]) // adx
            predicted = final[low] - off if dy < 0 else final[low] + off
            highroom, lowroom = (limit - predicted) & 0xffffffff, predicted
            room, value = min(highroom, lowroom) * 2, values[i]
            if value:
                flags[low] = flags[high] = flags[i] = True
                if value >= room:
                    final[i] = _clip16(value - lowroom + predicted if highroom > lowroom else
                                       predicted - value + highroom - 1)
                else:
                    final[i] = _clip16(predicted - (value + 1) // 2 if value & 1 else predicted + value // 2)
            else:
                final[i] = _clip16(predicted)
        output = [0.] * max(length, self.x[1])
        lx, ly = 0, final[0] * self.multiplier
        for pos in self.order[1:]:
            if flags[pos]:
                x1, y1 = self.x[pos], final[pos] * self.multiplier
                if lx < self.x[1]:
                    _render_line(lx, ly, min(x1, self.x[1]), y1, output)
                lx, ly = x1, y1
            if lx >= self.x[1]:
                break
        if lx < self.x[1]:
            _render_line(lx, ly, self.x[1], ly, output)
        return output[:length], False


def _bark(value):
    return (f(13.1) * math.atan(f(f(.00074) * value))
            + f(2.24) * math.atan(f(f(f(1.85e-8) * value) * value)) + f(f(1e-4) * value))


class Floor0:
    def __init__(self, bits, books, block_sizes):
        self.order, self.rate, self.bark_size = bits.read(8), bits.read(16), bits.read(16)
        self.amplitude_bits, self.amplitude_offset = bits.read(6), bits.read(8)
        self.book_list = [bits.index(8, len(books)) for _ in range(bits.read(4) + 1)]
        self.books = books
        if not self.order or not self.rate or not self.bark_size:
            raise VorbisError('Invalid Vorbis floor0 parameters')
        scale = self.bark_size / _bark(f(self.rate / 2))
        self.maps = {}
        for size in block_sizes:
            length = size // 2
            self.maps[length] = [min(self.bark_size - 1, math.floor(_bark(f(self.rate * i / (2 * length))) * scale))
                                 for i in range(length)]

    def decode(self, bits, length):
        amplitude = bits.read(self.amplitude_bits)
        if not amplitude:
            return [0.] * length, True
        index = bits.read(len(self.book_list).bit_length())
        if index >= len(self.book_list):
            index = 0  # The original decoder logs and selects its first book.
        book = self.books[self.book_list[index]]
        lsp, last = [], 0.
        while len(lsp) < self.order:
            row = [f(value + last) for value in book.vector(bits)]
            lsp.extend(row)
            last = row[-1]
        lsp = [f(2 * math.cos(value)) for value in lsp[:self.order]]
        wstep, output, previous = f(math.pi / self.bark_size), [], None
        for index in self.maps[length]:
            if index != previous:
                p, q = .5, .5
                cosine = f(2 * math.cos(f(wstep * index)))
                j = 0
                while j + 1 < self.order:
                    q, p = f(q * f(lsp[j] - cosine)), f(p * f(lsp[j + 1] - cosine))
                    j += 2
                if j == self.order:
                    p, q = f(p * f(p * f(2 - cosine))), f(q * f(q * f(2 + cosine)))
                else:
                    q = f(q * f(cosine - lsp[j]))
                    p, q = f(p * f(p * f(4 - f(cosine * cosine)))), f(q * q)
                total = f(p + q)
                if total == 0:
                    raise VorbisError('Invalid Vorbis floor0 product')
                root = math.sqrt(total) if total >= 0 else math.nan
                exponent = (((amplitude * self.amplitude_offset) & 0xffffffffffffffff)
                            / (((1 << self.amplitude_bits) - 1) * root)
                            - self.amplitude_offset) * f(.11512925)
                try:
                    value = f(math.exp(exponent))
                except OverflowError:
                    value = math.inf
                previous = index
            output.append(value)
        return output, False
