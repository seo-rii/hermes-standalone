"""Vorbis I LSB bit reader/codebooks, source-derived from pinned FFmpeg8.1.2."""
import math

from standalone.local_whisper_audio_aac_dsp import f32


class VorbisError(ValueError):
    def __init__(self, message, *, invalid_frame=False):
        super().__init__(message)
        # Mark only decoder initialization failures after successful original
        # Ogg header/duration parsing. Faster-Whisper filters these frames.
        self.invalid_frame = invalid_frame


class Bits:
    def __init__(self, data, *, padded=False):
        self.data = data
        self.position = 0
        self.padded = padded

    def read(self, count):
        if count < 0 or (not self.padded and self.position + count > 8 * len(self.data)):
            raise VorbisError('Truncated Vorbis bits')
        value = int.from_bytes(self.data[self.position // 8:(self.position + count + 7) // 8], 'little')
        value = (value >> (self.position & 7)) & ((1 << count) - 1)
        self.position += count
        if self.padded:
            # Original safe get_bits clamps at size_in_bits_plus8 and reads
            # the AV_INPUT_BUFFER_PADDING_SIZE zero bytes beyond payload EOF.
            self.position = min(self.position, 8 * len(self.data) + 8)
        return value

    def index(self, count, limit):
        value = self.read(count)
        if value >= limit:
            raise VorbisError('Vorbis setup index is out of range')
        return value


def unpack_float(value):
    mantissa = value & 0x1fffff
    if value & 0x80000000:
        mantissa = -mantissa
    try:
        result = f32(math.ldexp(mantissa, ((value & 0x7fe00000) >> 21) - 788))
    except OverflowError as exc:
        raise VorbisError('Invalid Vorbis codebook float') from exc
    if not math.isfinite(result):
        raise VorbisError('Invalid Vorbis codebook float')
    return result


def codes_from_lengths(lengths):
    """Original ff_vorbis_len2vlc tree-growth order, including sparse books."""
    codes = [0] * len(lengths)
    present = [i for i, length in enumerate(lengths) if length]
    if not present:
        return codes
    exits = [0] * 33
    first = present[0]
    if not 0 < lengths[first] <= 32:
        raise VorbisError('Invalid Vorbis codeword length')
    for i in range(lengths[first]):
        exits[i + 1] = 1 << i
    if len(present) == 1:
        return codes
    for p in present[1:]:
        length = lengths[p]
        if not 0 < length <= 32:
            raise VorbisError('Invalid Vorbis codeword length')
        level = length
        while level and not exits[level]:
            level -= 1
        if not level:
            raise VorbisError('Overspecified Vorbis codebook')
        code = exits[level]
        exits[level] = 0
        for j in range(level + 1, length + 1):
            exits[j] = code + (1 << (j - 1))
        codes[p] = code
    if any(exits):
        raise VorbisError('Underspecified Vorbis codebook')
    return codes


class Codebook:
    def __init__(self, bits):
        if bits.read(24) != 0x564342:
            raise VorbisError('Invalid Vorbis codebook signature')
        self.dimensions = bits.read(16)
        entries = bits.read(24)
        if not 1 <= self.dimensions <= 16 or entries > 65536:
            raise VorbisError('Invalid Vorbis codebook size')
        lengths = self._lengths(bits, entries)
        self.lookup = bits.read(4)
        self.vectors = None
        if self.lookup == 1:
            lengths = self._lookup_vectors(bits, entries, lengths)
        elif self.lookup:
            # This is the pinned original decoder's boundary, not an implicit
            # reinterpretation of lookup2 as lookup1.
            raise VorbisError('Original Vorbis codebook lookup type is unsupported')
        self._tree(lengths)

    @staticmethod
    def _lengths(bits, entries):
        if bits.read(1):
            length = bits.read(5) + 1
            lengths = []
            while len(lengths) < entries and length <= 32:
                number = bits.read((entries - len(lengths)).bit_length())
                if len(lengths) + number > entries:
                    raise VorbisError('Too many ordered Vorbis codewords')
                lengths.extend([length] * number)
                length += 1
            if len(lengths) != entries:
                raise VorbisError('Incomplete ordered Vorbis codebook')
        else:
            sparse = bits.read(1)
            lengths = [(bits.read(5) + 1 if not sparse or bits.read(1) else 0)
                       for _ in range(entries)]
        return lengths

    def _lookup_vectors(self, bits, entries, lengths):
        low, high = 0, entries + 1
        while high - low > 1:
            middle = (low + high) // 2
            if middle ** self.dimensions <= entries:
                low = middle
            else:
                high = middle
        count = low
        minimum, delta = unpack_float(bits.read(32)), unpack_float(bits.read(32))
        value_bits, sequence = bits.read(4) + 1, bits.read(1)
        multiplicands = [bits.read(value_bits) for _ in range(count)]
        vectors = []
        for i, length in enumerate(lengths):
            if not length:
                continue
            last, offset, row = 0., i, []
            for _ in range(self.dimensions):
                value = f32(f32(f32(multiplicands[offset % count] * delta) + minimum) + last)
                row.append(value)
                if sequence:
                    last = value
                offset //= count
            vectors.append(row)
        self.vectors = vectors
        return [length for length in lengths if length]

    def _tree(self, lengths):
        codes = codes_from_lengths(lengths)
        self.tree = {}
        self.max_length = max(lengths, default=0)
        for symbol, (length, code) in enumerate(zip(lengths, codes)):
            if not length:
                continue
            node = self.tree
            for i in range(length):
                bit = (code >> i) & 1
                if i + 1 == length:
                    if bit in node:
                        raise VorbisError('Colliding Vorbis codebook')
                    node[bit] = symbol
                else:
                    node = node.setdefault(bit, {})
                    if not isinstance(node, dict):
                        raise VorbisError('Colliding Vorbis codebook prefix')

    def read(self, bits):
        node = self.tree
        for _ in range(self.max_length):
            node = node.get(bits.read(1))
            if node is None:
                break
            if isinstance(node, int):
                return node
        raise VorbisError('Invalid Vorbis entropy codeword')

    def vector(self, bits):
        if self.vectors is None:
            raise VorbisError('Missing Vorbis VQ codebook')
        return self.vectors[self.read(bits)]
