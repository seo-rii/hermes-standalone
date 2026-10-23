"""Reversible gzip preprocessing for solid compression, independent of encoders.

Gzip headers/trailers, original DEFLATE block/tree bits, token choices and final
padding are preserved. Inflated bytes become visible to an outer compressor;
literal runs and original LZ77 matches form a separate compact recipe. Unpack
recreates the original bits directly and never calls a zlib/gzip encoder.

The format is private and versioned. It is deliberately not enabled by default
in the builder: solid-compression benefit must first be measured on its input.
RFC 1951/1952 define the Huffman codes, block framing and gzip envelope.
"""

from array import array
import hashlib
import struct
import zlib


MAGIC = b"HERMES-GZ1\x00"
MAX_CLEAR = 128 << 20
MAX_OUTPUT = 256 << 20
MAX_HEADER = 1 << 20
MAX_MEMBERS = 65536
MAX_BLOCKS = 65536
_END = 255
_LENGTH_BASE = (3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
                35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258)
_LENGTH_EXTRA = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0)
_DISTANCE_BASE = (1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
                  257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
                  8193, 12289, 16385, 24577)
_DISTANCE_EXTRA = (0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                   7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13)
# RFC order is explicit rather than inferred from the alphabet.
_CODE_ORDER = (16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15)


def _varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return result


class _Cursor:
    def __init__(self, data):
        self.data, self.position = memoryview(data), 0

    def read(self, size):
        end = self.position + size
        if size < 0 or end > len(self.data):
            raise ValueError("Truncated gzip recipe")
        result = self.data[self.position:end]
        self.position = end
        return result

    def varint(self):
        value = 0
        for group in range(9):
            byte = self.read(1)[0]
            value |= (byte & 127) << (7 * group)
            if not byte & 128:
                if group and byte == 0:
                    raise ValueError("Noncanonical gzip recipe integer")
                return value
        raise ValueError("Oversized gzip recipe integer")


class _Bits:
    def __init__(self, data):
        self.data, self.position, self.buffer, self.count = memoryview(data), 0, 0, 0

    def tell(self):
        return self.position * 8 - self.count

    def peek(self, width):
        while self.count < width and self.position < len(self.data):
            self.buffer |= self.data[self.position] << self.count
            self.position += 1
            self.count += 8
        return self.buffer & ((1 << width) - 1)

    def take(self, width):
        value = self.peek(width)
        if width > self.count:
            raise ValueError("Truncated DEFLATE bits")
        self.buffer >>= width
        self.count -= width
        return value

    def bytes(self, size):
        bit = self.tell()
        if bit % 8 or bit // 8 + size > len(self.data):
            raise ValueError("Invalid stored DEFLATE block")
        start, end = bit // 8, bit // 8 + size
        self.position, self.buffer, self.count = end, 0, 0
        return self.data[start:end]


class _Huffman:
    def __init__(self, lengths, maximum=15):
        counts = [0] * (maximum + 1)
        for length in lengths:
            if not 0 <= length <= maximum:
                raise ValueError("Invalid DEFLATE Huffman code length")
            counts[length] += 1
        self.width = max(lengths, default=0)
        self.codes = {}
        self.table = array("I", [0]) * (1 << self.width)
        left, code, next_codes = 1, 0, [0] * (maximum + 1)
        for bits in range(1, maximum + 1):
            left = left * 2 - counts[bits]
            if left < 0:
                raise ValueError("Oversubscribed DEFLATE Huffman tree")
            code = (code + (counts[bits-1] if bits > 1 else 0)) * 2
            next_codes[bits] = code
        self.incomplete = left > 0
        for symbol, length in enumerate(lengths):
            if not length:
                continue
            canonical = next_codes[length]
            next_codes[length] += 1
            reverse = 0
            for _ in range(length):
                reverse = reverse * 2 + (canonical & 1)
                canonical >>= 1
            self.codes[symbol] = (reverse, length)
            entry = (symbol << 4) | length
            for index in range(reverse, len(self.table), 1 << length):
                self.table[index] = entry

    def decode(self, bits):
        entry = self.table[bits.peek(self.width)]
        if not entry:
            raise ValueError("Invalid DEFLATE Huffman symbol")
        bits.take(entry & 15)
        return entry >> 4

    def encode(self, writer, symbol):
        try:
            code, width = self.codes[symbol]
        except KeyError:
            raise ValueError("Gzip recipe uses an absent Huffman symbol") from None
        writer.bits(code, width)


_FIXED_LITERAL = _Huffman([8]*144 + [9]*112 + [7]*24 + [8]*8)
_FIXED_DISTANCE = _Huffman([5]*32)


def _block_header(bits, offset=0):
    final, kind = bits.take(1), bits.take(2)
    if kind == 0:
        bits.take((-(bits.tell() + offset)) % 8)
        size, inverse = bits.take(16), bits.take(16)
        if size ^ inverse != 65535:
            raise ValueError("Invalid stored DEFLATE block length")
        return final, kind, size, None
    if kind == 1:
        return final, kind, _FIXED_LITERAL, _FIXED_DISTANCE
    if kind != 2:
        raise ValueError("Reserved DEFLATE block type")
    literal_count, distance_count, code_count = bits.take(5)+257, bits.take(5)+1, bits.take(4)+4
    if literal_count > 286:
        raise ValueError("Invalid DEFLATE literal alphabet")
    code_lengths = [0]*19
    for symbol in _CODE_ORDER[:code_count]:
        code_lengths[symbol] = bits.take(3)
    code_tree = _Huffman(code_lengths, 7)
    if code_tree.incomplete:
        raise ValueError("Incomplete DEFLATE code-length tree")
    lengths = []
    while len(lengths) < literal_count + distance_count:
        symbol = code_tree.decode(bits)
        if symbol < 16:
            lengths.append(symbol)
            continue
        if symbol == 16:
            if not lengths:
                raise ValueError("DEFLATE repeats a missing code length")
            value, count = lengths[-1], bits.take(2)+3
        elif symbol == 17:
            value, count = 0, bits.take(3)+3
        else:
            value, count = 0, bits.take(7)+11
        if len(lengths)+count > literal_count+distance_count:
            raise ValueError("DEFLATE code length repeat overflows its alphabet")
        lengths.extend([value]*count)
    literal = _Huffman(lengths[:literal_count])
    if 256 not in literal.codes:
        raise ValueError("DEFLATE tree has no end-of-block symbol")
    distance = _Huffman(lengths[literal_count:])
    if literal.incomplete and literal.width != 1 or distance.incomplete and distance.width > 1:
        raise ValueError("Incomplete DEFLATE literal or distance tree")
    return final, kind, literal, distance


def _bit_slice(data, start, end):
    width = end-start
    value = int.from_bytes(data[start//8:(end+7)//8], "little") >> (start % 8)
    return (value & ((1 << width)-1)).to_bytes((width+7)//8, "little")


def _gzip_header_end(data, start):
    if start+10 > len(data) or bytes(data[start:start+3]) != b"\x1f\x8b\x08" or data[start+3] & 224:
        raise ValueError("Invalid gzip header")
    flags, position = data[start+3], start+10
    if flags & 4:
        if position+2 > len(data):
            raise ValueError("Truncated gzip extra field")
        position += 2 + int.from_bytes(data[position:position+2], "little")
    for flag in (8, 16):
        if flags & flag:
            found = bytes(data[position:min(len(data), start+MAX_HEADER)]).find(b"\0")
            if found < 0:
                raise ValueError("Unterminated or oversized gzip text field")
            position += found+1
    if flags & 2:
        if position+2 > len(data) or int.from_bytes(data[position:position+2], "little") != zlib.crc32(data[start:position]) & 65535:
            raise ValueError("Invalid gzip header CRC")
        position += 2
    if position > len(data) or position-start > MAX_HEADER:
        raise ValueError("Truncated or oversized gzip header")
    return position


def _member_recipe(deflate, clear):
    bits, position, blocks = _Bits(deflate), 0, []
    while True:
        if len(blocks) >= MAX_BLOCKS:
            raise ValueError("Too many DEFLATE blocks")
        start = bits.tell()
        final, kind, literal, distance = _block_header(bits)
        prefix_width = bits.tell()-start
        prefix = _bit_slice(deflate, start, bits.tell())
        recipe, pending = bytearray(), 0
        if kind == 0:
            stored = bits.bytes(literal)
            if position+literal > len(clear) or stored != clear[position:position+literal]:
                raise ValueError("Stored DEFLATE bytes differ from the inflated data")
            position += literal
        else:
            while True:
                symbol = literal.decode(bits)
                if symbol < 256:
                    if position >= len(clear) or clear[position] != symbol:
                        raise ValueError("DEFLATE literal differs from the inflated data")
                    position, pending = position+1, pending+1
                    if pending == 128:
                        recipe.append(127)
                        pending = 0
                    continue
                if pending:
                    recipe.append(pending-1)
                    pending = 0
                if symbol == 256:
                    recipe.append(_END)
                    break
                if symbol > 285:
                    raise ValueError("Reserved DEFLATE length symbol")
                index = symbol-257
                extra = bits.take(_LENGTH_EXTRA[index])
                length = _LENGTH_BASE[index]+extra
                dist_symbol = distance.decode(bits)
                if dist_symbol >= 30:
                    raise ValueError("Reserved DEFLATE distance symbol")
                dist = _DISTANCE_BASE[dist_symbol]+bits.take(_DISTANCE_EXTRA[dist_symbol])
                if dist > position or position+length > len(clear):
                    raise ValueError("DEFLATE match exceeds its history or inflated data")
                recipe.append(128+index)
                if _LENGTH_EXTRA[index]:
                    recipe.append(extra)
                recipe.extend(_varint(dist))
                position += length
        blocks.append((prefix_width, prefix, recipe))
        if final:
            break
    padding_width = (-bits.tell()) % 8
    padding = bits.take(padding_width)
    if bits.tell() != len(deflate)*8 or position != len(clear):
        raise ValueError("DEFLATE member has trailing bytes or the wrong inflated length")
    return blocks, padding_width, padding


def pack(data, *, max_clear=MAX_CLEAR):
    """Return deterministic preprocessing bytes, or None for non-gzip input.

    The result usually expands its input. Its purpose is better compression
    across gzip resources inside an outer solid archive, not standalone size.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("Gzip preprocessing requires a bytes-like value")
    data = memoryview(data)
    if bytes(data[:2]) != b"\x1f\x8b":
        return None
    if type(max_clear) is not int or not 0 <= max_clear <= MAX_CLEAR or len(data) > MAX_OUTPUT:
        raise ValueError("Gzip preprocessing bound exceeded")
    members, position, total_clear = [], 0, 0
    while position < len(data):
        if len(members) >= MAX_MEMBERS:
            raise ValueError("Too many gzip members")
        header_end = _gzip_header_end(data, position)
        decoder = zlib.decompressobj(31)
        try:
            clear = decoder.decompress(data[position:], max_clear-total_clear+1)
        except zlib.error as error:
            raise ValueError("Invalid gzip member") from error
        total_clear += len(clear)
        if total_clear > max_clear or not decoder.eof:
            raise ValueError("Truncated gzip member or inflated-data bound exceeded")
        end = len(data)-len(decoder.unused_data)
        if end < header_end+8:
            raise ValueError("Truncated gzip member")
        blocks, padding_width, padding = _member_recipe(data[header_end:end-8], memoryview(clear))
        members.append((bytes(data[position:header_end]), clear, bytes(data[end-8:end]), blocks, padding_width, padding))
        position = end
    output = bytearray(MAGIC)
    output.extend(_varint(len(data)))
    output.extend(_varint(len(members)))
    output.extend(hashlib.sha256(data).digest())
    for header, clear, trailer, blocks, padding_width, padding in members:
        output.extend(_varint(len(header)))
        output.extend(header)
        output.extend(_varint(len(clear)))
        output.extend(clear)
        output.extend(trailer)
        output.extend(_varint(len(blocks)))
        for width, prefix, recipe in blocks:
            output.extend(_varint(width))
            output.extend(prefix)
            output.extend(_varint(len(recipe)))
            output.extend(recipe)
        output.extend((padding_width, padding))
    output.extend(hashlib.sha256(output).digest())
    return bytes(output)


class _Writer:
    def __init__(self, output, limit):
        self.output, self.limit, self.buffer, self.count, self.total = output, limit, 0, 0, 0

    def bytes(self, value):
        if self.count or len(self.output)+len(value) > self.limit:
            raise ValueError("Gzip reconstruction exceeds its output bound or byte alignment")
        self.output.extend(value)
        self.total += len(value)*8

    def bits(self, value, width):
        self.buffer |= value << self.count
        self.count += width
        self.total += width
        while self.count >= 8:
            if len(self.output) >= self.limit:
                raise ValueError("Gzip reconstruction exceeds its output bound")
            self.output.append(self.buffer & 255)
            self.buffer >>= 8
            self.count -= 8

    def prefix(self, data, width):
        for start in range(0, width, 8):
            self.bits(data[start//8], min(8, width-start))


def _reconstruct_block(writer, prefix, width, recipe, clear, position, final_expected):
    if width < 3 or width > 4096 or width % 8 and prefix[-1] >> (width % 8):
        raise ValueError("Invalid DEFLATE prefix recipe")
    bits = _Bits(prefix)
    final, kind, literal, distance = _block_header(bits, writer.total)
    if bits.tell() != width or bool(final) != final_expected:
        raise ValueError("DEFLATE prefix recipe has the wrong framing")
    writer.prefix(prefix, width)
    if kind == 0:
        if len(recipe) or position+literal > len(clear):
            raise ValueError("Invalid stored DEFLATE recipe")
        writer.bytes(clear[position:position+literal])
        return position+literal
    tokens = _Cursor(recipe)
    while tokens.position < len(tokens.data):
        command = tokens.read(1)[0]
        if command < 128:
            count = command+1
            if position+count > len(clear):
                raise ValueError("Gzip literal run exceeds its clear data")
            for symbol in clear[position:position+count]:
                literal.encode(writer, symbol)
            position += count
        elif command < 157:
            index = command-128
            extra = tokens.read(1)[0] if _LENGTH_EXTRA[index] else 0
            if extra >= 1 << _LENGTH_EXTRA[index]:
                raise ValueError("Invalid gzip match length extra bits")
            length, dist = _LENGTH_BASE[index]+extra, tokens.varint()
            if not 1 <= dist <= min(position, 32768) or position+length > len(clear):
                raise ValueError("Gzip match exceeds its history or clear data")
            origin = bytes(clear[position-dist:position-dist+min(dist, length)])
            if bytes(clear[position:position+length]) != (origin*((length+len(origin)-1)//len(origin)))[:length]:
                raise ValueError("Gzip match differs from its clear data")
            literal.encode(writer, 257+index)
            writer.bits(extra, _LENGTH_EXTRA[index])
            symbol = 0
            while symbol < 29 and dist >= _DISTANCE_BASE[symbol+1]:
                symbol += 1
            distance.encode(writer, symbol)
            writer.bits(dist-_DISTANCE_BASE[symbol], _DISTANCE_EXTRA[symbol])
            position += length
        elif command == _END:
            if tokens.position != len(tokens.data):
                raise ValueError("Gzip recipe has tokens after end-of-block")
            literal.encode(writer, 256)
            return position
        else:
            raise ValueError("Reserved gzip recipe token")
    raise ValueError("Gzip recipe has no end-of-block token")


def unpack(data, *, max_output=MAX_OUTPUT):
    """Reconstruct original gzip bytes with strict framing and output bounds."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("Gzip reconstruction requires a bytes-like value")
    if type(max_output) is not int or not 0 <= max_output <= MAX_OUTPUT:
        raise ValueError("Invalid gzip reconstruction output bound")
    view = memoryview(data)
    if len(view) < len(MAGIC)+66 or bytes(view[:len(MAGIC)]) != MAGIC:
        raise ValueError("Invalid gzip preprocessing format")
    cursor = _Cursor(view[:-32])
    cursor.read(len(MAGIC))
    original_size, member_count = cursor.varint(), cursor.varint()
    if original_size > max_output or not 1 <= member_count <= MAX_MEMBERS:
        raise ValueError("Gzip reconstruction output or member bound exceeded")
    if hashlib.sha256(view[:-32]).digest() != bytes(view[-32:]):
        raise ValueError("Gzip preprocessing checksum mismatch")
    expected_digest = bytes(cursor.read(32))
    output, total_clear = bytearray(), 0
    for _ in range(member_count):
        header_size = cursor.varint()
        if not 10 <= header_size <= MAX_HEADER:
            raise ValueError("Invalid gzip header recipe size")
        header = cursor.read(header_size)
        if _gzip_header_end(header, 0) != len(header):
            raise ValueError("Gzip header recipe has trailing bytes")
        clear_size = cursor.varint()
        total_clear += clear_size
        if total_clear > MAX_CLEAR:
            raise ValueError("Gzip recipe exceeds its clear-data bound")
        clear, trailer = cursor.read(clear_size), cursor.read(8)
        crc, size = struct.unpack("<II", trailer)
        if crc != zlib.crc32(clear) or size != clear_size & 0xffffffff:
            raise ValueError("Gzip recipe clear data differs from its trailer")
        count = cursor.varint()
        if not 1 <= count <= MAX_BLOCKS:
            raise ValueError("Invalid DEFLATE recipe block count")
        writer = _Writer(output, original_size)
        writer.bytes(header)
        # Prefix alignment is relative to DEFLATE, not the gzip envelope.
        writer.total = 0
        position = 0
        for index in range(count):
            width = cursor.varint()
            if not 3 <= width <= 4096:
                raise ValueError("Invalid DEFLATE recipe prefix length")
            prefix = cursor.read((width+7)//8)
            recipe_size = cursor.varint()
            if recipe_size > original_size*5+MAX_BLOCKS:
                raise ValueError("Gzip token recipe exceeds its size bound")
            recipe = cursor.read(recipe_size)
            position = _reconstruct_block(writer, prefix, width, recipe, clear, position, index == count-1)
        padding_width, padding = cursor.read(2)
        if padding_width != (-writer.total) % 8 or padding >= 1 << padding_width or position != clear_size:
            raise ValueError("Gzip recipe has invalid padding or an incomplete clear stream")
        writer.bits(padding, padding_width)
        writer.bytes(trailer)
    if cursor.position != len(cursor.data) or len(output) != original_size or hashlib.sha256(output).digest() != expected_digest:
        raise ValueError("Gzip reconstruction size, digest or trailing data mismatch")
    return bytes(output)
