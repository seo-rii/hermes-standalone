"""JPEG pixels using the reviewed pyjpeg primitives and explicit scan assembly.

Supports 8-bit Huffman sequential and progressive frames, including chroma
subsampling, restart intervals, DC/AC refinement, and Adobe CMYK/YCCK. The
application owns scan state because pyjpeg 0.9's high-level reader does not
assemble these scans. Arithmetic, lossless and hierarchical JPEG are rejected.
No Pillow, native extension, external process, or network is used.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass


class JPEGError(ValueError):
    """Malformed or unsupported JPEG input."""


DEFAULT_MAX_PIXELS = 40_000_000


def _dimensions(width, height, max_pixels):
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
            or width > 65535 or height > 65535 or type(max_pixels) is not int
            or max_pixels <= 0 or width * height > max_pixels):
        raise JPEGError("JPEG dimensions exceed the permitted pixel limit")


def _clip(value):
    return max(0, min(255, value))


def _segment(reader):
    marker = reader.read_marker()
    length = reader.read_u16()
    if length < 2:
        raise JPEGError("Invalid JPEG segment length")
    payload = reader.read(length - 2)
    return marker, payload


def _parse(marker, payload, parser):
    import pyjpeg
    reader = pyjpeg.BufferedReader(bytes((255, marker)) + (len(payload) + 2).to_bytes(2, "big") + payload)
    result = parser.read(reader)
    if reader.offset != len(reader.data):
        raise JPEGError("JPEG segment parser left unread data")
    return result


def _marker(reader):
    # JPEG permits repeated FF fill bytes before a marker.
    while reader.peek_u8() == 255 and reader.peek_u8(1) == 255:
        reader.read_u8()
    return reader.peek_marker()


def _bits(reader, count):
    value = 0
    for _ in range(count):
        value = (value << 1) | reader.read_bit()
    return value


def _signed(reader, count):
    if count == 0:
        return 0
    value = _bits(reader, count)
    return value if value >= 1 << (count - 1) else value - ((1 << count) - 1)


def _align(reader):
    if reader.bit_count and reader.data & ((1 << reader.bit_count) - 1) != (1 << reader.bit_count) - 1:
        raise JPEGError("Invalid JPEG entropy padding")


def _huffman(table):
    from pyjpeg.huffman import Decoder
    code = 0
    for length, symbols in enumerate(table, 1):
        # All-one code words are forbidden because they overlap scan padding.
        if code + len(symbols) >= 1 << length:
            raise JPEGError("Invalid JPEG Huffman code lengths")
        code = (code + len(symbols)) << 1
    if not any(table):
        raise JPEGError("Empty JPEG Huffman table")
    return Decoder(table)


@dataclass
class _Component:
    identifier: int
    horizontal: int
    vertical: int
    table: int
    width: int
    height: int
    stride: int
    rows: int
    coefficients: array
    progression: list
    quantization: list | None = None


def _components(frame, max_pixels):
    width, height = frame.samples_per_line, frame.number_of_lines
    _dimensions(width, height, max_pixels)
    if frame.n not in (0, 1, 2) or frame.precision != 8:
        raise JPEGError("Only 8-bit Huffman sequential/progressive JPEG is supported")
    if len(frame.components) not in (1, 3, 4):
        raise JPEGError("Unsupported JPEG component count")
    factors = [component.sampling_factor for component in frame.components]
    if (any(not (1 <= h <= 4 and 1 <= v <= 4) for h, v in factors)
            or len(factors) > 1 and sum(h * v for h, v in factors) > 10):
        raise JPEGError("Invalid JPEG sampling factors")
    hmax, vmax = max(h for h, _ in factors), max(v for _, v in factors)
    columns, rows = (width + 8 * hmax - 1) // (8 * hmax), (height + 8 * vmax - 1) // (8 * vmax)
    result = {}
    for component in frame.components:
        if component.id in result or component.quantization_table_index > 3:
            raise JPEGError("Invalid or duplicate JPEG frame component")
        h, v = component.sampling_factor
        stride, block_rows = columns * h, rows * v
        result[component.id] = _Component(
            component.id, h, v, component.quantization_table_index,
            (width * h + hmax - 1) // hmax, (height * v + vmax - 1) // vmax,
            stride, block_rows, array("i", [0]) * (stride * block_rows * 64), [None] * 64,
        )
    return result, columns, rows, hmax, vmax


def _read_block(bits, block, dc, ac, previous_dc, first, last, high, low, eob):
    if first == 0:
        if high:
            block[0] |= bits.read_bit() << low
        else:
            size = dc.read_symbol(bits)
            if size > 11:
                raise JPEGError("Invalid JPEG DC category")
            previous_dc += _signed(bits, size)
            block[0] = previous_dc << low
        if last == 0:
            return previous_dc, eob
        first = 1
    if not high:
        if eob:
            return previous_dc, eob - 1
        index = first
        while index <= last:
            symbol = ac.read_symbol(bits)
            run, size = symbol >> 4, symbol & 15
            if size:
                if size > 10 or index + run > last:
                    raise JPEGError("Invalid JPEG AC coefficient run")
                index += run
                block[index] = _signed(bits, size) << low
                index += 1
            elif run == 15:
                index += 16
                if index > last + 1:
                    raise JPEGError("JPEG zero run exceeds spectral band")
            else:
                return previous_dc, (1 << run) + _bits(bits, run) - 1
        return previous_dc, 0

    # Successive AC scans interleave correction bits for existing coefficients
    # with Huffman-coded runs introducing new coefficients (T.81 G.1.2.3).
    step = 1 << low
    index = first
    if not eob:
        while index <= last:
            symbol = ac.read_symbol(bits)
            run, size = symbol >> 4, symbol & 15
            if size:
                if size != 1:
                    raise JPEGError("Invalid progressive JPEG refinement category")
                new = step if bits.read_bit() else -step
            elif run != 15:
                eob = (1 << run) + _bits(bits, run)
                break
            else:
                run = 16
            while index <= last:
                old = block[index]
                if old:
                    if bits.read_bit() and not abs(old) & step:
                        block[index] += step if old > 0 else -step
                    index += 1
                elif run:
                    run -= 1
                    index += 1
                    if not size and run == 0:
                        break
                else:
                    break
            if size:
                if index > last:
                    raise JPEGError("Progressive JPEG run exceeds spectral band")
                block[index] = new
                index += 1
            elif run:
                raise JPEGError("Progressive JPEG zero run exceeds spectral band")
    if eob:
        while index <= last:
            old = block[index]
            if old and bits.read_bit() and not abs(old) & step:
                block[index] += step if old > 0 else -step
            index += 1
        eob -= 1
    return previous_dc, eob


def _scan(reader, frame, components, columns, rows, scan, tables, quantization, interval):
    from pyjpeg.scan import Reader
    first, last = scan.spectral_selection
    high, low = scan.point_transform >> 4, scan.point_transform & 15
    selected = [item.component_selector for item in scan.components]
    if not selected or len(selected) > 4 or len(set(selected)) != len(selected) or any(key not in components for key in selected):
        raise JPEGError("Invalid JPEG scan components")
    progressive = frame.n == 2
    if progressive:
        if not (0 <= first <= last <= 63 and low <= 13 and high <= 13) or (first == 0 and last != 0):
            raise JPEGError("Invalid progressive JPEG spectral selection")
        if first and len(selected) != 1 or high and high != low + 1:
            raise JPEGError("Invalid progressive JPEG approximation scan")
    elif (first, last, high, low) != (0, 63, 0, 0):
        raise JPEGError("Invalid sequential JPEG spectral selection")
    decoders = {}
    for item in scan.components:
        component = components[item.component_selector]
        expected = high if high else None
        if any(component.progression[index] != expected for index in range(first, last + 1)):
            raise JPEGError("JPEG coefficient scans are out of order")
        if first and component.progression[0] is None:
            raise JPEGError("JPEG AC scan precedes DC scan")
        if component.table not in quantization:
            raise JPEGError("Missing JPEG quantization table")
        if component.quantization is None:
            component.quantization = quantization[component.table][:]
        elif component.quantization != quantization[component.table]:
            raise JPEGError("JPEG quantization changes within a component")
        dc = tables.get((0, item.dc_table)) if first == 0 and not high else None
        ac = tables.get((1, item.ac_table)) if last > 0 else None
        if first == 0 and not high and dc is None or last > 0 and ac is None:
            raise JPEGError("Missing JPEG Huffman table")
        decoders[component.identifier] = dc, ac
        component.progression[first:last + 1] = [low] * (last - first + 1)
    interleaved = len(selected) > 1
    if interleaved:
        scan_columns, scan_rows = columns, rows
    else:
        component = components[selected[0]]
        scan_columns, scan_rows = (component.width + 7) // 8, (component.height + 7) // 8
    count = scan_columns * scan_rows
    bits = Reader(reader)
    predictors = dict.fromkeys(selected, 0)
    eob = 0
    restart = 0
    for mcu in range(count):
        if interval and mcu and mcu % interval == 0:
            if eob:
                raise JPEGError("JPEG end-of-band run crosses restart")
            _align(bits)
            if _marker(reader) != 0xD0 + restart:
                raise JPEGError("Missing or out-of-order JPEG restart marker")
            reader.read_marker()
            restart = (restart + 1) % 8
            bits = Reader(reader)
            predictors = dict.fromkeys(selected, 0)
        my, mx = divmod(mcu, scan_columns)
        for identifier in selected:
            component = components[identifier]
            h, v = (component.horizontal, component.vertical) if interleaved else (1, 1)
            for by in range(v):
                for bx in range(h):
                    offset = ((my * v + by) * component.stride + mx * h + bx) * 64
                    block = component.coefficients[offset:offset + 64]
                    dc, ac = decoders[identifier]
                    predictors[identifier], eob = _read_block(
                        bits, block, dc, ac, predictors[identifier], first, last, high, low, eob)
                    if not progressive and eob:
                        raise JPEGError("Progressive end-of-band run in sequential JPEG")
                    component.coefficients[offset:offset + 64] = block
    if eob:
        raise JPEGError("JPEG end-of-band run exceeds the scan")
    _align(bits)


def _plane(component):
    import pyjpeg
    result = bytearray(component.width * component.height)
    for by in range((component.height + 7) // 8):
        for bx in range((component.width + 7) // 8):
            offset = (by * component.stride + bx) * 64
            values = pyjpeg.idct(component.coefficients[offset:offset + 64], component.quantization, 8)
            length = min(8, component.width - bx * 8)
            for row in range(min(8, component.height - by * 8)):
                start = (by * 8 + row) * component.width + bx * 8
                result[start:start + length] = bytes(values[row * 8:row * 8 + length])
    return result


def _upsample(plane, component, width, height, hmax, vmax):
    if component.horizontal == hmax and component.vertical == vmax:
        return plane
    if component.width <= 2:
        # The reference JPEG upsampler uses box expansion for very narrow
        # components, where its neighboring-column filter cannot be applied.
        return bytearray(plane[min(component.height - 1, y * component.vertical // vmax) * component.width
                               + min(component.width - 1, x * component.horizontal // hmax)]
                         for y in range(height) for x in range(width))
    # Center-aligned bilinear chroma upsampling, with edge extension. Integer
    # coordinates preserve deterministic rounding and the common 3:1 filter.
    output = bytearray(width * height)
    dx, dy = 2 * hmax, 2 * vmax
    denominator = dx * dy
    for y in range(height):
        sy, fy = divmod((2 * y + 1) * component.vertical - vmax, dy)
        y0, y1 = max(0, min(sy, component.height - 1)), max(0, min(sy + 1, component.height - 1))
        for x in range(width):
            sx, fx = divmod((2 * x + 1) * component.horizontal - hmax, dx)
            x0, x1 = max(0, min(sx, component.width - 1)), max(0, min(sx + 1, component.width - 1))
            top = plane[y0 * component.width + x0] * (dx - fx) + plane[y0 * component.width + x1] * fx
            bottom = plane[y1 * component.width + x0] * (dx - fx) + plane[y1 * component.width + x1] * fx
            output[y * width + x] = (top * (dy - fy) + bottom * fy + denominator // 2) // denominator
    return output


def _rgb(y, cb, cr):
    cb, cr = cb - 128, cr - 128
    return (_clip(y + ((91881 * cr + 32768) >> 16)),
            _clip(y - ((22554 * cb + 46802 * cr + 32768) >> 16)),
            _clip(y + ((116130 * cb + 32768) >> 16)))


def _pixels(frame, components, hmax, vmax, adobe, jfif):
    width, height = frame.samples_per_line, frame.number_of_lines
    planes = [_upsample(_plane(item), item, width, height, hmax, vmax) for item in components.values()]
    if len(planes) == 1:
        return width, height, "L", bytes(planes[0])
    output = bytearray(width * height * 3)
    direct_rgb = len(planes) == 3 and (adobe == 0 or not jfif and list(components) == [82, 71, 66])
    if len(planes) == 3 and adobe not in (None, 0, 1) or len(planes) == 4 and adobe not in (None, 0, 2):
        raise JPEGError("Unsupported JPEG Adobe color transform")
    for index in range(width * height):
        values = [plane[index] for plane in planes]
        if len(planes) == 3:
            rgb = values if direct_rgb else _rgb(*values)
        elif adobe == 2:
            inverted = _rgb(*values[:3])
            rgb = [((255 - channel) * values[3] + 127) // 255 for channel in inverted]
        elif adobe == 0:
            rgb = [(channel * values[3] + 127) // 255 for channel in values[:3]]
        else:
            rgb = [((255 - channel) * (255 - values[3]) + 127) // 255 for channel in values[:3]]
        output[index * 3:index * 3 + 3] = bytes(rgb)
    return width, height, "RGB", bytes(output)


def decode(data: bytes, max_pixels: int = DEFAULT_MAX_PIXELS) -> tuple[int, int, str, bytes]:
    """Decode supported JPEG bytes to L/RGB pixels with a preallocation limit."""
    import pyjpeg
    from pyjpeg.scan import ReadError as ScanReadError
    reader = pyjpeg.BufferedReader(data)
    frame = None
    components = {}
    tables, quantization = {}, {}
    interval = 0
    adobe, jfif = None, False
    try:
        if reader.read(2) != b"\xff\xd8":
            raise JPEGError("Missing JPEG start marker")
        while True:
            marker = _marker(reader)
            if marker == 0xD9:
                reader.read_marker()
                if frame is None or any(item.progression[0] is None for item in components.values()):
                    raise JPEGError("JPEG ended before all image components were decoded")
                return _pixels(frame, components, hmax, vmax, adobe, jfif)
            marker, payload = _segment(reader)
            if marker in (0xC0, 0xC1, 0xC2):
                if frame is not None:
                    raise JPEGError("Multiple JPEG frames are unsupported")
                frame = _parse(marker, payload, pyjpeg.StartOfFrame)
                components, columns, rows, hmax, vmax = _components(frame, max_pixels)
            elif marker == 0xDB:
                for table in _parse(marker, payload, pyjpeg.DefineQuantizationTables).tables:
                    if len(table.values) != 64 or any(value == 0 for value in table.values):
                        raise JPEGError("Invalid JPEG quantization values")
                    quantization[table.destination] = table.values
            elif marker == 0xC4:
                for table in _parse(marker, payload, pyjpeg.DefineHuffmanTables).tables:
                    tables[table.table_class, table.destination] = _huffman(table.table)
            elif marker == 0xDD:
                if len(payload) != 2:
                    raise JPEGError("Invalid JPEG restart interval")
                interval = int.from_bytes(payload, "big")
            elif marker == 0xDA:
                if frame is None:
                    raise JPEGError("JPEG scan precedes its frame")
                scan = _parse(marker, payload, pyjpeg.StartOfScan)
                _scan(reader, frame, components, columns, rows, scan, tables, quantization, interval)
            elif 0xE0 <= marker <= 0xEF or marker == 0xFE:
                if marker == 0xE0 and payload.startswith(b"JFIF\x00"):
                    jfif = True
                if marker == 0xEE and payload.startswith(b"Adobe"):
                    if len(payload) < 12:
                        raise JPEGError("Truncated JPEG Adobe header")
                    adobe = payload[11]
            else:
                raise JPEGError(f"Unsupported JPEG marker 0x{marker:02x}")
    except JPEGError:
        raise
    except (EOFError, pyjpeg.ReadError, ScanReadError, AssertionError, IndexError, KeyError, OverflowError, ValueError) as exc:
        raise JPEGError("Malformed or truncated JPEG data") from exc


def encode(width: int, height: int, mode: str, pixels: bytes, quality: int = 75) -> bytes:
    """Encode L/RGB pixels as deterministic baseline JPEG with 4:4:4 chroma."""
    import pyjpeg
    from pyjpeg.huffman import Encoder
    from pyjpeg.huffman_dct_scan import Writer
    _dimensions(width, height, DEFAULT_MAX_PIXELS)
    channels = {"L": 1, "RGB": 3}.get(mode)
    if channels is None or len(pixels) != width * height * channels:
        raise JPEGError("JPEG encoding needs correctly sized L or RGB pixels")
    if type(quality) is not int or not 1 <= quality <= 100:
        raise JPEGError("JPEG quality must be an integer from 1 through 100")
    scale = 5000 // quality if quality < 50 else 200 - quality * 2
    quantization = [[max(1, min(255, (value * scale + 50) // 100)) for value in table] for table in (
        pyjpeg.standard_luminance_quantization_table, pyjpeg.standard_chrominance_quantization_table)]
    dc_tables = [pyjpeg.standard_luminance_dc_huffman_table, pyjpeg.standard_chrominance_dc_huffman_table]
    ac_tables = [pyjpeg.standard_luminance_ac_huffman_table, pyjpeg.standard_chrominance_ac_huffman_table]
    writer = pyjpeg.BufferedWriter()
    pyjpeg.StartOfImage().write(writer)
    pyjpeg.JfifHeader().write(writer)
    table_count = 1 if channels == 1 else 2
    pyjpeg.DefineQuantizationTables([pyjpeg.QuantizationTable(index, quantization[index]) for index in range(table_count)]).write(writer)
    pyjpeg.DefineHuffmanTables([table for index in range(table_count) for table in (
        pyjpeg.HuffmanTable.dc(index, dc_tables[index]), pyjpeg.HuffmanTable.ac(index, ac_tables[index]))]).write(writer)
    frame = [pyjpeg.FrameComponent.dct(index + 1, quantization_table_index=int(index > 0)) for index in range(channels)]
    pyjpeg.StartOfFrame.baseline(height, width, frame).write(writer)
    pyjpeg.StartOfScan.dct([pyjpeg.ScanComponent.dct(index + 1, int(index > 0), int(index > 0)) for index in range(channels)]).write(writer)
    entropy = Writer(writer)
    encoders = [(Encoder(dc_tables[index]), Encoder(ac_tables[index])) for index in range(table_count)]
    previous = [0] * channels
    for by in range(0, height, 8):
        for bx in range(0, width, 8):
            blocks = [[] for _ in range(channels)]
            for y in range(by, by + 8):
                for x in range(bx, bx + 8):
                    index = (min(y, height - 1) * width + min(x, width - 1)) * channels
                    if channels == 1:
                        blocks[0].append(pixels[index])
                    else:
                        red, green, blue = pixels[index:index + 3]
                        blocks[0].append(_clip((19595 * red + 38470 * green + 7471 * blue + 32768) >> 16))
                        blocks[1].append(_clip(((-11059 * red - 21709 * green + 32768 * blue + 32768) >> 16) + 128))
                        blocks[2].append(_clip(((32768 * red - 27439 * green - 5329 * blue + 32768) >> 16) + 128))
            for index, block in enumerate(blocks):
                table = int(index > 0)
                coefficients = pyjpeg.fdct(block, 8, quantization[table])
                entropy.write_data_unit(coefficients, *encoders[table], prev_dc=previous[index])
                previous[index] = coefficients[0]
    entropy.flush()
    pyjpeg.EndOfImage().write(writer)
    return bytes(writer.data)
