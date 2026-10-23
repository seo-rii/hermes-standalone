"""Pure Python VP8 keyframe decoder, translated from golang.org/x/image/vp8.

Copyright 2011, 2014 The Go Authors. All rights reserved.
BSD-3-Clause: see licenses/go-image-LICENSE and vp8-provenance.json for the
immutable reference revision, original file hashes, and license text.
Color conversion matches libwebp's fixed-point reconstruction; its BSD terms
and pinned source are in licenses/libwebp-COPYING and webp-provenance.json.
Format: https://www.rfc-editor.org/rfc/rfc6386

Only independent keyframes used by WebP are accepted; interframe video needs a
reference-frame store and is not part of a WebP bitstream. Pixel and partition
bounds are checked before image-plane allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from standalone import raster_vp8_filter as loop_filter
from standalone.raster_vp8_idct import inverse4, inverse_wht, signed16
from standalone.raster_vp8_predict import predFunc4, predFunc8, predFunc16
from standalone.raster_vp8_tables import (
    defaultTokenProb, tokenProbUpdateProb, predProb, dequantTableAC, dequantTableDC,
)


class VP8Error(ValueError):
    """Malformed, unsupported, or resource-exceeding VP8 data."""


def clip(value, minimum, maximum):
    return max(minimum, min(maximum, value))


class _Partition:
    def __init__(self, data):
        self.data = memoryview(data)
        self.offset = 0
        self.range = 254
        self.bits = self.count = 0

    def bit(self, probability=128):
        if self.count < 8:
            if self.offset >= len(self.data):
                raise VP8Error("Truncated VP8 arithmetic-coded partition")
            self.bits |= self.data[self.offset] << (8 - self.count)
            self.offset += 1
            self.count += 8
        split = ((self.range * probability) >> 8) + 1
        value = int(self.bits >= split << 8)
        if value:
            self.range -= split
            self.bits -= split << 8
        else:
            self.range = split - 1
        if self.range < 127:
            shift = 8 - (self.range + 1).bit_length()
            self.range = ((self.range + 1) << shift) - 1
            self.bits <<= shift
            self.count -= shift
        return value

    def uint(self, count, probability=128):
        result = 0
        for _ in range(count):
            result = (result << 1) | self.bit(probability)
        return result

    def signed(self, count):
        value = self.uint(count)
        return -value if self.bit() else value

    def optional(self, count):
        return self.signed(count) if self.bit() else 0


@dataclass
class _MB:
    predictions: list = field(default_factory=lambda: [0] * 4)
    nz: int = 0
    y16: int = 0


_BANDS = (0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0)
_ZIGZAG = (0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15)
_CATEGORIES = ((173, 148, 140), (176, 155, 140, 135), (180, 157, 141, 134, 130),
               (254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129))


class _Decoder:
    def __init__(self, data, maximum):
        if len(data) < 10 or data[0] & 1 or data[3:6] != b"\x9d\x01\x2a":
            raise VP8Error("WebP VP8 data must contain an independent keyframe")
        version = (data[0] >> 1) & 7
        if version > 3 or not data[0] & 16:
            raise VP8Error("Unsupported VP8 frame version or hidden frame")
        width = int.from_bytes(data[6:8], "little") & 16383
        height = int.from_bytes(data[8:10], "little") & 16383
        if (not width or not height or type(maximum) is not int or maximum < 1
                or width * height > maximum):
            raise VP8Error("VP8 dimensions exceed the permitted pixel limit")
        partition_length = int.from_bytes(data[:3], "little") >> 5
        if not partition_length or 10 + partition_length > len(data):
            raise VP8Error("Invalid VP8 first partition length")
        self.width, self.height = width, height
        self.mbw, self.mbh = (width + 15) // 16, (height + 15) // 16
        self.fp = _Partition(data[10:10 + partition_length])
        self.remainder = memoryview(data)[10 + partition_length:]
        self.tokenProb = bytearray(defaultTokenProb)
        self.segment = 0
        self.segmented = self.update_map = self.relative = False
        self.segment_q, self.segment_f, self.segment_p = [0] * 4, [0] * 4, [255] * 3
        self.simple = self.level = self.sharpness = self.delta = 0
        self.reference_delta, self.mode_delta = [0] * 4, [0] * 4
        self.coeff = [0] * 400
        self.ybr = [bytearray(32) for _ in range(26)]
        self.pred4 = [[0] * 4 for _ in range(4)]

    def headers(self):
        fp = self.fp
        if fp.bit():
            raise VP8Error("Unsupported VP8 color space")
        fp.bit()  # Pixel clamping convention; reconstructed pixels are clipped.
        self.segmented = bool(fp.bit())
        if self.segmented:
            self.update_map = bool(fp.bit())
            if fp.bit():
                self.relative = not fp.bit()
                self.segment_q = [fp.optional(7) for _ in range(4)]
                self.segment_f = [fp.optional(6) for _ in range(4)]
            if self.update_map:
                self.segment_p = [fp.uint(8) if fp.bit() else 255 for _ in range(3)]
        self.simple, self.level, self.sharpness, self.delta = fp.bit(), fp.uint(6), fp.uint(3), fp.bit()
        if self.delta and fp.bit():
            self.reference_delta = [fp.optional(6) for _ in range(4)]
            self.mode_delta = [fp.optional(6) for _ in range(4)]
        self.filter_parameters = self._filter_parameters()
        self.partition_count = 1 << fp.uint(2)
        table_size = 3 * (self.partition_count - 1)
        if len(self.remainder) < table_size:
            raise VP8Error("Truncated VP8 coefficient partition table")
        sizes = [int.from_bytes(self.remainder[index:index + 3], "little")
                 for index in range(0, table_size, 3)]
        final = len(self.remainder) - table_size - sum(sizes)
        if final < 0 or final >= 1 << 24:
            raise VP8Error("Invalid VP8 coefficient partition lengths")
        sizes.append(final)
        offset, self.partitions = table_size, []
        for size in sizes:
            self.partitions.append(_Partition(self.remainder[offset:offset + size]))
            offset += size
        base = fp.uint(7)
        dy1, dy2dc, dy2ac, duvdc, duvac = (fp.optional(4) for _ in range(5))
        self.quant = []
        for segment in range(4):
            q = base
            if self.segmented:
                q = base + self.segment_q[segment] if self.relative else self.segment_q[segment]
            self.quant.append(((dequantTableDC[clip(q + dy1, 0, 127)], dequantTableAC[clip(q, 0, 127)]),
                               (dequantTableDC[clip(q + dy2dc, 0, 127)] * 2,
                                max(8, dequantTableAC[clip(q + dy2ac, 0, 127)] * 155 // 100)),
                               (dequantTableDC[clip(q + duvdc, 0, 117)],
                                dequantTableAC[clip(q + duvac, 0, 127)])))
        fp.bit()  # Refresh entropy probabilities applies to subsequent frames.
        for index, probability in enumerate(tokenProbUpdateProb):
            if fp.bit(probability):
                self.tokenProb[index] = fp.uint(8)
        self.skip_probability = fp.uint(8) if fp.bit() else None

    def _filter_parameters(self):
        result = []
        for segment in range(4):
            base = self.level
            if self.segmented:
                base = self.segment_f[segment] + (self.level if self.relative else 0)
            modes = []
            for inner in (False, True):
                level = base
                if self.delta:
                    level += self.reference_delta[0] + (self.mode_delta[0] if inner else 0)
                level = clip(level, 0, 63)
                interior = level
                if self.sharpness:
                    interior >>= 2 if self.sharpness > 4 else 1
                    interior = min(interior, 9 - self.sharpness)
                interior = max(1, interior)
                high = 0 if level < 15 else 1 if level < 40 else 2
                modes.append((2 * level + interior if level else 0, interior, high, inner))
            result.append(modes)
        return result

    def _predict_modes(self, mx):
        fp = self.fp
        self.use16 = bool(fp.bit(145))
        if self.use16:
            if not fp.bit(156):
                mode = 2 if fp.bit(163) else 0
            else:
                mode = 1 if fp.bit(128) else 3
            self.pred16 = mode
            self.above[mx].predictions[:] = [mode] * 4
            self.left.predictions[:] = [mode] * 4
        else:
            for row in range(4):
                mode = self.left.predictions[row]
                for column in range(4):
                    base = (self.above[mx].predictions[column] * 10 + mode) * 9
                    p = predProb[base:base + 9]
                    if not fp.bit(p[0]):
                        mode = 0
                    elif not fp.bit(p[1]):
                        mode = 1
                    elif not fp.bit(p[2]):
                        mode = 2
                    elif not fp.bit(p[3]):
                        mode = 3 if not fp.bit(p[4]) else 4 if not fp.bit(p[5]) else 5
                    elif not fp.bit(p[6]):
                        mode = 6
                    elif not fp.bit(p[7]):
                        mode = 7
                    else:
                        mode = 8 if not fp.bit(p[8]) else 9
                    self.pred4[row][column] = mode
                    self.above[mx].predictions[column] = mode
                self.left.predictions[row] = mode
        self.pred8 = 0 if not fp.bit(142) else 2 if not fp.bit(114) else 3 if not fp.bit(183) else 1

    def _workspace(self, mx, my):
        for row in range(26):
            if mx:
                self.ybr[row][7] = self.ybr[row][23 if row < 17 else 15]
                if row >= 17:
                    self.ybr[row][23] = self.ybr[row][31]
            else:
                self.ybr[row][7] = 129
                if row >= 17:
                    self.ybr[row][23] = 129
        if not my:
            self.ybr[0][7:28] = bytes([127]) * 21
            self.ybr[17][7:16] = bytes([127]) * 9
            self.ybr[17][23:32] = bytes([127]) * 9
        else:
            start = (my * 16 - 1) * self.ys + mx * 16
            self.ybr[0][8:24] = self.y[start:start + 16]
            over = self.y[start + 16:start + 20] if mx + 1 < self.mbw else self.y[start + 15:start + 16] * 4
            self.ybr[0][24:28] = over
            start = (my * 8 - 1) * self.cs + mx * 8
            self.ybr[17][8:16] = self.u[start:start + 8]
            self.ybr[17][24:32] = self.v[start:start + 8]
        for row in (4, 8, 12):
            self.ybr[row][24:28] = self.ybr[0][24:28]

    def _prob(self, plane, band, context):
        start = (plane * 24 + band * 3 + context) * 11
        return self.tokenProb[start:start + 11]

    def _residual4(self, partition, plane, context, quant, skip_first, base):
        position = int(skip_first)
        p = self._prob(plane, _BANDS[position], context)
        if not partition.bit(p[0]):
            return 0
        while position != 16:
            position += 1
            if not partition.bit(p[1]):
                p = self._prob(plane, _BANDS[position], 0)
                continue
            if not partition.bit(p[2]):
                value = 1
                p = self._prob(plane, _BANDS[position], 1)
            else:
                if not partition.bit(p[3]):
                    value = 2 if not partition.bit(p[4]) else 3 + partition.bit(p[5])
                elif not partition.bit(p[6]):
                    value = (5 + partition.bit(159) if not partition.bit(p[7])
                             else 7 + 2 * partition.bit(165) + partition.bit(145))
                else:
                    high = partition.bit(p[8])
                    category = 2 * high + partition.bit(p[9 + high])
                    value = 0
                    for probability in _CATEGORIES[category]:
                        value = value * 2 + partition.bit(probability)
                    value += 3 + (8 << category)
                p = self._prob(plane, _BANDS[position], 2)
            zigzag = _ZIGZAG[position - 1]
            coefficient = value * quant[int(zigzag > 0)]
            if partition.bit():
                coefficient = -coefficient
            self.coeff[base + zigzag] = signed16(coefficient)
            if position == 16 or not partition.bit(p[0]):
                return 1
        return 1

    def _residuals(self, mx, my):
        partition = self.partitions[my & (self.partition_count - 1)]
        quant1, quant2, quantuv = self.quant[self.segment]
        plane = 3
        if self.use16:
            nz = self._residual4(partition, 1, self.left.y16 + self.above[mx].y16, quant2, False, 384)
            self.left.y16 = self.above[mx].y16 = nz
            inverse_wht(self)
            plane = 0
        left = [(self.left.nz >> shift) & 1 for shift in range(4)]
        above = [(self.above[mx].nz >> shift) & 1 for shift in range(4)]
        dc_mask, ac_mask, base = 0, 0, 0
        for row in range(4):
            nz = left[row]
            for column in range(4):
                nz = self._residual4(partition, plane, nz + above[column], quant1, self.use16, base)
                above[column] = nz
                ac_mask |= nz << (base // 16)
                dc_mask |= int(self.coeff[base] != 0) << (base // 16)
                base += 16
            left[row] = nz
        left_mask = sum(value << shift for shift, value in enumerate(left))
        above_mask = sum(value << shift for shift, value in enumerate(above))
        left = [(self.left.nz >> (shift + 4)) & 1 for shift in range(4)]
        above = [(self.above[mx].nz >> (shift + 4)) & 1 for shift in range(4)]
        for channel in (0, 2):
            for row in range(2):
                nz = left[row + channel]
                for column in range(2):
                    nz = self._residual4(partition, 2, nz + above[column + channel], quantuv, False, base)
                    above[column + channel] = nz
                    ac_mask |= nz << (base // 16)
                    dc_mask |= int(self.coeff[base] != 0) << (base // 16)
                    base += 16
                left[row + channel] = nz
        self.left.nz = left_mask | sum(value << (shift + 4) for shift, value in enumerate(left))
        self.above[mx].nz = above_mask | sum(value << (shift + 4) for shift, value in enumerate(above))
        self.dc_mask, self.ac_mask = dc_mask, ac_mask
        return not (dc_mask or ac_mask)

    @staticmethod
    def _edge_mode(mx, my, mode):
        if mode:
            return mode
        return 12 if mx == my == 0 else 11 if not mx else 10 if not my else 0

    def _reconstruct(self, mx, my):
        if self.update_map:
            if not self.fp.bit(self.segment_p[0]):
                self.segment = self.fp.bit(self.segment_p[1])
            else:
                self.segment = self.fp.bit(self.segment_p[2]) + 2
        skip = bool(self.fp.bit(self.skip_probability)) if self.skip_probability is not None else False
        self.coeff[:] = [0] * 400
        self._workspace(mx, my)
        self._predict_modes(mx)
        if skip:
            if self.use16:
                self.left.y16 = self.above[mx].y16 = 0
            self.left.nz = self.above[mx].nz = 0
            self.dc_mask = self.ac_mask = 0
        else:
            skip = self._residuals(mx, my)
        if self.use16:
            predFunc16[self._edge_mode(mx, my, self.pred16)](self, 1, 8)
        for row in range(4):
            for column in range(4):
                index, y, x = row * 4 + column, row * 4 + 1, column * 4 + 8
                if not self.use16:
                    predFunc4[self.pred4[row][column]](self, y, x)
                if self.ac_mask & (1 << index):
                    inverse4(self, y, x, index * 16)
                elif self.dc_mask & (1 << index):
                    inverse4(self, y, x, index * 16, True)
        mode = self._edge_mode(mx, my, self.pred8)
        for channel, x in enumerate((8, 24)):
            predFunc8[mode](self, 18, x)
            for row in range(2):
                for column in range(2):
                    index = 16 + channel * 4 + row * 2 + column
                    if self.ac_mask & (1 << index):
                        inverse4(self, 18 + row * 4, x + column * 4, index * 16)
                    elif self.dc_mask & (1 << index):
                        inverse4(self, 18 + row * 4, x + column * 4, index * 16, True)
        for row in range(16):
            start = (my * 16 + row) * self.ys + mx * 16
            self.y[start:start + 16] = self.ybr[row + 1][8:24]
        for row in range(8):
            start = (my * 8 + row) * self.cs + mx * 8
            self.u[start:start + 8] = self.ybr[row + 18][8:16]
            self.v[start:start + 8] = self.ybr[row + 18][24:32]
        level, interior, high, inner = self.filter_parameters[self.segment][int(not self.use16)]
        self.filters.append((level, interior, high, inner or not skip))

    def decode(self):
        self.headers()
        self.ys, self.cs = self.mbw * 16, self.mbw * 8
        self.y = bytearray(self.ys * self.mbh * 16)
        self.u = bytearray(self.cs * self.mbh * 8)
        self.v = bytearray(self.cs * self.mbh * 8)
        self.above = [_MB() for _ in range(self.mbw)]
        self.filters = []
        for my in range(self.mbh):
            self.left = _MB()
            for mx in range(self.mbw):
                self._reconstruct(mx, my)
        if self.level:
            loop_filter.apply(self)
        return self


def decode(data, max_pixels=40_000_000):
    """Return width, height, RGB bytes for an independently coded VP8 frame."""
    decoder = _Decoder(data, max_pixels).decode()
    return decoder.width, decoder.height, "RGB", _rgb(decoder)


def _rgb(decoder):
    # Chroma reconstruction and fixed-point color conversion are deliberately
    # kept separate from entropy/reconstruction state for independent checking.
    # The fixed-point stages match libwebp, including its intermediate rounding.
    result = bytearray(decoder.width * decoder.height * 3)
    cw, ch = (decoder.width + 1) // 2, (decoder.height + 1) // 2
    for y in range(decoder.height):
        sy, fy = divmod(2 * y - 1, 4)
        y0, y1 = clip(sy, 0, ch - 1), clip(sy + 1, 0, ch - 1)
        for x in range(decoder.width):
            sx, fx = divmod(2 * x - 1, 4)
            x0, x1 = clip(sx, 0, cw - 1), clip(sx + 1, 0, cw - 1)
            chroma = []
            for plane in (decoder.u, decoder.v):
                top = plane[y0 * decoder.cs + x0] * (4 - fx) + plane[y0 * decoder.cs + x1] * fx
                bottom = plane[y1 * decoder.cs + x0] * (4 - fx) + plane[y1 * decoder.cs + x1] * fx
                chroma.append((top * (4 - fy) + bottom * fy + 8) >> 4)
            luma = (decoder.y[y * decoder.ys + x] * 19077) >> 8
            u, v = chroma
            index = (y * decoder.width + x) * 3
            result[index:index + 3] = bytes((clip((luma + ((v * 26149) >> 8) - 14234) >> 6, 0, 255),
                                             clip((luma - ((u * 6419) >> 8) - ((v * 13320) >> 8) + 8708) >> 6, 0, 255),
                                             clip((luma + ((u * 33050) >> 8) - 17685) >> 6, 0, 255)))
    return bytes(result)
