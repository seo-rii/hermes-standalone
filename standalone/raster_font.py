"""TrueType/OpenType outline fonts rendered with embedded, pure fontTools.

This backend reads real font outlines, including quadratic/cubic curves and
composite glyphs, and uses an antialiased nonzero-winding scan converter. The
default Aileron subset is the one shipped by Pillow (font-provenance.json).
It does not implement FreeType hinting or HarfBuzz complex-script shaping.
"""

from __future__ import annotations

from enum import IntEnum
from functools import lru_cache
import io
import math
from pathlib import Path
import struct


MAX_FONT_BYTES = 64 * 1024 * 1024
MAX_TEXT_LENGTH = 4096
MAX_MASK_PIXELS = 20_000_000


class FontError(OSError):
    pass


class Layout(IntEnum):
    BASIC = 0
    RAQM = 1


def _text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or len(value) > MAX_TEXT_LENGTH:
        raise ValueError("Text must be a string within the font layout limit")
    return value


def _pen(glyphs, scale):
    from fontTools.pens.basePen import BasePen

    class FlattenPen(BasePen):
        def __init__(self):
            super().__init__(glyphs)
            self.contours, self.current = [], []
            self.edges = self.components = 0

        def _point(self, value):
            return value[0] * scale, -value[1] * scale

        def _append(self, point):
            self.edges += 1
            if self.edges > 100000:
                raise FontError("Font glyph exceeds the outline complexity limit")
            self.current.append(point)

        def _moveTo(self, point):
            self.current = []
            self._append(self._point(point))

        def _lineTo(self, point):
            self._append(self._point(point))

        def _qCurveToOne(self, control, point):
            start, control, end = self.current[-1], self._point(control), self._point(point)
            self._quadratic(start, control, end, 0)

        def _quadratic(self, start, control, end, depth):
            distance = math.hypot(start[0] - 2 * control[0] + end[0],
                                  start[1] - 2 * control[1] + end[1])
            if distance <= 0.3 or depth >= 12:
                self._append(end)
                return
            a = ((start[0] + control[0]) / 2, (start[1] + control[1]) / 2)
            b = ((control[0] + end[0]) / 2, (control[1] + end[1]) / 2)
            middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            self._quadratic(start, a, middle, depth + 1)
            self._quadratic(middle, b, end, depth + 1)

        def _curveToOne(self, control1, control2, point):
            self._cubic(self.current[-1], self._point(control1), self._point(control2), self._point(point), 0)

        def _cubic(self, start, first, second, end, depth):
            flatness = max(math.hypot(start[0] - 2 * first[0] + second[0], start[1] - 2 * first[1] + second[1]),
                           math.hypot(first[0] - 2 * second[0] + end[0], first[1] - 2 * second[1] + end[1]))
            if flatness <= 0.3 or depth >= 12:
                self._append(end)
                return
            midpoint = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            a, b, c = midpoint(start, first), midpoint(first, second), midpoint(second, end)
            d, e = midpoint(a, b), midpoint(b, c)
            middle = midpoint(d, e)
            self._cubic(start, a, d, middle, depth + 1)
            self._cubic(middle, e, c, end, depth + 1)

        def _closePath(self):
            if self.current:
                if self.current[-1] != self.current[0]:
                    self._append(self.current[0])
                self.contours.append(tuple(self.current))
                self.current = []

        _endPath = _closePath

        def addComponent(self, glyphName, transformation):
            self.components += 1
            if self.components > 64:
                raise FontError("Font composite glyph nesting exceeds its limit")
            try:
                super().addComponent(glyphName, transformation)
            finally:
                self.components -= 1

    return FlattenPen()


def _scan(contours, box, supersample):
    from standalone import raster
    left, top, right, bottom = box
    width, height = max(1, right - left), max(1, bottom - top)
    if width * height > MAX_MASK_PIXELS:
        raise FontError("Rendered text exceeds the mask pixel limit")
    coverage = bytearray(width * height)
    edges = []
    for contour in contours:
        for (x0, y0), (x1, y1) in zip(contour, contour[1:]):
            if y0 != y1:
                edges.append((x0, y0, x1, y1))
    if len(edges) > 1_000_000:
        raise FontError("Rendered text exceeds the outline complexity limit")
    for subrow in range(height * supersample):
        y = top + (subrow + 0.5) / supersample
        intersections = []
        for x0, y0, x1, y1 in edges:
            if min(y0, y1) <= y < max(y0, y1):
                x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                intersections.append((x, 1 if y1 > y0 else -1))
        intersections.sort()
        winding, start = 0, None
        for x, delta in intersections:
            old, winding = winding, winding + delta
            if not old and winding:
                start = x
            elif old and not winding:
                first = max(0, math.ceil((start - left) * supersample - 0.5))
                last = min(width * supersample, math.ceil((x - left) * supersample - 0.5))
                base = (subrow // supersample) * width
                for column in range(first, last):
                    coverage[base + column // supersample] += 1
    denominator = supersample * supersample
    pixels = bytes((value * 255 + denominator // 2) // denominator for value in coverage)
    return raster.frombytes("L", (width, height), pixels)


class FreeTypeFont:
    """The image backend's outline-font interface; no native FreeType is used."""

    def __init__(self, font, size=10, index=0, encoding="", layout_engine=None, *, kerning=True):
        from fontTools.ttLib import TTFont, TTLibError
        if isinstance(size, bool) or not isinstance(size, (int, float)) or not 0 < size <= 2048:
            raise ValueError("Font size must be positive and at most 2048 pixels")
        if not math.isfinite(size) or type(index) is not int or index < 0:
            raise ValueError("Invalid font size or collection index")
        if encoding not in ("", "unic"):
            raise NotImplementedError("Only Unicode font encoding is available")
        if layout_engine not in (None, Layout.BASIC):
            raise NotImplementedError("Complex-script font shaping requires a shaping backend")
        self.path = str(font) if isinstance(font, (str, Path)) else None
        if hasattr(font, "read"):
            data = font.read(MAX_FONT_BYTES + 1)
        else:
            path = Path(font)
            if path.stat().st_size > MAX_FONT_BYTES:
                raise FontError("Font exceeds the input size limit")
            data = path.read_bytes()
        if not isinstance(data, bytes) or len(data) > MAX_FONT_BYTES:
            raise FontError("Font exceeds the input size limit")
        if data[:4] == b"wOF2":
            raise FontError("WOFF2 fonts need decompression; provide the source TTF or OTF")
        self._data = data
        try:
            self._font = TTFont(io.BytesIO(data), fontNumber=index, lazy=True, checkChecksums=2)
            self._glyphs = self._font.getGlyphSet()
            self._cmap = self._font.getBestCmap() or {}
            self._metrics = self._font["hmtx"].metrics
            units = self._font["head"].unitsPerEm
            if not 16 <= units <= 16384 or len(self._glyphs) > 65536:
                raise FontError("Invalid font dimensions or glyph count")
            self.size, self.index, self.encoding, self.layout_engine = size, index, encoding, Layout.BASIC
            self._scale = size / units
            self._ascent = math.ceil(self._font["hhea"].ascent * self._scale)
            self._descent = math.ceil(-self._font["hhea"].descent * self._scale)
            self._kerning = kerning
            self._kern = {}
            if "kern" in self._font:
                for table in self._font["kern"].kernTables:
                    if getattr(table, "coverage", 0) & 1:
                        self._kern.update(getattr(table, "kernTable", {}))
            self._gpos = self._pair_tables()
        except FontError:
            raise
        except (TTLibError, KeyError, ValueError, AssertionError, EOFError, IndexError, struct.error) as exc:
            raise FontError("Invalid or unsupported outline font") from exc

    def _pair_tables(self):
        if "GPOS" not in self._font:
            return []
        table = self._font["GPOS"].table
        if not table.FeatureList or not table.LookupList:
            return []
        indices = set()
        for record in table.FeatureList.FeatureRecord:
            if record.FeatureTag == "kern":
                indices.update(record.Feature.LookupListIndex)
        result = []
        for index in sorted(indices):
            lookup = table.LookupList.Lookup[index]
            for subtable in lookup.SubTable:
                kind = lookup.LookupType
                if kind == 9:
                    kind, subtable = subtable.ExtensionLookupType, subtable.ExtSubTable
                if kind == 2:
                    result.append(subtable)
        return result

    def _pair(self, left, right):
        value, found = 0, False
        for table in self._gpos:
            if left not in table.Coverage.glyphs:
                continue
            if table.Format == 1:
                record = table.PairSet[table.Coverage.glyphs.index(left)]
                pair = next((pair for pair in record.PairValueRecord if pair.SecondGlyph == right), None)
            elif table.Format == 2:
                c1 = table.ClassDef1.classDefs.get(left, 0)
                c2 = table.ClassDef2.classDefs.get(right, 0)
                pair = table.Class1Record[c1].Class2Record[c2]
            else:
                continue
            if pair is not None:
                found = True
                value += getattr(pair.Value1, "XAdvance", 0)
        return (value if found else self._kern.get((left, right), 0)) * self._scale

    @lru_cache(maxsize=128)
    def _outline(self, glyph):
        pen = _pen(self._glyphs, self._scale)
        try:
            self._glyphs[glyph].draw(pen)
            if pen.current:
                pen._closePath()
        except FontError:
            raise
        except (KeyError, IndexError, ValueError, RecursionError, AssertionError) as exc:
            raise FontError("Invalid font glyph outline") from exc
        return tuple(pen.contours)

    def _layout(self, text, direction=None, features=None, language=None):
        text = _text(text)
        if direction not in (None, "ltr") or language is not None:
            raise NotImplementedError("Font language and bidirectional shaping are unavailable")
        if features is not None and set(features) - {"kern", "-kern"}:
            raise NotImplementedError("Only the kerning font feature is available")
        kerning = self._kerning and (not features or "-kern" not in features)
        cursor, previous, positions = 0.0, None, []
        for char in text:
            glyph = self._cmap.get(ord(char), ".notdef")
            if glyph not in self._glyphs:
                glyph = next(iter(self._glyphs))
            if previous is not None and kerning:
                cursor += self._pair(previous, glyph)
            positions.append((glyph, cursor))
            cursor += math.floor(self._metrics[glyph][0] * self._scale + 0.5)
            previous = glyph
        return positions, cursor

    def getlength(self, text, mode="", direction=None, features=None, language=None):
        return self._layout(text, direction, features, language)[1]

    def getmetrics(self):
        return self._ascent, self._descent

    def getname(self):
        names = self._font["name"]
        return names.getDebugName(1) or "Unknown", names.getDebugName(2) or "Regular"

    def _geometry(self, text, anchor=None, direction=None, features=None, language=None):
        positions, advance = self._layout(text, direction, features, language)
        contours = []
        points = 0
        for glyph, cursor in positions:
            for contour in self._outline(glyph):
                points += len(contour)
                if points > 1_000_000:
                    raise FontError("Rendered text exceeds the outline complexity limit")
                contours.append(tuple((x + cursor, y) for x, y in contour))
        ink = [point for contour in contours for point in contour]
        if ink:
            left, top = min(x for x, _ in ink), min(y for _, y in ink)
            right, bottom = max(x for x, _ in ink), max(y for _, y in ink)
        else:
            left = right = top = bottom = 0
        anchor = anchor or "la"
        if len(anchor) != 2 or anchor[0] not in "lmr" or anchor[1] not in "atmsbd":
            raise ValueError("Unsupported text anchor")
        dx = {"l": 0, "m": -advance / 2, "r": -advance}[anchor[0]]
        dy = {"a": self._ascent, "t": -top, "m": (self._ascent - self._descent) / 2,
              "s": 0, "b": -bottom, "d": -self._descent}[anchor[1]]
        box = (math.floor(min(0, left) + dx), math.floor(top + dy),
               math.ceil(max(advance, right) + dx), math.ceil(bottom + dy))
        if not positions:
            box = 0, 0, 0, 0
        return contours, box, dx, dy

    def getbbox(self, text, mode="", direction=None, features=None, language=None, stroke_width=0, anchor=None):
        _, box, _, _ = self._geometry(text, anchor, direction, features, language)
        if stroke_width:
            return box[0] - stroke_width, box[1] - stroke_width, box[2] + stroke_width, box[3] + stroke_width
        return box

    def render(self, text, *, anchor=None, direction=None, features=None, language=None):
        contours, box, dx, dy = self._geometry(text, anchor, direction, features, language)
        contours = [tuple((x + dx, y + dy) for x, y in contour) for contour in contours]
        return _scan(contours, box, 4 if self.size <= 64 else 2), box[:2]

    def getmask(self, text, mode="L", **options):
        if mode not in ("", "L"):
            raise NotImplementedError("Outline text masks use grayscale coverage")
        return self.render(text, **options)[0]

    def getmask2(self, text, mode="L", **options):
        if mode not in ("", "L"):
            raise NotImplementedError("Outline text masks use grayscale coverage")
        return self.render(text, **options)

    def font_variant(self, font=None, size=None, index=None, encoding=None, layout_engine=None):
        return FreeTypeFont(font or io.BytesIO(self._data), self.size if size is None else size,
                            self.index if index is None else index, self.encoding if encoding is None else encoding,
                            self.layout_engine if layout_engine is None else layout_engine, kerning=self._kerning)


def truetype(font, size=10, index=0, encoding="", layout_engine=None):
    return FreeTypeFont(font, size, index, encoding, layout_engine)


ImageFont = FreeTypeFont


@lru_cache(maxsize=16)
def load_default(size=None):
    from standalone.raster_fontdata import default_font_bytes
    return FreeTypeFont(io.BytesIO(default_font_bytes()), 10 if size is None else size, kerning=False)
