"""Pixel geometry and outline text for browser overlays and video reviews."""

import math


def color(value, mode):
    names = {"red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255),
             "white": (255, 255, 255), "black": (0, 0, 0), "yellow": (255, 255, 0),
             "gray": (128, 128, 128), "grey": (128, 128, 128), "silver": (192, 192, 192),
             "orange": (255, 165, 0), "purple": (128, 0, 128), "pink": (255, 192, 203),
             "cyan": (0, 255, 255), "aqua": (0, 255, 255), "magenta": (255, 0, 255),
             "fuchsia": (255, 0, 255), "lime": (0, 255, 0), "navy": (0, 0, 128),
             "teal": (0, 128, 128), "olive": (128, 128, 0), "maroon": (128, 0, 0),
             "transparent": (0, 0, 0, 0)}
    if isinstance(value, str):
        if value.startswith("#") and len(value) in (4, 5, 7, 9):
            value = value[1:]
            value = tuple(int(c * 2, 16) for c in value) if len(value) in (3, 4) else tuple(int(value[i:i + 2], 16) for i in range(0, len(value), 2))
        else:
            value = names[value.lower()]
    if isinstance(value, int):
        value = (value,) * len(mode)
    if mode in ("L", "LA") and len(value) >= 3:
        gray = (19595 * value[0] + 38470 * value[1] + 7471 * value[2] + 32768) >> 16
        value = (gray,) if mode == "L" else (gray, value[3] if len(value) == 4 else 255)
    if len(mode) == 4 and len(value) == 3:
        value = (*value, 255)
    if len(mode) == 3 and len(value) == 4:
        value = value[:3]
    return value


class Draw:
    def __init__(self, image):
        self.image = image

    def _set(self, x, y, fill):
        if 0 <= x < self.image.width and 0 <= y < self.image.height:
            self.image.putpixel((x, y), fill)

    def rectangle(self, box, fill=None, outline=None, width=1):
        left, top, right, bottom = map(round, box)
        if fill is not None:
            fill = color(fill, self.image.mode)
        if outline is not None:
            outline = color(outline, self.image.mode)
        for y in range(max(0, top), min(self.image.height, bottom + 1)):
            for x in range(max(0, left), min(self.image.width, right + 1)):
                edge = min(x - left, right - x, y - top, bottom - y) < width
                value = outline if outline is not None and edge else fill
                if value is not None:
                    self._set(x, y, value)

    def ellipse(self, box, fill=None, outline=None, width=1):
        left, top, right, bottom = box
        cx, cy = (left + right) / 2, (top + bottom) / 2
        rx, ry = (right - left) / 2, (bottom - top) / 2
        if rx <= 0 or ry <= 0:
            return
        fill = color(fill, self.image.mode) if fill is not None else None
        outline = color(outline, self.image.mode) if outline is not None else None
        for y in range(max(0, math.floor(top)), min(self.image.height, math.ceil(bottom) + 1)):
            for x in range(max(0, math.floor(left)), min(self.image.width, math.ceil(right) + 1)):
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 > 1:
                    continue
                inner = rx > width and ry > width and ((x - cx) / (rx - width)) ** 2 + ((y - cy) / (ry - width)) ** 2 < 1
                value = outline if outline is not None and not inner else fill
                if value is not None:
                    self._set(x, y, value)

    def line(self, points, fill=None, width=1):
        if not points:
            return
        points = list(zip(points[::2], points[1::2])) if isinstance(points[0], (int, float)) else points
        fill, radius = color(fill if fill is not None else "black", self.image.mode), max(0, width - 1) / 2
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            steps = max(1, math.ceil(max(abs(x1 - x0), abs(y1 - y0))))
            for step in range(steps + 1):
                x, y = round(x0 + (x1 - x0) * step / steps), round(y0 + (y1 - y0) * step / steps)
                for dy in range(-math.floor(radius), math.ceil(radius) + 1):
                    for dx in range(-math.floor(radius), math.ceil(radius) + 1):
                        self._set(x + dx, y + dy, fill)

    def rounded_rectangle(self, box, radius=0, fill=None, outline=None, width=1):
        left, top, right, bottom = map(round, box)
        if right < left or bottom < top:
            raise ValueError("Rectangle coordinates must be ordered")
        radius = max(0, min(float(radius), (right - left) / 2, (bottom - top) / 2))
        if not radius:
            return self.rectangle(box, fill=fill, outline=outline, width=width)
        fill = color(fill, self.image.mode) if fill is not None else None
        outline = color(outline, self.image.mode) if outline is not None else None

        def inside(x, y, inset):
            lo, hi, upper, lower = left + inset, right - inset, top + inset, bottom - inset
            if lo > hi or upper > lower or not (lo <= x <= hi and upper <= y <= lower):
                return False
            r = max(0, radius - inset)
            cx = max(lo + r, min(hi - r, x))
            cy = max(upper + r, min(lower - r, y))
            return (x - cx) ** 2 + (y - cy) ** 2 <= r * r

        for y in range(max(0, top), min(self.image.height, bottom + 1)):
            for x in range(max(0, left), min(self.image.width, right + 1)):
                if inside(x, y, 0):
                    value = outline if outline is not None and not inside(x, y, width) else fill
                    if value is not None:
                        self._set(x, y, value)

    def textbbox(self, xy, text, font=None, anchor=None, spacing=4, align="left", direction=None,
                 features=None, language=None, stroke_width=0, embedded_color=False, **options):
        if options or embedded_color:
            raise NotImplementedError("Unsupported outline text option")
        from standalone.raster_font import load_default
        font = font or load_default()
        if "\n" in text:
            return self.multiline_textbbox(xy, text, font, anchor, spacing, align, direction,
                                           features, language, stroke_width)
        box = font.getbbox(text, anchor=anchor, direction=direction, features=features,
                           language=language, stroke_width=stroke_width)
        return xy[0] + box[0], xy[1] + box[1], xy[0] + box[2], xy[1] + box[3]

    def textlength(self, text, font=None, direction=None, features=None, language=None, embedded_color=False):
        if embedded_color:
            raise NotImplementedError("Color glyph layout is unavailable")
        from standalone.raster_font import load_default
        return (font or load_default()).getlength(text, direction=direction, features=features, language=language)

    def text(self, xy, text, fill=None, font=None, anchor=None, spacing=4, align="left", direction=None,
             features=None, language=None, stroke_width=0, stroke_fill=None, embedded_color=False, **options):
        if options or embedded_color:
            raise NotImplementedError("Unsupported outline text option")
        if not text:
            return
        if "\n" in text:
            return self.multiline_text(xy, text, fill, font, anchor, spacing, align, direction,
                                       features, language, stroke_width, stroke_fill)
        from standalone.raster_font import load_default
        font = font or load_default()
        mask, offset = font.render(text, anchor=anchor, direction=direction, features=features, language=language)
        location = round(xy[0] + offset[0]), round(xy[1] + offset[1])
        foreground = color(fill if fill is not None else "white", self.image.mode)
        if stroke_width:
            if type(stroke_width) is not int or not 0 <= stroke_width <= 64:
                raise ValueError("Text stroke width must be an integer from 0 through 64")
            if mask.width * mask.height * (2 * stroke_width + 1) ** 2 > 100_000_000:
                raise ValueError("Text stroke exceeds the drawing pixel budget")
            stroke = color(stroke_fill if stroke_fill is not None else foreground, self.image.mode)
            for dy in range(-stroke_width, stroke_width + 1):
                for dx in range(-stroke_width, stroke_width + 1):
                    if dx * dx + dy * dy <= stroke_width * stroke_width:
                        self.image.paste(stroke, (location[0] + dx, location[1] + dy,
                                                  location[0] + dx + mask.width, location[1] + dy + mask.height), mask)
        self.image.paste(foreground, (*location, location[0] + mask.width, location[1] + mask.height), mask)

    def _lines(self, xy, text, font, spacing, align, direction, features, language):
        from standalone.raster_font import load_default
        font = font or load_default()
        if align not in ("left", "center", "right"):
            raise ValueError("Text alignment must be left, center, or right")
        lines = text.split("\n")
        lengths = [font.getlength(line, direction=direction, features=features, language=language) for line in lines]
        maximum = max(lengths, default=0)
        ascent, descent = font.getmetrics()
        for index, (line, length) in enumerate(zip(lines, lengths)):
            offset = 0 if align == "left" else (maximum - length) / 2 if align == "center" else maximum - length
            yield (xy[0] + offset, xy[1] + index * (ascent + descent + spacing)), line, font

    def multiline_text(self, xy, text, fill=None, font=None, anchor=None, spacing=4, align="left", direction=None,
                       features=None, language=None, stroke_width=0, stroke_fill=None, **options):
        if anchor not in (None, "la") or options:
            raise NotImplementedError("Multiline text currently uses the left ascender anchor")
        for location, line, font in self._lines(xy, text, font, spacing, align, direction, features, language):
            self.text(location, line, fill, font, direction=direction, features=features,
                      language=language, stroke_width=stroke_width, stroke_fill=stroke_fill)

    def multiline_textbbox(self, xy, text, font=None, anchor=None, spacing=4, align="left", direction=None,
                           features=None, language=None, stroke_width=0, **options):
        if anchor not in (None, "la") or options:
            raise NotImplementedError("Multiline text currently uses the left ascender anchor")
        boxes = [self.textbbox(location, line, font, direction=direction, features=features,
                               language=language, stroke_width=stroke_width)
                 for location, line, font in self._lines(xy, text, font, spacing, align, direction, features, language)]
        return (min(box[0] for box in boxes), min(box[1] for box in boxes),
                max(box[2] for box in boxes), max(box[3] for box in boxes))


class ImageDraw:
    Draw = Draw
