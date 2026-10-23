"""VP8 loop filtering translated from x/image/vp8/filter.go.

Copyright 2014 The Go Authors. All rights reserved.
BSD-3-Clause: see licenses/go-image-LICENSE and vp8-provenance.json.
"""


def clip(value, low, high):
    return max(low, min(high, value))


def filter2(pixels, level, index, along, across):
    for _ in range(16):
        p1, p0 = pixels[index - 2 * across], pixels[index - across]
        q0, q1 = pixels[index], pixels[index + across]
        if 2 * abs(p0 - q0) + abs(p1 - q1) // 2 <= level:
            delta = 3 * (q0 - p0) + clip(p1 - q1, -128, 127)
            a = clip((delta + 4) >> 3, -16, 15)
            b = clip((delta + 3) >> 3, -16, 15)
            pixels[index - across] = clip(p0 + b, 0, 255)
            pixels[index] = clip(q0 - a, 0, 255)
        index += along


def filter246(pixels, count, level, inner, high, index, along, across, four):
    for _ in range(count):
        p3, p2, p1, p0 = (pixels[index - offset * across] for offset in (4, 3, 2, 1))
        q0, q1, q2, q3 = (pixels[index + offset * across] for offset in (0, 1, 2, 3))
        boundary = 2 * abs(p0 - q0) + abs(p1 - q1) // 2 <= level
        interior = max(abs(p3 - p2), abs(p2 - p1), abs(p1 - p0),
                       abs(q1 - q0), abs(q2 - q1), abs(q3 - q2)) <= inner
        if boundary and interior:
            if abs(p1 - p0) > high or abs(q1 - q0) > high:
                delta = 3 * (q0 - p0) + clip(p1 - q1, -128, 127)
                a, b = clip((delta + 4) >> 3, -16, 15), clip((delta + 3) >> 3, -16, 15)
                pixels[index - across] = clip(p0 + b, 0, 255)
                pixels[index] = clip(q0 - a, 0, 255)
            elif four:
                delta = 3 * (q0 - p0)
                a, b = clip((delta + 4) >> 3, -16, 15), clip((delta + 3) >> 3, -16, 15)
                c = (a + 1) >> 1
                pixels[index - 2 * across] = clip(p1 + c, 0, 255)
                pixels[index - across] = clip(p0 + b, 0, 255)
                pixels[index] = clip(q0 - a, 0, 255)
                pixels[index + across] = clip(q1 - c, 0, 255)
            else:
                delta = clip(3 * (q0 - p0) + clip(p1 - q1, -128, 127), -128, 127)
                a, b, c = ((factor * delta + 63) >> 7 for factor in (27, 18, 9))
                pixels[index - 3 * across] = clip(p2 + c, 0, 255)
                pixels[index - 2 * across] = clip(p1 + b, 0, 255)
                pixels[index - across] = clip(p0 + a, 0, 255)
                pixels[index] = clip(q0 - a, 0, 255)
                pixels[index + across] = clip(q1 - b, 0, 255)
                pixels[index + 2 * across] = clip(q2 - c, 0, 255)
        index += along


def apply(decoder):
    for my in range(decoder.mbh):
        for mx in range(decoder.mbw):
            level, inner, high, include_inner = decoder.filters[my * decoder.mbw + mx]
            if not level:
                continue
            yi = (my * decoder.ys + mx) * 16
            ci = (my * decoder.cs + mx) * 8
            if decoder.simple:
                if mx:
                    filter2(decoder.y, level + 4, yi, decoder.ys, 1)
                if include_inner:
                    for offset in (4, 8, 12):
                        filter2(decoder.y, level, yi + offset, decoder.ys, 1)
                if my:
                    filter2(decoder.y, level + 4, yi, 1, decoder.ys)
                if include_inner:
                    for offset in (4, 8, 12):
                        filter2(decoder.y, level, yi + offset * decoder.ys, 1, decoder.ys)
            else:
                planes = ((decoder.y, 16, yi, decoder.ys),
                          (decoder.u, 8, ci, decoder.cs), (decoder.v, 8, ci, decoder.cs))
                for pixels, size, start, stride in planes:
                    if mx:
                        filter246(pixels, size, level + 4, inner, high, start, stride, 1, False)
                    if include_inner:
                        for offset in range(4, size, 4):
                            filter246(pixels, size, level, inner, high, start + offset, stride, 1, True)
                for pixels, size, start, stride in planes:
                    if my:
                        filter246(pixels, size, level + 4, inner, high, start, 1, stride, False)
                    if include_inner:
                        for offset in range(4, size, 4):
                            filter246(pixels, size, level, inner, high, start + offset * stride, 1, stride, True)
