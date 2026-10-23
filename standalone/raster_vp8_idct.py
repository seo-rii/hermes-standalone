"""VP8 inverse transforms translated from x/image/vp8/idct.go.

Copyright 2011 The Go Authors. All rights reserved.
BSD-3-Clause: see licenses/go-image-LICENSE and vp8-provenance.json.
"""


def signed16(value):
    return ((value + 32768) & 65535) - 32768


def _multiply(value, multiplier):
    product = value * multiplier
    return (((product + (1 << 31)) & 0xffffffff) - (1 << 31)) >> 16


def inverse4(decoder, y, x, base, dc_only=False):
    coefficients, pixels = decoder.coeff, decoder.ybr
    if dc_only:
        delta = (coefficients[base] + 4) >> 3
        for row in range(y, y + 4):
            for column in range(x, x + 4):
                pixels[row][column] = max(0, min(255, pixels[row][column] + delta))
        return
    matrix = [[0] * 4 for _ in range(4)]
    for index in range(4):
        a = coefficients[base + index] + coefficients[base + index + 8]
        b = coefficients[base + index] - coefficients[base + index + 8]
        c = _multiply(coefficients[base + index + 4], 35468) - _multiply(coefficients[base + index + 12], 85627)
        d = _multiply(coefficients[base + index + 4], 85627) + _multiply(coefficients[base + index + 12], 35468)
        matrix[index] = a + d, b + c, b - c, a - d
    for row in range(4):
        dc = matrix[0][row] + 4
        a, b = dc + matrix[2][row], dc - matrix[2][row]
        c = _multiply(matrix[1][row], 35468) - _multiply(matrix[3][row], 85627)
        d = _multiply(matrix[1][row], 85627) + _multiply(matrix[3][row], 35468)
        for column, value in enumerate((a + d, b + c, b - c, a - d)):
            pixels[y + row][x + column] = max(0, min(255, pixels[y + row][x + column] + (value >> 3)))


def inverse_wht(decoder):
    coefficients = decoder.coeff
    matrix = [0] * 16
    for index in range(4):
        a = coefficients[384 + index] + coefficients[396 + index]
        b = coefficients[388 + index] + coefficients[392 + index]
        c = coefficients[388 + index] - coefficients[392 + index]
        d = coefficients[384 + index] - coefficients[396 + index]
        matrix[index], matrix[8 + index] = a + b, a - b
        matrix[4 + index], matrix[12 + index] = d + c, d - c
    for row in range(4):
        dc = matrix[row * 4] + 3
        a, b = dc + matrix[row * 4 + 3], matrix[row * 4 + 1] + matrix[row * 4 + 2]
        c, d = matrix[row * 4 + 1] - matrix[row * 4 + 2], dc - matrix[row * 4 + 3]
        for column, value in enumerate((a + b, d + c, a - b, d - c)):
            coefficients[row * 64 + column * 16] = signed16(value >> 3)
