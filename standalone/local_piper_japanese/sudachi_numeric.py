"""Sudachi's source string arithmetic and kanji/Arabic numeric path parser."""


class StringNumber:
    def __init__(self):
        self.clear()

    def clear(self):
        self.significand = ''
        self.scale = 0
        self.point = -1
        self.is_all_zero = True

    def append(self, value):
        if value:
            self.is_all_zero = False
        self.significand += str(value)

    def shift_scale(self, value):
        if self.is_zero():
            self.significand += '1'
        self.scale += value

    def add(self, number):
        if number.is_zero():
            return True
        if self.is_zero():
            self.significand += number.significand
            self.scale, self.point = number.scale, number.point
            return True
        self.normalize_scale()
        length = number.int_length()
        if self.scale < length:
            return False
        self.significand += '0' * (self.scale - length)
        if number.point >= 0:
            self.point = len(self.significand) + number.point
        self.significand += number.significand
        self.scale = number.scale
        return True

    def set_point(self):
        if self.scale == 0 and self.point < 0:
            self.point = len(self.significand)
            return True
        return False

    def int_length(self):
        self.normalize_scale()
        return self.point if self.point >= 0 else len(self.significand) + self.scale

    def is_zero(self):
        return not self.significand

    def normalized(self):
        if self.is_zero():
            return '0'
        self.normalize_scale()
        if self.scale > 0:
            self.significand += '0' * self.scale
        elif self.point >= 0:
            self.significand = self.significand[:self.point] + '.' + self.significand[self.point:]
            if self.point == 0:
                self.significand = '0' + self.significand
            self.significand = self.significand.rstrip('0').removesuffix('.')
        return self.significand

    def normalize_scale(self):
        if self.point < 0:
            return
        scale = len(self.significand) - self.point
        if scale > self.scale:
            self.point += self.scale
            self.scale = 0
        else:
            self.scale -= scale
            self.point = -1


CHAR_TO_NUM = {char: index for index, char in enumerate('〇一二三四五六七八九')}
CHAR_TO_NUM.update({str(value): value for value in range(10)})
CHAR_TO_NUM.update(十=-1, 百=-2, 千=-3, 万=-4, 億=-8, 兆=-12)


class NumericParser:
    def __init__(self):
        self.total = StringNumber()
        self.subtotal = StringNumber()
        self.tmp = StringNumber()
        self.clear()

    def clear(self):
        self.digit_length = 0
        self.is_first_digit = True
        self.has_comma = False
        self.has_hanging_point = False
        self.error = None
        self.total.clear()
        self.subtotal.clear()
        self.tmp.clear()

    def check_comma(self):
        if self.is_first_digit:
            return False
        if not self.has_comma:
            return self.digit_length <= 3 and not self.tmp.is_zero() and not self.tmp.is_all_zero
        return self.digit_length == 3

    def _point(self):
        self.has_hanging_point = True
        if self.is_first_digit:
            self.error = 'Point'
            return False
        if self.has_comma and not self.check_comma():
            self.error = 'Comma'
            return False
        if not self.tmp.set_point():
            self.error = 'Point'
            return False
        self.has_comma = False
        return True

    def _comma(self):
        if not self.check_comma():
            self.error = 'Comma'
            return False
        self.has_comma = True
        self.digit_length = 0
        return True

    def _unit(self, value):
        if value >= -3:
            self.tmp.shift_scale(-value)
            if not self.subtotal.add(self.tmp):
                return False
            self.tmp.clear()
        else:
            if not self.subtotal.add(self.tmp) or self.subtotal.is_zero():
                return False
            self.subtotal.shift_scale(-value)
            if not self.total.add(self.subtotal):
                return False
            self.subtotal.clear()
            self.tmp.clear()
        self.is_first_digit = True
        self.digit_length = 0
        self.has_comma = False
        return True

    def append(self, char):
        if char == '.':
            return self._point()
        if char == ',':
            return self._comma()
        value = CHAR_TO_NUM.get(char)
        if value is None:
            return False
        if value < 0:
            return self._unit(value)
        self.tmp.append(value)
        self.is_first_digit = False
        self.digit_length += 1
        self.has_hanging_point = False
        return True

    def done(self):
        result = self.subtotal.add(self.tmp) and self.total.add(self.subtotal)
        if self.has_hanging_point:
            self.error = 'Point'
            return False
        if self.has_comma and self.digit_length != 3:
            self.error = 'Comma'
            return False
        return result

    def normalized(self):
        return self.total.normalized()
