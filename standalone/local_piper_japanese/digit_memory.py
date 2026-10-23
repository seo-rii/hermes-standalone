"""Byte strings, references and integer operations used by the digit program."""
from dataclasses import dataclass
import operator


class CString:
    __slots__ = ('storage', 'offset')

    def __init__(self, value=b'', offset=0):
        self.storage = value if isinstance(value, bytearray) else bytearray(value) + b'\0'
        self.offset = offset

    def __add__(self, distance):
        return CString(self.storage, self.offset + distance)

    def __radd__(self, distance):
        return self + distance

    def __sub__(self, distance):
        if isinstance(distance, CString):
            assert self.storage is distance.storage
            return self.offset - distance.offset
        return self + -distance

    def __getitem__(self, index):
        return self.storage[self.offset + index]

    def __setitem__(self, index, value):
        self.storage[self.offset + index] = value

    def bytes(self):
        end = self.storage.find(0, self.offset)
        return bytes(self.storage[self.offset:end if end >= 0 else len(self.storage)])

    def write(self, value):
        ending = self.offset + len(value)
        if ending > len(self.storage):
            raise ValueError('original digit character buffer overflow')
        self.storage[self.offset:ending] = value


@dataclass
class Ref:
    container: object
    key: object
    member: bool = False

    def get(self):
        return getattr(self.container, self.key) if self.member else self.container[self.key]

    def set(self, value):
        if self.member:
            setattr(self.container, self.key, value)
        else:
            self.container[self.key] = value
        return value

    def address(self):
        if isinstance(self.container, CString):
            return self.container + self.key
        return self


class CArray(list):
    def __init__(self, values, item_size):
        super().__init__(values)
        self.item_size = item_size


@dataclass(frozen=True)
class TypeSize:
    name: str


def dereference(value):
    return value.get() if isinstance(value, Ref) else value


def text(value):
    return dereference(value).bytes().decode('utf-8')


def size_of(value):
    if isinstance(value, TypeSize):
        return value
    if isinstance(value, CString):
        return len(value.storage) - value.offset
    if isinstance(value, CArray):
        return len(value) * value.item_size
    return 4 if isinstance(value, int) else 8


def integer_divide(left, right):
    quotient = abs(left) // abs(right)
    return -quotient if (left < 0) != (right < 0) else quotient


def integer_remainder(left, right):
    return left - integer_divide(left, right) * right


def equal(left, right):
    if isinstance(left, CString) and isinstance(right, CString):
        return left.storage is right.storage and left.offset == right.offset
    return left == right


BINARY = {
    '+': operator.add, '-': operator.sub, '*': operator.mul,
    '/': integer_divide, '%': integer_remainder,
    '==': equal, '!=': lambda a, b: not equal(a, b),
    '<': operator.lt, '<=': operator.le, '>': operator.gt, '>=': operator.ge,
    '&': operator.and_, '|': operator.or_, '^': operator.xor,
    '<<': operator.lshift, '>>': operator.rshift,
}
