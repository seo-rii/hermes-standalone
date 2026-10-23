"""Digit-local implementation of original NJD node allocation and C helpers."""
from dataclasses import dataclass
import re

from .digit_memory import CString, Ref, TypeSize, dereference, text
from .njd import NJDNode, STRING_FIELDS, load_feature


@dataclass(eq=False)
class NumberSequence:
    start: object = None
    end: object = None
    next: object = None


def allocate(count, kind):
    assert count == 1 and isinstance(kind, TypeSize)
    return {'NJDNode': NJDNode, 'NJDNumberSequence': NumberSequence}[kind.name]()


def initialize(node):
    node = dereference(node)
    for name, value in NJDNode().values().items():
        setattr(node, name, value)
    node.prev = node.next = None


def load_node(node, feature):
    node = dereference(node)
    feature = text(feature)
    fresh = load_feature(feature)
    if len(fresh) != 1:
        raise ValueError('compound digit node not present in pinned source')
    original_flag = node.chain_flag
    fields = feature.split(',')
    for name, value in fresh[0].values().items():
        setattr(node, name, value)
    if len(fields) < 13 or fields[12] not in ('0', '1'):
        node.chain_flag = original_flag


def insert_node(previous, following, node):
    node.prev, node.next = previous, following
    previous.next, following.prev = node, node
    return node


def getter(name):
    def get(node):
        value = dereference(node).get(name)
        return CString(value.encode()) if isinstance(value, str) else value
    return get


def setter(name):
    def set_value(node, value):
        node = dereference(node)
        if name in STRING_FIELDS and isinstance(value, CString):
            value = text(value)
        node.set(name, value)
    return set_value


def compare(left, right):
    a, b = left.bytes(), right.bytes()
    return (a > b) - (a < b)


def compare_prefix(left, right, count):
    a = bytes(left.storage[left.offset:left.offset + count]).split(b'\0', 1)[0]
    b = bytes(right.storage[right.offset:right.offset + count]).split(b'\0', 1)[0]
    return (a > b) - (a < b)


def substring(left, right):
    position = left.bytes().find(right.bytes())
    return None if position < 0 else left + position


def copy_string(target, source):
    target.write(source.bytes() + b'\0')
    return target


def copy_prefix(target, source, count):
    value = source.bytes()[:count]
    target.write(value + b'\0' * (count - len(value)))
    return target


def append_string(target, source):
    (target + len(target.bytes())).write(source.bytes() + b'\0')
    return target


def format_string(target, capacity, format_, *values):
    arguments = tuple(value.bytes() if isinstance(value, CString) else value for value in values)
    value = format_.bytes() % arguments
    if capacity:
        target.write(value[:capacity - 1] + b'\0')
    return len(value)


def parse_integer(value):
    match = re.match(rb'\s*[+-]?\d+', value.bytes())
    return int(match.group()) if match else 0


def builtins():
    result = {
        'calloc': allocate, 'free': lambda value: None,
        'NJDNode_initialize': initialize, 'NJDNode_clear': initialize,
        'NJDNode_load': load_node, 'NJDNode_insert': insert_node,
        'NJD_remove_node': lambda njd, node: njd.remove(node),
        'NJD_remove_silent_node': lambda njd: njd.remove_silent(),
        'strlen': lambda value: len(value.bytes()), 'strcmp': compare,
        'strncmp': compare_prefix, 'strstr': substring,
        'strcpy': copy_string, 'strncpy': copy_prefix, 'strcat': append_string,
        'snprintf': format_string, 'atoi': parse_integer,
    }
    for name in NJDNode().values():
        result['NJDNode_get_' + name] = getter(name)
        result['NJDNode_set_' + name] = setter(name)
    return result
