"""Qualified Relay 0.10.0 typed metric schema; no native SDK namespace."""
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math
import re

METRIC_SCHEMA = {'name': 'nemo.relay.metric_measurements', 'version': '1'}
_MIN_I64, _MAX_I64, _MAX_U64 = -(1 << 63), (1 << 63) - 1, (1 << 64) - 1
_WHITESPACE = '\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000'


class _MetricEnum(Enum):
    def __int__(self):
        return self.value

    def __eq__(self, other):
        if type(other) is type(self):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        return NotImplemented

    __hash__ = None

    def __repr__(self):
        return type(self).__name__ + '.' + self.name

    __str__ = __repr__


class MetricKind(_MetricEnum):
    Counter = 0
    UpDownCounter = 1
    Gauge = 2
    Histogram = 3


class MetricValueType(_MetricEnum):
    U64 = 0
    I64 = 1
    F64 = 2


_KINDS = ('counter', 'up_down_counter', 'gauge', 'histogram')
_TYPES = ('u64', 'i64', 'f64')


def _json(value, active=None):
    if value is None or type(value) in (bool, str):
        return value
    if isinstance(value, int):
        if not _MIN_I64 <= value <= _MAX_U64:
            raise ValueError('Failed to convert to JSON: JSON number out of range')
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dict, list, tuple)):
        active = set() if active is None else active
        identity = id(value)
        if identity in active:
            raise ValueError('Failed to convert to JSON: circular reference detected')
        active.add(identity)
        try:
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError('Failed to convert to JSON: key must be a string')
                return {key: _json(item, active) for key, item in value.items()}
            return [_json(item, active) for item in value]
        finally:
            active.remove(identity)
    raise ValueError('Failed to convert to JSON: unsupported type: ' + type(value).__name__)


class MetricMeasurement:
    """Owned immutable measurement; publication validates the complete envelope."""
    __slots__ = ('_fields', '_kind', '_value_type')

    def __init__(self, name, kind, value_type, value, unit=None, description=None, attributes=None, boundaries=None):
        for argument in (name, unit, description):
            if argument is not None and not isinstance(argument, str):
                raise TypeError("'" + type(argument).__name__ + "' object is not an instance of 'str'")
        if name is None:
            raise TypeError("'NoneType' object is not an instance of 'str'")
        if not isinstance(kind, MetricKind):
            raise TypeError("argument 'kind': '" + type(kind).__name__ + "' object cannot be converted to 'MetricKind'")
        if not isinstance(value_type, MetricValueType):
            raise TypeError("argument 'value_type': '" + type(value_type).__name__ + "' object cannot be converted to 'MetricValueType'")
        if isinstance(boundaries, str):
            raise TypeError("Can't extract `str` to `Vec`")
        if boundaries is not None:
            boundaries = [float(item) for item in boundaries]
        self._kind, self._value_type = kind, value_type
        self._fields = {'name': name, 'kind': _KINDS[int(kind)], 'value_type': _TYPES[int(value_type)],
                        'value': _json(value), 'unit': unit, 'description': description,
                        'attributes': _json(attributes), 'boundaries': boundaries}

    @property
    def name(self):
        return self._fields['name']

    @property
    def kind(self):
        return self._kind

    @property
    def value_type(self):
        return self._value_type

    @property
    def value(self):
        return deepcopy(self._fields['value'])

    @property
    def unit(self):
        return self._fields['unit']

    @property
    def description(self):
        return self._fields['description']

    @property
    def attributes(self):
        return deepcopy(self._fields['attributes'])

    @property
    def boundaries(self):
        return deepcopy(self._fields['boundaries'])

    def to_dict(self):
        return deepcopy(self._fields)

    def __repr__(self):
        import json
        return '<MetricMeasurement name=' + json.dumps(self.name, ensure_ascii=False) + ' kind=' + self._fields['kind'] + ' value_type=' + self._fields['value_type'] + '>'


def _attribute(value):
    if type(value) in (str, bool):
        return value
    if type(value) is int:
        if _MIN_I64 <= value <= _MAX_I64:
            return value
        raise ValueError('attribute values must not exceed the maximum signed 64-bit integer')
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('must be finite')
        return value
    if value is None:
        raise ValueError('attribute values must not be null')
    if isinstance(value, list):
        if not value:
            raise ValueError('attribute arrays must not be empty and untyped')
        first = _attribute(value[0])
        if isinstance(first, list):
            raise ValueError('attribute arrays must contain primitive values')
        if type(first) is float:
            valid = all((type(item) is float and math.isfinite(item))
                        or (type(item) is int and _MAX_I64 < item <= _MAX_U64) for item in value)
            if valid:
                return [float(item) for item in value]
        elif all(type(item) is type(first) and (type(item) is not int or _MIN_I64 <= item <= _MAX_I64) for item in value):
            return list(value)
        raise ValueError('attribute arrays must contain one primitive type')
    raise ValueError('attribute values must be primitive values or homogeneous primitive arrays')


@dataclass(frozen=True)
class ValidatedMeasurement:
    name: str
    kind: str
    value_type: str
    value: int | float
    unit: str | None
    description: str | None
    attributes: dict
    boundaries: tuple[float, ...] | None

    @property
    def identity(self):
        return self.kind, self.value_type, self.unit


def _validate(wire):
    name, kind, value_type = wire['name'], wire['kind'], wire['value_type']
    if not isinstance(name, str) or re.fullmatch(r'[A-Za-z][A-Za-z0-9_.\-/]{0,254}', name) is None:
        raise ValueError("name must be 1-255 ASCII bytes, start with a letter, and contain only letters, digits, '_', '.', '-', or '/'")
    boundaries = wire.get('boundaries')
    if boundaries is not None:
        if len(boundaries) > 64:
            raise ValueError('boundaries must contain at most 64 entries')
        if any(not math.isfinite(value) for value in boundaries):
            raise ValueError('must be finite')
        if any(first >= second for first, second in zip(boundaries, boundaries[1:])):
            raise ValueError('boundaries must be strictly increasing without duplicates')
        if kind != 'histogram':
            raise ValueError('boundaries are only valid for histogram measurements')
    unit = wire.get('unit')
    if unit is not None and (not unit.isascii() or len(unit) > 63):
        raise ValueError('unit must be ASCII and at most 63 bytes')
    value = wire['value']
    if value_type == 'u64':
        if type(value) is not int or not 0 <= value <= _MAX_I64:
            raise ValueError('value must be an unsigned integer no greater than i64::MAX')
    elif value_type == 'i64':
        if type(value) is not int or not _MIN_I64 <= value <= _MAX_I64:
            raise ValueError('value must be a signed integer')
    else:
        if type(value) not in (int, float):
            raise ValueError('value must be a number')
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('must be finite')
    if kind == 'counter' and value_type == 'f64' and value < 0:
        raise ValueError('counter values must be non-negative')
    supported = {'counter': ('u64', 'f64'), 'up_down_counter': ('i64', 'f64'),
                 'gauge': ('u64', 'i64', 'f64'), 'histogram': ('u64', 'f64')}
    if value_type not in supported[kind]:
        raise ValueError('kind ' + kind + ' does not support value_type ' + value_type)
    attributes = wire.get('attributes')
    if attributes is not None and not isinstance(attributes, dict):
        raise ValueError('attributes must be a JSON object')
    result = {}
    for key, item in sorted((attributes or {}).items()):
        if not key.strip(_WHITESPACE):
            raise ValueError('attributes contains a blank attribute key')
        result[key] = _attribute(item)
    return ValidatedMeasurement(name, kind, value_type, value, unit, wire.get('description'), result,
                                None if boundaries is None else tuple(boundaries))


def validate_measurements(measurements):
    if not measurements:
        raise ValueError('measurements must contain at least one entry')
    parsed = []
    for index, wire in enumerate(measurements):
        try:
            parsed.append(_validate(wire))
        except ValueError as error:
            raise ValueError('measurements[' + str(index) + '] ' + str(error)) from error
    descriptors = {}
    for index, measurement in enumerate(parsed):
        key = measurement.name.lower()
        previous = descriptors.get(key)
        if previous is not None and parsed[previous].identity != measurement.identity:
            raise ValueError('measurements[' + str(index) + '] conflicts with the descriptor for measurements[' + str(previous) + ']')
        descriptors[key] = index
    return tuple(parsed)


def metric_payload(measurements):
    rows = []
    for measurement in measurements:
        if not isinstance(measurement, MetricMeasurement):
            raise TypeError("'" + type(measurement).__name__ + "' object is not an instance of 'MetricMeasurement'")
        rows.append(measurement.to_dict())
    try:
        validate_measurements(rows)
    except ValueError as error:
        raise ValueError('invalid argument: ' + str(error)) from error
    return {'measurements': rows}


def classify_metric_mark(event):
    if event.get('scope_category') is not None or (event.get('data_schema') or {}).get('name') != METRIC_SCHEMA['name']:
        return None
    version = event['data_schema'].get('version')
    if version != '1':
        import json
        raise ValueError('unsupported metric schema version ' + json.dumps(version))
    payload = event.get('data')
    if payload is None:
        raise ValueError('metric mark data is missing')
    try:
        if isinstance(payload, list):
            if len(payload) != 1:
                expected = 'struct MetricEnvelope with 1 element' if len(payload) < 1 else 'fewer elements in array'
                raise ValueError('invalid length ' + str(len(payload)) + ', expected ' + expected)
            measurements = payload[0]
        elif isinstance(payload, dict):
            if set(payload) - {'measurements'}:
                field = sorted(set(payload) - {'measurements'})[0]
                raise ValueError('unknown field `' + field + '`, expected `measurements`')
            if 'measurements' not in payload:
                raise ValueError('missing field `measurements`')
            measurements = payload['measurements']
        else:
            raise ValueError('invalid type: ' + _unexpected(payload) + ', expected struct MetricEnvelope')
        if not isinstance(measurements, list):
            raise ValueError('invalid type: ' + _unexpected(measurements) + ', expected a sequence')
        parsed = [_measurement_fields(wire) for wire in measurements]
    except ValueError as error:
        raise ValueError('invalid metric envelope: ' + str(error)) from error
    return validate_measurements(parsed)


def _unexpected(value):
    import json
    if value is None:
        return 'null'
    if type(value) is bool:
        return 'boolean `' + str(value).lower() + '`'
    if type(value) in (int, float):
        return ('integer `' if type(value) is int else 'floating point `') + repr(value) + '`'
    if isinstance(value, str):
        return 'string ' + json.dumps(value, ensure_ascii=False)
    return 'map' if isinstance(value, dict) else 'sequence'


def _enum_field(value, variants):
    unit = None
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError('invalid value: map, expected map with a single key')
        value, unit = next(iter(value.items()))
    if not isinstance(value, str):
        raise ValueError('invalid type: ' + _unexpected(value) + ', expected string or map')
    if value not in variants:
        raise ValueError('unknown variant `' + value + '`, expected one of ' + ', '.join('`' + item + '`' for item in variants))
    if unit is not None:
        raise ValueError('invalid type: ' + _unexpected(unit) + ', expected unit')
    return value


def _measurement_fields(wire):
    fields = ('name', 'kind', 'value_type', 'value', 'unit', 'description', 'attributes', 'boundaries')
    if isinstance(wire, list):
        if len(wire) != 8:
            expected = 'struct MetricMeasurement with 8 elements' if len(wire) < 8 else 'fewer elements in array'
            raise ValueError('invalid length ' + str(len(wire)) + ', expected ' + expected)
        wire = dict(zip(fields, wire))
    elif not isinstance(wire, dict):
        raise ValueError('invalid type: ' + _unexpected(wire) + ', expected struct MetricMeasurement')
    result = {}
    for key, value in sorted(wire.items()):
        if key not in fields:
            raise ValueError('unknown field `' + key + '`, expected one of ' + ', '.join('`' + item + '`' for item in fields))
        if key in ('name', 'unit', 'description'):
            if not isinstance(value, str) and (key == 'name' or value is not None):
                raise ValueError('invalid type: ' + _unexpected(value) + ', expected a string')
        elif key in ('kind', 'value_type'):
            value = _enum_field(value, _KINDS if key == 'kind' else _TYPES)
        elif key == 'boundaries' and value is not None:
            if not isinstance(value, list):
                raise ValueError('invalid type: ' + _unexpected(value) + ', expected a sequence')
            for item in value:
                if type(item) not in (int, float):
                    raise ValueError('invalid type: ' + _unexpected(item) + ', expected f64')
            value = [float(item) for item in value]
        result[key] = value
    for field in fields[:4]:
        if field not in result:
            raise ValueError('missing field `' + field + '`')
    return result
