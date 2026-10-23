"""Typed synchronous Relay metric aggregation, translated from the pinned Rust SDK."""
from bisect import bisect_left
from copy import deepcopy
import hashlib
import json
import struct
import threading
import time

from .relay_metrics import _WHITESPACE, classify_metric_mark

DEFAULT_BOUNDS = (0., 5., 10., 25., 50., 75., 100., 250., 500., 750., 1000., 2500., 5000., 7500., 10000.)
_U64_MASK = (1 << 64) - 1
_OVERFLOW = {'otel.metric.overflow': True}


def parse_temporality(value):
    if not isinstance(value, str):
        raise ValueError('invalid metric temporality')
    result = value.strip(_WHITESPACE).lower()
    if result == 'lowmemory':
        result = 'low_memory'
    if result not in ('cumulative', 'delta', 'low_memory'):
        raise ValueError('invalid metric temporality')
    return result


def attribute_fingerprint(attributes):
    """Relay's typed SHA256 identity, including IEEE signed-zero bits."""
    if not attributes:
        return None
    digest = hashlib.sha256()

    def count(value):
        digest.update(struct.pack('>Q', value))

    def text(value):
        data = value.encode('utf-8')
        count(len(data))
        digest.update(data)

    for key, value in sorted(attributes.items()):
        text(key)
        values = value if isinstance(value, list) else [value]
        typ = {str: b's', bool: b'b', int: b'i', float: b'f'}[type(values[0])]
        digest.update(typ.upper() if isinstance(value, list) else typ)
        if isinstance(value, list):
            count(len(value))
        for item in values:
            if typ == b's':
                text(item)
            elif typ == b'b':
                digest.update(bytes([item]))
            else:
                digest.update(struct.pack('>q' if typ == b'i' else '>d', item))
    return digest.digest()


class MetricDiagnostics:
    """Bounded, sorted diagnostics with native saturating counts and messages."""
    def __init__(self):
        self._values = {}
        self._lock = threading.RLock()

    def record(self, code, message, count=1):
        message = message if len(message) <= 1024 else message[:1023] + '…'
        with self._lock:
            existing = self._values.get(code)
            if existing is None:
                if len(self._values) == 32:
                    return 0
                existing = self._values[code] = {'count': 0, 'messages': set()}
            existing['count'] = min(_U64_MASK, existing['count'] + max(1, count))
            messages = existing['messages'] | {message}
            if len('\n'.join(sorted(messages))) <= 1024:
                existing['messages'] = messages
            return existing['count']

    def snapshot(self):
        with self._lock:
            return [dict(code=code, count=value['count'], message='\n'.join(sorted(value['messages'])))
                    for code, value in sorted(self._values.items())]


def _wrap(value, value_type):
    if value_type == 'f64':
        return value
    value &= _U64_MASK
    return value - (1 << 64) if value_type == 'i64' and value >= (1 << 63) else value


class _Series:
    def __init__(self, attributes, descriptor, bounds):
        self.attributes, self.updated = deepcopy(attributes), False
        self.reset(descriptor, bounds)

    def reset(self, descriptor, bounds):
        self.value = 0.0 if descriptor.value_type == 'f64' else 0
        self.count = 0
        self.buckets = [0] * (len(bounds) + 1) if bounds else []
        self.minimum = float('inf')
        self.maximum = float('-inf')

    def update(self, descriptor, bounds, value):
        self.updated = True
        if descriptor.kind == 'gauge':
            self.value = value
        else:
            self.value = _wrap(self.value + value, descriptor.value_type)
        if descriptor.kind == 'histogram':
            self.count = (self.count + 1) & _U64_MASK
            if self.buckets:
                index = bisect_left(bounds, float(value))
                self.buckets[index] = (self.buckets[index] + 1) & _U64_MASK
            if value < self.minimum:
                self.minimum = value
            if value > self.maximum:
                self.maximum = value

    def point(self, descriptor, bounds, start, current):
        result = {'attributes': deepcopy(self.attributes), 'start_time_unix_nano': start, 'time_unix_nano': current}
        if descriptor.kind == 'histogram':
            result.update(count=self.count, sum=float(self.value), min=float(self.minimum), max=float(self.maximum),
                          explicit_bounds=list(bounds), bucket_counts=list(self.buckets))
        elif descriptor.value_type == 'f64':
            result['as_double'] = self.value
        else:
            value = self.value
            # The pinned Rust OTLP transform uses i64::try_from(u64).unwrap_or_default().
            result['as_int'] = 0 if value >= (1 << 63) else value
        return result


class _Instrument:
    def __init__(self, descriptor, temporality, cardinality_limit, clock):
        self.descriptor = descriptor
        self.bounds = DEFAULT_BOUNDS if descriptor.boundaries is None else descriptor.boundaries
        self.delta = (temporality == 'delta' and descriptor.kind != 'up_down_counter') or (
            temporality == 'low_memory' and descriptor.kind in ('counter', 'histogram'))
        self.start = clock()
        self.cardinality_limit = cardinality_limit
        self.series = {}
        self.series_count = 0
        self.attribute_sets = set()

    def record(self, measurement):
        attributes = measurement.attributes
        key = attribute_fingerprint(attributes)
        if key not in self.series:
            if key is None:
                self.series[key] = _Series(attributes, self.descriptor, self.bounds)
            elif self.series_count < self.cardinality_limit:
                self.series[key] = _Series(attributes, self.descriptor, self.bounds)
                self.series_count += 1
            else:
                key = attribute_fingerprint(_OVERFLOW)
                if key not in self.series:
                    self.series[key] = _Series(_OVERFLOW, self.descriptor, self.bounds)
        self.series[key].update(self.descriptor, self.bounds, measurement.value)

    def collect(self, name, clock):
        descriptor = self.descriptor
        current = clock()
        points = []
        for key, series in tuple(self.series.items()):
            if not self.delta or series.updated:
                points.append(series.point(descriptor, self.bounds, self.start, current))
                if self.delta:
                    series.updated = False
                    series.reset(descriptor, self.bounds)
            elif key is not None and series.attributes != _OVERFLOW:
                del self.series[key]
                self.series_count -= 1
        if self.delta:
            self.start = current
        if not points:
            return None
        points.sort(key=lambda value: json.dumps(value['attributes'], sort_keys=True))
        kind = {'counter': 'sum', 'up_down_counter': 'sum', 'gauge': 'gauge', 'histogram': 'histogram'}[descriptor.kind]
        result = dict(name=name, description=descriptor.description or '', unit=descriptor.unit or '', type=kind, data_points=points)
        if kind in ('sum', 'histogram'):
            result['aggregation_temporality'] = 1 if self.delta else 2
        if kind == 'sum':
            result['is_monotonic'] = descriptor.kind == 'counter'
        return result


class MetricProcessor:
    """One independently owned meter; preflight every mark before any update."""
    def __init__(self, temporality='cumulative', max_instruments=256, cardinality_limit=2000, *, clock=time.time_ns, diagnostics=None):
        self.temporality = parse_temporality(temporality)
        if max_instruments <= 0 or cardinality_limit <= 0 or cardinality_limit >= _U64_MASK:
            raise ValueError('metric instrument and cardinality limits must be positive bounded integers')
        self.max_instruments, self.cardinality_limit = max_instruments, cardinality_limit
        self.clock, self.diagnostics = clock, diagnostics if diagnostics is not None else MetricDiagnostics()
        self.instruments, self.rejected_marks = {}, 0
        self._lock = threading.RLock()

    def _reject(self, name, code, message):
        self.rejected_marks = min(_U64_MASK, self.rejected_marks + 1)
        self.diagnostics.record(code, 'OpenTelemetry metric mark ' + json.dumps(name, ensure_ascii=False)
                                + ' was dropped atomically: ' + message)

    def process(self, event):
        try:
            measurements = classify_metric_mark(event)
        except ValueError as error:
            with self._lock:
                self._reject(event['name'], 'otel.metric_mark_invalid', str(error))
            return False
        return False if measurements is None else self.record_validated(measurements, event['name'])

    def record_validated(self, measurements, mark_name='metric'):
        with self._lock:
            proposed = {}
            for measurement in measurements:
                key = measurement.name.lower()
                existing = self.instruments.get(key)
                if existing is not None and existing.descriptor.identity != measurement.identity:
                    self._reject(mark_name, 'otel.metric_descriptor_conflict', 'metric ' + json.dumps(measurement.name, ensure_ascii=False)
                                 + ' conflicts with its existing instrument descriptor')
                    return False
                proposed.setdefault(key, measurement)
            new_count = sum(key not in self.instruments for key in proposed)
            if len(self.instruments) + new_count > self.max_instruments:
                self._reject(mark_name, 'otel.metric_instrument_limit', 'metric mark exceeds the endpoint limit of '
                             + str(self.max_instruments) + ' distinct instruments')
                return False
            for key, measurement in proposed.items():
                if key not in self.instruments:
                    self.instruments[key] = _Instrument(measurement, self.temporality, self.cardinality_limit, self.clock)
            for measurement in measurements:
                instrument = self.instruments[measurement.name.lower()]
                fingerprint = attribute_fingerprint(measurement.attributes)
                if fingerprint is not None and fingerprint not in instrument.attribute_sets:
                    if len(instrument.attribute_sets) >= self.cardinality_limit:
                        self.diagnostics.record('otel.metric_cardinality_limit', 'OpenTelemetry metric '
                            + json.dumps(measurement.name, ensure_ascii=False) + ' exceeded the endpoint cardinality limit of '
                            + str(self.cardinality_limit) + '; additional attribute sets use the SDK overflow series')
                    else:
                        instrument.attribute_sets.add(fingerprint)
                instrument.record(measurement)
            return True

    def collect(self):
        with self._lock:
            result = [instrument.collect(name, self.clock) for name, instrument in sorted(self.instruments.items())]
            return [metric for metric in result if metric is not None]
