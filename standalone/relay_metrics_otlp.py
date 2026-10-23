"""Owned OTLP metric providers using official messages and stdlib transports."""
from copy import deepcopy
import ipaddress
import logging
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit

from .relay_metrics_route import MetricRouter
from .relay_metrics_aggregate import parse_temporality


def _value(value):
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue
    if type(value) is bool:
        return AnyValue(bool_value=value)
    if type(value) is int:
        return AnyValue(int_value=value)
    if type(value) is float:
        return AnyValue(double_value=value)
    if isinstance(value, list):
        return AnyValue(array_value=ArrayValue(values=[_value(item) for item in value]))
    return AnyValue(string_value=value)


def _attributes(values):
    from opentelemetry.proto.common.v1.common_pb2 import KeyValue
    return [KeyValue(key=key, value=_value(value)) for key, value in sorted(values.items())]


def resource_attributes(values):
    """Preserve resource env/config precedence and report the actual implementation."""
    service = os.environ.get('OTEL_SERVICE_NAME')
    defaults = {'service.name':service if service else 'unknown_service:' + Path(sys.executable).name}
    for entry in os.environ.get('OTEL_RESOURCE_ATTRIBUTES', '').split(','):
        if '=' in entry:
            key, value = entry.split('=', 1)
            defaults[key.strip()] = value.strip()
    defaults.update({'telemetry.sdk.name':'hermes-standalone', 'telemetry.sdk.language':'python', 'telemetry.sdk.version':'0.10.0-stdlib'})
    defaults.update(values)
    return defaults


def serialize_metrics(metrics, values, endpoint):
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
    from opentelemetry.proto.common.v1.common_pb2 import InstrumentationScope
    from opentelemetry.proto.metrics.v1.metrics_pb2 import (
        Gauge, Histogram, HistogramDataPoint, Metric, NumberDataPoint, ResourceMetrics, ScopeMetrics, Sum)
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource
    output = []
    for row in metrics:
        points = []
        for original in row['data_points']:
            point = dict(original)
            point['attributes'] = _attributes(point['attributes'])
            points.append((HistogramDataPoint if row['type'] == 'histogram' else NumberDataPoint)(**point))
        if row['type'] == 'gauge':
            data = Gauge(data_points=points)
        elif row['type'] == 'sum':
            data = Sum(data_points=points, aggregation_temporality=row['aggregation_temporality'], is_monotonic=row['is_monotonic'])
        else:
            data = Histogram(data_points=points, aggregation_temporality=row['aggregation_temporality'])
        output.append(Metric(name=row['name'], description=row['description'], unit=row['unit'], **{row['type']:data}))
    message = ExportMetricsServiceRequest(resource_metrics=[ResourceMetrics(resource=Resource(attributes=_attributes(resource_attributes(values))),
        scope_metrics=[ScopeMetrics(scope=InstrumentationScope(name=endpoint.get('instrumentation_scope', 'opentelemetry')), metrics=output)])])
    return message.SerializeToString()


def _destination_key(endpoint):
    from standalone.relay_plugin import _url
    from standalone.relay_otlp import _signal_url
    url = _url(endpoint['endpoint'].strip())
    if endpoint.get('transport', 'http_binary') == 'http_binary':
        url = _signal_url(url, 'metrics')
    parts = urlsplit(url)
    host = parts.hostname
    domain = host.removesuffix('.')
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = domain == 'localhost' or domain.endswith('.localhost')
    if local:
        host = '<loopback>'
    elif not host.isascii():
        host = host.encode('idna').decode('ascii')
    path = '/'.join(part for part in parts.path.split('/') if part)
    path = '/' + path
    return endpoint.get('transport', 'http_binary'), parts.scheme, host, parts.port or (443 if parts.scheme == 'https' else 80), path, (
        parts.query if '?' in url.split('#', 1)[0] else None)


def resolve_metric_endpoints(config):
    from standalone.relay_otlp import _signal_url
    section = config.get('metrics') or {}
    explicit = section.get('endpoints')
    source = explicit if explicit is not None else config.get('endpoints', config.get('traces', []))
    if not isinstance(source, list):
        raise ValueError('Relay metrics endpoints must be an array')
    if not source:
        raise ValueError('Enabled Relay metrics requires at least one explicit endpoint' if explicit is not None else
                         'Enabled Relay metrics has no explicit or derivable trace endpoints')
    endpoints, skipped = [], []
    for index, original in enumerate(source):
        if not isinstance(original, dict):
            raise ValueError('Relay metrics endpoint must be an object')
        endpoint = deepcopy(original)
        endpoint['_metric_index'] = index
        url = endpoint.get('endpoint', '')
        if not isinstance(url, str):
            raise ValueError('Relay metrics endpoint must be a URL string')
        if endpoint.get('transport', 'http_binary') == 'http_binary':
            path = urlsplit(url.strip()).path
            if explicit is None:
                if path not in ('', '/') and not path.endswith('/v1/traces'):
                    skipped.append('OpenTelemetry endpoints[' + str(index) + '] cannot derive metrics from its configured path; configure metrics.endpoints explicitly')
                    continue
                endpoint['endpoint'] = _signal_url(url.strip(), 'metrics')
            elif any(path.rstrip('/').endswith('/v1/' + signal) for signal in ('traces', 'logs')):
                raise ValueError('Explicit Relay metrics endpoint must select metrics or a bare authority')
        endpoints.append(endpoint)
    destinations = set()
    for endpoint in endpoints:
        destination = _destination_key(endpoint)
        if destination in destinations:
            raise ValueError('Relay metrics endpoints must use distinct destinations')
        destinations.add(destination)
    return endpoints, skipped


def validate_metrics(config):
    from standalone.relay_plugin import _header_destination, _headers
    section = config.get('metrics') or {}
    if not section.get('enabled'):
        return []
    for name, default in [('export_interval_millis',60000), ('max_instruments',256), ('cardinality_limit',2000)]:
        value = section.get(name, default)
        if type(value) is not int or value <= 0 or value > (1 << 64) - 1 or (name == 'cardinality_limit' and value == (1 << 64) - 1):
            raise ValueError('Relay metrics ' + name + ' must be a positive bounded integer')
    parse_temporality(section.get('temporality', 'cumulative'))
    endpoints, _ = resolve_metric_endpoints(config)
    for endpoint in endpoints:
        if not isinstance(endpoint, dict) or endpoint.get('transport', 'http_binary') not in ('http_binary', 'grpc'):
            raise ValueError('Relay metrics requires HTTP/protobuf or gRPC transport')
        _header_destination(endpoint.get('endpoint', ''), endpoint, signal='metrics')
        _headers(endpoint)
        timeout = endpoint.get('timeout_millis', 3000)
        if type(timeout) is not int or timeout <= 0 or timeout > (1 << 64) - 1:
            raise ValueError('Relay metrics timeout_millis must be a positive integer')
        attributes = endpoint.get('resource_attributes') or {}
        if not isinstance(attributes, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in attributes.items()):
            raise ValueError('Relay metrics resource_attributes must contain strings')
        if any(key in attributes for key in ('telemetry.sdk.name', 'telemetry.sdk.language', 'telemetry.sdk.version')):
            raise ValueError('Relay metrics telemetry.sdk resource attributes are supplied by the implementation')
        prefixes = endpoint.get('promote_resource_metadata_prefixes') or []
        if not isinstance(prefixes, list) or any(not isinstance(prefix, str) for prefix in prefixes) or len(set(prefixes)) != len(prefixes):
            raise ValueError('Relay metrics metadata promotion prefixes must be a unique array')
        for prefix in prefixes:
            segments = prefix.removesuffix('.').split('.') if isinstance(prefix, str) else []
            if not segments or any(not segment or any(not (char.isascii() and (char.isalnum() or char in '_-')) for char in segment) for segment in segments):
                raise ValueError('Relay metrics metadata promotion prefixes must be literal dotted names')
    return endpoints


class OtlpMetricsSink:
    """Own each endpoint's aggregation and collection/export lifecycle."""
    def __init__(self, config):
        # Fail at activation if the declared official protobuf closure is absent.
        from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
        endpoints = validate_metrics(config)
        self.config = deepcopy(config.get('metrics') or {})
        if self.config.get('enabled'):
            _, self.skipped_endpoints = resolve_metric_endpoints(config)
            for message in self.skipped_endpoints:
                logging.getLogger(__name__).warning('%s', message)
        else:
            self.skipped_endpoints = []
        self._stop, self._lock = threading.Event(), threading.RLock()
        self._closed = False
        self._workers, self._provider_locks = {}, {}
        self.routers = [MetricRouter(endpoint, self.config, provider_created=self._start_provider) for endpoint in deepcopy(endpoints)]
        self._worker = None
        for router in self.routers:
            self._start_provider(router, None)

    def __call__(self, event):
        value = event.to_dict() if hasattr(event, 'to_dict') else deepcopy(event)
        with self._lock:
            if self._closed:
                return
            for router in self.routers:
                router.process(value)

    def _start_provider(self, router, key):
        with self._lock:
            if self._closed:
                return
            identity = id(router), key
            self._provider_locks[identity] = threading.RLock()
            worker = threading.Thread(target=self._periodic, args=(router, key), name='relay-metrics-periodic', daemon=True)
            self._workers[identity] = worker
            self._worker = self._worker or worker
            worker.start()

    def _periodic(self, router, key):
        interval = self.config.get('export_interval_millis', 60000) / 1000
        due = time.monotonic() + interval
        while not self._stop.is_set():
            if self._stop.wait(min(threading.TIMEOUT_MAX, max(0, due - time.monotonic()))):
                return
            if time.monotonic() < due:
                continue
            try:
                self._collect_provider(router, key)
            except (OSError, ValueError):
                pass  # Each provider retains its diagnostics and next collection.
            due += interval

    def _collect_provider(self, router, key):
        from standalone.relay_plugin import _header_destination, _post
        from standalone.relay_otlp import _signal_url
        with self._provider_locks[id(router), key]:
            resource, metrics = router.collect_provider(key)
            if not metrics:
                return
            endpoint = router.endpoint
            try:
                payload = serialize_metrics(metrics, resource, endpoint)
                url = endpoint['endpoint']
                _header_destination(url, endpoint, signal='metrics')
                if endpoint.get('transport', 'http_binary') == 'grpc':
                    from standalone.relay_grpc import post
                    post(url, payload, endpoint, 'metrics')
                else:
                    _post(_signal_url(url, 'metrics'), payload, endpoint, 'application/x-protobuf', no_redirect=True)
            except (OSError, ValueError) as error:
                router.diagnostics.record('otel.metrics_export_failed',
                    'OpenTelemetry metric export to endpoint ' + endpoint['endpoint'] + ' failed: ' + str(error))
                raise

    def _collect_export(self, final=False):
        errors = []
        for router in self.routers:
            keys = router.provider_keys()
            if final:
                keys = (*keys[1:], None)
            for key in keys:
                try:
                    self._collect_provider(router, key)
                except (OSError, ValueError) as error:
                    errors.append(error)
        if errors:
            raise errors[0]

    def flush(self):
        if not self._closed:
            self._collect_export()

    def close(self):
        self._stop.set()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            workers = tuple(self._workers.values())
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join()
        self._collect_export(final=True)

    def runtime_diagnostics(self):
        return [entry for router in self.routers for entry in router.diagnostics.snapshot()]

    def delivery_failure_summary(self):
        failures = sum(entry['count'] for entry in self.runtime_diagnostics() if entry['code'] == 'otel.metrics_export_failed')
        return 'otel.metrics_export_failed (' + str(failures) + ')' if failures else None

    def plugin_diagnostics(self):
        return [{**entry, 'component':'observability',
                 'field':'opentelemetry.metrics.endpoints[' + str(router.endpoint['_metric_index']) + '].endpoint'}
                for router in self.routers for entry in router.diagnostics.snapshot()]
