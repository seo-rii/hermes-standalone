"""Relay metric resource lineage and standard GenAI first-chunk measurement."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import re
import threading

from .relay_metrics import ValidatedMeasurement
from .relay_metrics_aggregate import MetricDiagnostics, MetricProcessor

_RESERVED = ('error.', 'exception.', 'gen_ai.', 'input.', 'llm.', 'nemo_relay.', 'openinference.', 'output.',
             'server.', 'service.', 'session.', 'tool.', 'tool_call.', 'user.')
_SDK_KEYS = ('telemetry.sdk.name', 'telemetry.sdk.language', 'telemetry.sdk.version')


def _metadata_attribute(value):
    if type(value) in (str, bool, float):
        return value
    if type(value) is int:
        if value > (1 << 63) - 1:
            raise ValueError('unsigned integer is larger than OTLP i64')
        return value
    if value is None:
        raise ValueError('null values are not OTLP attributes')
    if isinstance(value, dict):
        raise ValueError('object values are not supported')
    if not value:
        return []
    first = value[0]
    if first is None:
        raise ValueError('arrays of null are not OTLP attributes')
    if isinstance(first, (list, dict)):
        raise ValueError('nested arrays and objects are not supported')
    if type(first) in (str, bool):
        if all(type(item) is type(first) for item in value):
            return list(value)
        raise ValueError('array values must have one primitive type')
    if any(type(item) not in (int, float) for item in value):
        raise ValueError('array values must have one primitive type')
    if any(type(item) is int and item > (1 << 63) - 1 for item in value):
        raise ValueError('array contains an unsigned integer larger than OTLP i64')
    return list(value) if all(type(item) is int for item in value) else [float(item) for item in value]


def configured_resource(config):
    result = {}
    for field in ('name', 'namespace', 'version'):
        if config.get('service_' + field) is not None:
            result['service.' + field] = config['service_' + field]
    result.update(config.get('resource_attributes') or {})
    return result


def _nanoseconds(value):
    stamp = re.fullmatch(r'(.*T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})', value)
    if stamp is None:
        raise ValueError('invalid Relay event timestamp')
    delta = datetime.fromisoformat(stamp[1] + stamp[3].replace('Z', '+00:00')) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000 + int((stamp[2] or '').ljust(9, '0'))


class MetricRouter:
    """Retain at most sixteen dynamic providers and 4096 active resource scopes."""
    def __init__(self, endpoint, settings, *, clock=None, provider_created=None):
        self.endpoint, self.settings = deepcopy(endpoint), deepcopy(settings)
        self.diagnostics = MetricDiagnostics()
        self._clock = {} if clock is None else {'clock': clock}
        self.base = self._new_processor()
        self.pipelines = {}
        self.resources = {}
        self.active, self.completed = {}, {}
        self._lock = threading.RLock()
        self._provider_created = provider_created

    def _new_processor(self):
        return MetricProcessor(temporality=self.settings.get('temporality', 'cumulative'),
            max_instruments=self.settings.get('max_instruments', 256),
            cardinality_limit=self.settings.get('cardinality_limit', 2000), diagnostics=self.diagnostics, **self._clock)

    def _promote(self, event):
        prefixes = self.endpoint.get('promote_resource_metadata_prefixes') or []
        if not prefixes or (event.get('parent_uuid') is not None and event.get('propagation_parent_uuid') == event['parent_uuid']):
            return None
        base = configured_resource(self.endpoint)
        values = dict(base)
        protected = set(base) | set(_SDK_KEYS)
        metadata = event.get('metadata') or {}
        if not isinstance(metadata, dict):
            return None
        for key, value in sorted(metadata.items()):
            if not any(key.startswith(prefix) for prefix in prefixes):
                continue
            reason = None
            if key == 'metadata' or key.startswith(_RESERVED):
                reason = 'attribute key is reserved by Relay or an OpenTelemetry projection'
            elif key not in protected:
                try:
                    values[key] = _metadata_attribute(value)
                except ValueError as error:
                    reason = str(error)
            if reason is not None:
                self.diagnostics.record('otel.resource_metadata_promotion_value_unsupported.' + key,
                    'OpenTelemetry resource_metadata attribute ' + json.dumps(key, ensure_ascii=False) + ' was not promoted: ' + reason)
        if values == base:
            return None
        key = json.dumps(values, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        if key not in self.pipelines:
            if len(self.pipelines) == 16:
                self.diagnostics.record('otel.resource_metadata_pipeline_limit',
                    'OpenTelemetry metrics resource pipeline limit of 16 reached; new resources use the configured base resource')
                return None
            self.pipelines[key] = self._new_processor()
            self.resources[key] = values
            if self._provider_created is not None:
                self._provider_created(self, key)
        return key

    def process(self, event):
        with self._lock:
            now = _nanoseconds(event['timestamp'])
            expired = [identity for identity, (closed, _) in self.completed.items() if now - closed > 60_000_000_000]
            for identity in expired:
                del self.completed[identity]
            phase, identity = event.get('scope_category'), event['uuid']
            parent = event.get('parent_uuid')
            route = self.active.get(parent)
            if route is None and parent in self.completed:
                route = self.completed[parent][1]
            if phase == 'start':
                if identity in self.active:
                    route = self.active[identity]
                else:
                    self.completed.pop(identity, None)
                    if len(self.active) >= 4096:
                        self.diagnostics.record('otel.resource_metadata_active_scope_limit',
                            'OpenTelemetry active resource scope limit of 4096 reached; new scopes use the configured base resource until capacity is available')
                        route = None
                    else:
                        route = route if route is not None else self._promote(event)
                        if route is not None:
                            self.active[identity] = route
            elif phase == 'end':
                route = self.active.pop(identity, None)
                if route is not None:
                    self.completed[identity] = now, route
            processor = self.base if route is None else self.pipelines[route]
            measurement = first_chunk_measurement(event)
            if measurement is not None:
                processor.record_validated((measurement,), event['name'])
            processor.process(event)

    def collect(self):
        with self._lock:
            result = [(configured_resource(self.endpoint), self.base.collect())]
            result.extend((self.resources[key], processor.collect()) for key, processor in self.pipelines.items())
            return [(resource, metrics) for resource, metrics in result if metrics]

    def provider_keys(self):
        with self._lock:
            return (None, *self.pipelines)

    def collect_provider(self, key):
        with self._lock:
            if key is None:
                return configured_resource(self.endpoint), self.base.collect()
            return self.resources[key], self.pipelines[key].collect()


def _scalar(event, keys, kind):
    objects = [(event.get('category_profile') or {})]
    for source in (event.get('metadata'), event.get('data')):
        if isinstance(source, dict):
            objects.append(source)
    for index, source in enumerate(objects):
        for key in keys:
            value = source.get(key)
            if index and key not in source:
                for container in ('usage', 'request', 'response'):
                    if isinstance(source.get(container), dict) and key in source[container]:
                        value = source[container][key]
                        break
            if kind == 'int' and type(value) is int and -(1 << 63) <= value < (1 << 63):
                return value
            if kind == 'str':
                if isinstance(value, str):
                    return value
                if type(value) in (bool, int, float):
                    return json.dumps(value, separators=(',', ':'))
    return None


def first_chunk_measurement(event):
    if event.get('scope_category') != 'end' or event.get('category') != 'llm':
        return None
    profile = event.get('category_profile') or {}
    elapsed = profile.get('time_to_first_chunk')
    if type(elapsed) not in (float, int) or not math.isfinite(elapsed):
        return None
    provider = _scalar(event, ('gen_ai.provider.name', 'provider_name', 'provider'), 'str')
    name = event['name'].lower()
    if provider is None:
        for needle, value in [('azure_ai_inference', 'azure.ai.inference'), ('azure ai inference', 'azure.ai.inference'),
            ('azure_openai', 'azure.ai.openai'), ('azure openai', 'azure.ai.openai'), ('anthropic', 'anthropic'),
            ('claude', 'anthropic'), ('bedrock', 'aws.bedrock'), ('cohere', 'cohere'), ('deepseek', 'deepseek'),
            ('gemini', 'gcp.gemini'), ('vertex', 'gcp.vertex_ai'), ('groq', 'groq'), ('mistral', 'mistral_ai'),
            ('openai', 'openai'), ('gpt', 'openai'), ('perplexity', 'perplexity')]:
            if needle in name:
                provider = value
                break
    request = profile.get('annotated_request') or {}
    response = profile.get('annotated_response') or {}
    if provider is None:
        api = request.get('api_specific') or {}
        provider = {'anthropic_messages': 'anthropic', 'openai_chat': 'openai', 'openai_responses': 'openai',
                    'oci_genai': 'oci.genai'}.get(api.get('api'))
    if provider is None:
        return None
    operation = 'generate_content' if 'generate_content' in name or 'generatecontent' in name else (
        'text_completion' if 'completion' in name and 'chat' not in name else 'chat')
    attributes = {'gen_ai.operation.name': operation, 'gen_ai.provider.name': provider}
    model = request.get('model')
    if model is None:
        model = profile.get('model_name')
    if model is None:
        model = _scalar(event, ('gen_ai.request.model', 'model', 'model_name'), 'str')
    for key, value in [('gen_ai.request.model', model), ('gen_ai.response.model', response.get('model')),
        ('server.address', _scalar(event, ('server.address', 'server_address'), 'str')),
        ('server.port', _scalar(event, ('server.port', 'server_port'), 'int'))]:
        if value is not None:
            attributes[key] = value
    return ValidatedMeasurement('gen_ai.client.operation.time_to_first_chunk', 'histogram', 'f64', float(elapsed),
                                's', None, dict(sorted(attributes.items())), None)
