"""Relay OTLP protobuf export using embedded official protobuf messages.

HTTP uses urllib and gRPC uses the embedded pure Python HTTP/2 implementation.
Activation remains owned by Relay plugin config.
"""

from datetime import datetime
import json
from urllib.parse import urlsplit, urlunsplit
import uuid

from standalone.relay_plugin import _post


def _nanoseconds(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1_000_000_000)


def _value(value):
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue
    if type(value) is bool:
        return AnyValue(bool_value=value)
    if type(value) is int and -(1 << 63) <= value < (1 << 63):
        return AnyValue(int_value=value)
    if type(value) is float:
        return AnyValue(double_value=value)
    return AnyValue(string_value=value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _attributes(values):
    from opentelemetry.proto.common.v1.common_pb2 import KeyValue
    return [KeyValue(key=str(key), value=_value(value)) for key, value in values.items() if value is not None]


def _resource(config):
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource
    values = {"service.name": config.get("service_name", "unknown_service"), **config.get("resource_attributes", {})}
    for field in ("namespace", "version"):
        if config.get("service_" + field):
            values["service." + field] = config["service_" + field]
    return Resource(attributes=_attributes(values))


def _signal_url(url, signal):
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return url
    path = parts.path
    if signal == "traces":
        # The native trace exporter treats an explicit trailing slash as a
        # deliberate collector-root destination. Logs/metrics derive paths
        # from the parsed root or a terminal /v1/traces suffix instead.
        if path:
            return url
        path = "/v1/traces"
    elif path in ("", "/"):
        path = "/v1/" + signal
    elif path.endswith("/v1/traces"):
        path = path[:-len("/v1/traces")] + "/v1/" + signal
    else:
        return url
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


class OtlpSink:
    def __init__(self, config):
        # Import the original generated types now so an absent declared closure
        # fails activation rather than losing the first emitted span.
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
        self.config, self.starts, self.roots, self.parents, self.marks, self.pending = config, {}, {}, {}, {}, []
        self.files, self.metrics = [], None
        try:
            if (config.get("metrics") or {}).get("enabled"):
                from standalone.relay_metrics_otlp import OtlpMetricsSink
                self.metrics = OtlpMetricsSink(config)
            from standalone.relay_files import FileDestination, validate_destinations
            for destination in validate_destinations(config.get('file_sinks', [])):
                self.files.append((destination, FileDestination(destination)))
        except BaseException:
            if self.metrics is not None:
                self.metrics.close()
            for _, destination in self.files:
                destination.close()
            raise

    def __call__(self, event):
        if self.metrics is not None:
            self.metrics(event)
        value = event.to_dict()
        identity, parent = value["uuid"], value.get("parent_uuid")
        if value["kind"] == "mark":
            # Reserved metric marks never become trace events or log records.
            if (value.get("data_schema") or {}).get("name") == "nemo.relay.metric_measurements":
                return
            self.marks.setdefault(parent, []).append(value)
            self._logs(value)
            return
        if value["scope_category"] == "start":
            self.starts[identity] = value
            self.parents[identity] = parent
            self.roots[identity] = self.roots.get(parent, parent or identity)
            return
        start = self.starts.pop(identity, None)
        if start is None:
            return
        for endpoint in self.config.get("endpoints", self.config.get("traces", [])):
            if endpoint.get("type", "full") != "full" and value.get("category") != "llm":
                continue
            self._trace(start, value, endpoint)
        for config, destination in self.files:
            if config.get("type", "full") != "full" and value.get("category") != "llm":
                continue
            self._trace(start, value, config, destination)
        self.marks.pop(identity, None)
        self.roots.pop(identity, None)
        self.parents.pop(identity, None)

    def _trace(self, start, end, endpoint, destination=None):
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
        from opentelemetry.proto.common.v1.common_pb2 import InstrumentationScope
        from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status
        identity = uuid.UUID(start["uuid"])
        trace = uuid.UUID(self.roots.get(start["uuid"], start["uuid"]))
        attributes = {"nemo.relay.uuid": start["uuid"], "nemo.relay.category": start.get("category"),
                      **(start.get("metadata") or {}), **(end.get("metadata") or {})}
        profile = start.get("category_profile") or {}
        if start.get("category") == "llm":
            attributes.update({"gen_ai.operation.name": "chat", "gen_ai.request.model": profile.get("model_name")})
            request = (start.get("data") or {}).get("content", {}) if isinstance(start.get("data"), dict) else {}
            if request.get("messages") is not None:
                attributes["gen_ai.input.messages"] = request["messages"]
            if end.get("data") is not None:
                attributes["gen_ai.output.messages"] = end["data"]
            if endpoint.get("type") == "openinference":
                attributes["openinference.span.kind"] = "LLM"
                attributes["llm.model_name"] = profile.get("model_name")
        elif endpoint.get("type", "full") == "full":
            attributes.update({"nemo.relay.input": start.get("data"), "nemo.relay.output": end.get("data")})
        events = []
        for mark in self.marks.get(start["uuid"], []):
            if mark["name"] in endpoint.get("mark_exclude_names", ["llm.chunk"]):
                continue
            events.append(Span.Event(time_unix_nano=_nanoseconds(mark["timestamp"]), name=mark["name"],
                                     attributes=_attributes({**(mark.get("metadata") or {}), "data": mark.get("data")})))
        span = Span(trace_id=trace.bytes, span_id=identity.bytes[-8:],
                    parent_span_id=uuid.UUID(start["parent_uuid"]).bytes[-8:] if start.get("parent_uuid") else b"",
                    name=start["name"], kind=Span.SPAN_KIND_CLIENT if start.get("category") == "llm" else Span.SPAN_KIND_INTERNAL,
                    start_time_unix_nano=_nanoseconds(start["timestamp"]), end_time_unix_nano=_nanoseconds(end["timestamp"]),
                    attributes=_attributes(attributes), events=events,
                    status=Status(code=Status.STATUS_CODE_ERROR if ((end.get("metadata") or {}).get("error")
                        or (end.get("metadata") or {}).get("otel.status_code") == "ERROR") else Status.STATUS_CODE_UNSET))
        message = ExportTraceServiceRequest(resource_spans=[ResourceSpans(resource=_resource(endpoint), scope_spans=[
            ScopeSpans(scope=InstrumentationScope(name=endpoint.get("instrumentation_scope", "opentelemetry")), spans=[span])])])
        self.pending.append((endpoint.get("endpoint"), message.SerializeToString(), endpoint, "traces", destination))
        self._flush_pending()

    def _logs(self, event):
        section = self.config.get("logs") or {}
        if not section.get("enabled") or event["name"] == "llm.chunk":
            return
        from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
        from opentelemetry.proto.common.v1.common_pb2 import InstrumentationScope
        from opentelemetry.proto.logs.v1.logs_pb2 import LogRecord, ResourceLogs, ScopeLogs
        levels = {"trace": 1, "debug": 5, "info": 9, "warn": 13, "warning": 13, "error": 17}
        severity = str((event.get("metadata") or {}).get("nemo.relay.log.severity", "info")).lower()
        if levels.get(severity, 9) < levels.get(section.get("minimum_severity", "info"), 9):
            return
        endpoints = section.get("endpoints")
        if endpoints is None:
            endpoints = self.config.get("endpoints", self.config.get("traces", []))
        for endpoint in endpoints:
            record = LogRecord(time_unix_nano=_nanoseconds(event["timestamp"]), severity_number=levels.get(severity, 9),
                               severity_text=severity.upper(), body=_value(event["name"]),
                               attributes=_attributes({**(event.get("metadata") or {}), "data": event.get("data")}))
            message = ExportLogsServiceRequest(resource_logs=[ResourceLogs(resource=_resource(endpoint), scope_logs=[
                ScopeLogs(scope=InstrumentationScope(name=endpoint.get("instrumentation_scope", "opentelemetry")), log_records=[record])])])
            self.pending.append((endpoint["endpoint"], message.SerializeToString(), endpoint, "logs", None))
        self._flush_pending()

    def _flush_pending(self):
        while self.pending:
            url, payload, endpoint, signal, destination = self.pending[0]
            transport = endpoint.get("transport", "http_binary")
            if destination is not None:
                destination.write(payload)
            elif transport == "grpc":
                from standalone.relay_grpc import post
                post(url, payload, endpoint, signal)
            elif transport == "http_binary":
                from standalone.relay_plugin import _header_destination
                target = _signal_url(url, signal)
                _header_destination(target, endpoint, signal=signal)
                _post(target, payload, endpoint, "application/x-protobuf", no_redirect=True)
            else:
                raise ValueError("Unknown Relay OTLP transport")
            self.pending.pop(0)

    def flush(self):
        errors = []
        try:
            self._flush_pending()
        except (OSError, ValueError) as error:
            errors.append(error)
        if self.metrics is not None:
            try:
                self.metrics.flush()
            except (OSError, ValueError) as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def close(self):
        metric_error = None
        if self.metrics is not None:
            try:
                self.metrics.close()
            except (OSError, ValueError) as error:
                metric_error = error
        # Pending trace/file delivery remains retryable. Metric collection has
        # the native reset-before-export policy and is never replayed here.
        self._flush_pending()
        for _, destination in self.files:
            destination.close()
        summary = self.metrics.delivery_failure_summary() if self.metrics is not None else None
        if summary:
            from standalone.relay_plugin import DeliveryFailure
            raise DeliveryFailure('OpenTelemetry runtime delivery failures: ' + summary)
        if metric_error is not None:
            raise metric_error

    def runtime_diagnostics(self):
        if self.metrics is None:
            return []
        return self.metrics.plugin_diagnostics()
