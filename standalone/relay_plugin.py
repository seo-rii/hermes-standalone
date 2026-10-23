"""Relay plugin configuration and actual opt-in exporters for the qualified port.

Layer ordering and list merging follow pinned nemo-relay 0.10.0 source. Native
engines and transports outside this boundary produce explicit diagnostics;
validation never loads a plugin or opens an exporter. No installer is invoked.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import ipaddress
import os
from pathlib import Path
import re
import stat
import threading
import tomllib
from urllib import request as http
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
import uuid

from standalone.relay_types import json_copy


_LOCK, _ACTIVE = threading.RLock(), None


@dataclass
class ComponentSpec:
    kind: str
    config: object = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self):
        return {"kind": self.kind, "enabled": self.enabled, "config": _document(self.config)}


@dataclass
class PluginConfig:
    version: int = 1
    components: list = field(default_factory=list)
    policy: object = field(default_factory=dict)

    def to_dict(self):
        return {"version": self.version, "components": [_document(item) for item in self.components],
                "policy": _document(self.policy)}


def _document(value):
    if hasattr(value, "to_dict"):
        return _document(value.to_dict())
    if isinstance(value, dict):
        return {key: _document(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_document(item) for item in value]
    return json_copy(value)


def _merge(left, right, *, component=False, path=(), observability=False):
    if isinstance(left, dict) and isinstance(right, dict):
        result = deepcopy(left)
        for key, value in right.items():
            if not path and not component and key == "components" and isinstance(result.get(key), list) and isinstance(value, list):
                base, used = list(result[key]), {}
                for item in value:
                    kind = item.get("kind") if isinstance(item, dict) else None
                    matches = [index for index, existing in enumerate(base) if isinstance(existing, dict) and existing.get("kind") == kind]
                    ordinal = used.get(kind, 0)
                    used[kind] = ordinal + 1
                    if kind is not None and ordinal < len(matches):
                        index = matches[ordinal]
                        result[key][index] = _merge(result[key][index], item, component=True, observability=kind == "observability")
                    else:
                        result[key].append(deepcopy(item))
            elif key in result:
                nested = () if component and not path and key == "config" else (*path, key)
                result[key] = _merge(result[key], value, component=component, path=nested, observability=observability)
            else:
                result[key] = deepcopy(value)
        return result
    concatenate = component and (len(path) == 1 or (observability and path in {
        ("atof", "sinks"), ("opentelemetry", "endpoints"), ("opentelemetry", "traces"), ("atif", "storage"),
        ("opentelemetry", "logs", "endpoints"), ("opentelemetry", "metrics", "endpoints")}))
    if isinstance(left, list) and isinstance(right, list) and concatenate:
        clears = path in {("opentelemetry", "logs", "endpoints"), ("opentelemetry", "metrics", "endpoints")}
        return deepcopy(right if not right and clears else right + left)
    return deepcopy(right)


def _config_paths(explicit):
    if explicit is not None:
        user = Path(explicit).expanduser()
    elif "XDG_CONFIG_HOME" in os.environ:
        user = Path(os.environ["XDG_CONFIG_HOME"]) / "nemo-relay/plugins.toml"
    elif home := os.environ.get("HOME", os.environ.get("USERPROFILE")):
        user = Path(home) / ".config/nemo-relay/plugins.toml"
    else:
        user = None
    system = (Path(os.environ.get("ProgramData", "C:\\ProgramData")) / "nemo-relay/plugins.toml"
              if os.name == "nt" else Path("/etc/nemo-relay/plugins.toml"))
    paths = []
    for path in (user, system):
        if path is not None and path not in paths:
            paths.append(path)
    return paths


def _resolve(config, explicit=None):
    combined, paths = {}, []
    for path in _config_paths(explicit):
        if not path.exists():
            if explicit is not None and path == Path(explicit).expanduser():
                raise FileNotFoundError(path)
            continue
        with path.open("rb") as source:
            document = tomllib.load(source)
        components = document.get("components", [])
        if not isinstance(components, list):
            raise ValueError("Relay components must be an array")
        kinds = [item.get("kind") for item in components if isinstance(item, dict)]
        if len(kinds) != len(set(kinds)):
            raise ValueError("Duplicate Relay component kind in " + str(path))
        combined = _merge(combined, document)
        paths.append(str(path))
    return _merge(combined, _document(config)), paths


def _redact(value, *, secret=False):
    if isinstance(value, dict):
        return {key: _redact(item, secret=secret or key in {"headers", "authorization", "token", "password", "secret"})
                for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret=secret) for item in value]
    return "<redacted>" if secret and value is not None else deepcopy(value)


def _report(document, paths):
    diagnostics, dynamic, resolved = [], [], deepcopy(document)
    resolved.setdefault("version", 1)
    resolved.setdefault("components", [])

    def issue(message, code="standalone.relay.unsupported", level="error"):
        diagnostics.append({"level": level, "code": code, "message": message})

    if type(resolved["version"]) is not int or resolved["version"] != 1:
        issue("Relay plugin configuration requires version = 1", "plugin.invalid_version")
    components = resolved["components"]
    if not isinstance(components, list):
        issue("Relay components must be an array", "plugin.invalid_components")
        components = []
    for component in components:
        if not isinstance(component, dict) or not isinstance(component.get("kind"), str):
            issue("Relay component requires a kind", "plugin.invalid_component")
            continue
        component.setdefault("enabled", True)
        if type(component["enabled"]) is not bool:
            issue("Relay component enabled must be Boolean", "plugin.invalid_component")
            continue
        if not component["enabled"]:
            continue
        kind, config = component["kind"], component.get("config", {})
        if kind == "pricing":
            from standalone.relay_pricing import PricingResolver
            try:
                PricingResolver(config)
            except ValueError as error:
                issue(str(error), "pricing.invalid_config")
            if sum(isinstance(item, dict) and item.get("enabled", True) and item.get("kind") == "pricing" for item in components) > 1:
                issue("Relay pricing permits only one enabled component", "plugin.duplicate_component")
            continue
        if kind != "observability":
            if kind in {"adaptive", "pii_redaction", "model_pricing", "resource_metrics", "nemo_guardrails"}:
                issue(f"Relay component {kind!r} requires an engine outside the standalone Relay boundary")
            else:
                policy = (document.get("policy") or {}).get("unknown_component", "warn")
                if policy != "ignore":
                    issue(f"Unknown Relay component {kind!r}", "plugin.unknown_component", "error" if policy == "error" else "warning")
            continue
        if not isinstance(config, dict):
            issue("Relay observability config must be an object")
            continue
        if config.get("version", 4) != 4:
            issue("Unsupported Relay observability schema version")
        if type(config.get("enable_full_payloads", False)) is not bool:
            issue("Relay enable_full_payloads must be Boolean")
        for section in ("atof", "atif", "opentelemetry"):
            value = config.get(section)
            if value is None:
                continue
            if not isinstance(value, dict) or type(value.get("enabled", False)) is not bool:
                issue(f"Relay {section}.enabled must be Boolean")
                continue
            if not value.get("enabled"):
                continue
            if section == "atof":
                sinks = value.get("sinks")
                if sinks is not None and not isinstance(sinks, list):
                    issue("Relay ATOF sinks must be an array")
                for sink in sinks or []:
                    if not isinstance(sink, dict) or sink.get("type", "file") not in {"file", "stream"}:
                        issue("Unknown Relay ATOF sink type")
                        continue
                    if sink.get("mode", "append") not in {"append", "overwrite"}:
                        issue("Relay ATOF file mode must be append or overwrite")
                    if sink.get("type", "file") == "stream":
                        try:
                            if sink.get("transport", "http_post") == "http_post":
                                _url(sink.get("url", ""))
                            else:
                                from standalone.relay_atof_stream import validate_stream
                                validate_stream(sink)
                        except ValueError as error:
                            issue(str(error))
                        if sink.get("field_name_policy", "preserve") not in {"preserve", "replace_dots"}:
                            issue("Unsupported ATOF field_name_policy")
            elif section == "atif":
                from standalone.relay_atif_storage import validate_config
                try:
                    validate_config(value)
                except (OSError, ValueError) as error:
                    issue(str(error))
            else:
                if (value.get("metrics") or {}).get("enabled"):
                    from standalone.relay_metrics_otlp import validate_metrics
                    try:
                        validate_metrics(value)
                    except ValueError as error:
                        issue(str(error))
                from standalone.relay_files import validate_destinations
                file_sinks = value.get("file_sinks", [])
                try:
                    validate_destinations(file_sinks)
                except ValueError as error:
                    issue(str(error))
                endpoints = value.get("endpoints", value.get("traces", []))
                if not isinstance(endpoints, list):
                    issue("Relay OTLP endpoints must be an array")
                    continue
                logs = value.get("logs") or {}
                if not endpoints and not file_sinks and not logs.get("enabled") and not (value.get("metrics") or {}).get("enabled"):
                    issue("Enabled Relay OpenTelemetry requires an endpoint or enabled signal")
                for endpoint in endpoints:
                    if not isinstance(endpoint, dict) or endpoint.get("transport", "http_binary") not in {"http_binary", "grpc"}:
                        issue("Standalone Relay OTLP requires HTTP/protobuf or gRPC transport")
                    elif endpoint.get("type", "full") not in {"full", "gen_ai", "openinference"}:
                        issue("Unknown Relay OTLP trace projection")
                if logs.get("enabled"):
                    for endpoint in logs.get("endpoints") or []:
                        if not isinstance(endpoint, dict) or endpoint.get("transport", "http_binary") not in {"http_binary", "grpc"}:
                            issue("Standalone Relay OTLP logs require HTTP/protobuf or gRPC transport")
    plugins = document.get("plugins", {})
    if not isinstance(plugins, dict):
        issue("Relay plugins must be a table")
    else:
        policy = plugins.get("policy", {})
        default_allowed = (policy.get("defaults", {}) if isinstance(policy, dict) else {}).get("allowed", True)
        for declaration in plugins.get("dynamic", []):
            if not isinstance(declaration, dict):
                issue("Malformed Relay dynamic plugin declaration")
                continue
            # A rule may override defaults using manifest identity. Until the
            # native manifest/trust host is ported, refuse such declarations
            # rather than silently suppressing a potentially selected plugin.
            selected = default_allowed is not False or bool(policy.get("rules"))
            dynamic.append({"manifest_ref": declaration.get("manifest", ""), "selected": selected,
                            "status": {"environment": "invalid" if selected else "unknown"}})
            if selected:
                issue("Selected Relay dynamic plugin requires the native ABI or worker host; no plugin code was loaded")
    for path in paths:
        issue("Inherited Relay plugin configuration: " + path, "plugin.configuration_inherited", "warning")
    return {"config_paths": list(paths), "resolved_config": _redact(resolved), "dynamic_plugins": dynamic,
            "config": {"diagnostics": diagnostics, "runtime_diagnostics": []}}


def validate(config, *, additional_plugins_toml=None):
    return _report(*_resolve(config, additional_plugins_toml))


def validate_exact(config):
    return _report(_document(config), [])


def _url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("Relay exporter requires an HTTP(S) URL without embedded credentials")
    return value


class _NoRedirect(http.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def _headers(config):
    sources = [config.get(key) or {} for key in ("headers", "header_env", "header_file")]
    names = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Relay exporter header sources must be objects")
        for name in source:
            if not isinstance(name, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name):
                raise ValueError("Invalid Relay exporter header name")
            if name.lower() in names:
                raise ValueError("Relay exporter header names must be unique across sources")
            names.add(name.lower())
    result = dict(sources[0])
    for name, variable in sources[1].items():
        if not isinstance(variable, str) or not variable.strip() or variable != variable.strip() or "\0" in variable or "=" in variable:
            raise ValueError("Invalid Relay exporter header environment variable")
        if variable not in os.environ:
            raise ValueError("Relay exporter header environment variable is missing")
        value = os.environ[variable]
        if not value.strip() or value != value.strip():
            raise ValueError("Relay exporter environment header must be nonblank and unpadded")
        result[name] = value
    for name, filename in sources[2].items():
        descriptor = os.open(Path(filename).expanduser(), os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Relay exporter header source must be a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                content = source.read(65537)
        finally:
            os.close(descriptor)
        if len(content) > 65536:
            raise ValueError("Relay exporter header file exceeds 64 KiB")
        try:
            result[name] = content.decode().rstrip()
        except UnicodeError:
            raise ValueError("Relay exporter header file is not valid UTF-8") from None
        if not result[name]:
            raise ValueError("Relay exporter header file is blank")
    if any(not isinstance(value, str) or any(ord(char) < 32 and char != "\t" or ord(char) == 127 for char in value) for value in result.values()):
        raise ValueError("Invalid Relay exporter header")
    return result


def _header_destination(url, config, *, signal=None):
    """Match the native exporter's credential transport policy before I/O."""
    endpoint = urlsplit(_url(url))
    protected = signal == 'traces' or bool(config.get('header_file')) or signal in {'logs', 'metrics'} and any(
        config.get(key) for key in ('headers', 'header_env'))
    if not protected or endpoint.scheme == "https":
        return
    host = endpoint.hostname
    local = host.lower() == "localhost"
    if not local:
        try:
            local = "%" not in host and ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
    if not local:
        raise ValueError("Relay OTLP traces and configured signal credentials require HTTPS for remote endpoints")


def _post(url, payload, config, content_type, *, no_redirect=False):
    _header_destination(url, config)
    headers = _headers(config)
    headers["Content-Type"] = content_type
    request = http.Request(_url(url), data=payload, headers=headers, method="POST")
    timeout = config.get("timeout_millis", 3000)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("Relay exporter timeout must be positive")
    # OTLP never redirects. Other legacy transports also refuse redirects
    # whenever configured credentials or metadata are attached.
    opener = http.build_opener(_NoRedirect()) if no_redirect or len(headers) > 1 else http.build_opener()
    try:
        with opener.open(request, timeout=timeout / 1000) as response:
            if not 200 <= response.status < 300:
                raise OSError("Relay exporter returned a non-success response")
            response.read(65536)
    except HTTPError as error:
        code = error.code
        error.close()
        raise OSError("Relay exporter returned HTTP " + str(code)) from None
    except URLError:
        raise OSError("Relay exporter connection failed") from None


def _replace_dots(value):
    if isinstance(value, dict):
        result = {}
        # The pinned exporter orders dotted keys first and retains every
        # collision with _2, _3, ... suffixes instead of dropping a value.
        for key in sorted(value, key=lambda name: ("." not in name, name)):
            base, suffix = key.replace(".", "_"), 1
            target = base
            while target in result:
                suffix += 1
                target = f"{base}_{suffix}"
            result[target] = _replace_dots(value[key])
        return result
    if isinstance(value, list):
        return [_replace_dots(item) for item in value]
    return value


def _private_file(directory, filename, *, append=False):
    if not filename or Path(filename).name != filename or filename in {".", ".."} or "\x00" in filename:
        raise ValueError("Relay exporter filename must remain inside its output directory")
    directory = Path(directory or Path.cwd()).expanduser().resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory / filename, flags, 0o600)
    if os.name != "nt":
        os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "wb")


class _AtofSink:
    def __init__(self, config):
        self.config, self.pending, self.writer, self.stream = config, [], None, None
        if config.get("type", "file") == "file":
            filename = config.get("filename") or "nemo-relay-events-" + datetime.now(timezone.utc).strftime("%Y-%m-%d-%H.%M.%S") + ".jsonl"
            self.writer = _private_file(config.get("output_directory"), filename, append=config.get("mode", "append") == "append")
        elif config.get("transport", "http_post") in ("websocket", "ndjson"):
            from standalone.relay_atof_stream import AtofStreamingSink
            self.stream = AtofStreamingSink(config)

    def __call__(self, event):
        value = event.to_dict()
        if self.config.get("field_name_policy") == "replace_dots":
            value = _replace_dots(value)
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        if self.writer is not None:
            self.writer.write(payload)
            self.writer.flush()
        elif self.stream is not None:
            self.stream.write(payload[:-1])
        else:
            self.pending.append(payload)
            self.flush()

    def flush(self):
        if self.stream is not None:
            self.stream.flush()
        while self.pending:
            _post(self.config["url"], self.pending[0], self.config, "application/x-ndjson")
            self.pending.pop(0)

    def close(self):
        if self.stream is not None:
            self.stream.close()
            return
        self.flush()
        if self.writer is not None:
            self.writer.close()


class _AtifSink:
    def __init__(self, config):
        from standalone.relay_atif_storage import AtifStorage
        self.config, self.owners, self.events, self.roots = config, {}, {}, {}
        self.storage = AtifStorage(config)
        self._lock, self._closed, self._fatal = threading.RLock(), False, None

    def __call__(self, event):
        with self._lock:
            if self._closed or self._fatal is not None:
                return
            try:
                self._observe(event)
            except ValueError:
                self._fatal = 'ATIF trajectory filename or serialization failed'

    def _observe(self, event):
        from standalone.relay_atif_storage import render_filename, DEFAULT_TEMPLATE
        value = event.to_dict()
        root = self.owners.get(value["uuid"]) or self.owners.get(value.get("parent_uuid"))
        if value.get("kind") == "scope" and value.get("category") == "agent" and value.get("scope_category") == "start" and root is None:
            root = value["uuid"]
            render_filename(self.config.get('filename_template', DEFAULT_TEMPLATE), root, value.get('metadata'))
            self.events[root], self.roots[root] = [], value
        if root is None:
            return
        self.owners[value["uuid"]] = root
        self.events[root].append(event)
        if value.get("scope_category") == "end" and value["uuid"] == root:
            self._export(root)

    def _export(self, root):
        from standalone import relay_codecs
        steps, last_user, calls, parents, turns = [], None, {}, {}, set()
        for event in self.events[root]:
            value = event.to_dict()
            if value.get("kind") == "scope" and value.get("scope_category") == "start":
                parents[value["uuid"]] = value
            if value.get("category") == "tool" and value.get("scope_category") == "end":
                call_id = (value.get("category_profile") or {}).get("tool_call_id")
                step = calls.get(call_id)
                if step is not None:
                    step.setdefault("observation", {"results": []})["results"].append({
                        "source_call_id": call_id, "content": value.get("data")})
                continue
            if value.get("category") != "llm":
                continue
            profile = value.get("category_profile") or {}
            if value.get("scope_category") == "start":
                request = profile.get("annotated_request") or {}
                messages = request.get("messages", [])
                user = next((item.get("content") for item in reversed(messages) if item.get("role") == "user"), None)
                ancestor, seen = parents.get(value.get("parent_uuid")), set()
                while ancestor is not None and ancestor["uuid"] not in seen and ancestor.get("name") != "hermes.turn":
                    seen.add(ancestor["uuid"])
                    ancestor = parents.get(ancestor.get("parent_uuid"))
                turn = ancestor["uuid"] if ancestor is not None and ancestor.get("name") == "hermes.turn" else None
                fresh = (turn not in turns) if turn is not None else user != last_user
                if user is not None and fresh:
                    steps.append({"step_id": len(steps) + 1, "source": "user", "message": user, "timestamp": value["timestamp"]})
                    last_user = user
                    if turn is not None:
                        turns.add(turn)
            else:
                annotation = event.annotated_response
                if annotation is None and isinstance(value.get("data"), dict):
                    raw = value["data"]
                    codec = (relay_codecs.OpenAIChatCodec() if "choices" in raw else
                             relay_codecs.AnthropicMessagesCodec() if "content" in raw else relay_codecs.OpenAIResponsesCodec())
                    annotation = codec.decode_response(raw)
                if annotation is None:
                    continue
                step = {"step_id": len(steps) + 1, "source": "agent", "message": annotation.message or "",
                        "timestamp": value["timestamp"], "llm_call_count": 1}
                if annotation.model:
                    step["model_name"] = annotation.model
                if annotation.tool_calls:
                    step["tool_calls"] = [{"tool_call_id": item.get("id"), "function_name": item.get("name"),
                                           "arguments": item.get("arguments")} for item in annotation.tool_calls]
                    calls.update({item["id"]: step for item in annotation.tool_calls if item.get("id")})
                if annotation.usage:
                    step["metrics"] = {key: annotation.usage[key] for key in (
                        "prompt_tokens", "completion_tokens") if key in annotation.usage}
                request_start = parents.get(value["uuid"])
                step["extra"] = {"ancestry": {"function_id": value["uuid"], "function_name": value["name"]},
                                 "llm_response": value.get("data")}
                if request_start is not None:
                    step["extra"]["llm_request"] = (request_start.get("data") or {}).get("content", request_start.get("data"))
                steps.append(step)
        config = self.config
        payload = {"schema_version": "ATIF-v1.7", "session_id": root, "trajectory_id": root,
                   "agent": {"name": config.get("agent_name", "NeMo Relay"), "version": config.get("agent_version", "0.10.0"),
                             "model_name": config.get("model_name", "unknown")}, "steps": steps}
        for key in ("tool_definitions", "extra"):
            if key in config:
                payload["agent"][key] = json_copy(config[key])
        content = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2).encode() + b"\n"
        self.storage.write(root, self.roots[root].get('metadata'), content)
        self.events.pop(root)
        self.roots.pop(root)
        self.owners = {key: value for key, value in self.owners.items() if value != root}

    def runtime_diagnostics(self):
        return self.storage.runtime_diagnostics()

    def close(self):
        with self._lock:
            self._closed = True
            if self._fatal is None:
                for root in tuple(self.events):
                    try:
                        self._export(root)
                    except ValueError:
                        self._fatal = 'ATIF trajectory filename or serialization failed'
                        break
            if self._fatal is not None:
                raise DeliveryFailure(self._fatal)
            self.storage.close()


class DeliveryFailure(RuntimeError):
    """A removed exporter reports its historical failed deliveries."""


class Activation:
    def __init__(self, report, sinks, *, pricing=False):
        self._report, self._sinks, self._closed = report, sinks, False
        self._all_sinks = tuple(sinks)
        self._pricing = pricing
        self._close_error = None
        self._close_lock = threading.Lock()

    @property
    def is_active(self):
        return not self._closed

    @property
    def report(self):
        report = deepcopy(self._report)
        report['config']['runtime_diagnostics'] = [entry for _, sink in self._all_sinks
            if hasattr(sink, 'runtime_diagnostics') for entry in sink.runtime_diagnostics()]
        return report

    async def close(self):
        from standalone import relay
        await relay.subscribers.flush_async()
        # Native shutdown performs blocking exporter work outside the Python
        # event loop. An HTTP/2 peer may itself need this caller's loop.
        await asyncio.to_thread(self._close)

    def _close(self):
        global _ACTIVE
        from standalone import relay
        with self._close_lock:
            if self._closed:
                if self._close_error is not None:
                    raise self._close_error
                return
            errors, retained = [], []
            for name, sink in reversed(self._sinks):
                try:
                    sink.close()
                except DeliveryFailure as error:
                    errors.append(error)
                except Exception as error:
                    errors.append(error)
                    retained.append((name, sink))
                    continue
                relay.subscribers.deregister(name)
            self._sinks = list(reversed(retained))
            if not retained:
                with _LOCK:
                    if self._pricing:
                        from standalone.relay_pricing import set_active
                        set_active(None)
                    self._closed = True
                    relay._FULL_PAYLOADS = False
                    if _ACTIVE is self:
                        _ACTIVE = None
            if errors:
                error = RuntimeError('registration failed: plugin teardown failed: ' + '; '.join(map(str, errors)))
                if self._closed:
                    self._close_error = error
                raise error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exception):
        await self.close()


async def initialize(config, *, additional_plugins_toml=None):
    global _ACTIVE
    from standalone import relay
    document, paths = _resolve(config, additional_plugins_toml)
    report = _report(document, paths)
    if errors := [item["message"] for item in report["config"]["diagnostics"] if item["level"] == "error"]:
        raise ValueError("; ".join(errors))
    with _LOCK:
        if _ACTIVE is not None:
            raise RuntimeError("conflict: plugin configuration is owned by an active dynamic plugin host")
        sinks, pricing = [], None
        try:
            for component in document.get("components", []):
                if not component.get("enabled", True):
                    continue
                settings = component.get("config", {})
                if component.get("kind") == "pricing":
                    from standalone.relay_pricing import PricingResolver
                    pricing = PricingResolver(settings)
                    continue
                if component.get("kind") != "observability":
                    continue
                relay._FULL_PAYLOADS = bool(settings.get("enable_full_payloads", False))
                if (settings.get("atof") or {}).get("enabled"):
                    for config in settings["atof"].get("sinks") or [{"type": "file"}]:
                        sinks.append(("standalone.relay.atof." + uuid.uuid4().hex, _AtofSink(config)))
                if (settings.get("atif") or {}).get("enabled"):
                    sinks.append(("standalone.relay.atif." + uuid.uuid4().hex, _AtifSink(settings["atif"])))
                if (settings.get("opentelemetry") or {}).get("enabled"):
                    from standalone.relay_otlp import OtlpSink
                    sinks.append(("standalone.relay.otlp." + uuid.uuid4().hex, OtlpSink(settings["opentelemetry"])))
            for name, sink in sinks:
                relay.subscribers.register(name, sink)
            if pricing is not None:
                from standalone.relay_pricing import set_active
                set_active(pricing)
            _ACTIVE = Activation(report, sinks, pricing=pricing is not None)
            return _ACTIVE
        except BaseException:
            for name, sink in reversed(sinks):
                relay.subscribers.deregister(name)
                sink.close()
            relay._FULL_PAYLOADS = False
            raise
