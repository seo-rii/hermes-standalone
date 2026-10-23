"""Qualified standard-library Relay execution and event runtime for Hermes.

Owns real context stacks, middleware, asynchronous streams and subscriber
delivery. The original application keeps its session/profile/consent policy.
Native plugin engines are validated separately by relay_plugin and never
silently treated as active. Pinned source contracts: relay-provenance.json.
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import contextmanager
import contextvars
from copy import deepcopy
from dataclasses import dataclass
import inspect
import logging
import re
import threading
import time
import sys
from types import SimpleNamespace
import uuid as _uuid

from standalone.relay_types import (
    AnnotatedLLMRequest, AnnotatedLLMResponse, Event, LLMAttributes, LLMHandle,
    LLMRequest, LLMRequestInterceptOutcome, MarkEvent, PendingMarkSpec,
    ScopeAttributes, ScopeEvent, ScopeHandle, ScopeType, ToolAttributes,
    ToolExecutionInterceptOutcome, ToolExecutionResult, ToolHandle,
    json_copy, timestamp,
)
from standalone.relay_metrics import MetricKind, MetricValueType, MetricMeasurement
from standalone import relay_codecs as codecs
from standalone import relay_observability as observability


logger = logging.getLogger(__name__)
_STACK = contextvars.ContextVar("hermes_standalone_relay_stack", default=None)
_CALLBACK = contextvars.ContextVar("hermes_standalone_relay_publication", default=False)
_HANDLES, _FRESH_AGENTS, _STATE_LOCK = {}, set(), threading.RLock()
_FULL_PAYLOADS = False


def _id():
    return str(_uuid.uuid7() if hasattr(_uuid, "uuid7") else _uuid.uuid4())


class ScopeStack(list):
    def __init__(self, values=()):
        super().__init__(values)
        self.lock = threading.RLock()


def create_scope_stack():
    return ScopeStack()


def get_scope_stack():
    stack = _STACK.get()
    if stack is None:
        stack = ScopeStack()
        _STACK.set(stack)
    return stack


def scope_stack_active():
    return _STACK.get() is not None


def set_thread_scope_stack(stack):
    if not isinstance(stack, ScopeStack):
        raise TypeError("Relay requires an owned ScopeStack")
    _STACK.set(stack)


def propagate_scope_to_thread():
    if not scope_stack_active():
        raise RuntimeError("No active Relay scope stack")
    return get_scope_stack()


def fork_asyncio_context():
    context = contextvars.copy_context()
    context.run(_STACK.set, ScopeStack(get_scope_stack()))
    return context


async def _await(value):
    return await value if inspect.isawaitable(value) else value


def _resolve(value, *, timeout=10):
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    result, context = {}, contextvars.copy_context()

    def invoke():
        try:
            result["value"] = context.run(asyncio.run, value)
        except BaseException as error:
            result["error"] = error

    worker = threading.Thread(target=invoke, daemon=True, name="hermes-relay-callback")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError("Relay asynchronous callback exceeded 10 seconds")
    if "error" in result:
        raise result["error"]
    return result["value"]


class _Subscribers:
    def __init__(self):
        self._condition = threading.Condition()
        self._callbacks, self._queue = {}, deque()
        self._submitted = self._completed = 0
        self._thread = None

    def register(self, name, callback):
        if not isinstance(name, str) or not name or not callable(callback):
            raise ValueError("Relay subscriber needs a name and callable")
        with self._condition:
            self._callbacks[name] = callback

    def deregister(self, name):
        with self._condition:
            return self._callbacks.pop(name, None) is not None

    def snapshot(self):
        with self._condition:
            return tuple(self._callbacks.values())

    def publish(self, event, *, callbacks=None, force=False):
        with self._condition:
            callbacks = tuple(self._callbacks.values()) if callbacks is None else callbacks
            if not callbacks and not force:
                return
            self._submitted += 1
            self._queue.append((self._submitted, event, callbacks))
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True, name="hermes-relay-subscribers")
                self._thread.start()
            self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue)
                sequence, event, callbacks = self._queue.popleft()
            token = _CALLBACK.set(True)
            try:
                from standalone.relay_stream import DeferredEvent
                if isinstance(event, DeferredEvent):
                    try:
                        event.dispatch(callbacks)
                    except Exception:
                        logger.warning("Relay deferred publication failed", exc_info=True)
                    continue
                for callback in callbacks:
                    try:
                        _resolve(callback(event))
                    except Exception:
                        logger.warning("Relay subscriber failed", exc_info=True)
            finally:
                _CALLBACK.reset(token)
                with self._condition:
                    self._completed = sequence
                    self._condition.notify_all()

    def flush(self):
        # A callback cannot wait for its own publication barrier.
        if _CALLBACK.get() or threading.current_thread() is self._thread:
            return
        with self._condition:
            target = self._submitted
            self._condition.wait_for(lambda: self._completed >= target)

    async def flush_async(self):
        await asyncio.to_thread(self.flush)


subscribers = _Subscribers()


class _Middleware:
    def __init__(self, kinds):
        self._kinds, self._entries, self._lock, self._order = set(kinds), {}, threading.RLock(), 0

    def __getattr__(self, name):
        for prefix, method in (("register_", self._register), ("deregister_", self._deregister)):
            if name.startswith(prefix) and name[len(prefix):] in self._kinds:
                return lambda *args, _kind=name[len(prefix):], _method=method: _method(_kind, *args)
        raise AttributeError(name)

    def _register(self, kind, name, priority, *arguments):
        callback = arguments[-1] if arguments else None
        break_chain = arguments[0] if len(arguments) == 2 else False
        if not isinstance(name, str) or not name or type(priority) is not int or not callable(callback):
            raise ValueError("Invalid Relay middleware registration")
        if len(arguments) not in (1, 2) or type(break_chain) is not bool:
            raise TypeError("Invalid Relay middleware arguments")
        with self._lock:
            self._order += 1
            self._entries.setdefault(kind, {})[name] = (priority, self._order, break_chain, callback)

    def _deregister(self, kind, name):
        with self._lock:
            return self._entries.get(kind, {}).pop(name, None) is not None

    def snapshot(self, kind):
        with self._lock:
            return tuple(sorted(self._entries.get(kind, {}).values(), key=lambda item: item[:2]))


intercepts = _Middleware(("tool_request", "tool_execution", "llm_request", "llm_execution", "llm_stream_execution"))
guardrails = _Middleware(("tool_sanitize_request", "tool_sanitize_response", "tool_conditional_execution",
                         "llm_sanitize_request", "llm_sanitize_response", "llm_conditional_execution",
                         "mark_sanitize", "scope_sanitize_start", "scope_sanitize_end"))
event_metadata = _Middleware(("injector",))


def _merge(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return {**deepcopy(left), **deepcopy(right)}
    return deepcopy(right if right is not None else left)


def _emit(record, *, annotated_request=None, annotated_response=None):
    record = json_copy(record)
    event = Event(record, annotated_request=annotated_request, annotated_response=annotated_response)
    kind = "mark_sanitize" if record["kind"] == "mark" else "scope_sanitize_" + record["scope_category"]
    fields = {name: deepcopy(record.get(name)) for name in ("data", "category_profile", "metadata")}
    token = _CALLBACK.set(True)
    try:
        for _, _, _, callback in guardrails.snapshot(kind):
            try:
                result = _resolve(callback(event, deepcopy(fields)))
                if not isinstance(result, dict) or set(result) != set(fields):
                    raise ValueError("Invalid Relay sanitizer result")
                fields = json_copy(result)
            except Exception:
                fields = dict.fromkeys(fields)
                break  # Fail closed: no remaining sanitizer sees the raw payload.
        record.update(fields)
        for _, _, _, callback in event_metadata.snapshot("injector"):
            try:
                additions = json_copy(_resolve(callback(Event(record))))
                if not isinstance(additions, dict) or any(not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", key) for key in additions):
                    raise ValueError("Invalid Relay metadata injector keys")
                for value in additions.values():
                    if isinstance(value, list):
                        if any(type(item) not in (str, bool, int, float) for item in value) or len({type(item) for item in value}) > 1:
                            raise ValueError("Relay metadata arrays must be homogeneous scalar values")
                    elif type(value) not in (str, bool, int, float):
                        raise ValueError("Relay metadata must contain scalar values")
                record["metadata"] = {**additions, **(record.get("metadata") or {})}
            except Exception:
                logger.warning("Relay metadata injector failed", exc_info=True)
    finally:
        _CALLBACK.reset(token)
    subscribers.publish(Event(record, annotated_request=annotated_request, annotated_response=annotated_response))


def _record(handle, phase, data, metadata=None, at=None):
    return {"kind": "scope", "atof_version": "0.1", "uuid": handle.uuid,
            "parent_uuid": handle.parent_uuid, "timestamp": timestamp(at) if at is not None else (handle.started_at if phase == "start" else timestamp()),
            "name": handle.name, "scope_category": phase, "attributes": [], "category": str(handle.scope_type),
            "category_profile": deepcopy(handle.profile), "data": json_copy(data), "data_schema": None,
            "metadata": _merge(handle.metadata, metadata)}


def _new_handle(name, kind, *, handle=None, attributes=0, data=None, metadata=None, timestamp_value=None, profile=None):
    stack = get_scope_stack()
    with stack.lock:
        parent = handle if handle is not None else (stack[-1] if stack else None)
    if parent is not None and not isinstance(parent, ScopeHandle):
        raise TypeError("Relay parent must be an owned scope handle")
    result = ScopeHandle(_id(), str(name), ScopeType(kind), None if parent is None else parent.uuid,
                         json_copy(data), json_copy(metadata), attributes or 0, timestamp(timestamp_value), profile)
    with _STATE_LOCK:
        _HANDLES[result.uuid] = result
        if result.scope_type is ScopeType.Agent:
            _FRESH_AGENTS.add(result.uuid)
    return result


class _Scope:
    def metric(self, name, measurements, *, handle=None, metadata=None, timestamp=None):
        from standalone.relay_metrics import METRIC_SCHEMA, metric_payload
        if not isinstance(name, str):
            raise TypeError("'" + type(name).__name__ + "' object is not an instance of 'str'")
        prepared_metadata = json_copy(metadata)
        if timestamp is not None:
            globals()["timestamp"](timestamp)
        payload = metric_payload(measurements)
        self.event(name, handle=handle, data=payload, data_schema=METRIC_SCHEMA,
                   metadata=prepared_metadata, timestamp=timestamp)

    def get_handle(self):
        stack = get_scope_stack()
        with stack.lock:
            if not stack:
                raise RuntimeError("Relay scope stack is empty")
            return stack[-1]

    def push(self, name, scope_type, *, handle=None, attributes=None, data=None, metadata=None, input=None, timestamp=None):
        stack = get_scope_stack()
        with stack.lock:
            result = _new_handle(name, scope_type, handle=handle, attributes=attributes, data=data,
                                 metadata=metadata, timestamp_value=timestamp)
            stack.append(result)
            _emit(_record(result, "start", input))
        return result

    def pop(self, handle, *, output=None, metadata=None, timestamp=None):
        stack = get_scope_stack()
        with stack.lock:
            if not stack or stack[-1].uuid != handle.uuid:
                raise RuntimeError("scope handle is not at the top of the stack")
            stack.pop()
            _emit(_record(handle, "end", output, metadata, timestamp))
        with _STATE_LOCK:
            _HANDLES.pop(handle.uuid, None)
            _FRESH_AGENTS.discard(handle.uuid)

    def event(self, name, *, handle=None, data=None, data_schema=None, metadata=None, severity=None, timestamp=None):
        stack = get_scope_stack()
        with stack.lock:
            parent = handle if handle is not None else (stack[-1] if stack else None)
        if name == "compaction":
            with _STATE_LOCK:
                current, seen = parent, set()
                while current is not None and current.uuid not in seen:
                    seen.add(current.uuid)
                    if current.scope_type is ScopeType.Agent:
                        _FRESH_AGENTS.add(current.uuid)
                        break
                    current = _HANDLES.get(current.parent_uuid)
        record = {"kind": "mark", "atof_version": "0.1", "uuid": _id(),
                  "parent_uuid": None if parent is None else parent.uuid,
                  "timestamp": globals()["timestamp"](timestamp), "name": str(name),
                  "data": json_copy(data), "data_schema": json_copy(data_schema), "metadata": json_copy(metadata),
                  "category": None, "category_profile": None}
        if severity is not None:
            record["metadata"] = {**(record["metadata"] or {}), "nemo.relay.log.severity": str(severity).lower()}
        _emit(record)

    @contextmanager
    def scope(self, name, scope_type, **kwargs):
        handle = self.push(name, scope_type, **kwargs)
        try:
            yield handle
        finally:
            self.pop(handle)


scope = _Scope()


async def _conditional(kind, *arguments):
    for _, _, _, callback in guardrails.snapshot(kind):
        rejection = await _await(callback(*arguments))
        if rejection is not None:
            raise RuntimeError("Relay conditional execution rejected: " + str(rejection))


def _sanitize(kind, value, context):
    for _, _, _, callback in guardrails.snapshot(kind):
        try:
            value = _resolve(callback(context, deepcopy(value)) if kind.startswith("tool_")
                             else callback(deepcopy(value), context))
        except Exception:
            return None
        if value is None:
            return None
    return value


async def _branch(callback, *args):
    context = fork_asyncio_context()
    return await context.run(asyncio.create_task, _await(callback(*args)))


class _Tools:
    def request_intercepts(self, name, args):
        value = json_copy(args)
        for _, _, stop, callback in intercepts.snapshot("tool_request"):
            value = json_copy(_resolve(callback(name, value)))
            if stop:
                break
        return value

    def conditional_execution(self, name, args):
        return _conditional("tool_conditional_execution", name, args)

    def call(self, name, args, *, handle=None, attributes=None, data=None, metadata=None, tool_call_id=None, timestamp=None):
        profile = {} if tool_call_id is None else {"tool_call_id": tool_call_id}
        result = _new_handle(name, ScopeType.Tool, handle=handle, attributes=attributes, data=data, metadata=metadata,
                             timestamp_value=timestamp, profile=profile)
        _emit(_record(result, "start", _sanitize("tool_sanitize_request", args, name)))
        return result

    def call_end(self, handle, result, *, data=None, metadata=None, timestamp=None):
        if not isinstance(result, ToolExecutionResult):
            raise TypeError("Relay tool result requires ToolExecutionResult")
        record = _record(handle, "end", _sanitize("tool_sanitize_response", result.result, handle.name), metadata, timestamp)
        if result.annotation is not None:
            record["category_profile"] = {**(record["category_profile"] or {}), "tool_result_annotation": json_copy(result.annotation)}
        _emit(record)
        with _STATE_LOCK:
            _HANDLES.pop(handle.uuid, None)

    async def execute(self, name, args, callback, **kwargs):
        await _conditional("tool_conditional_execution", name, args)
        args = self.request_intercepts(name, args)
        entries = intercepts.snapshot("tool_execution")
        handle = self.call(name, args, **kwargs)
        pending_marks = []

        async def invoke(index, current):
            if index == len(entries):
                result = await _await(callback(current))
                if not isinstance(result, ToolExecutionResult):
                    raise TypeError("Relay tool callback requires ToolExecutionResult")
                return result
            active, pending = True, set()

            async def next_call(value):
                if not active:
                    raise RuntimeError("Relay continuation is no longer active")
                task = asyncio.create_task(_branch(invoke, index + 1, json_copy(value)))
                pending.add(task)
                try:
                    return await task
                finally:
                    pending.discard(task)

            context = SimpleNamespace(tool_name=name, args=json_copy(current), tool_call_id=kwargs.get("tool_call_id"))
            try:
                result = await _await(entries[index][3](context, next_call))
                if not isinstance(result, ToolExecutionInterceptOutcome):
                    raise TypeError("Relay tool execution intercept requires an outcome")
                pending_marks.extend(result.pending_marks)
                return ToolExecutionResult(result.result, result.annotation)
            finally:
                active = False
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

        try:
            result = await invoke(0, args)
        except BaseException as error:
            self.call_end(handle, ToolExecutionResult(None), metadata={"otel.status_code": "ERROR",
                          "otel.status_description": type(error).__name__})
            raise
        self.call_end(handle, result, metadata={"otel.status_code": "OK"})
        for mark in pending_marks:
            scope.event(mark.name, handle=handle, data=mark.data, metadata=mark.metadata, data_schema=mark.data_schema)
        return result


tools = _Tools()


def _codec_context(codec):
    identity = SimpleNamespace(kind="builtin" if codec else "none", id=type(codec).__name__ if codec else None)
    return SimpleNamespace(codec=identity, resolve_codec=lambda: codec)


def _observable_request(request, codec, parent):
    credentials = {"authorization", "proxy-authorization", "cookie", "x-api-key", "api-key", "anthropic-api-key", "x-goog-api-key"}
    headers = {key: value for key, value in request.headers.items() if key.casefold() not in credentials}
    body = request.content
    # Hermes carries SDK transport headers in extra_headers. They remain on
    # the provider request but must not become exported conversation data.
    if isinstance(body.get("extra_headers"), dict):
        body["extra_headers"] = {key: value for key, value in body["extra_headers"].items() if key.casefold() not in credentials}
    observable = _sanitize("llm_sanitize_request", LLMRequest(headers, body), _codec_context(codec))
    if observable is None:
        return None, None
    annotation = codec.decode(observable) if codec else None
    with _STATE_LOCK:
        current, seen, fresh = parent, set(), False
        while current is not None and current.uuid not in seen:
            seen.add(current.uuid)
            if current.scope_type is ScopeType.Agent:
                fresh = current.uuid in _FRESH_AGENTS
                _FRESH_AGENTS.discard(current.uuid)
                break
            current = _HANDLES.get(current.parent_uuid)
    if not _FULL_PAYLOADS and not fresh and annotation is not None:
        last_user = next((index for index in range(len(annotation.messages) - 1, -1, -1)
                          if annotation.messages[index].get("role") == "user"), None)
        if last_user is not None and last_user > 0:
            annotation.messages = annotation.messages[last_user:]
            observable = codec.encode(annotation, observable)
    return {"headers": observable.headers, "content": observable.content}, annotation


def _trace_request(request, handle):
    root, current, seen = handle, handle, set()
    with _STATE_LOCK:
        while current is not None and current.uuid not in seen:
            seen.add(current.uuid)
            root = current
            current = _HANDLES.get(current.parent_uuid)
    trace = _uuid.UUID(root.uuid).hex
    span = _uuid.UUID(handle.uuid).hex[-16:]
    headers = {key: value for key, value in request.headers.items() if key.casefold() not in {"traceparent", "tracestate"}}
    headers["traceparent"] = f"00-{trace}-{span}-01"
    return LLMRequest(headers, request.content)


class _Llm:
    def call(self, name, request, *, handle=None, attributes=None, data=None, metadata=None, model_name=None, timestamp=None):
        result = _new_handle(name, ScopeType.Llm, handle=handle, attributes=attributes, data=data, metadata=metadata,
                             timestamp_value=timestamp, profile={"model_name": model_name or "unknown"})
        observable, annotation = _observable_request(request, None, handle)
        _emit(_record(result, "start", observable), annotated_request=annotation)
        return result

    def call_end(self, handle, response, *, data=None, metadata=None, annotated_response=None, response_codec=None, timestamp=None):
        sanitized = _sanitize("llm_sanitize_response", response, _codec_context(response_codec))
        annotation = annotated_response
        if sanitized is None:
            annotation = None
        elif annotation is None and response_codec is not None:
            annotation = response_codec.decode_response(sanitized)
        record = _record(handle, "end", sanitized if sanitized is not None else data, metadata, timestamp)
        if annotation is not None:
            record["category_profile"] = {**(record["category_profile"] or {}), "annotated_response": annotation.to_dict()
                                         if hasattr(annotation, "to_dict") else json_copy(annotation)}
        _emit(record, annotated_response=annotation)
        with _STATE_LOCK:
            _HANDLES.pop(handle.uuid, None)

    async def _prepare(self, name, request, codec):
        await _conditional("llm_conditional_execution", request)
        annotation = codec.decode(request) if codec else None
        for _, _, stop, callback in intercepts.snapshot("llm_request"):
            baseline_request = request
            baseline_annotation = annotation.to_dict() if annotation is not None else None
            result = await _await(callback(name, request, annotation))
            if not isinstance(result, LLMRequestInterceptOutcome):
                raise TypeError("Relay request intercept requires LLMRequestInterceptOutcome")
            request, annotation = result.request, result.annotated_request
            if codec is not None and annotation is not None and annotation.to_dict() != baseline_annotation:
                encoded, before, rewritten = codec.encode(annotation, baseline_request).content, baseline_request.content, request.content
                for key in before.keys() | encoded.keys():
                    if key not in encoded:
                        rewritten.pop(key, None)
                    elif before.get(key) != encoded[key]:
                        rewritten[key] = encoded[key]
                request = LLMRequest(request.headers, rewritten)
            for mark in result.pending_marks:
                scope.event(mark.name, data=mark.data, metadata=mark.metadata, data_schema=mark.data_schema)
            if stop:
                break
        return request

    def _start(self, name, request, kwargs):
        codec = kwargs.get("codec")
        selected = {key: value for key, value in kwargs.items() if key not in ("codec", "response_codec")}
        result = _new_handle(name, ScopeType.Llm, handle=selected.get("handle"), attributes=selected.get("attributes"),
                             data=selected.get("data"), metadata=selected.get("metadata"),
                             profile={"model_name": selected.get("model_name") or "unknown"})
        observable, annotation = _observable_request(request, codec, selected.get("handle"))
        record = _record(result, "start", observable)
        if annotation is not None:
            record["category_profile"]["annotated_request"] = annotation.to_dict()
        _emit(record, annotated_request=annotation)
        return result

    async def execute(self, name, request, callback, **kwargs):
        codec = kwargs.get("codec")
        request = await self._prepare(name, request, codec)
        entries = intercepts.snapshot("llm_execution")
        handle = self._start(name, request, kwargs)
        request = _trace_request(request, handle)
        context = SimpleNamespace(request_codec=_codec_context(codec), response_codec=(
            _codec_context(kwargs["response_codec"]) if kwargs.get("response_codec") is not None else None))

        async def invoke(index, current):
            if index == len(entries):
                return json_copy(await _await(callback(current)))
            active, pending = True, set()

            async def next_call(value):
                if not active:
                    raise RuntimeError("Relay continuation is no longer active")
                task = asyncio.create_task(_branch(invoke, index + 1, value))
                pending.add(task)
                try:
                    return await task
                finally:
                    pending.discard(task)

            try:
                return await _await(entries[index][3](name, current, context, next_call))
            finally:
                active = False
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

        try:
            response = await invoke(0, request)
        except BaseException as error:
            self.call_end(handle, None, metadata={"otel.status_code": "ERROR", "otel.status_description": type(error).__name__})
            raise
        self.call_end(handle, response, response_codec=kwargs.get("response_codec"), metadata={"otel.status_code": "OK"})
        return response

    async def stream_execute(self, name, request, callback, collector, finalizer, **kwargs):
        request = await self._prepare(name, request, kwargs.get("codec"))
        lifecycle_subscribers = subscribers.snapshot()
        handle = self._start(name, request, kwargs)
        request = _trace_request(request, handle)
        stream_started_at = time.monotonic_ns()
        entries = intercepts.snapshot("llm_stream_execution")
        context = SimpleNamespace(request_codec=_codec_context(kwargs.get("codec")),
                                  response_codec=(_codec_context(kwargs["response_codec"])
                                                  if kwargs.get("response_codec") is not None else None))

        async def invoke(index, current):
            if index == len(entries):
                result = await _await(callback(current))
            else:
                result = await _await(entries[index][3](name, current, context, lambda value: invoke(index + 1, value)))
            if not hasattr(result, "__aiter__"):
                raise TypeError("Relay streaming callback requires an asynchronous iterator")
            return result.__aiter__()

        try:
            source = await invoke(0, request)
        except BaseException as error:
            from standalone.relay_stream import failed_start
            failed_start(sys.modules[__name__], handle, error, lifecycle_subscribers)
            raise
        from standalone.relay_stream import LlmStream
        return LlmStream(sys.modules[__name__], source, handle, collector, finalizer,
                         kwargs.get("response_codec"), stream_started_at, lifecycle_subscribers)


llm = _Llm()

# Imported last so plugin activation can own this runtime's subscribers without
# introducing a native or global SDK namespace.
from standalone import relay_plugin as plugin
from standalone import relay_model_pricing as model_pricing
