"""JSON-owned types for the qualified Hermes Relay runtime.

The consumed contracts are derived from nemo-relay 0.10.0; see
relay-provenance.json. No native binding or global package is impersonated.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntFlag, StrEnum
import json
from typing import Any, Literal

Json = Any
JsonObject = dict[str, Any]
UnsupportedBehavior = Literal["ignore", "warn", "error"]


def json_copy(value):
    # Refuse values the Rust JSON boundary would refuse; retain JSON null.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def timestamp(value=None):
    value = datetime.now(timezone.utc) if value is None else value
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Relay timestamps require timezone-aware datetime values")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class ScopeType(StrEnum):
    Agent = "agent"
    Function = "function"
    Tool = "tool"
    Llm = "llm"
    Retriever = "retriever"
    Embedder = "embedder"
    Reranker = "reranker"
    Guardrail = "guardrail"
    Evaluator = "evaluator"
    Custom = "custom"
    Unknown = "unknown"


class ScopeAttributes(IntFlag):
    PARALLEL = 1
    RELOCATABLE = 2


class ToolAttributes(IntFlag):
    STATEFUL = 1
    SIDE_EFFECTS = 2


class LLMAttributes(IntFlag):
    STATEFUL = 1
    STREAMING = 2


@dataclass(frozen=True)
class ScopeHandle:
    uuid: str
    name: str
    scope_type: ScopeType
    parent_uuid: str | None
    data: Any = None
    metadata: Any = None
    attributes: Any = 0
    started_at: str = ""
    profile: Any = None


ToolHandle = ScopeHandle
LLMHandle = ScopeHandle


class LLMRequest:
    def __init__(self, headers, content):
        if not isinstance(headers, dict) or not isinstance(content, dict):
            raise TypeError("Relay LLM headers and content must be JSON objects")
        self._headers, self._content = json_copy(headers), json_copy(content)

    @property
    def headers(self):
        return deepcopy(self._headers)

    @property
    def content(self):
        return deepcopy(self._content)


@dataclass(frozen=True)
class ToolExecutionResult:
    result: Any
    annotation: Any = None


@dataclass(frozen=True, init=False)
class ToolExecutionInterceptOutcome:
    result: Any
    annotation: Any = None
    pending_marks: list = field(default_factory=list)

    def __init__(self, result, pending_marks=None, *, annotation=None):
        object.__setattr__(self, "result", json_copy(result))
        object.__setattr__(self, "annotation", json_copy(annotation))
        object.__setattr__(self, "pending_marks", list(pending_marks or []))


@dataclass(frozen=True)
class LLMRequestInterceptOutcome:
    request: LLMRequest
    annotated_request: Any = None
    pending_marks: list = field(default_factory=list)
    optimization_contributions: list = field(default_factory=list)


@dataclass(frozen=True)
class PendingMarkSpec:
    name: str
    category: Any = None
    category_profile: Any = None
    data: Any = None
    metadata: Any = None
    data_schema: Any = None
    severity: Any = None


class AnnotatedLLMRequest:
    def __init__(self, messages, *, instructions=None, model=None, params=None, tools=None,
                 tool_choice=None, api_specific=None, extra=None):
        for key, value in locals().copy().items():
            if key != "self":
                setattr(self, key, json_copy(value))
        self.extra = self.extra or {}

    def system_prompt(self):
        if isinstance(self.instructions, str):
            return self.instructions
        return next((item.get("content") for item in self.messages if item.get("role") == "system"), None)

    def last_user_message(self):
        return next((item.get("content") for item in reversed(self.messages) if item.get("role") == "user"), None)

    def has_tool_calls(self):
        return bool(self.tools)

    def to_dict(self):
        return json_copy({**self.extra, **{name: getattr(self, name) for name in (
            "messages", "instructions", "model", "params", "tools", "tool_choice", "api_specific")
            if getattr(self, name) is not None}})


class AnnotatedLLMResponse:
    def __init__(self, id=None, model=None, message=None, tool_calls=None, finish_reason=None,
                 usage=None, api_specific=None, extra=None):
        for key, value in locals().copy().items():
            if key != "self":
                setattr(self, key, json_copy(value))
        self.extra = self.extra or {}
        self.optimization_summary = None

    def response_text(self):
        if isinstance(self.message, str):
            return self.message
        if isinstance(self.message, list):
            return "".join(item.get("text", "") for item in self.message if isinstance(item, dict))
        return None

    def has_tool_calls(self):
        return bool(self.tool_calls)

    def to_dict(self):
        fields = {name: getattr(self, name) for name in (
            "id", "model", "message", "tool_calls", "finish_reason", "usage", "api_specific")
            if getattr(self, name) is not None}
        if self.finish_reason is not None:
            if getattr(self, '_finish_source', object()) == self.finish_reason:
                fields['finish_reason'] = self._finish_json
            elif self.finish_reason not in ('complete', 'length', 'tool_use', 'content_filter'):
                fields['finish_reason'] = {'unknown':self.finish_reason}
        return json_copy({**fields, **self.extra})


class Event:
    def __init__(self, record, *, annotated_request=None, annotated_response=None):
        object.__setattr__(self, "_record", json_copy(record))
        object.__setattr__(self, "annotated_request", annotated_request)
        object.__setattr__(self, "annotated_response", annotated_response)

    def __getattr__(self, name):
        if name not in self._record:
            raise AttributeError(name)
        return deepcopy(self._record[name])

    def __setattr__(self, name, value):
        raise AttributeError("Relay events are immutable")

    def to_dict(self):
        return deepcopy(self._record)

    def to_json(self):
        return json.dumps(self._record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


ScopeEvent = Event
MarkEvent = Event
