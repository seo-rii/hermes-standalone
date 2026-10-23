"""Small wire objects, provider errors and SSE decoding for bundled transports.

These objects implement the Hermes client boundary. They are not replacements
for Pydantic models or a process-wide emulation of any provider SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


_OPTIONAL_FIELDS = frozenset({
    "content", "refusal", "tool_calls", "function_call", "finish_reason", "logprobs",
    "usage", "system_fingerprint", "service_tier", "audio", "reasoning",
    "reasoning_content", "reasoning_details", "prompt_tokens_details",
    "completion_tokens_details", "cached_tokens", "reasoning_tokens", "provider",
})


class WireObject(dict):
    """JSON mapping with the attribute and dump access used by Hermes adapters.

    Keeping mappings as dict subclasses also preserves provider-specific blocks
    whose consumers intentionally use ``isinstance(block, dict)``.
    """

    def __init__(self, value=None, **kwargs):
        super().__init__({key: wire_object(item) for key, item in dict(value or {}, **kwargs).items()})

    def __getattr__(self, name):
        if name in self:
            return self[name]
        if name in _OPTIONAL_FIELDS:
            return None
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = wire_object(value)

    @property
    def model_extra(self):
        # This boundary is intentionally schema-free: extensions retain their
        # exact JSON values, including when a later provider adds a new field.
        return dict(self)

    @property
    def model_fields_set(self):
        return set(self)

    def model_dump(self, *, exclude_none=False, include=None, exclude=None, **_kwargs):
        value = json_value(self, exclude_none=exclude_none)
        if include is not None:
            value = {key: item for key, item in value.items() if key in include}
        if exclude is not None:
            value = {key: item for key, item in value.items() if key not in exclude}
        return value

    def model_dump_json(self, **kwargs):
        indent = kwargs.pop("indent", None)
        return json.dumps(self.model_dump(**kwargs), ensure_ascii=False, indent=indent)

    def to_dict(self, **kwargs):
        return self.model_dump(**kwargs)

    def model_copy(self, *, update=None, deep=False):
        import copy
        value = copy.deepcopy(self) if deep else WireObject(self)
        if update:
            value.update({key: wire_object(item) for key, item in update.items()})
        return value


def wire_object(value):
    if isinstance(value, dict):
        return WireObject(value)
    if isinstance(value, (list, tuple)):
        return [wire_object(item) for item in value]
    return value


def json_value(value, *, exclude_none=False):
    """Serialize only JSON data and explicit model dumps, never object reprs."""
    if isinstance(value, dict):
        return {key: json_value(item, exclude_none=exclude_none) for key, item in value.items()
                if not (exclude_none and item is None)}
    if isinstance(value, (list, tuple)):
        return [json_value(item, exclude_none=exclude_none) for item in value]
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump(mode="json"), exclude_none=exclude_none)
    return value


class APIError(Exception):
    def __init__(self, message, *, request=None, body=None):
        super().__init__(message)
        self.message, self.request, self.body = message, request, body
        details = body.get("error", body) if isinstance(body, dict) else {}
        details = details if isinstance(details, dict) else {}
        self.code = details.get("code")
        self.type = details.get("type")
        self.param = details.get("param")


class APIStatusError(APIError):
    def __init__(self, message, *, response, body=None, status_code=None):
        super().__init__(message, request=response.request, body=body)
        self.response = response
        self.status_code = response.status_code if status_code is None else status_code
        self.request_id = response.headers.get("x-request-id") or response.headers.get("request-id")


class BadRequestError(APIStatusError):
    pass


class AuthenticationError(APIStatusError):
    pass


class PermissionDeniedError(APIStatusError):
    pass


class NotFoundError(APIStatusError):
    pass


class ConflictError(APIStatusError):
    pass


class UnprocessableEntityError(APIStatusError):
    pass


class RateLimitError(APIStatusError):
    pass


class InternalServerError(APIStatusError):
    pass


class APIConnectionError(APIError):
    pass


class APITimeoutError(APIConnectionError):
    pass


class APIResponseValidationError(APIStatusError):
    pass


def status_error(response, body=None, *, status_code=None):
    if body is None:
        try:
            body = response.json()
        except (ValueError, UnicodeError):
            body = response.text
    details = body.get("error", body) if isinstance(body, dict) else body
    message = details.get("message") if isinstance(details, dict) else details
    status = response.status_code if status_code is None else status_code
    cls = {400: BadRequestError, 401: AuthenticationError, 403: PermissionDeniedError,
           404: NotFoundError, 409: ConflictError, 422: UnprocessableEntityError,
           429: RateLimitError}.get(status, InternalServerError if status >= 500 else APIStatusError)
    return cls(f"HTTP {status}: {message or body}", response=response, body=body, status_code=status)


@dataclass(frozen=True)
class SSEEvent:
    event: str | None
    data: str
    id: str | None = None
    retry: int | None = None

    def json(self):
        return json.loads(self.data)


class SSEDecoder:
    """SSE line decoder: UTF-8 and transport buffering are owned by httpx."""

    def __init__(self):
        self.data = []
        self.event = None
        self.id = None
        self.retry = None
        self.first = True

    def feed(self, line):
        if self.first:
            line = line.removeprefix("\ufeff")
            self.first = False
        if line == "":
            result = SSEEvent(self.event, "\n".join(self.data), self.id, self.retry) if self.data else None
            self.data, self.event, self.retry = [], None, None
            return result
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            self.data.append(value)
        elif field == "event":
            self.event = value
        elif field == "id" and "\0" not in value:
            self.id = value
        elif field == "retry" and value.isascii() and value.isdecimal():
            self.retry = int(value)
        return None


def iter_sse(response):
    decoder = SSEDecoder()
    response.encoding = "utf-8"
    for line in response.iter_lines():
        event = decoder.feed(line)
        if event is not None:
            yield event
    # A dropped partial event is not a completed frame. The Hermes stream
    # consumer decides whether the absence of a finish marker permits a retry.


async def aiter_sse(response):
    decoder = SSEDecoder()
    response.encoding = "utf-8"
    async for line in response.aiter_lines():
        event = decoder.feed(line)
        if event is not None:
            yield event


def event_object(event, response):
    try:
        body = event.json()
    except (ValueError, UnicodeError) as exc:
        raise APIResponseValidationError("Invalid JSON in provider event stream", response=response,
                                         body=event.data) from exc
    if event.event == "error" or isinstance(body, dict) and body.get("error"):
        details = body.get("error", body) if isinstance(body, dict) else {}
        code = details.get("status_code", details.get("status", details.get("code"))) if isinstance(details, dict) else None
        try:
            status = int(code)
        except (ValueError, TypeError):
            status = 500
        if not 400 <= status <= 599:
            status = 500
        raise status_error(response, body, status_code=status)
    return wire_object(body)


class JSONStream:
    """One response lease; closing a stream never closes its parent client."""

    def __init__(self, response):
        self.response = response
        self._iterator = self._iterate()

    def _iterate(self):
        try:
            for event in iter_sse(self.response):
                if event.data.strip() == "[DONE]":
                    return
                yield event_object(event, self.response)
        finally:
            self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self):
        self.response.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class AsyncJSONStream:
    def __init__(self, response):
        self.response = response
        self._iterator = self._iterate()

    async def _iterate(self):
        try:
            async for event in aiter_sse(self.response):
                if event.data.strip() == "[DONE]":
                    return
                yield event_object(event, self.response)
        finally:
            await self.close()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._iterator.__anext__()

    async def close(self):
        await self.response.aclose()

    aclose = close

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        await self.close()
