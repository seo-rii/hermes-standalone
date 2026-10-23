"""Anthropic Messages wire client with ordered, lossless SSE accumulation.

This implements the Messages client boundary used by the original Hermes
adapter. It does not install an ``anthropic`` module or emulate Pydantic.
"""

from __future__ import annotations

import json

from standalone.transport import SyncHTTPClient, UNSET, split_request_options
from standalone.transport_chat import response_object
from standalone.transport_types import (
    APIConnectionError, APIError, APIResponseValidationError, APITimeoutError,
    APIStatusError, AuthenticationError, BadRequestError, InternalServerError,
    JSONStream, NotFoundError, PermissionDeniedError, RateLimitError,
    UnprocessableEntityError, WireObject, wire_object,
)


class Omit:
    """Explicitly remove an inherited header, case-insensitively."""


def _merge_headers(*mappings):
    headers = {}
    for mapping in mappings:
        for name, value in (mapping or {}).items():
            key = str(name).lower()
            if isinstance(value, Omit):
                headers.pop(key, None)
            else:
                headers[key] = value
    return headers


class MessageStream:
    """Own one response, exposing raw events and an accumulated Message snapshot."""

    def __init__(self, response, events=None):
        self.response = response
        self._events = JSONStream(response) if events is None else events
        self._message = None
        self._complete = False
        self._inputs = {}
        self._stopped = set()
        self._iterator = self._iterate()

    def _invalid(self, message, event=None):
        raise APIResponseValidationError(message, response=self.response, body=event)

    @property
    def current_message_snapshot(self):
        if self._message is None:
            raise RuntimeError("The message stream has not started")
        return self._message

    def _block(self, event):
        index = event.get("index")
        content = self.current_message_snapshot.content
        if type(index) is not int or not 0 <= index < len(content) or index in self._stopped:
            self._invalid("Unexpected content block index in message stream", event)
        return index, content[index]

    def _delta(self, event):
        index, block = self._block(event)
        delta = event.get("delta")
        if not isinstance(delta, dict):
            self._invalid("Invalid content block delta", event)
        kind = delta.get("type")
        field = {"text_delta": "text", "thinking_delta": "thinking", "signature_delta": "signature"}.get(kind)
        if field:
            value = delta.get(field)
            if not isinstance(value, str) or not isinstance(block.get(field, ""), str):
                self._invalid("Invalid text in message stream delta", event)
            block[field] = block.get(field, "") + value
        elif kind == "input_json_delta":
            value = delta.get("partial_json")
            if not isinstance(value, str):
                self._invalid("Invalid tool JSON delta", event)
            self._inputs[index] = self._inputs.get(index, "") + value
            # A complete prefix is useful to interruption/runaway handling;
            # an incomplete prefix is never substituted for the final input.
            try:
                block.input = json.loads(self._inputs[index])
            except ValueError:
                pass
        elif kind == "citations_delta":
            citation = delta.get("citation")
            if not isinstance(citation, dict):
                self._invalid("Invalid citation delta", event)
            if block.get("citations") is None:
                block.citations = []
            block.citations.append(wire_object(citation))
        else:
            # Preserve novel deltas for consumers without inventing a schema
            # or silently interpreting a provider's unknown append operation.
            block.setdefault("provider_deltas", []).append(wire_object(delta))

    def _finish_input(self, index, block):
        if index not in self._inputs:
            return
        try:
            value = json.loads(self._inputs[index])
        except ValueError as exc:
            raise APIResponseValidationError("Incomplete tool JSON in message stream",
                                             response=self.response, body=self._inputs[index]) from exc
        if not isinstance(value, dict):
            self._invalid("Tool input must be a JSON object", value)
        block.input = value

    def _accumulate(self, event):
        if not isinstance(event, dict):
            self._invalid("Expected an object in message stream", event)
        kind = event.get("type")
        if kind == "ping":
            return
        if kind == "message_start":
            if self._message is not None or not isinstance(event.get("message"), dict):
                self._invalid("Unexpected message_start in message stream", event)
            self._message = event.message
            if not isinstance(self._message.get("content"), list):
                self._invalid("Message content must be a list", event)
            if self._message.get("usage") is None:
                self._message.usage = {"input_tokens": 0, "output_tokens": 0}
            self._message.setdefault("stop_reason", None)
            self._message.setdefault("stop_sequence", None)
            return
        if self._message is None:
            self._invalid("Message event preceded message_start", event)
        if kind == "content_block_start":
            index, block = event.get("index"), event.get("content_block")
            if type(index) is not int or index != len(self._message.content) or not isinstance(block, dict):
                self._invalid("Unexpected content_block_start in message stream", event)
            self._message.content.append(wire_object(block))
        elif kind == "content_block_delta":
            self._delta(event)
        elif kind == "content_block_stop":
            index, block = self._block(event)
            self._finish_input(index, block)
            self._stopped.add(index)
        elif kind == "message_delta":
            delta = event.get("delta")
            if not isinstance(delta, dict):
                self._invalid("Invalid message_delta", event)
            for key, value in delta.items():
                self._message[key] = wire_object(value)
            usage = event.get("usage")
            if usage is not None:
                if not isinstance(usage, dict):
                    self._invalid("Invalid message usage", event)
                for key, value in usage.items():
                    if value is not None:
                        self._message.usage[key] = wire_object(value)
        elif kind == "message_stop":
            for index, block in enumerate(self._message.content):
                self._finish_input(index, block)
            self._complete = True

    def _iterate(self):
        try:
            for event in self._events:
                self._accumulate(event)
                yield event
                if self._complete:
                    break
        finally:
            self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    @property
    def text_stream(self):
        for event in self:
            if event.get("type") == "content_block_delta" and event.delta.get("type") == "text_delta":
                yield event.delta.text

    def get_final_message(self):
        for _event in self:
            pass
        if not self._complete:
            self._invalid("Messages stream ended before message_stop")
        return self.current_message_snapshot

    def get_final_text(self):
        return "".join(block.get("text", "") for block in self.get_final_message().content
                       if block.get("type") == "text")

    def close(self):
        self._events.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class _StreamManager:
    def __init__(self, messages, kwargs):
        self._messages, self._kwargs = messages, kwargs
        self._stream = None

    def __enter__(self):
        if self._stream is not None:
            raise RuntimeError("Message stream manager cannot be entered twice")
        self._stream = self._messages.create(**{**self._kwargs, "stream": True})
        return self._stream

    def __exit__(self, *_args):
        if self._stream is not None:
            self._stream.close()


class Messages:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        body, options = split_request_options(kwargs)
        streaming = bool(body.get("stream", False))
        response = self._client.post("v1/messages", body=body, stream=streaming, **options)
        if streaming:
            return MessageStream(response)
        try:
            return response_object(response)
        finally:
            response.close()

    def stream(self, **kwargs):
        return _StreamManager(self, kwargs)

    def count_tokens(self, **kwargs):
        body, options = split_request_options(kwargs)
        response = self._client.post("v1/messages/count_tokens", body=body, **options)
        try:
            return response_object(response)
        finally:
            response.close()


class Anthropic(SyncHTTPClient):
    def __init__(self, *, api_key=None, auth_token=None, base_url="https://api.anthropic.com",
                 timeout=UNSET, max_retries=2, default_headers=None, default_query=None,
                 http_client=None):
        self.api_key, self.auth_token = api_key, auth_token
        super().__init__(base_url=base_url, timeout=timeout, max_retries=max_retries,
                         default_headers=default_headers, default_query=default_query,
                         http_client=http_client)

    @property
    def messages(self):
        return Messages(self)

    @property
    def default_headers(self):
        return _merge_headers({"accept": "application/json", "anthropic-version": "2023-06-01"})

    def _prepare_headers(self, headers):
        auth = {}
        if self.api_key:
            auth["x-api-key"] = self.api_key
        if self.auth_token:
            auth["authorization"] = "Bearer " + self.auth_token
        return _merge_headers(auth, self._custom_headers, headers)


def __getattr__(name):
    if name == "AnthropicBedrock":
        from standalone.transport_bedrock import AnthropicBedrock
        return AnthropicBedrock
    raise AttributeError(name)
