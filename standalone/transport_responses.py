"""Responses resources over the same owned HTTP connections as Chat Completions.

Hermes' existing Codex response assembler owns event accumulation, incomplete
response policy, reasoning replay and tool-call identity. This boundary carries
every JSON field through unchanged, including unknown provider extensions.
"""

from __future__ import annotations

from urllib.parse import quote

from standalone.transport import split_request_options
from standalone.transport_chat import response_object
from standalone.transport_types import AsyncJSONStream, JSONStream, WireObject


class ResponseObject(WireObject):
    @property
    def output_text(self):
        if "output_text" in self:
            return self["output_text"]
        return "".join(
            part.get("text", "")
            for item in self.get("output", []) if item.get("type") == "message"
            for part in item.get("content", []) if part.get("type") == "output_text"
        )


def _response(response):
    result = ResponseObject(response_object(response))
    result._request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
    return result


class Responses:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        body, options = split_request_options(kwargs)
        streaming = bool(body.get("stream", False))
        response = self._client.post("responses", body=body, stream=streaming, **options)
        if streaming:
            return JSONStream(response)
        try:
            return _response(response)
        finally:
            response.close()

    def compact(self, **kwargs):
        body, options = split_request_options(kwargs)
        response = self._client.post("responses/compact", body=body, **options)
        try:
            return _response(response)
        finally:
            response.close()

    def retrieve(self, response_id, **kwargs):
        params, options = split_request_options(kwargs)
        options["query"] = {**params, **(options["query"] or {})}
        response = self._client.get("responses/" + quote(str(response_id), safe=""), **options)
        try:
            return _response(response)
        finally:
            response.close()


class AsyncResponses:
    def __init__(self, client):
        self._client = client

    async def create(self, **kwargs):
        body, options = split_request_options(kwargs)
        streaming = bool(body.get("stream", False))
        response = await self._client.post("responses", body=body, stream=streaming, **options)
        if streaming:
            return AsyncJSONStream(response)
        try:
            return _response(response)
        finally:
            await response.aclose()

    async def compact(self, **kwargs):
        body, options = split_request_options(kwargs)
        response = await self._client.post("responses/compact", body=body, **options)
        try:
            return _response(response)
        finally:
            await response.aclose()

    async def retrieve(self, response_id, **kwargs):
        params, options = split_request_options(kwargs)
        options["query"] = {**params, **(options["query"] or {})}
        response = await self._client.get("responses/" + quote(str(response_id), safe=""), **options)
        try:
            return _response(response)
        finally:
            await response.aclose()
