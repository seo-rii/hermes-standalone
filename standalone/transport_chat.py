"""Chat Completions wire adapter at Hermes' existing client boundary."""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

from standalone.transport import AsyncHTTPClient, SyncHTTPClient, UNSET, split_request_options
from standalone.transport_types import (
    APIResponseValidationError, AsyncJSONStream, JSONStream, wire_object,
)


def response_object(response):
    try:
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("Expected an object response")
        result = wire_object(value)
        result._request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        return result
    except (ValueError, UnicodeError) as exc:
        raise APIResponseValidationError("Provider returned an invalid JSON object", response=response,
                                         body=response.text) from exc


class Completions:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        body, options = split_request_options(kwargs)
        streaming = bool(body.get("stream", False))
        response = self._client.post("chat/completions", body=body, stream=streaming, **options)
        if streaming:
            return JSONStream(response)
        try:
            return response_object(response)
        finally:
            response.close()


class AsyncCompletions:
    def __init__(self, client):
        self._client = client

    async def create(self, **kwargs):
        body, options = split_request_options(kwargs)
        streaming = bool(body.get("stream", False))
        response = await self._client.post("chat/completions", body=body, stream=streaming, **options)
        if streaming:
            return AsyncJSONStream(response)
        try:
            return response_object(response)
        finally:
            await response.aclose()


class Models:
    def __init__(self, client):
        self._client = client

    def list(self, **kwargs):
        body, options = split_request_options(kwargs)
        if body:
            options["query"] = {**body, **(options["query"] or {})}
        response = self._client.get("models", **options)
        try:
            return response_object(response)
        finally:
            response.close()


class AsyncModels:
    def __init__(self, client):
        self._client = client

    async def list(self, **kwargs):
        body, options = split_request_options(kwargs)
        if body:
            options["query"] = {**body, **(options["query"] or {})}
        response = await self._client.get("models", **options)
        try:
            return response_object(response)
        finally:
            await response.aclose()


class _ClientIdentity:
    def _identity(self, api_key, organization, project):
        key = os.environ.get("OPENAI_API_KEY", "") if api_key is None else api_key
        self._api_key_provider = key if callable(key) else None
        self.api_key = "" if callable(key) else key
        self.organization = organization
        self.project = project

    @property
    def auth_headers(self):
        return self._auth_headers(self.api_key)

    def _auth_headers(self, token):
        headers = {}
        if token:
            if not isinstance(token, str):
                raise TypeError("api_key must be a string or a callable returning a string")
            headers["Authorization"] = f"Bearer {token}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        return headers

    @property
    def default_headers(self):
        import httpx
        headers = httpx.Headers({"Accept": "application/json", **self.auth_headers})
        headers.update(self._custom_headers)
        return dict(headers)

    def _credential_headers(self, token, headers):
        import httpx
        result = httpx.Headers(self._auth_headers(token))
        result.update(self._custom_headers)
        result.update(headers or {})
        return result


class OpenAI(_ClientIdentity, SyncHTTPClient):
    """Hermes' synchronous, provider-neutral OpenAI-compatible wire client."""

    def __init__(self, *, api_key=None, base_url=None, organization=None, project=None,
                 timeout=UNSET, max_retries=2, default_headers=None, default_query=None,
                 http_client=None, _strict_response_validation=False):
        self._identity(api_key, organization, project)
        super().__init__(base_url=base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
                         timeout=timeout, max_retries=max_retries, default_headers=default_headers,
                         default_query=default_query, http_client=http_client)

    def _prepare_headers(self, headers):
        token = self._api_key_provider() if self._api_key_provider else self.api_key
        if inspect.isawaitable(token):
            if inspect.iscoroutine(token):
                token.close()
            raise TypeError("A synchronous client cannot await an asynchronous api_key provider")
        return self._credential_headers(token, headers)

    @property
    def chat(self):
        return SimpleNamespace(completions=Completions(self))

    @property
    def models(self):
        return Models(self)

    @property
    def responses(self):
        from standalone.transport_responses import Responses
        return Responses(self)

    @property
    def audio(self):
        from standalone.transport_media import audio
        return audio(self)

    @property
    def images(self):
        from standalone.transport_media import Images
        return Images(self)

    @property
    def videos(self):
        from standalone.transport_media import Videos
        return Videos(self)


class AsyncOpenAI(_ClientIdentity, AsyncHTTPClient):
    def __init__(self, *, api_key=None, base_url=None, organization=None, project=None,
                 timeout=UNSET, max_retries=2, default_headers=None, default_query=None,
                 http_client=None, _strict_response_validation=False):
        self._identity(api_key, organization, project)
        super().__init__(base_url=base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
                         timeout=timeout, max_retries=max_retries, default_headers=default_headers,
                         default_query=default_query, http_client=http_client)

    async def _prepare_headers(self, headers):
        token = self._api_key_provider() if self._api_key_provider else self.api_key
        if inspect.isawaitable(token):
            token = await token
        return self._credential_headers(token, headers)

    @property
    def chat(self):
        return SimpleNamespace(completions=AsyncCompletions(self))

    @property
    def models(self):
        return AsyncModels(self)

    @property
    def responses(self):
        from standalone.transport_responses import AsyncResponses
        return AsyncResponses(self)
