"""HTTPX-backed wire primitives shared by the standalone provider adapters.

No provider SDK is imported. Hermes still owns main-turn retries, socket aborts,
provider routing and request policy. The bounded retry option serves auxiliary
clients that historically relied on their SDK's retries.
"""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import threading
import time

from standalone.transport_types import (
    APIConnectionError, APITimeoutError, json_value, status_error,
)


UNSET = object()


def split_request_options(kwargs):
    """Separate request options from JSON, with extra_body taking precedence."""
    body = dict(kwargs)
    options = {"headers": body.pop("extra_headers", None), "query": body.pop("extra_query", None),
               "timeout": body.pop("timeout", UNSET)}
    extra = body.pop("extra_body", None)
    if extra is not None:
        if not isinstance(extra, dict):
            raise TypeError("extra_body must be a mapping")
        body.update(extra)
    return body, options


class _ClientLease:
    def __init__(self, client):
        self.client, self.count = client, 1
        self.lock = threading.Lock()

    def retain(self):
        with self.lock:
            self.count += 1

    def release(self):
        with self.lock:
            self.count -= 1
            return self.count == 0


def _retry_delay(response, attempt):
    if response is not None:
        headers = response.headers
        try:
            if "retry-after-ms" in headers:
                delay = float(headers["retry-after-ms"]) / 1000
            elif "retry-after" in headers:
                try:
                    delay = float(headers["retry-after"])
                except ValueError:
                    date = parsedate_to_datetime(headers["retry-after"])
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    delay = (date - datetime.now(timezone.utc)).total_seconds()
            else:
                delay = None
            if delay is not None and 0 <= delay <= 60:
                return delay
        except (ValueError, TypeError, OverflowError):
            pass
    return min(0.5 * (2 ** min(attempt, 5)), 8.0)


def _should_retry(response):
    instructed = response.headers.get("x-should-retry")
    if instructed in ("true", "false"):
        return instructed == "true"
    return response.status_code in (408, 409, 429) or response.status_code >= 500


class _HTTPSettings:
    def _initialize(self, *, base_url, timeout, max_retries, default_headers, default_query,
                    http_client, async_client):
        import httpx
        if not isinstance(max_retries, int) or not 0 <= max_retries <= 100:
            raise ValueError("max_retries must be an integer between 0 and 100")
        self.base_url = httpx.URL(str(base_url))
        if self.base_url.scheme not in ("http", "https") or not self.base_url.host:
            raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
        if not self.base_url.path.endswith("/"):
            self.base_url = self.base_url.copy_with(path=self.base_url.path + "/")
        self.max_retries = max_retries
        self._custom_headers = dict(default_headers or {})
        self.default_query = dict(default_query or {})
        client_type = httpx.AsyncClient if async_client else httpx.Client
        if http_client is not None and not isinstance(http_client, client_type):
            raise TypeError(f"http_client must be an {client_type.__name__}")
        self.timeout = (http_client.timeout if http_client is not None else httpx.Timeout(600, connect=5)) if timeout is UNSET else timeout
        self._client = http_client if http_client is not None else client_type(timeout=self.timeout, follow_redirects=True)
        self._lease = _ClientLease(self._client)
        self._closed = False

    @property
    def default_headers(self):
        return {"Accept": "application/json", **self._custom_headers}

    def is_closed(self):
        return self._closed or self._client.is_closed

    def with_options(self, *, default_headers=UNSET, default_query=UNSET, timeout=UNSET,
                     max_retries=UNSET, base_url=UNSET, **kwargs):
        if self.is_closed():
            raise RuntimeError("Cannot copy a closed client")
        clone = copy.copy(self)
        clone._custom_headers = dict(self._custom_headers)
        clone.default_query = dict(self.default_query)
        if default_headers is not UNSET:
            clone._custom_headers.update(default_headers or {})
        if default_query is not UNSET:
            clone.default_query.update(default_query or {})
        if timeout is not UNSET:
            clone.timeout = timeout
        if max_retries is not UNSET:
            if not isinstance(max_retries, int) or not 0 <= max_retries <= 100:
                raise ValueError("max_retries must be an integer between 0 and 100")
            clone.max_retries = max_retries
        if base_url is not UNSET:
            import httpx
            url = httpx.URL(str(base_url))
            if url.scheme not in ("http", "https") or not url.host:
                raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
            clone.base_url = url.copy_with(path=url.path.rstrip("/") + "/")
        for name, value in kwargs.items():
            if name not in ("api_key", "organization", "project"):
                raise TypeError(f"Unsupported client option: {name}")
            if name == "api_key":
                clone._api_key_provider = value if callable(value) else None
                clone.api_key = "" if callable(value) else value
            else:
                setattr(clone, name, value)
        clone._lease.retain()
        clone._closed = False
        return clone

    def _build_request(self, method, path, *, body, headers, query, timeout,
                       files=None, data=None, raw_content=UNSET):
        import httpx
        if self.is_closed():
            raise RuntimeError("Cannot send a request, as the client has been closed")
        relative = httpx.URL(path)
        if relative.is_absolute_url:
            if (relative.scheme, relative.host, relative.port) != (self.base_url.scheme, self.base_url.host, self.base_url.port):
                raise ValueError("Provider requests must use the configured endpoint origin")
            url = relative
        else:
            url = self.base_url.join(path.lstrip("/"))
        # Joining a relative path discards the base query; preserve its gateway
        # parameters, then let explicit client and request parameters override.
        params = self.base_url.params.merge(self.default_query).merge(url.params).merge(query or {})
        merged = httpx.Headers(self.default_headers)
        merged.update(headers or {})
        content = None if raw_content is UNSET else raw_content
        if body is not None:
            if files is not None or data is not None or raw_content is not UNSET:
                raise TypeError("JSON and multipart/raw request bodies are mutually exclusive")
            content = json.dumps(json_value(body), ensure_ascii=False, allow_nan=False,
                                 separators=(",", ":")).encode("utf-8")
            if "content-type" not in merged:
                merged["content-type"] = "application/json"
        return self._client.build_request(method, url.copy_with(query=b""), params=params, content=content,
                                          files=files, data=data,
                                          headers=merged, timeout=self.timeout if timeout is UNSET else timeout)


class SyncHTTPClient(_HTTPSettings):
    def __init__(self, *, base_url, timeout=UNSET, max_retries=0, default_headers=None,
                 default_query=None, http_client=None):
        self._initialize(base_url=base_url, timeout=timeout, max_retries=max_retries,
                         default_headers=default_headers, default_query=default_query,
                         http_client=http_client, async_client=False)

    def _prepare_headers(self, headers):
        return headers

    def request(self, method, path, *, body=None, headers=None, query=None, timeout=UNSET, stream=False,
                files=None, data=None, raw_content=UNSET):
        import httpx
        for attempt in range(self.max_retries + 1):
            request = self._build_request(method, path, body=body, headers=self._prepare_headers(headers),
                                          query=query, timeout=timeout, files=files, data=data, raw_content=raw_content)
            try:
                response = self._client.send(request, stream=stream)
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    time.sleep(_retry_delay(None, attempt))
                    continue
                raise APITimeoutError("Provider request timed out", request=request) from exc
            except httpx.TransportError as exc:
                if attempt < self.max_retries:
                    time.sleep(_retry_delay(None, attempt))
                    continue
                raise APIConnectionError(f"Provider connection failed: {exc}", request=request) from exc
            if response.is_success:
                return response
            try:
                response.read()
                if attempt < self.max_retries and _should_retry(response):
                    delay = _retry_delay(response, attempt)
                else:
                    raise status_error(response)
            finally:
                response.close()
            time.sleep(delay)
        raise AssertionError("Unreachable retry state")

    def post(self, path, *, body=None, **kwargs):
        return self.request("POST", path, body=body, **kwargs)

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def close(self):
        if not self._closed:
            self._closed = True
            if self._lease.release():
                self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class AsyncHTTPClient(_HTTPSettings):
    def __init__(self, *, base_url, timeout=UNSET, max_retries=0, default_headers=None,
                 default_query=None, http_client=None):
        self._initialize(base_url=base_url, timeout=timeout, max_retries=max_retries,
                         default_headers=default_headers, default_query=default_query,
                         http_client=http_client, async_client=True)

    async def _prepare_headers(self, headers):
        return headers

    async def request(self, method, path, *, body=None, headers=None, query=None, timeout=UNSET, stream=False,
                      files=None, data=None, raw_content=UNSET):
        import httpx
        for attempt in range(self.max_retries + 1):
            request = self._build_request(method, path, body=body, headers=await self._prepare_headers(headers),
                                          query=query, timeout=timeout, files=files, data=data, raw_content=raw_content)
            try:
                response = await self._client.send(request, stream=stream)
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    await asyncio.sleep(_retry_delay(None, attempt))
                    continue
                raise APITimeoutError("Provider request timed out", request=request) from exc
            except httpx.TransportError as exc:
                if attempt < self.max_retries:
                    await asyncio.sleep(_retry_delay(None, attempt))
                    continue
                raise APIConnectionError(f"Provider connection failed: {exc}", request=request) from exc
            if response.is_success:
                return response
            try:
                await response.aread()
                if attempt < self.max_retries and _should_retry(response):
                    delay = _retry_delay(response, attempt)
                else:
                    raise status_error(response)
            finally:
                await response.aclose()
            await asyncio.sleep(delay)
        raise AssertionError("Unreachable retry state")

    async def post(self, path, *, body=None, **kwargs):
        return await self.request("POST", path, body=body, **kwargs)

    async def get(self, path, **kwargs):
        return await self.request("GET", path, **kwargs)

    async def close(self):
        if not self._closed:
            self._closed = True
            if self._lease.release():
                await self._client.aclose()

    aclose = close

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        await self.close()
