"""Fetch service wire operation used by the embedded browser HTTP helper.

The upstream helper consumes only ``fetch_sync(...).text``. This qualified
application boundary sends the pinned client's published /fetch JSON contract;
it does not publish a replacement fetch_use namespace.
"""

import base64
from dataclasses import dataclass
import json
import os
from urllib.parse import urlsplit
import uuid


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchResponse:
    status_code: int
    text: str
    headers: dict
    url: str


def fetch_sync(url, *, headers=None, timeout_ms=30000):
    import httpx
    from agent.ssl_verify import resolve_httpx_verify

    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Browser fetch requires an HTTP(S) target URL")
    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError("Browser fetch timeout must be a positive integer")
    key = os.environ.get("BROWSER_USE_API_KEY")
    if not key:
        raise ValueError("BROWSER_USE_API_KEY is required for the configured fetch service")
    endpoint = os.environ.get("FETCH_USE_URL", "https://fetch.browser-use.com").rstrip("/")
    service = urlsplit(endpoint)
    if service.scheme not in ("http", "https") or not service.hostname or service.username or service.password:
        raise ValueError("Invalid fetch service URL")
    session = os.environ.get("SESSION_ID") or str(uuid.uuid4())
    payload = {"url": url, "method": "GET", "timeout_ms": min(timeout_ms, 120000),
               "follow_redirects": True, "max_redirects": 10, "proxy_country": "US", "session_id": session}
    if headers:
        payload["headers"] = dict(headers)
    auth = {"X-Browser-Use-API-Key": key}
    if os.environ.get("FETCH_USE_AUTH_TOKEN"):
        auth["X-Fetch-Token"] = os.environ["FETCH_USE_AUTH_TOKEN"]
    with httpx.Client(verify=resolve_httpx_verify(), timeout=timeout_ms / 1000 + 10, follow_redirects=False) as client:
        with client.stream("POST", endpoint + "/fetch", json=payload, headers=auth) as response:
            content = bytearray()
            for block in response.iter_bytes():
                content.extend(block)
                if len(content) > 32 * 1024 * 1024:
                    raise FetchError("Fetch service response exceeds the size limit")
            if not 200 <= response.status_code < 300:
                raise FetchError(f"Fetch service returned HTTP {response.status_code}")
    try:
        result = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise FetchError("Fetch service returned malformed JSON") from exc
    if not isinstance(result, dict) or result.get("error"):
        raise FetchError("Fetch service reported a failed request")
    status, body, response_headers = result.get("status_code"), result.get("body", ""), result.get("headers", {})
    if type(status) is not int or not isinstance(body, str) or not isinstance(response_headers, dict):
        raise FetchError("Fetch service response has an invalid shape")
    if result.get("is_binary"):
        try:
            body = base64.b64decode(result.get("body_base64", ""), validate=True).decode("utf-8", "replace")
        except (ValueError, TypeError) as exc:
            raise FetchError("Fetch service returned malformed binary data") from exc
    return FetchResponse(status, body, response_headers, result.get("final_url") or url)
