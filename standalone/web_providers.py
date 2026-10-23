"""Qualified HTTP boundaries for Hermes' existing Exa, Firecrawl and Parallel plugins.

Wire contracts reviewed against uv.lock's Exa 2.10.2, Firecrawl 4.17.0 and
Parallel 0.4.2 sources and the vendors' API references:
https://exa.ai/docs/reference/search
https://docs.firecrawl.dev/api-reference/endpoint/search
https://docs.parallel.ai/api-reference/search/search

These clients implement only the methods consumed by the original plugins.
They are not imported under third-party SDK names. Credential selection,
profile scoping, billing routes, caches and result normalization remain in
the original provider implementations.
"""

import os
import time
from urllib.parse import urljoin

from standalone.transport import AsyncHTTPClient, SyncHTTPClient
from standalone.transport_types import APIConnectionError, APIStatusError, WireObject


EXA_BASE_URL = "https://api.exa.ai"
PARALLEL_BASE_URL = "https://api.parallel.ai"
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev"


def _client_options(base_url, headers, retries=0):
    import httpx

    # API keys must remain on the configured vendor/gateway origin. A redirect
    # is surfaced as an HTTP error, including when it points to another host.
    return dict(base_url=base_url, max_retries=retries, default_headers=headers,
                http_client=httpx.Client(timeout=60.0, follow_redirects=False))


def _object(response):
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("Web provider returned a non-object JSON response")
    return value


def _records(value):
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("Web provider returned invalid result records")
    return [WireObject({"url": None, "title": None, "text": None, "highlights": None,
                        "excerpts": None, "full_content": None, **row}) for row in value]


def _results(value):
    result = WireObject(value)
    result.results = _records(value.get("results", []))
    result.errors = [WireObject({"url": None, "content": None, "error_type": None, **row})
                     for row in value.get("errors") or []]
    return result


class Exa:
    def __init__(self, *, api_key, base_url=None):
        self.base_url = base_url or EXA_BASE_URL
        self.headers = {"x-api-key": api_key, "User-Agent": "hermes-standalone/exa-2.10.2",
                        "Content-Type": "application/json"}

    def _post(self, endpoint, payload):
        try:
            with SyncHTTPClient(**_client_options(self.base_url, self.headers)) as client:
                response = client.post(endpoint, body=payload)
                try:
                    return _results(_object(response))
                finally:
                    response.close()
        except APIStatusError as error:
            raise ValueError(f"Request failed with status code {error.status_code}: {error.response.text}") from error

    def search(self, query, *, num_results=10, contents=None):
        return self._post("search", {"query": query, "numResults": num_results,
                                    "contents": {"text": {"maxCharacters": 10000}} if contents is None else contents})

    def get_contents(self, urls, *, text=True):
        return self._post("contents", {"urls": [urls] if isinstance(urls, str) else list(urls), "text": text})


class Parallel:
    def __init__(self, *, api_key, base_url=None):
        self.base_url = base_url or os.environ.get("PARALLEL_BASE_URL") or PARALLEL_BASE_URL
        self.headers = {"x-api-key": api_key, "parallel-beta": "search-extract-2025-10-10",
                        "User-Agent": "hermes-standalone/parallel-0.4.2"}
        self.beta = self

    def search(self, *, search_queries, objective, mode, max_results):
        with SyncHTTPClient(**_client_options(self.base_url, self.headers, retries=2)) as client:
            response = client.post("v1beta/search", body={"search_queries": search_queries, "objective": objective,
                                                         "mode": mode, "max_results": max_results})
            try:
                return _results(_object(response))
            finally:
                response.close()


class AsyncParallel(Parallel):
    async def extract(self, *, urls, full_content=True):
        import httpx

        async with AsyncHTTPClient(base_url=self.base_url, max_retries=2, default_headers=self.headers,
                http_client=httpx.AsyncClient(timeout=60.0, follow_redirects=False)) as client:
            response = await client.post("v1beta/extract", body={"urls": list(urls), "full_content": full_content})
            try:
                return _results(_object(response))
            finally:
                await response.aclose()


class FirecrawlError(RuntimeError):
    def __init__(self, message, response):
        super().__init__(message)
        self.status_code, self.response = response.status_code, response


def _firecrawl_error(response, action):
    try:
        value = response.json()
        message, detail = value.get("error", "No error message provided."), value.get("details", "No additional error details provided.")
    except (ValueError, AttributeError):
        message = f"Server returned non-JSON response: {response.text[:500]}"
        detail = f"Full response status: {response.status_code}"
    labels = {400: "Bad Request", 401: "Unauthorized", 402: "Payment Required", 403: "Website Not Supported",
              408: "Request Timeout", 429: "Rate Limit Exceeded", 500: "Internal Server Error"}
    label = labels.get(response.status_code, f"Unexpected error (HTTP {response.status_code})")
    return FirecrawlError(f"{label}: Failed to {action}. {message} - {detail}", response)


class Firecrawl:
    def __init__(self, *, api_key=None, api_url=None):
        self.api_url = api_url or FIRECRAWL_BASE_URL
        self.headers = {"Content-Type": "application/json", "User-Agent": "hermes-standalone/firecrawl-4.17.0"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key
        elif "api.firecrawl.dev" in self.api_url.lower():
            raise ValueError("API key is required for the cloud API. Set FIRECRAWL_API_KEY or pass api_key.")

    def _post(self, action, payload):
        payload = {**payload, "origin": "python-sdk@4.17.0"}
        for attempt in range(3):
            try:
                with SyncHTTPClient(**_client_options(self.api_url, self.headers)) as client:
                    response = client.post(urljoin(self.api_url.rstrip("/") + "/", "/v2/" + action), body=payload)
                    try:
                        value = _object(response)
                        if not value.get("success"):
                            raise _firecrawl_error(response, action)
                        return value
                    finally:
                        response.close()
            except APIStatusError as error:
                if error.status_code != 502 or attempt == 2:
                    raise _firecrawl_error(error.response, action) from error
            except APIConnectionError:
                if attempt == 2:
                    raise
            time.sleep(0.5 * 2 ** attempt)
        raise AssertionError("Unreachable Firecrawl retry state")

    def search(self, *, query, limit=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Limit must be positive and cannot exceed 100")
        return self._post("search", {"query": query, "limit": limit})

    def scrape(self, *, url, formats, timeout=None):
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL cannot be empty")
        payload = {"url": url.strip(), "formats": formats}
        if timeout is not None:
            payload["timeout"] = timeout
        return self._post("scrape", payload)
