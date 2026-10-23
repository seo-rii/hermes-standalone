"""Async JSON-RPC transports preserving Hermes' existing MCP lifecycle ownership."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager, suppress
import json
import os
import sys
from urllib.parse import urljoin, urlsplit

from standalone.mcp_types import LATEST_HANDSHAKE_VERSION, RequestFailure, validate_message
from standalone.transport_types import aiter_sse, json_value


MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def _header_value(value):
    text = str(value)
    if (text != text.strip() or any(ord(char) < 32 or ord(char) > 126 for char in text)
            or text.startswith("=?base64?") and text.endswith("?=")):
        return "=?base64?" + base64.b64encode(text.encode("utf-8")).decode("ascii") + "?="
    return text


class _QueueTransport:
    def __init__(self):
        self.incoming = asyncio.Queue(maxsize=256)
        self.jobs = set()
        self.protocol_version = LATEST_HANDSHAKE_VERSION
        self.closed = False

    async def receive(self):
        item = await self.incoming.get()
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, RequestFailure) else validate_message(item)

    def _start(self, coroutine, *, request_id=None):
        async def run():
            try:
                await coroutine
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if not self.closed:
                    await self.incoming.put(RequestFailure(request_id, error) if request_id is not None else error)
        task = asyncio.create_task(run())
        self.jobs.add(task)
        task.add_done_callback(self.jobs.discard)
        return task

    async def _close_jobs(self):
        self.closed = True
        jobs = list(self.jobs)
        for task in jobs:
            task.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        self.jobs.clear()


class _StdioTransport(_QueueTransport):
    def __init__(self, process, parameters):
        super().__init__()
        self.process, self.parameters = process, parameters
        self._write_lock = asyncio.Lock()
        self._start(self._read())

    async def _read(self):
        while True:
            line = await self.process.stdout.readline()
            if not line:
                raise ConnectionError("MCP stdio connection closed")
            if len(line) > MAX_MESSAGE_BYTES:
                raise ValueError("MCP stdio message exceeds the wire limit")
            value = json.loads(line.decode(self.parameters.encoding, self.parameters.encoding_error_handler))
            await self.incoming.put(validate_message(value))

    async def send(self, message):
        if self.closed or self.process.returncode is not None:
            raise ConnectionError("MCP stdio connection closed")
        data = json.dumps(json_value(message), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(self.parameters.encoding) + b"\n"
        if len(data) > MAX_MESSAGE_BYTES:
            raise ValueError("MCP stdio message exceeds the wire limit")
        async with self._write_lock:
            self.process.stdin.write(data)
            await self.process.stdin.drain()

    async def close(self):
        await self._close_jobs()
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=0.5)
            return
        except TimeoutError:
            pass
        # Give the parent its opportunity to reap descendants. Hermes' existing
        # MCP PID/group ledger owns the later descendant sweep and crash cleanup.
        with suppress(ProcessLookupError):
            self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=2)
        except TimeoutError:
            with suppress(ProcessLookupError):
                self.process.kill()
            await self.process.wait()


@asynccontextmanager
async def stdio_client(server, errlog=None):
    # Hermes passes its profile-scoped, filtered environment. Direct users of
    # this helper receive the same minimal ambient keys as the upstream SDK.
    environment = server.env
    if environment is None:
        names = ("PATH", "HOME", "USER", "USERNAME", "USERPROFILE", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
        environment = {name: os.environ[name] for name in names if name in os.environ}
    process = await asyncio.create_subprocess_exec(
        server.command, *server.args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=errlog if errlog is not None else sys.stderr, cwd=server.cwd, env=environment,
        start_new_session=os.name != "nt", limit=MAX_MESSAGE_BYTES + 1,
    )
    transport = _StdioTransport(process, server)
    try:
        yield transport, transport
    finally:
        await transport.close()


class _HTTPTransport(_QueueTransport):
    def __init__(self, url, client, *, legacy_sse=False):
        super().__init__()
        self.url, self.client, self.legacy_sse = url, client, legacy_sse
        self.endpoint = None if legacy_sse else url
        self.session_id = None
        self._request_jobs = {}
        self._get_started = False
        self.ready = asyncio.get_running_loop().create_future()
        self.tool_schemas = {}

    def _headers(self, message=None):
        headers = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": self.protocol_version}
        if self.session_id and self.protocol_version != "2026-07-28":
            headers["MCP-Session-Id"] = self.session_id
        if message and self.protocol_version == "2026-07-28" and "method" in message:
            method, params = message["method"], message.get("params", {})
            headers["Mcp-Method"] = method
            if method in ("tools/call", "resources/read", "prompts/get"):
                headers["Mcp-Name"] = _header_value(params.get("uri" if method == "resources/read" else "name", ""))
            if method == "tools/call":
                self._tool_headers(headers, self.tool_schemas.get(params.get("name"), {}), params.get("arguments", {}))
        return headers

    def _tool_headers(self, headers, schema, arguments):
        from standalone.mcp_headers import parameter_headers
        headers.update({name: _header_value(value) for name, value in parameter_headers(schema, arguments).items()})

    async def send(self, message):
        if self.closed:
            raise ConnectionError("MCP HTTP connection closed")
        task = self._start(self._post(message), request_id=message.get("id") if "method" in message else None)
        if "id" in message and "method" in message:
            identifier = message["id"]
            self._request_jobs[identifier] = task
            task.add_done_callback(lambda _task: self._request_jobs.pop(identifier, None))
        else:
            # Handshake notifications must reach the server before tools/list.
            await task

    async def cancel_request(self, identifier):
        task = self._request_jobs.get(identifier)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _post(self, message):
        content = json.dumps(json_value(message), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(content) > MAX_MESSAGE_BYTES:
            raise ValueError("MCP HTTP message exceeds the wire limit")
        headers = {**self._headers(message), "Content-Type": "application/json"}
        async with self.client.stream("POST", self.endpoint, content=content, headers=headers) as response:
            if not response.is_success:
                await response.aread()
                if response.status_code not in (401, 403):
                    with suppress(ValueError, UnicodeError):
                        error_body = response.json()
                        if isinstance(error_body, dict) and isinstance(error_body.get("error"), dict) and "id" in message:
                            failure = {"jsonrpc": "2.0", "id": message["id"], "error": error_body["error"]}
                            await self.incoming.put(validate_message(failure))
                            return
                response.raise_for_status()
            identifier = response.headers.get("mcp-session-id")
            if identifier:
                if any(ord(char) < 33 or ord(char) > 126 for char in identifier):
                    raise ValueError("MCP server supplied an invalid session identifier")
                self.session_id = identifier
            if response.status_code == 202:
                if "id" in message and "method" in message and not self.legacy_sse:
                    raise ValueError("MCP request received HTTP 202 without a result stream")
                if message.get("method") == "notifications/initialized" and not self._get_started and not self.legacy_sse:
                    self._get_started = True
                    self._start(self._get())
                return
            mime = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if mime == "application/json":
                data = await response.aread()
                if len(data) > MAX_MESSAGE_BYTES:
                    raise ValueError("MCP HTTP response exceeds the wire limit")
                await self.incoming.put(validate_message(json.loads(data)))
            elif mime == "text/event-stream":
                completed, last_id, delay = await self._consume(response, message.get("id"))
                if not completed and "id" in message and "method" in message:
                    if last_id and self.protocol_version != "2026-07-28":
                        await self._resume(message["id"], last_id, delay)
                    else:
                        raise ConnectionError("MCP response stream closed before its result; request was not replayed")
            else:
                raise ValueError(f"Unexpected MCP response content type: {mime}")

    async def _consume(self, response, identifier=None):
        last_id, delay = None, 0.5
        async for event in aiter_sse(response):
            if len(event.data.encode("utf-8")) > MAX_MESSAGE_BYTES:
                raise ValueError("MCP SSE message exceeds the wire limit")
            last_id = event.id or last_id
            if event.retry is not None:
                delay = min(event.retry / 1000, 60)
            if self.legacy_sse and event.event == "endpoint":
                endpoint = urljoin(self.url, event.data)
                original, candidate = urlsplit(self.url), urlsplit(endpoint)
                if (candidate.scheme, candidate.hostname, candidate.port) != (original.scheme, original.hostname, original.port):
                    raise ValueError("MCP SSE endpoint must retain the configured origin")
                self.endpoint = endpoint
                if not self.ready.done():
                    self.ready.set_result(None)
                continue
            if not event.data or event.event not in (None, "", "message"):
                continue
            message = validate_message(json.loads(event.data))
            await self.incoming.put(message)
            if identifier is not None and message.get("id") == identifier and "method" not in message:
                return True, last_id, delay
        return False, last_id, delay

    async def _resume(self, identifier, last_id, delay):
        # Resume a server-assigned stream; never re-POST an outcome-uncertain call.
        for _attempt in range(3):
            await asyncio.sleep(delay)
            async with self.client.stream("GET", self.url, headers={**self._headers(), "Last-Event-ID": last_id}) as response:
                response.raise_for_status()
                complete, next_id, delay = await self._consume(response, identifier)
                if complete:
                    return
                last_id = next_id or last_id
        raise ConnectionError("MCP response stream could not be resumed; request was not replayed")

    async def _get(self):
        try:
            async with self.client.stream("GET", self.url, headers=self._headers()) as response:
                if response.status_code == 405 and not self.legacy_sse:
                    return
                response.raise_for_status()
                if response.headers.get("content-type", "").split(";", 1)[0].strip() != "text/event-stream":
                    raise ValueError("MCP GET endpoint did not return an event stream")
                await self._consume(response)
            if not self.closed:
                raise ConnectionError("MCP notification stream closed")
        except BaseException as error:
            if self.legacy_sse and not self.ready.done():
                self.ready.set_exception(error)
            raise

    async def close(self):
        if self.closed:
            return
        await self._close_jobs()
        if self.session_id and not self.legacy_sse and self.protocol_version != "2026-07-28":
            with suppress(Exception):
                response = await self.client.delete(self.url, headers=self._headers(), timeout=2)
                await response.aclose()


@asynccontextmanager
async def streamable_http_client(url, *, http_client=None, **kwargs):
    import httpx
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(30, read=300), follow_redirects=True, **kwargs)
    transport = _HTTPTransport(url, client)
    try:
        yield transport, transport, lambda: transport.session_id
    finally:
        await transport.close()
        if owns_client:
            await client.aclose()


@asynccontextmanager
async def sse_client(url, *, headers=None, timeout=30, sse_read_timeout=300, auth=None,
                     httpx_client_factory=None):
    import httpx
    settings = {"headers": headers, "timeout": httpx.Timeout(timeout, read=sse_read_timeout), "auth": auth}
    client = (httpx_client_factory(**settings) if httpx_client_factory else
              httpx.AsyncClient(follow_redirects=True, **settings))
    transport = _HTTPTransport(url, client, legacy_sse=True)
    transport._start(transport._get())
    try:
        await asyncio.wait_for(transport.ready, timeout=timeout)
        yield transport, transport
    finally:
        await transport.close()
        await client.aclose()
