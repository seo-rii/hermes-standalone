"""MCP client session for Hermes, using real JSON-RPC over owned transports.

Supports handshake-era sessions and the 2026-07-28 per-request protocol. Hermes'
existing callbacks remain responsible for sampling limits, consent and scope.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import inspect
from types import SimpleNamespace

from standalone.mcp_types import (
    ErrorData, LATEST_HANDSHAKE_VERSION, LATEST_PROTOCOL_VERSION, MCPObject, McpError,
    PromptListChangedNotification, RequestFailure, ResourceListChangedNotification, ServerNotification,
    ToolListChangedNotification, mcp_object,
)
from standalone.transport_types import json_value


class ClientSession:
    def __init__(self, read_stream, write_stream, *, read_timeout_seconds=None,
                 sampling_callback=None, sampling_capabilities=None, elicitation_callback=None,
                 list_roots_callback=None, message_handler=None, logging_callback=None, client_info=None):
        self._read, self._write = read_stream, write_stream
        self._timeout = (read_timeout_seconds.total_seconds() if hasattr(read_timeout_seconds, "total_seconds")
                         else read_timeout_seconds)
        self._callbacks = {"sampling/createMessage": sampling_callback, "elicitation/create": elicitation_callback,
                           "roots/list": list_roots_callback}
        self._sampling_capabilities = sampling_capabilities
        self._message_handler, self._logging_callback = message_handler, logging_callback
        self._client_info = client_info or MCPObject(name="hermes-standalone", version="1")
        self._protocol_version = LATEST_HANDSHAKE_VERSION
        self._pending, self._progress, self._callback_tasks = {}, {}, set()
        self._next_id = 0
        self._reader = None
        self._closed = False
        self._closed_event = asyncio.Event()
        self._transport_error = None

    async def __aenter__(self):
        if self._closed:
            raise ConnectionError("MCP session is closed")
        self._reader = asyncio.create_task(self._receive_loop())
        return self

    async def __aexit__(self, *_args):
        self._closed = True
        self._closed_event.set()
        tasks = list(self._callback_tasks)
        if self._reader is not None:
            tasks.append(self._reader)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._fail_pending(ConnectionError("MCP session closed"))

    def _fail_pending(self, error):
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)

    async def _receive_loop(self):
        try:
            while True:
                message = await self._read.receive()
                if isinstance(message, RequestFailure):
                    future = self._pending.get(message.identifier)
                    if future is not None and not future.done():
                        future.set_exception(message.error)
                    continue
                if "method" not in message:
                    future = self._pending.get(message["id"])
                    if future is not None and not future.done():
                        if "error" in message:
                            future.set_exception(McpError(message["error"]))
                        else:
                            if self._protocol_version == LATEST_PROTOCOL_VERSION:
                                result = message["result"]
                                if not isinstance(result, dict) or result.get("resultType", "complete") not in ("complete", "input_required"):
                                    raise ValueError("Unsupported MCP resultType from the server")
                            future.set_result(mcp_object(message["result"]))
                elif "id" in message:
                    if self._protocol_version == LATEST_PROTOCOL_VERSION:
                        raise ValueError("Modern MCP server input requests must use input_required results")
                    task = asyncio.create_task(self._answer(message))
                    self._callback_tasks.add(task)
                    task.add_done_callback(self._callback_tasks.discard)
                else:
                    await self._notification(message)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._closed = True
            self._transport_error = error
            self._closed_event.set()
            self._fail_pending(error)
            if self._message_handler:
                with suppress(Exception):
                    await self._invoke(self._message_handler, error)

    async def wait_closed(self):
        await self._closed_event.wait()
        if self._transport_error is not None:
            raise self._transport_error

    @staticmethod
    async def _invoke(callback, *args):
        value = callback(*args)
        return await value if inspect.isawaitable(value) else value

    async def _input(self, message):
        method = message.get("method")
        if method == "ping":
            return MCPObject()
        callback = self._callbacks.get(method)
        if callback is None:
            return ErrorData(code=-32601, message=f"Client capability is not enabled: {method}")
        context = SimpleNamespace(session=self, request_id=message.get("id"), meta=message.get("params", {}).get("_meta"))
        try:
            return await self._invoke(callback, context, mcp_object(message.get("params", {})))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return ErrorData(code=-32603, message=str(error)[:1000])

    async def _answer(self, message):
        try:
            result = await self._input(message)
            await self._write.send({"jsonrpc": "2.0", "id": message["id"],
                                    "error" if isinstance(result, ErrorData) else "result": json_value(result)})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._fail_pending(error)

    async def _notification(self, message):
        method, params = message["method"], mcp_object(message.get("params", {}))
        if method == "notifications/progress":
            callback = self._progress.get(params.get("progressToken"))
            if callback:
                await self._invoke(callback, params.get("progress"), params.get("total"), params.get("message"))
        elif method == "notifications/message" and self._logging_callback:
            await self._invoke(self._logging_callback, params)
        if self._message_handler:
            cls = {"notifications/tools/list_changed": ToolListChangedNotification,
                   "notifications/prompts/list_changed": PromptListChangedNotification,
                   "notifications/resources/list_changed": ResourceListChangedNotification}.get(method, MCPObject)
            await self._invoke(self._message_handler, ServerNotification(cls(message)))

    def _build_capabilities(self, _version=None):
        result = MCPObject()
        if self._callbacks["sampling/createMessage"]:
            result["sampling"] = json_value(self._sampling_capabilities or {})
        if self._callbacks["elicitation/create"]:
            result["elicitation"] = {"form": {}}
        if self._callbacks["roots/list"]:
            result["roots"] = {}
        return result

    def _set_protocol(self, version):
        self._protocol_version = version
        self._write.protocol_version = version

    def adopt(self, result):
        version = getattr(result, "protocol_version", None) or getattr(result, "protocolVersion", None)
        if not isinstance(version, str):
            raise ValueError("MCP initialization result has no protocol version")
        self._set_protocol(version)

    async def initialize(self):
        self._set_protocol(LATEST_HANDSHAKE_VERSION)
        result = await self._request("initialize", {"protocolVersion": LATEST_HANDSHAKE_VERSION,
            "capabilities": self._build_capabilities(), "clientInfo": self._client_info})
        version = result.get("protocolVersion")
        if version not in ("2024-11-05", "2025-03-26", "2025-06-18", LATEST_HANDSHAKE_VERSION):
            raise ValueError(f"Unsupported protocol version from the server: {version}")
        self.adopt(result)
        await self.send_notification({"method": "notifications/initialized"})
        return result

    async def complete_handshake_at_offered_version(self):
        self._set_protocol(LATEST_HANDSHAKE_VERSION)
        result = await self._request("initialize", {"protocolVersion": LATEST_HANDSHAKE_VERSION,
            "capabilities": self._build_capabilities(), "clientInfo": self._client_info})
        await self.send_notification({"method": "notifications/initialized"})
        return result

    async def discover(self):
        self._set_protocol(LATEST_PROTOCOL_VERSION)
        result = await self._request("server/discover", {})
        versions = result.get("supportedVersions", [])
        if LATEST_PROTOCOL_VERSION not in versions:
            raise ValueError("MCP server does not support the requested protocol version")
        return result

    async def send_notification(self, notification):
        message = {"jsonrpc": "2.0", **json_value(notification)}
        await self._write.send(message)

    async def _request(self, method, params=None, *, progress_callback=None):
        if self._closed or self._reader is None:
            raise ConnectionError("MCP session is not open")
        self._next_id += 1
        identifier = self._next_id
        values = dict(params or {})
        metadata = dict(values.get("_meta") or {})
        if self._protocol_version == LATEST_PROTOCOL_VERSION:
            metadata.update({"io.modelcontextprotocol/protocolVersion": self._protocol_version,
                             "io.modelcontextprotocol/clientInfo": self._client_info,
                             "io.modelcontextprotocol/clientCapabilities": self._build_capabilities()})
        if progress_callback:
            metadata["progressToken"] = identifier
            self._progress[identifier] = progress_callback
        if metadata:
            values["_meta"] = metadata
        future = asyncio.get_running_loop().create_future()
        self._pending[identifier] = future
        try:
            await self._write.send({"jsonrpc": "2.0", "id": identifier, "method": method, "params": values})
            return await asyncio.wait_for(future, timeout=self._timeout)
        except (asyncio.CancelledError, TimeoutError):
            if method != "initialize":
                is_modern_http = self._protocol_version == LATEST_PROTOCOL_VERSION and hasattr(self._write, "cancel_request")
                if not is_modern_http:
                    with suppress(Exception):
                        await asyncio.wait_for(self.send_notification({"method": "notifications/cancelled",
                            "params": {"requestId": identifier, "reason": "Client cancelled request"}}), timeout=2)
                if hasattr(self._write, "cancel_request"):
                    await self._write.cancel_request(identifier)
            raise
        finally:
            self._pending.pop(identifier, None)
            self._progress.pop(identifier, None)

    async def _operation(self, method, params, *, progress_callback=None):
        values = dict(params)
        for _round in range(16):
            result = await self._request(method, values, progress_callback=progress_callback)
            if not isinstance(result, dict) or result.get("resultType") != "input_required":
                return result
            requests = result.get("inputRequests", {})
            if not isinstance(requests, dict) or len(requests) > 64:
                raise ValueError("Invalid or excessive MCP input requests")
            responses = {}
            for name, request in requests.items():
                if not isinstance(request, dict):
                    raise ValueError("Invalid MCP input request")
                response = await self._input(request)
                if isinstance(response, ErrorData):
                    raise McpError(response)
                responses[name] = json_value(response)
            values = {**params, "inputResponses": responses}
            if "requestState" in result:
                if not isinstance(result["requestState"], str):
                    raise ValueError("MCP requestState must be an opaque string")
                values["requestState"] = result["requestState"]
        raise McpError(ErrorData(code=-32603, message="MCP input-request round limit reached"))

    async def list_tools(self, cursor=None):
        result = await self._request("tools/list", {"cursor": cursor} if cursor is not None else {})
        if hasattr(self._write, "tool_schemas"):
            from standalone.mcp_headers import header_fields
            accepted = []
            for tool in result.get("tools", []):
                if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                    if self._protocol_version == LATEST_PROTOCOL_VERSION:
                        try:
                            header_fields(tool.get("inputSchema", {}))
                        except ValueError:
                            import logging
                            logging.getLogger(__name__).warning("Rejected MCP tool %r: invalid header annotation", tool["name"])
                            continue
                    self._write.tool_schemas[tool["name"]] = tool.get("inputSchema", {})
                    accepted.append(tool)
            result["tools"] = accepted
        return result

    async def call_tool(self, name, arguments=None, *, progress_callback=None, **kwargs):
        return await self._operation("tools/call", {"name": name, "arguments": arguments or {}, **kwargs},
                                     progress_callback=progress_callback)

    async def list_resources(self, cursor=None):
        return await self._request("resources/list", {"cursor": cursor} if cursor is not None else {})

    async def list_resource_templates(self, cursor=None):
        return await self._request("resources/templates/list", {"cursor": cursor} if cursor is not None else {})

    async def read_resource(self, uri):
        return await self._operation("resources/read", {"uri": str(uri)})

    async def list_prompts(self, cursor=None):
        return await self._request("prompts/list", {"cursor": cursor} if cursor is not None else {})

    async def get_prompt(self, name, arguments=None):
        return await self._operation("prompts/get", {"name": name, "arguments": arguments or {}})

    async def send_ping(self):
        return await self._request("ping")

    async def set_logging_level(self, level):
        return await self._request("logging/setLevel", {"level": level})
