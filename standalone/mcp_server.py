"""Application-owned MCP stdio tool server for the embedded browser helper.

This implements real JSON-RPC lifecycle, discovery, strict tool arguments,
concurrent request routing, cancellation, and text/structured tool results.
Modern requests are self-contained; legacy initialize state is separate.
This is an explicit backend import, never installed as an SDK namespace.
https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import math
import sys
import types
import typing


MAX_REQUEST_BYTES = 8 * 1024 * 1024
_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}
_MODERN_VERSION = "2026-07-28"
_PROTOCOL = "io.modelcontextprotocol/"


def _schema(annotation):
    if annotation in (inspect.Parameter.empty, typing.Any):
        return {}
    if annotation is None or annotation is type(None):
        return {"type": "null"}
    origin, arguments = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return {"anyOf": [_schema(argument) for argument in arguments]}
    if origin is typing.Annotated:
        return _schema(arguments[0])
    if origin is typing.Literal:
        return {"enum": list(arguments)}
    if annotation in (str, int, float, bool):
        return {"type": {str: "string", int: "integer", float: "number", bool: "boolean"}[annotation]}
    if annotation is dict or origin is dict:
        return {"type": "object", "additionalProperties": _schema(arguments[1]) if len(arguments) == 2 else True}
    if annotation is list or origin is list:
        return {"type": "array", "items": _schema(arguments[0]) if arguments else {}}
    raise TypeError(f"Unsupported MCP tool annotation: {annotation!r}")


def _valid(value, schema):
    if "anyOf" in schema:
        return any(_valid(value, branch) for branch in schema["anyOf"])
    if "enum" in schema:
        return any(type(value) is type(choice) and value == choice for choice in schema["enum"])
    kind = schema.get("type")
    if kind == "null":
        return value is None
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return type(value) is int
    if kind == "number":
        return type(value) in (int, float) and (type(value) is int or math.isfinite(value))
    if kind == "boolean":
        return type(value) is bool
    if kind == "array":
        return isinstance(value, list) and all(_valid(item, schema.get("items", {})) for item in value)
    if kind == "object":
        if not isinstance(value, dict):
            return False
        properties = schema.get("properties", {})
        if any(name not in value for name in schema.get("required", [])):
            return False
        additional = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in properties:
                if not _valid(item, properties[name]):
                    return False
            elif additional is False:
                return False
            elif isinstance(additional, dict) and not _valid(item, additional):
                return False
    return True


class _RPCError(Exception):
    def __init__(self, code, message, data=None):
        self.code, self.message, self.data = code, message, data


class MCPServer:
    def __init__(self, name, *, version="1.0.0", instructions=None):
        if not isinstance(name, str) or not name:
            raise ValueError("MCP server name must be a nonempty string")
        self.name, self.version, self.instructions = name, version, instructions
        self._tools = {}
        self._initialized = self._ready = False
        self._call_lock = None

    def tool(self, name=None, description=None, *, annotations=None):
        def register(function):
            tool_name = name or function.__name__
            if not isinstance(tool_name, str) or not tool_name or tool_name in self._tools:
                raise ValueError("MCP tool names must be unique nonempty strings")
            signature = inspect.signature(function)
            hints = typing.get_type_hints(function)
            properties, required = {}, []
            for parameter in signature.parameters.values():
                if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    raise TypeError("MCP tools use named parameters")
                field = _schema(hints.get(parameter.name, parameter.annotation))
                if parameter.default is inspect.Parameter.empty:
                    required.append(parameter.name)
                else:
                    json.dumps(parameter.default, allow_nan=False)
                    field = dict(field, default=parameter.default)
                properties[parameter.name] = field
            schema = {"type": "object", "properties": properties, "additionalProperties": False}
            if required:
                schema["required"] = required
            document = {"name": tool_name, "description": description or function.__doc__ or "", "inputSchema": schema}
            if annotations is not None:
                document["annotations"] = dict(annotations)
            self._tools[tool_name] = function, document
            return function
        return register

    async def _execute(self, function, arguments):
        # The browser's synchronous helpers redirect stdout while executing.
        # Serialize tool functions, while allowing ping/list/cancel traffic.
        async with self._call_lock:
            if inspect.iscoroutinefunction(function):
                with contextlib.redirect_stdout(sys.stderr):
                    return await function(**arguments)

            def execute():
                with contextlib.redirect_stdout(sys.stderr):
                    return function(**arguments)
            operation = asyncio.create_task(asyncio.to_thread(execute))
            try:
                result = await asyncio.shield(operation)
            except asyncio.CancelledError:
                # A Python thread cannot be forcefully cancelled. Keep the
                # serialization fence until its helper has restored stdout and
                # finished its browser action, then suppress its late response.
                with contextlib.suppress(Exception):
                    await operation
                raise
            return await result if inspect.isawaitable(result) else result

    def _modern(self, method, parameters):
        metadata = parameters.get("_meta", {})
        if not isinstance(metadata, dict):
            raise _RPCError(-32602, "Request metadata must be an object")
        modern = method == "server/discover" or any(key.startswith(_PROTOCOL) for key in metadata)
        if not modern:
            return False
        version = metadata.get(_PROTOCOL + "protocolVersion")
        capabilities = metadata.get(_PROTOCOL + "clientCapabilities")
        if not isinstance(version, str) or not isinstance(capabilities, dict):
            raise _RPCError(-32602, "Per-request protocol version and client capabilities are required")
        if version != _MODERN_VERSION:
            raise _RPCError(-32022, "Unsupported protocol version", {
                "supported": [_MODERN_VERSION, *sorted(_VERSIONS, reverse=True)], "requested": version})
        identity = metadata.get(_PROTOCOL + "clientInfo")
        if _PROTOCOL + "clientInfo" in metadata and (not isinstance(identity, dict) or not isinstance(identity.get("name"), str)
                                                    or not isinstance(identity.get("version"), str)):
            raise _RPCError(-32602, "Invalid per-request client information")
        return True

    async def _dispatch(self, method, parameters, modern=False):
        if method == "ping":
            return {}
        if method == "server/discover":
            result = {"supportedVersions": [_MODERN_VERSION, *sorted(_VERSIONS, reverse=True)],
                      "capabilities": {"tools": {"listChanged": False}}}
            if self.instructions:
                result["instructions"] = self.instructions
            return result
        if method == "initialize":
            if modern:
                raise _RPCError(-32601, "initialize requires a legacy protocol version")
            if self._initialized:
                raise _RPCError(-32600, "MCP session is already initialized")
            requested = parameters.get("protocolVersion")
            if not isinstance(requested, str) or not isinstance(parameters.get("capabilities", {}), dict):
                raise _RPCError(-32602, "Invalid initialization parameters")
            self._initialized = True
            result = {"protocolVersion": requested if requested in _VERSIONS else "2025-11-25",
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": self.name, "version": self.version}}
            if self.instructions:
                result["instructions"] = self.instructions
            return result
        if not modern and not self._ready:
            raise _RPCError(-32002, "MCP initialization is incomplete")
        if method == "tools/list":
            if parameters.get("cursor") is not None:
                raise _RPCError(-32602, "Invalid tool list cursor")
            return {"tools": [document for _, document in self._tools.values()]}
        if method == "tools/call":
            name = parameters.get("name")
            tool = self._tools.get(name) if isinstance(name, str) else None
            if tool is None:
                raise _RPCError(-32602, "Unknown MCP tool")
            function, document = tool
            arguments = parameters.get("arguments", {})
            if not _valid(arguments, document["inputSchema"]):
                raise _RPCError(-32602, "Tool arguments do not match the input schema")
            try:
                result = await self._execute(function, arguments)
                text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, allow_nan=False)
                response = {"content": [{"type": "text", "text": text}], "isError": False}
                if isinstance(result, dict):
                    response["structuredContent"] = result
                return response
            except Exception as exc:
                return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        raise _RPCError(-32601, "Method not found")

    async def _serve(self, reader, writer):
        self._call_lock = asyncio.Lock()
        pending = {}

        def send(message):
            encoded = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
            if hasattr(writer, "buffer"):
                writer.buffer.write(encoded.encode("utf-8"))
                writer.buffer.flush()
            else:
                writer.write(encoded)
                writer.flush()

        async def answer(identifier, method, parameters):
            try:
                modern = self._modern(method, parameters)
                result = await self._dispatch(method, parameters, modern)
                if modern:
                    result = {**result, "resultType": "complete", "_meta": {
                        _PROTOCOL + "serverInfo": {"name": self.name, "version": self.version}}}
                send({"jsonrpc": "2.0", "id": identifier, "result": result})
            except _RPCError as error:
                document = {"code": error.code, "message": error.message}
                if error.data is not None:
                    document["data"] = error.data
                send({"jsonrpc": "2.0", "id": identifier, "error": document})
            except asyncio.CancelledError:
                pass
            except Exception:
                send({"jsonrpc": "2.0", "id": identifier,
                      "error": {"code": -32603, "message": "Internal server error"}})
            finally:
                pending.pop(identifier, None)

        def invalid_constant(value):
            raise ValueError("Nonfinite JSON number")

        try:
            while True:
                line = await asyncio.to_thread(reader.readline, MAX_REQUEST_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_REQUEST_BYTES:
                    raise ValueError("MCP request exceeds its byte limit")
                try:
                    message = json.loads(line, parse_constant=invalid_constant)
                except (ValueError, UnicodeError, RecursionError):
                    send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                    continue
                if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
                    send({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}})
                    continue
                identifier = message.get("id")
                method, parameters = message["method"], message.get("params", {})
                if not isinstance(parameters, dict):
                    if "id" in message:
                        send({"jsonrpc": "2.0", "id": identifier, "error": {"code": -32602, "message": "Invalid params"}})
                    continue
                if "id" not in message:
                    if method == "notifications/initialized" and self._initialized:
                        self._ready = True
                    elif method == "notifications/cancelled":
                        target = parameters.get("requestId")
                        if type(target) in (str, int) and target in pending:
                            pending[target].cancel()
                    continue
                if type(identifier) not in (str, int) or identifier in pending:
                    send({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid or duplicate request id"}})
                    continue
                pending[identifier] = asyncio.create_task(answer(identifier, method, parameters))
        finally:
            if pending:
                for operation in tuple(pending.values()):
                    operation.cancel()
                await asyncio.gather(*tuple(pending.values()), return_exceptions=True)

    def run(self, transport="stdio"):
        if transport != "stdio":
            raise NotImplementedError("This embedded MCP server exposes only stdio")
        reader = getattr(sys.stdin, "buffer", sys.stdin)
        # Retain the actual protocol writer across helper stdout redirection.
        asyncio.run(self._serve(reader, sys.stdout))
