"""Actual stdio, Streamable HTTP and legacy SSE MCP protocol fixtures."""

from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import unittest

from standalone.tests import test_transport as _wire_harness


class MCPDependencyTests(_wire_harness.TransportDependencyTests):
    CASES = "standalone.tests.test_mcp_transport.MCPWireTests"


def _result(message):
    method, params = message.get("method"), message.get("params", {})
    if method == "initialize":
        return {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}, "prompts": {}, "resources": {}},
                "serverInfo": {"name": "local-mcp-fixture", "version": "1"}}
    if method == "server/discover":
        return {"resultType": "complete", "supportedVersions": ["2026-07-28"], "capabilities": {"tools": {}}}
    if method == "tools/list":
        if params.get("cursor"):
            return {"tools": []}
        return {"tools": [{"name": "echo", "description": "Echo fixture", "annotations": {"readOnlyHint": True},
                           "inputSchema": {"type": "object", "properties": {"text": {"type": "string", "x-mcp-header": "Text"}}}},
                          {"name": "write", "description": "Write fixture", "annotations": {"readOnlyHint": False},
                           "inputSchema": {"type": "object", "properties": {}}}],
                **({"nextCursor": "second"} if "cursor" not in params else {})}
    if method == "tools/call":
        if params["name"] == "need-input" and "inputResponses" not in params:
            return {"resultType": "input_required", "requestState": "opaque-fixture-state",
                    "inputRequests": {"consent": {"method": "elicitation/create", "params": {
                        "mode": "form", "message": "fixture permission", "requestedSchema": {"type": "object"}}}}}
        if params["name"] == "need-input":
            return {"resultType": "complete", "content": [{"type": "text", "text": params["inputResponses"]["consent"]["action"]}],
                    "isError": False}
        return {"content": [{"type": "text", "text": params.get("arguments", {}).get("text", "fixture")}],
                "structuredContent": {"arguments": params.get("arguments", {})}, "isError": False}
    if method == "resources/list":
        return {"resources": [{"uri": "fixture://read", "name": "fixture"}]}
    if method == "resources/read":
        return {"contents": [{"uri": params["uri"], "mimeType": "text/plain", "text": "resource fixture"}]}
    if method == "prompts/list":
        return {"prompts": [{"name": "fixture"}]}
    if method == "prompts/get":
        return {"messages": [{"role": "user", "content": {"type": "text", "text": "prompt fixture"}}]}
    if method == "ping":
        return {}
    return None


_STDIO = r'''
import json, os, pathlib, sys
sys.path.insert(0, sys.argv[1])
from standalone.tests.test_mcp_transport import _result
def send(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)
for line in sys.stdin:
    message = json.loads(line)
    with open(os.environ["MCP_FIXTURE_LOG"], "a", encoding="utf-8") as log:
        log.write(json.dumps(message) + "\n")
    if "id" not in message:
        continue
    if message.get("method") == "tools/call" and message["params"]["name"] == "stall":
        continue
    if message.get("method") == "tools/call" and message["params"]["name"] == "environment":
        result = {"content":[{"type":"text","text": json.dumps(dict(os.environ))}],"isError":False}
    else:
        result = _result(message)
    if message.get("method") == "tools/call":
        token = message["params"].get("_meta", {}).get("progressToken")
        if token is not None:
            send({"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":token,"progress":1,"total":1,"message":"done"}})
        send({"jsonrpc":"2.0","method":"notifications/tools/list_changed"})
    reply = {"jsonrpc":"2.0","id":message["id"]}
    reply.update({"result":result} if result is not None else {"error":{"code":-32601,"message":"Method not found"}})
    send(reply)
'''


class _HTTPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def _bytes(self, data=b"", status=200, mime="application/json", headers=None):
        self.send_response(status)
        self.send_header("content-type", mime)
        self.send_header("content-length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        notification_stream = self.path == "/mcp" and getattr(self.server, "dynamic_notifications", False)
        if self.path not in ("/sse", "/bad-sse") and not notification_stream:
            self._bytes(status=405)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("connection", "close")
        self.end_headers()
        endpoint = "/messages" if self.path == "/sse" else "http://outside.invalid/messages"
        pending = self.server.events
        if notification_stream:
            pending = queue.Queue()
            self.server.notification_queues.append(pending)
        try:
            if not notification_stream:
                self.wfile.write(f"event: endpoint\ndata: {endpoint}\n\n".encode())
                self.wfile.flush()
            while not self.server.stopping.is_set():
                try:
                    message = pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                self.wfile.write(("event: message\ndata: " + json.dumps(message) + "\n\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def do_DELETE(self):
        self.server.deletes.append(dict(self.headers))
        self._bytes(status=200)

    def do_POST(self):
        message = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.server.requests.append({"message": message, "headers": {key.lower(): value for key, value in self.headers.items()}})
        if "id" not in message:
            self._bytes(status=202)
            return
        if self.path == "/modern-only" and message.get("method") == "initialize":
            self._bytes(json.dumps({"jsonrpc": "2.0", "id": message["id"], "error": {
                "code": -32022, "message": "Unsupported protocol version"}}).encode(), status=400)
            return
        result = _result(message)
        if getattr(self.server, "dynamic_notifications", False):
            if message["method"] == "tools/call" and message["params"]["name"] == "echo":
                self.server.dynamic_tool = True
                for pending in self.server.notification_queues:
                    pending.put({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
            elif message["method"] == "tools/list" and getattr(self.server, "dynamic_tool", False) and not message.get("params", {}).get("cursor"):
                result["tools"].append({"name": "dynamic", "description": "Newly announced tool",
                    "annotations": {"readOnlyHint": True}, "inputSchema": {"type": "object", "properties": {}}})
        response = {"jsonrpc": "2.0", "id": message["id"]}
        response.update({"result": result} if result is not None else {"error": {"code": -32601, "message": "unknown"}})
        if self.path == "/messages":
            self.server.events.put(response)
            self._bytes(status=202)
        elif self.path == "/stream":
            self._bytes(("event: message\ndata: " + json.dumps(response) + "\n\n").encode(), mime="text/event-stream")
        else:
            headers = {"mcp-session-id": "fixture-session"} if message.get("method") == "initialize" else None
            self._bytes(json.dumps(response).encode(), headers=headers)


@unittest.skipUnless(importlib.util.find_spec("httpx"), "MCP wire cases run in the isolated dependency subprocess")
class MCPWireTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mcp-wire-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def server(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _HTTPHandler)
        server.daemon_threads = True
        server.requests, server.deletes = [], []
        server.events, server.stopping = queue.Queue(), threading.Event()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def close():
            server.stopping.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        self.addCleanup(close)
        return server, f"http://127.0.0.1:{server.server_port}"

    def test_stdio_real_process_progress_pagination_resources_and_cancellation(self):
        from standalone.mcp_client import ClientSession
        from standalone.mcp_transport import stdio_client
        from standalone.mcp_types import StdioServerParameters, ToolListChangedNotification

        script, log = self.directory / "server.py", self.directory / "calls.jsonl"
        script.write_text(_STDIO, encoding="utf-8")
        parameters = StdioServerParameters(command=sys.executable,
            args=["-I", "-S", "-B", str(script), str(_wire_harness.PROJECT)],
            env={"MCP_FIXTURE_LOG": str(log), "MCP_FIXTURE_VALUE": "explicit-profile"}, cwd=str(self.directory))
        progress, notifications = [], []
        async def on_progress(*args):
            progress.append(args)
        async def on_message(message):
            notifications.append(message)
        async def run():
            async with stdio_client(parameters) as (read, write):
                process = read.process
                async with ClientSession(read, write, message_handler=on_message) as session:
                    self.assertEqual((await session.initialize()).protocol_version, "2025-11-25")
                    self.assertEqual((await session.list_tools()).next_cursor, "second")
                    self.assertNotIn("nextCursor", await session.list_tools(cursor="second"))
                    result = await session.call_tool("echo", {"text": "안녕"}, progress_callback=on_progress)
                    self.assertEqual(result.content[0].text, "안녕")
                    self.assertFalse(result.is_error)
                    self.assertEqual(progress, [(1, 1, "done")])
                    self.assertTrue(any(isinstance(item.root, ToolListChangedNotification) for item in notifications))
                    self.assertEqual((await session.read_resource("fixture://read")).contents[0].text, "resource fixture")
                    self.assertEqual((await session.get_prompt("fixture")).messages[0].content.text, "prompt fixture")
                    environment = json.loads((await session.call_tool("environment")).content[0].text)
                    self.assertEqual(environment["MCP_FIXTURE_VALUE"], "explicit-profile")
                    self.assertNotIn("HOME", environment)
                    task = asyncio.create_task(session.call_tool("stall"))
                    await asyncio.sleep(0.03)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    await session.send_ping()
            self.assertIsNotNone(process.returncode)
        asyncio.run(run())
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(calls[1]["method"], "notifications/initialized")
        self.assertTrue(any(item.get("method") == "notifications/cancelled" for item in calls))

    def test_streamable_http_session_headers_and_sse_response(self):
        from standalone.mcp_client import ClientSession
        from standalone.mcp_transport import streamable_http_client
        server, base = self.server()
        async def run(path):
            async with streamable_http_client(base + path) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    self.assertEqual((await session.call_tool("echo", {"text": "wire"})).content[0].text, "wire")
        asyncio.run(run("/mcp"))
        self.assertTrue(any(item["headers"].get("mcp-session-id") == "fixture-session" for item in server.requests[1:]))
        self.assertEqual(server.deletes[0].get("MCP-Session-Id"), "fixture-session")
        asyncio.run(run("/stream"))

    def test_modern_discover_headers_and_input_required_respects_decline(self):
        from standalone.mcp_client import ClientSession
        from standalone.mcp_transport import streamable_http_client
        from standalone.mcp_types import ElicitResult
        server, base = self.server()
        prompts = []
        async def decline(context, params):
            prompts.append(params.message)
            return ElicitResult(action="decline")
        async def run():
            async with streamable_http_client(base + "/mcp") as streams:
                async with ClientSession(streams[0], streams[1], elicitation_callback=decline) as session:
                    self.assertEqual((await session.discover()).supported_versions, ["2026-07-28"])
                    await session.list_tools()
                    await session.call_tool("echo", {"text": "한글"})
                    response = await session.call_tool("need-input", {})
                    self.assertEqual(response.content[0].text, "decline")
        asyncio.run(run())
        self.assertEqual(prompts, ["fixture permission"])
        self.assertEqual(server.deletes, [])
        for item in server.requests:
            meta = item["message"]["params"]["_meta"]
            self.assertEqual(meta["io.modelcontextprotocol/protocolVersion"], "2026-07-28")
            self.assertEqual(item["headers"]["mcp-method"], item["message"]["method"])
        echo = next(item for item in server.requests if item["message"]["params"].get("name") == "echo")
        self.assertTrue(echo["headers"]["mcp-param-text"].startswith("=?base64?"))
        last = server.requests[-1]["message"]
        self.assertEqual(last["params"]["requestState"], "opaque-fixture-state")
        self.assertEqual(last["params"]["inputResponses"], {"consent": {"action": "decline"}})

    def test_legacy_sse_and_foreign_endpoint_rejection(self):
        from standalone.mcp_client import ClientSession
        from standalone.mcp_transport import sse_client
        _, base = self.server()
        async def run():
            async with sse_client(base + "/sse", timeout=3) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    self.assertEqual((await session.call_tool("echo", {"text": "SSE"})).content[0].text, "SSE")
            with self.assertRaisesRegex(ValueError, "configured origin"):
                async with sse_client(base + "/bad-sse", timeout=3):
                    self.fail("A foreign advertised endpoint must never be accepted")
        asyncio.run(run())

    def test_http_protocol_rejection_does_not_poison_discover_fallback(self):
        from standalone.mcp_client import ClientSession
        from standalone.mcp_transport import streamable_http_client
        from standalone.mcp_types import McpError
        server, base = self.server()
        async def run():
            async with streamable_http_client(base + "/modern-only") as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    with self.assertRaises(McpError) as caught:
                        await session.initialize()
                    self.assertEqual(caught.exception.error.code, -32022)
                    await session.discover()
                    self.assertEqual((await session.call_tool("echo", {"text": "fallback"})).content[0].text, "fallback")
        asyncio.run(run())
        self.assertEqual([entry["message"]["method"] for entry in server.requests],
                         ["initialize", "server/discover", "tools/call"])

    def test_header_annotations_validate_paths_types_collisions_and_injection(self):
        from standalone.mcp_headers import header_fields, parameter_headers
        from standalone.mcp_transport import _header_value
        good = {"properties": {"where": {"type": "object", "properties": {
            "region": {"type": "string", "x-mcp-header": "Region_Value"},
            "count": {"type": "integer", "x-mcp-header": "Count"}}}}}
        self.assertEqual(parameter_headers(good, {"where": {"region": "east", "count": 2}}),
                         {"Mcp-Param-Region_Value": "east", "Mcp-Param-Count": "2"})
        self.assertEqual(parameter_headers(good, {}), {})
        for schema in (
            {"properties": {"bad": {"type": "number", "x-mcp-header": "Float"}}},
            {"items": {"type": "string", "x-mcp-header": "Array"}},
            {"properties": {"bad": {"type": "string", "x-mcp-header": "Bad\r\nHeader"}}},
            {"properties": {"a": {"type": "string", "x-mcp-header": "Same"},
                            "b": {"type": "string", "x-mcp-header": "same"}}},
        ):
            with self.assertRaises(ValueError):
                header_fields(schema)
        with self.assertRaises(ValueError):
            parameter_headers(good, {"where": {"count": 2**60}})
        for value in ("안녕", "line\nvalue", " padded ", "=?base64?literal?="):
            self.assertTrue(_header_value(value).startswith("=?base64?"))


if __name__ == "__main__":
    unittest.main()
