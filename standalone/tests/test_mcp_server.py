"""Actual MCP stdio server wire, validation, stdout isolation and cancellation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[2]
_SERVER = r'''
import asyncio,sys,time
sys.path.insert(0,PROJECT_PATH)
from standalone.mcp_server import MCPServer
server=MCPServer('fixture-tools',version='1.2')
busy=False
@server.tool(description='Strict scalar and Unicode list arguments')
def echo(number:int,enabled:bool=True,labels:list[str]|None=None):
    print('helper diagnostic',flush=True)
    return {'number':number,'enabled':enabled,'labels':labels,'busy':busy}
@server.tool()
def blocking(delay:float):
    global busy
    busy=True
    print('blocking-start',flush=True)
    time.sleep(delay)
    busy=False
    return 'late blocking result'
@server.tool()
async def asynchronous():
    await asyncio.Event().wait()
def isolate(event,args):
    if (event.startswith('socket.') and event != 'socket.__new__') or event in ('subprocess.Popen','os.system'):
        raise RuntimeError('MCP fixture cannot access the network or external programs')
sys.addaudithook(isolate)
server.run()
'''


class MCPServerTests(unittest.TestCase):
    async def start(self, scratch):
        environment = {"HOME": str(scratch), "USERPROFILE": str(scratch)}
        for key in ("SYSTEMROOT", "WINDIR"):
            if key in os.environ:
                environment[key] = os.environ[key]
        return await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-S", "-B", "-c", _SERVER.replace("PROJECT_PATH", repr(str(PROJECT))),
            cwd=scratch, env=environment, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )

    async def send(self, process, message):
        process.stdin.write((json.dumps(message) + "\n").encode())
        await process.stdin.drain()

    async def read(self, process):
        line = await asyncio.wait_for(process.stdout.readline(), 5)
        self.assertTrue(line, await process.stderr.read() if process.returncode is not None else "MCP server exited")
        return json.loads(line)

    async def initialize(self, process, requested="2025-11-25"):
        await self.send(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": requested, "capabilities": {}, "clientInfo": {"name": "fixture", "version": "1"},
        }})
        response = await self.read(process)
        self.assertEqual(response["result"]["protocolVersion"], "2025-11-25")
        self.assertEqual(response["result"]["serverInfo"], {"name": "fixture-tools", "version": "1.2"})
        await self.send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})

    def test_lifecycle_discovery_schema_and_real_tool_results(self):
        async def scenario(scratch):
            process = await self.start(scratch)
            try:
                await self.send(process, {"jsonrpc": "2.0", "id": 0, "method": "tools/list"})
                self.assertEqual((await self.read(process))["error"]["code"], -32002)
                # The 2026 stateless/discover revision must negotiate down to
                # the handshake revision this stdio server actually implements.
                await self.initialize(process, "2026-07-28")
                await self.send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
                tools = (await self.read(process))["result"]["tools"]
                schema = next(tool["inputSchema"] for tool in tools if tool["name"] == "echo")
                self.assertEqual(schema["required"], ["number"])
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(schema["properties"]["enabled"], {"type": "boolean", "default": True})
                arguments = {"number": 3, "labels": ["한글", "Unicode"]}
                await self.send(process, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                          "params": {"name": "echo", "arguments": arguments}})
                result = (await self.read(process))["result"]
                self.assertFalse(result["isError"])
                self.assertEqual(result["structuredContent"], dict(arguments, enabled=True, busy=False))
                self.assertEqual(json.loads(result["content"][0]["text"]), result["structuredContent"])
                for index, bad in enumerate(({"number": True}, {"number": 1, "enabled": 1},
                                              {"number": 1, "unknown": 2}, {"labels": []}), 4):
                    await self.send(process, {"jsonrpc": "2.0", "id": index, "method": "tools/call",
                                              "params": {"name": "echo", "arguments": bad}})
                    self.assertEqual((await self.read(process))["error"]["code"], -32602)
                process.stdin.write(b"not-json\n");await process.stdin.drain()
                self.assertEqual((await self.read(process))["error"]["code"], -32700)
                process.stdin.close()
                await asyncio.wait_for(process.wait(), 5)
                self.assertEqual(process.returncode, 0, (await process.stderr.read()).decode())
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        with tempfile.TemporaryDirectory(prefix="hermes-mcp-server-") as directory:
            asyncio.run(scenario(Path(directory)))

    def test_cancellation_keeps_the_sync_helper_fence_and_suppresses_late_reply(self):
        async def scenario(scratch):
            process = await self.start(scratch)
            try:
                await self.initialize(process)
                await self.send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                          "params": {"name": "blocking", "arguments": {"delay": 0.15}}})
                self.assertEqual((await asyncio.wait_for(process.stderr.readline(), 5)).strip(), b"blocking-start")
                await self.send(process, {"jsonrpc": "2.0", "method": "notifications/cancelled",
                                          "params": {"requestId": 2, "reason": "fixture cancellation"}})
                await self.send(process, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                          "params": {"name": "echo", "arguments": {"number": 5}}})
                reply = await self.read(process)
                self.assertEqual(reply["id"], 3)
                self.assertFalse(reply["result"]["structuredContent"]["busy"])
                await self.send(process, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                          "params": {"name": "asynchronous", "arguments": {}}})
                await self.send(process, {"jsonrpc": "2.0", "method": "notifications/cancelled",
                                          "params": {"requestId": 4}})
                await self.send(process, {"jsonrpc": "2.0", "id": 5, "method": "ping"})
                self.assertEqual((await self.read(process)), {"jsonrpc": "2.0", "id": 5, "result": {}})
                process.stdin.close()
                self.assertEqual(await asyncio.wait_for(process.stdout.read(), 5), b"")
                await asyncio.wait_for(process.wait(), 5)
                self.assertEqual(process.returncode, 0, (await process.stderr.read()).decode())
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        with tempfile.TemporaryDirectory(prefix="hermes-mcp-server-cancel-") as directory:
            asyncio.run(scenario(Path(directory)))

    def test_modern_requests_are_self_contained_and_legacy_can_share_the_process(self):
        async def scenario(scratch):
            process = await self.start(scratch)
            metadata = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
                        "io.modelcontextprotocol/clientCapabilities": {}}
            try:
                await self.send(process, {"jsonrpc": "2.0", "id": "discover", "method": "server/discover",
                                          "params": {"_meta": metadata}})
                discovered = (await self.read(process))["result"]
                self.assertEqual(discovered["resultType"], "complete")
                self.assertIn("2026-07-28", discovered["supportedVersions"])
                self.assertEqual(discovered["_meta"]["io.modelcontextprotocol/serverInfo"],
                                 {"name": "fixture-tools", "version": "1.2"})
                await self.send(process, {"jsonrpc": "2.0", "id": "list", "method": "tools/list",
                                          "params": {"_meta": metadata}})
                listing = (await self.read(process))["result"]
                self.assertEqual(listing["resultType"], "complete")
                self.assertEqual(len(listing["tools"]), 3)
                for index, fields in enumerate(({}, {"io.modelcontextprotocol/protocolVersion": "2026-07-28"},
                                                {**metadata, "io.modelcontextprotocol/clientCapabilities": []}),
                                              20):
                    await self.send(process, {"jsonrpc": "2.0", "id": index, "method": "server/discover",
                                              "params": {"_meta": fields}})
                    self.assertEqual((await self.read(process))["error"]["code"], -32602)
                await self.send(process, {"jsonrpc": "2.0", "id": "future", "method": "tools/call", "params": {
                    "name": "echo", "arguments": {"number": 7}, "_meta": {
                        **metadata, "io.modelcontextprotocol/protocolVersion": "2999-01-01"}}})
                error = (await self.read(process))["error"]
                self.assertEqual(error["code"], -32022)
                self.assertEqual(error["data"]["requested"], "2999-01-01")
                self.assertIn("2026-07-28", error["data"]["supported"])
                # Modern discovery does not select or initialize legacy state.
                await self.initialize(process)
                await self.send(process, {"jsonrpc": "2.0", "id": "legacy", "method": "tools/list"})
                legacy = (await self.read(process))["result"]
                self.assertNotIn("resultType", legacy)
                self.assertEqual(legacy["tools"], listing["tools"])
                await self.send(process, {"jsonrpc": "2.0", "id": "other-client", "method": "tools/call", "params": {
                    "name": "echo", "arguments": {"number": 9}, "_meta": {
                        **metadata, "io.modelcontextprotocol/clientInfo": {"name": "other", "version": "2"}}}})
                reply = (await self.read(process))["result"]
                self.assertEqual(reply["resultType"], "complete")
                self.assertEqual(reply["structuredContent"]["number"], 9)
                process.stdin.close()
                await asyncio.wait_for(process.wait(), 5)
                self.assertEqual(process.returncode, 0, (await process.stderr.read()).decode())
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        with tempfile.TemporaryDirectory(prefix="hermes-mcp-modern-server-") as directory:
            asyncio.run(scenario(Path(directory)))

    def test_actual_qualified_client_and_server_interoperate_in_both_protocol_eras(self):
        from standalone.mcp_client import ClientSession
        from standalone.mcp_transport import stdio_client
        from standalone.mcp_types import McpError, StdioServerParameters

        async def scenario(scratch, modern):
            environment = {"HOME": str(scratch), "USERPROFILE": str(scratch)}
            for key in ("SYSTEMROOT", "WINDIR"):
                if key in os.environ:
                    environment[key] = os.environ[key]
            parameters = StdioServerParameters(sys.executable,
                ["-I", "-S", "-B", "-c", _SERVER.replace("PROJECT_PATH", repr(str(PROJECT)))],
                env=environment, cwd=str(scratch))
            with (scratch / "server.log").open("wb") as errlog:
                async with stdio_client(parameters, errlog) as (read, write):
                    async with ClientSession(read, write, read_timeout_seconds=5) as session:
                        if modern:
                            discovered = await session.discover()
                            self.assertIn("2026-07-28", discovered.supported_versions)
                            self.assertEqual(discovered.resultType, "complete")
                            self.assertEqual(session._protocol_version, "2026-07-28")
                        else:
                            initialized = await session.initialize()
                            self.assertEqual(initialized.protocolVersion, "2025-11-25")
                        tools = await session.list_tools()
                        self.assertEqual({tool.name for tool in tools.tools}, {"echo", "blocking", "asynchronous"})
                        reply = await session.call_tool("echo", {"number": 17, "labels": ["한글"]})
                        self.assertEqual(reply.structuredContent.labels, ["한글"])
                        self.assertFalse(reply.isError)
                        if modern:
                            self.assertEqual(reply.resultType, "complete")
                            self.assertEqual(write.protocol_version, "2026-07-28")
                        with self.assertRaises(McpError) as failure:
                            await session.call_tool("echo", {"number": True})
                        self.assertEqual(failure.exception.code, -32602)
                        await session.send_ping()
        for modern in (False, True):
            with self.subTest(modern=modern), tempfile.TemporaryDirectory(prefix="hermes-mcp-paired-") as directory:
                asyncio.run(scenario(Path(directory), modern))

    def test_stdin_eof_cancels_an_async_helper_and_exits(self):
        async def scenario(scratch):
            process = await self.start(scratch)
            try:
                await self.initialize(process)
                await self.send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                          "params": {"name": "asynchronous"}})
                await self.send(process, {"jsonrpc": "2.0", "id": 3, "method": "ping"})
                self.assertEqual((await self.read(process))["id"], 3)
                process.stdin.close()
                await asyncio.wait_for(process.wait(), 5)
                self.assertEqual(process.returncode, 0, (await process.stderr.read()).decode())
                self.assertEqual(await process.stdout.read(), b"")
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        with tempfile.TemporaryDirectory(prefix="hermes-mcp-eof-") as directory:
            asyncio.run(scenario(Path(directory)))

    def test_current_client_refuses_an_unnegotiated_result_variant(self):
        from standalone.mcp_client import ClientSession
        from standalone.mcp_transport import stdio_client
        from standalone.mcp_types import StdioServerParameters

        malformed_server = r'''
import json,sys
for line in sys.stdin:
    message=json.loads(line)
    if 'id' in message:
        print(json.dumps({'jsonrpc':'2.0','id':message['id'],'result':{'resultType':'unnegotiated-extension'}}),flush=True)
'''
        async def scenario(scratch):
            parameters = StdioServerParameters(sys.executable, ["-I", "-S", "-B", "-c", malformed_server],
                                               env={"HOME": str(scratch)}, cwd=str(scratch))
            with (scratch / "server.log").open("wb") as errlog:
                async with stdio_client(parameters, errlog) as (read, write):
                    async with ClientSession(read, write, read_timeout_seconds=5) as session:
                        with self.assertRaisesRegex(ValueError, "resultType"):
                            await session.discover()
        with tempfile.TemporaryDirectory(prefix="hermes-mcp-invalid-result-") as directory:
            asyncio.run(scenario(Path(directory)))


if __name__ == "__main__":
    unittest.main()
