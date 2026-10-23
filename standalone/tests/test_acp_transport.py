"""The retained ACP transport runs both request directions over real asyncio sockets."""

import asyncio
import unittest

from acp_adapter import _protocol as acp
from acp_adapter._protocol import schema
from acp_adapter._protocol.agent.connection import AgentSideConnection


class _Agent:
    def __init__(self):
        self.waiting = asyncio.Event()
        self.cancelled = asyncio.Event()

    def on_connect(self, connection):
        self.connection = connection

    async def initialize(self, protocol_version, **kwargs):
        return schema.InitializeResponse(protocol_version=protocol_version,
                                         agent_capabilities=schema.AgentCapabilities())

    async def new_session(self, cwd, mcp_servers, **kwargs):
        return schema.NewSessionResponse(session_id="fixture-session")

    async def prompt(self, prompt, session_id, **kwargs):
        if prompt[0].text == "wait for cancel":
            self.waiting.set()
            await self.cancelled.wait()
            return schema.PromptResponse(stop_reason="cancelled")
        decision = await self.connection.request_permission(
            session_id=session_id,
            tool_call=acp.update_tool_call("fixture-tool", title="Read fixture", kind="read", status="pending"),
            options=[schema.PermissionOption(option_id="allow", name="Allow once", kind="allow_once")],
        )
        if not isinstance(decision.outcome, schema.AllowedOutcome) or decision.outcome.option_id != "allow":
            raise AssertionError("Permission outcome lost its declared variant")
        await self.connection.session_update(session_id=session_id, update=acp.update_agent_message_text("allowed"))
        return schema.PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id, **kwargs):
        self.cancelled.set()

    async def ext_method(self, name, payload):
        return {"name": name, "payload": payload}


class _Client:
    def __init__(self):
        self.notifications = []
        self.updated = asyncio.Event()
        self.permissions = []

    async def request_permission(self, options, session_id, tool_call, **kwargs):
        self.permissions.append((session_id, tool_call.tool_call_id, options[0].option_id))
        return schema.RequestPermissionResponse(outcome=schema.AllowedOutcome(option_id="allow", outcome="selected"))

    async def session_update(self, session_id, update, **kwargs):
        self.notifications.append((session_id, update))
        self.updated.set()


class ACPTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_bidirectional_permission_notifications_and_concurrent_cancel(self):
        agent, client = _Agent(), _Client()
        accepted = asyncio.get_running_loop().create_future()
        server_writers = []

        async def serve(reader, writer):
            server_writers.append(writer)
            connection = AgentSideConnection(agent, writer, reader)
            accepted.set_result(connection)

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
        connection = acp.connect_to_agent(client, writer, reader)
        agent_connection = await accepted
        try:
            initialized = await asyncio.wait_for(connection.initialize(protocol_version=1), timeout=5)
            self.assertEqual(initialized.protocol_version, 1)
            session = await asyncio.wait_for(connection.new_session(cwd="/fixture", mcp_servers=[]), timeout=5)
            self.assertEqual(session.session_id, "fixture-session")
            response = await asyncio.wait_for(connection.prompt(
                prompt=[acp.text_block("request permission")], session_id=session.session_id,
            ), timeout=5)
            self.assertEqual(response.stop_reason, "end_turn")
            await asyncio.wait_for(client.updated.wait(), timeout=5)
            self.assertEqual(client.permissions, [("fixture-session", "fixture-tool", "allow")])
            self.assertEqual(client.notifications[0][1].content.text, "allowed")
            extension = await asyncio.wait_for(connection.ext_method("fixture", {"opaque": {"null": None}}), timeout=5)
            self.assertEqual(extension, {"name": "fixture", "payload": {"opaque": {"null": None}}})
            pending = asyncio.create_task(connection.prompt(
                prompt=[acp.text_block("wait for cancel")], session_id=session.session_id,
            ))
            await asyncio.wait_for(agent.waiting.wait(), timeout=5)
            await connection.cancel(session_id=session.session_id)
            stopped = await asyncio.wait_for(pending, timeout=5)
            self.assertEqual(stopped.stop_reason, "cancelled")
        finally:
            await connection.close()
            await agent_connection.close()
            writer.close()
            await writer.wait_closed()
            for server_writer in server_writers:
                server_writer.close()
                await server_writer.wait_closed()
            server.close()
            await server.wait_closed()


if __name__ == "__main__":
    unittest.main()
