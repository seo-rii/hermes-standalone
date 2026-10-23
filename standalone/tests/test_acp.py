"""Pinned ACP records and the original adapter over actual stdio and gateway I/O."""

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from typing import Annotated
import unittest

from acp_adapter._protocol import schema
from standalone.web_schema import ConfigDict, Field, RequestModel, RootModel, ValidationError


class Aliased(RequestModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    session_id: Annotated[str, Field(alias="sessionId", description="Session identity")]
    note: str | None = Field(None)


class RootRecord(RootModel[Aliased]):
    pass


class ACPRecordTests(unittest.TestCase):
    def test_annotated_aliases_defaults_extras_and_generic_unwrapped_root(self):
        value = RootRecord.model_validate({"sessionId": "one"})
        self.assertEqual(value.root.session_id, "one")
        self.assertEqual(value.model_dump(by_alias=True), {"sessionId": "one", "note": None})
        self.assertEqual(Aliased.model_fields["session_id"].description, "Session identity")
        self.assertEqual(Aliased(session_id="two").session_id, "two")
        for invalid in ({"sessionId": 3}, {"sessionId": "one", "other": True}, {}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                RootRecord.model_validate(invalid)

    def test_all_records_resolve_and_wire_tags_preserve_typed_variants(self):
        records = [value for value in vars(schema).values()
                   if isinstance(value, type) and issubclass(value, RequestModel)]
        self.assertEqual(len(records), 145)
        for record in records:
            record._fields()
        update = schema.SessionNotification.model_validate({"sessionId": "session", "update": {
            "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "hello"},
            "_meta": {"opaque": {"null": None}},
        }})
        self.assertIsInstance(update.update, schema.AgentMessageChunk)
        self.assertEqual(update.model_dump(by_alias=True, exclude_none=True)["update"]["_meta"],
                         {"opaque": {"null": None}})
        response = schema.InitializeResponse.model_validate({"protocolVersion": 1, "authMethods": [{
            "id": "setup", "name": "Setup", "type": "terminal", "args": ["--setup"],
        }]})
        self.assertIsInstance(response.auth_methods[0], schema.TerminalAuthMethod)
        self.assertEqual(response.model_dump(by_alias=True, exclude_unset=True)["authMethods"][0]["type"], "terminal")
        with self.assertRaises(ValidationError):
            schema.SessionNotification.model_validate({"sessionId": "session", "update": {
                "sessionUpdate": "unknown", "content": {"type": "text", "text": "hello"},
            }})
        with self.assertRaises(ValidationError):
            schema.InitializeRequest.model_validate({"protocolVersion": -1})


@unittest.skipUnless(sys.platform.startswith("linux") and sys.version_info >= (3, 14),
                     "Original ACP gateway integration currently verified on Linux/Python 3.14")
class OriginalACPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from standalone.tests.test_entrypoints import BundledSourceTestCase
        BundledSourceTestCase.setUpClass.__func__(cls)

    def test_original_acp_stdio_uses_canonical_gateway_and_real_http_tool_turn(self):
        from standalone.tests.test_chat_entry import _LAUNCHER, OriginalChatTests
        from standalone.tests.test_entrypoints import support
        from standalone.tests.test_transport_agent import _AgentHTTPHandler
        with tempfile.TemporaryDirectory(prefix="acp-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _AgentHTTPHandler)
            server.requests = []
            server.daemon_threads = True
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            context = dict(self.context, artifact=str(scratch / "hermes.py"))
            context["manifest"] = dict(context["manifest"], runtime={"hermes_python": "3.14"})
            Path(context["artifact"]).write_text(
                _LAUNCHER.replace("CONTEXT_JSON", repr(json.dumps(context)))
                .replace("SCRATCH_ROOT", repr(str(scratch)))
                .replace("ENDPOINT_PORT", str(server.server_port)), encoding="utf-8")
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            environment["PATH"] = ""
            home = Path(environment["HERMES_HOME"])
            home.mkdir()
            (home / "config.yaml").write_text(json.dumps({
                "model": {"default": "standalone-wire-fixture", "provider": "custom",
                          "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                          "api_key": "local-http-fixture-token", "context_length": 131072},
                "agent": {"environment_probe": False, "api_max_retries": 1, "auto_recovery_cycles": 0,
                          "gateway_startup_warmup_timeout": 0, "agent_cache": {"memory_high_mb": 0}},
                "gateway": {"multiplex_profiles": False, "loop_watchdog": False},
                "compression": {"enabled": False}, "tools": {"tool_search": {"enabled": False}},
                "platform_toolsets": {"acp": ["todo"]},
                "model_catalog": {"enabled": False, "excluded_providers": ["copilot"]},
                "auxiliary": {"title_generation": {"enabled": False}},
            }), encoding="utf-8")
            # The original ACP picker reads its normal profile cache. Seed a
            # fresh local registry so this fixture needs only the provider wire.
            (home / "models_dev_cache.json").write_text(json.dumps({
                "standalone-fixture": {"id": "standalone-fixture", "name": "Fixture", "env": [],
                    "models": {"standalone-wire-fixture": {"id": "standalone-wire-fixture",
                        "name": "Fixture", "tool_call": True, "limit": {"context": 131072, "output": 4096}}}},
            }), encoding="utf-8")
            command = [sys.executable, "-I", "-S", "-B", "-X", "utf8", context["artifact"], "acp"]
            checked = subprocess.run([*command, "--check"], env=environment, cwd=scratch / "work",
                                     capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("Hermes ACP check OK", checked.stdout)
            child = subprocess.Popen(command, env=environment, cwd=scratch / "work", stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            messages, errors = queue.Queue(), bytearray()
            reader = threading.Thread(target=lambda: self._read_messages(child, messages), daemon=True)
            reader.start()
            def capture_errors():
                for line in child.stderr:
                    errors.extend(line)
            stderr = threading.Thread(target=capture_errors, daemon=True)
            stderr.start()
            notices = []
            try:
                def request(number, method, params):
                    child.stdin.write((json.dumps({"jsonrpc": "2.0", "id": number, "method": method,
                                                  "params": params}) + "\n").encode())
                    child.stdin.flush()
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        try:
                            message = messages.get(timeout=max(0.01, deadline - time.monotonic()))
                        except queue.Empty:
                            self.fail("ACP reply timed out: " + errors.decode(errors="replace")[-4000:])
                        if isinstance(message, BaseException):
                            raise message
                        if message is None:
                            self.fail("ACP closed before reply: " + errors.decode(errors="replace")[-4000:])
                        if message.get("id") == number and "method" not in message:
                            return message
                        if message.get("method") == "session/request_permission" and "id" in message:
                            choices = message["params"]["options"]
                            selected = next(item for item in choices if item["kind"] == "allow_once")
                            child.stdin.write((json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": {
                                "outcome": {"outcome": "selected", "optionId": selected["optionId"]},
                            }}) + "\n").encode())
                            child.stdin.flush()
                        notices.append(message)
                    self.fail("ACP reply deadline expired")

                initialized = request(1, "initialize", {"protocolVersion": 1, "clientCapabilities": {
                    "fs": {"readTextFile": True, "writeTextFile": True}, "auth": {"terminal": True}}})
                self.assertEqual(initialized["result"]["protocolVersion"], 1, initialized)
                self.assertTrue(initialized["result"]["agentCapabilities"]["loadSession"])
                invalid = request(2, "session/new", {"mcpServers": []})
                self.assertEqual(invalid["error"]["code"], -32602, invalid)
                unknown = request(3, "ping", {})
                self.assertEqual(unknown["error"]["code"], -32601, unknown)
                created = request(4, "session/new", {"cwd": str(scratch / "work"), "mcpServers": []})
                self.assertIn("result", created, (created, errors.decode(errors="replace")[-5000:]))
                session_id = created["result"]["sessionId"]
                response = request(5, "session/prompt", {"sessionId": session_id, "prompt": [{
                    "type": "text", "text": "Record the HTTP verification task as completed, then confirm completion.",
                }]})
                self.assertEqual(response["result"]["stopReason"], "end_turn", response)
                updates = [message["params"]["update"] for message in notices
                           if message.get("method") == "session/update"]
                text = "".join(update.get("content", {}).get("text", "") for update in updates
                               if update.get("sessionUpdate") == "agent_message_chunk")
                self.assertIn("HTTP fixture complete", text, updates)
                self.assertTrue(any(update.get("sessionUpdate") == "tool_call" for update in updates), updates)
                listing = request(6, "session/list", {})
                self.assertIsInstance(listing["result"]["sessions"], list, listing)
                resumed = request(7, "session/resume", {"sessionId": session_id,
                    "cwd": str(scratch / "work"), "mcpServers": []})
                self.assertIn("result", resumed, resumed)
                self.assertEqual(resumed["result"]["_meta"]["hermes"]["sessionProvenance"]["acpSessionId"],
                                 session_id, resumed)
                self.assertEqual(len(server.requests), 2)
                with sqlite3.connect(home / "state.db") as database:
                    persisted = database.execute("SELECT role, content FROM messages ORDER BY id").fetchall()
                self.assertIn(("assistant", "HTTP fixture complete"), persisted)
                self.assertTrue(any(role == "tool" for role, _ in persisted))
                child.stdin.close()
                self.assertEqual(child.wait(timeout=10), 0, errors.decode(errors="replace")[-5000:])
            except BaseException as error:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=5)
                OriginalChatTests._stop_fixture_processes(self, scratch)
                logs = {path.name: path.read_text(errors="replace")[-6000:]
                        for path in (home / "logs").glob("*.log")}
                events = scratch / "guard-events.jsonl"
                if events.exists():
                    logs["guard-events"] = events.read_text()[-4000:]
                logs["last_notices"] = notices[-8:]
                raise AssertionError(f"{error}\nCompleted fixture logs: {logs}") from error
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=5)
                OriginalChatTests._stop_fixture_processes(self, scratch)
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=5)

    @staticmethod
    def _read_messages(child, messages):
        try:
            for line in child.stdout:
                messages.put(json.loads(line))
        except BaseException as error:
            messages.put(error)
        finally:
            messages.put(None)


if __name__ == "__main__":
    unittest.main()
