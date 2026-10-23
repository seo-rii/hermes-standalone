"""Original AIAgent constructor and tool turn through an actual HTTP endpoint."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest


PROJECT = Path(__file__).resolve().parents[2]
TODO = {"id": "wire-probe", "content": "Verify original HTTP tool turn", "status": "completed"}


class _AgentHTTPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.server.requests.append({"path": self.path, "body": body})
        replies = [item for item in body.get("messages", []) if item.get("role") == "tool"]
        title_schema = body.get("response_format", {}).get("json_schema", {}).get("name") == "session_title"
        if title_schema:
            message = {"role": "assistant", "content": json.dumps({"title": "Verify HTTP task completion"}),
                       "tool_calls": None}
            finish = "stop"
        elif replies:
            message = {"role": "assistant", "content": "HTTP fixture complete", "tool_calls": None}
            finish = "stop"
        else:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "http-tool-1", "type": "function", "function": {
                    "name": "todo_list", "arguments": json.dumps({"todos": [TODO]})}}]}
            finish = "tool_calls"
        response = {"id": f"http-response-{len(self.server.requests)}", "model": "standalone-wire-fixture",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        if body.get("stream"):
            delta = dict(message)
            if delta.get("tool_calls"):
                delta["tool_calls"] = [{"index": 0, **delta["tool_calls"][0]}]
            content = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in (
                {"id": response["id"], "model": response["model"], "choices": [{"index": 0, "delta": delta}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                {"choices": [], "usage": response["usage"]},
            )) + "data: [DONE]\n\n"
            mime = "text/event-stream"
        else:
            content, mime = json.dumps(response), "application/json"
        payload = content.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", mime)
        self.send_header("content-length", str(len(payload)))
        self.send_header("x-litellm-model-id", "local-fixture-deployment")
        self.end_headers()
        try:
            self.wfile.write(payload)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class _ProtocolHTTPHandler(_AgentHTTPHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.server.requests.append({"path": self.path, "body": body})
        if self.path.endswith("/responses"):
            self._responses(body)
        elif self.path.endswith("/messages"):
            self._messages(body)
        else:
            self.send_error(404)

    def _send(self, body, events=None):
        if events is None:
            payload, mime = json.dumps(body).encode(), "application/json"
        else:
            payload = "".join("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n"
                              for event in events).encode()
            mime = "text/event-stream"
        self.send_response(200)
        self.send_header("content-type", mime)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _responses(self, body):
        replies = [item for item in body.get("input", []) if item.get("type") == "function_call_output"]
        if replies:
            output = [{"id": "msg-http-final", "type": "message", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": "HTTP fixture complete", "annotations": []}]}]
        else:
            output = [{"id": "fc-http-tool-1", "call_id": "http-tool-1", "type": "function_call", "status": "completed",
                       "name": "todo_list", "arguments": json.dumps({"todos": [TODO]})}]
        response = {"id": "resp-http-fixture", "object": "response", "status": "completed", "model": "standalone-wire-fixture",
                    "output": output, "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
        events = [{"type": "response.created", "response": {**response, "output": [], "status": "in_progress"}},
                  {"type": "response.output_item.done", "output_index": 0, "item": output[0]},
                  {"type": "response.completed", "response": response}]
        self._send(response, events if body.get("stream") else None)

    def _messages(self, body):
        replies = [block for item in body.get("messages", []) if isinstance(item.get("content"), list)
                   for block in item["content"] if block.get("type") == "tool_result"]
        if replies:
            block, reason = {"type": "text", "text": "HTTP fixture complete"}, "end_turn"
        else:
            block, reason = {"type": "tool_use", "id": "http-tool-1", "name": "todo_list", "input": {"todos": [TODO]}}, "tool_use"
        response = {"id": "msg-http-fixture", "type": "message", "role": "assistant", "model": "standalone-wire-fixture",
                    "content": [block], "stop_reason": reason, "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 5}}
        if block["type"] == "tool_use":
            start = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        else:
            start, delta = {"type": "text", "text": ""}, {"type": "text_delta", "text": block["text"]}
        events = [
            {"type": "message_start", "message": {**response, "content": [], "stop_reason": None,
                                                    "usage": None if self.server.null_usage else {"input_tokens": 10, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": start},
            {"type": "content_block_delta", "index": 0, "delta": delta},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None},
             "usage": None if self.server.null_usage else {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        self._send(response, events if body.get("stream") else None)


_WORKER = r'''
import contextlib, io, json, os, pathlib, sys, traceback
source, vendor, scratch, endpoint, streaming = sys.argv[1:6]
protocol = sys.argv[6] if len(sys.argv) > 6 else "chat"
api_mode = {"chat": "chat_completions", "responses": "codex_responses", "messages": "anthropic_messages"}[protocol]
source, vendor, scratch = map(pathlib.Path, (source, vendor, scratch))
sys.path[:0] = [str(source), str(vendor)]
from standalone.runtime import initialize_bundle
initialize_bundle({"root": str(source.parent), "artifact": str(scratch / "fixture.py"),
                   "bundle_sha256": source.parent.name, "manifest": {"project": "hermes-standalone"}})
home = pathlib.Path(os.environ["HERMES_HOME"])
home.mkdir(parents=True)
(home / "config.yaml").write_text(json.dumps({
    "model": {"default": "standalone-wire-fixture", "provider": "standalone-wire-fixture",
              "base_url": endpoint, "context_length": 131072, "streaming": streaming == "true"},
    "agent": {"environment_probe": False, "api_max_retries": 1, "auto_recovery_cycles": 0},
    "compression": {"enabled": False},
    "tools": {"tool_search": {"enabled": False}},
}), encoding="utf-8")
from urllib.parse import urlsplit
address = urlsplit(endpoint)
blocked = []
def audit(event, args):
    if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork"):
        blocked.append(event)
        raise PermissionError("HTTP fixture disallows process execution")
    if event == "socket.getaddrinfo" and (args[0] != "127.0.0.1" or int(args[1]) != address.port):
        blocked.append(event)
        raise PermissionError("HTTP fixture disallows external name resolution")
    if event == "socket.connect" and args[1] != ("127.0.0.1", address.port):
        blocked.append(event)
        raise PermissionError("HTTP fixture allows only its local server")
    if event == "import" and args[0].split(".")[0] in {"pip", "uv", "ensurepip"}:
        blocked.append(event)
        raise PermissionError("HTTP fixture disallows installers")
sys.addaudithook(audit)
output, errors = io.StringIO(), io.StringIO()
report = {"ok": False, "blocked": blocked, "isolated": bool(sys.flags.isolated and sys.flags.no_site)}
try:
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        from run_agent import AIAgent
        agent = AIAgent(model="standalone-wire-fixture", provider="standalone-wire-fixture",
                        api_mode=api_mode, base_url=endpoint, api_key="local-http-fixture-token",
                        enabled_toolsets=["todo"], skip_context_files=True, skip_memory=True,
                        skip_background_review=True, quiet_mode=True, save_trajectories=False,
                        max_iterations=3, cwd=str(scratch / "work"))
        try:
            report["client_module"] = type(agent._anthropic_client if protocol == "messages" else agent.client).__module__
            conversation = agent.run_conversation(
                "Record the HTTP verification task as completed, then confirm completion.",
                system_message="Use the todo_list tool once, then report completion.")
            report["final_response"] = conversation.get("final_response")
            report["roles"] = [item.get("role") for item in conversation.get("messages", [])]
            report["tool_results"] = [item.get("content") for item in conversation.get("messages", []) if item.get("role") == "tool"]
            report["todos"] = agent._todo_store.read()
            report["served_model"] = getattr(agent, "last_served_model", None)
            report["ok"] = report["final_response"] == "HTTP fixture complete"
        finally:
            agent.close()
except BaseException as error:
    report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
report["stdout"] = output.getvalue()[-2000:]
report["stderr"] = errors.getvalue()[-3000:]
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(not report["ok"] or bool(blocked))
'''


@unittest.skipIf(sys.version_info < (3, 14), "The upstream agent targets Python 3.14")
class AgentHTTPIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from standalone.offline_agent_probe import _source_snapshot, _vendor_sources
        cls.temporary = tempfile.TemporaryDirectory(prefix="hermes-agent-http-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        downloads = PROJECT / "standalone" / ".cache" / "downloads"
        manifest = json.loads((PROJECT / "standalone" / "dependencies.json").read_text(encoding="utf-8"))
        if any(not (downloads / item["filename"]).is_file() for item in manifest["packages"]):
            raise unittest.SkipTest("Reviewed cached inputs are required for the original agent HTTP integration")
        snapshot = cls.directory / "snapshot"
        snapshot.mkdir()
        vendor = cls.directory / "vendor"
        vendor.mkdir()
        _vendor_sources(PROJECT, downloads, vendor)
        _, digest = _source_snapshot(PROJECT, snapshot)
        bundle = cls.directory / digest
        bundle.mkdir()
        cls.source, cls.vendor = bundle / "app", bundle / "vendor"
        snapshot.rename(cls.source)
        vendor.rename(cls.vendor)
        cls.worker = cls.directory / "worker.py"
        cls.worker.write_text(_WORKER, encoding="utf-8")

    def run_agent_fixture(self, *, streaming, protocol="chat", null_usage=False):
        from standalone.agent_probe_worker import fixture_environment
        scratch = self.directory / (protocol + ("-stream" if streaming else "-json"))
        for name in ("home", "work", "tmp"):
            (scratch / name).mkdir(parents=True)
        server = ThreadingHTTPServer(("127.0.0.1", 0), _AgentHTTPHandler if protocol == "chat" else _ProtocolHTTPHandler)
        server.daemon_threads = True
        server.requests = []
        server.null_usage = null_usage
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run([
                sys.executable, "-I", "-S", "-B", "-X", "utf8", str(self.worker), str(self.source),
                str(self.vendor), str(scratch), f"http://127.0.0.1:{server.server_port}/v1",
                "true" if streaming else "false", protocol,
            ], env=fixture_environment(scratch), cwd=scratch / "work", capture_output=True,
               text=True, encoding="utf-8", timeout=60)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["isolated"])
        self.assertEqual(report["client_module"], "standalone.transport_messages" if protocol == "messages" else "standalone.transport_chat")
        self.assertEqual(report["roles"], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(report["todos"], [TODO], report)
        if protocol == "chat":
            self.assertEqual(report["served_model"], "local-fixture-deployment")
        self.assertEqual(report["blocked"], [])
        self.assertEqual(len(server.requests), 2, server.requests)
        first, second = [item["body"] for item in server.requests]
        expected_path = "/v1/" + {"chat": "chat/completions", "responses": "responses", "messages": "messages"}[protocol]
        self.assertTrue(all(item["path"] == expected_path for item in server.requests), server.requests)
        self.assertEqual(bool(first.get("stream")), streaming)
        history = "input" if protocol == "responses" else "messages"
        self.assertEqual(first[history], second[history][:len(first[history])])
        self.assertEqual(first["tools"], second["tools"])
        self.assertIn("todo_list", {tool["function"]["name"] if protocol == "chat" else tool["name"] for tool in first["tools"]})
        if protocol == "responses":
            replies = [item for item in second["input"] if item.get("type") == "function_call_output"]
            self.assertEqual(len(replies), 1)
            self.assertEqual(replies[0]["call_id"], "http-tool-1")
            self.assertEqual(json.loads(replies[0]["output"])["todos"], [TODO])
            return
        if protocol == "messages":
            self.assertEqual(first.get("system"), second.get("system"))
            replies = [block for item in second["messages"] if isinstance(item.get("content"), list)
                       for block in item["content"] if block.get("type") == "tool_result"]
            self.assertEqual(len(replies), 1)
            self.assertEqual(replies[0]["tool_use_id"], "http-tool-1")
            content = replies[0]["content"]
            if isinstance(content, list):
                content = "".join(item.get("text", "") for item in content)
            self.assertEqual(json.loads(content)["todos"], [TODO])
            return
        replies = [item for item in second["messages"] if item.get("role") == "tool"]
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]["tool_call_id"], "http-tool-1")
        self.assertEqual(json.loads(replies[0]["content"])["todos"], [TODO])

    def test_original_constructor_tool_turn_over_json_http(self):
        self.run_agent_fixture(streaming=False)

    def test_original_constructor_tool_turn_over_sse_http(self):
        self.run_agent_fixture(streaming=True)

    def test_original_responses_constructor_tool_turn_over_sse_http(self):
        self.run_agent_fixture(streaming=True, protocol="responses")

    def test_original_anthropic_constructor_tool_turn_over_json_http(self):
        self.run_agent_fixture(streaming=False, protocol="messages")

    def test_original_anthropic_constructor_tool_turn_over_sse_http(self):
        self.run_agent_fixture(streaming=True, protocol="messages", null_usage=True)


if __name__ == "__main__":
    unittest.main()
