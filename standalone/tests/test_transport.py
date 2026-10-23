"""Real loopback HTTP tests, run with reviewed cached wheels and no site-packages."""

from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.parse import parse_qs, urlsplit


PROJECT = Path(__file__).resolve().parents[2]


class TransportDependencyTests(unittest.TestCase):
    CASES = "standalone.tests.test_transport.TransportWireTests"
    DEPENDENCIES = ("httpx",)

    def test_real_wire_contract_without_site_packages(self):
        from standalone import vendor as vendor_tool

        downloads = PROJECT / "standalone" / ".cache" / "downloads"
        lock = vendor_tool.load_dependencies(PROJECT / "standalone" / "dependencies.json")
        packages = {item["name"]: item for item in lock["packages"]}
        # The lock targets 3.14, where these conditional edges are absent.
        # Transport/build checks also run on the bootstrap's older Python.
        wanted, pending = set(), [*self.DEPENDENCIES, "typing-extensions"]
        while pending:
            name = pending.pop()
            if name not in wanted:
                wanted.add(name)
                pending.extend(set(packages[name].get("dependencies", []))
                               - set(packages[name].get("replaced_dependencies", {})))
        if any(not (downloads / packages[name]["filename"]).is_file() for name in wanted):
            self.skipTest("Reviewed cached wheels are required for transport integration")
        with tempfile.TemporaryDirectory(prefix="hermes-wire-tests-") as directory:
            root = Path(directory)
            vendor = root / "vendor"
            vendor.mkdir()
            paths = vendor_tool.ArchivePaths()
            for name in sorted(wanted):
                files, _ = vendor_tool.read_wheel(packages[name], downloads)
                for path, data in files.items():
                    paths.add(path)
                    target = vendor / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
            source = (
                f"import sys, unittest; sys.path[:0] = [{str(PROJECT)!r}, {str(vendor)!r}]; "
                f"suite = unittest.defaultTestLoader.loadTestsFromName({self.CASES!r}); "
                "result = unittest.TextTestRunner(verbosity=2).run(suite); "
                "raise SystemExit(not result.wasSuccessful())"
            )
            environment = {"HOME": str(root), "USERPROFILE": str(root), "HERMES_HOME": str(root / "hermes")}
            for name in ("SYSTEMROOT", "WINDIR"):
                if name in os.environ:
                    environment[name] = os.environ[name]
            result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", source],
                                    env=environment, cwd=root, capture_output=True, text=True,
                                    encoding="utf-8", timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class _FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        self._respond(None)

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        self._respond(json.loads(self.rfile.read(length)))

    def _send(self, content, *, status=200, mime="application/json", headers=None):
        self.send_response(status)
        self.send_header("content-type", mime)
        self.send_header("content-length", str(len(content)))
        self.send_header("x-request-id", "wire-request-17")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            # Deliberately split UTF-8 sequences and CRLF boundaries on the wire.
            for start in range(0, len(content), 7):
                self.wfile.write(content[start:start + 7])
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _respond(self, body):
        record = {"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body}
        with self.server.records_lock:
            self.server.records.append(record)
        mode = (body or {}).get("fixture", "")
        if mode == "slow":
            time.sleep(0.2)
        if mode in ("error", "retry"):
            prior = sum(item["body"] and item["body"].get("fixture") == mode for item in self.server.records)
            if mode == "error" or prior == 1:
                payload = {"error": {"message": "fixture rate limit", "code": "rate_limit", "type": "quota"}}
                self._send(json.dumps(payload).encode(), status=429, headers={"retry-after": "0"})
                return
        if mode == "plain-error":
            self._send(b"fixture forbidden", status=403, mime="text/plain")
            return
        if mode == "invalid-json":
            self._send(b"not-json")
            return
        if (body or {}).get("stream"):
            if mode == "error-event":
                self._send(b'event: error\ndata: {"error":{"message":"stream rejected","code":429}}\n\n',
                           mime="text/event-stream")
                return
            if mode == "invalid-event":
                self._send(b"data: invalid\n\n", mime="text/event-stream")
                return
            if mode == "stall":
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n')
                self.wfile.flush()
                self.server.release.wait(3)
                self.close_connection = True
                return
            chunks = [
                {"id": "chat-fixture", "model": "local-fixture", "choices": [{"index": 0, "delta": {
                    "role": "assistant", "reasoning_content": "생각", "reasoning_details": [{"type": "reasoning.text", "text": "생각"}]}}]},
                {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "tool-1", "type": "function",
                    "function": {"name": "calculate", "arguments": '{"x":'}}]}}]},
                {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "2}"}}]}}]},
                {"choices": [{"index": 0, "delta": {"content": "완료"}, "finish_reason": "tool_calls"}]},
                {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}},
            ]
            lines = ["\ufeff: fixture heartbeat\r\nid: frame-1\r\nretry: 10\r\n"]
            for item in chunks:
                # Pretty JSON exercises multi-line data fields.
                lines.append("".join("data: " + line + "\r\n" for line in json.dumps(item, ensure_ascii=False, indent=1).splitlines()) + "\r\n")
            lines.append("data: [DONE]\r\n\r\n")
            self._send("".join(lines).encode("utf-8"), mime="text/event-stream")
            return
        if self.command == "GET":
            response = {"object": "list", "data": [{"id": "local-fixture", "object": "model"}]}
        else:
            response = {"id": "chat-fixture", "model": "local-fixture", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "안녕하세요", "reasoning_content": "생각",
                            "reasoning_details": [{"type": "reasoning.text", "text": "생각"}],
                            "tool_calls": [{"id": "tool-1", "type": "function", "function": {
                                "name": "calculate", "arguments": '{"x":2}'}}]}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10,
                          "prompt_tokens_details": {"cached_tokens": 2}}}
        self._send(json.dumps(response, ensure_ascii=False).encode("utf-8"))


@unittest.skipUnless(importlib.util.find_spec("httpx"), "Wire cases run in the isolated dependency subprocess")
class TransportWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
        cls.server.daemon_threads = True
        cls.server.records = []
        cls.server.records_lock = threading.Lock()
        cls.server.release = threading.Event()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.release.set()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def setUp(self):
        with self.server.records_lock:
            self.server.records.clear()

    def client(self, **kwargs):
        from standalone.transport_chat import OpenAI
        client = OpenAI(base_url=self.url, api_key="local-fixture-token", max_retries=0, **kwargs)
        self.addCleanup(client.close)
        return client

    def test_json_request_options_and_unknown_response_fields(self):
        from standalone.transport_chat import OpenAI
        client = OpenAI(base_url=self.url + "?api-version=base", api_key="local-fixture-token", max_retries=0,
                        default_headers={"X-Profile": "default"}, default_query={"client": "yes"})
        self.addCleanup(client.close)
        messages = [{"role": "user", "content": [{"type": "text", "text": "안녕"}]}]
        response = client.chat.completions.create(model="fixture", messages=messages, temperature=0.8,
            tools=[{"type": "function", "function": {"name": "calculate", "parameters": {"type": "object"}}}],
            extra_body={"temperature": 0, "enabled": False, "nullable": None},
            extra_headers={"x-profile": "request"}, extra_query={"api-version": "request"})
        record = self.server.records[-1]
        self.assertEqual(urlsplit(record["path"]).path, "/v1/chat/completions")
        self.assertEqual(parse_qs(urlsplit(record["path"]).query), {"api-version": ["request"], "client": ["yes"]})
        self.assertEqual(record["body"]["messages"], messages)
        self.assertEqual(record["body"]["temperature"], 0)
        self.assertIs(record["body"]["enabled"], False)
        self.assertIsNone(record["body"]["nullable"])
        self.assertNotIn("extra_headers", record["body"])
        headers = {key.lower(): value for key, value in record["headers"].items()}
        self.assertEqual(headers["x-profile"], "request")
        self.assertEqual(headers["authorization"], "Bearer local-fixture-token")
        self.assertEqual(response.choices[0].message.tool_calls[0].function.name, "calculate")
        self.assertEqual(response.choices[0].message.model_extra["reasoning_content"], "생각")
        self.assertIsInstance(response.choices[0].message.reasoning_details[0], dict)
        self.assertEqual(response.usage.prompt_tokens_details.cached_tokens, 2)
        self.assertEqual(response.model_dump()["choices"][0]["message"]["content"], "안녕하세요")
        self.assertEqual(response._request_id, "wire-request-17")
        self.assertIsNone(response.choices[0].message.refusal)

    def test_sse_tools_reasoning_usage_and_multiline_utf8(self):
        client = self.client()
        stream = client.chat.completions.create(model="fixture", messages=[], stream=True)
        chunks = list(stream)
        self.assertEqual(len(chunks), 5)
        self.assertEqual(chunks[0].choices[0].delta.reasoning_content, "생각")
        self.assertEqual(chunks[1].choices[0].delta.tool_calls[0].function.arguments, '{"x":')
        self.assertEqual(chunks[2].choices[0].delta.tool_calls[0].function.arguments, "2}")
        self.assertEqual(chunks[3].choices[0].delta.content, "완료")
        self.assertEqual(chunks[-1].usage.total_tokens, 10)
        self.assertTrue(stream.response.is_closed)
        self.assertFalse(client.is_closed())

    def test_error_status_body_and_headers_remain_inspectable(self):
        from standalone.transport_types import PermissionDeniedError, RateLimitError
        client = self.client()
        with self.assertRaises(RateLimitError) as caught:
            client.chat.completions.create(model="fixture", messages=[], fixture="error")
        error = caught.exception
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.body["error"]["code"], "rate_limit")
        self.assertEqual(error.type, "quota")
        self.assertEqual(error.response.headers["retry-after"], "0")
        self.assertEqual(error.request_id, "wire-request-17")
        self.assertTrue(error.response.is_closed)
        with self.assertRaises(PermissionDeniedError) as caught:
            client.chat.completions.create(model="fixture", messages=[], fixture="plain-error")
        self.assertEqual(caught.exception.body, "fixture forbidden")

    def test_auxiliary_retries_and_main_no_retry(self):
        from standalone.transport_chat import OpenAI
        with OpenAI(base_url=self.url, api_key="fixture", max_retries=1) as client:
            result = client.chat.completions.create(model="fixture", messages=[], fixture="retry")
        self.assertEqual(result.id, "chat-fixture")
        self.assertEqual(len(self.server.records), 2)

    def test_bad_json_and_stream_error_are_explicit_and_close_response(self):
        from standalone.transport_types import APIResponseValidationError, RateLimitError
        client = self.client()
        with self.assertRaises(APIResponseValidationError):
            client.chat.completions.create(model="fixture", messages=[], fixture="invalid-json")
        for mode, error_class in (("invalid-event", APIResponseValidationError), ("error-event", RateLimitError)):
            stream = client.chat.completions.create(model="fixture", messages=[], stream=True, fixture=mode)
            with self.assertRaises(error_class):
                next(stream)
            self.assertTrue(stream.response.is_closed)

    def test_stream_close_releases_only_its_response(self):
        client = self.client()
        stream = client.chat.completions.create(model="fixture", messages=[], stream=True, fixture="stall")
        self.assertEqual(next(stream).choices[0].delta.content, "first")
        stream.close()
        self.assertTrue(stream.response.is_closed)
        self.assertFalse(client.is_closed())
        self.assertEqual(client.chat.completions.create(model="fixture", messages=[]).id, "chat-fixture")

    def test_timeout_override_is_applied_to_real_socket(self):
        from standalone.transport_types import APITimeoutError
        client = self.client(timeout=5)
        with self.assertRaises(APITimeoutError) as caught:
            client.chat.completions.create(model="fixture", messages=[], fixture="slow", timeout=0.025)
        self.assertEqual(caught.exception.request.extensions["timeout"]["read"], 0.025)
        self.assertFalse(client.is_closed())

    def test_callable_credentials_and_copied_client_ownership(self):
        from standalone.transport_chat import OpenAI
        count = 0
        def token():
            nonlocal count
            count += 1
            return f"fixture-{count}"
        client = OpenAI(base_url=self.url, api_key=token, max_retries=0, default_headers={"X-Base": "retained"})
        clone = client.with_options(default_headers={"X-Route": "clone"})
        self.addCleanup(client.close)
        self.addCleanup(clone.close)
        client.close()
        self.assertFalse(clone.is_closed())
        self.assertEqual(clone.models.list().data[0].id, "local-fixture")
        clone.chat.completions.create(model="fixture", messages=[])
        auth = [{k.lower(): v for k, v in item["headers"].items()}["authorization"] for item in self.server.records]
        self.assertEqual(auth, ["Bearer fixture-1", "Bearer fixture-2"])
        clone.close()
        self.assertTrue(clone._client.is_closed)
        self.assertTrue(client.is_closed())

    def test_async_json_stream_credentials_and_cancel(self):
        from standalone.transport_chat import AsyncOpenAI

        async def run():
            async def token():
                return "async-fixture"
            async with AsyncOpenAI(base_url=self.url, api_key=token, max_retries=0) as client:
                response = await client.chat.completions.create(model="fixture", messages=[])
                self.assertEqual(response.id, "chat-fixture")
                stream = await client.chat.completions.create(model="fixture", messages=[], stream=True)
                chunks = [chunk async for chunk in stream]
                self.assertEqual(chunks[-1].usage.total_tokens, 10)
                self.assertTrue(stream.response.is_closed)
                stalled = await client.chat.completions.create(model="fixture", messages=[], stream=True, fixture="stall")
                self.assertEqual((await anext(stalled)).choices[0].delta.content, "first")
                task = asyncio.create_task(anext(stalled))
                await asyncio.sleep(0.02)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(stalled.response.is_closed)
                self.assertFalse(client.is_closed())
                self.assertEqual((await client.models.list()).data[0].id, "local-fixture")
            self.assertTrue(client.is_closed())
        asyncio.run(run())

    def test_cross_origin_raw_request_is_rejected_before_network(self):
        client = self.client()
        with self.assertRaisesRegex(ValueError, "configured endpoint origin"):
            client.get("http://provider.invalid/models")
        self.assertEqual(self.server.records, [])


if __name__ == "__main__":
    unittest.main()
