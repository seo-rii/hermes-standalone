"""Messages and Responses contracts against an actual local HTTP server."""

from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import threading
import unittest

from standalone.tests import test_transport as _transport_fixture


class ProtocolDependencyTests(_transport_fixture.TransportDependencyTests):
    CASES = "standalone.tests.test_transport_protocols.ProtocolWireTests"


def message_events(*, malformed=False, incomplete=False):
    events = [
        {"type": "message_start", "message": {"id": "msg-1", "type": "message", "role": "assistant",
            "model": "fixture", "content": [], "usage": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "검토"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "signed-opaque"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tool-1", "name": "todo_list", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"task":'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"전체 이식"}' if not malformed else 'broken'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": "", "citations": None}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "완료"}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "citations_delta", "citation": {"type": "char_location", "cited_text": "본문", "start_char_index": 0, "end_char_index": 2}}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None,
             "stop_details": {"type": "refusal", "unknown": "retained"}},
             "usage": {"output_tokens": 7, "cache_read_input_tokens": 3}},
        {"type": "message_delta", "delta": {}, "usage": None},
    ]
    if not incomplete:
        events.append({"type": "message_stop"})
    return events


class ProtocolHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.server.records.append((self.path, dict(self.headers), body))
        response = {"id": "response-1", "status": "completed", "output": [
            {"id": "reason-1", "type": "reasoning", "encrypted_content": "opaque-encrypted", "summary": []},
            {"id": "fc-1", "type": "function_call", "call_id": "call-1", "name": "todo_list", "arguments": "{}"},
            {"id": "text-1", "type": "message", "phase": "final_answer", "content": [{"type": "output_text", "text": "완료", "annotations": []}]},
        ], "usage": {"input_tokens": 8, "output_tokens": 7, "total_tokens": 15}, "provider_extension": {"keep": True}}
        if self.path.startswith("/v1/messages"):
            if self.path.endswith("count_tokens"):
                payload, mime = json.dumps({"input_tokens": 8}).encode(), "application/json"
            elif body.get("stream"):
                events = message_events(malformed=body.get("malformed", False), incomplete=body.get("incomplete", False))
                payload = "".join("event: " + event["type"] + "\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events).encode()
                mime = "text/event-stream"
            else:
                payload = json.dumps({"id": "msg-1", "content": [{"type": "text", "text": "완료"}], "usage": {"input_tokens": 8, "output_tokens": 7}}).encode()
                mime = "application/json"
        elif body.get("stream"):
            events = [
                {"type": "response.created", "response": {"id": "response-1", "status": "in_progress"}},
                *[{"type": "response.output_item.done", "output_index": index, "item": item} for index, item in enumerate(response["output"])],
                {"type": "response.completed", "response": response},
            ]
            payload = "".join("event: " + event["type"] + "\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events).encode()
            mime = "text/event-stream"
        else:
            payload, mime = json.dumps(response).encode(), "application/json"
        self.send_response(200)
        self.send_header("content-type", mime)
        self.send_header("content-length", str(len(payload)))
        self.send_header("request-id", "fixture-protocol-1")
        self.end_headers()
        # Exercise split codepoints as well as normal HTTP buffering.
        for index in range(0, len(payload), 11):
            self.wfile.write(payload[index:index + 11])


@unittest.skipUnless(importlib.util.find_spec("httpx"), "Runs inside isolated dependency fixture")
class ProtocolWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ProtocolHandler)
        cls.server.daemon_threads = True
        cls.server.records = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(3)

    def messages_client(self):
        from standalone.transport_messages import Anthropic, Omit
        client = Anthropic(base_url=self.url, api_key="fixture-key", auth_token="foreign-token",
                           default_headers={"Authorization": Omit(), "X-Profile": "fixture"})
        self.addCleanup(client.close)
        return client

    def test_messages_ordered_thinking_signature_tool_json_and_usage(self):
        client = self.messages_client()
        with client.messages.stream(model="fixture", messages=[{"role": "user", "content": "작업"}], max_tokens=200) as stream:
            self.assertIsNotNone(stream.response)
            events = list(stream)
            message = stream.get_final_message()
        self.assertEqual(events[-1].type, "message_stop")
        self.assertEqual([block.type for block in message.content], ["thinking", "tool_use", "text"])
        self.assertEqual(message.content[0].signature, "signed-opaque")
        self.assertEqual(message.content[0].thinking, "검토")
        self.assertEqual(message.content[1].input, {"task": "전체 이식"})
        self.assertEqual(message.content[2].citations[0].cited_text, "본문")
        self.assertEqual(message.usage.output_tokens, 7)
        self.assertEqual(message.usage.cache_read_input_tokens, 3)
        self.assertEqual(message.stop_details.unknown, "retained")
        path, headers, body = self.server.records[-1]
        headers = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(path, "/v1/messages")
        self.assertNotIn("authorization", headers)
        self.assertEqual(headers["x-api-key"], "fixture-key")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertTrue(stream.response.is_closed)
        self.assertFalse(client.is_closed())

    def test_messages_missing_stop_and_malformed_tool_input_fail(self):
        from standalone.transport_types import APIResponseValidationError
        for mode in ("malformed", "incomplete"):
            with self.subTest(mode=mode):
                with self.messages_client().messages.stream(model="fixture", extra_body={mode: True}) as stream:
                    with self.assertRaises(APIResponseValidationError):
                        stream.get_final_message()
                self.assertTrue(stream.response.is_closed)

    def test_messages_sync_json_and_count_tokens(self):
        client = self.messages_client()
        result = client.messages.create(model="fixture", messages=[], max_tokens=20)
        self.assertEqual(result.content[0].text, "완료")
        self.assertEqual(client.messages.count_tokens(model="fixture", messages=[]).input_tokens, 8)

    def test_responses_preserves_reasoning_items_identifiers_and_extensions(self):
        from standalone.transport_chat import OpenAI
        with OpenAI(base_url=self.url + "/v1", api_key="fixture") as client:
            response = client.responses.create(model="fixture", extra_body={"input": [{"role": "user", "content": "작업"}]})
            self.assertEqual(response.output_text, "완료")
            self.assertEqual(response.output[0].encrypted_content, "opaque-encrypted")
            self.assertEqual(response.output[1].call_id, "call-1")
            self.assertEqual(response.provider_extension.keep, True)
            self.assertEqual(response._request_id, "fixture-protocol-1")
            with client.responses.create(model="fixture", input=[], stream=True) as stream:
                events = list(stream)
            self.assertEqual(events[-1].response.output, response.output)
            self.assertTrue(stream.response.is_closed)
            client.responses.compact(model="fixture", input=[])
            self.assertEqual(self.server.records[-1][0], "/v1/responses/compact")

    def test_responses_async_json_and_sse(self):
        from standalone.transport_chat import AsyncOpenAI

        async def run():
            async with AsyncOpenAI(base_url=self.url + "/v1", api_key="fixture") as client:
                response = await client.responses.create(model="fixture", input=[])
                stream = await client.responses.create(model="fixture", input=[], stream=True)
                async with stream:
                    events = [event async for event in stream]
                self.assertEqual(events[-1].response.output, response.output)
                self.assertEqual(response.output_text, "완료")
                self.assertTrue(stream.response.is_closed)

        asyncio.run(run())
