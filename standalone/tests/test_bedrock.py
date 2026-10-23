"""Real HTTP signing and AWS binary event framing, with no AWS network calls."""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import struct
import threading
import unittest
import zlib

from standalone.tests import test_transport as _fixture
from standalone.tests.test_transport_protocols import message_events


class BedrockDependencyTests(_fixture.TransportDependencyTests):
    DEPENDENCIES = ("httpx", "boto3")
    CASES = "standalone.tests.test_bedrock.BedrockWireTests"


def event_frame(payload, *, kind="chunk", message_type="event"):
    headers = bytearray()
    kind_header = ":exception-type" if message_type == "exception" else ":event-type"
    for name, value in ((":message-type", message_type), (kind_header, kind),
                        (":content-type", "application/json")):
        key, value = name.encode(), value.encode()
        headers.extend(bytes([len(key)]) + key + b"\x07" + struct.pack(">H", len(value)) + value)
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", len(headers) + len(body) + 16, len(headers))
    frame = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + body
    return frame + struct.pack(">I", zlib.crc32(frame))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.server.records.append((self.path, dict(self.headers), body))
        if self.path.endswith("invoke-with-response-stream"):
            payload = b"".join(event_frame({"bytes": base64.b64encode(json.dumps(event).encode()).decode()})
                               for event in message_events())
            if body.get("fixture_bad_crc"):
                payload = payload[:12] + bytes([payload[12] ^ 1]) + payload[13:]
            if body.get("fixture_error"):
                payload = event_frame({"message": "fixture throttling"}, kind="throttlingException",
                                      message_type="exception")
            mime = "application/vnd.amazon.eventstream"
        else:
            payload = json.dumps({"id": "bedrock-1", "type": "message", "role": "assistant",
                                  "content": [{"type": "text", "text": "서명 확인"}],
                                  "usage": {"input_tokens": 2, "output_tokens": 3}}).encode()
            mime = "application/json"
        self.send_response(200)
        self.send_header("content-type", mime)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        for index in range(0, len(payload), 13):
            self.wfile.write(payload[index:index + 13])


@unittest.skipUnless(importlib.util.find_spec("boto3"), "Runs in the isolated dependency subprocess")
class BedrockWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
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

    def client(self, **kwargs):
        from standalone.transport_bedrock import AnthropicBedrock
        client = AnthropicBedrock(base_url=self.url, aws_region="us-east-1", max_retries=0, **kwargs)
        self.addCleanup(client.close)
        return client

    def test_signed_messages_and_binary_event_frames(self):
        client = self.client(aws_access_key="fixture-access", aws_secret_key="fixture-secret",
                             aws_session_token="fixture-session",
                             default_headers={"anthropic-beta": "fixture-one,fixture-two"})
        result = client.messages.create(model="anthropic.fixture:0", messages=[], max_tokens=100)
        self.assertEqual(result.content[0].text, "서명 확인")
        path, headers, body = self.server.records[-1]
        headers = {name.lower(): value for name, value in headers.items()}
        self.assertEqual(path, "/model/anthropic.fixture:0/invoke")
        self.assertIn("Credential=fixture-access/", headers["authorization"])
        self.assertIn("/us-east-1/bedrock/aws4_request", headers["authorization"])
        self.assertEqual(headers["x-amz-security-token"], "fixture-session")
        self.assertNotIn("model", body)
        self.assertNotIn("stream", body)
        self.assertEqual(body["anthropic_version"], "bedrock-2023-05-31")
        self.assertEqual(body["anthropic_beta"], ["fixture-one", "fixture-two"])
        with client.messages.stream(model="anthropic.fixture:0", messages=[], max_tokens=100) as stream:
            final = stream.get_final_message()
        self.assertEqual(final.content[1].input, {"task": "전체 이식"})
        self.assertEqual(final.content[0].signature, "signed-opaque")
        self.assertTrue(stream.response.is_closed)

    def test_bearer_and_event_errors(self):
        from botocore.eventstream import ChecksumMismatch
        from standalone.transport_types import RateLimitError
        client = self.client(api_key="fixture-bearer")
        client.messages.create(model="fixture", messages=[], max_tokens=10)
        headers = {name.lower(): value for name, value in self.server.records[-1][1].items()}
        self.assertEqual(headers["authorization"], "Bearer fixture-bearer")
        self.assertNotIn("x-api-key", headers)
        for key, error in (("fixture_bad_crc", ChecksumMismatch), ("fixture_error", RateLimitError)):
            with self.subTest(key=key):
                with client.messages.stream(model="fixture", messages=[], max_tokens=10,
                                            extra_body={key: True}) as stream:
                    with self.assertRaises(error):
                        stream.get_final_message()
                self.assertTrue(stream.response.is_closed)
