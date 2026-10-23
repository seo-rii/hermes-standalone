"""Multipart uploads and incremental downloads against actual loopback HTTP."""

from __future__ import annotations

from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from standalone.tests import test_transport as _fixture


class MediaDependencyTests(_fixture.TransportDependencyTests):
    CASES = "standalone.tests.test_media.MediaWireTests"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.respond(None)

    def do_POST(self):
        payload = self.rfile.read(int(self.headers["content-length"]))
        if self.headers["content-type"].startswith("multipart/"):
            message = BytesParser(policy=default).parsebytes(
                b"Content-Type: " + self.headers["content-type"].encode() + b"\r\n\r\n" + payload)
            body = [(part.get_param("name", header="content-disposition"), part.get_filename(),
                     part.get_payload(decode=True)) for part in message.iter_parts()]
        else:
            body = json.loads(payload)
        self.respond(body)

    def respond(self, body):
        self.server.records.append((self.path, dict(self.headers), body))
        if self.path.endswith("audio/speech"):
            value, mime = b"\x00\x01\x02\x03" * 1500, "audio/pcm; rate=24000"
        elif self.path.endswith("/content"):
            value, mime = b"fixture-video-bytes", "video/mp4"
        elif self.path.endswith("audio/transcriptions"):
            text = any(key == "response_format" and value == b"text" for key, _, value in body)
            value, mime = ("정상 전사".encode(), "text/plain") if text else (json.dumps({"text": "정상 전사"}).encode(), "application/json")
        elif self.path.endswith(("images/edits", "images/generations")):
            value, mime = json.dumps({"data": [{"b64_json": "cG5n"}], "usage": {"total_tokens": 2}}).encode(), "application/json"
        else:
            value, mime = json.dumps({"id": "video-1", "status": "completed"}).encode(), "application/json"
        retry = self.server.retry_once
        self.server.retry_once = False
        self.send_response(503 if retry else 200)
        self.send_header("content-type", mime)
        self.send_header("content-length", str(len(value)))
        self.send_header("retry-after-ms", "0")
        self.end_headers()
        self.wfile.write(value)


@unittest.skipUnless(importlib.util.find_spec("httpx"), "Runs in the isolated dependency subprocess")
class MediaWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.daemon_threads = True
        cls.server.records = []
        cls.server.retry_once = False
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(3)

    def client(self):
        from standalone.transport_chat import OpenAI
        client = OpenAI(base_url=self.url, api_key="fixture", max_retries=1)
        self.addCleanup(client.close)
        return client

    def test_audio_file_replay_text_json_and_speech_stream_lifetime(self):
        client = self.client()
        self.server.retry_once = True
        source = io.BytesIO(b"fixture-waveform")
        source.name = "fixture.wav"
        text = client.audio.transcriptions.create(file=source, model="fixture", response_format="text",
                                                  extra_body={"languages": ["ko"], "condition": False})
        self.assertEqual(text, "정상 전사")
        first, second = self.server.records[-2:]
        self.assertEqual(first[2], second[2])
        self.assertIn(("file", "fixture.wav", b"fixture-waveform"), second[2])
        self.assertIn(("languages[]", None, b"ko"), second[2])
        self.assertIn(("condition", None, b"false"), second[2])
        self.assertFalse(source.closed)
        result = client.audio.transcriptions.create(file=("fixture.wav", b"audio"), model="fixture")
        self.assertEqual(result.text, "정상 전사")
        with client.audio.speech.with_streaming_response.create(model="fixture", input="안녕", voice="fixture") as stream:
            self.assertIn("rate=24000", stream.headers["content-type"])
            self.assertEqual(b"".join(stream.iter_bytes(511)), b"\x00\x01\x02\x03" * 1500)
        self.assertTrue(stream.response.is_closed)
        self.assertFalse(client.is_closed())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.pcm"
            client.audio.speech.create(model="fixture", input="안녕", voice="fixture").stream_to_file(path)
            self.assertEqual(path.stat().st_size, 6000)

    def test_multiple_image_files_masks_and_video_job_content(self):
        client = self.client()
        response = client.images.generate(model="fixture", prompt="그림", extra_body={"n": 1})
        self.assertEqual(response.data[0].b64_json, "cG5n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.png"
            path.write_bytes(b"image-one")
            client.images.edit(model="fixture", prompt="편집", image=[path, ("second.png", b"image-two", "image/png")],
                               mask=("mask.png", b"mask"))
        parts = self.server.records[-1][2]
        self.assertIn(("image[]", "fixture.png", b"image-one"), parts)
        self.assertIn(("image[]", "second.png", b"image-two"), parts)
        self.assertIn(("mask", "mask.png", b"mask"), parts)
        created = client.videos.create(model="fixture", prompt="영상", seconds="4", extra_body={"seed": 0})
        self.assertEqual(created.id, "video-1")
        self.assertIn(("seed", None, b"0"), self.server.records[-1][2])
        self.assertEqual(client.videos.retrieve(created.id).status, "completed")
        self.assertEqual(client.videos.download_content(created.id).read(), b"fixture-video-bytes")
