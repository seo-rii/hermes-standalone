"""Exercise reviewed dependency implementations without installed packages."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone import vendor


STANDALONE = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.version_info >= (3, 14), "Runtime dependency behavior targets Python 3.14")
class RuntimeDependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = vendor.load_dependencies(STANDALONE / "dependencies.json")
        cache = STANDALONE / ".cache" / "downloads"
        if any(not (cache / item["filename"]).is_file() for item in manifest["packages"]):
            raise unittest.SkipTest("Run the explicit vendor fetch command before dependency integration tests")
        cls.temporary = tempfile.TemporaryDirectory(prefix="standalone-dependency-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.library = cls.directory / "vendor"
        paths = vendor.ArchivePaths()
        for package in manifest["packages"]:
            files, _ = vendor.read_wheel(package, cache)
            for name, data in files.items():
                paths.add(name)
                path = cls.library / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)

    def run_isolated(self, source: str) -> dict:
        profile = self.directory / self._testMethodName
        profile.mkdir()
        environment = {"HOME": str(profile), "USERPROFILE": str(profile)}
        for name in ("SYSTEMROOT", "WINDIR"):
            if name in os.environ:
                environment[name] = os.environ[name]
        code = f"import sys; sys.path[:0] = {[str(self.library), str(STANDALONE.parent)]!r}\n" + source
        child = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", code],
            cwd=profile, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(child.returncode, 0, child.stderr + child.stdout)
        return json.loads(child.stdout)

    def test_python_source_fallbacks_preserve_template_yaml_and_png_behavior(self):
        report = self.run_isolated('''
import io, json
import jinja2, markupsafe, png, qrcode, yaml
from qrcode.image.pure import PyPNGImage
assert markupsafe._escape_inner.__module__ == "markupsafe._native"
assert yaml.__with_libyaml__ is False
assert jinja2.Environment(autoescape=True).from_string("{{ text }}").render(text="<tag>&") == "&lt;tag&gt;&amp;"
document = {"nested": [True, 3, "한글"], "mapping": {"key": "value"}}
assert yaml.safe_load(yaml.safe_dump(document, allow_unicode=True)) == document
image = qrcode.make("offline dependency fixture", image_factory=PyPNGImage)
output = io.BytesIO()
image.save(output)
width, height, rows, metadata = png.Reader(bytes=output.getvalue()).read()
assert width == height and width > 0 and len(list(rows)) == height
assert not any(name in sys.modules for name in ("markupsafe._speedups", "yaml._yaml"))
print(json.dumps({"template_escape": True, "yaml_roundtrip": True, "png_roundtrip": True}))
''')
        self.assertEqual(report, {"template_escape": True, "yaml_roundtrip": True, "png_roundtrip": True})

    def test_aiohttp_handles_http_body_and_websocket_frames_without_native_extensions(self):
        report = self.run_isolated('''
import asyncio, json
import aiohttp
from aiohttp import web
from aiohttp.http_parser import HttpRequestParser, HttpRequestParserPy
assert HttpRequestParser is HttpRequestParserPy

async def main():
    async def echo(request):
        return web.json_response({"received": await request.json()})
    async def socket(request):
        response = web.WebSocketResponse()
        await response.prepare(request)
        incoming = await response.receive_str()
        await response.send_str(incoming + " acknowledged")
        await response.close()
        return response
    app = web.Application()
    app.router.add_post("/echo", echo)
    app.router.add_get("/socket", socket)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"http://127.0.0.1:{port}/echo", json={"value": "fixture"}) as response:
                assert response.status == 200
                assert await response.json() == {"received": {"value": "fixture"}}
            async with session.ws_connect(f"http://127.0.0.1:{port}/socket") as websocket:
                await websocket.send_str("local fixture")
                assert await websocket.receive_str() == "local fixture acknowledged"
    finally:
        await runner.cleanup()
    assert not any(name in sys.modules for name in ("aiohttp._http_parser", "multidict._multidict", "frozenlist._frozenlist", "yarl._quoting_c"))
    return {"http": True, "websocket": True, "native_extensions": False}
print(json.dumps(asyncio.run(main())))
''')
        self.assertEqual(report, {"http": True, "websocket": True, "native_extensions": False})

    def test_gateway_libraries_and_asgi_uploads_work_with_only_reviewed_packages(self):
        report = self.run_isolated('''
import asyncio, json, ssl
import fire, h2.connection, httpx, jwt, slack_bolt.async_app, telegram, truststore, uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from telegram.request import HTTPXRequest

async def upload(request):
    async with request.form() as form:
        file = form["upload"]
        return JSONResponse({"name": file.filename, "text": (await file.read()).decode()})

async def main():
    app = Starlette(routes=[Route("/upload", upload, methods=["POST"])])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://fixture.invalid") as client:
        response = await client.post("/upload", files={"upload": ("fixture.txt", b"multipart fixture")})
        assert response.status_code == 200 and response.json() == {"name": "fixture.txt", "text": "multipart fixture"}
    server = uvicorn.Config(app, http="h11", loop="asyncio", ws="websockets", lifespan="off", log_config=None)
    server.load()
    request = HTTPXRequest()
    await request.shutdown()
    return {"multipart": True, "server_config": True, "telegram_request": True}

token = jwt.encode({"subject": "offline fixture"}, "offline-fixture-key-of-at-least-32-bytes", algorithm="HS256")
assert jwt.decode(token, "offline-fixture-key-of-at-least-32-bytes", algorithms=["HS256"])["subject"] == "offline fixture"
try:
    jwt.decode(token, "incorrect-fixture-key-of-at-least-32-bytes", algorithms=["HS256"])
except jwt.InvalidSignatureError:
    pass
else:
    raise AssertionError("JWT signature checking was bypassed")
context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
assert context.verify_mode == ssl.CERT_REQUIRED
connection = h2.connection.H2Connection()
connection.initiate_connection()
assert connection.data_to_send().startswith(b"PRI * HTTP/2.0")
result = asyncio.run(main())
result.update(jwt_signature=True, system_trust=True, http2=True)
print(json.dumps(result))
''')
        self.assertTrue(all(report.values()), report)

    def test_bedrock_signed_loopback_request_and_pure_serialization_libraries(self):
        report = self.run_isolated('''
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import boto3, jmespath, msgpack
from botocore.config import Config
from google.protobuf.struct_pb2 import Struct
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

observed = []
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        observed.append((self.path, body, self.headers.get("Authorization", "")))
        reply = {"output": {"message": {"role": "assistant", "content": [{"text": "fixture answer"}]}},
                 "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3},
                 "metrics": {"latencyMs": 1}}
        content = json.dumps(reply).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content))); self.end_headers(); self.wfile.write(content)

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
client = boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="fixture-key",
                      aws_secret_access_key="fixture-secret", endpoint_url=f"http://127.0.0.1:{server.server_port}",
                      config=Config(retries={"max_attempts": 0}, proxies={}))
try:
    result = client.converse(modelId="fixture-model", messages=[{"role": "user", "content": [{"text": "offline"}]}])
    assert result["output"]["message"]["content"][0]["text"] == "fixture answer"
    assert observed[0][0] == "/model/fixture-model/converse"
    assert observed[0][2].startswith("AWS4-HMAC-SHA256 Credential=fixture-key/")
    assert observed[0][1]["messages"][0]["content"] == [{"text": "offline"}]
finally:
    client.close(); server.shutdown(); thread.join(); server.server_close()
assert jmespath.search("items[?enabled].name", {"items": [{"enabled": True, "name": "fixture"}]}) == ["fixture"]
document = {"text": "한글", "values": [1, True, None]}
assert msgpack.unpackb(msgpack.packb(document, use_bin_type=True), raw=False) == document
assert msgpack.Packer.__module__ == "msgpack.fallback"
message = Struct(); message.update(document)
assert Struct.FromString(message.SerializeToString()) == message
trace = ExportTraceServiceRequest(); trace.resource_spans.add().schema_url = "https://fixture.invalid/schema"
assert ExportTraceServiceRequest.FromString(trace.SerializeToString()) == trace
from google.protobuf.internal import api_implementation
assert api_implementation.Type() == "python"
print(json.dumps({"bedrock_http_signing": True, "jmespath": True, "msgpack": True, "protobuf": True, "otel": True}))
''')
        self.assertTrue(all(report.values()), report)

    def test_discord_gateway_message_parser_and_pcm_volume_without_native_modules(self):
        report = self.run_isolated('''
import json, struct
import discord
from standalone.pcm_audio import mul

class FixtureClient(discord.Client):
    def dispatch(self, event, *arguments, **kwargs):
        self.events.append((event, arguments))

client = FixtureClient(intents=discord.Intents.default())
client.events = []
client._connection.parse_message_create({
    "id": "100", "channel_id": "200", "type": 0, "content": "offline gateway message",
    "author": {"id": "300", "username": "fixture", "discriminator": "0001", "avatar": None},
    "attachments": [], "embeds": [], "mentions": [], "mention_roles": [],
    "timestamp": "2026-01-01T00:00:00+00:00", "edited_timestamp": None,
    "flags": 0, "tts": False, "mention_everyone": False, "pinned": False,
})
assert client.events[0][0] == "message"
message = client.events[0][1][0]
assert message.content == "offline gateway message" and message.author.name == "fixture"
fragment = struct.pack("=7h", -32768, -3, -1, 0, 1, 3, 32767)
class PCM(discord.AudioSource):
    def read(self): return fragment
source = discord.PCMVolumeTransformer(PCM(), volume=0.5)
assert struct.unpack("=7h", source.read()) == (-16384, -2, -1, 0, 0, 1, 16383)
assert struct.unpack("=7h", mul(fragment, 2, 2.0)) == (-32768, -6, -2, 0, 2, 6, 32767)
assert "audioop" not in sys.modules and "audioop_lts" not in sys.modules
print(json.dumps({"message_event": True, "pcm_volume": True, "native_audioop": False}))
''')
        self.assertEqual(report, {"message_event": True, "pcm_volume": True, "native_audioop": False})


if __name__ == "__main__":
    unittest.main()
