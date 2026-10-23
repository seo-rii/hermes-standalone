"""Unmodified pure platform SDKs use real TLS/HTTP and webhook ingress."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from urllib.parse import parse_qs


_CERTIFICATE = r'''
import base64, pathlib, sys
sys.path.insert(0, VENDOR_PATH)
from tlslite.utils.keyfactory import parsePEMKey
root = pathlib.Path(FIXTURE_PATH)
key = parsePEMKey((root / "key.pem").read_text(), private=True, implementations=["python"])
def tlv(tag, data):
    size = len(data)
    length = bytes([size]) if size < 128 else bytes([128 + (size.bit_length() + 7) // 8]) + size.to_bytes((size.bit_length() + 7) // 8, "big")
    return bytes([tag]) + length + data
def sequence(*values): return tlv(0x30, b"".join(values))
def integer(number):
    data = number.to_bytes(max(1, (number.bit_length() + 7) // 8), "big")
    return tlv(2, b"\0" + data if data[0] & 128 else data)
algorithm = sequence(bytes.fromhex("06092a864886f70d01010b"), b"\x05\x00")
name = sequence(tlv(0x31, sequence(bytes.fromhex("0603550403"), tlv(12, b"localhost"))))
spki = sequence(sequence(bytes.fromhex("06092a864886f70d010101"), b"\x05\x00"),
                tlv(3, b"\0" + sequence(integer(key.n), integer(key.e))))
constraints = sequence(bytes.fromhex("0603551d13"), b"\x01\x01\xff", tlv(4, sequence(b"\x01\x01\xff")))
names = sequence(bytes.fromhex("0603551d11"), tlv(4, sequence(tlv(0x87, bytes([127, 0, 0, 1])), tlv(0x82, b"localhost"))))
tbs = sequence(tlv(0xa0, integer(2)), integer(47), algorithm, name,
    sequence(tlv(0x17, b"200101000000Z"), tlv(0x18, b"20990101000000Z")), name, spki, tlv(0xa3, sequence(constraints, names)))
certificate = sequence(tbs, algorithm, tlv(3, b"\0" + bytes(key.hashAndSign(tbs, hAlg="sha256"))))
text = base64.b64encode(certificate).decode()
(root / "cert.pem").write_text("-----BEGIN CERTIFICATE-----\n" + "\n".join(text[i:i+64] for i in range(0, len(text), 64)) + "\n-----END CERTIFICATE-----\n")
import json
def unsigned(number): return base64.urlsafe_b64encode(number.to_bytes((number.bit_length()+7)//8, "big")).decode().rstrip("=")
(root / "jwk.json").write_text(json.dumps({"kty": "RSA", "n": unsigned(key.n), "e": unsigned(key.e), "kid": "fixture-teams-key", "alg": "RS256", "use": "sig"}))
'''

_FEISHU = r'''
import asyncio, hashlib, time, traceback
from gateway.config import PlatformConfig
from plugins.platforms.feishu import adapter as feishu
import httpx
report = {}
async def exercise():
    assert feishu.feishu_deps_present() and feishu._load_lark_oapi()
    feishu.FEISHU_DOMAIN = "https://127.0.0.1:" + str(endpoint_port)
    adapter = feishu.FeishuAdapter(PlatformConfig(enabled=True, extra={
        "app_id": "fixture-feishu-app", "app_secret": "fixture-feishu-secret", "connection_mode": "webhook",
        "verification_token": "fixture-verification", "webhook_host": "127.0.0.1", "webhook_port": 8787,
        "text_batch_window_seconds": 0, "text_batch_max_messages": 1,
    }))
    adapter._webhook_port = 0
    received, delivered = [], asyncio.Event()
    async def on_message(event):
        received.append(event)
        delivered.set()
    adapter.set_message_handler(on_message)
    assert await adapter.connect(), adapter._fatal_error_message
    try:
        assert adapter._bot_open_id == "ou-fixture-bot"
        address = adapter._webhook_runner.addresses[0]
        endpoint = "http://127.0.0.1:" + str(address[1]) + adapter._webhook_path
        async with httpx.AsyncClient(timeout=10) as client:
            rejected = await client.post(endpoint, json={"type": "url_verification", "token": "other", "challenge": "challenge"})
            assert rejected.status_code == 401
            challenge = await client.post(endpoint, json={"type": "url_verification", "token": "fixture-verification", "challenge": "challenge"})
            assert challenge.json() == {"challenge": "challenge"}
            message = {"schema": "2.0", "header": {"event_id": "fixture-feishu-event", "event_type": "im.message.receive_v1", "token": "fixture-verification"},
                "event": {"sender": {"sender_id": {"open_id": "ou-fixture-user", "union_id": "on-fixture-user"}, "sender_type": "user", "tenant_key": "fixture-tenant"},
                    "message": {"message_id": "om-fixture-inbound", "chat_id": "oc-fixture-chat", "chat_type": "p2p", "message_type": "text", "content": json.dumps({"text": "Feishu inbound"}), "create_time": str(int(time.time()*1000))}}}
            accepted = await client.post(endpoint, json=message)
            assert accepted.status_code == 200 and accepted.json()["code"] == 0, accepted.text
            await asyncio.wait_for(delivered.wait(), 10)
            assert received[0].text == "Feishu inbound" and received[0].source.user_id == "ou-fixture-user", received
            assert received[0].source.user_id_alt == "on-fixture-user"
            await client.post(endpoint, json=message)
        sent = await adapter.send("oc-fixture-chat", "Feishu outbound")
        assert sent.success and sent.message_id == "om-fixture-sent", sent
        await asyncio.sleep(0.1)
        assert len(received) == 1, received
        # The native AES dependency has been replaced only in its real SDK decryptor.
        from lark_oapi.core.utils.decryptor import AESCipher
        from standalone.crypto_backend import aes_cbc_encrypt
        key = hashlib.sha256(b"fixture-encrypt-key").digest()
        clear = b"encrypted Feishu event"
        pad = 16-len(clear)%16
        cipher = bytes(range(16)) + aes_cbc_encrypt(key, bytes(range(16)), clear + bytes([pad])*pad)
        assert AESCipher("fixture-encrypt-key").decrypt(cipher) == clear
        for invalid in (b"short", cipher[:-1]):
            try: AESCipher("fixture-encrypt-key").decrypt(invalid)
            except ValueError: pass
            else: raise AssertionError("Invalid Feishu ciphertext accepted")
        assert not any(name == "Crypto" or name.startswith("Crypto.") for name in sys.modules)
        report.update(feishu_ingress=True, feishu_send=True, dedup=True, aes_boundary=True)
    finally:
        await adapter.disconnect()
try:
    asyncio.run(exercise())
except BaseException:
    report["error"] = traceback.format_exc()
report["blocked_events"] = guard.events
print(json.dumps(report), flush=True)
os._exit(0)
'''

_TEAMS = r'''
import asyncio, time, traceback
import httpx, jwt
from gateway.config import PlatformConfig
from plugins.platforms.teams import adapter as teams
from standalone import platform_teams
origin = "https://127.0.0.1:" + str(endpoint_port)
platform_teams.JWKS_URL = origin + "/jwks"
platform_teams.AUTHORITY_URL = origin
platform_teams.SERVICE_URL = origin + "/teams/"
private_key = (scratch / "key.pem").read_bytes()
report = {}
async def exercise():
    assert teams.check_requirements() and teams.check_teams_requirements()
    adapter = teams.TeamsAdapter(PlatformConfig(enabled=True, extra={
        "client_id": "fixture-teams-app", "client_secret": "fixture-teams-secret", "tenant_id": "fixture-teams-tenant", "host": "127.0.0.1"}))
    adapter._port = 0
    received, delivered = [], asyncio.Event()
    async def on_message(event):
        received.append(event)
        delivered.set()
    adapter.set_message_handler(on_message)
    assert await adapter.connect(), getattr(adapter, "_fatal_error_message", "Teams connect failed")
    now = int(time.time())
    claims = {"aud": "fixture-teams-app", "iss": platform_teams.TOKEN_ISSUER, "iat": now, "exp": now+3600,
              "serviceurl": origin + "/teams/"}
    def signed(values):
        return jwt.encode(values, private_key, algorithm="RS256", headers={"kid": "fixture-teams-key"})
    activity = {"type": "message", "id": "teams-inbound", "serviceUrl": origin+"/teams/", "channelId": "msteams",
        "from": {"id": "29:fixture-user", "aadObjectId": "fixture-user-aad", "name": "Fixture user"},
        "recipient": {"id": "28:fixture-teams-app"}, "conversation": {"id": "19:fixture-chat@thread.tacv2", "conversationType": "personal", "tenantId": "fixture-teams-tenant"},
        "text": "Teams inbound"}
    try:
        endpoint = "http://127.0.0.1:" + str(adapter._runner.addresses[0][1]) + "/api/messages"
        async with httpx.AsyncClient(timeout=10) as client:
            denied = await client.post(endpoint, json=activity)
            assert denied.status_code == 401, denied.text
            tokens = [signed({**claims, **change}) for change in (
                {"aud": "another-app"}, {"iss": "https://attacker.invalid"}, {"exp": now-1000}, {"serviceurl": origin+"/other/"})]
            token = signed(claims)
            parts = token.split(".")
            forged = parts[0] + "." + parts[1] + "." + ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
            for bad in [*tokens, forged]:
                denied = await client.post(endpoint, json=activity, headers={"Authorization": "Bearer " + bad})
                assert denied.status_code == 401, denied.text
            assert not received
            accepted = await client.post(endpoint, json=activity, headers={"Authorization": "Bearer " + token})
            assert accepted.status_code == 200, accepted.text
            await asyncio.wait_for(delivered.wait(), 10)
            assert received[0].text == "Teams inbound" and received[0].source.user_id == "fixture-user-aad"
            await client.post(endpoint, json=activity, headers={"Authorization": "Bearer " + token})
            card = {**activity, "id": "teams-card", "type": "invoke", "name": "adaptiveCard/action",
                    "value": {"action": {"type": "Action.Execute", "data": {"hermes_action": "approve_once", "session_key": "fixture-session", "opaque": None}}}}
            denied_card = await client.post(endpoint, json=card, headers={"Authorization": "Bearer " + token})
            assert denied_card.status_code == 200 and denied_card.json()["type"] == "application/vnd.microsoft.activity.message", denied_card.text
            os.environ["TEAMS_ALLOWED_USERS"] = "fixture-user-aad"
            allowed_card = await client.post(endpoint, json=card, headers={"Authorization": "Bearer " + token})
            assert allowed_card.status_code == 200 and allowed_card.json()["type"] == "application/vnd.microsoft.card.adaptive", allowed_card.text
        assert len(received) == 1
        sent = await adapter.send("19:fixture-chat@thread.tacv2", "Teams outbound")
        assert sent.success and sent.message_id == "teams-fixture-sent", sent
        await adapter.send_typing("19:fixture-chat@thread.tacv2")
        media = await adapter.send_image("19:fixture-chat@thread.tacv2", "https://8.8.8.8/image.png", caption="Fixture caption")
        assert media.success, media
        assert not any(name == "microsoft_teams" or name.startswith("microsoft_teams.") or name == "pydantic" or name.startswith("dependency_injector") for name in sys.modules)
        report.update(teams_ingress=True, teams_send=True, jwt_refusals=True, approval_gate=True, dedup=True)
    finally:
        await adapter.disconnect()
try:
    asyncio.run(exercise())
except BaseException:
    report["error"] = traceback.format_exc()
report["blocked_events"] = guard.events
print(json.dumps(report), flush=True)
os._exit(0)
'''

_DINGTALK = r'''
import asyncio, time, traceback
import httpx
from websockets.asyncio.server import serve
from gateway.config import PlatformConfig
from plugins.platforms.dingtalk import adapter as dingtalk
import dingtalk_stream
import dingtalk_stream.stream as stream_module
from standalone import platform_dingtalk
origin = "https://127.0.0.1:" + str(endpoint_port)
dingtalk_stream.DingTalkStreamClient.OPEN_CONNECTION_API = origin + "/dingtalk/open"
stream_module.DINGTALK_OPENAPI_ENDPOINT = origin
platform_dingtalk.API_URL = origin
# The SDK's optional UDP local-address discovery is fixture metadata only.
dingtalk_stream.DingTalkStreamClient.get_host_ip = lambda self: "127.0.0.1"
dingtalk._DINGTALK_WEBHOOK_RE = re.compile("^" + re.escape(origin) + "/")
report = {}
async def exercise():
    assert dingtalk.dingtalk_deps_present()
    adapter = dingtalk.DingTalkAdapter(PlatformConfig(enabled=True, extra={
        "client_id": "fixture-dingtalk-app", "client_secret": "fixture-dingtalk-secret", "require_mention": True,
        "card_template_id": "fixture-card-template"}))
    received, delivered, acknowledged, release = [], asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def on_message(event):
        received.append(event)
        delivered.set()
    adapter.set_message_handler(on_message)
    async def stream(connection):
        assert connection.request.path == "/stream?ticket=fixture-ticket", connection.request.path
        payload = {"msgId": "ding-inbound", "conversationId": "ding-chat", "conversationType": "1",
            "senderId": "ding-user", "senderStaffId": "ding-user-staff", "senderNick": "Fixture user",
            "msgtype": "text", "text": {"content": "DingTalk inbound"}, "sessionWebhook": origin + "/dingtalk/reply",
            "sessionWebhookExpiredTime": int((time.time()+3600)*1000), "createAt": int(time.time()*1000)}
        frame = {"specVersion": "1.0", "type": "CALLBACK", "headers": {"messageId": "ding-frame",
            "topic": dingtalk_stream.ChatbotMessage.TOPIC, "contentType": "application/json"}, "data": json.dumps(payload)}
        for _ in range(2):
            await connection.send(json.dumps(frame))
            ack = json.loads(await asyncio.wait_for(connection.recv(), 5))
            assert ack["code"] == 200, ack
        acknowledged.set()
        await release.wait()
    server = await serve(stream, "127.0.0.1", 0)
    try:
        ws_url = "ws://127.0.0.1:" + str(server.sockets[0].getsockname()[1]) + "/stream"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(origin + "/dingtalk/register", json={"endpoint": ws_url})
            assert response.status_code == 200
        assert await adapter.connect()
        await asyncio.wait_for(acknowledged.wait(), 10)
        await asyncio.wait_for(delivered.wait(), 10)
        assert len(received) == 1 and received[0].text == "DingTalk inbound", received
        assert received[0].source.user_id == "ding-user" and received[0].source.user_id_alt == "ding-user-staff"
        result = await adapter.send("ding-chat", "DingTalk card draft")
        assert result.success and result.message_id.startswith("hermes_"), result
        final = await adapter.edit_message("ding-chat", result.message_id, "DingTalk card final", finalize=True)
        assert final.success, final
        target = {"downloadCode": "fixture-download-code"}
        await adapter._fetch_download_url("fixture-download-code", "fixture-dingtalk-app",
            await adapter._get_access_token(), target, "downloadCode")
        assert target["downloadCode"] == "https://8.8.8.8/dingtalk-media", target
        # A rejected card follows the original adapter's webhook fallback.
        adapter._card_template_id = "reject-card"
        result = await adapter.send("ding-chat", "DingTalk outbound")
        assert result.success, result
        if adapter._bg_tasks:
            await asyncio.gather(*tuple(adapter._bg_tasks), return_exceptions=True)
        report.update(dingtalk_stream=True, dingtalk_send=True, sdk_ack=True, dedup=True, cards=True, media_url=True)
    finally:
        await adapter.disconnect()
        release.set()
        server.close()
        await server.wait_closed()
try:
    asyncio.run(exercise())
except BaseException:
    report["error"] = traceback.format_exc()
report["blocked_events"] = guard.events
print(json.dumps(report), flush=True)
os._exit(0)
'''


class _PlatformHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.respond(None)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = {key: value[0] for key, value in parse_qs(raw.decode()).items()} if self.headers.get_content_type() == "application/x-www-form-urlencoded" else json.loads(raw)
        self.respond(body)

    do_PUT = do_POST

    def respond(self, body):
        self.server.records.append({"method": self.command, "path": self.path, "body": body, "headers": dict(self.headers)})
        status = 200
        if self.path == "/dingtalk/register":
            self.server.dingtalk_endpoint = body["endpoint"]
            payload = {"ok": True}
        elif self.path == "/dingtalk/open":
            payload = {"endpoint": self.server.dingtalk_endpoint, "ticket": "fixture-ticket"}
        elif self.path == "/dingtalk/reply":
            payload = {"errcode": 0, "errmsg": "ok"}
        elif self.path == "/v1.0/oauth2/accessToken":
            payload = {"accessToken": "fixture-dingtalk-token", "expireIn": 7200}
        elif self.path == "/v1.0/robot/messageFiles/download":
            payload = {"downloadUrl": "https://8.8.8.8/dingtalk-media"}
        elif self.path == "/v1.0/card/instances" and body.get("cardTemplateId") == "reject-card":
            status, payload = 429, {"code": "Throttling", "message": "fixture rejected card", "requestid": "fixture-card-request"}
        elif self.path == "/jwks":
            payload = {"keys": [self.server.jwk]}
        elif self.path == "/fixture-teams-tenant/oauth2/v2.0/token":
            payload = {"access_token": "fixture-bot-connector-token", "expires_in": 3600, "token_type": "Bearer"}
        elif self.path.startswith("/teams/v3/conversations/"):
            payload = {"id": "teams-fixture-sent"}
        elif self.path == "/open-apis/auth/v3/tenant_access_token/internal":
            payload = {"code": 0, "tenant_access_token": "fixture-tenant-token", "expire": 7200}
        elif self.path == "/open-apis/bot/v3/info":
            payload = {"code": 0, "bot": {"open_id": "ou-fixture-bot", "app_name": "Fixture bot"}}
        elif self.path.startswith("/open-apis/contact/v3/users/"):
            payload = {"code": 0, "data": {"user": {"name": "Fixture user"}}}
        elif self.path.startswith("/open-apis/im/v1/chats/"):
            payload = {"code": 0, "data": {"chat_id": "oc-fixture-chat", "name": "Fixture chat", "chat_type": "p2p"}}
        elif self.path.startswith("/open-apis/im/v1/messages/") and self.path.endswith("/reactions"):
            payload = {"code": 0, "data": {"reaction_id": "fixture-reaction"}}
        elif self.path.startswith("/open-apis/im/v1/messages"):
            payload = {"code": 0, "data": {"message_id": "om-fixture-sent"}}
        else:
            payload = {"code": 0, "data": {}}
        content = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


@unittest.skipUnless(sys.platform.startswith("linux") and sys.version_info >= (3, 14),
                     "Original platform SDK integration currently verified on Linux/Python 3.14")
class OriginalPlatformSDKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from standalone.tests.test_entrypoints import BundledSourceTestCase, STANDALONE
        BundledSourceTestCase.setUpClass.__func__(cls)
        from standalone import vendor
        package = next(item for item in vendor.load_dependencies(STANDALONE / "dependencies.json")["packages"] if item["name"] == "tlslite-ng")
        cls.tls = cls.directory / "tls"
        cls.tls.mkdir()
        with tarfile.open(fileobj=io.BytesIO(vendor._verified_bytes(package, STANDALONE / ".cache/downloads")), mode="r:gz") as archive:
            (cls.tls / "key.pem").write_bytes(archive.extractfile("tlslite_ng-0.8.2/tests/serverX509Key.pem").read())
        generator = _CERTIFICATE.replace("VENDOR_PATH", repr(str(cls.bundle / "vendor"))).replace("FIXTURE_PATH", repr(str(cls.tls)))
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", generator], capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise AssertionError(result.stderr)

    def invoke(self, program, handler=_PlatformHandler):
        from standalone.tests.test_chat_entry import _LAUNCHER, OriginalChatTests
        from standalone.tests.test_entrypoints import support
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads, server.records = True, []
        server.jwk = json.loads((self.tls / "jwk.json").read_text())
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(self.tls / "cert.pem", self.tls / "key.pem")
        server.socket = tls.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory(prefix="p-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            (scratch / "cert.pem").write_bytes((self.tls / "cert.pem").read_bytes())
            (scratch / "key.pem").write_bytes((self.tls / "key.pem").read_bytes())
            context = dict(self.context, artifact=str(scratch / "hermes.py"))
            prefix = _LAUNCHER.split("arguments = sys.argv[1:]")[0].replace("CONTEXT_JSON", repr(json.dumps(context))).replace(
                "SCRATCH_ROOT", repr(str(scratch))).replace("ENDPOINT_PORT", str(server.server_port))
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            environment.update(PATH="", SSL_CERT_FILE=str(scratch / "cert.pem"), REQUESTS_CA_BUNDLE=str(scratch / "cert.pem"))
            home = Path(environment["HERMES_HOME"])
            home.mkdir(parents=True, exist_ok=True)
            (home / "config.yaml").write_text(json.dumps({"agent": {"environment_probe": False},
                "tools": {"tool_search": {"enabled": False}}, "model_catalog": {"enabled": False}}))
            try:
                result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", prefix + program],
                    env=environment, cwd=scratch / "work", capture_output=True, text=True, encoding="utf-8", timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr[-5000:])
                report = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertNotIn("error", report, report.get("error"))
                self.assertTrue(all(event.get("recoverable_capability_probe") for event in report["blocked_events"]), report)
                return report, list(server.records)
            finally:
                OriginalChatTests._stop_fixture_processes(self, scratch)

    def test_feishu_original_sdk_tls_send_and_authenticated_webhook_ingress(self):
        report, records = self.invoke(_FEISHU)
        self.assertTrue(report["feishu_ingress"] and report["feishu_send"] and report["dedup"] and report["aes_boundary"])
        token = next(row for row in records if row["path"].startswith("/open-apis/auth/v3/"))
        self.assertEqual(token["body"], {"app_id": "fixture-feishu-app", "app_secret": "fixture-feishu-secret"})
        sent = next(row for row in records if row["path"].startswith("/open-apis/im/v1/messages?") and row["body"].get("receive_id"))
        self.assertEqual(sent["body"]["content"], json.dumps({"text": "Feishu outbound"}, ensure_ascii=False))
        self.assertEqual(sent["headers"]["Authorization"], "Bearer fixture-tenant-token")

    def test_teams_original_adapter_signed_jwt_ingress_cards_and_outbound_wire(self):
        report, records = self.invoke(_TEAMS)
        self.assertTrue(all(report[name] for name in ("teams_ingress", "teams_send", "jwt_refusals", "approval_gate", "dedup")))
        tokens = [row for row in records if row["path"].endswith("/oauth2/v2.0/token")]
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0]["body"], {"grant_type": "client_credentials", "client_id": "fixture-teams-app",
            "client_secret": "fixture-teams-secret", "scope": "https://api.botframework.com/.default"})
        sends = [row for row in records if row["path"].startswith("/teams/v3/conversations/")]
        self.assertTrue(any(row["body"].get("text") == "Teams outbound" for row in sends))
        self.assertTrue(any(row["body"]["type"] == "typing" for row in sends))
        attachments = next(row["body"]["attachments"] for row in sends if row["body"].get("text") == "Fixture caption")
        self.assertEqual(attachments[0], {"contentType": "image/png", "contentUrl": "https://8.8.8.8/image.png"})
        self.assertTrue(all(row["headers"]["Authorization"] == "Bearer fixture-bot-connector-token" for row in sends))

    def test_dingtalk_original_stream_sdk_callback_ack_and_webhook_reply(self):
        report, records = self.invoke(_DINGTALK)
        self.assertTrue(all(report[name] for name in ("dingtalk_stream", "dingtalk_send", "sdk_ack", "dedup", "cards", "media_url")))
        request = next(row for row in records if row["path"] == "/dingtalk/open")
        self.assertEqual(request["body"]["clientId"], "fixture-dingtalk-app")
        self.assertEqual(request["body"]["clientSecret"], "fixture-dingtalk-secret")
        self.assertEqual(request["body"]["subscriptions"], [{"type": "CALLBACK", "topic": "/v1.0/im/bot/messages/get"}])
        reply = next(row for row in records if row["path"] == "/dingtalk/reply")
        self.assertEqual(reply["body"], {"msgtype": "markdown", "markdown": {"title": "Hermes", "text": "DingTalk outbound"}})
        cards = [row for row in records if row["path"] == "/v1.0/card/instances"]
        self.assertEqual(cards[0]["body"]["cardTemplateId"], "fixture-card-template")
        self.assertEqual(cards[0]["body"]["cardData"], {"cardParamMap": {"content": ""}})
        self.assertEqual(cards[0]["headers"]["x-acs-dingtalk-access-token"], "fixture-dingtalk-token")
        stream = [row for row in records if row["path"] == "/v1.0/card/streaming"]
        self.assertTrue(all(row["method"] == "PUT" for row in stream))
        self.assertFalse(stream[0]["body"]["isFinalize"])
        self.assertTrue(stream[1]["body"]["isFinalize"])
        self.assertEqual(stream[1]["body"]["content"], "DingTalk card final")
        download = next(row for row in records if row["path"] == "/v1.0/robot/messageFiles/download")
        self.assertEqual(download["body"], {"downloadCode": "fixture-download-code", "robotCode": "fixture-dingtalk-app"})
        self.assertTrue(any(row["path"] == "/v1.0/robot/emotion/reply" for row in records))


if __name__ == "__main__":
    unittest.main()
