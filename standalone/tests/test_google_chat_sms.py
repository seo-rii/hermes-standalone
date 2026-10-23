"""Original Google Chat and Twilio SMS adapters over isolated HTTP/TLS fixtures."""

import base64
from http.server import BaseHTTPRequestHandler
import json
import re
import sys
import threading
import unittest
from urllib.parse import parse_qs, urlsplit

from standalone.tests import test_platform_sdks as platform_fixture


_PUBSUB = r'''
import threading, time, traceback
import httpx
from google.oauth2.credentials import Credentials
from google.api_core import exceptions
from standalone import google_pubsub
origin = "https://127.0.0.1:" + str(endpoint_port)
google_pubsub.API_URL = origin + "/pubsub/v1"
report = {}
def control(name, messages):
    with httpx.Client() as http:
        response = http.post(origin + "/fixture/pubsub", json={"name": name, "messages": messages})
        response.raise_for_status()
def state():
    with httpx.Client() as http:
        return http.get(origin + "/fixture/state").json()
def wait_for(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = state()
        if predicate(current): return current
        threading.Event().wait(0.05)
    raise AssertionError(state())
def message(identifier, size=70):
    import base64
    return {"ackId": identifier, "message": {"data": base64.b64encode(b"x"*size).decode(), "messageId": identifier,
            "attributes": {"ce-type": "fixture"}}, "deliveryAttempt": 2}
try:
    client = google_pubsub.SubscriberClient(credentials=Credentials("fixture-pubsub-token"))
    try:
        path = "projects/fixture/subscriptions/"
        assert client.get_subscription({"subscription": path+"retry"})["name"] == path+"retry"
        for suffix, error in (("missing", exceptions.NotFound), ("denied", exceptions.PermissionDenied)):
            try: client.get_subscription({"subscription": path+suffix})
            except error: pass
            else: raise AssertionError("Unclassified subscription failure: " + suffix)
        control("flow", [message("flow-1"), message("flow-2"), message("flow-3")])
        received, failures, first, complete = [], [], threading.Event(), threading.Event()
        def callback(item):
            try:
                received.append(item.message_id)
                assert item.data == b"x"*70 and item.delivery_attempt == 2
                if item.message_id == "flow-1":
                    received.append(item)
                    first.set()
                elif item.message_id == "flow-2":
                    assert "flow-1" in state()["acks"], "Byte budget released before acknowledgment reached server"
                    item.nack()
                else:
                    complete.set()
                    raise RuntimeError("Fixture callback failure must nack")
            except BaseException as exc:
                if item.message_id != "flow-3": failures.append(str(exc))
                raise
        future = client.subscribe(path+"flow", callback=callback,
            flow_control=google_pubsub.FlowControl(max_messages=2, max_bytes=100))
        assert first.wait(10)
        held = next(item for item in received if not isinstance(item, str))
        held.ack()
        assert complete.wait(10)
        wait_for(lambda value: {"flow-2", "flow-3"}.issubset(value["nacks"]))
        assert not failures, failures
        assert [value for value in received if isinstance(value, str)] == ["flow-1", "flow-2", "flow-3"]
        assert future.cancel() and future.cancelled() and future.result(10) is True
        control("heartbeat", [message("heartbeat-unacked")])
        beating = threading.Event()
        heartbeat = client.subscribe(path+"heartbeat", callback=lambda item: beating.set(),
            flow_control=google_pubsub.FlowControl(max_messages=1, max_duration_per_lease_extension=10))
        assert beating.wait(10)
        wait_for(lambda value: value["extensions"].get("heartbeat-unacked", 0) >= 2)
        heartbeat.cancel()
        assert heartbeat.result(10) is True
        control("lease", [message("lease-expired")])
        leased = threading.Event()
        lease = client.subscribe(path+"lease", callback=lambda item: leased.set(),
            flow_control=google_pubsub.FlowControl(max_messages=1, max_lease_duration=0.1))
        assert leased.wait(10)
        wait_for(lambda value: "lease-expired" in value["nacks"])
        lease.cancel()
        assert lease.result(10) is True
        control("shutdown", [message("shutdown-unacked")])
        pending = threading.Event()
        shutdown = client.subscribe(path+"shutdown", callback=lambda item: pending.set())
        assert pending.wait(10)
        shutdown.cancel()
        assert shutdown.result(10) is True
        assert "shutdown-unacked" in state()["nacks"]
        for invalid in ("https://attacker.invalid/subscription", "projects/p/subscriptions/s?x=1"):
            try: client.get_subscription({"subscription": invalid})
            except ValueError: pass
            else: raise AssertionError("Unsafe subscription accepted")
    finally:
        client.close()
    assert not any(thread.name == "hermes-pubsub-pull" and thread.is_alive() for thread in threading.enumerate())
    assert "google.cloud.pubsub_v1" not in sys.modules
    report.update(rest_pull=True, auth=True, typed_errors=True, retry=True, ack_nack=True,
                  callback_failure=True, flow_control=True, heartbeat=True, lease_expiry=True, shutdown=True)
except BaseException:
    report["error"] = traceback.format_exc()
report["blocked_events"] = guard.events
print(json.dumps(report), flush=True)
os._exit(0)
'''


_CHAT = r'''
import asyncio, base64, time, traceback
import httpx, jwt
from aiohttp import web
from googleapiclient import discovery, discovery_cache
from google.auth import _regional_access_boundary_utils
from google.oauth2 import credentials as user_credentials, id_token
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from plugins.platforms.google_chat import adapter as chat, oauth
from standalone import google_pubsub
from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
from agent.secret_scope import get_secret, set_secret_scope, reset_secret_scope
origin = "https://127.0.0.1:" + str(endpoint_port)
google_pubsub.API_URL = origin + "/pubsub/v1"
_regional_access_boundary_utils._get_domain = lambda: "127.0.0.1:" + str(endpoint_port)
assert chat.check_google_chat_requirements()
assert chat.pubsub_v1 is google_pubsub
chat.httplib2.CA_CERTS = str(scratch / "cert.pem")
document = json.loads(discovery_cache.get_static_doc("chat", "v1"))
document.update(rootUrl=origin+"/", mtlsRootUrl=origin+"/", baseUrl=origin+"/v1/")
def service(api, version, **kwargs):
    assert (api, version) == ("chat", "v1")
    return discovery.build_from_document(document, credentials=kwargs["credentials"])
chat.build_service = discovery.build = service
id_token._GOOGLE_OAUTH2_CERTS_URL = origin + "/google-certs"
user_credentials._GOOGLE_OAUTH2_TOKEN_ENDPOINT = origin + "/oauth2/token"
info = {"type": "service_account", "project_id": "fixture", "private_key_id": "fixture-google-key",
        "private_key": (scratch / "key.pem").read_text(), "client_email": "fixture@fixture.iam.gserviceaccount.com",
        "token_uri": origin + "/oauth2/token"}
def envelope(identifier, text):
    return {"type": "MESSAGE", "message": {"name": "spaces/fixture/messages/"+identifier,
        "sender": {"name": "users/fixture-user", "email": "person@example.invalid", "type": "HUMAN"},
        "text": text, "thread": {"name": "spaces/fixture/threads/fixture-thread"}},
        "space": {"name": "spaces/fixture", "spaceType": "DIRECT_MESSAGE"}}
report = {}
async def exercise():
    profile_home = scratch / "home/secondary-chat"
    profile_home.mkdir()
    home_token = set_hermes_home_override(profile_home)
    secret_token = set_secret_scope({"GOOGLE_CHAT_FIXTURE_SCOPE": "secondary"}, profile_home=str(profile_home))
    adapter = chat.GoogleChatAdapter(PlatformConfig(enabled=True, extra={"project_id": "fixture",
        "subscription_name": "projects/fixture/subscriptions/chat", "service_account_json": json.dumps(info)}))
    received, ready = [], asyncio.Event()
    async def on_message(event):
        received.append(event)
        assert get_hermes_home() == profile_home
        assert get_secret("GOOGLE_CHAT_FIXTURE_SCOPE") == "secondary"
        ready.set()
    adapter.set_message_handler(on_message)
    async with httpx.AsyncClient() as http:
        data = envelope("pubsub-inbound", "Google PubSub inbound")
        records = [{"ackId": "chat-"+str(i), "message": {"messageId": str(i), "data": base64.b64encode(json.dumps(item).encode()).decode()}}
                   for i, item in enumerate((data, data, {"unknown": True}))]
        records.append({"ackId": "chat-invalid-json", "message": {"data": base64.b64encode(b"not json").decode()}})
        await http.post(origin + "/fixture/pubsub", json={"name": "chat", "messages": records})
    try:
        assert not oauth._missing_required_packages(), oauth._missing_required_packages()
        assert await adapter.connect(), getattr(adapter, "_fatal_error_message", "connect failed")
        await asyncio.wait_for(ready.wait(), 10)
        sent = await adapter.send("spaces/fixture", "Google outbound", metadata={"thread_id": "spaces/fixture/threads/reply"})
        assert sent.success and sent.message_id == "spaces/fixture/messages/sent", sent
        # The real per-user OAuth refresh and media-upload client remains embedded.
        token_path = oauth._token_path("person@example.invalid")
        oauth._write_private_json(token_path, {"type": "authorized_user", "client_id": "fixture-user-client",
            "client_secret": "fixture-user-secret", "refresh_token": "fixture-user-refresh", "token": "expired",
            "expiry": "2000-01-01T00:00:00Z", "scopes": oauth.SCOPES})
        upload = scratch / "work/fixture.txt"
        upload.write_text("Google native attachment")
        delivered = await adapter.send_document("spaces/fixture", str(upload), caption="Native attachment")
        assert delivered.success, delivered
        assert json.loads(token_path.read_text())["token"] == "fixture-user-access"
        assert token_path.stat().st_mode & 0o777 == 0o600
        await adapter.disconnect()
        assert len(received) == 1, received
        # HTTP ingress runs the original API-server callback and original Google
        # ID-token certificate verification, with only the cert endpoint rewired.
        callback = chat.GoogleChatAdapter(PlatformConfig(enabled=True, extra={"service_account_json": json.dumps(info),
            "http_events_url": "https://chat.example.invalid/events", "http_events_audience": "fixture-chat-audience",
            "http_events_service_account_email": "chat-api-push@system.gserviceaccount.com"}))
        http_received = []
        async def receive_http(event): http_received.append(event)
        callback.set_message_handler(receive_http)
        assert await callback.connect()
        api = APIServerAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1"}))
        app = web.Application()
        app["platform_event_adapters"] = {"google_chat": callback}
        app.router.add_post("/api/platforms/{platform}/events", api._handle_platform_event_callback)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            endpoint = "http://127.0.0.1:" + str(runner.addresses[0][1]) + "/api/platforms/google_chat/events"
            now = int(time.time())
            claims = {"iss": "https://accounts.google.com", "aud": "fixture-chat-audience", "iat": now, "exp": now+3600,
                      "email": "chat-api-push@system.gserviceaccount.com"}
            def signed(values): return jwt.encode(values, info["private_key"], algorithm="RS256", headers={"kid": "fixture-google-key"})
            async with httpx.AsyncClient() as http:
                payload = envelope("http-inbound", "Google HTTP inbound")
                missing = await http.post(endpoint, json=payload)
                assert missing.status_code == 401, missing.text
                for changed in ({"aud": "other-audience"}, {"iss": "https://attacker.invalid"}, {"exp": now-100}, {"email": "other@example.invalid"}):
                    rejected = await http.post(endpoint, json=payload, headers={"Authorization": "Bearer " + signed({**claims, **changed})})
                    assert rejected.status_code == 401, rejected.text
                headers = {"Authorization": "Bearer " + signed(claims)}
                accepted = await http.post(endpoint, json=payload, headers=headers)
                assert accepted.status_code == 200, accepted.text
                duplicate = await http.post(endpoint, json=payload, headers=headers)
                assert duplicate.status_code == 200
                assert len(http_received) == 1 and http_received[0].text == "Google HTTP inbound", http_received
                assert http_received[0].source.user_id == "person@example.invalid"
        finally:
            await runner.cleanup()
            await callback.disconnect()
        assert "google.cloud.pubsub_v1" not in sys.modules
        report.update(original_pubsub_ingress=True, profile_scope=True, duplicate_filter=True,
            original_chat_sdk=True, service_account_auth=True, user_oauth_refresh=True, native_attachment=True,
            signed_http_ingress=True, invalid_tokens_rejected=True)
    finally:
        await adapter.disconnect()
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
try:
    asyncio.run(exercise())
except BaseException:
    report["error"] = traceback.format_exc()
report["blocked_events"] = guard.events
print(json.dumps(report), flush=True)
os._exit(0)
'''


_SMS = r'''
import asyncio, base64, hashlib, hmac, traceback, urllib.parse
import httpx
from gateway.config import PlatformConfig
from plugins.platforms.sms import adapter as sms
from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_active
origin = "https://127.0.0.1:" + str(endpoint_port)
sms.TWILIO_API_BASE = origin + "/2010-04-01/Accounts"
report = {}
async def exercise():
    os.environ.update(TWILIO_ACCOUNT_SID="foreign-account", TWILIO_AUTH_TOKEN="foreign-token", TWILIO_PHONE_NUMBER="+19990000000")
    set_multiplex_active(True)
    scope = set_secret_scope({"TWILIO_ACCOUNT_SID": "ACfixture", "TWILIO_AUTH_TOKEN": "fixture-token",
        "TWILIO_PHONE_NUMBER": "+15550000001", "SMS_WEBHOOK_URL": "https://public.example.invalid/webhooks/twilio",
        "SMS_WEBHOOK_HOST": "127.0.0.1", "SMS_WEBHOOK_PORT": "8787"})
    adapter = sms.SmsAdapter(PlatformConfig(enabled=True))
    adapter._webhook_port = 0
    received, ready = [], asyncio.Event()
    async def inbound(event):
        received.append(event)
        ready.set()
    adapter.set_message_handler(inbound)
    try:
        assert sms.check_sms_requirements()
        assert await adapter.connect()
        endpoint = "http://127.0.0.1:" + str(adapter._runner.addresses[0][1]) + "/webhooks/twilio"
        params = {"From": "+15550000002", "To": "+15550000001", "Body": "SMS inbound 한글", "MessageSid": "SMfixture-inbound"}
        def signed(values):
            clear = adapter._webhook_url + "".join(k+values[k] for k in sorted(values))
            return base64.b64encode(hmac.new(b"fixture-token", clear.encode(), hashlib.sha1).digest()).decode()
        async with httpx.AsyncClient() as http:
            for signature in (None, "invalid"):
                headers = {} if signature is None else {"X-Twilio-Signature": signature}
                denied = await http.post(endpoint, data=params, headers=headers)
                assert denied.status_code == 403 and not received, denied.text
            accepted = await http.post(endpoint, data=params, headers={"X-Twilio-Signature": signed(params)})
            assert accepted.status_code == 200 and "<Response" in accepted.text, accepted.text
            await asyncio.wait_for(ready.wait(), 10)
            assert received[0].text == params["Body"] and received[0].source.chat_id == params["From"], received
            echo = {**params, "From": "+15550000001", "MessageSid": "SMfixture-echo"}
            assert (await http.post(endpoint, data=echo, headers={"X-Twilio-Signature": signed(echo)})).status_code == 200
            oversized = await http.post(endpoint, content=b"Body=" + b"x"*(sms._TWILIO_WEBHOOK_MAX_BODY_BYTES+1))
            assert oversized.status_code == 413, oversized.text
        assert len(received) == 1
        sent = await adapter.send("+15550000002", "**" + "x"*3400 + "**")
        assert sent.success and sent.message_id.startswith("SMfixture-"), sent
        failed = await adapter.send("+19990000001", "rejected destination")
        assert not failed.success and "Twilio 400: fixture rejected destination" in failed.error, failed
        standalone = await sms._standalone_send(PlatformConfig(enabled=True), "+15550000002", "**Standalone SMS**")
        assert standalone["success"], standalone
        missing_scope = set_secret_scope({})
        try:
            assert not sms.check_sms_requirements()
            refused = await sms._standalone_send(PlatformConfig(enabled=True), "+15550000002", "must not borrow")
            assert not refused.get("success", False) and "not configured" in refused["error"], refused
        finally:
            reset_secret_scope(missing_scope)
        assert "twilio" not in sys.modules
        report.update(original_sms_ingress=True, signature_validation=True, oversized_body=True,
                      outbound=True, chunking=True, vendor_error=True, standalone_send=True, profile_scope=True)
    finally:
        await adapter.disconnect()
        reset_secret_scope(scope)
        set_multiplex_active(False)
try:
    asyncio.run(exercise())
except BaseException:
    report["error"] = traceback.format_exc()
report["blocked_events"] = guard.events
print(json.dumps(report), flush=True)
os._exit(0)
'''


class _GoogleSMSHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    def do_PATCH(self):
        self._dispatch()

    def _dispatch(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        content_type = self.headers.get("Content-Type", "")
        if "application/json" in content_type:
            body = json.loads(raw) if raw else {}
        elif "application/x-www-form-urlencoded" in content_type:
            body = {key: values[0] for key, values in parse_qs(raw.decode(), keep_blank_values=True).items()}
        else:
            body = {"raw_bytes": len(raw), "contains_attachment": b"Google native attachment" in raw}
        self.server.records.append({"method": self.command, "path": self.path,
                                    "headers": {key.lower(): value for key, value in self.headers.items()}, "body": body})
        state = self.server.__dict__.setdefault("pubsub_state", {
            "lock": threading.Lock(), "messages": {}, "acks": [], "nacks": [], "pulls": {}, "extensions": {}, "retries": 0})
        path, status = urlsplit(self.path).path, 200
        with state["lock"]:
            if path == "/fixture/pubsub":
                state["messages"][body["name"]] = list(body["messages"])
                payload = {"ok": True}
            elif path == "/fixture/state":
                payload = {key: state[key] for key in ("acks", "nacks", "pulls", "extensions")}
            elif path == "/google-certs":
                payload = {"fixture-google-key": self.server.certificate}
            elif path.startswith("/v1/projects/-/serviceAccounts/") and path.endswith("/allowedLocations"):
                payload = {"locations": [], "encodedLocations": "0x0"}
            elif path == "/oauth2/token":
                token = "fixture-user-access" if body.get("grant_type") == "refresh_token" else "fixture-google-access"
                payload = {"access_token": token, "token_type": "Bearer", "expires_in": 3600}
            elif path.startswith("/pubsub/v1/projects/fixture/subscriptions/"):
                name, _, action = path.rsplit("/", 1)[-1].partition(":")
                if self.headers.get("Authorization") not in ("Bearer fixture-pubsub-token", "Bearer fixture-google-access"):
                    status, payload = 401, {"error": {"message": "fixture authentication failure"}}
                elif name in ("missing", "denied"):
                    status, payload = (404 if name == "missing" else 403), {"error": {"message": "fixture subscription error"}}
                elif name == "retry" and not state["retries"]:
                    state["retries"] += 1
                    status, payload = 503, {"error": {"message": "fixture transient failure"}}
                elif action == "pull":
                    state["pulls"][name] = state["pulls"].get(name, 0) + 1
                    messages = state["messages"].setdefault(name, [])
                    count = body["maxMessages"]
                    payload = {"receivedMessages": messages[:count]}
                    del messages[:count]
                elif action == "acknowledge":
                    state["acks"].extend(body["ackIds"])
                    payload = {}
                elif action == "modifyAckDeadline":
                    if body["ackDeadlineSeconds"] == 0:
                        state["nacks"].extend(body["ackIds"])
                    else:
                        for ack_id in body["ackIds"]:
                            state["extensions"][ack_id] = state["extensions"].get(ack_id, 0) + 1
                    payload = {}
                else:
                    payload = {"name": "projects/fixture/subscriptions/"+name, "ackDeadlineSeconds": 10}
            elif path.startswith("/2010-04-01/Accounts/"):
                if body.get("To") == "+19990000001":
                    status, payload = 400, {"code": 21211, "message": "fixture rejected destination"}
                else:
                    payload = {"sid": "SMfixture-"+str(len(self.server.records)), "status": "queued"}
            elif path.startswith("/upload/"):
                payload = {"attachmentDataRef": {"resourceName": "spaces/fixture/attachments/fixture-upload"}}
            elif path.startswith("/v1/spaces/") and "/messages" in path:
                payload = {"name": "spaces/fixture/messages/sent", "thread": {"name": "spaces/fixture/threads/reply"}}
            else:
                status, payload = 404, {"error": {"message": "Unknown fixture path"}}
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@unittest.skipUnless(sys.platform.startswith("linux") and sys.version_info >= (3, 14),
                     "Original Google Chat/SMS integration currently verified on Linux/Python 3.14")
class OriginalGoogleChatSMSIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        platform_fixture.OriginalPlatformSDKTests.setUpClass.__func__(cls)

    def invoke(self, program):
        certificate = (self.tls / "cert.pem").read_text()
        class Handler(_GoogleSMSHandler):
            def setup(self):
                self.server.certificate = certificate
                super().setup()
        return platform_fixture.OriginalPlatformSDKTests.invoke(self, program, Handler)

    def test_pubsub_rest_callback_ack_nack_flow_leases_errors_and_shutdown(self):
        report, records = self.invoke(_PUBSUB)
        self.assertTrue(all(report[name] for name in ("rest_pull", "auth", "typed_errors", "retry", "ack_nack",
                                                     "callback_failure", "flow_control", "heartbeat", "lease_expiry", "shutdown")))
        calls = [row for row in records if row["path"].startswith("/pubsub/v1/")]
        self.assertTrue(all(row["headers"]["authorization"] == "Bearer fixture-pubsub-token" for row in calls))
        self.assertEqual(len([row for row in calls if row["path"].endswith("/retry")]), 2)
        pulls = [row for row in calls if row["path"].endswith("/flow:pull")]
        self.assertEqual(pulls[0]["body"], {"maxMessages": 2})
        self.assertTrue(any(row["body"].get("ackDeadlineSeconds") == 60 for row in calls))

    def test_original_google_chat_pubsub_service_account_user_oauth_and_signed_http(self):
        report, records = self.invoke(_CHAT)
        self.assertTrue(all(report[name] for name in ("original_pubsub_ingress", "profile_scope", "duplicate_filter",
            "original_chat_sdk", "service_account_auth", "user_oauth_refresh", "native_attachment",
            "signed_http_ingress", "invalid_tokens_rejected")))
        token = next(row for row in records if row["path"] == "/oauth2/token" and "assertion" in row["body"])
        self.assertEqual(token["body"]["grant_type"], "urn:ietf:params:oauth:grant-type:jwt-bearer")
        claims = json.loads(base64.urlsafe_b64decode(token["body"]["assertion"].split(".")[1] + "=="))
        self.assertEqual(claims["iss"], "fixture@fixture.iam.gserviceaccount.com")
        self.assertIn("https://www.googleapis.com/auth/pubsub", claims["scope"])
        sent = next(row for row in records if row["body"].get("text") == "Google outbound")
        self.assertEqual(sent["headers"]["authorization"], "Bearer fixture-google-access")
        self.assertEqual(sent["body"]["thread"], {"name": "spaces/fixture/threads/reply"})
        refresh = next(row for row in records if row["body"].get("grant_type") == "refresh_token")
        self.assertEqual(refresh["body"]["refresh_token"], "fixture-user-refresh")
        upload = next(row for row in records if row["path"].startswith("/upload/"))
        self.assertTrue(upload["body"]["contains_attachment"])
        self.assertEqual(upload["headers"]["authorization"], "Bearer fixture-user-access")
        self.assertEqual(len([row for row in records if row["path"] == "/google-certs"]), 1)

    def test_original_sms_signed_ingress_chunking_errors_and_scoped_standalone_send(self):
        report, records = self.invoke(_SMS)
        self.assertTrue(all(report[name] for name in ("original_sms_ingress", "signature_validation", "oversized_body",
                                                     "outbound", "chunking", "vendor_error", "standalone_send", "profile_scope")))
        sent = [row for row in records if row["path"] == "/2010-04-01/Accounts/ACfixture/Messages.json"]
        chunks = [row["body"]["Body"] for row in sent if row["body"]["Body"].startswith("x")]
        self.assertEqual("".join(re.sub(r" \(\d+/\d+\)$", "", chunk) for chunk in chunks), "x"*3400)
        self.assertTrue(all(len(chunk) <= 1600 for chunk in chunks))
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(row["body"]["From"] == "+15550000001" for row in sent))
        self.assertTrue(all(row["headers"]["authorization"] == "Basic " + base64.b64encode(b"ACfixture:fixture-token").decode() for row in sent))
        self.assertTrue(any(row["body"]["Body"] == "Standalone SMS" for row in sent))


if __name__ == "__main__":
    unittest.main()
