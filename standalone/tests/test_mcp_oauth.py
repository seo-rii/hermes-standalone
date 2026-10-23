"""Original MCP OAuth storage/policy with real local authorization servers."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from urllib.parse import parse_qs, urlencode, urlsplit


class _OAuthHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def reply(self, body, status=200, headers=None):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def record(self, body=None):
        self.server.records.append({"method": self.command, "path": self.path,
                                    "headers": {key.lower(): value for key, value in self.headers.items()}, "body": body})

    def do_GET(self):
        self.record()
        origin = self.server.origin
        path = urlsplit(self.path).path
        if path == "/mcp":
            if self.headers.get("authorization") == "Bearer " + self.server.access:
                return self.reply({"authenticated": True})
            return self.reply({}, 401, {"www-authenticate": f'Bearer resource_metadata="{origin}/resource-metadata", scope="read"'})
        if path in ("/resource-metadata", "/.well-known/oauth-protected-resource/mcp"):
            return self.reply({"resource": origin + ("/different" if self.server.bad_resource else "/mcp"),
                               "authorization_servers": [origin + "/auth"], "scopes_supported": ["read"]})
        if path == "/.well-known/oauth-authorization-server/auth":
            return self.reply(self.server.metadata())
        if path == "/auth/authorize":
            query = {name: values[0] for name, values in parse_qs(urlsplit(self.path).query).items()}
            self.server.authorization = query
            callback = {"code": "local-authorization-code", "state": query["state"], "iss": origin + "/auth"}
            if self.server.callback_mode == "state":
                callback["state"] = "different-state"
            elif self.server.callback_mode == "issuer":
                callback["iss"] = origin + "/different-issuer"
            elif self.server.callback_mode == "missing_issuer":
                callback.pop("iss")
            return self.reply({}, 302, {"location": query["redirect_uri"] + "?" + urlencode(callback)})
        return self.reply({}, 404)

    def do_POST(self):
        data = self.rfile.read(int(self.headers.get("content-length", "0")))
        body = json.loads(data) if self.headers.get("content-type", "").startswith("application/json") else {
            name: values[0] for name, values in parse_qs(data.decode()).items()}
        self.record(body)
        if self.path == "/auth/register":
            # Real providers can omit the method while returning a secret;
            # Hermes' existing coercion must still select client_secret_post.
            return self.reply({"client_id": "local-client", "client_secret": "local-client-secret"}, 201)
        if self.path == "/auth/device":
            return self.reply({"device_code": "local-device-code", "user_code": "TEST-ONLY",
                               "verification_uri": self.server.origin + "/verify", "interval": 0.01, "expires_in": 60})
        if self.path == "/auth/token-redirect":
            return self.reply({}, 307, {"location": self.server.origin + "/credential-sink"})
        if self.path == "/credential-sink":
            return self.reply({"unexpected": True})
        if self.path != "/auth/token":
            return self.reply({}, 404)
        grant = body.get("grant_type")
        if grant == "authorization_code":
            verifier = body.get("code_verifier", "")
            digest = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
            expected = self.server.authorization
            if (body.get("code") != "local-authorization-code" or digest != expected.get("code_challenge")
                    or expected.get("code_challenge_method") != "S256" or body.get("resource") != self.server.origin + "/mcp"):
                return self.reply({"error": "invalid_grant"}, 400)
        elif grant == "refresh_token":
            self.server.refresh_started.set()
            if self.server.delay_refresh:
                self.server.refresh_release.wait(3)
            if body.get("refresh_token") != self.server.refresh:
                return self.reply({"error": "invalid_grant"}, 400)
        elif grant == "urn:ietf:params:oauth:grant-type:device_code":
            self.server.device_polls += 1
            if self.server.device_polls == 1:
                return self.reply({"error": "authorization_pending"}, 400)
        else:
            return self.reply({"error": "unsupported_grant_type"}, 400)
        self.server.generation += 1
        self.server.access = "local-access-" + str(self.server.generation)
        self.server.refresh = "local-refresh-" + str(self.server.generation)
        return self.reply({"access_token": self.server.access, "refresh_token": self.server.refresh,
                           "token_type": "Bearer", "scope": "read", "expires_in": 3600}, 201)


@unittest.skipUnless(os.environ.get("HERMES_MCP_OAUTH_FIXTURE") == "1", "Runs inside the isolated OAuth fixture")
class OAuthWireTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _OAuthHandler)
        self.server.daemon_threads = True
        self.server.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.server.access, self.server.refresh = "no-access-yet", "local-refresh-0"
        self.server.generation = self.server.device_polls = 0
        self.server.records, self.server.authorization = [], {}
        self.server.callback_mode = "valid"
        self.server.bad_resource = self.server.delay_refresh = False
        self.server.refresh_started, self.server.refresh_release = threading.Event(), threading.Event()
        self.server.metadata = lambda: {
            "issuer": self.server.origin + "/auth", "authorization_endpoint": self.server.origin + "/auth/authorize",
            "token_endpoint": self.server.origin + "/auth/token", "registration_endpoint": self.server.origin + "/auth/register",
            "device_authorization_endpoint": self.server.origin + "/auth/device",
            "code_challenge_methods_supported": ["S256"], "authorization_response_iss_parameter_supported": True,
            "grant_types_supported": ["authorization_code", "refresh_token", "urn:ietf:params:oauth:grant-type:device_code"],
        }
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.refresh_release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)

    @asynccontextmanager
    async def browser_provider(self, suffix="", *, cached=False):
        import httpx
        from tools import mcp_oauth as original
        from tools.mcp_oauth_manager import HermesMCPOAuthProvider
        from tools.mcp_tool_errors import _make_redirect_header_stripper
        handler, result = original._make_callback_handler()
        callback = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=callback.serve_forever, daemon=True)
        thread.start()
        name = self._testMethodName + suffix
        metadata = original._build_client_metadata({"_resolved_port": callback.server_port})
        storage = original.HermesTokenStorage(name)

        async def redirect(url):
            async with httpx.AsyncClient(follow_redirects=True, timeout=2) as browser:
                response = await browser.get(url)
                response.raise_for_status()

        async def complete():
            if result.get("auth_code") is None:
                raise RuntimeError("The real callback listener did not receive an authorization code")
            return original._authorization_code_result(result["auth_code"], result["state"], result["iss"])

        provider = HermesMCPOAuthProvider(server_url=self.server.origin + "/mcp", server_name=name,
            client_metadata=metadata, storage=storage, redirect_handler=redirect, callback_handler=complete)
        if cached:
            await self.seed_expired(provider)
        factory = _make_redirect_header_stripper(httpx, httpx.URL(self.server.origin + "/mcp"))
        try:
            async with factory(auth=provider, follow_redirects=True, timeout=5) as client:
                yield provider, client
        finally:
            callback.shutdown()
            callback.server_close()
            thread.join(3)

    async def seed_expired(self, provider):
        from standalone.mcp_oauth_types import OAuthClientInformationFull, OAuthMetadata, OAuthToken
        storage = provider.context.storage
        await storage.set_client_info(OAuthClientInformationFull(
            client_id="local-client", client_secret="local-client-secret", token_endpoint_auth_method="client_secret_post",
            redirect_uris=provider.context.client_metadata.redirect_uris, issuer=self.server.origin + "/auth"))
        storage.save_oauth_metadata(OAuthMetadata(self.server.metadata()))
        storage.bind_issuer(self.server.origin + "/auth")
        await storage.set_tokens(OAuthToken(access_token="expired-local-access", refresh_token=self.server.refresh,
                                            scope="read", expires_in=0))

    def test_browser_pkce_real_callback_refresh_and_private_storage(self):
        async def run():
            async with self.browser_provider() as (provider, client):
                response = await client.get(self.server.origin + "/mcp")
                self.assertTrue(response.json()["authenticated"])
                self.assertEqual(provider.context.client_info.token_endpoint_auth_method, "client_secret_post")
                provider.context.token_expiry_time = time.time() - 1
                responses = await asyncio.gather(*(client.get(self.server.origin + "/mcp") for _ in range(4)))
                self.assertTrue(all(item.status_code == 200 for item in responses))
                self.assertEqual(self.server.generation, 2)
                saved = json.loads(provider.context.storage._tokens_path().read_text())
                self.assertEqual(saved["refresh_token"], "local-refresh-2")
                self.assertEqual(saved["hermes_issuer"], self.server.origin + "/auth")
                self.assertGreater(saved["expires_at"], time.time())
                if os.name != "nt":
                    self.assertEqual(provider.context.storage._tokens_path().stat().st_mode & 0o777, 0o600)
        asyncio.run(run())
        exchanges = [item for item in self.server.records if item["path"] == "/auth/token"]
        self.assertEqual([item["body"]["grant_type"] for item in exchanges], ["authorization_code", "refresh_token"])
        self.assertTrue(all(item["body"]["resource"] == self.server.origin + "/mcp" for item in exchanges))
        self.assertTrue(all(item["body"]["client_secret"] == "local-client-secret" for item in exchanges))
        self.assertTrue(all(item["headers"].get("user-agent", "").startswith("Hermes-Agent/") for item in exchanges))
        self.assertNotIn("mcp", sys.modules)
        self.assertNotIn("pydantic", sys.modules)

    def test_state_issuer_and_resource_validation_precede_token_exchange(self):
        async def run():
            from standalone.mcp_oauth_types import OAuthFlowError
            for mode in ("state", "issuer", "missing_issuer"):
                self.server.callback_mode = mode
                async with self.browser_provider(mode) as (_provider, client):
                    with self.assertRaises(OAuthFlowError):
                        await client.get(self.server.origin + "/mcp")
            self.server.bad_resource = True
            async with self.browser_provider("resource") as (_provider, client):
                with self.assertRaisesRegex(OAuthFlowError, "resource"):
                    await client.get(self.server.origin + "/mcp")
        asyncio.run(run())
        self.assertFalse(any(item["path"] == "/auth/token" for item in self.server.records))

    def test_cancelled_refresh_releases_original_cross_process_fence(self):
        async def run():
            from tools.mcp_oauth import acquire_refresh_fence, release_refresh_fence
            self.server.delay_refresh = True
            async with self.browser_provider(cached=True) as (provider, client):
                task = asyncio.create_task(client.get(self.server.origin + "/mcp"))
                self.assertTrue(await asyncio.to_thread(self.server.refresh_started.wait, 3))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                descriptor = await acquire_refresh_fence(provider.context.storage._tokens_path(), timeout=0.3)
                release_refresh_fence(descriptor)
                self.assertIsNone(provider._hermes_fence)
                self.server.refresh_release.set()
        asyncio.run(run())

    def test_two_provider_instances_share_one_refresh_generation(self):
        async def run():
            self.server.delay_refresh = True
            async with self.browser_provider("shared", cached=True) as (first, first_client):
                async with self.browser_provider("shared") as (second, second_client):
                    pending = asyncio.gather(first_client.get(self.server.origin + "/mcp"),
                                             second_client.get(self.server.origin + "/mcp"))
                    self.assertTrue(await asyncio.to_thread(self.server.refresh_started.wait, 3))
                    await asyncio.sleep(0.05)
                    self.server.refresh_release.set()
                    responses = await pending
                    self.assertEqual([item.status_code for item in responses], [200, 200])
                    self.assertEqual(first.context.current_tokens.refresh_token, "local-refresh-1")
                    self.assertEqual(second.context.current_tokens.refresh_token, "local-refresh-1")
                    self.assertIsNone(first._hermes_fence)
                    self.assertIsNone(second._hermes_fence)
        asyncio.run(run())
        exchanges = [item for item in self.server.records if item["path"] == "/auth/token"]
        self.assertEqual(len(exchanges), 1)
        self.assertEqual(exchanges[0]["body"]["grant_type"], "refresh_token")

    def test_normal_manager_reuses_cached_client_and_refreshes_without_consent(self):
        async def run():
            import httpx
            from tools.mcp_oauth_manager import get_manager
            from tools.mcp_tool_errors import _make_redirect_header_stripper
            async with self.browser_provider(cached=True) as (seed, _client):
                manager = get_manager()
                config = {"client_id": "local-client", "client_secret": "local-client-secret", "cimd": False}
                provider = manager.get_or_build_provider(self._testMethodName, self.server.origin + "/mcp", config)
                self.assertIs(provider, manager.get_or_build_provider(self._testMethodName, self.server.origin + "/mcp", config))
                self.assertEqual(provider.context.storage._tokens_path(), seed.context.storage._tokens_path())
                factory = _make_redirect_header_stripper(httpx, httpx.URL(self.server.origin + "/mcp"))
                async with factory(auth=provider, timeout=3) as client:
                    self.assertEqual((await client.get(self.server.origin + "/mcp")).status_code, 200)
                self.assertEqual(provider.context.current_tokens.refresh_token, "local-refresh-1")
                manager.evict(self._testMethodName)
        asyncio.run(run())
        self.assertFalse(any(item["path"].startswith("/auth/authorize") for item in self.server.records))

    def test_cimd_client_identifier_skips_registration(self):
        async def run():
            self.server.metadata = lambda: {
                "issuer": self.server.origin + "/auth", "authorization_endpoint": self.server.origin + "/auth/authorize",
                "token_endpoint": self.server.origin + "/auth/token", "code_challenge_methods_supported": ["S256"],
                "authorization_response_iss_parameter_supported": True, "client_id_metadata_document_supported": True,
            }
            async with self.browser_provider() as (provider, client):
                provider.context.client_metadata_url = "https://client.example.test/hermes.json"
                self.assertEqual((await client.get(self.server.origin + "/mcp")).status_code, 200)
        asyncio.run(run())
        self.assertFalse(any(item["path"] == "/auth/register" for item in self.server.records))
        exchange = next(item for item in self.server.records if item["path"] == "/auth/token")
        self.assertEqual(exchange["body"]["client_id"], "https://client.example.test/hermes.json")
        self.assertNotIn("client_secret", exchange["body"])

    def test_token_redirect_does_not_forward_credential_body(self):
        async def run():
            import httpx
            from standalone.mcp_oauth_types import OAuthMetadata
            async with self.browser_provider(cached=True) as (provider, client):
                metadata = {**self.server.metadata(), "token_endpoint": self.server.origin + "/auth/token-redirect"}
                provider.context.storage.save_oauth_metadata(OAuthMetadata(metadata))
                with self.assertRaisesRegex(httpx.RequestError, "cannot follow redirects"):
                    await client.get(self.server.origin + "/mcp")
                self.assertIsNone(provider._hermes_fence)
        asyncio.run(run())
        self.assertFalse(any(item["path"] == "/credential-sink" for item in self.server.records))

    def test_original_device_login_pending_then_success(self):
        async def run():
            from tools.mcp_oauth_device import login_device
            from tools.mcp_oauth import HermesTokenStorage
            await login_device(self._testMethodName, self.server.origin + "/mcp",
                               {"client_id": "local-client", "client_secret": "local-client-secret", "timeout": 5})
            tokens = await HermesTokenStorage(self._testMethodName).get_tokens()
            self.assertEqual(tokens.access_token, "local-access-1")
            self.assertEqual(tokens.refresh_token, "local-refresh-1")
        asyncio.run(run())
        self.assertEqual(self.server.device_polls, 2)
        self.assertFalse(any(item["path"] == "/auth/register" or item["path"].startswith("/auth/authorize")
                             for item in self.server.records))


_WORKER = r'''
import importlib.util, os, pathlib, sys, unittest
source, vendor, scratch, testfile = map(pathlib.Path, sys.argv[1:])
sys.path[:0] = [str(source), str(vendor)]
from standalone.runtime import initialize_bundle
initialize_bundle({"root": str(source.parent), "artifact": str(scratch / "fixture.py"),
                   "bundle_sha256": source.parent.name, "manifest": {"project": "hermes-standalone"}})
os.environ["HERMES_MCP_OAUTH_FIXTURE"] = "1"
def audit(event, args):
    if event == "socket.getaddrinfo" and args[0] not in ("127.0.0.1", "localhost", "::1"):
        raise PermissionError("OAuth fixture forbids external name resolution")
    if event == "socket.connect" and (not isinstance(args[1], tuple) or args[1][0] not in ("127.0.0.1", "::1")):
        raise PermissionError("OAuth fixture allows only loopback connections")
    if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork"):
        raise PermissionError("OAuth fixture forbids process execution")
sys.addaudithook(audit)
spec = importlib.util.spec_from_file_location("oauth_loopback_cases", testfile)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.OAuthWireTests)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
'''


@unittest.skipIf(sys.version_info < (3, 14), "The upstream runtime targets Python 3.14")
class OAuthAgentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from standalone.tests.test_transport_agent import AgentHTTPIntegrationTests
        AgentHTTPIntegrationTests.setUpClass.__func__(cls)

    def test_original_oauth_with_local_authorization_server(self):
        from standalone.agent_probe_worker import fixture_environment
        scratch = self.directory / "oauth"
        for name in ("home", "work", "tmp"):
            (scratch / name).mkdir(parents=True)
        worker = scratch / "worker.py"
        worker.write_text(_WORKER)
        process = subprocess.run([
            sys.executable, "-I", "-S", "-B", "-X", "utf8", str(worker), str(self.source), str(self.vendor),
            str(scratch), str(Path(__file__).resolve()),
        ], cwd=scratch / "work", env=fixture_environment(scratch), capture_output=True,
           text=True, encoding="utf-8", timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
