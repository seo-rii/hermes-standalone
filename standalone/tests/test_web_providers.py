"""Original web tools use the reviewed HTTP boundaries in an isolated bundle."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from urllib.parse import parse_qs


_CONFIGURE = r'''
from standalone import web_providers, web_ddgs
origin = "http://127.0.0.1:" + str(endpoint_port)
web_providers.EXA_BASE_URL = origin + "/exa"
web_providers.PARALLEL_BASE_URL = origin + "/parallel"
web_ddgs.SEARCH_URL = origin + "/ddg/html/"
'''

_DRIVER = r'''
import asyncio, base64, time, traceback
from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_context, reset_multiplex_context
from tools import web_tools
home = pathlib.Path(os.environ["HERMES_HOME"])
home.mkdir(parents=True, exist_ok=True)
def configure(backend):
    (home / "config.yaml").write_text(json.dumps({"web": {"backend": backend,
        "keyless_fallback": False, "keyless_rescue": False}, "agent": {"environment_probe": False}}))
def search(query, limit=5):
    value = json.loads(web_tools.web_search_tool(query, limit))
    assert value.get("success"), value
    return value["data"]["web"]
def extract(urls):
    value = json.loads(asyncio.run(web_tools.web_extract_tool(urls)))
    assert "results" in value, value
    return value["results"]
report = {}
try:
    configure("exa")
    rows = search("exa-highlights", 2)
    assert rows == [{"url": "https://8.8.8.8/exa", "title": "Exa result", "description": "first second", "position": 1}], rows
    rows = extract(["https://8.8.8.8/exa-body"])
    assert rows[0]["content"] == "Exa full text", rows
    multiplex_token = set_multiplex_context(True)
    try:
        for query, key in (("exa-profile-a", "profile-a-exa"), ("exa-profile-b", "profile-b-exa")):
            token = set_secret_scope({"EXA_API_KEY": key})
            try:
                search(query)
            finally:
                reset_secret_scope(token)
        token = set_secret_scope({})
        try:
            denied = json.loads(web_tools.web_search_tool("exa-scoped-missing"))
            assert not denied["success"] and "EXA_API_KEY" in denied["error"], denied
        finally:
            reset_secret_scope(token)
    finally:
        reset_multiplex_context(multiplex_token)
    denied = json.loads(web_tools.web_search_tool("exa-denied"))
    assert not denied["success"] and "status code 401" in denied["error"], denied
    report["exa"] = True

    configure("parallel")
    rows = search("parallel-retry", 100)
    assert rows[0]["description"] == "one two", rows
    rows = extract(["https://8.8.8.8/parallel-body", "https://8.8.8.8/parallel-error"])
    assert rows[0]["content"] == "Parallel full text", rows
    assert rows[1]["error"] == "fixture extraction blocked", rows
    report["parallel"] = True

    configure("firecrawl")
    rows = search("firecrawl-retry")
    assert rows[0]["description"] == "Firecrawl snippet", rows
    rows = extract(["https://8.8.8.8/firecrawl-body", "https://8.8.8.8/firecrawl-redirect"])
    assert rows[0]["content"] == "Firecrawl markdown", rows
    assert rows[1]["url"] == "http://127.0.0.1/private" and "private or internal" in rows[1]["error"], rows
    denied = json.loads(web_tools.web_search_tool("firecrawl-quota"))
    assert not denied["success"] and "Payment Required" in denied["error"], denied
    os.environ.pop("FIRECRAWL_API_KEY")
    search("firecrawl-selfhost-keyless")
    # A JWT is only a local entitlement snapshot, as in the original account helper;
    # no external account or token is used by this fixture.
    def segment(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
    jwt = segment({"alg": "none"}) + "." + segment({"exp": int(time.time()) + 3600, "paid_access": True}) + ".fixture"
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {"nous": {"access_token": jwt}}}))
    os.environ["TOOL_GATEWAY_USER_TOKEN"] = "fixture-managed-token"
    os.environ["FIRECRAWL_GATEWAY_URL"] = origin
    os.environ["PERPLEXITY_GATEWAY_URL"] = origin + "/perplexity"
    os.environ["FIRECRAWL_API_KEY"] = "direct-key-must-not-bill-managed"
    configure("nous")
    search("firecrawl-managed")
    rows = extract(["https://8.8.8.8/firecrawl-managed-body"])
    assert rows[0]["content"] == "Firecrawl markdown", rows
    report["firecrawl"] = True

    configure("ddgs")
    rows = search("ddg-pages", 2)
    assert [row["title"] for row in rows] == ["Duck & first", "Duck second"], rows
    assert rows[0]["url"] == "https://8.8.8.8/ddg-one" and rows[0]["description"] == "first snippet", rows
    from plugins.web.ddgs import provider as ddgs_provider
    assert ddgs_provider._last_worker_proc.poll() == 0
    denied = json.loads(web_tools.web_search_tool("ddg-challenge"))
    assert not denied["success"] and "bot challenge" in denied["error"], denied
    assert ddgs_provider._last_worker_proc.poll() is not None
    assert not any(name in sys.modules for name in ("exa_py", "parallel", "firecrawl", "ddgs", "primp", "pydantic"))
    report["ddgs"] = True
    report["blocked_events"] = guard.events
except BaseException:
    report["error"] = traceback.format_exc()
print(json.dumps(report), flush=True)
os._exit(0)
'''

_DEADLINE_DRIVER = r'''
import threading, time, traceback
from plugins.web.ddgs import provider
from tools.interrupt import set_interrupt
from tools.web_tools import web_search_tool
home = pathlib.Path(os.environ["HERMES_HOME"])
home.mkdir(parents=True, exist_ok=True)
(home / "config.yaml").write_text(json.dumps({"web": {"backend": "ddgs", "keyless_fallback": False,
    "keyless_rescue": False}, "agent": {"environment_probe": False}}))
report = {}
bounded_calls = []
original_bounded = provider._run_ddgs_search_bounded
def measured_bounded(*args, **kwargs):
    began = time.monotonic()
    try:
        return original_bounded(*args, **kwargs)
    finally:
        bounded_calls.append(time.monotonic() - began)
provider._run_ddgs_search_bounded = measured_bounded
try:
    provider._test_hook = "sleep"
    provider._SEARCH_TIMEOUT_SECS = 0.3
    started = time.monotonic()
    timed = json.loads(web_search_tool("ddg-owned-timeout"))
    assert not timed["success"] and "timed out" in timed["error"], timed
    assert provider._last_worker_proc.poll() is not None
    elapsed = time.monotonic() - started
    # Cold tool/plugin discovery is outside the disposable search worker's
    # deadline. The observed failure spent 5.83 s in the tool and 0.497 s in
    # the actual bounded operation, including worker termination and reaping.
    assert len(bounded_calls) == 1 and bounded_calls[0] < 5, bounded_calls
    report["timeout_seconds"] = bounded_calls[0]
    report["tool_initialization_seconds"] = elapsed - bounded_calls[0]
    report["timeout_reaped"] = True
    provider._SEARCH_TIMEOUT_SECS = 10
    main_tid = threading.get_ident()
    timer = threading.Timer(0.5, lambda: set_interrupt(True, main_tid))
    timer.start()
    try:
        cancelled = json.loads(web_search_tool("ddg-owned-interrupt"))
    finally:
        timer.join(2)
        set_interrupt(False)
    assert not cancelled["success"] and "interrupted" in cancelled["error"].lower(), cancelled
    assert provider._last_worker_proc.poll() is not None
    report["interrupt_reaped"] = True
    report["blocked_events"] = guard.events
except BaseException:
    report["error"] = traceback.format_exc()
print(json.dumps(report), flush=True)
os._exit(0)
'''


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def reply(self, payload, status=200, *, html=False, headers=None):
        content = payload.encode() if html else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
        self.send_header("Content-Length", str(len(content)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = {key: value[0] for key, value in parse_qs(raw.decode(), keep_blank_values=True).items()} if self.path == "/ddg/html/" else json.loads(raw)
        record = {"path": self.path, "body": body, "headers": {key.lower(): value for key, value in self.headers.items()}}
        with self.server.record_lock:
            self.server.records.append(record)
            prior = sum(item["path"] == self.path and item["body"] == body for item in self.server.records)
        if body.get("query") == "exa-denied":
            return self.reply({"error": "fixture unauthorized"}, 401)
        if self.path == "/exa/search":
            return self.reply({"results": [{"url": "https://8.8.8.8/exa", "title": "Exa result", "highlights": ["first", "second"]}]})
        if self.path == "/exa/contents":
            return self.reply({"results": [{"url": body["urls"][0], "title": "Exa body", "text": "Exa full text"}]})
        if self.path == "/parallel/v1beta/search":
            if prior == 1:
                return self.reply({"error": {"message": "fixture rate limit"}}, 429, headers={"Retry-After": "0"})
            return self.reply({"results": [{"url": "https://8.8.8.8/parallel", "title": "Parallel", "excerpts": ["one", "two"]}]})
        if self.path == "/parallel/v1beta/extract":
            return self.reply({"results": [{"url": body["urls"][0], "title": "Parallel body", "full_content": "Parallel full text"}],
                               "errors": [{"url": body["urls"][1], "content": "fixture extraction blocked", "error_type": "access_denied"}]})
        if self.path == "/v2/search":
            if body["query"] == "firecrawl-retry" and prior == 1:
                return self.reply({"error": "fixture bad gateway"}, 502)
            if body["query"] == "firecrawl-quota":
                return self.reply({"error": "fixture quota exhausted"}, 402)
            return self.reply({"success": True, "data": {"web": [{"url": "https://8.8.8.8/firecrawl", "title": "Firecrawl", "description": "Firecrawl snippet"}]}})
        if self.path == "/v2/scrape":
            final = "http://127.0.0.1/private" if body["url"].endswith("-redirect") else body["url"]
            return self.reply({"success": True, "data": {"markdown": "Firecrawl markdown", "metadata": {"title": "Firecrawl body", "sourceURL": final}}})
        if self.path == "/perplexity/search":
            return self.reply({"error": "fixture managed primary unavailable"}, 503)
        if self.path == "/ddg/html/":
            if body["q"] == "ddg-challenge":
                return self.reply('<form id="challenge-form">Bots use DuckDuckGo too</form>', 202, html=True)
            if "s" not in body:
                return self.reply('<div class="result"><div class="result__body"><h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2F8.8.8.8%2Fddg-one">Duck &amp; first</a></h2><a class="result__snippet">first <b>snippet</b></a></div></div>', html=True)
            if body["s"] == "10":
                return self.reply('<div class="body"><h2>Duck second</h2><a href="https://8.8.8.8/ddg-two">second snippet</a></div>', html=True)
            return self.reply("<html>No further results</html>", html=True)
        self.reply({"error": "Unexpected fixture path"}, 404)


class DuckDuckGoParserTests(unittest.TestCase):
    def test_results_unwrap_redirects_drop_ads_and_preserve_text(self):
        from standalone.web_ddgs import _Results
        parser = _Results()
        parser.feed('<div class="body"><h2>A &amp; <b>B</b></h2><a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2F">a<br>b</a></div>'
                    '<div class="body"><h2>Ad</h2><a href="https://duckduckgo.com/y.js?ad=1">skip</a></div>')
        parser.close()
        self.assertEqual(parser.rows, [{"title": "A & B", "href": "https://example.org/", "body": "a b"}])


@unittest.skipUnless(sys.platform.startswith("linux") and sys.version_info >= (3, 14),
                     "Original web tool integration currently verified on Linux/Python 3.14")
class OriginalWebProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from standalone.tests.test_entrypoints import BundledSourceTestCase
        BundledSourceTestCase.setUpClass.__func__(cls)

    def invoke(self, driver):
        from standalone.tests.test_chat_entry import _LAUNCHER, OriginalChatTests
        from standalone.tests.test_entrypoints import support
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        server.records, server.record_lock = [], threading.Lock()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory(prefix="w-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            context = dict(self.context, artifact=str(scratch / "hermes.py"))
            context["manifest"] = dict(context["manifest"], internal_modules=[
                *context["manifest"]["internal_modules"], "plugins.web.ddgs._search_worker"])
            launcher = _LAUNCHER.replace("CONTEXT_JSON", repr(json.dumps(context))).replace(
                "SCRATCH_ROOT", repr(str(scratch))).replace("ENDPOINT_PORT", str(server.server_port))
            # Numeric getaddrinfo performs no DNS lookup. It exercises the real
            # URL safety check, while every connection still targets loopback.
            launcher = launcher.replace('        if event == "socket.getaddrinfo":',
                '        if event == "socket.getaddrinfo" and args[0] in {"8.8.8.8", "127.0.0.1"}:\n'
                '            return\n        if event == "socket.getaddrinfo":')
            launcher = launcher.replace("arguments = sys.argv[1:]", _CONFIGURE + "\narguments = sys.argv[1:]")
            Path(context["artifact"]).write_text(launcher, encoding="utf-8")
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            environment.update(PATH="", EXA_API_KEY="ambient-exa", PARALLEL_API_KEY="fixture-parallel",
                               PARALLEL_SEARCH_MODE="fast", FIRECRAWL_API_KEY="fixture-firecrawl",
                               FIRECRAWL_API_URL=f"http://127.0.0.1:{server.server_port}/ignored-base-path")
            source = launcher.split("arguments = sys.argv[1:]")[0] + driver
            try:
                result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", source],
                    env=environment, cwd=scratch / "work", capture_output=True, text=True, encoding="utf-8", timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertNotIn("error", report, report.get("error"))
                self.assertTrue(all(event.get("recoverable_capability_probe") for event in report["blocked_events"]), report)
                with server.record_lock:
                    return report, list(server.records)
            finally:
                OriginalChatTests._stop_fixture_processes(self, scratch)

    def test_original_search_extract_key_scope_and_vendor_wire(self):
        report, records = self.invoke(_DRIVER)
        self.assertTrue(all(report[name] for name in ("exa", "parallel", "firecrawl", "ddgs")))
        def calls(path, query=None):
            return [row for row in records if row["path"] == path and (query is None or row["body"].get("query") == query)]
        exa = calls("/exa/search", "exa-highlights")[0]
        self.assertEqual(exa["body"], {"query": "exa-highlights", "numResults": 10, "contents": {"highlights": True}})
        self.assertEqual(exa["headers"]["x-exa-integration"], "hermes-agent")
        self.assertEqual(calls("/exa/search", "exa-profile-a")[0]["headers"]["x-api-key"], "profile-a-exa")
        self.assertEqual(calls("/exa/search", "exa-profile-b")[0]["headers"]["x-api-key"], "profile-b-exa")
        self.assertEqual(calls("/exa/search", "exa-scoped-missing"), [])
        self.assertEqual(len(calls("/exa/search", "exa-denied")), 1)
        self.assertEqual(calls("/exa/contents")[0]["body"], {"urls": ["https://8.8.8.8/exa-body"], "text": True})
        parallel = calls("/parallel/v1beta/search")
        self.assertEqual(len(parallel), 2)
        self.assertEqual(parallel[0]["body"], {"search_queries": ["parallel-retry"], "objective": "parallel-retry", "mode": "fast", "max_results": 20})
        self.assertEqual(parallel[0]["headers"]["parallel-beta"], "search-extract-2025-10-10")
        self.assertEqual(parallel[0]["headers"]["x-api-key"], "fixture-parallel")
        self.assertTrue(calls("/parallel/v1beta/extract")[0]["body"]["full_content"])
        firecrawl = calls("/v2/search", "firecrawl-retry")
        self.assertEqual(len(firecrawl), 2)
        self.assertEqual(firecrawl[0]["body"], {"query": "firecrawl-retry", "limit": 10, "origin": "python-sdk@4.17.0"})
        self.assertEqual(firecrawl[0]["headers"]["authorization"], "Bearer fixture-firecrawl")
        self.assertEqual(len(calls("/v2/search", "firecrawl-quota")), 1)
        self.assertNotIn("authorization", calls("/v2/search", "firecrawl-selfhost-keyless")[0]["headers"])
        self.assertEqual(calls("/v2/search", "firecrawl-managed")[0]["headers"]["authorization"], "Bearer fixture-managed-token")
        self.assertEqual(calls("/v2/scrape")[0]["body"]["timeout"], 60000)
        self.assertEqual([row["body"].get("s") for row in calls("/ddg/html/") if row["body"]["q"] == "ddg-pages"], [None, "10", "25"])

    def test_original_ddgs_deadline_and_interrupt_reap_owned_children(self):
        report, records = self.invoke(_DEADLINE_DRIVER)
        self.assertTrue(report["timeout_reaped"] and report["interrupt_reaped"])
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
