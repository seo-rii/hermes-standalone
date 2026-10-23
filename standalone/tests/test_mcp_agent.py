"""Original Hermes MCP discovery, registry dispatch and fail-closed trust policy."""

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import queue
import subprocess
import sys
import threading
import unittest

from standalone.tests import test_transport_agent as _agent_fixture
from standalone.tests.test_mcp_transport import _HTTPHandler


_WORKER = r'''
import contextlib, io, json, os, pathlib, sys, time, traceback
source, vendor, scratch, endpoint = sys.argv[1:]
source, vendor, scratch = map(pathlib.Path, (source, vendor, scratch))
sys.path[:0] = [str(source), str(vendor)]
from standalone.runtime import initialize_bundle
initialize_bundle({"root": str(source.parent), "artifact": str(scratch / "fixture.py"),
                   "bundle_sha256": source.parent.name, "manifest": {"project": "hermes-standalone"}})
home = pathlib.Path(os.environ["HERMES_HOME"])
home.mkdir(parents=True)
(home / "config.yaml").write_text(json.dumps({"mcp_servers": {"wire": {
    "url": endpoint, "trust": "untrusted", "connect_timeout": 5, "timeout": 5,
    "headers": {"X-Fixture-Profile": "temporary-only"}}}}), encoding="utf-8")
os.environ["HERMES_SINGLE_QUERY_SESSION"] = "1"
output, errors = io.StringIO(), io.StringIO()
report = {"ok": False}
try:
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        from tools.mcp_tool_discovery import discover_mcp_tools
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        from tools.registry import registry
        from tools import mcp_tool
        names = discover_mcp_tools()
        report["names"] = names
        report["session_module"] = mcp_tool.ClientSession.__module__
        report["read"] = registry.dispatch("mcp__wire__echo", {"text": "original-registry-result"})
        deadline = time.monotonic() + 3
        while registry.get_entry("mcp__wire__dynamic") is None and time.monotonic() < deadline:
            time.sleep(0.01)
        report["refreshed"] = registry.get_entry("mcp__wire__dynamic") is not None
        report["write"] = registry.dispatch("mcp__wire__write", {})
        report["resource"] = registry.dispatch("mcp__wire__read_resource", {"uri": "fixture://read"})
        report["prompt"] = registry.dispatch("mcp__wire__get_prompt", {"name": "fixture"})
        shutdown_mcp_servers()
        report["remaining_servers"] = len(mcp_tool._servers)
        report["global_sdk_injected"] = "mcp" in sys.modules
        report["ok"] = ("original-registry-result" in str(report["read"]) and "NOT run" in str(report["write"])
                        and report["refreshed"])
except BaseException as error:
    report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
report["stdout"] = output.getvalue()[-2000:]
report["stderr"] = errors.getvalue()[-3000:]
print(json.dumps(report))
raise SystemExit(not report["ok"])
'''


@unittest.skipIf(sys.version_info < (3, 14), "The upstream runtime targets Python 3.14")
class MCPAgentIntegrationTests(unittest.TestCase):
    setUpClass = classmethod(_agent_fixture.AgentHTTPIntegrationTests.setUpClass.__func__)

    def test_original_discovery_dispatch_trust_and_shutdown(self):
        from standalone.agent_probe_worker import fixture_environment
        scratch = self.directory / "mcp"
        for name in ("home", "work", "tmp"):
            (scratch / name).mkdir(parents=True)
        worker = scratch / "worker.py"
        worker.write_text(_WORKER, encoding="utf-8")
        server = ThreadingHTTPServer(("127.0.0.1", 0), _HTTPHandler)
        server.daemon_threads = True
        server.dynamic_notifications = True
        server.notification_queues = []
        server.requests, server.deletes = [], []
        server.events, server.stopping = queue.Queue(), threading.Event()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            process = subprocess.run([
                sys.executable, "-I", "-S", "-B", "-X", "utf8", str(worker), str(self.source), str(self.vendor),
                str(scratch), f"http://127.0.0.1:{server.server_port}/mcp",
            ], cwd=scratch / "work", env=fixture_environment(scratch), capture_output=True,
               text=True, encoding="utf-8", timeout=40)
        finally:
            server.stopping.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads(process.stdout)
        self.assertEqual(report["session_module"], "standalone.mcp_client")
        self.assertFalse(report["global_sdk_injected"])
        self.assertEqual(report["remaining_servers"], 0)
        self.assertTrue(report["refreshed"])
        self.assertIn("resource fixture", report["resource"])
        self.assertIn("prompt fixture", report["prompt"])
        calls = [item for item in server.requests if item["message"].get("method") == "tools/call"]
        self.assertEqual([item["message"]["params"]["name"] for item in calls], ["echo"])
        self.assertEqual(calls[0]["headers"]["x-fixture-profile"], "temporary-only")


if __name__ == "__main__":
    unittest.main()
