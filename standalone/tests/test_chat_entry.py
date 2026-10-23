"""The original CLI, canonical gateway and managed worker use a local HTTP provider."""

from http.server import ThreadingHTTPServer
import json
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest

from standalone import processes
from standalone.tests.test_entrypoints import BundledSourceTestCase, support
from standalone.tests.test_transport_agent import _AgentHTTPHandler


_LAUNCHER = r'''
import json, os, pathlib, re, runpy, socket, sys
import zoneinfo
zoneinfo.reset_tzpath(())  # Use the embedded IANA database, independently of host files.
context = json.loads(CONTEXT_JSON)
scratch = pathlib.Path(SCRATCH_ROOT)
endpoint_port = ENDPOINT_PORT
source = pathlib.Path(context["root"]) / "app"
sys.path[:0] = [str(source), str(source.parent / "vendor")]
from standalone.runtime import initialize_bundle, bundle_module_argv
initialize_bundle(context)
from standalone import processes
if os.environ.get("ENTRY_FIXTURE_STACKS"):
    import faulthandler, signal
    trace_stream = open(scratch / (str(os.getpid()) + ".trace"),
        "w", opener=lambda path, flags: os.open(path, flags, 0o600))
    faulthandler.register(signal.SIGUSR2, file=trace_stream, all_threads=True)
with (scratch / "owned-processes.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"pid": os.getpid(), "birth": processes.Process().create_time()}) + "\n")
from standalone.agent_probe_worker import ProbeGuard, ProbeBlocked

class Guard(ProbeGuard):
    def allowed_port(self, port):
        if port == endpoint_port or port in getattr(self, "http_ports", ()):
            return True
        try:
            owned = {json.loads(line)["pid"]: json.loads(line)["birth"]
                     for line in (scratch / "owned-processes.jsonl").read_text().splitlines()}
            for line in (scratch / "owned-listeners.jsonl").read_text().splitlines():
                record = json.loads(line)
                if record["port"] == port and record["pid"] in owned:
                    try:
                        if processes.Process(record["pid"]).create_time() == owned[record["pid"]]:
                            return True
                    except (processes.Error, OSError):
                        continue
            return False
        except (OSError, ValueError, KeyError):
            return False

    def deny(self, event, operation, path=None, *, recoverable=False):
        try:
            super().deny(event, operation, path, recoverable=recoverable)
        finally:
            if event.startswith("socket."):
                self.events[-1]["address"] = getattr(self, "address", None)
            if event == "subprocess.Popen":
                self.events[-1]["fixture_command"] = getattr(self, "fixture_command", None)
            with (scratch / "guard-events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(self.events[-1]) + "\n")

    def __call__(self, event, args):
        if event == "socket.sendmsg":
            # Python 3.14 asyncio batches HTTP chunks with sendmsg. Permit
            # only a socket already connected to an admitted owned listener.
            try:
                peer = args[0].getpeername()
            except OSError:
                peer = None
            if isinstance(peer, tuple) and peer[0] == "127.0.0.1" and self.allowed_port(peer[1]):
                return
        if event in {"os.fork", "os.forkpty"} and getattr(self, "pty_spawning", False):
            return
        if event == "os.exec" and getattr(self, "pty_spawning", False) and bundle_module_argv(args[1]) is not None:
            return
        if event in {"socket.bind", "socket.connect"}:
            address = args[1]
            self.address = repr(address)
            if isinstance(address, (str, bytes)) and pathlib.Path(os.fsdecode(address)).is_relative_to(scratch):
                return
            if event == "socket.bind" and address == ("127.0.0.1", 0):
                return
            if event == "socket.connect" and isinstance(address, tuple) and address[0] == "127.0.0.1" and self.allowed_port(address[1]):
                return
        if event == "socket.getaddrinfo":
            self.address = repr(args[:2])
            if args[0] == "127.0.0.1" and (int(args[1]) == 0 or self.allowed_port(int(args[1]))):
                return
        if event == "subprocess.Popen":
            command = args[1]
            self.fixture_command = list(command) if isinstance(command, (list, tuple)) else str(command)
            if isinstance(command, (list, tuple)) and bundle_module_argv(command) is not None:
                return
            if isinstance(command, (list, tuple)) and command and command[0] == "git":
                self.deny(event, "unavailable_optional_git", recoverable=True)
            if isinstance(command, (list, tuple)) and len(command) == 3 and pathlib.Path(command[0]).name == "gh" and list(command[1:]) == ["auth", "token"]:
                self.deny(event, "unavailable_optional_github_credentials", recoverable=True)
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            if path == "/dev/tty" and getattr(self, "pty_spawning", False):
                return
            if path == "/etc/ssh/sshd_config" or path.startswith("/etc/ssh/sshd_config.d/"):
                self.deny(event, "unavailable_host_security_configuration", recoverable=True)
            match = re.fullmatch(r"/proc/([0-9]+)/(?:cmdline|environ|statm)", path)
            if match:
                owned = {json.loads(line)["pid"] for line in (scratch / "owned-processes.jsonl").read_text().splitlines()}
                if int(match[1]) in owned:
                    return
                # Canonical ownership discovery must be able to skip unrelated PIDs.
                raise PermissionError("The fixture does not inspect unrelated processes")
        super().__call__(event, args)

guard = Guard(source, source.parent / "vendor", scratch)
sys.addaudithook(guard)
original_bind = socket.socket.bind
def fixture_bind(listener, address):
    original_bind(listener, address)
    bound = listener.getsockname()
    if isinstance(bound, tuple) and bound[0] == "127.0.0.1":
        with (scratch / "owned-listeners.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "port": bound[1]}) + "\n")
socket.socket.bind = fixture_bind
arguments = sys.argv[1:]
if arguments[:1] == ["--cache-dir"]:
    arguments = arguments[2:]
if arguments[:1] == ["--internal-module"]:
    module, arguments = arguments[1], arguments[2:]
    assert module in context["manifest"]["internal_modules"]
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
else:
    from standalone.cli import main
    raise SystemExit(main(arguments, context))
'''

_DASHBOARD_DRIVER = r'''
import contextlib, threading, time, types, urllib.error, urllib.request
from hermes_cli.main_dashboard import _resolve_dashboard_web_dist
_resolve_dashboard_web_dist(types.SimpleNamespace(skip_build=False), False)
from hermes_cli import web_server as web
from hermes_cli.web_server_chat import _resolve_chat_argv
from hermes_cli.pty_bridge import PtyBridge
import uvicorn
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
port = listener.getsockname()[1]
guard.http_ports = {port}
server = uvicorn.Server(uvicorn.Config(web.app, log_level="error", lifespan="off"))
server_thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
server_thread.start()
client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
origin = f"http://127.0.0.1:{port}"
bridge = None
try:
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "Dashboard HTTP listener did not start"
    with client.open(origin + "/", timeout=5) as response:
        html = response.read().decode()
        assert response.status == 200 and "no-store" in response.headers["Cache-Control"]
    asset_path = re.search(r'src="(/assets/[^\"]+\.js)"', html).group(1)
    with client.open(origin + asset_path, timeout=5) as response:
        assert response.status == 200 and "javascript" in response.headers["Content-Type"]
        assert "immutable" in response.headers["Cache-Control"]
        assert len(response.read()) > 1000
    try:
        client.open(origin + "/api/config", timeout=5)
    except urllib.error.HTTPError as error:
        denied = error.code
    else:
        raise AssertionError("Dashboard configuration is exposed without authentication")
    request = urllib.request.Request(origin + "/api/config", headers={web._SESSION_HEADER_NAME: web._SESSION_TOKEN})
    with client.open(request, timeout=5) as response:
        assert response.status == 200 and isinstance(json.load(response), dict)
    argv, cwd, env = _resolve_chat_argv(workspace_cwd=str(scratch / "work"),
                                      active_session_file=str(scratch / "active-session.json"))
    module, arguments = bundle_module_argv(argv)
    assert module == "hermes_cli.main" and arguments[:2] == ["chat", "--cli"]
    argv.extend(["--provider", "custom", "--model", "standalone-wire-fixture", "--base-url",
                 f"http://127.0.0.1:{endpoint_port}/v1", "--api-key", "local-http-fixture-token",
                 "--toolsets", "todo", "--ignore-rules", "-Q", "-q",
                 "Record the HTTP verification task as completed, then confirm completion."])
    guard.pty_spawning = True
    try:
        bridge = PtyBridge.spawn(argv, cwd=cwd, env=env, cols=100, rows=30)
    finally:
        guard.pty_spawning = False
    output = bytearray()
    # This functional fixture cold-starts the canonical gateway and a distinct
    # worker from verified sources. Use the same bounded startup/output window
    # as the classic interactive fixture below; this is not a latency contract.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        chunk = bridge.read(timeout=0.1)
        if chunk is None:
            break
        output.extend(chunk)
        assert len(output) < 1024 * 1024
    else:
        import signal
        os.kill(bridge.pid, signal.SIGUSR2)
        raise AssertionError("Dashboard terminal did not finish: " + output.decode(errors="replace")[-3000:])
    text = output.decode(errors="replace")
    assert "HTTP fixture complete" in text, text[-3000:]
    report = {"http_root": 200, "http_asset": 200, "http_unauthorized": denied, "http_config": 200,
              "embedded_assets": pathlib.Path(os.environ["HERMES_WEB_DIST"]) == source / "hermes_cli" / "web_dist",
              "terminal_pid": bridge.pid, "stdout": text, "module": module,
              "marker": json.loads((scratch / "active-session.json").read_text())}
finally:
    if bridge is not None:
        bridge.close()
    server.should_exit = True
    server_thread.join(timeout=10)
    listener.close()
assert not server_thread.is_alive()
print(json.dumps(report), flush=True)
os._exit(0)
'''

_TERMINAL_ECHO = r'''
def observed_echo(output, text):
    visible = re.sub(rb"\x1b\[[0-?]*[ -/]*[@-~]", b"", bytes(output))
    return b"".join(text.split()) in b"".join(visible.split())
'''

_INTERACTIVE_DRIVER = _TERMINAL_ECHO + r'''
import asyncio, time
from hermes_cli.pty_bridge import PtyBridge
argv = [sys.executable, "-I", "-S", "-B", "-X", "utf8", context["artifact"]]
guard.pty_spawning = True
try:
    bridge = PtyBridge.spawn(argv, cwd=str(scratch / "work"), env=dict(os.environ), cols=100, rows=30)
finally:
    guard.pty_spawning = False
output = bytearray()
submitted, turn_sent, quitting, quit_sent = False, False, False, False
prompt = b"Record the HTTP verification task as completed, then confirm completion."
echo_at, quit_echo_at = None, None
try:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        chunk = bridge.read(timeout=0.1)
        if chunk is None:
            break
        output.extend(chunk)
        assert len(output) < 1024 * 1024
        if not submitted and (scratch / "active-session.json").exists() and b"\x1b[?2004h" in output:
            assert asyncio.run(bridge.write(prompt))
            submitted = True
        if submitted and not turn_sent and observed_echo(output, prompt):
            if echo_at is None:
                echo_at = time.monotonic()
            # The original REPL treats Enter within 50 ms of new text as a
            # pasted newline. Submit after its observed echo has settled.
            if time.monotonic() - echo_at >= 0.1:
                assert asyncio.run(bridge.write(b"\r"))
                turn_sent = True
        if turn_sent and not quitting and b"HTTP fixture complete" in output:
            assert asyncio.run(bridge.write(b"/quit"))
            quitting = True
        if quitting and not quit_sent and observed_echo(output, b"/quit"):
            if quit_echo_at is None:
                quit_echo_at = time.monotonic()
            if time.monotonic() - quit_echo_at >= 0.1:
                assert asyncio.run(bridge.write(b"\r"))
                quit_sent = True
    else:
        if bridge.is_alive():
            import signal
            os.kill(bridge.pid, signal.SIGUSR2)
        raise AssertionError("Default terminal did not finish: " + repr({
            "submitted": submitted, "turn_sent": turn_sent, "quitting": quitting, "quit_sent": quit_sent,
            "output": output.decode(errors="replace")[-3000:]}))
    text = output.decode(errors="replace")
    assert submitted and quitting and "HTTP fixture complete" in text, text[-3000:]
    print(json.dumps({"stdout": text, "no_command": True, "terminal_pid": bridge.pid,
                      "marker": json.loads((scratch / "active-session.json").read_text())}), flush=True)
finally:
    bridge.close()
os._exit(0)
'''


@unittest.skipUnless(sys.platform.startswith("linux") and sys.version_info >= (3, 14),
                     "Original managed gateway integration currently verified on Linux/Python 3.14")
class OriginalChatTests(BundledSourceTestCase):
    def test_oneshot_reaches_original_gateway_worker_and_persists_http_turn(self):
        self._exercise_entry(dashboard=False)

    def test_dashboard_http_assets_and_real_python_terminal_persist_session(self):
        self._exercise_entry(dashboard=True)

    def test_no_command_starts_original_classic_repl_and_executes_real_turn(self):
        self._exercise_entry(dashboard=False, interactive=True)

    def _exercise_entry(self, *, dashboard, interactive=False):
        with tempfile.TemporaryDirectory(prefix="chat-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _AgentHTTPHandler)
            server.requests = []
            server.daemon_threads = True
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            context = dict(self.context, artifact=str(scratch / "hermes.py"))
            Path(context["artifact"]).write_text(
                _LAUNCHER.replace("CONTEXT_JSON", repr(json.dumps(context)))
                .replace("SCRATCH_ROOT", repr(str(scratch)))
                .replace("ENDPOINT_PORT", str(server.server_port)), encoding="utf-8",
            )
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            environment["PATH"] = ""
            if dashboard or interactive:
                environment["ENTRY_FIXTURE_STACKS"] = "1"
            environment["HERMES_TUI_ACTIVE_SESSION_FILE"] = str(scratch / "active-session.json")
            home = Path(environment["HERMES_HOME"])
            home.mkdir()
            # Interactive startup reads the original shared model registry even
            # when catalog background refresh is disabled. Keep that cache local.
            (home / "models_dev_cache.json").write_text(json.dumps({
                "openai": {"id": "openai", "name": "Fixture registry", "models": {}}
            }), encoding="utf-8")
            endpoint = f"http://127.0.0.1:{server.server_port}/v1"
            (home / "config.yaml").write_text(json.dumps({
                "model": {"default": "standalone-wire-fixture", "provider": "custom",
                          "base_url": endpoint, "api_key": "local-http-fixture-token", "context_length": 131072},
                "agent": {"environment_probe": False, "api_max_retries": 1, "auto_recovery_cycles": 0,
                          "gateway_startup_warmup_timeout": 0},
                "gateway": {"multiplex_profiles": False, "loop_watchdog": False, "managed_workers": not interactive},
                "compression": {"enabled": False},
                "tools": {"tool_search": {"enabled": False}},
                "platform_toolsets": {"cli": ["todo"]},
                "model_catalog": {"enabled": False},
                "display": {"interface": "tui"},
                "telemetry": {"shared_metrics": {"enabled": False, "send": False, "offer_version": 2}},
            }), encoding="utf-8")
            command = [sys.executable, "-I", "-S", "-B", "-X", "utf8", context["artifact"]]
            try:
                if dashboard or interactive:
                    driver = _LAUNCHER.split("arguments = sys.argv[1:]")[0] + (
                        _INTERACTIVE_DRIVER if interactive else _DASHBOARD_DRIVER)
                    driver = driver.replace("CONTEXT_JSON", repr(json.dumps(context))).replace(
                        "SCRATCH_ROOT", repr(str(scratch))).replace("ENDPOINT_PORT", str(server.server_port))
                    command = [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", driver]
                else:
                    command.extend(["chat", "--provider", "custom", "--model", "standalone-wire-fixture",
                                    "--base-url", endpoint, "--api-key", "local-http-fixture-token", "--toolsets", "todo",
                                    "--ignore-rules", "-Q", "-q", "Record the HTTP verification task as completed, then confirm completion."])
                try:
                    result = subprocess.run(
                        command,
                        env=environment, cwd=scratch / "work", capture_output=True, text=True, timeout=120 if dashboard or interactive else 60,
                    )
                except subprocess.TimeoutExpired as error:
                    result = subprocess.CompletedProcess(
                        command, 124, (error.stdout or b"").decode(errors="replace"),
                        (error.stderr or b"").decode(errors="replace"),
                    )
                gateway_log = home / "logs" / "gateway-stdio.log"
                diagnostic = {"stdout": result.stdout[-2000:], "stderr": result.stderr[-3000:],
                              "provider_port": server.server_port,
                              "requests": [{"path": request["path"], "has_tools": bool(request["body"].get("tools"))} for request in server.requests],
                              "gateway_log": gateway_log.read_text()[-5000:] if gateway_log.exists() else ""}
                diagnostic["guard_events"] = (scratch / "guard-events.jsonl").read_text() if (scratch / "guard-events.jsonl").exists() else ""
                diagnostic["logs"] = {str(path.relative_to(home)): path.read_text(errors="replace")[-4000:]
                                      for path in (home / "logs").glob("*.log") if path != gateway_log}
                if dashboard or interactive:
                    diagnostic["terminal_stacks"] = {path.name: path.read_text(errors="replace")[-7000:]
                                                       for path in scratch.glob("*.trace") if path.stat().st_size}
                self.assertEqual(result.returncode, 0, diagnostic)
                self.assertIn("HTTP fixture complete", result.stdout)
                if dashboard:
                    report = json.loads(result.stdout)
                    self.assertTrue(report["embedded_assets"], report)
                    self.assertEqual([report[field] for field in ("http_root", "http_asset", "http_config")], [200, 200, 200])
                    self.assertIn(report["http_unauthorized"], (401, 403))
                turn_requests = [request for request in server.requests if request["body"].get("tools")]
                request_summary = [{"path": request["path"], "roles": [item.get("role") for item in request["body"].get("messages", [])],
                                    "has_tools": bool(request["body"].get("tools"))} for request in server.requests]
                self.assertEqual(len(turn_requests), 2, request_summary)
                self.assertTrue(any(item.get("role") == "tool" for item in turn_requests[1]["body"]["messages"]))
                with closing(sqlite3.connect(home / "state.db")) as database:
                    messages = database.execute("SELECT role, content FROM messages ORDER BY id").fetchall()
                    session_id = json.loads((scratch / "active-session.json").read_text())["session_id"]
                    self.assertIsNotNone(database.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone())
                    if not interactive:
                        workers = database.execute("SELECT status, last_sequence FROM worker_executions WHERE session_id=?",
                                                   (session_id,)).fetchall()
                        self.assertTrue(any(status == "terminal" and sequence > 0 for status, sequence in workers), workers)
                owned = [json.loads(line) for line in (scratch / "owned-processes.jsonl").read_text().splitlines()]
                self.assertGreaterEqual(len({record["pid"] for record in owned}), 2 if interactive else 3)
                self.assertIn(("assistant", "HTTP fixture complete"), messages)
                self.assertTrue(any(role == "tool" for role, _ in messages), messages)
                events_file = scratch / "guard-events.jsonl"
                events = [json.loads(line) for line in events_file.read_text().splitlines()] if events_file.exists() else []
                self.assertTrue(all(event.get("recoverable_capability_probe") for event in events), events)
            finally:
                self._stop_fixture_processes(scratch)
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def _stop_fixture_processes(self, scratch):
        records = scratch / "owned-processes.jsonl"
        if not records.exists():
            return
        for line in reversed(records.read_text().splitlines()):
            record = json.loads(line)
            try:
                process = processes.Process(record["pid"])
                if process.create_time() == record["birth"]:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except processes.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
            except processes.NoSuchProcess:
                pass


class TerminalEchoTests(unittest.TestCase):
    def test_actual_wrapped_cursor_movement_and_partial_echo_acknowledgement(self):
        import re
        namespace={"re":re}
        exec(_TERMINAL_ECHO,namespace)
        contains=namespace['observed_echo']
        prompt=b"Record the HTTP verification task as completed, then confirm completion."
        # This rendering came from the failed full-suite PTY. The terminal
        # advances over the space before 'confirm' instead of printing it.
        first=b"\x1b[0mRecord the HTTP verification task as completed, then"
        second=b"\x1b[Cconfirm completion.\x1b[0m\n\x1b[2A\x1b[16C"
        self.assertNotIn(prompt,first+second)
        self.assertFalse(contains(first,prompt))
        self.assertTrue(contains(first+second,prompt))
        self.assertFalse(contains(first+second,prompt.replace(b'completed',b'canceled')))
        self.assertTrue(contains(b'\x1b[0m/qu\x1b[?25h it',b'/quit'))


if __name__ == "__main__":
    unittest.main()
