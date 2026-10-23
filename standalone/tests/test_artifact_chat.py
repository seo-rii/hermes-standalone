"""Opt-in acceptance of the relocated, generated single-file CLI and workers."""

from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

from standalone import processes
from standalone.agent_probe_worker import fixture_environment
from standalone.tests import test_artifact
from standalone.tests.test_transport_agent import _AgentHTTPHandler


class ArtifactManagedChatTests(unittest.TestCase):
    def test_relocated_artifact_managed_gateway_http_tool_turn_and_durable_receipts(self):
        if test_artifact.ARTIFACT is None:
            self.skipTest("Pass --artifact to standalone/check.py to exercise the generated file")
        if sys.version_info < (3, 14):
            self.skipTest("The original Hermes runtime requires CPython 3.14 or later")
        with tempfile.TemporaryDirectory(prefix="hermes actual artifact chat ") as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            artifact = scratch / "copied hermes.py"
            shutil.copyfile(test_artifact.ARTIFACT, artifact)
            environment = fixture_environment(scratch)
            environment["PATH"] = ""
            environment["HERMES_TUI_ACTIVE_SESSION_FILE"] = str(scratch / "session.json")
            home = Path(environment["HERMES_HOME"])
            home.mkdir()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _AgentHTTPHandler)
            server.requests = []
            server.daemon_threads = True
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            endpoint = f"http://127.0.0.1:{server.server_port}/v1"
            (home / "config.yaml").write_text(json.dumps({
                "model": {"default": "standalone-wire-fixture", "provider": "custom", "base_url": endpoint,
                          "api_key": "local-http-fixture-token", "context_length": 131072},
                "agent": {"environment_probe": False, "api_max_retries": 1, "auto_recovery_cycles": 0,
                          "gateway_startup_warmup_timeout": 0},
                "gateway": {"multiplex_profiles": False, "loop_watchdog": False, "managed_workers": True},
                "compression": {"enabled": False}, "model_catalog": {"enabled": False},
                "tools": {"tool_search": {"enabled": False}},
                "telemetry": {"shared_metrics": {"enabled": False, "send": False, "offer_version": 2}},
            }), encoding="utf-8")
            (home / "models_dev_cache.json").write_text(json.dumps({
                "openai": {"id": "openai", "name": "Fixture registry", "models": {}}
            }), encoding="utf-8")
            command = [sys.executable, "-I", "-S", "-B", "-X", "utf8", str(artifact),
                       "--cache-dir", str(scratch / "cache"), "chat", "--provider", "custom",
                       "--model", "standalone-wire-fixture", "--base-url", endpoint,
                       "--api-key", "local-http-fixture-token", "--toolsets", "todo", "--ignore-rules",
                       "-Q", "-q", "Record the HTTP verification task as completed, then confirm completion."]
            owned = {}
            def identify(pid):
                try:
                    process = processes.Process(pid)
                    birth = process.create_time()
                    argv = process.cmdline()
                    if str(artifact) not in argv or Path(process.environ().get("HERMES_HOME", "")).resolve() != home.resolve():
                        return None
                    role = "worker" if "agent.managed_worker" in argv else "gateway" if (
                        "gateway.run" in argv or ("hermes_cli.main" in argv and "gateway" in argv)) else "cli"
                    if process.create_time() != birth:
                        return None
                    record = {"pid": pid, "birth": birth, "artifact": str(artifact), "home": str(home), "role": role}
                    previous = owned.get(pid)
                    if previous is not None and previous != record:
                        return None
                    owned[pid] = record
                    return process
                except (processes.Error, OSError, ValueError):
                    return None
            def observe():
                identify(child.pid)
                record_path = home / "gateway.pid"
                try:
                    record = json.loads(record_path.read_text())
                    if Path(record["hermes_home"]).resolve() == home.resolve():
                        process = identify(record["pid"])
                        if process is not None:
                            if sys.platform.startswith("linux"):
                                raw = Path(f"/proc/{process.pid}/stat").read_text()
                                token = int(raw[raw.rfind(")") + 2:].split()[19])
                            else:
                                token = int(process.create_time() * 100)
                            if token != record["start_time"] or owned[process.pid]["role"] != "gateway":
                                owned.pop(process.pid, None)
                except (OSError, ValueError, KeyError, TypeError, processes.Error):
                    pass
                for record in tuple(owned.values()):
                    process = identify(record["pid"])
                    if process is not None and process.create_time() == record["birth"]:
                        try:
                            for descendant in process.children(recursive=True):
                                identify(descendant.pid)
                        except processes.Error:
                            pass
                (scratch / "owned-processes.json").write_text(json.dumps(list(owned.values())), encoding="utf-8")
            def cleanup():
                # Each signal requires all four recorded ownership facts to
                # still match. Stop the gateway first so it reaps its workers.
                for record in sorted(owned.values(), key=lambda value: value["role"] != "gateway"):
                    process = identify(record["pid"])
                    if process is None or process.create_time() != record["birth"]:
                        continue
                    try:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except processes.TimeoutExpired:
                            process = identify(record["pid"])
                            if process is not None and process.create_time() == record["birth"]:
                                process.kill()
                                process.wait(timeout=5)
                    except processes.NoSuchProcess:
                        pass
            result = None
            out_path, err_path = scratch / "stdout.log", scratch / "stderr.log"
            for path in (out_path, err_path):
                path.touch(mode=0o600)
            try:
                with out_path.open("wb") as stdout, err_path.open("wb") as stderr:
                    child = subprocess.Popen(command, cwd=scratch / "work", env=environment, stdout=stdout, stderr=stderr)
                    identify(child.pid)
                    deadline = time.monotonic() + test_artifact.ARTIFACT_TIMEOUT
                    interval = threading.Event()
                    while child.poll() is None and time.monotonic() < deadline:
                        observe()
                        interval.wait(0.05)
                    observe()
                    result = child.poll()
                    if result is None:
                        cleanup()
                        child.wait(timeout=10)
                stdout = out_path.read_text(encoding="utf-8", errors="replace")
                stderr = err_path.read_text(encoding="utf-8", errors="replace")
                diagnostic = {"stdout": stdout[-2500:], "stderr": stderr[-2500:],
                              "gateway_log": (home / "logs/gateway-stdio.log").read_text(errors="replace")[-4000:]
                              if (home / "logs/gateway-stdio.log").exists() else "", "owned": list(owned.values())}
                self.assertEqual(result, 0, diagnostic)
                self.assertIn("HTTP fixture complete", stdout, diagnostic)
                self.assertTrue(any(record["role"] == "gateway" for record in owned.values()), diagnostic)
                self.assertTrue(any(record["role"] == "worker" for record in owned.values()), diagnostic)
                self.assertGreaterEqual(len(owned), 3, diagnostic)
                turn_requests = [request for request in server.requests if request["body"].get("tools")]
                self.assertEqual(len(turn_requests), 2)
                self.assertTrue(any(message.get("role") == "tool" for message in turn_requests[1]["body"]["messages"]))
                session = json.loads((scratch / "session.json").read_text())["session_id"]
                with closing(sqlite3.connect(home / "state.db")) as database:
                    messages = database.execute("SELECT role,content FROM messages WHERE session_id=?", (session,)).fetchall()
                    executions = database.execute("SELECT execution_id,status,last_sequence FROM worker_executions WHERE session_id=?", (session,)).fetchall()
                    self.assertIn(("assistant", "HTTP fixture complete"), messages)
                    self.assertTrue(any(role == "tool" for role, _ in messages))
                    self.assertTrue(any(status == "terminal" and sequence > 0 for _, status, sequence in executions), executions)
                    self.assertGreater(database.execute("SELECT count(*) FROM worker_receipts WHERE execution_id IN "
                                                        "(SELECT execution_id FROM worker_executions WHERE session_id=?)", (session,)).fetchone()[0], 0)
            finally:
                cleanup()
                if "child" in locals():
                    child.wait(timeout=10)
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
