"""Original command dispatch and process re-entry under isolated bundled sources."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


STANDALONE = Path(__file__).resolve().parents[1]
PROJECT = STANDALONE.parent
spec = importlib.util.spec_from_file_location("entry_probe_support", STANDALONE / "offline_agent_probe.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)

_COMMAND_WORKER = r'''
import contextlib, io, json, os, pathlib, sys, traceback
context = json.loads(sys.argv[1])
arguments = json.loads(sys.argv[2])
scratch = pathlib.Path(sys.argv[3])
source = pathlib.Path(context["root"]) / "app"
vendor = source.parent / "vendor"
sys.path[:0] = [str(source), str(vendor)]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
from standalone.agent_probe_worker import ProbeGuard
os.chdir(scratch / "work")
guard = ProbeGuard(source, vendor, scratch)
sys.addaudithook(guard)
out, err = io.StringIO(), io.StringIO()
report = {"exit_code": 0}
try:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        sys.argv = ["hermes", *arguments]
        from standalone.cli import main
        report["exit_code"] = main(arguments, context)
except SystemExit as error:
    report["exit_code"] = error.code or 0
except BaseException as error:
    report.update(exit_code=1, error_type=type(error).__name__, error=str(error),
                  frames=[{"path": str(frame.filename), "line": frame.lineno, "function": frame.name}
                          for frame in traceback.StackSummary.extract(
                              traceback.walk_tb(error.__traceback__), lookup_lines=False)[-14:]])
report.update(stdout=out.getvalue(), stderr=err.getvalue(), blocked_events=guard.events)
print(json.dumps(report), flush=True)
os._exit(0)
'''

_INSPECT_WORKER = r'''
import json, os, pathlib, subprocess, sys
context = json.loads(sys.argv[1])
scratch = pathlib.Path(sys.argv[2])
source = pathlib.Path(context["root"]) / "app"
sys.path[:0] = [str(source), str(source.parent / "vendor")]
from standalone.runtime import initialize_bundle, bundled_command
initialize_bundle(context)
from standalone.agent_probe_worker import ProbeGuard
guard = ProbeGuard(source, source.parent / "vendor", scratch)
sys.addaudithook(guard)
from hermes_cli._launchers import runtime_command, installation_command
from hermes_cli.gateway_runtime_service_identity import verify_gateway_argv
from gateway.status import looks_like_gateway_command_line
from gateway.runtime_entry import hermes_command
from hermes_state_holders import _looks_like_hermes
from hermes_cli.gateway_windows import _gateway_run_argv
profile = scratch / "home" / ".hermes" / "profiles" / "research"
arguments = ["--profile", "research", "gateway", "run"]
gateway = runtime_command(source, arguments, home=profile)
verify_gateway_argv(gateway, profile)
wrong_profile_rejected = False
try:
    verify_gateway_argv(gateway, profile.with_name("other"))
except ValueError:
    wrong_profile_rejected = True
watcher = bundled_command(source, "gateway.restart_watcher", ["1", "0", *gateway])
foreign = list(gateway)
foreign[6] = str(scratch / "foreign.py")
foreign_rejected = False
try:
    verify_gateway_argv(foreign, profile)
except ValueError:
    foreign_rejected = True
query = runtime_command(source, ["config", "get", "terminal.backend", "--json"],
                        home=scratch / "home" / "fallback")
report = {
    "query": query, "gateway": gateway,
    "same_installation_command": installation_command(source, arguments, home=profile) == gateway,
    "same_restart_command": hermes_command() == bundled_command(source, "hermes_cli.main"),
    "windows_command": _gateway_run_argv(sys.executable, "--profile research"),
    "gateway_recognized": looks_like_gateway_command_line(subprocess.list2cmdline(gateway)),
    "direct_recognized": looks_like_gateway_command_line(subprocess.list2cmdline(
        bundled_command(source, "gateway.run"))),
    "watcher_rejected": not looks_like_gateway_command_line(subprocess.list2cmdline(watcher)),
    "wrong_profile_rejected": wrong_profile_rejected, "foreign_rejected": foreign_rejected,
    "holder_recognized": _looks_like_hermes(gateway), "foreign_holder_rejected": not _looks_like_hermes(foreign),
    "blocked_events": guard.events,
}
print(json.dumps(report), flush=True)
os._exit(0)
'''

_FIXTURE_ARTIFACT = r'''
import contextlib, io, json, os, pathlib, runpy, sys, traceback
context = json.loads(CONTEXT_JSON)
source = pathlib.Path(context["root"]) / "app"
sys.path[:0] = [str(source), str(source.parent / "vendor")]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
from standalone.agent_probe_worker import ProbeGuard
scratch = pathlib.Path(os.environ["ENTRY_FIXTURE_SCRATCH"])
guard = ProbeGuard(source, source.parent / "vendor", scratch)
sys.addaudithook(guard)
arguments = sys.argv[1:]
assert arguments[:1] == ["--cache-dir"]
assert arguments[2:3] == ["--internal-module"]
module = arguments[3]
assert module in context["manifest"]["internal_modules"]
if module == "agent.managed_worker":
    sys.argv = [module, *arguments[4:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    raise AssertionError("managed worker did not exit")
if module == "standalone.profile_worker":
    sys.argv = [module, *arguments[4:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    os._exit(0)
out, err = io.StringIO(), io.StringIO()
report = {"exit_code": 0}
try:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        sys.argv = [module, *arguments[4:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
except SystemExit as error:
    report["exit_code"] = error.code or 0
except BaseException as error:
    report.update(exit_code=1, error_type=type(error).__name__, error=str(error),
                  traceback=traceback.format_exc())
report.update(stdout=out.getvalue(), stderr=err.getvalue(), blocked_events=guard.events)
print(json.dumps(report), flush=True)
os._exit(0)
'''

_MANAGED_VERIFY_WORKER = r'''
import json, os, pathlib, subprocess, sys
context = json.loads(sys.argv[1])
scratch = pathlib.Path(sys.argv[2])
source = pathlib.Path(context["root"]) / "app"
sys.path[:0] = [str(source), str(source.parent / "vendor")]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
from standalone.agent_probe_worker import ProbeGuard, fixture_environment
from hermes_cli.runtime_module import python_module_command
environment = fixture_environment(scratch)
environment["ENTRY_FIXTURE_SCRATCH"] = str(scratch)
child = subprocess.Popen(python_module_command("agent.managed_worker"), stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, cwd=scratch / "work")
try:
    guard = ProbeGuard(source, source.parent / "vendor", scratch)
    sys.addaudithook(guard)
    from agent.managed_worker import read_frame
    from gateway.session_worker_reservation import verify_worker_identity
    from hermes_state_runtime import RuntimeStoreError
    hello = read_frame(child.stdout)
    pid, birth = verify_worker_identity(child, hello)
    stale = dict(hello, birth=hello["birth"] - 1)
    stale_rejected = False
    try:
        verify_worker_identity(child, stale)
    except RuntimeStoreError as error:
        stale_rejected = error.reason == "permission_denied"
    report = {"worker_pid_matches_handle": pid == child.pid, "birth": birth,
              "stale_birth_rejected": stale_rejected, "worker_still_alive": child.poll() is None,
              "no_global_psutil": "psutil" not in sys.modules, "blocked_events": guard.events}
    print(json.dumps(report), flush=True)
finally:
    child.terminate()
    child.communicate(timeout=5)
os._exit(0)
'''

_GATEWAY_WORKER = r'''
import contextlib, io, json, os, pathlib, sys, traceback
context = json.loads(sys.argv[1])
scratch = pathlib.Path(sys.argv[2])
source = pathlib.Path(context["root"]) / "app"
sys.path[:0] = [str(source), str(source.parent / "vendor")]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
from standalone.agent_probe_worker import ProbeGuard
guard = ProbeGuard(source, source.parent / "vendor", scratch)
sys.addaudithook(guard)
out, err = io.StringIO(), io.StringIO()
report = {"exit_code": 0}
try:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        import asyncio
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig, Platform
        from gateway.session import SessionSource, SessionStore
        home = pathlib.Path(os.environ["HERMES_HOME"])
        config = GatewayConfig(sessions_dir=home / "sessions", multiplex_profiles=False, loop_watchdog=False)
        runner = GatewayRunner(config)
        report["gateway_constructed"] = True
        source_event = SessionSource(platform=Platform.TELEGRAM, chat_id="offline-fixture", user_id="offline-user")
        entry = runner.session_store.get_or_create_session(source_event)
        session_id, session_key = entry.session_id, entry.session_key
        runner.session_store.close_all_db_handles()
        reopened = SessionStore(home / "sessions", config)
        restored = reopened.get_or_create_session(source_event)
        report["session_restored"] = restored.session_id == session_id and restored.session_key == session_key
        reopened.close_all_db_handles()
        async def lifecycle():
            report["gateway_started"] = await runner.start()
            report["running_before_stop"] = runner._running
            service = await runner._ensure_hosted_room_worker()
            report["hosted_room_started"] = service is not None and service.runtime.status().get("running")
            await runner.stop()
            report["stopped"] = not runner._running
            report["hosted_room_stopped"] = not service.runtime.status().get("running")
        asyncio.run(lifecycle())
except BaseException as error:
    report.update(exit_code=1, error_type=type(error).__name__, error=str(error),
                  frames=[{"path": str(frame.filename), "line": frame.lineno, "function": frame.name}
                          for frame in traceback.StackSummary.extract(
                              traceback.walk_tb(error.__traceback__), lookup_lines=False)[-14:]])
report.update(stdout=out.getvalue(), stderr=err.getvalue(), blocked_events=guard.events)
print(json.dumps(report), flush=True)
os._exit(0)
'''

_DASHBOARD_WORKER = r'''
import contextlib, io, json, os, pathlib, sys, types
import zoneinfo
zoneinfo.reset_tzpath(())
context = json.loads(sys.argv[1])
scratch = pathlib.Path(sys.argv[2])
source = pathlib.Path(context["root"]) / "app"
sys.path[:0] = [str(source), str(source.parent / "vendor")]
from standalone.runtime import initialize_bundle, bundle_module_argv
initialize_bundle(context)
from standalone.agent_probe_worker import ProbeGuard
guard = ProbeGuard(source, source.parent / "vendor", scratch)
sys.addaudithook(guard)
from hermes_cli.main_dashboard import _resolve_dashboard_web_dist
from hermes_cli.main_tui_launch import _resolve_use_tui
from hermes_cli.web_server_chat import _resolve_chat_argv
args = types.SimpleNamespace(skip_build=False)
dist = scratch / "web-assets"
dist.mkdir()
(dist / "index.html").write_text("<!doctype html><title>External fixture</title>")
os.environ["HERMES_WEB_DIST"] = str(dist)
_resolve_dashboard_web_dist(args, False)
external = os.environ["HERMES_WEB_DIST"] == str(dist)
os.environ["HERMES_WEB_DIST"] = str(dist / "absent")
missing_refused = False
with contextlib.redirect_stderr(io.StringIO()):
    try:
        _resolve_dashboard_web_dist(args, False)
    except SystemExit as error:
        missing_refused = error.code == 1
os.environ["HERMES_TUI"] = "1"
classic = _resolve_use_tui(types.SimpleNamespace()) is False
argv, cwd, env = _resolve_chat_argv(workspace_cwd=str(scratch / "work"),
                                   active_session_file=str(scratch / "active-session.json"))
module, arguments = bundle_module_argv(argv)
from hermes_cli.standalone_terminal import write_active_session
os.environ["HERMES_TUI_ACTIVE_SESSION_FILE"] = env["HERMES_TUI_ACTIVE_SESSION_FILE"]
write_active_session("local-fixture-canonical-id")
report = {"external_assets_selected": external, "missing_assets_refused": missing_refused,
          "classic_selected": classic, "module": module, "arguments": arguments, "cwd": cwd,
          "profile_preserved": env["HERMES_HOME"] == os.environ["HERMES_HOME"],
          "marker": json.loads((scratch / "active-session.json").read_text()), "blocked_events": guard.events}
print(json.dumps(report), flush=True)
os._exit(0)
'''

_PROFILE_SEED_WORKER = r'''
import json, os, pathlib, sys
context = json.loads(sys.argv[1])
scratch = pathlib.Path(sys.argv[2])
source = pathlib.Path(context["root"]) / "app"
sys.path[:0] = [str(source), str(source.parent / "vendor")]
from standalone.runtime import initialize_bundle, bundle_module_argv
initialize_bundle(context)
from standalone.agent_probe_worker import ProbeGuard
class Guard(ProbeGuard):
    def __call__(self, event, args):
        if event == "subprocess.Popen" and isinstance(args[1], (list, tuple)) and bundle_module_argv(args[1]) is not None:
            return
        super().__call__(event, args)
guard = Guard(source, source.parent / "vendor", scratch)
sys.addaudithook(guard)
from hermes_cli.profiles import seed_profile_skills
os.environ["ENTRY_FIXTURE_SCRATCH"] = str(scratch)
profile = scratch / "home" / "selected"
profile.mkdir()
(profile / "config.yaml").write_text('{"skills": {"bundled": true}}')
launch_skills = pathlib.Path(os.environ["HERMES_HOME"]) / "skills"
before = sorted(path.relative_to(launch_skills).as_posix() for path in launch_skills.rglob("SKILL.md"))
result = seed_profile_skills(profile, quiet=True)
assert result is not None and result.get("total_bundled", 0) > 0, result
assert list((profile / "skills").rglob("SKILL.md"))
assert sorted(path.relative_to(launch_skills).as_posix() for path in launch_skills.rglob("SKILL.md")) == before
print(json.dumps({"selected_home_seeded": True, "blocked_events": guard.events}), flush=True)
os._exit(0)
'''

_PROCESS_LIFECYCLE_WORKER = r'''
import json, os, pathlib, subprocess, sys, time
context = json.loads(sys.argv[1])
scratch = pathlib.Path(sys.argv[2])
source = pathlib.Path(context["root"]) / "app"
sys.path[:0] = [str(source), str(source.parent / "vendor")]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
from standalone import processes
script = """
import json,os,pathlib,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[1])
child=subprocess.Popen([sys.executable,'-I','-S','-B','-c','import time; time.sleep(60)'])
def stop(sig,frame):
    (root/'parent-term.json').write_text(json.dumps({'child_alive':child.poll() is None}))
    child.terminate()
    child.wait(timeout=3)
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
print(json.dumps({'child':child.pid}),flush=True)
while True:time.sleep(0.01)
"""
child = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", script, str(scratch)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=dict(os.environ), cwd=scratch / "work")
grandchild = json.loads(child.stdout.readline())["child"]
from standalone.agent_probe_worker import ProbeGuard
guard = ProbeGuard(source, source.parent / "vendor", scratch)
sys.addaudithook(guard)
try:
    from tools.process_registry_termination import ProcessTerminationMixin
    class Owner(ProcessTerminationMixin):
        @staticmethod
        def _host_pid_is_ours(pid, expected):
            return processes.Process(pid).create_time() == expected
        @staticmethod
        def _daemon_term_grace_seconds():
            return 1.0
    birth = processes.Process(child.pid).create_time()
    Owner._terminate_host_pid(child.pid, birth - 1)
    assert child.poll() is None
    Owner._terminate_host_pid(child.pid, birth)
    child.wait(timeout=5)
    assert child.returncode == 0
    assert json.loads((scratch / "parent-term.json").read_text())["child_alive"] is True
    assert not processes.pid_exists(grandchild)
    assert "psutil" not in sys.modules
    print(json.dumps({"parent_reaped_own_child": True, "stale_identity_refused": True,
                      "blocked_events": guard.events}), flush=True)
finally:
    for pid in (grandchild, child.pid):
        try:
            process = processes.Process(pid)
            process.kill()
            process.wait(timeout=3)
        except processes.NoSuchProcess:
            pass
    child.communicate(timeout=3)
os._exit(0)
'''


class BundledSourceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="hermes-entry-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        downloads = STANDALONE / ".cache" / "downloads"
        manifest = json.loads((STANDALONE / "dependencies.json").read_text(encoding="utf-8"))
        if any(not (downloads / item["filename"]).is_file() for item in manifest["packages"]):
            raise unittest.SkipTest("Reviewed cached dependencies are needed for original entry tests")
        source, vendor = cls.directory / "app", cls.directory / "vendor"
        source.mkdir()
        vendor.mkdir()
        support._vendor_sources(PROJECT, downloads, vendor)
        _, digest = support._source_snapshot(PROJECT, source)
        cls.bundle = cls.directory / digest
        cls.bundle.mkdir()
        source.rename(cls.bundle / "app")
        vendor.rename(cls.bundle / "vendor")
        cls.context = {
            "root": str(cls.bundle), "artifact": str(cls.directory / "hermes.py"),
            "bundle_sha256": digest,
            "manifest": {"project": "hermes-standalone", "internal_modules": [
                "hermes_cli.main", "gateway.run", "standalone.entry_worker", "gateway.restart_watcher",
                "agent.managed_worker", "hermes_cli.kanban_worker_client",
                "tools.mcp_death_supervisor",
                "standalone.profile_worker",
                "browser_harness.run", "browser_harness.daemon",
            ]},
        }
        Path(cls.context["artifact"]).write_text(
            _FIXTURE_ARTIFACT.replace("CONTEXT_JSON", repr(json.dumps(cls.context))), encoding="utf-8",
        )


class OriginalCommandTests(BundledSourceTestCase):
    def invoke(self, arguments: list[str], *, config: dict | None = None) -> dict:
        with tempfile.TemporaryDirectory(prefix="command-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            if config is not None:
                profile = scratch / "home" / "hermes"
                profile.mkdir()
                (profile / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            process = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", _COMMAND_WORKER,
                 json.dumps(self.context), json.dumps(arguments), str(scratch)],
                env=environment, cwd=scratch / "work", capture_output=True, text=True,
                encoding="utf-8", timeout=30,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(process.stdout)
            report["expected_home"] = str(scratch / "home" / "hermes")
            return report

    def assert_success(self, report: dict) -> None:
        self.assertEqual(report["exit_code"], 0, report)
        self.assertTrue(all(event.get("recoverable_capability_probe") for event in report["blocked_events"]), report)

    @unittest.skipUnless(sys.version_info < (3, 14), "Only older bootstrap interpreters are refused")
    def test_commands_reject_unsupported_python_before_original_imports(self):
        report = self.invoke(["--help"])
        self.assertEqual(report["exit_code"], 2, report)
        self.assertIn("require Python 3.14", report["stderr"])
        self.assertEqual(report["blocked_events"], [])

    @unittest.skipUnless(sys.version_info >= (3, 14), "Original Hermes commands require Python 3.14")
    def test_original_help_lists_real_commands(self):
        report = self.invoke(["--help"])
        self.assert_success(report)
        self.assertIn("sessions", report["stdout"])
        self.assertIn("gateway", report["stdout"])

    @unittest.skipUnless(sys.version_info >= (3, 14), "Original Hermes commands require Python 3.14")
    def test_original_config_reads_only_temporary_profile(self):
        report = self.invoke(["config", "get", "terminal.backend", "--json"], config={"terminal": {"backend": "local"}})
        self.assert_success(report)
        self.assertEqual(json.loads(report["stdout"]), "local")

    @unittest.skipUnless(sys.version_info >= (3, 14), "Original Hermes commands require Python 3.14")
    def test_original_sessions_lists_empty_temporary_database(self):
        report = self.invoke(["sessions", "list"])
        self.assert_success(report)
        self.assertIn("session", report["stdout"].lower())

    def test_gateway_commands_keep_canonical_process_and_profile_identity(self):
        with tempfile.TemporaryDirectory(prefix="identity-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", _INSPECT_WORKER,
                 json.dumps(self.context), str(scratch)], env=environment, cwd=scratch / "work",
                capture_output=True, text=True, encoding="utf-8", timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            for field in ("same_installation_command", "same_restart_command", "gateway_recognized",
                          "direct_recognized", "watcher_rejected", "wrong_profile_rejected", "foreign_rejected",
                          "holder_recognized", "foreign_holder_rejected"):
                self.assertTrue(report[field], (field, report))
            self.assertEqual(report["windows_command"][6], self.context["artifact"])
            self.assertEqual(report["blocked_events"], [])

    def test_real_child_reentry_reads_home_fallback_and_explicit_profile(self):
        with tempfile.TemporaryDirectory(prefix="reentry-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", _INSPECT_WORKER,
                 json.dumps(self.context), str(scratch)], env=environment, cwd=scratch / "work",
                capture_output=True, text=True, encoding="utf-8", timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            command = json.loads(result.stdout)["query"]
            environment["ENTRY_FIXTURE_SCRATCH"] = str(scratch)
            for home_name in ("fallback", "selected"):
                home = scratch / "home" / home_name
                home.mkdir()
                (home / "config.yaml").write_text(json.dumps({"terminal": {"backend": home_name}}), encoding="utf-8")
            for explicit_home in (None, scratch / "home" / "selected"):
                with self.subTest(explicit_home=explicit_home):
                    environment.pop("HERMES_HOME", None)
                    if explicit_home is not None:
                        environment["HERMES_HOME"] = str(explicit_home)
                    result = subprocess.run(command, env=environment, cwd=scratch / "work",
                                            capture_output=True, text=True, encoding="utf-8", timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assert_success(report)
                    self.assertEqual(json.loads(report["stdout"]), "selected" if explicit_home else "fallback")

    def invoke_worker(self, worker, *, config=None):
        with tempfile.TemporaryDirectory(prefix="worker-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            if config is not None:
                profile = scratch / "home" / "hermes"
                profile.mkdir()
                (profile / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", worker,
                 json.dumps(self.context), str(scratch)], env=environment, cwd=scratch / "work",
                capture_output=True, text=True, encoding="utf-8", timeout=45,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_original_managed_worker_hello_is_verified_by_actual_owner(self):
        report = self.invoke_worker(_MANAGED_VERIFY_WORKER)
        for field in ("worker_pid_matches_handle", "stale_birth_rejected", "worker_still_alive", "no_global_psutil"):
            self.assertTrue(report[field], report)
        self.assertEqual(report["blocked_events"], [])

    def test_original_gateway_start_stop_and_session_persistence(self):
        report = self.invoke_worker(_GATEWAY_WORKER, config={
            "agent": {"gateway_startup_warmup_timeout": 0},
            "gateway": {"multiplex_profiles": False, "loop_watchdog": False},
        })
        self.assert_success(report)
        for field in ("gateway_constructed", "session_restored", "gateway_started", "running_before_stop", "stopped",
                      "hosted_room_started", "hosted_room_stopped"):
            self.assertTrue(report[field], report)

    def test_dashboard_uses_python_terminal_and_validates_prebuilt_assets(self):
        report = self.invoke_worker(_DASHBOARD_WORKER)
        for field in ("external_assets_selected", "missing_assets_refused", "classic_selected", "profile_preserved"):
            self.assertTrue(report[field], report)
        self.assertEqual(report["module"], "hermes_cli.main")
        self.assertEqual(report["arguments"][:2], ["chat", "--cli"])
        self.assertEqual(report["arguments"][2:], ["--in", report["cwd"]])
        self.assertEqual(report["marker"], {"session_id": "local-fixture-canonical-id"})
        self.assertTrue(all(item.get("recoverable_capability_probe") for item in report["blocked_events"]), report)

    def test_original_profile_skill_seeding_reenters_bound_runtime_in_selected_home(self):
        report = self.invoke_worker(_PROFILE_SEED_WORKER)
        self.assertTrue(report["selected_home_seeded"], report)
        self.assertEqual(report["blocked_events"], [])

    @unittest.skipIf(sys.platform == "win32", "Original POSIX parent-first termination path")
    def test_original_terminal_teardown_preserves_parent_reaping_and_refuses_stale_birth(self):
        report = self.invoke_worker(_PROCESS_LIFECYCLE_WORKER)
        self.assertTrue(report["parent_reaped_own_child"], report)
        self.assertTrue(report["stale_identity_refused"], report)
        self.assertEqual(report["blocked_events"], [])


if __name__ == "__main__":
    unittest.main()
