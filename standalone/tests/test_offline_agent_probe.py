from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


STANDALONE = Path(__file__).resolve().parents[1]
PROJECT = STANDALONE.parent
spec = importlib.util.spec_from_file_location("offline_probe_under_test", STANDALONE / "offline_agent_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class OfflineProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="offline-probe-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "source"
        self.root.mkdir()
        builder = probe._load_helper("build")
        for name in builder.RUNTIME_DIRECTORIES + builder.RESOURCE_DIRECTORIES + ("standalone",):
            (self.root / name).mkdir()
        for name in builder.REQUIRED_ROOT_RESOURCES:
            (self.root / name).write_text("fixture\n", encoding="utf-8")
        for name, value in {
            "CONTRACT.json": {"project": "fixture"},
            "capabilities.json": {"agent_runtime_ready": False},
            "dependencies.json": {"format_version": 1, "packages": []},
        }.items():
            (self.root / "standalone" / name).write_text(json.dumps(value), encoding="utf-8")
        (self.root / "standalone" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "standalone" / "runtime.py").write_bytes((STANDALONE / "runtime.py").read_bytes())
        (self.root / "standalone" / "agent_probe_worker.py").write_bytes(
            (STANDALONE / "agent_probe_worker.py").read_bytes()
        )

    def run_source(self, source: str, *, stage="import") -> dict:
        (self.root / "run_agent.py").write_text(source, encoding="utf-8")
        return probe.run_probe(self.root, self.root / "absent-downloads", stage=stage)

    def test_constructor_probe_uses_temporary_state_and_reports_its_limited_scope(self):
        source = """
import os
from pathlib import Path
assert "PROBE_SECRET_SENTINEL" not in os.environ
assert Path(os.environ["HERMES_HOME"]).is_relative_to(Path(os.environ["HOME"]))
class AIAgent:
    def __init__(self, **kwargs):
        assert kwargs["enabled_toolsets"] == []
        assert kwargs["skip_context_files"] and kwargs["skip_memory"]
        self.fixture = Path(os.environ["HERMES_HOME"]) / "probe.txt"
        self.fixture.parent.mkdir(parents=True, exist_ok=True)
        self.fixture.write_text("temporary only")
    def close(self):
        self.fixture.unlink()
"""
        with mock.patch.dict(os.environ, {"PROBE_SECRET_SENTINEL": "private-fixture-value"}):
            report = self.run_source(source, stage="construct")
        self.assertEqual(report["status"], "passed", report)
        self.assertTrue(report["agent_imported"])
        self.assertTrue(report["agent_constructed"])
        self.assertTrue(report["original_constructor_verified"])
        self.assertFalse(report["agent_turn_verified"])
        self.assertFalse(report["live_provider_verified"])
        self.assertFalse(report["artifact_integrity_verified"])
        self.assertTrue(report["isolated"])
        self.assertEqual(report["verified_distributions"], 0)
        self.assertEqual(report["blocked_events"], [])
        self.assertNotIn("private-fixture-value", json.dumps(report))

    def test_missing_dependency_is_a_failure_instead_of_a_fake_agent(self):
        report = self.run_source("import missing_native_dependency_fixture\n")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["missing_module"], "missing_native_dependency_fixture")
        self.assertFalse(report["agent_imported"])
        self.assertFalse(report["agent_constructed"])
        self.assertTrue(any(frame["path"] == "source/run_agent.py" for frame in report["frames"]))

    def test_network_process_and_installer_attempts_are_blocked(self):
        cases = (
            ("import socket\nsocket.create_connection(('127.0.0.1', 9))\n", "network"),
            ("import subprocess\nsubprocess.run(['unexpected-installer'])\n", "process_execution"),
            ("import ensurepip\n", "package_installer_import"),
        )
        for source, operation in cases:
            with self.subTest(operation=operation):
                report = self.run_source(source)
                self.assertEqual(report["status"], "blocked", report)
                self.assertEqual(report["blocked_events"][-1]["operation"], operation)
                self.assertFalse(report["agent_imported"])

    def test_outside_files_are_neither_read_nor_modified(self):
        outside = Path(self.temporary.name) / "private-fixture.txt"
        outside.write_text("unchanged-private-fixture", encoding="utf-8")
        for mode, operation in (("r", "read_outside_fixture"), ("w", "write_outside_fixture")):
            with self.subTest(mode=mode):
                report = self.run_source(f"open({str(outside)!r}, {mode!r})\n")
                self.assertEqual(report["status"], "blocked", report)
                self.assertEqual(report["blocked_events"][-1]["operation"], operation)
                self.assertEqual(outside.read_text(encoding="utf-8"), "unchanged-private-fixture")
                self.assertNotIn("unchanged-private-fixture", json.dumps(report))

    def test_constructor_blocker_does_not_erase_successful_import(self):
        report = self.run_source("""
class AIAgent:
    def __init__(self, **kwargs):
        import socket
        socket.getaddrinfo("standalone.invalid", 443)
""", stage="construct")
        self.assertEqual(report["status"], "blocked", report)
        self.assertTrue(report["agent_imported"])
        self.assertFalse(report["agent_constructed"])

    def test_background_guard_failure_is_not_reported_as_success(self):
        report = self.run_source("""
import threading
def unexpected_background_operation():
    import socket
    socket.getaddrinfo("standalone.invalid", 443)
thread = threading.Thread(target=unexpected_background_operation)
thread.start()
thread.join()
""")
        self.assertEqual(report["status"], "blocked", report)
        self.assertTrue(report["agent_imported"])
        self.assertEqual(report["blocked_events"][-1]["operation"], "network")

    def test_fixture_cannot_write_bundled_sources(self):
        report = self.run_source("from pathlib import Path\nPath(__file__).write_text('changed')\n")
        self.assertEqual(report["status"], "blocked", report)
        self.assertEqual(report["blocked_events"][-1]["operation"], "bundled_source_write")

    def test_copy_reads_bundled_resource_but_cannot_replace_it(self):
        source = """
import os, shutil
from pathlib import Path
destination = Path(os.environ["HOME"]) / "copied-resource.py"
shutil.copy2(__file__, destination)
assert destination.read_bytes() == Path(__file__).read_bytes()
class AIAgent:
    pass
"""
        report = self.run_source(source)
        self.assertEqual(report["status"], "passed", report)
        report = self.run_source(source + "\nshutil.copy2(destination, __file__)\n")
        self.assertEqual(report["status"], "blocked", report)
        self.assertEqual(report["blocked_events"][-1]["operation"], "bundled_source_write")

    def test_runtime_worker_requires_existing_verified_binding(self):
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c",
             f"import sys; sys.path.insert(0, {str(PROJECT)!r}); "
             "from standalone.agent_probe_worker import main; main([])"],
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("require the verified standalone bundle runtime", result.stderr)

    def test_runtime_worker_reports_before_parent_removes_nested_profile(self):
        (self.root / "run_agent.py").write_text("""
import os
from pathlib import Path
state = Path(os.environ["HERMES_HOME"]) / "nested" / "probe.txt"
state.parent.mkdir(parents=True)
state.write_text("disposable fixture", encoding="utf-8")
""", encoding="utf-8")
        bundle = Path(self.temporary.name) / ("f" * 64)
        source = bundle / "app"
        source.mkdir(parents=True)
        (bundle / "vendor").mkdir()
        probe._source_snapshot(self.root, source)
        context = {
            "root": str(bundle), "artifact": str(STANDALONE / "agent_probe_worker.py"),
            "bundle_sha256": bundle.name, "manifest": {"project": "hermes-standalone"},
        }
        with tempfile.TemporaryDirectory(prefix="parent-owned-", dir=self.temporary.name) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            command = (
                f"import runpy, sys; sys.path.insert(0, {str(source)!r}); "
                "from standalone.runtime import initialize_bundle; "
                f"initialize_bundle({context!r}); "
                f"sys.argv = ['standalone.agent_probe_worker', '--scratch-root', {str(scratch)!r}]; "
                "runpy.run_module('standalone.agent_probe_worker', run_name='__main__')"
            )
            process = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", command],
                cwd=self.root, capture_output=True, text=True, encoding="utf-8", timeout=10,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(process.stdout)
            self.assertEqual(report["status"], "passed", report)
            self.assertEqual(report["blocked_events"], [])
            self.assertEqual(
                (scratch / "home" / "hermes" / "nested" / "probe.txt").read_text(encoding="utf-8"),
                "disposable fixture",
            )
        self.assertFalse(scratch.exists(), "Only the parent removes the completed worker's profile")

    def test_runtime_worker_rejects_preexisting_profile_state(self):
        worker = probe._load_helper("agent_probe_worker")
        scratch = Path(self.temporary.name) / "preexisting-profile"
        scratch.mkdir()
        for name in ("home", "work", "tmp"):
            (scratch / name).mkdir()
        (scratch / "home" / "config.yaml").write_text("fixture only", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be real and empty"):
            worker._validate_scratch(scratch, self.root)
        with self.assertRaisesRegex(ValueError, "separate from bundled sources"):
            worker._validate_scratch(self.root, self.root)


class FileSyncStandaloneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="file-sync-stdlib-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.vendor = cls.directory / "vendor"
        cls.vendor.mkdir()
        downloads = STANDALONE / ".cache" / "downloads"
        manifest = json.loads((STANDALONE / "dependencies.json").read_text(encoding="utf-8"))
        if any(not (downloads / item["filename"]).is_file() for item in manifest["packages"]):
            raise unittest.SkipTest("Reviewed cached wheels are required for the original file-sync integration")
        probe._vendor_sources(PROJECT, downloads, cls.vendor)

    def run_script(self, source: str) -> dict:
        home = self.directory / "home"
        home.mkdir(exist_ok=True)
        environment = {
            "HOME": str(home), "USERPROFILE": str(home), "HERMES_HOME": str(home / "hermes"),
        }
        for name in ("SYSTEMROOT", "WINDIR"):
            if name in os.environ:
                environment[name] = os.environ[name]
        prefix = f"import sys\nsys.path[:0] = [{str(PROJECT)!r}, {str(self.vendor)!r}]\n"
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", prefix + source],
            env=environment, cwd=self.directory, capture_output=True, text=True,
            encoding="utf-8", timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_original_sync_owner_probe_tracks_our_child_without_psutil(self):
        report = self.run_script("""
import json, subprocess
from tools.environments.file_sync import _temp_entry_owner_alive, _SYNC_BACK_TEMP_PREFIX
assert "psutil" not in sys.modules
child = subprocess.Popen([sys.executable, "-I", "-S", "-c", "import sys; sys.stdin.read()"],
                         stdin=subprocess.PIPE)
attempted = []
def deny_additional_processes(event, args):
    if event == "subprocess.Popen":
        attempted.append(event)
        raise RuntimeError("PID existence must not execute an external program")
sys.addaudithook(deny_additional_processes)
try:
    name = _SYNC_BACK_TEMP_PREFIX + str(child.pid) + "-fixture.tar"
    alive = _temp_entry_owner_alive(name)
    child.stdin.close()
    child.wait(timeout=5)
    dead = not _temp_entry_owner_alive(name)
    malformed = _temp_entry_owner_alive(_SYNC_BACK_TEMP_PREFIX + "legacy-name.tar")
    print(json.dumps({"alive": alive, "dead": dead, "malformed_preserved": malformed,
                      "additional_processes": len(attempted)}))
finally:
    if child.poll() is None:
        child.terminate()
        child.wait(timeout=5)
""")
        self.assertEqual(report, {
            "alive": True, "dead": True, "malformed_preserved": True, "additional_processes": 0,
        })

    @unittest.skipUnless(hasattr(os, "fork") and hasattr(os, "WNOWAIT"), "Requires POSIX unreaped-child observation")
    def test_unreaped_zombie_still_owns_its_sync_temp_entry(self):
        report = self.run_script("""
import json, os
from tools.environments.file_sync import _temp_entry_owner_alive, _SYNC_BACK_TEMP_PREFIX
pid = os.fork()
if pid == 0:
    os._exit(0)
try:
    exited = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
    name = _SYNC_BACK_TEMP_PREFIX + str(pid) + "-fixture.tar"
    print(json.dumps({"child_exited": exited.si_pid == pid,
                      "unreaped_pid_still_exists": _temp_entry_owner_alive(name)}))
finally:
    os.waitpid(pid, 0)
""")
        self.assertEqual(report, {"child_exited": True, "unreaped_pid_still_exists": True})

    def test_broken_installed_psutil_does_not_silently_select_the_fallback(self):
        report = self.run_script("""
import importlib.abc, json
class BrokenNativeDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "psutil":
            raise ModuleNotFoundError("broken native extension", name="psutil._psutil_linux")
sys.meta_path.insert(0, BrokenNativeDependency())
try:
    import tools.environments.file_sync
except ModuleNotFoundError as error:
    print(json.dumps({"missing": error.name}))
else:
    raise AssertionError("A broken installed dependency was silently ignored")
""")
        self.assertEqual(report, {"missing": "psutil._psutil_linux"})

    def test_original_agent_executes_tool_result_final_under_offline_fixture(self):
        report = probe.run_probe(PROJECT, STANDALONE / ".cache" / "downloads", stage="fixture-turn")
        self.assertEqual(report["status"], "passed", report)
        self.assertTrue(report["agent_imported"])
        self.assertTrue(report["agent_constructed"])
        self.assertTrue(report["fixture_turn_verified"])
        self.assertTrue(report["fixture_tool_result_verified"])
        self.assertTrue(report["fixture_todo_state_verified"])
        self.assertEqual(report["fixture_tool"], "todo_list")
        self.assertEqual(report["fixture_model_calls"], 2)
        self.assertEqual(report["fixture_conversation_roles"], ["user", "assistant", "tool", "assistant"])
        self.assertFalse(report["agent_turn_verified"])
        self.assertFalse(report["original_constructor_verified"])
        self.assertFalse(report["live_provider_verified"])
        self.assertTrue(all(item.get("recoverable_capability_probe") for item in report["blocked_events"]))
        self.assertEqual(report["diagnostic_stderr"], "")

    def test_original_constructor_uses_bundled_transport_without_sdk_shims(self):
        report = probe.run_probe(PROJECT, STANDALONE / ".cache" / "downloads", stage="construct", timeout=60)
        self.assertEqual(report["status"], "passed", report)
        self.assertTrue(report["agent_imported"])
        self.assertTrue(report["agent_constructed"])
        self.assertTrue(report["original_constructor_verified"])
        self.assertFalse(report["live_provider_verified"])
        self.assertFalse(report["fixture_turn_verified"])
        self.assertTrue(all(item.get("recoverable_capability_probe") for item in report["blocked_events"]))


if __name__ == "__main__":
    unittest.main()
