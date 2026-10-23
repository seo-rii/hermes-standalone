"""Original Desktop SSH trust checks and isolated artifact launch ownership.

Linux/macOS cases exercise the platform-independent selectors and real process
birth/argv checks. Native secure-file lifecycle runs only on actual Windows.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hermes_cli import windows_ssh_runtime as ssh
from standalone import windows_files


PROJECT = Path(__file__).resolve().parents[2]
OWNER, NONCE = "a" * 32, "0123456789abcdef"


class SSHBoundaryTests(unittest.TestCase):
    def test_ordinary_source_preserves_its_win32_selection_seam(self):
        source_file, source_constants = object(), object()
        with patch.object(ssh, "_bundle_runtime", return_value=None), patch.object(
            ssh, "_win32", return_value=SimpleNamespace(win32file=source_file, win32con=source_constants)
        ):
            self.assertIs(ssh._file_api(), source_file)
            self.assertIs(ssh._constants(), source_constants)
            self.assertEqual(ssh._installation_command("/unverified/other.py"), ["/unverified/other.py"])
        if sys.platform != "win32":
            with self.assertRaisesRegex(RuntimeError, "only available on Windows"):
                ssh._win32()

    def test_original_token_path_nonce_length_and_delete_flags_are_preserved(self):
        with tempfile.TemporaryDirectory(prefix="hermes-ssh-token-guard-") as temporary:
            root = Path(temporary)
            path = root / OWNER / (NONCE + ".token")
            opened, closed = [], []
            api = SimpleNamespace(ReadFile=lambda handle, limit: (0, b"b" * 64), CloseHandle=closed.append)
            def open_file(path, access, creation, flags, share=0):
                opened.append((path, access, creation, flags, share))
                return 123
            with patch.object(ssh, "_root", return_value=root), patch.object(ssh, "_constants", return_value=windows_files.constants), \
                 patch.object(ssh, "_file_api", return_value=api), patch.object(ssh, "_open", side_effect=open_file):
                self.assertEqual(ssh.read_token(str(path)), "b" * 64)
                self.assertEqual(closed, [123])
                self.assertTrue(opened[0][1] & windows_files.constants.DELETE)
                self.assertEqual(opened[0][3] & (0x04000000 | 0x00200000), 0x04000000 | 0x00200000)
                for content in (b"b" * 63, b"b" * 65, b"b" * 64 + b"\xff", b"G" * 64, b"b" * 63 + b"\xff"):
                    api.ReadFile = lambda handle, limit, content=content: (0, content)
                    with self.subTest(length=len(content)), self.assertRaises(SystemExit):
                        ssh.read_token(str(path))
                before = len(opened)
                for candidate in (root.parent / "outside.token", root / ".." / path.name,
                                  root / OWNER / ("Z" * 16 + ".token"), root / "wrong-owner" / path.name,
                                  root / OWNER / "deeper" / path.name):
                    with self.subTest(path=candidate), self.assertRaises(SystemExit):
                        ssh.read_token(str(candidate))
                self.assertEqual(len(opened), before)
                for owner, nonce, token in (("invalid", NONCE, b"b" * 64),
                                            (OWNER, "invalid", b"b" * 64),
                                            (OWNER, NONCE, b"b" * 65)):
                    with self.assertRaises(ValueError):
                        ssh.upload_token(owner, nonce, token)


_FIXTURE = r'''
import json, pathlib, sys, time
args = sys.argv[1:]
if "--version" in args:
    print("Hermes SSH fixture")
elif "--help" in args:
    print("serve --ssh-session-token-file --ssh-owner-nonce")
elif "serve" in args:
    print("READY", flush=True)
    time.sleep(60)
else:
    raise SystemExit("Unexpected fixture entry arguments")
'''


class SSHRuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hermes ssh binding 한글 ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.digest = "c" * 64
        self.root = self.directory / self.digest
        self.app = self.root / "app"
        for name in ("standalone", "hermes_cli", "hermes_platform"):
            (self.app / name).mkdir(parents=True)
        for name in (
            "standalone/__init__.py", "standalone/runtime.py", "standalone/windows_acl.py",
            "standalone/windows_files.py", "standalone/processes.py", "standalone/process_errors.py",
            "standalone/process_linux.py", "standalone/process_darwin.py", "standalone/process_windows.py",
            "hermes_cli/__init__.py", "hermes_cli/windows_ssh_runtime.py",
            "hermes_platform/__init__.py", "hermes_platform/processes.py", "hermes_constants.py",
        ):
            shutil.copyfile(PROJECT / name, self.app / name)
        self.artifact = self.directory / "hermes fixture.py"
        self.artifact.write_text(_FIXTURE, encoding="utf-8")

    def invoke(self, body):
        context = {"root": str(self.root), "artifact": str(self.artifact), "bundle_sha256": self.digest,
                   "manifest": {"project": "hermes-standalone", "internal_modules": ["hermes_cli.main"]}}
        environment = {"PATH": "", "HOME": str(self.directory / "home"),
                       "USERPROFILE": str(self.directory / "home"), "HERMES_HOME": str(self.directory / "profile"),
                       "PYTHONUTF8": "1", "LANG": "C.UTF-8"}
        for name in ("SystemRoot", "WINDIR", "SYSTEMDRIVE", "TEMP", "TMP"):
            if name in os.environ:
                environment[name] = os.environ[name]
        worker = r'''
import importlib.abc, json, os, pathlib, subprocess, sys
context = json.loads(sys.argv[1])
source = pathlib.Path(context["root"]) / "app"
sys.path.insert(0, str(source))
from standalone import runtime
runtime.initialize_bundle(context)
from hermes_cli import windows_ssh_runtime as ssh
class NoNativePackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("psutil", "win32api", "win32con", "win32file", "win32security", "pywintypes", "ntsecuritycon"):
            raise AssertionError("Unqualified native dependency reached: " + fullname)
sys.meta_path.insert(0, NoNativePackages())
def no_network(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "os.system"):
        raise AssertionError("Unexpected external operation: " + event)
sys.addaudithook(no_network)
'''
        return subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", worker + body,
                               json.dumps(context)], cwd=self.directory, env=environment,
                              capture_output=True, text=True, encoding="utf-8", timeout=35)

    def test_bound_artifact_resolution_inspection_and_real_process_identity(self):
        result = self.invoke(r'''
from standalone import windows_files
from hermes_platform.processes import process_api
assert ssh._file_api() is windows_files
assert ssh._constants() is windows_files.constants
artifact = context["artifact"]
command = ssh._resolve_direct_command(artifact)
assert runtime.bundle_module_argv(command) == ("hermes_cli.main", [])
assert command[:6] == [sys.executable, "-I", "-S", "-B", "-X", "utf8"]
assert ssh._installation_command(str(pathlib.Path(artifact).with_name("foreign.py"))) == [str(pathlib.Path(artifact).with_name("foreign.py"))]
report = ssh.inspect_hermes(artifact)
assert report["supported"] and report["version"] == "Hermes SSH fixture", report
nonce = "0123456789abcdef"
args = ["--profile", "research", "serve", "--isolated", "--ssh-owner-nonce", nonce]
child = subprocess.Popen([*command, *args], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
try:
    assert child.stdout.readline().strip() == b"READY"
    created = int(process_api().Process(child.pid).create_time() * 1_000_000_000)
    state = ssh.process_state(child.pid, created, artifact, nonce)
    assert state["alive"] and state["owned"] and not state["indeterminate"], state
    assert not ssh.process_state(child.pid, created + 1, artifact, nonce)["owned"]
    assert not ssh.process_state(child.pid, created, artifact, "fedcba9876543210")["owned"]
    # A source-style launch carrying the same nonce is not this artifact.
    source_child = subprocess.Popen([sys.executable, "-I", "-S", "-c", "import time; time.sleep(60) # hermes_cli.main", *args],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        source_created = int(process_api().Process(source_child.pid).create_time() * 1_000_000_000)
        assert not ssh.process_state(source_child.pid, source_created, artifact, nonce)["owned"]
        assert not ssh.terminate_owned(source_child.pid, source_created, artifact, nonce)
        assert source_child.poll() is None
    finally:
        source_child.terminate()
        source_child.wait(5)
    assert not ssh.terminate_owned(child.pid, created + 1, artifact, nonce)
    assert child.poll() is None
    assert ssh.terminate_owned(child.pid, created, artifact, nonce)
    child.wait(5)
finally:
    if child.poll() is None:
        child.kill()
        child.wait(5)
    child.stdout.close()
    child.stderr.close()
print("bound artifact identity and native dependency isolation passed")
''')
        self.assertEqual(result.returncode, 0, result.stderr[-5000:])
        self.assertIn("bound artifact identity", result.stdout)

    @unittest.skipUnless(os.name == "nt", "Requires actual Windows secure files and detached process creation")
    def test_native_original_token_lock_log_and_backend_lifecycle(self):
        result = self.invoke(r'''
from hermes_platform.processes import process_api
owner, nonce = "a" * 32, "0123456789abcdef"
token = "b" * 64
record = ssh.upload_token(owner, nonce, token.encode())
assert pathlib.Path(record["path"]).is_file()
assert ssh.read_token(record["path"]) == token
assert not pathlib.Path(record["path"]).exists()
assert not ssh.remove_artifact(pathlib.Path(record["path"]))
ssh.upload_token(owner, nonce, token.encode())
spawn = ssh.spawn_backend({"ownershipId": owner, "spawnNonce": nonce,
                           "hermesPath": context["artifact"], "profile": "research"})
try:
    created = int(spawn["creationTimeNs"])
    assert ssh.process_state(spawn["pid"], created, context["artifact"], nonce)["owned"]
    ssh.write_lock(owner, {"spawnNonce": nonce, "pid": spawn["pid"], "creationTimeNs": str(created)})
    lock = ssh.read_lock(owner)
    assert lock["pid"] == spawn["pid"] and lock["logPath"].endswith("/" + owner + "/" + nonce + ".log")
    import time
    deadline = time.monotonic() + 5
    while "READY" not in ssh._read_log(owner, nonce)["content"]:
        if time.monotonic() >= deadline:
            raise AssertionError("Detached fixture did not produce its log")
        time.sleep(.05)
finally:
    ssh.terminate_owned(spawn["pid"], int(spawn["creationTimeNs"]), context["artifact"], nonce)
assert ssh.remove_artifact(ssh._lock_path(owner))
assert ssh.remove_artifact(ssh._token_path(owner, nonce))
assert ssh.remove_artifact(ssh._log_path(owner, nonce))
print("native original token, lock, log and detached backend passed")
''')
        self.assertEqual(result.returncode, 0, result.stderr[-5000:])
        self.assertIn("native original token", result.stdout)
