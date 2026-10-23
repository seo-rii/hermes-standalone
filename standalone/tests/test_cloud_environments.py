"""Original cloud shell environments over actual loopback REST/toolbox traffic.

The fixture server executes only this test's generated shell scripts in a private
remote directory, with an empty credential environment and no login files. The
Hermes process itself cannot start commands or reach any other network endpoint.
No cloud account, sandbox or deployment is accessed.
"""

from __future__ import annotations

import base64
import email.parser
import email.policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from urllib.parse import parse_qs, urlsplit

from standalone.tests import test_entrypoints as entry_tests


class _CloudHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def emit(self, value, *, content_type="application/json", status=200):
        content = json.dumps(value).encode() if not isinstance(value, bytes) else value
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def remote_path(self, value):
        target = Path(value).resolve()
        if target.is_relative_to(self.server.remote):
            return target
        if value.startswith("/tmp/"):
            value = str(self.server.remote / "tmp") + value[len("/tmp"):]
        target = Path(value).resolve()
        if not target.is_relative_to(self.server.remote):
            raise AssertionError("Fixture refuses remote filesystem access outside its private root")
        return target

    def metadata(self):
        if self.server.family == "vercel":
            return {"id": "sb-fixture", "status": self.server.state, "cwd": str(self.server.remote),
                    "memory": 2048, "vcpus": 1, "region": "fixture", "runtime": "node22", "timeout": 300000,
                    "requestedAt": 1, "createdAt": 1, "updatedAt": 1}
        return {"id": "sb-fixture", "state": self.server.state, "toolboxProxyUrl": self.server.origin + "/toolbox"}

    def execute_shell(self, command, args=None, *, cwd=None):
        if command in ("bash", "sh") and args and args[0] in ("-c", "-lc", "-l"):
            script = args[-1]
        else:
            script = command
        script = script.replace("/tmp/.hermes_sync.", str(self.server.remote / "tmp/.hermes_sync."))
        # This server represents a clean remote container. Login setup is empty,
        # so never read the host account's real login files while emulating it.
        environment = {"HOME": str(self.server.remote), "PATH": os.defpath, "USER": "fixture",
                       "TMPDIR": str(self.server.remote / "tmp"), "LANG": "C.UTF-8"}
        result = subprocess.run(["/bin/bash", "--noprofile", "--norc", "-c", script],
                                cwd=self.remote_path(cwd or str(self.server.remote)), env=environment,
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=15)
        return result.stdout.decode(errors="replace"), result.stderr.decode(errors="replace"), result.returncode

    def do_GET(self):
        try:
            self.dispatch("GET")
        except Exception as error:
            self.server.errors.append(repr(error))
            self.emit({"message": str(error)}, status=500)

    def do_POST(self):
        try:
            self.dispatch("POST")
        except Exception as error:
            self.server.errors.append(repr(error))
            self.emit({"message": str(error)}, status=500)

    def do_DELETE(self):
        try:
            self.dispatch("DELETE")
        except Exception as error:
            self.server.errors.append(repr(error))
            self.emit({"message": str(error)}, status=500)

    def dispatch(self, method):
        url = urlsplit(self.path)
        path, query = url.path, parse_qs(url.query)
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        content_type = self.headers.get("Content-Type", "")
        data = json.loads(body) if body and "application/json" in content_type else None
        assert self.headers.get("Authorization") == "Bearer fixture-cloud-token"
        self.server.requests.append({"method": method, "path": path, "query": query, "body": data})
        if self.server.family == "vercel":
            assert query.get("teamId") == ["fixture-team"]
            if path == "/v1/sandboxes" and method == "POST":
                assert data["projectId"] == "fixture-project"
                assert data["resources"] == {"vcpus": 1, "memory": 2048}
                self.server.state = "running"
                return self.emit({"sandbox": self.metadata(), "routes": []})
            if path == "/v1/sandboxes/sb-fixture" and method == "GET":
                return self.emit({"sandbox": self.metadata(), "routes": []})
            if path.endswith("/cmd") and method == "POST":
                stdout, stderr, code = self.execute_shell(data["command"], data["args"], cwd=data.get("cwd"))
                identifier = str(len(self.server.commands) + 1)
                self.server.commands[identifier] = (stdout, stderr, code)
                return self.emit({"command": {"id": identifier, "exitCode": None}})
            match = re.fullmatch(r"/v1/sandboxes/sb-fixture/cmd/([0-9]+)(/logs)?", path)
            if match:
                stdout, stderr, code = self.server.commands[match[1]]
                if match[2]:
                    content = b"".join(json.dumps({"stream": name, "data": text}).encode() + b"\n"
                                       for name, text in (("stdout", stdout), ("stderr", stderr)) if text)
                    return self.emit(content, content_type="text/event-stream")
                assert query.get("wait") == ["true"]
                return self.emit({"command": {"id": match[1], "exitCode": code}})
            if path.endswith("/fs/write"):
                assert self.headers.get("x-cwd") == "/"
                with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
                    for member in archive.getmembers():
                        assert member.isfile()
                        target = self.remote_path("/" + member.name)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.extractfile(member).read())
                        target.chmod(member.mode)
                return self.emit({})
            if path.endswith("/fs/read"):
                return self.emit(self.remote_path(data["path"]).read_bytes(), content_type="application/octet-stream")
            if path.endswith("/snapshot"):
                self.server.state = "stopped"
                return self.emit({"sandbox": self.metadata(), "snapshot": {"id": "snap-fixture"}})
            if path.endswith("/stop"):
                self.server.state = "stopped"
                return self.emit({"sandbox": self.metadata()})
        else:
            if path == "/api/sandbox" and method == "POST":
                assert data["autoStopInterval"] == 0 and data["cpu"] == 1
                assert data["buildInfo"]["dockerfileContent"].startswith("FROM ")
                self.server.state = "started"
                return self.emit(self.metadata())
            if path == "/api/sandbox/paginated":
                return self.emit({"items": [self.metadata()], "page": 1, "totalPages": 1})
            if path == "/api/sandbox/sb-fixture":
                if method == "DELETE":
                    self.server.state = "destroyed"
                    return self.emit({})
                return self.emit(self.metadata())
            if path.endswith("/start"):
                self.server.state = "started"
                return self.emit(self.metadata())
            if path.endswith("/stop"):
                self.server.state = "stopped"
                return self.emit({})
            if path.endswith("/process/execute"):
                match = re.search(r"echo '([a-zA-Z0-9+/=]+)' \| base64 -d \| sh", data["command"])
                assert match
                original = base64.b64decode(match[1]).decode()
                arguments = shlex.split(original)
                stdout, stderr, code = self.execute_shell(arguments[0], arguments[1:]) if arguments and arguments[0] in ("bash", "sh") else self.execute_shell(original)
                return self.emit({"exitCode": code, "result": stdout + stderr})
            if path.endswith("/files/bulk-upload"):
                message = email.parser.BytesParser(policy=email.policy.default).parsebytes(
                    ("Content-Type: " + content_type + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + body)
                parts = {part.get_param("name", header="Content-Disposition"): part.get_payload(decode=True)
                         for part in message.iter_parts()}
                for name, value in parts.items():
                    if name.endswith(".path"):
                        target = self.remote_path(value.decode())
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(parts[name[:-4] + "file"])
                return self.emit({})
            if path.endswith("/files/download"):
                return self.emit(self.remote_path(query["path"][0]).read_bytes(), content_type="application/octet-stream")
            if path.endswith("/files/permissions"):
                self.remote_path(query["path"][0]).chmod(int(query["mode"][0], 8))
                return self.emit({})
            if path.endswith("/files") and method == "DELETE":
                self.remote_path(query["path"][0]).unlink(missing_ok=True)
                return self.emit({})
        raise AssertionError("Unexpected cloud fixture request: " + method + " " + path)


_WORKER = r'''
import json, os, pathlib, socket, sys
context, scratch, endpoint, family = json.loads(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
source = pathlib.Path(context["root"])/"app"
sys.path[:0] = [str(source), str(source.parent/"vendor")]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
from standalone.agent_probe_worker import ProbeGuard
class Guard(ProbeGuard):
    def __call__(self,event,args):
        if event in ("socket.connect","socket.getaddrinfo"):
            address = args[1] if event == "socket.connect" else args[:2]
            if isinstance(address,tuple) and address[0] == "127.0.0.1" and int(address[1]) == int(endpoint.rsplit(":",1)[1]):
                return
        if event == "subprocess.Popen" and args[1] == ["uname","-p"]:
            self.deny(event,"unavailable_optional_uname",recoverable=True)
        return super().__call__(event,args)
guard = Guard(source,source.parent/"vendor",scratch)
sys.addaudithook(guard)
remote = scratch/"remote"
if family == "vercel":
    from standalone import cloud_vercel as api
    api.API_ORIGIN = endpoint
    from tools.environments.vercel_sandbox import VercelSandboxEnvironment
    class Environment(VercelSandboxEnvironment):
        def get_temp_dir(self): return str(remote/"tmp")
    import tools.environments.vercel_sandbox as environment_module
    environment_module.VercelSandboxEnvironment = Environment
    environment = Environment(cwd=str(remote),memory=2048,persistent_filesystem=True,task_id="wire-fixture")
else:
    from standalone import cloud_daytona as api
    from tools.environments.daytona import DaytonaEnvironment
    class Environment(DaytonaEnvironment):
        def get_temp_dir(self): return str(remote/"tmp")
    import tools.environments.daytona as environment_module
    environment_module.DaytonaEnvironment = Environment
    environment = Environment(image="fixture/image:tag",cwd=str(remote),persistent_filesystem=False,task_id="wire-fixture")
assert environment._snapshot_ready
try:
    first = environment.execute("export WIRE_FIXTURE=persisted; mkdir -p next; cd next; printf 'remote fixture'",timeout=10)
    assert first["returncode"] == 0 and "remote fixture" in first["output"],first
    assert environment.cwd == str(remote/"next"),environment.cwd
    second = environment.execute("printf %s \"$WIRE_FIXTURE\"",timeout=10)
    assert second["returncode"] == 0 and "persisted" in second["output"],second
    payload = "private fixture stdin\nsecond line \"' $ < >\n"
    result = environment.execute("cat",stdin_data=payload,timeout=10)
    assert result["returncode"] == 0 and payload.rstrip() in result["output"],result
    if family == "vercel":
        sandbox = environment._sandbox
        path = str(remote/"binary")
        sandbox.write_files([{"path":path,"content":b"\x00\xffbinary","mode":0o600}])
        sandbox.download_file(path,scratch/"downloaded")
    else:
        sandbox = environment._sandbox
        sandbox.fs.upload_file(b"\x00\xffbinary",str(remote/"binary"))
        sandbox.fs.download_file(str(remote/"binary"),str(scratch/"downloaded"))
    assert (scratch/"downloaded").read_bytes() == b"\x00\xffbinary"
    os.environ.update(TERMINAL_ENV="vercel_sandbox" if family == "vercel" else "daytona",
                      TERMINAL_CWD=str(remote),TERMINAL_CONTAINER_MEMORY="2048",
                      TERMINAL_CONTAINER_CPU="1",TERMINAL_CONTAINER_DISK="51200",
                      TERMINAL_CONTAINER_PERSISTENT="true" if family == "vercel" else "false")
    from tools import terminal_tool as terminal
    assert terminal.check_terminal_requirements(),terminal.terminal_backend_unavailable_reason()
    try:
        public = json.loads(terminal.terminal_tool("printf 'public cloud fixture'",task_id="public-wire",timeout=10))
        assert public.get("exit_code") == 0 and not public.get("error") and "public cloud fixture" in public.get("output",""),public
    finally:
        terminal.cleanup_all_environments()
finally:
    environment.cleanup()
assert all(event.get("recoverable_capability_probe") for event in guard.events),guard.events
if family == "vercel":
    assert json.loads((pathlib.Path(os.environ["HERMES_HOME"])/"vercel_sandbox_snapshots.json").read_text())["wire-fixture"] == "snap-fixture"
assert "daytona" not in sys.modules and "vercel.sandbox" not in sys.modules
print(json.dumps({"ok":True,"family":family,"no_global_sdk":True}),flush=True)
os._exit(0)
'''


class CloudEnvironmentTests(entry_tests.BundledSourceTestCase):
    def test_original_vercel_environment_exec_stdin_sync_and_snapshot_over_http(self):
        self.exercise("vercel")

    def test_original_daytona_environment_exec_stdin_files_and_delete_over_http(self):
        self.exercise("daytona")

    def exercise(self, family):
        with tempfile.TemporaryDirectory(prefix="cloud-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp", "remote", "remote/tmp", "remote/.hermes"):
                (scratch / name).mkdir()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _CloudHandler)
            server.remote, server.family = scratch / "remote", family
            server.state = "running" if family == "vercel" else "started"
            server.requests, server.commands, server.errors = [], {}, []
            server.origin = f"http://127.0.0.1:{server.server_port}"
            server.daemon_threads = True
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            environment = entry_tests.support._load_helper("agent_probe_worker").fixture_environment(scratch)
            environment.update(PATH="", VERCEL_TOKEN="fixture-cloud-token", VERCEL_TEAM_ID="fixture-team",
                               VERCEL_PROJECT_ID="fixture-project", DAYTONA_API_KEY="fixture-cloud-token",
                               DAYTONA_API_URL=server.origin + "/api")
            Path(environment["HERMES_HOME"]).mkdir()
            try:
                result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", _WORKER,
                                         json.dumps(self.context), str(scratch), server.origin, family],
                                        cwd=scratch / "work", env=environment, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout + repr(server.errors))
                self.assertTrue(json.loads(result.stdout)["ok"])
                self.assertFalse(server.errors)
                self.assertTrue(any(request["path"].endswith("/stop" if family == "vercel" else "/sb-fixture")
                                    and request["method"] == ("POST" if family == "vercel" else "DELETE") for request in server.requests))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
