"""Explicit Vercel Sandbox HTTP boundary used by the original environment.

Wire operations were reviewed against vercel 0.7.2 (SHA256
fceab7b354a19d088e2dd6463d3c9f1358ec71bdad2d9452e10ffa9092101066).
The original vercel-oidc credentials implementation remains embedded, including
refresh and request-local credential discovery. No global vercel SDK is faked.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
from datetime import timedelta
from enum import StrEnum
import io
import json
import os
from pathlib import Path
import posixpath
import tarfile
import threading
import time
from types import SimpleNamespace
from typing import NotRequired, TypedDict
from urllib.parse import quote

import httpx


API_ORIGIN = "https://api.vercel.com"


class SandboxStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    ABORTED = "aborted"
    SNAPSHOTTING = "snapshotting"


class WriteFile(TypedDict):
    path: str
    content: bytes
    mode: NotRequired[int]


@dataclass(frozen=True)
class Resources:
    vcpus: int | None = None
    memory: int | None = None

    def __post_init__(self):
        if any(value is not None and type(value) is not int for value in (self.vcpus, self.memory)):
            raise TypeError("Vercel sandbox resources must be integers")
        if self.vcpus is not None and self.vcpus != 1 and self.vcpus % 2:
            raise ValueError("Vercel sandbox vcpus must be 1 or even")
        if self.vcpus is not None and self.memory is not None and self.memory != self.vcpus * 2048:
            raise ValueError("Vercel sandbox memory must equal vcpus * 2048")


class SandboxError(RuntimeError):
    def __init__(self, response):
        self.response, self.status_code = response, response.status_code
        try:
            error = response.json().get("error", {})
            detail = error.get("message") if isinstance(error, dict) else str(error)
        except ValueError:
            detail = None
        super().__init__(f"Vercel sandbox HTTP {self.status_code}" + (": " + detail if detail else ""))


def _seconds(value):
    return value.total_seconds() if isinstance(value, timedelta) else float(value)


def _segment(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Vercel sandbox identifiers must be nonempty strings")
    return quote(value, safe="")


class _Client:
    def __init__(self):
        self.http = httpx.Client(base_url=API_ORIGIN, timeout=180,
                                 headers={"User-Agent": "hermes-standalone/vercel-sandbox"})

    def credentials(self):
        from vercel.oidc import get_credentials
        return get_credentials()

    def request(self, method, path, *, stream=False, **kwargs):
        credentials = self.credentials()
        headers = {"Authorization": "Bearer " + credentials.token, **kwargs.pop("headers", {})}
        params = dict(kwargs.pop("params", {}))
        if credentials.team_id:
            params.setdefault("teamId", credentials.team_id)
        request = self.http.build_request(method, path, headers=headers, params=params, **kwargs)
        response = self.http.send(request, stream=stream)
        if not response.is_success:
            response.read()
            try:
                raise SandboxError(response)
            finally:
                response.close()
        return response

    def json(self, method, path, **kwargs):
        with closing(self.request(method, path, **kwargs)) as response:
            value = response.json()
        if not isinstance(value, dict):
            raise ValueError("Vercel sandbox response must be an object")
        return value

    def close(self):
        self.http.close()


class _Command:
    def __init__(self, sandbox, payload):
        self._sandbox = sandbox
        self.cmd_id = payload["id"]
        _segment(self.cmd_id)
        self.exit_code = payload.get("exitCode")

    def wait(self):
        response = self._sandbox.client.json("GET", self._sandbox._path + "/cmd/" + _segment(self.cmd_id), params={"wait": "true"})
        code = response["command"].get("exitCode")
        if type(code) is not int:
            raise ValueError("Vercel sandbox command did not provide an exit code")
        self.exit_code = code
        return self

    def output(self, stream="both"):
        output = []
        response = self._sandbox.client.request("GET", self._sandbox._path + "/cmd/" + _segment(self.cmd_id) + "/logs",
                                                 headers={"Accept": "text/event-stream"}, stream=True)
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict) and item.get("stream") in ("stdout", "stderr") and isinstance(item.get("data"), str):
                    if stream == "both" or item["stream"] == stream:
                        output.append(item["data"])
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ProtocolError, httpx.TransportError):
            pass  # The original SDK retains complete records before transport EOF.
        finally:
            response.close()
        return "".join(output)


class Sandbox:
    def __init__(self, client, response):
        self.client = client
        self._update(response)

    def _update(self, response):
        record = response["sandbox"]
        if not isinstance(record, dict) or not isinstance(record.get("cwd"), str) or not record["cwd"].startswith("/"):
            raise ValueError("Invalid Vercel sandbox metadata")
        _segment(record.get("id"))
        self.sandbox = SimpleNamespace(**{**record, "status": SandboxStatus(record["status"])})
        self._path = "/v1/sandboxes/" + _segment(self.sandbox.id)

    @property
    def status(self):
        return self.sandbox.status

    @classmethod
    def create(cls, *, timeout=None, resources=None, runtime=None, image=None, source=None):
        if runtime is not None and image is not None:
            raise ValueError("Vercel sandbox runtime and image are mutually exclusive")
        if source is not None and (not isinstance(source, dict) or source.get("type") != "snapshot" or not isinstance(source.get("snapshot_id"), str)):
            raise ValueError("Vercel environment requires a snapshot source with snapshot_id")
        if source is not None and runtime is not None:
            raise ValueError("Vercel snapshot sources cannot be combined with runtime")
        if resources is not None and not isinstance(resources, Resources):
            raise TypeError("Vercel sandbox resources must be Resources")
        client = _Client()
        try:
            credentials = client.credentials()
            if not credentials.project_id:
                raise RuntimeError("Sandbox credentials did not include a project ID")
            body = {"projectId": credentials.project_id}
            if timeout is not None:
                body["timeout"] = int(timeout.total_seconds() * 1000) if isinstance(timeout, timedelta) else timeout
            if runtime is not None:
                body["runtime"] = runtime
            if image is not None:
                body["image"] = image
            if resources is not None:
                body["resources"] = {name: value for name, value in vars(resources).items() if value is not None}
            if source is not None:
                body["source"] = {"type": "snapshot", "snapshotId": source["snapshot_id"]}
            return cls(client, client.json("POST", "/v1/sandboxes", json=body))
        except BaseException:
            client.close()
            raise

    def refresh(self):
        self._update(self.client.json("GET", self._path))
        return self

    def wait_for_status(self, status, *, timeout=timedelta(seconds=30), poll_interval=timedelta(milliseconds=250)):
        deadline = time.monotonic() + _seconds(timeout)
        interval = threading.Event()
        while self.status != status:
            if time.monotonic() >= deadline:
                raise TimeoutError("Vercel sandbox did not reach " + str(status))
            interval.wait(min(_seconds(poll_interval), max(0, deadline - time.monotonic())))
            self.refresh()

    def run_command(self, command, args=None, *, cwd=None, env=None, sudo=False):
        if not isinstance(command, str) or not isinstance(args or [], list):
            raise TypeError("Vercel command and argument list have invalid types")
        body = {"command": command, "args": args or [], "env": env or {}, "sudo": sudo}
        if cwd is not None:
            body["cwd"] = cwd
        payload = self.client.json("POST", self._path + "/cmd", json=body)
        return _Command(self, payload["command"]).wait()

    def write_files(self, files):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for entry in files:
                path, content = entry["path"], entry["content"]
                if not isinstance(path, str) or not isinstance(content, bytes):
                    raise TypeError("Vercel write_files requires string paths and byte contents")
                mode = entry.get("mode")
                if mode is not None and (type(mode) is not int or not 0 <= mode <= 0o777):
                    raise ValueError("Vercel file mode must be an integer between 0 and 0o777")
                absolute = posixpath.normpath(path if posixpath.isabs(path) else posixpath.join(self.sandbox.cwd, path))
                info = tarfile.TarInfo(posixpath.relpath(absolute, "/"))
                info.size = len(content)
                if mode is not None:
                    info.mode = mode
                archive.addfile(info, io.BytesIO(content))
        with closing(self.client.request("POST", self._path + "/fs/write", content=buffer.getvalue(),
                                         headers={"x-cwd": "/", "Content-Type": "application/gzip"})):
            pass

    def download_file(self, path, destination):
        destination = Path(destination)
        partial = destination.with_name(destination.name + ".part")
        response = self.client.request("POST", self._path + "/fs/read", json={"path": path}, stream=True)
        try:
            with partial.open("wb") as stream:
                for block in response.iter_bytes():
                    stream.write(block)
            os.replace(partial, destination)
        finally:
            response.close()
            partial.unlink(missing_ok=True)

    def snapshot(self):
        response = self.client.json("POST", self._path + "/snapshot")
        self._update(response)
        snapshot_id = response["snapshot"]["id"]
        _segment(snapshot_id)
        return SimpleNamespace(snapshot_id=snapshot_id)

    def stop(self, *, blocking=False, timeout=timedelta(seconds=30), poll_interval=timedelta(milliseconds=250)):
        self._update(self.client.json("POST", self._path + "/stop"))
        if blocking:
            self.wait_for_status(SandboxStatus.STOPPED, timeout=timeout, poll_interval=poll_interval)
