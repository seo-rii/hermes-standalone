"""Explicit Daytona REST/toolbox operations for the original Hermes environment.

Reviewed wire sources: Daytona 0.155.0, daytona-api-client 0.155.0 and
daytona-toolbox-api-client 0.155.0 from uv.lock hash-verified archives. This
qualified boundary replaces their native object-store/Pydantic closure for the
operations Hermes actually calls. Remote shell, sync and cancellation policy
remain in tools.environments.daytona.
"""

from __future__ import annotations

import base64
from contextlib import ExitStack, closing
from dataclasses import dataclass
from enum import StrEnum
import io
import json
import os
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace
from urllib.parse import quote, urlsplit

import httpx


class DaytonaError(RuntimeError):
    def __init__(self, message, *, response=None):
        self.response = response
        self.status_code = response.status_code if response is not None else None
        super().__init__(message)


class SandboxState(StrEnum):
    CREATING = "creating"
    RESTORING = "restoring"
    DESTROYED = "destroyed"
    DESTROYING = "destroying"
    STARTED = "started"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"
    BUILD_FAILED = "build_failed"
    PENDING_BUILD = "pending_build"
    BUILDING_SNAPSHOT = "building_snapshot"
    UNKNOWN = "unknown"
    PULLING_SNAPSHOT = "pulling_snapshot"
    ARCHIVED = "archived"
    ARCHIVING = "archiving"
    RESIZING = "resizing"


@dataclass(frozen=True)
class Resources:
    cpu: int | None = None
    memory: int | None = None
    disk: int | None = None
    gpu: int | None = None


@dataclass(frozen=True)
class CreateSandboxFromImageParams:
    image: str
    name: str | None = None
    labels: dict | None = None
    auto_stop_interval: int | None = None
    resources: Resources | None = None


@dataclass(frozen=True)
class FileUpload:
    source: str | bytes | bytearray
    destination: str


def _segment(value):
    if not isinstance(value, str) or not value:
        raise DaytonaError("Daytona sandbox identifiers must be nonempty strings")
    return quote(value, safe="")


def _origin(value):
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise DaytonaError("Invalid Daytona API origin")
    return value.rstrip("/")


def _secret(name):
    from agent.secret_scope import get_secret
    return get_secret(name)


class Daytona:
    def __init__(self):
        key, jwt = _secret("DAYTONA_API_KEY"), _secret("DAYTONA_JWT_TOKEN")
        if not key and not jwt:
            raise DaytonaError("DAYTONA_API_KEY or DAYTONA_JWT_TOKEN is required")
        organization = _secret("DAYTONA_ORGANIZATION_ID")
        if not key and not organization:
            raise DaytonaError("Organization ID is required when using JWT token")
        self.origin = _origin(_secret("DAYTONA_API_URL") or _secret("DAYTONA_SERVER_URL") or "https://app.daytona.io/api")
        self.target = _secret("DAYTONA_TARGET")
        headers = {"Authorization": "Bearer " + (key or jwt), "X-Daytona-Source": "python-sdk",
                   "X-Daytona-SDK-Version": "0.155.0"}
        if not key:
            headers["X-Daytona-Organization-ID"] = organization
        self.http = httpx.Client(headers=headers, timeout=60)

    def _request(self, method, path, *, origin=None, stream=False, **kwargs):
        try:
            request = self.http.build_request(method, (origin or self.origin) + path, **kwargs)
            response = self.http.send(request, stream=stream)
        except httpx.HTTPError as exc:
            raise DaytonaError("Daytona transport failed: " + str(exc)) from exc
        if not response.is_success:
            response.read()
            try:
                try:
                    body = response.json()
                    message = body.get("message") if isinstance(body, dict) else None
                except ValueError:
                    message = None
                raise DaytonaError(f"Daytona HTTP {response.status_code}" + (": " + str(message) if message else ""), response=response)
            finally:
                response.close()
        return response

    def _json(self, method, path, **kwargs):
        with closing(self._request(method, path, **kwargs)) as response:
            if not response.content:
                return None
            result = response.json()
        if not isinstance(result, dict):
            raise DaytonaError("Daytona response must be an object")
        return result

    def get(self, name):
        return Sandbox(self, self._json("GET", "/sandbox/" + _segment(name)))

    def list(self, *, labels=None, limit=None):
        page = 1
        while True:
            params = {"page": page}
            if labels is not None:
                params["labels"] = json.dumps(labels)
            if limit is not None:
                params["limit"] = limit
            response = self._json("GET", "/sandbox/paginated", params=params)
            items = response["items"]
            if not isinstance(items, list):
                raise DaytonaError("Invalid Daytona sandbox listing")
            for item in items:
                yield Sandbox(self, item)
            total_pages = response.get("totalPages", response.get("total_pages", page))
            if type(total_pages) is not int or page >= total_pages or not items:
                return
            page += 1

    def create(self, params, *, timeout=60):
        if not isinstance(params, CreateSandboxFromImageParams) or not isinstance(params.image, str) or not params.image:
            raise DaytonaError("Daytona environment requires an image reference")
        body = {"env": {}, "volumes": [], "buildInfo": {"dockerfileContent": "FROM " + params.image + "\n"}}
        for name, value in (("name", params.name), ("labels", params.labels), ("target", self.target),
                            ("autoStopInterval", params.auto_stop_interval)):
            if value is not None:
                body[name] = value
        if params.resources is not None:
            body.update({name: value for name, value in vars(params.resources).items() if value is not None})
        sandbox = Sandbox(self, self._json("POST", "/sandbox", json=body, timeout=timeout or None))
        sandbox._wait(SandboxState.STARTED, timeout)
        return sandbox

    def delete(self, sandbox, *, timeout=60):
        with closing(self._request("DELETE", "/sandbox/" + _segment(sandbox.id), timeout=timeout or None)):
            pass
        sandbox.state = SandboxState.DESTROYED

    def close(self):
        self.http.close()


class _Process:
    def __init__(self, sandbox):
        self.sandbox = sandbox

    def exec(self, command, cwd=None, env=None, timeout=None):
        if not isinstance(command, str):
            raise TypeError("Daytona shell commands must be strings")
        encoded = base64.b64encode(command.encode()).decode("ascii")
        wrapped = "echo '" + encoded + "' | base64 -d | sh"
        if env:
            exports = []
            for name, value in env.items():
                if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name):
                    raise ValueError("Invalid Daytona environment variable name")
                exports.append("export " + name + "=$(echo '" + base64.b64encode(value.encode()).decode("ascii") + "' | base64 -d)")
            wrapped = ";".join(exports) + "; " + wrapped
        body = {"command": 'sh -c "' + wrapped + '"'}
        if cwd is not None:
            body["cwd"] = cwd
        if timeout is not None:
            body["timeout"] = timeout
        response = self.sandbox._json("POST", "/process/execute", json=body,
                                      timeout=None if timeout == 0 else (timeout + 30 if timeout is not None else 60))
        output, charts = [], []
        for line in response["result"].splitlines():
            if line.startswith("dtn_artifact_k39fd2:"):
                artifact = json.loads(line.removeprefix("dtn_artifact_k39fd2:").strip())
                if artifact.get("type") == "chart":
                    charts.append(artifact.get("value", {}))
            else:
                output.append(line)
        result = "\n".join(output)
        code = response.get("exitCode", response.get("code"))
        if type(code) is not int:
            raise DaytonaError("Daytona command did not provide an exit code")
        return SimpleNamespace(result=result, exit_code=code, artifacts=SimpleNamespace(stdout=result, charts=charts))


class _Files:
    def __init__(self, sandbox):
        self.sandbox = sandbox

    def upload_file(self, source, destination, *, timeout=1800):
        self.upload_files([FileUpload(source, destination)], timeout=timeout)

    def upload_files(self, files, timeout=1800):
        fields, payloads = {}, {}
        with ExitStack() as stack:
            for index, entry in enumerate(files):
                fields[f"files[{index}].path"] = entry.destination
                stream = io.BytesIO(entry.source) if isinstance(entry.source, (bytes, bytearray)) else stack.enter_context(open(entry.source, "rb"))
                payloads[f"files[{index}].file"] = (entry.destination, stream)
            with closing(self.sandbox._request("POST", "/files/bulk-upload", data=fields, files=payloads, timeout=timeout or None)):
                pass

    def download_file(self, source, destination=None, *, timeout=1800):
        response = self.sandbox._request("GET", "/files/download", params={"path": source}, timeout=timeout or None, stream=True)
        partial = None
        try:
            if destination is None:
                return response.read()
            target = Path(destination)
            partial = target.with_name(target.name + ".part")
            with partial.open("wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
            os.replace(partial, target)
        finally:
            response.close()
            if partial is not None:
                partial.unlink(missing_ok=True)

    def delete_file(self, path, recursive=False):
        with closing(self.sandbox._request("DELETE", "/files", params={"path": path, "recursive": str(bool(recursive)).lower()})):
            pass

    def set_file_permissions(self, path, *, mode=None, owner=None, group=None):
        params = {name: value for name, value in (("path", path), ("mode", mode), ("owner", owner), ("group", group)) if value is not None}
        with closing(self.sandbox._request("POST", "/files/permissions", params=params)):
            pass


class Sandbox:
    def __init__(self, daytona, response):
        self.daytona = daytona
        self._update(response)
        self.process, self.fs = _Process(self), _Files(self)

    def _update(self, response):
        if not isinstance(response, dict):
            raise DaytonaError("Invalid Daytona sandbox metadata")
        self.id = response["id"]
        _segment(self.id)
        self.state = SandboxState(response["state"])
        self.error_reason = response.get("errorReason")
        self.toolbox_proxy_url = _origin(response["toolboxProxyUrl"])
        self._path = "/" + _segment(self.id)

    def _request(self, method, path, **kwargs):
        return self.daytona._request(method, self._path + path, origin=self.toolbox_proxy_url, **kwargs)

    def _json(self, method, path, **kwargs):
        return self.daytona._json(method, self._path + path, origin=self.toolbox_proxy_url, **kwargs)

    def refresh_data(self):
        self._update(self.daytona._json("GET", "/sandbox/" + _segment(self.id)))

    def _wait(self, state, timeout):
        deadline = time.monotonic() + timeout if timeout else None
        interval = threading.Event()
        while self.state != state:
            if self.state in (SandboxState.ERROR, SandboxState.BUILD_FAILED):
                raise DaytonaError("Daytona sandbox failed: " + str(self.error_reason or self.state))
            if state == SandboxState.STOPPED and self.state == SandboxState.DESTROYED:
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Daytona sandbox did not reach " + str(state))
            interval.wait(0.1)
            self.refresh_data()

    def start(self, timeout=60):
        self._update(self.daytona._json("POST", "/sandbox/" + _segment(self.id) + "/start", timeout=timeout or None))
        self._wait(SandboxState.STARTED, timeout)

    def stop(self, timeout=60):
        with closing(self.daytona._request("POST", "/sandbox/" + _segment(self.id) + "/stop", timeout=timeout or None)):
            pass
        self.refresh_data()
        self._wait(SandboxState.STOPPED, timeout)
