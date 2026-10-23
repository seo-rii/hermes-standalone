"""Typed dashboard requests, auth dependencies and streaming scope contracts."""

from __future__ import annotations

from contextvars import ContextVar
import importlib.util
import json
from typing import Literal
import unittest

from standalone.tests import test_transport as _fixture
from standalone.web_schema import Field, RequestModel, SecretStr, StrictBool, StrictInt, ValidationError, field_validator


class Settings(RequestModel):
    revision: StrictInt = Field(0, ge=0)
    enabled: StrictBool = False
    endpoint: Literal["chat", "messages"] = "chat"
    note: str | None = None
    secret: SecretStr | None = None
    rows: list[dict[str, int]] = []


class Nested(RequestModel):
    settings: Settings
    mapping: dict[str, Settings] = {}


class Validated(RequestModel):
    timeout: float | None = None

    @field_validator("timeout", mode="before")
    @classmethod
    def positive(cls, value):
        if value is not None and (isinstance(value, bool) or float(value) <= 0):
            raise ValueError("timeout must be positive")
        return value


class SchemaTests(unittest.TestCase):
    def test_strict_boolean_revision_and_enum_reject_ambiguous_payloads(self):
        for values in ({"revision": True}, {"revision": "1"}, {"revision": -1},
                       {"enabled": 1}, {"enabled": "false"}, {"endpoint": "unknown"},
                       {"note": 3}, {"rows": [{"n": "bad"}]}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Settings(**values)

    def test_nested_models_secret_masking_and_omitted_null_are_preserved(self):
        value = Nested.model_validate({"settings": {"revision": 2, "note": None, "secret": "private-fixture"},
                                       "mapping": {"second": {"enabled": True}}})
        self.assertEqual(value.settings.model_fields_set, {"revision", "note", "secret"})
        self.assertEqual(value.settings.secret.get_secret_value(), "private-fixture")
        self.assertNotIn("private-fixture", value.model_dump_json())
        self.assertEqual(value.mapping["second"].enabled, True)
        self.assertEqual(value.settings.model_dump(exclude_unset=True)["note"], None)
        self.assertNotIn("note", Settings().model_dump(exclude_unset=True))
        self.assertNotIn("note", value.settings.model_dump(exclude_none=True))

    def test_mutable_defaults_and_before_validator(self):
        first, second = Settings(), Settings()
        first.rows.append({"count": 3})
        self.assertEqual(second.rows, [])
        self.assertEqual(Validated(timeout="2.5").timeout, 2.5)
        for value in (True, 0, -1, "bad"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Validated(timeout=value)


class DashboardDependencyTests(_fixture.TransportDependencyTests):
    DEPENDENCIES = ("httpx", "starlette", "python-multipart")
    CASES = "standalone.tests.test_dashboard.ASGITests"


@unittest.skipUnless(importlib.util.find_spec("starlette"), "Runs in the isolated dependency subprocess")
class ASGITests(unittest.TestCase):
    def test_real_request_models_query_bounds_and_multipart(self):
        from standalone.web import FastAPI, File, Form, Query, UploadFile
        from starlette.testclient import TestClient
        # The real endpoints use module-level annotation bindings, too.
        globals()["UploadFile"] = UploadFile
        app = FastAPI()

        @app.post("/settings/{item}")
        async def settings(item: int, body: Nested, limit: int = Query(20, ge=0, le=50)):
            return {"item": item, "body": body, "limit": limit, "fields": sorted(body.settings.model_fields_set)}

        @app.post("/file")
        async def upload(file: UploadFile = File(...), enabled: bool = Form(False)):
            return {"filename": file.filename, "content": (await file.read()).decode(), "enabled": enabled}

        with TestClient(app) as client:
            result = client.post("/settings/3?limit=8", json={"settings": {"enabled": True, "secret": "fixture"}})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()["fields"], ["enabled", "secret"])
            self.assertEqual(result.json()["body"]["settings"]["secret"], "**********")
            for url, body in (("/settings/a", {"settings": {}}), ("/settings/3?limit=-1", {"settings": {}}),
                              ("/settings/3", {"settings": {"enabled": "true"}})):
                response = client.post(url, json=body)
                self.assertEqual(response.status_code, 422, response.text)
            response = client.post("/settings/3", content=b"not-json", headers={"content-type": "application/json"})
            self.assertEqual(response.status_code, 422)
            response = client.post("/file", files={"file": ("fixture.txt", b"safe contents")}, data={"enabled": "true"})
            self.assertEqual(response.json(), {"filename": "fixture.txt", "content": "safe contents", "enabled": True})

    def test_auth_and_profile_dependency_live_through_stream_and_close(self):
        from standalone.web import APIRouter, Depends, FastAPI, HTTPConnection, HTTPException, StreamingResponse
        from starlette.testclient import TestClient
        globals()["HTTPConnection"] = HTTPConnection
        scope = ContextVar("dashboard-test-profile", default="unbound")
        transitions = []

        async def profile(conn: HTTPConnection, profile: str = "default"):
            if conn.headers.get("authorization") != "Bearer fixture":
                raise HTTPException(401, "Missing auth")
            token = scope.set(profile)
            transitions.append(("enter", scope.get()))
            try:
                yield
            finally:
                transitions.append(("exit", scope.get()))
                scope.reset(token)

        router = APIRouter(prefix="/api")

        @router.get("/stream")
        async def stream():
            async def rows():
                yield scope.get()
            return StreamingResponse(rows())

        @router.get("/sync")
        def sync_handler():
            return {"profile": scope.get()}

        app = FastAPI()
        app.include_router(router, dependencies=[Depends(profile)])
        with TestClient(app) as client:
            response = client.get("/api/stream")
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json(), {"detail": "Missing auth"})
            response = client.get("/api/stream?profile=satellite", headers={"authorization": "Bearer fixture"})
            self.assertEqual(response.text, "satellite")
            response = client.get("/api/sync?profile=second", headers={"authorization": "Bearer fixture"})
            self.assertEqual(response.json(), {"profile": "second"})
        self.assertEqual(transitions, [("enter", "satellite"), ("exit", "satellite"), ("enter", "second"), ("exit", "second")])
        self.assertEqual(scope.get(), "unbound")

    def test_websocket_dependency_and_parameters(self):
        from standalone.web import Depends, FastAPI, HTTPConnection, WebSocket, WebSocketException
        from starlette.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect
        globals().update(HTTPConnection=HTTPConnection, WebSocket=WebSocket)
        app = FastAPI()

        async def auth(conn: HTTPConnection, token: str):
            if token != "fixture":
                raise WebSocketException(1008)

        @app.websocket("/events/{number}", dependencies=[Depends(auth)])
        async def events(socket: WebSocket, number: int):
            await socket.accept()
            await socket.send_json({"number": number})
            await socket.close()

        with TestClient(app) as client:
            with client.websocket_connect("/events/3?token=fixture") as socket:
                self.assertEqual(socket.receive_json(), {"number": 3})
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect("/events/3?token=wrong"):
                    pass
