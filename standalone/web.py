"""Hermes dashboard routing on the embedded Starlette ASGI implementation.

The original middleware, handlers and profile dependencies remain the owners of
authorization and side effects. This module replaces only typed parameter
binding and route registration; it is not installed under the FastAPI name.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import inspect
import json
from pathlib import Path
import sys
from typing import get_args, get_origin, get_type_hints

from starlette.applications import Starlette
from starlette.background import BackgroundTasks
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException, WebSocketException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import HTTPConnection, Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route, Router, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette import status

from standalone.web_schema import (
    MISSING, Field, FieldInfo, RequestModel, SecretStr, StrictBool, StrictInt,
    ValidationError, constrain, dump_value, field_validator, validate,
)

BaseModel = RequestModel


@dataclass
class Parameter(FieldInfo):
    location: str = "query"


def Query(default=MISSING, **kwargs):
    return Parameter(default=default, location="query", **kwargs)


def Body(default=MISSING, **kwargs):
    return Parameter(default=default, location="body", **kwargs)


def Form(default=MISSING, **kwargs):
    return Parameter(default=default, location="form", **kwargs)


def File(default=MISSING, **kwargs):
    return Parameter(default=default, location="form", **kwargs)


@dataclass(frozen=True)
class Depends:
    dependency: object
    use_cache: bool = True


def jsonable_encoder(value, *, include=None, exclude=None, by_alias=True, exclude_unset=False,
                     exclude_defaults=False, exclude_none=False, custom_encoder=None, **_kwargs):
    if custom_encoder and type(value) in custom_encoder:
        return custom_encoder[type(value)](value)
    if isinstance(value, RequestModel):
        value = value.model_dump(mode="json", include=include, exclude=exclude, by_alias=by_alias,
                                 exclude_unset=exclude_unset, exclude_defaults=exclude_defaults,
                                 exclude_none=exclude_none)
    if isinstance(value, dict):
        return {str(key): jsonable_encoder(item, exclude_none=exclude_none, custom_encoder=custom_encoder)
                for key, item in value.items() if not (exclude_none and item is None)}
    if isinstance(value, (list, tuple, set)):
        return [jsonable_encoder(item, exclude_none=exclude_none, custom_encoder=custom_encoder) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return jsonable_encoder(value.value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return dump_value(value, mode="json", exclude_none=exclude_none)


def _body_annotation(annotation):
    if annotation in (dict, list) or get_origin(annotation) in (dict, list):
        return True
    if isinstance(annotation, type) and issubclass(annotation, RequestModel):
        return True
    return any(_body_annotation(part) for part in get_args(annotation))


class _Binding:
    def __init__(self, connection, stack):
        self.connection, self.stack = connection, stack
        self.cache = {}
        self.body = MISSING
        self.form = None

    async def _parameter(self, name, annotation, default):
        if annotation in (Request, WebSocket, HTTPConnection):
            return self.connection
        if isinstance(default, Depends):
            return await self.invoke(default.dependency, dependency=default)
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Dashboard parameter {name!r} requires an explicit annotation")
        field = default if isinstance(default, Parameter) else Parameter(default=default)
        if field.default is inspect.Parameter.empty:
            field = Parameter(location=field.location)
        key = field.alias or name
        location = "path" if name in self.connection.path_params else field.location
        if not isinstance(default, Parameter) and _body_annotation(annotation):
            location = "body"
        value = MISSING
        if location == "path":
            value = self.connection.path_params[name]
        elif location == "body":
            if not isinstance(self.connection, Request):
                raise TypeError("A WebSocket endpoint cannot declare a JSON request body")
            if self.body is MISSING:
                raw = await self.connection.body()
                try:
                    self.body = json.loads(raw) if raw else MISSING
                except (ValueError, UnicodeError):
                    raise ValidationError([{"loc": ("body",), "type": "json_invalid", "msg": "Invalid JSON"}]) from None
            value = self.body
        elif location == "form":
            if self.form is None:
                self.form = await self.connection.form()
                self.stack.push_async_callback(self.form.close)
            value = self.form.get(key, MISSING)
        else:
            query = self.connection.query_params
            if key in query:
                value = query.getlist(key) if get_origin(annotation) is list else query[key]
        if value is MISSING:
            if field.default is MISSING or field.default is Ellipsis:
                raise ValidationError([{"loc": (location, name), "type": "missing", "msg": "Field required"}])
            return field.default
        if annotation is UploadFile:
            if not isinstance(value, UploadFile):
                raise ValidationError([{"loc": (location, name), "type": "file_type", "msg": "Upload file required"}])
            return value
        position = ("body",) if location == "body" else (location, name)
        return constrain(validate(annotation, value, position), field, position)

    async def invoke(self, function, *, dependency=None):
        if dependency and dependency.use_cache and function in self.cache:
            return self.cache[function]
        hints = get_type_hints(function, include_extras=True)
        values = {}
        for name, parameter in inspect.signature(function).parameters.items():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise TypeError("Dashboard endpoints must declare concrete parameters")
            values[name] = await self._parameter(name, hints.get(name, parameter.annotation), parameter.default)
        if inspect.isasyncgenfunction(function):
            result = await self.stack.enter_async_context(asynccontextmanager(function)(**values))
        elif inspect.isgeneratorfunction(function):
            # Hermes profile dependencies are async generators so their context
            # tokens stay in this request's task. Synchronous contexts run here
            # rather than entering and exiting in unrelated worker contexts.
            result = self.stack.enter_context(contextmanager(function)(**values))
        elif inspect.iscoroutinefunction(function):
            result = await function(**values)
        else:
            result = await run_in_threadpool(function, **values)
        if dependency and dependency.use_cache:
            self.cache[function] = result
        return result


@dataclass
class _Spec:
    function: object
    dependencies: tuple
    options: dict
    websocket: bool = False


def _endpoint(spec):
    async def handler(connection):
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            binding = _Binding(connection, stack)
            try:
                for dependency in spec.dependencies:
                    await binding.invoke(dependency.dependency, dependency=dependency)
                result = await binding.invoke(spec.function)
            except ValidationError as exc:
                if spec.websocket:
                    raise WebSocketException(code=1008, reason="Invalid request parameters") from exc
                return _ScopedResponse(JSONResponse({"detail": exc.errors()}, status_code=422), stack)
            if spec.websocket:
                await stack.aclose()
                return result
            if isinstance(result, Response):
                return _ScopedResponse(result, stack)
            response_model = spec.options.get("response_model")
            if response_model is not None:
                result = validate(response_model, result)
            status_code = spec.options.get("status_code") or 200
            if status_code in (204, 304):
                return _ScopedResponse(Response(status_code=status_code), stack)
            response_class = spec.options.get("response_class", JSONResponse)
            return _ScopedResponse(response_class(jsonable_encoder(result), status_code=status_code), stack)
        except BaseException:
            await stack.__aexit__(*sys.exc_info())
            raise

    return handler


class _ScopedResponse(Response):
    """Keep profile yield dependencies alive through streaming and disconnects."""

    def __init__(self, response, stack):
        self._response, self._stack = response, stack

    def __getattr__(self, name):
        return getattr(self._response, name)

    async def __call__(self, scope, receive, send):
        try:
            await self._response(scope, receive, send)
        finally:
            await self._stack.__aexit__(*sys.exc_info())


class _Registration:
    def add_api_route(self, path, endpoint, *, methods=None, dependencies=None, name=None, **options):
        spec = _Spec(endpoint, tuple(self.dependencies) + tuple(dependencies or ()), dict(options))
        route = Route(self.prefix + path, _endpoint(spec), methods=methods or ["GET"], name=name or endpoint.__name__)
        route._hermes_spec = spec
        self.routes.append(route)

    def add_api_websocket_route(self, path, endpoint, *, dependencies=None, name=None, **options):
        spec = _Spec(endpoint, tuple(self.dependencies) + tuple(dependencies or ()), dict(options), True)
        route = WebSocketRoute(self.prefix + path, _endpoint(spec), name=name or endpoint.__name__)
        route._hermes_spec = spec
        self.routes.append(route)

    def api_route(self, path, *, methods=None, **options):
        def decorate(function):
            self.add_api_route(path, function, methods=methods, **options)
            return function
        return decorate

    def get(self, path, **options):
        return self.api_route(path, methods=["GET"], **options)

    def head(self, path, **options):
        return self.api_route(path, methods=["HEAD"], **options)

    def post(self, path, **options):
        return self.api_route(path, methods=["POST"], **options)

    def put(self, path, **options):
        return self.api_route(path, methods=["PUT"], **options)

    def patch(self, path, **options):
        return self.api_route(path, methods=["PATCH"], **options)

    def delete(self, path, **options):
        return self.api_route(path, methods=["DELETE"], **options)

    def options(self, path, **options):
        return self.api_route(path, methods=["OPTIONS"], **options)

    def websocket(self, path, **options):
        def decorate(function):
            self.add_api_websocket_route(path, function, **options)
            return function
        return decorate

    def include_router(self, router, *, prefix="", dependencies=None, **_metadata):
        for route in router.routes:
            spec = getattr(route, "_hermes_spec", None)
            if spec is None:
                raise TypeError("Dashboard subrouters must retain their typed parameter bindings")
            combined = [*(dependencies or ()), *spec.dependencies]
            register = self.add_api_websocket_route if spec.websocket else self.add_api_route
            kwargs = {} if spec.websocket else {"methods": list(route.methods)}
            register(prefix + route.path, spec.function, dependencies=combined, name=route.name, **kwargs, **spec.options)


class APIRouter(_Registration, Router):
    def __init__(self, *, prefix="", dependencies=None, tags=None, **kwargs):
        super().__init__(**kwargs)
        self.prefix, self.dependencies, self.tags = prefix, tuple(dependencies or ()), tags or []


async def _http_exception(_request, exc):
    if exc.status_code in (204, 304):
        return Response(status_code=exc.status_code, headers=exc.headers)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)


class FastAPI(_Registration, Starlette):
    def __init__(self, *, title="Hermes", version="", lifespan=None, dependencies=None, **kwargs):
        self.prefix, self.dependencies = "", tuple(dependencies or ())
        self.title, self.version = title, version
        super().__init__(lifespan=lifespan, **kwargs)
        self.add_exception_handler(HTTPException, _http_exception)

    def middleware(self, kind):
        if kind != "http":
            raise TypeError("Only HTTP function middleware is supported")

        def decorate(function):
            self.add_middleware(BaseHTTPMiddleware, dispatch=function)
            return function

        return decorate
