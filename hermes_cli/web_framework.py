"""Dashboard dependency boundary for source installs and the standalone build."""

from agent.sdk_boundary import bundled

if bundled():
    from standalone.web import (
        APIRouter, BackgroundTasks, BaseModel, Body, CORSMiddleware, Depends,
        FastAPI, Field, File, FileResponse, Form, HTMLResponse, HTTPException,
        JSONResponse, Query, RedirectResponse, Request, Response, SecretStr,
        StaticFiles, StreamingResponse, StrictBool, StrictInt, UploadFile,
        ValidationError, WebSocket, WebSocketDisconnect, WebSocketException,
        field_validator, jsonable_encoder, status,
    )
else:
    from fastapi import (
        APIRouter, BackgroundTasks, Body, Depends, FastAPI, File, Form,
        HTTPException, Query, Request, UploadFile, WebSocket,
        WebSocketDisconnect, WebSocketException, status,
    )
    from fastapi.encoders import jsonable_encoder
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field, SecretStr, StrictBool, StrictInt, ValidationError, field_validator
