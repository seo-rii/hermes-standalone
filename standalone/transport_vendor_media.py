"""Mistral and ElevenLabs HTTP boundaries consumed by Hermes' voice tools."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from types import SimpleNamespace
from urllib.parse import quote

from standalone.transport import SyncHTTPClient, split_request_options
from standalone.transport_media import _multipart_request
from standalone.transport_types import APIResponseValidationError, JSONStream, wire_object


MAX_AUDIO_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 4 * ((MAX_AUDIO_BYTES + 2) // 3) + 256 * 1024


def _json_response(response):
    data = bytearray()
    try:
        for chunk in response.iter_bytes(64 * 1024):
            data.extend(chunk)
            if len(data) > MAX_JSON_BYTES:
                raise APIResponseValidationError("Voice response exceeds the size limit", response=response)
        try:
            value = json.loads(data)
        except (ValueError, UnicodeError):
            raise APIResponseValidationError("Invalid voice JSON response", response=response) from None
        if not isinstance(value, dict):
            raise APIResponseValidationError("Voice response must be a JSON object", response=response)
        return wire_object(value)
    finally:
        response.close()


class _MistralTranscriptions:
    def __init__(self, client):
        self._client = client

    def complete(self, **kwargs):
        body, options = split_request_options(kwargs)
        upload = body.get("file")
        if isinstance(upload, dict):
            if "content" not in upload:
                raise TypeError("Mistral transcription file requires content")
            body["file"] = (upload.get("file_name", "audio"), upload["content"])
        streaming = bool(body.get("stream", False))
        response = _multipart_request(self._client, "v1/audio/transcriptions", body, options, ("file",), stream=True)
        return JSONStream(response) if streaming else _json_response(response)


class _MistralSpeech:
    def __init__(self, client):
        self._client = client

    def complete(self, **kwargs):
        body, options = split_request_options(kwargs)
        extra = body.pop("additional_properties", None)
        if extra is not None:
            if not isinstance(extra, dict):
                raise TypeError("additional_properties must be a mapping")
            body.update(extra)
        response = self._client.post("v1/audio/speech", body=body, stream=True, **options)
        if body.get("stream"):
            return JSONStream(response)
        result = _json_response(response)
        encoded = result.get("audio_data")
        if not isinstance(encoded, str):
            raise APIResponseValidationError("Mistral speech response has no audio_data", response=response)
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except ValueError:
            raise APIResponseValidationError("Mistral speech audio_data is invalid base64", response=response) from None
        if len(decoded) > MAX_AUDIO_BYTES:
            raise APIResponseValidationError("Mistral speech exceeds the audio size limit", response=response)
        return result


class Mistral(SyncHTTPClient):
    def __init__(self, *, api_key, server_url="https://api.mistral.ai", timeout=120, max_retries=0, http_client=None):
        self.api_key = api_key
        # Hermes' existing Mistral TTS override is server_url, while other voice
        # provider configs commonly include /v1. Accept either spelling once.
        base = str(server_url).rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        super().__init__(base_url=base, timeout=timeout, max_retries=max_retries, http_client=http_client)

    @property
    def default_headers(self):
        return {"Accept": "application/json", "Authorization": "Bearer " + self.api_key, **self._custom_headers}

    @property
    def audio(self):
        return SimpleNamespace(transcriptions=_MistralTranscriptions(self), speech=_MistralSpeech(self))


@dataclass(frozen=True)
class ElevenLabsEnvironment:
    base: str
    wss: str = "wss://api.elevenlabs.io"


class AudioChunks:
    """Bounded binary response whose explicit close also works before iteration."""

    def __init__(self, response):
        self.response = response
        self._iterator = self._iterate()

    def _iterate(self):
        total = 0
        try:
            for chunk in self.response.iter_bytes(64 * 1024):
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    raise APIResponseValidationError("Speech response exceeds the audio size limit", response=self.response)
                yield chunk
        finally:
            self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self):
        self.response.close()


class _ElevenLabsSpeech:
    def __init__(self, client):
        self._client = client

    def convert(self, *, voice_id, **kwargs):
        if not isinstance(voice_id, str) or not voice_id or voice_id in {".", ".."}:
            raise ValueError("ElevenLabs voice_id must identify a voice")
        body, options = split_request_options(kwargs)
        query = dict(options.get("query") or {})
        for name in ("output_format", "enable_logging", "optimize_streaming_latency"):
            if name in body:
                query[name] = body.pop(name)
        options["query"] = query
        response = self._client.post("v1/text-to-speech/" + quote(voice_id, safe=""), body=body, stream=True, **options)
        return AudioChunks(response)


class ElevenLabs(SyncHTTPClient):
    def __init__(self, *, api_key, environment=None, base_url=None, timeout=60, max_retries=0, httpx_client=None):
        self.api_key = api_key
        base = base_url or (environment.base if environment is not None else "https://api.elevenlabs.io")
        base = str(base).rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        super().__init__(base_url=base, timeout=timeout, max_retries=max_retries, http_client=httpx_client)

    @property
    def default_headers(self):
        return {"Accept": "audio/mpeg", "xi-api-key": self.api_key, **self._custom_headers}

    @property
    def text_to_speech(self):
        return _ElevenLabsSpeech(self)
