"""Audio, image and video HTTP endpoints consumed by the original Hermes tools."""

from __future__ import annotations

from contextlib import ExitStack
import mimetypes
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

from standalone.transport import split_request_options
from standalone.transport_chat import response_object
from standalone.transport_types import JSONStream, json_value


class BinaryResponse:
    """One owned response, including streaming headers and incremental file writes."""

    def __init__(self, response):
        self.response = response
        self.headers = response.headers
        self.status_code = response.status_code

    def read(self):
        try:
            return self.response.read()
        finally:
            self.close()

    def iter_bytes(self, chunk_size=None):
        try:
            yield from self.response.iter_bytes(chunk_size)
        finally:
            self.close()

    def stream_to_file(self, path):
        with open(path, "wb") as target:
            for chunk in self.iter_bytes():
                target.write(chunk)

    def close(self):
        self.response.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _form_fields(value, prefix=""):
    """Match bracketed multipart keys, retaining false, zero and ordered arrays."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _form_fields(item, f"{prefix}[{key}]" if prefix else str(key))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _form_fields(item, prefix + "[]")
    elif value is not None:
        yield prefix, str(value).lower() if isinstance(value, bool) else str(value)


def _file_part(value, stack):
    if isinstance(value, Path):
        stream = stack.enter_context(value.open("rb"))
        return value.name, stream, mimetypes.guess_type(value.name)[0] or "application/octet-stream"
    return value


def _multipart_request(client, path, body, options, names, *, stream=False):
    with ExitStack() as stack:
        files = []
        for name in names:
            value = body.pop(name, None)
            if value is None:
                continue
            # A tuple is HTTPX's (filename, bytes/file, content_type) input;
            # a list means a repeated upload field, as in images.edit.
            values = value if isinstance(value, list) else [value]
            key = name + "[]" if isinstance(value, list) else name
            files.extend((key, _file_part(item, stack)) for item in values)
        # Text fields as filename=None multipart parts retain duplicate [] keys.
        files.extend((key, (None, value)) for key, value in _form_fields(json_value(body)))
        return client.post(path, files=files, stream=stream, **options)


class Speech:
    def __init__(self, client):
        self._client = client

    @property
    def with_streaming_response(self):
        return self

    def create(self, **kwargs):
        body, options = split_request_options(kwargs)
        return BinaryResponse(self._client.post("audio/speech", body=body, stream=True, **options))


class Transcriptions:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        body, options = split_request_options(kwargs)
        streaming = bool(body.get("stream", False))
        text = body.get("response_format") in ("text", "srt", "vtt")
        response = _multipart_request(self._client, "audio/transcriptions", body, options, ("file",), stream=streaming)
        if streaming:
            return JSONStream(response)
        try:
            return response.text if text else response_object(response)
        finally:
            response.close()


def audio(client):
    return SimpleNamespace(speech=Speech(client), transcriptions=Transcriptions(client))


class Images:
    def __init__(self, client):
        self._client = client

    def _call(self, path, kwargs, files=()):
        body, options = split_request_options(kwargs)
        streaming = bool(body.get("stream", False))
        response = (_multipart_request(self._client, path, body, options, files, stream=streaming) if files
                    else self._client.post(path, body=body, stream=streaming, **options))
        if streaming:
            return JSONStream(response)
        try:
            return response_object(response)
        finally:
            response.close()

    def generate(self, **kwargs):
        return self._call("images/generations", kwargs)

    def edit(self, **kwargs):
        return self._call("images/edits", kwargs, ("image", "mask"))


class Videos:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        body, options = split_request_options(kwargs)
        response = _multipart_request(self._client, "videos", body, options, ("input_reference",))
        try:
            return response_object(response)
        finally:
            response.close()

    def retrieve(self, video_id, **kwargs):
        query, options = split_request_options(kwargs)
        options["query"] = {**query, **(options["query"] or {})}
        response = self._client.get("videos/" + quote(video_id, safe=""), **options)
        try:
            return response_object(response)
        finally:
            response.close()

    def download_content(self, video_id, **kwargs):
        query, options = split_request_options(kwargs)
        options["query"] = {**query, **(options["query"] or {})}
        return BinaryResponse(self._client.get("videos/" + quote(video_id, safe="") + "/content",
                                              stream=True, **options))
