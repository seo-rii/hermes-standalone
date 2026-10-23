"""Qualified Relay OTLP file export, derived from the pinned Rust exporter.

JSON uses OTLP hex identifiers and the native exporter's explicit defaults.
Binary records use a four-byte big-endian length followed by protobuf. Open
writers share a handle by resolved parent and reject conflicting modes/formats.
The configured directory is a trusted anchor; newly created descendants and
output files are opened relative to its handle without following symlinks.
"""
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import threading
import weakref


_LOCK = threading.RLock()
_WRITERS = weakref.WeakValueDictionary()


def validate_config(config):
    """Validate file settings without creating directories or opening a file."""
    directory = config.get('output_directory')
    if not isinstance(directory, str) or not directory.strip() or '\0' in directory:
        raise ValueError('Relay OTLP output_directory must be a nonblank path')
    directory = directory.strip()
    if '..' in Path(directory).parts:
        raise ValueError('Relay OTLP output_directory contains unsafe traversal')
    filename = config.get('filename')
    if filename is not None and (not isinstance(filename, str) or not filename or filename != filename.strip()
            or len(Path(filename).parts) != 1 or Path(filename).name in {'', '.', '..'} or '\0' in filename):
        raise ValueError('Relay OTLP filename must be a single nonblank unpadded path component')
    if config.get('format', 'json_lines') not in ('json_lines', 'proto'):
        raise ValueError('Relay OTLP format must be json_lines or proto')
    if config.get('mode', 'overwrite') not in ('append', 'overwrite'):
        raise ValueError('Relay OTLP mode must be append or overwrite')
    if config.get('type', 'full') not in ('full', 'gen_ai', 'openinference'):
        raise ValueError('Unknown Relay OTLP trace projection')
    return directory


def validate_destinations(destinations):
    if not isinstance(destinations, list):
        raise ValueError('Relay OTLP file_sinks must be an array')
    seen = set()
    for config in destinations:
        if not isinstance(config, dict):
            raise ValueError('Relay OTLP file sink must be an object')
        directory = validate_config(config)
        extension = 'otlp.pb' if config.get('format') == 'proto' else 'jsonl'
        filename = config.get('filename') or '<default>.' + extension
        path = Path(directory).absolute() / filename
        if path in seen:
            raise ValueError('Relay OTLP file sinks require distinct output paths')
        seen.add(path)
    return destinations


def _anchor(directory):
    """Find an existing trusted anchor and lexical missing components."""
    path = Path(directory).absolute()
    missing = []
    while True:
        try:
            metadata = path.stat()
        except FileNotFoundError:
            if path.parent == path:
                raise OSError('Relay output has no existing filesystem anchor') from None
            missing.append(path.name)
            path = path.parent
        else:
            if not stat.S_ISDIR(metadata.st_mode):
                raise NotADirectoryError('Relay output anchor is not a directory')
            return path, list(reversed(missing))


def _open_private(directory, filename, append):
    anchor, missing = _anchor(directory)
    if os.name == 'nt':
        from standalone.relay_files_windows import open_private
        return open_private(anchor, missing, filename, append)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_CLOEXEC', 0)
    parent = os.open(anchor, directory_flags)
    try:
        for component in missing:
            try:
                os.mkdir(component, mode=0o700, dir_fd=parent)
            except FileExistsError:
                pass
            child = os.open(component, directory_flags | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, 'O_CLOEXEC', 0)
        descriptor = os.open(filename, flags | (os.O_APPEND if append else 0), 0o600, dir_fd=parent)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError('Relay output must be a regular file')
            os.fchmod(descriptor, 0o600)
            if not append:
                os.ftruncate(descriptor, 0)
            return os.fdopen(descriptor, 'wb')
        except BaseException:
            os.close(descriptor)
            raise
    finally:
        os.close(parent)


def _hex_identifiers(value):
    if isinstance(value, list):
        return [_hex_identifiers(item) for item in value]
    if isinstance(value, dict):
        return {key: base64.b64decode(item).hex() if key in {'traceId', 'spanId', 'parentSpanId'}
                else _hex_identifiers(item) for key, item in value.items()}
    return value


def encode_request(payload, format):
    """Serialize an official trace request; an empty batch has no file record."""
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    message = ExportTraceServiceRequest.FromString(payload)
    if not any(scope.spans for resource in message.resource_spans for scope in resource.scope_spans):
        return b''
    if format == 'proto':
        if len(payload) > 0xffffffff:
            raise ValueError('Relay OTLP record exceeds the four-byte frame length')
        return len(payload).to_bytes(4, 'big') + payload
    if format != 'json_lines':
        raise ValueError('Unknown Relay OTLP file format')
    from google.protobuf.json_format import MessageToDict
    value = MessageToDict(message, use_integers_for_enums=True, always_print_fields_with_no_presence=True)
    return json.dumps(_hex_identifiers(value), ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode() + b'\n'


class _SharedWriter:
    def __init__(self, directory, filename, format, append):
        self.format, self.append = format, append
        self.lock = threading.RLock()
        self.file = None
        self.file = _open_private(directory, filename, append)

    def __del__(self):
        if self.file is not None:
            self.file.close()


class FileDestination:
    def __init__(self, config):
        directory = validate_config(config)
        self.format = config.get('format', 'json_lines')
        extension = 'jsonl' if self.format == 'json_lines' else 'otlp.pb'
        filename = Path(config['filename']).name if config.get('filename') is not None else 'nemo-relay-otlp-' + datetime.now(timezone.utc).strftime('%Y-%m-%d-%H.%M.%S') + '.' + extension
        self.path = Path(directory).absolute() / filename
        append = config.get('mode', 'overwrite') == 'append'
        # Only the configured directory can contain trusted filesystem aliases;
        # the leaf itself must never be resolved through a symlink.
        key = self.path.parent.resolve() / filename
        with _LOCK:
            shared = _WRITERS.get(key)
            if shared is not None:
                if shared.format != self.format or shared.append != append:
                    raise ValueError('Relay OTLP file is already open with a different format or mode')
            else:
                shared = _SharedWriter(directory, filename, self.format, append)
                _WRITERS[key] = shared
            self.writer = shared

    def write(self, payload):
        record = encode_request(payload, self.format)
        shared = self.writer
        if shared is None:
            raise OSError('Relay OTLP file destination is closed')
        if not record:
            return
        with shared.lock:
            shared.file.write(record)
            shared.file.flush()
            # Delivery is acknowledged only after the complete record reaches
            # disk, as with the pinned native file exporter.
            os.fsync(shared.file.fileno())

    def close(self):
        with _LOCK:
            shared, self.writer = self.writer, None
            if shared is not None:
                with shared.lock:
                    shared.file.flush()
                    os.fsync(shared.file.fileno())
            # Shared file ownership ends with the last destination. The weak
            # registry never retains a descriptor after that owner is dropped.
