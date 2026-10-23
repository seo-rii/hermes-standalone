"""Cancellable PulseAudio native mainloop, also usable with PipeWire-Pulse."""

from __future__ import annotations

import ctypes
import os
import threading
import time

from standalone.audio_abi import bind, wait
from standalone.audio_device import AudioError, AudioOverflowError


class SampleSpec(ctypes.Structure):
    _fields_ = [("format", ctypes.c_int), ("rate", ctypes.c_uint32), ("channels", ctypes.c_uint8)]


class BufferAttr(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ("maxlength", "tlength", "prebuf", "minreq", "fragsize")]


class DevicePrefix(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("index", ctypes.c_uint32), ("description", ctypes.c_char_p),
                ("sample_spec", SampleSpec)]


class ServerInfoPrefix(ctypes.Structure):
    _fields_ = [(name, ctypes.c_char_p) for name in ("user_name", "host_name", "server_version", "server_name")]
    _fields_ += [("sample_spec", SampleSpec), ("default_sink_name", ctypes.c_char_p),
                 ("default_source_name", ctypes.c_char_p)]


_DEVICE_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(DevicePrefix), ctypes.c_int, ctypes.c_void_p)
_SERVER_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ServerInfoPrefix), ctypes.c_void_p)
_NOTIFY_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)


class PulseAudio:
    def __init__(self):
        self.library = library = ctypes.CDLL("libpulse.so.0")
        ptr, integer, size = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        bind(library, "pa_mainloop_new", [], ptr)
        bind(library, "pa_mainloop_get_api", [ptr], ptr)
        bind(library, "pa_mainloop_iterate", [ptr, integer, ctypes.POINTER(integer)], integer)
        bind(library, "pa_mainloop_free", [ptr], None)
        bind(library, "pa_context_new", [ptr, ctypes.c_char_p], ptr)
        bind(library, "pa_context_connect", [ptr, ctypes.c_char_p, integer, ptr], integer)
        bind(library, "pa_context_get_state", [ptr], integer)
        bind(library, "pa_context_errno", [ptr], integer)
        bind(library, "pa_strerror", [integer], ctypes.c_char_p)
        for name in ("pa_context_disconnect", "pa_context_unref", "pa_stream_unref", "pa_operation_unref"):
            bind(library, name, [ptr], None)
        bind(library, "pa_operation_get_state", [ptr], integer)
        bind(library, "pa_operation_cancel", [ptr], None)
        for kind in ("source", "sink"):
            bind(library, "pa_context_get_" + kind + "_info_list", [ptr, _DEVICE_CALLBACK, ptr], ptr)
        bind(library, "pa_context_get_server_info", [ptr, _SERVER_CALLBACK, ptr], ptr)
        bind(library, "pa_stream_new", [ptr, ctypes.c_char_p, ctypes.POINTER(SampleSpec), ptr], ptr)
        bind(library, "pa_stream_connect_record", [ptr, ctypes.c_char_p, ctypes.POINTER(BufferAttr), integer], integer)
        bind(library, "pa_stream_connect_playback", [ptr, ctypes.c_char_p, ctypes.POINTER(BufferAttr), integer, ptr, ptr], integer)
        bind(library, "pa_stream_get_state", [ptr], integer)
        bind(library, "pa_stream_readable_size", [ptr], size)
        bind(library, "pa_stream_writable_size", [ptr], size)
        bind(library, "pa_stream_peek", [ptr, ctypes.POINTER(ptr), ctypes.POINTER(size)], integer)
        bind(library, "pa_stream_drop", [ptr], integer)
        bind(library, "pa_stream_write", [ptr, ptr, size, ptr, ctypes.c_int64, integer], integer)
        bind(library, "pa_stream_disconnect", [ptr], integer)
        bind(library, "pa_stream_drain", [ptr, ptr, ptr], ptr)
        bind(library, "pa_stream_set_overflow_callback", [ptr, _NOTIFY_CALLBACK, ptr], None)

    def open(self, direction, rate, channels, blocksize, stop, *, device=None):
        return PulsePCM(self, direction, rate, channels, blocksize, stop, device=device)

    def query_devices(self):
        result, failed, defaults = [], [], {}
        with PulseConnection(self, threading.Event()) as connection:
            def server_info(_context, pointer, _userdata):
                if not pointer:
                    failed.append(True)
                    return
                defaults.update(input=pointer.contents.default_source_name, output=pointer.contents.default_sink_name)
            server_callback = _SERVER_CALLBACK(server_info)
            connection.operation(self.library.pa_context_get_server_info(connection.context, server_callback, None))
            for kind, direction in (("source", "input"), ("sink", "output")):
                def found(_context, pointer, end, _userdata):
                    if end < 0:
                        failed.append(True)
                    elif not end and pointer:
                        info = pointer.contents
                        if len(result) >= 1024:
                            failed.append(True)
                            return
                        result.append({"id": (info.name or b"").decode("utf-8", "replace"),
                                       "name": (info.description or info.name or b"").decode("utf-8", "replace"),
                                       "native_backend": "pulse", "directions": [direction],
                                       "default_samplerate": info.sample_spec.rate,
                                       "max_" + direction + "_channels": info.sample_spec.channels,
                                       "default": info.name == defaults.get(direction)})
                callback = _DEVICE_CALLBACK(found)
                operation = getattr(self.library, "pa_context_get_" + kind + "_info_list")(connection.context, callback, None)
                connection.operation(operation)
        if failed:
            raise AudioError("PulseAudio device enumeration failed or exceeded its bound")
        return result


class PulseConnection:
    def __init__(self, api, stop):
        self.api, self.stop, self.context = api, stop, None
        self.mainloop = api.library.pa_mainloop_new()
        if not self.mainloop:
            raise AudioError("PulseAudio could not create a native mainloop")
        try:
            self.context = api.library.pa_context_new(api.library.pa_mainloop_get_api(self.mainloop), b"Hermes PCM")
            if not self.context:
                raise AudioError("PulseAudio could not create a client context")
            server = os.environ.get("PULSE_SERVER")
            self.check(api.library.pa_context_connect(self.context, server.encode() if server else None, 1, None), "connect")
            deadline = time.monotonic() + 5
            while api.library.pa_context_get_state(self.context) != 4:
                self.pump()
                wait(stop, deadline, "PulseAudio server connection timed out")
        except Exception:
            self.close()
            raise

    def check(self, result, operation):
        if result < 0:
            code = self.api.library.pa_context_errno(self.context)
            message = self.api.library.pa_strerror(code).decode("utf-8", "replace")
            raise AudioError(f"PulseAudio {operation} failed: {message}")
        return result

    def pump(self):
        if self.stop.is_set():
            raise InterruptedError("PulseAudio operation cancelled")
        self.check(self.api.library.pa_mainloop_iterate(self.mainloop, 0, None), "mainloop")
        if self.api.library.pa_context_get_state(self.context) in (5, 6):
            self.check(-1, "connection")

    def operation(self, operation, *, timeout=5):
        if not operation:
            self.check(-1, "operation")
        try:
            deadline = time.monotonic() + timeout
            while self.api.library.pa_operation_get_state(operation) == 0:
                self.pump()
                wait(self.stop, deadline, "PulseAudio operation timed out")
            if self.api.library.pa_operation_get_state(operation) != 1:
                raise AudioError("PulseAudio operation was cancelled")
        finally:
            if self.api.library.pa_operation_get_state(operation) == 0:
                self.api.library.pa_operation_cancel(operation)
            self.api.library.pa_operation_unref(operation)

    def close(self):
        if self.context:
            self.api.library.pa_context_disconnect(self.context)
            self.api.library.pa_context_unref(self.context)
            self.context = None
        if self.mainloop:
            self.api.library.pa_mainloop_free(self.mainloop)
            self.mainloop = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class PulsePCM:
    def __init__(self, api, direction, rate, channels, blocksize, stop, *, device=None):
        self.api, self.direction, self.rate, self.channels, self.stop = api, direction, rate, channels, stop
        self.buffer_seconds = blocksize * 8 / rate
        self.connection = PulseConnection(api, stop)
        self.stream, self._pending, self._fault = None, bytearray(), None
        try:
            self.stream = api.library.pa_stream_new(self.connection.context, b"Hermes capture" if direction == "input" else b"Hermes playback",
                                                     ctypes.byref(SampleSpec(3, rate, channels)), None)
            if not self.stream:
                self.connection.check(-1, "stream creation")
            size = blocksize * channels * 2
            attr = BufferAttr(size * 8, size * 2, 0, size, size)
            self._overflow_callback = _NOTIFY_CALLBACK(self._overflow)
            if direction == "input":
                api.library.pa_stream_set_overflow_callback(self.stream, self._overflow_callback, None)
                result = api.library.pa_stream_connect_record(self.stream, device.encode() if device else None, ctypes.byref(attr), 0)
            else:
                result = api.library.pa_stream_connect_playback(self.stream, device.encode() if device else None, ctypes.byref(attr), 0, None, None)
            self.connection.check(result, "stream connection")
            deadline = time.monotonic() + 5
            while api.library.pa_stream_get_state(self.stream) != 2:
                self._pump()
                wait(stop, deadline, "PulseAudio PCM stream connection timed out")
        except Exception:
            self.close()
            raise

    def _overflow(self, _stream, _userdata):
        self._fault = AudioOverflowError("PulseAudio input overflow discarded capture samples")

    def _pump(self):
        self.connection.pump()
        if self.api.library.pa_stream_get_state(self.stream) in (3, 4):
            self.connection.check(-1, "PCM stream")
        if self._fault is not None:
            raise self._fault

    def read(self, frames):
        needed, deadline = frames * self.channels * 2, time.monotonic() + max(5, frames / self.rate + 2)
        while len(self._pending) < needed:
            self._pump()
            size = self.api.library.pa_stream_readable_size(self.stream)
            if size == ctypes.c_size_t(-1).value:
                self.connection.check(-1, "read size")
            if size:
                address, amount = ctypes.c_void_p(), ctypes.c_size_t()
                self.connection.check(self.api.library.pa_stream_peek(self.stream, ctypes.byref(address), ctypes.byref(amount)), "read")
                if amount.value > 1_048_576 or amount.value % (self.channels * 2):
                    raise AudioError("PulseAudio returned an oversized or misaligned capture block")
                if amount.value:
                    self._pending.extend(ctypes.string_at(address, amount.value) if address else b"\0" * amount.value)
                    self.connection.check(self.api.library.pa_stream_drop(self.stream), "read completion")
                    continue
            wait(self.stop, deadline, "PulseAudio capture timed out")
        result = bytes(self._pending[:needed])
        del self._pending[:needed]
        return result

    def write(self, data):
        offset, deadline = 0, time.monotonic() + max(5, len(data) / (self.rate * self.channels * 2) + 2)
        while offset < len(data):
            self._pump()
            available = self.api.library.pa_stream_writable_size(self.stream)
            if available == ctypes.c_size_t(-1).value:
                self.connection.check(-1, "write size")
            count = min(available, len(data) - offset, 262144)
            count -= count % (self.channels * 2)
            if count:
                raw = ctypes.create_string_buffer(data[offset:offset + count], count)
                self.connection.check(self.api.library.pa_stream_write(self.stream, raw, count, None, 0, 0), "write")
                offset += count
            else:
                wait(self.stop, deadline, "PulseAudio playback timed out")

    def drain(self):
        self.connection.operation(self.api.library.pa_stream_drain(self.stream, None, None),
                                  timeout=max(5, self.buffer_seconds + 2))

    def close(self):
        if self.stream:
            self.api.library.pa_stream_disconnect(self.stream)
            self.api.library.pa_stream_unref(self.stream)
            self.stream = None
        self.connection.close()
