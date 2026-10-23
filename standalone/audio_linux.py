"""Linux PCM through PulseAudio/PipeWire's Pulse service or ALSA OS APIs."""

from __future__ import annotations

import ctypes
import os
import time

from standalone.audio_abi import bind, wait
from standalone.audio_device import AudioError, AudioOverflowError


class ALSA:
    def __init__(self):
        self.library = library = ctypes.CDLL("libasound.so.2")
        ptr, integer, uint, frames, signed = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_ulong, ctypes.c_long
        bind(library, "snd_pcm_open", [ctypes.POINTER(ptr), ctypes.c_char_p, integer, integer], integer)
        bind(library, "snd_pcm_set_params", [ptr, integer, integer, uint, uint, integer, uint], integer)
        bind(library, "snd_pcm_readi", [ptr, ptr, frames], signed)
        bind(library, "snd_pcm_writei", [ptr, ptr, frames], signed)
        bind(library, "snd_pcm_wait", [ptr, integer], integer)
        for name in ("snd_pcm_drop", "snd_pcm_drain", "snd_pcm_close"):
            bind(library, name, [ptr], integer)
        bind(library, "snd_strerror", [integer], ctypes.c_char_p)
        bind(library, "snd_device_name_hint", [integer, ctypes.c_char_p, ctypes.POINTER(ptr)], integer)
        bind(library, "snd_device_name_get_hint", [ptr, ctypes.c_char_p], ptr)
        bind(library, "snd_device_name_free_hint", [ptr], integer)
        self.libc = ctypes.CDLL(None)
        bind(self.libc, "free", [ptr], None)

    def check(self, result, operation):
        if result < 0:
            message = self.library.snd_strerror(result).decode("utf-8", "replace")
            if result == -32:
                raise AudioOverflowError("ALSA PCM overrun or underrun: " + message)
            raise AudioError(f"ALSA {operation} failed: {message}")
        return result

    def open(self, direction, rate, channels, blocksize, stop, *, device="default"):
        return ALSAPCM(self, direction, rate, channels, blocksize, stop, device)

    def query_devices(self):
        hints, result = ctypes.c_void_p(), []
        self.check(self.library.snd_device_name_hint(-1, b"pcm", ctypes.byref(hints)), "device query")
        if not hints:
            return result
        try:
            array = ctypes.cast(hints, ctypes.POINTER(ctypes.c_void_p))
            for index in range(1024):
                if not array[index]:
                    break
                values = []
                for field in (b"NAME", b"DESC", b"IOID"):
                    value = self.library.snd_device_name_get_hint(array[index], field)
                    try:
                        values.append(ctypes.string_at(value).decode("utf-8", "replace") if value else "")
                    finally:
                        if value:
                            self.libc.free(value)
                name, description, io = values
                if not name or name == "null":
                    continue
                result.append({"id": name, "name": description or name, "native_backend": "alsa",
                               "directions": [io.lower()] if io else ["input", "output"],
                               "default_samplerate": 16000, "default": name == "default"})
            else:
                raise AudioError("ALSA device hints exceed the enumeration limit")
        finally:
            self.library.snd_device_name_free_hint(hints)
        return result


class ALSAPCM:
    def __init__(self, api, direction, rate, channels, blocksize, stop, device):
        self.api, self.direction, self.rate, self.channels, self.stop = api, direction, rate, channels, stop
        self.handle = ctypes.c_void_p()
        api.check(api.library.snd_pcm_open(ctypes.byref(self.handle), device.encode(), 1 if direction == "input" else 0, 1), "open")
        try:
            api.check(api.library.snd_pcm_set_params(self.handle, 2, 3, channels, rate, 1, 200000), "format selection")
        except Exception:
            self.close()
            raise

    def _transfer(self, buffer, frames, output):
        completed, deadline = 0, time.monotonic() + max(3, frames / self.rate + 2)
        operation = self.api.library.snd_pcm_writei if output else self.api.library.snd_pcm_readi
        while completed < frames:
            if self.stop.is_set():
                raise InterruptedError("ALSA PCM operation cancelled")
            result = operation(self.handle, ctypes.byref(buffer, completed * self.channels * 2), frames - completed)
            if result in (0, -11, -4):
                wait(self.stop, deadline, "ALSA PCM transfer timed out")
                continue
            self.api.check(result, "write" if output else "read")
            if result > frames - completed:
                raise AudioError("ALSA returned an invalid PCM frame count")
            completed += result

    def read(self, frames):
        buffer = ctypes.create_string_buffer(frames * self.channels * 2)
        self._transfer(buffer, frames, False)
        return buffer.raw

    def write(self, data):
        buffer = ctypes.create_string_buffer(data, len(data))
        self._transfer(buffer, len(data) // (self.channels * 2), True)

    def drain(self):
        deadline = time.monotonic() + 5
        while True:
            result = self.api.library.snd_pcm_drain(self.handle)
            if result not in (-11, -4):
                self.api.check(result, "drain")
                return
            wait(self.stop, deadline, "ALSA output drain timed out")

    def close(self):
        if self.handle:
            self.api.library.snd_pcm_drop(self.handle)
            self.api.library.snd_pcm_close(self.handle)
            self.handle = ctypes.c_void_p()


class LinuxAudio:
    def __init__(self):
        from standalone.audio_pulse import PulseAudio
        self.pulse = self.alsa = None
        try:
            self.pulse = PulseAudio()
        except OSError:
            pass
        try:
            self.alsa = ALSA()
        except OSError:
            pass
        if self.pulse is None and self.alsa is None:
            raise AudioError("Linux PulseAudio and ALSA operating-system libraries are unavailable")

    def open(self, direction, rate, channels, blocksize, stop, *, device=None):
        if device is not None:
            if not isinstance(device, dict) or direction not in device.get("directions", ()):
                raise AudioError("Explicit Linux audio selection requires an enumerated device")
            backend = {"alsa": self.alsa, "pulse": self.pulse}.get(device.get("native_backend"))
            if backend is None:
                raise AudioError("The selected Linux audio backend is unavailable")
            return backend.open(direction, rate, channels, blocksize, stop, device=device["id"])
        if os.environ.get("PIPEWIRE_REMOTE") and not os.environ.get("PULSE_SERVER"):
            if self.alsa is None:
                raise AudioError("The configured PipeWire remote requires the OS ALSA PipeWire plugin")
            return self.alsa.open(direction, rate, channels, blocksize, stop, device="pipewire")
        if self.pulse is not None:
            try:
                return self.pulse.open(direction, rate, channels, blocksize, stop)
            except InterruptedError:
                raise
            except OSError:
                if os.environ.get("PULSE_SERVER") or self.alsa is None:
                    raise
        elif os.environ.get("PULSE_SERVER"):
            raise AudioError("The configured PulseAudio server requires the OS libpulse client library")
        if self.alsa is None:
            raise AudioError("No usable Linux PCM backend is available")
        return self.alsa.open(direction, rate, channels, blocksize, stop,
                              device="pipewire" if os.environ.get("PIPEWIRE_REMOTE") else "default")

    def query_devices(self):
        if os.environ.get("PIPEWIRE_REMOTE") and not os.environ.get("PULSE_SERVER"):
            if self.alsa is None:
                raise AudioError("The configured PipeWire remote requires the OS ALSA PipeWire plugin")
            return self.alsa.query_devices()
        if self.pulse is not None:
            try:
                result = self.pulse.query_devices()
                if result or os.environ.get("PULSE_SERVER"):
                    return result
            except OSError:
                if os.environ.get("PULSE_SERVER") or self.alsa is None:
                    raise
        return self.alsa.query_devices() if self.alsa is not None else []
