"""Windows PCM capture/playback using the operating system's winMM API."""

from __future__ import annotations

import ctypes
import time

from standalone.audio_abi import bind, wait
from standalone.audio_device import AudioError


class WaveFormat(ctypes.Structure):
    _layout_ = "ms"
    _pack_ = 2
    _fields_ = [("format", ctypes.c_uint16), ("channels", ctypes.c_uint16), ("rate", ctypes.c_uint32),
                ("bytes_per_second", ctypes.c_uint32), ("block_align", ctypes.c_uint16),
                ("bits", ctypes.c_uint16), ("extra", ctypes.c_uint16)]


class WaveHeader(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("length", ctypes.c_uint32), ("recorded", ctypes.c_uint32),
                ("user", ctypes.c_size_t), ("flags", ctypes.c_uint32), ("loops", ctypes.c_uint32),
                ("next", ctypes.c_void_p), ("reserved", ctypes.c_size_t)]


class WaveInCaps(ctypes.Structure):
    _fields_ = [("manufacturer", ctypes.c_uint16), ("product", ctypes.c_uint16), ("version", ctypes.c_uint32),
                ("name", ctypes.c_uint16 * 32), ("formats", ctypes.c_uint32),
                ("channels", ctypes.c_uint16), ("reserved", ctypes.c_uint16)]


class WaveOutCaps(ctypes.Structure):
    _fields_ = WaveInCaps._fields_ + [("support", ctypes.c_uint32)]


class WindowsAudio:
    def __init__(self):
        self.library = library = ctypes.WinDLL("winmm", use_last_error=True)
        self._quarantined = []
        ptr, uint = ctypes.c_void_p, ctypes.c_uint32
        for prefix, caps in (("waveIn", WaveInCaps), ("waveOut", WaveOutCaps)):
            bind(library, prefix + "GetNumDevs", [], uint)
            bind(library, prefix + "GetDevCapsW", [ctypes.c_size_t, ctypes.POINTER(caps), uint], uint)
            bind(library, prefix + "GetErrorTextW", [uint, ctypes.c_wchar_p, uint], uint)
            bind(library, prefix + "Open", [ctypes.POINTER(ptr), uint, ctypes.POINTER(WaveFormat), ptr, ptr, uint], uint)
            for name in ("PrepareHeader", "UnprepareHeader"):
                bind(library, prefix + name, [ptr, ctypes.POINTER(WaveHeader), uint], uint)
            for name in ("Reset", "Close"):
                bind(library, prefix + name, [ptr], uint)
        bind(library, "waveInAddBuffer", [ptr, ctypes.POINTER(WaveHeader), uint], uint)
        bind(library, "waveInStart", [ptr], uint)
        bind(library, "waveOutWrite", [ptr, ctypes.POINTER(WaveHeader), uint], uint)

    def check(self, code, prefix, operation):
        if code:
            text = ctypes.create_unicode_buffer(256)
            getattr(self.library, prefix + "GetErrorTextW")(code, text, len(text))
            raise AudioError(f"Windows PCM {operation} failed ({code}): {text.value}")

    def query_devices(self):
        devices = []
        for prefix, kind, caps_type in (("waveIn", "input", WaveInCaps), ("waveOut", "output", WaveOutCaps)):
            count = getattr(self.library, prefix + "GetNumDevs")()
            if count > 1024:
                raise AudioError("Windows audio device count exceeds the enumeration limit")
            for index in range(count):
                caps = caps_type()
                self.check(getattr(self.library, prefix + "GetDevCapsW")(index, ctypes.byref(caps), ctypes.sizeof(caps)), prefix, "device query")
                name = bytes(caps.name).decode("utf-16-le", "replace").split("\0", 1)[0]
                devices.append({"id": f"{kind}:{index}", "name": name, "native_backend": "winmm",
                                "directions": [kind], "default_samplerate": 16000,
                                "max_" + kind + "_channels": caps.channels})
        return devices

    def open(self, direction, rate, channels, blocksize, stop, *, device=None):
        if len(self._quarantined) >= 16:
            raise AudioError("Windows retained failed audio buffers; restart the audio process before opening another stream")
        identifier = 0xFFFFFFFF
        if device is not None:
            if (not isinstance(device, dict) or device.get("native_backend") != "winmm"
                    or direction not in device.get("directions", ())):
                raise AudioError("Explicit Windows audio selection requires an enumerated device")
            prefix, separator, index = str(device.get("id", "")).partition(":")
            if prefix != direction or not separator or not index.isascii() or not index.isdecimal() or not 0 <= int(index) < 1024:
                raise AudioError("Invalid Windows native audio device ID")
            identifier = int(index)
        return WavePCM(self, direction, rate, channels, blocksize, stop, device=identifier)


class WavePCM:
    def __init__(self, api, direction, rate, channels, blocksize, stop, *, device=0xFFFFFFFF):
        self.api, self.direction, self.rate, self.channels, self.stop = api, direction, rate, channels, stop
        self.prefix = "waveIn" if direction == "input" else "waveOut"
        self.handle, self.buffers, self._pending, self._index = ctypes.c_void_p(), [], bytearray(), 0
        self.blockbytes = blocksize * channels * 2
        format = WaveFormat(1, channels, rate, rate * channels * 2, channels * 2, 16, 0)
        api.check(getattr(api.library, self.prefix + "Open")(ctypes.byref(self.handle), device,
                                                            ctypes.byref(format), None, None, 0), self.prefix, "open")
        try:
            if direction == "input":
                for _ in range(4):
                    raw = ctypes.create_string_buffer(self.blockbytes)
                    header = WaveHeader(ctypes.cast(raw, ctypes.c_void_p), self.blockbytes, 0, 0, 0, 0, None, 0)
                    api.check(api.library.waveInPrepareHeader(self.handle, ctypes.byref(header), ctypes.sizeof(header)), self.prefix, "prepare input")
                    self.buffers.append((raw, header))
                    api.check(api.library.waveInAddBuffer(self.handle, ctypes.byref(header), ctypes.sizeof(header)), self.prefix, "queue input")
                api.check(api.library.waveInStart(self.handle), self.prefix, "start capture")
        except Exception:
            self.close()
            raise

    def read(self, frames):
        needed, deadline = frames * self.channels * 2, time.monotonic() + max(5, frames / self.rate + 2)
        while len(self._pending) < needed:
            if self.stop.is_set():
                raise InterruptedError("Windows capture cancelled")
            raw, header = self.buffers[self._index]
            if header.flags & 1:
                amount = header.recorded
                if amount > self.blockbytes or amount % (self.channels * 2):
                    raise AudioError("Windows returned an invalid capture byte count")
                self._pending.extend(raw.raw[:amount])
                header.recorded = 0
                self.api.check(self.api.library.waveInAddBuffer(self.handle, ctypes.byref(header), ctypes.sizeof(header)), self.prefix, "requeue input")
                self._index = (self._index + 1) % len(self.buffers)
                if not amount:
                    wait(self.stop, deadline, "Windows audio capture returned empty buffers")
            else:
                wait(self.stop, deadline, "Windows audio capture timed out")
        result = bytes(self._pending[:needed])
        del self._pending[:needed]
        return result

    def write(self, data):
        for offset in range(0, len(data), self.blockbytes):
            if self.stop.is_set():
                raise InterruptedError("Windows playback cancelled")
            packet = data[offset:offset + self.blockbytes]
            raw = ctypes.create_string_buffer(packet, len(packet))
            header = WaveHeader(ctypes.cast(raw, ctypes.c_void_p), len(packet), 0, 0, 0, 0, None, 0)
            self.api.check(self.api.library.waveOutPrepareHeader(self.handle, ctypes.byref(header), ctypes.sizeof(header)), self.prefix, "prepare output")
            self.buffers.append((raw, header))
            try:
                self.api.check(self.api.library.waveOutWrite(self.handle, ctypes.byref(header), ctypes.sizeof(header)), self.prefix, "queue output")
                deadline = time.monotonic() + len(packet) / (self.rate * self.channels * 2) + 3
                while not header.flags & 1:
                    wait(self.stop, deadline, "Windows audio playback timed out")
            finally:
                if not header.flags & 1:
                    self.api.library.waveOutReset(self.handle)
                code = self.api.library.waveOutUnprepareHeader(self.handle, ctypes.byref(header), ctypes.sizeof(header))
                if not code:
                    self.buffers.remove((raw, header))
                else:
                    self.api.check(code, self.prefix, "release output buffer")

    def drain(self):
        # Each waveOutWrite packet waits for WHDR_DONE before returning.
        return None

    def close(self):
        if not self.handle:
            return
        library = self.api.library
        getattr(library, self.prefix + "Reset")(self.handle)
        for _raw, header in self.buffers:
            getattr(library, self.prefix + "UnprepareHeader")(self.handle, ctypes.byref(header), ctypes.sizeof(header))
        result = getattr(library, self.prefix + "Close")(self.handle)
        if result:
            # Preserve native-owned backing memory on failed release, rather
            # than let a driver retain a dangling Python allocation.
            self.api._quarantined.append((self.handle, self.buffers))
            self.handle, self.buffers = ctypes.c_void_p(), []
            self.api.check(result, self.prefix, "close")
        self.handle, self.buffers = ctypes.c_void_p(), []
