"""macOS PCM capture/playback through AudioQueue and CoreAudio OS APIs."""

from __future__ import annotations

import ctypes
import math
import queue
import threading
import time

from standalone.audio_abi import bind, wait
from standalone.audio_device import AudioError, AudioOverflowError


def fourcc(value):
    return int.from_bytes(value.encode("ascii"), "big")


class StreamDescription(ctypes.Structure):
    _fields_ = [("rate", ctypes.c_double), ("format", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("packet_bytes", ctypes.c_uint32), ("packet_frames", ctypes.c_uint32),
                ("frame_bytes", ctypes.c_uint32), ("channels", ctypes.c_uint32),
                ("bits", ctypes.c_uint32), ("reserved", ctypes.c_uint32)]


class QueueBuffer(ctypes.Structure):
    _fields_ = [("capacity", ctypes.c_uint32), ("data", ctypes.c_void_p), ("size", ctypes.c_uint32),
                ("user", ctypes.c_void_p), ("packet_capacity", ctypes.c_uint32),
                ("packets", ctypes.c_void_p), ("packet_count", ctypes.c_uint32)]


class PropertyAddress(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]


_BUFFER = ctypes.POINTER(QueueBuffer)
_INPUT_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, _BUFFER, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
_OUTPUT_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, _BUFFER)


class AudioBuffer(ctypes.Structure):
    _fields_ = [("channels", ctypes.c_uint32), ("size", ctypes.c_uint32), ("data", ctypes.c_void_p)]


class AudioBufferList(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint32), ("buffer", AudioBuffer * 1)]


class MacAudio:
    def __init__(self):
        self.library = library = ctypes.CDLL("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")
        self.core = core = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        self.foundation = foundation = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        self._quarantined = []
        ptr, uint, status = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32
        for name, callback in (("AudioQueueNewInput", _INPUT_CALLBACK), ("AudioQueueNewOutput", _OUTPUT_CALLBACK)):
            bind(library, name, [ctypes.POINTER(StreamDescription), callback, ptr, ptr, ptr, uint, ctypes.POINTER(ptr)], status)
        bind(library, "AudioQueueAllocateBuffer", [ptr, uint, ctypes.POINTER(_BUFFER)], status)
        bind(library, "AudioQueueEnqueueBuffer", [ptr, _BUFFER, uint, ptr], status)
        bind(library, "AudioQueueStart", [ptr, ptr], status)
        bind(library, "AudioQueueStop", [ptr, ctypes.c_ubyte], status)
        bind(library, "AudioQueueDispose", [ptr, ctypes.c_ubyte], status)
        bind(library, "AudioQueueGetProperty", [ptr, uint, ptr, ctypes.POINTER(uint)], status)
        bind(library, "AudioQueueSetProperty", [ptr, uint, ptr, uint], status)
        bind(core, "AudioObjectGetPropertyData", [uint, ctypes.POINTER(PropertyAddress), uint, ptr, ctypes.POINTER(uint), ptr], status)
        bind(core, "AudioObjectGetPropertyDataSize", [uint, ctypes.POINTER(PropertyAddress), uint, ptr, ctypes.POINTER(uint)], status)
        bind(foundation, "CFStringGetCString", [ptr, ptr, ctypes.c_long, uint], ctypes.c_ubyte)
        bind(foundation, "CFStringCreateWithCString", [ptr, ctypes.c_char_p, uint], ptr)
        bind(foundation, "CFRelease", [ptr], None)

    @staticmethod
    def check(status, operation):
        if status:
            raise AudioError(f"macOS AudioQueue {operation} failed (OSStatus {status})")

    def _property(self, identifier, selector, *, scope="glob", maximum=65536):
        address = PropertyAddress(fourcc(selector), fourcc(scope), 0)
        length = ctypes.c_uint32()
        self.check(self.core.AudioObjectGetPropertyDataSize(identifier, ctypes.byref(address), 0, None,
                   ctypes.byref(length)), "device property size")
        if length.value > maximum:
            raise AudioError("macOS audio device property exceeds its bound")
        if not length.value:
            return b""
        raw = ctypes.create_string_buffer(length.value)
        capacity = length.value
        self.check(self.core.AudioObjectGetPropertyData(identifier, ctypes.byref(address), 0, None,
                   ctypes.byref(length), raw), "device property")
        if length.value > capacity:
            raise AudioError("macOS audio device property grew past its buffer")
        return raw.raw[:length.value]

    def _string(self, identifier, selector):
        raw = self._property(identifier, selector, maximum=ctypes.sizeof(ctypes.c_void_p))
        if len(raw) != ctypes.sizeof(ctypes.c_void_p):
            raise AudioError("macOS returned an invalid audio device string")
        reference = ctypes.c_void_p.from_buffer_copy(raw)
        if not reference.value:
            raise AudioError("macOS returned an empty audio device string")
        try:
            output = ctypes.create_string_buffer(16384)
            if not self.foundation.CFStringGetCString(reference, output, len(output), 0x08000100):
                raise AudioError("macOS audio device string exceeds its UTF-8 buffer")
            return output.value.decode("utf-8")
        finally:
            self.foundation.CFRelease(reference)

    def query_devices(self):
        defaults, result = {}, []
        for selector, kind in (("dIn ", "input"), ("dOut", "output")):
            raw = self._property(1, selector, maximum=4)
            if len(raw) != 4:
                raise AudioError("macOS returned an invalid audio device ID size")
            defaults[kind] = ctypes.c_uint32.from_buffer_copy(raw).value
        raw = self._property(1, "dev#", maximum=4096)
        if len(raw) % 4:
            raise AudioError("macOS returned an invalid audio device list")
        identifiers = (ctypes.c_uint32 * (len(raw) // 4)).from_buffer_copy(raw)
        for identifier in identifiers:
            channels = {}
            for scope, kind in (("inpt", "input"), ("outp", "output")):
                configuration = self._property(identifier, "slay", scope=scope)
                if len(configuration) < 4:
                    raise AudioError("macOS returned an invalid audio buffer list")
                count = ctypes.c_uint32.from_buffer_copy(configuration).value
                offset = AudioBufferList.buffer.offset
                if count > 1024 or (count and offset + count * ctypes.sizeof(AudioBuffer) > len(configuration)):
                    raise AudioError("macOS returned an invalid audio buffer count")
                total = sum(AudioBuffer.from_buffer_copy(configuration, offset + i * ctypes.sizeof(AudioBuffer)).channels
                            for i in range(count))
                if total > 1024:
                    raise AudioError("macOS audio channel count exceeds its bound")
                channels[kind] = total
            directions = [kind for kind in channels if channels[kind]]
            if not directions:
                continue
            try:
                raw_rate = self._property(identifier, "nsrt", maximum=8)
                rate = ctypes.c_double.from_buffer_copy(raw_rate).value if len(raw_rate) == 8 else 0
                selected_rate = round(rate) if math.isfinite(rate) and rate > 0 else None
            except AudioError:
                selected_rate = None
            result.append({"id": identifier, "uid": self._string(identifier, "uid "),
                           "name": self._string(identifier, "lnam"), "native_backend": "audioqueue",
                           "directions": directions, "default_samplerate": selected_rate,
                           "max_input_channels": channels["input"], "max_output_channels": channels["output"],
                           "default_input": defaults["input"] == identifier,
                           "default_output": defaults["output"] == identifier})
        return result

    def select_queue_device(self, handle, uid):
        if not isinstance(uid, str) or not 1 <= len(uid) <= 4096 or "\0" in uid:
            raise AudioError("macOS audio selection requires a bounded nonempty device UID")
        reference = self.foundation.CFStringCreateWithCString(None, uid.encode("utf-8"), 0x08000100)
        if not reference:
            raise AudioError("macOS could not create the selected device UID")
        try:
            value = ctypes.c_void_p(reference)
            self.check(self.library.AudioQueueSetProperty(handle, fourcc("aqcd"), ctypes.byref(value),
                       ctypes.sizeof(value)), "select audio device")
        finally:
            self.foundation.CFRelease(reference)

    def open(self, direction, rate, channels, blocksize, stop, *, device=None):
        if len(self._quarantined) >= 16:
            raise AudioError("macOS retained failed audio queues; restart before opening another stream")
        uid = None
        if device is not None:
            if (not isinstance(device, dict) or device.get("native_backend") != "audioqueue"
                    or direction not in device.get("directions", ())):
                raise AudioError("Explicit macOS audio selection requires an enumerated device")
            uid = device.get("uid")
            if not isinstance(uid, str) or not uid:
                raise AudioError("The selected macOS device has no stable UID")
        return QueuePCM(self, direction, rate, channels, blocksize, stop, device_uid=uid)


class QueuePCM:
    def __init__(self, api, direction, rate, channels, blocksize, stop, *, device_uid=None):
        self.api, self.direction, self.rate, self.channels, self.stop = api, direction, rate, channels, stop
        self.handle, self.buffers = ctypes.c_void_p(), []
        self._input, self._free = queue.Queue(maxsize=8), queue.Queue(maxsize=4)
        self._lock, self._pending, self._error = threading.Lock(), bytearray(), None
        self._closing, self._started, self._inflight = False, False, 0
        self.blockbytes = blocksize * channels * 2
        description = StreamDescription(rate, fourcc("lpcm"), 12, channels * 2, 1, channels * 2, channels, 16, 0)
        callback_type = _INPUT_CALLBACK if direction == "input" else _OUTPUT_CALLBACK
        self._callback = callback_type(self._input_ready if direction == "input" else self._output_ready)
        function = api.library.AudioQueueNewInput if direction == "input" else api.library.AudioQueueNewOutput
        api.check(function(ctypes.byref(description), self._callback, None, None, None, 0, ctypes.byref(self.handle)), "open")
        try:
            if device_uid is not None:
                api.select_queue_device(self.handle, device_uid)
            for _ in range(4):
                buffer = _BUFFER()
                api.check(api.library.AudioQueueAllocateBuffer(self.handle, self.blockbytes, ctypes.byref(buffer)), "allocate buffer")
                if not buffer or buffer.contents.capacity < self.blockbytes or not buffer.contents.data:
                    raise AudioError("macOS allocated an invalid PCM buffer")
                self.buffers.append(buffer)
                if direction == "input":
                    api.check(api.library.AudioQueueEnqueueBuffer(self.handle, buffer, 0, None), "queue capture buffer")
                else:
                    self._free.put_nowait(buffer)
            if direction == "input":
                api.check(api.library.AudioQueueStart(self.handle, None), "start capture")
                self._started = True
        except Exception:
            self.close()
            raise

    def _input_ready(self, _user, native_queue, buffer, _time, _packets, _descriptions):
        try:
            with self._lock:
                if self._closing:
                    return
                info = buffer.contents
                if info.size > self.blockbytes or info.size > info.capacity or info.size % (self.channels * 2):
                    raise AudioError("macOS returned an invalid capture block size")
                if info.size:
                    try:
                        self._input.put_nowait(ctypes.string_at(info.data, info.size))
                    except queue.Full:
                        raise AudioOverflowError("macOS capture queue overflow discarded samples") from None
                self.api.check(self.api.library.AudioQueueEnqueueBuffer(native_queue, buffer, 0, None), "requeue capture buffer")
        except Exception as error:
            self._error = error

    def _output_ready(self, _user, _native_queue, buffer):
        try:
            with self._lock:
                if self._closing:
                    return
                self._inflight -= 1
                if self._inflight < 0:
                    raise AudioError("macOS returned an unqueued output buffer")
                self._free.put_nowait(buffer)
        except Exception as error:
            self._error = error

    def _check(self):
        if self._error is not None:
            raise self._error
        if self.stop.is_set() or self._closing:
            raise InterruptedError("macOS PCM operation cancelled")

    def read(self, frames):
        needed, deadline = frames * self.channels * 2, time.monotonic() + max(5, frames / self.rate + 2)
        while len(self._pending) < needed:
            self._check()
            try:
                self._pending.extend(self._input.get_nowait())
            except queue.Empty:
                wait(self.stop, deadline, "macOS audio capture timed out")
        result = bytes(self._pending[:needed])
        del self._pending[:needed]
        return result

    def write(self, data):
        deadline = time.monotonic() + len(data) / (self.rate * self.channels * 2) + 5
        for offset in range(0, len(data), self.blockbytes):
            while True:
                self._check()
                try:
                    buffer = self._free.get_nowait()
                    break
                except queue.Empty:
                    wait(self.stop, deadline, "macOS playback buffer timed out")
            packet = data[offset:offset + self.blockbytes]
            ctypes.memmove(buffer.contents.data, packet, len(packet))
            buffer.contents.size = len(packet)
            with self._lock:
                self._inflight += 1
            self.api.check(self.api.library.AudioQueueEnqueueBuffer(self.handle, buffer, 0, None), "queue output")
            if not self._started:
                self.api.check(self.api.library.AudioQueueStart(self.handle, None), "start playback")
                self._started = True

    def drain(self):
        if not self._started:
            return
        deadline = time.monotonic() + max(5, 4 * self.blockbytes / (self.rate * self.channels * 2) + 2)
        self.api.check(self.api.library.AudioQueueStop(self.handle, 0), "drain output")
        while True:
            self._check()
            running, length = ctypes.c_uint32(), ctypes.c_uint32(4)
            self.api.check(self.api.library.AudioQueueGetProperty(self.handle, fourcc("aqrn"), ctypes.byref(running), ctypes.byref(length)), "output state")
            if length.value != 4:
                raise AudioError("macOS returned an invalid playback state size")
            if not running.value:
                self._started = False
                return
            wait(self.stop, deadline, "macOS output drain timed out")

    def close(self):
        if not self.handle:
            return
        with self._lock:
            self._closing = True
        self.api.library.AudioQueueStop(self.handle, 1)
        status = self.api.library.AudioQueueDispose(self.handle, 1)
        if status:
            self.api._quarantined.append(self)
            self.api.check(status, "dispose")
        self.handle = ctypes.c_void_p()
        self.buffers.clear()
