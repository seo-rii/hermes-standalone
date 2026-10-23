"""Meet virtual routing and named-device PCM over OS APIs, without shell tools."""

from __future__ import annotations

import contextlib
import ctypes
import re
import threading

from standalone.audio_abi import bind
from standalone.audio_device import AudioError, OutputStream
from standalone.audio_pulse import PulseAudio, PulseConnection


_INDEX_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
_SUCCESS_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)


class PulseModules:
    """Keep one connection's module IDs; never reconnect and unload reused IDs."""

    def __init__(self, *, api=None, connection=None):
        self.api = api or PulseAudio()
        self.connection, self.modules = connection, []
        library, ptr = self.api.library, ctypes.c_void_p
        bind(library, "pa_context_load_module", [ptr, ctypes.c_char_p, ctypes.c_char_p, _INDEX_CALLBACK, ptr], ptr)
        bind(library, "pa_context_unload_module", [ptr, ctypes.c_uint32, _SUCCESS_CALLBACK, ptr], ptr)

    def load(self, name, arguments):
        if self.connection is None:
            self.connection = PulseConnection(self.api, threading.Event())
        received = []
        callback = _INDEX_CALLBACK(lambda _context, index, _user: received.append(index))
        self.connection.operation(self.api.library.pa_context_load_module(
            self.connection.context, name.encode(), arguments.encode(), callback, None))
        if len(received) != 1 or received[0] == 0xFFFFFFFF:
            raise AudioError("PulseAudio could not create the requested Meet virtual audio module")
        self.modules.append(received[0])
        return received[0]

    def close(self):
        if self.connection is None:
            return
        errors = []
        try:
            for identifier in reversed(self.modules):
                try:
                    completed = []
                    callback = _SUCCESS_CALLBACK(lambda _context, success, _user: completed.append(success))
                    self.connection.operation(self.api.library.pa_context_unload_module(
                        self.connection.context, identifier, callback, None))
                    if completed != [1]:
                        raise AudioError("PulseAudio could not unload an owned Meet virtual audio module")
                except OSError as error:
                    errors.append(error)
        finally:
            self.modules.clear()
            self.connection.close()
            self.connection = None
        if errors:
            raise AudioError("Meet virtual audio cleanup failed: " + str(errors[0]))


class MacDevices:
    """Read CoreAudio names/UIDs and select an exact output without default fallback."""

    def __init__(self, *, api=None, foundation=None):
        from standalone.audio_macos import MacAudio
        self.api = api or MacAudio()
        self.foundation = foundation or ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        from standalone.audio_macos import PropertyAddress
        ptr, uint = ctypes.c_void_p, ctypes.c_uint32
        bind(self.api.core, "AudioObjectGetPropertyDataSize", [uint, ctypes.POINTER(PropertyAddress), uint, ptr, ctypes.POINTER(uint)], ctypes.c_int32)
        bind(self.api.library, "AudioQueueSetProperty", [ptr, uint, ptr, uint], ctypes.c_int32)
        bind(self.foundation, "CFStringGetCString", [ptr, ctypes.c_void_p, ctypes.c_long, uint], ctypes.c_ubyte)
        bind(self.foundation, "CFStringCreateWithCString", [ptr, ctypes.c_char_p, uint], ptr)
        bind(self.foundation, "CFRelease", [ptr], None)

    def _string(self, device, selector):
        from standalone.audio_macos import PropertyAddress, fourcc
        value, size = ctypes.c_void_p(), ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        address = PropertyAddress(fourcc(selector), fourcc("glob"), 0)
        self.api.check(self.api.core.AudioObjectGetPropertyData(device, ctypes.byref(address), 0, None,
                       ctypes.byref(size), ctypes.byref(value)), "named audio device property")
        if size.value != ctypes.sizeof(value) or not value:
            raise AudioError("macOS returned an invalid named audio device string")
        try:
            raw = ctypes.create_string_buffer(16384)
            if not self.foundation.CFStringGetCString(value, raw, len(raw), 0x08000100):
                raise AudioError("macOS audio device string exceeds its bounded UTF-8 buffer")
            return raw.value.decode("utf-8")
        finally:
            self.foundation.CFRelease(value)

    def find(self, name):
        from standalone.audio_macos import PropertyAddress, fourcc
        if not isinstance(name, str) or not 1 <= len(name) <= 256 or "\0" in name:
            raise ValueError("Meet virtual audio selection requires a bounded device name")
        address, size = PropertyAddress(fourcc("dev#"), fourcc("glob"), 0), ctypes.c_uint32()
        self.api.check(self.api.core.AudioObjectGetPropertyDataSize(1, ctypes.byref(address), 0, None,
                       ctypes.byref(size)), "audio device list size")
        if size.value > 4096 or size.value % 4:
            raise AudioError("macOS audio device list exceeds its bounded capacity")
        identifiers = (ctypes.c_uint32 * (size.value // 4))()
        capacity = size.value
        self.api.check(self.api.core.AudioObjectGetPropertyData(1, ctypes.byref(address), 0, None,
                       ctypes.byref(size), identifiers), "audio device list")
        if size.value > capacity or size.value % 4:
            raise AudioError("macOS returned an invalid audio device list size")
        matches = [self._string(identifier, "uid ") for identifier in identifiers[:size.value // 4]
                   if self._string(identifier, "lnam").casefold() == name.casefold()]
        if len(matches) != 1:
            raise AudioError("The selected macOS virtual audio device is missing or ambiguous: " + name)
        if not matches[0]:
            raise AudioError("The selected macOS virtual audio device has no stable UID")
        return matches[0]

    def open(self, direction, rate, channels, blocksize, stop, *, uid):
        from standalone.audio_macos import fourcc
        if direction != "output":
            raise ValueError("Meet injects PCM into an explicit virtual output device")
        if not isinstance(uid, str) or not 1 <= len(uid) <= 4096 or "\0" in uid:
            raise ValueError("Meet virtual audio output requires a bounded nonempty device UID")
        stream = self.api.open(direction, rate, channels, blocksize, stop)
        try:
            value = self.foundation.CFStringCreateWithCString(None, uid.encode(), 0x08000100)
            if not value:
                raise AudioError("macOS could not create the explicit virtual-device UID")
            try:
                reference = ctypes.c_void_p(value)
                self.api.check(self.api.library.AudioQueueSetProperty(stream.handle, fourcc("aqcd"),
                               ctypes.byref(reference), ctypes.sizeof(reference)), "select virtual output device")
            finally:
                self.foundation.CFRelease(value)
        except BaseException:
            stream.close()
            raise
        return stream


class NativeBridge:
    def __init__(self, prefix="hermes_meet"):
        if not isinstance(prefix, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", prefix) is None:
            raise ValueError("Meet audio device prefix must be a bounded identifier")
        self.prefix, self.modules = prefix, None

    def setup(self):
        if self.modules is not None:
            raise AudioError("Meet virtual audio routing is already active")
        from hermes_platform.host.facts import os_family
        family = os_family()
        if family.startswith("linux"):
            sink, source = self.prefix + "_sink", self.prefix + "_src"
            self.modules = PulseModules()
            try:
                self.modules.load("module-null-sink", f"sink_name={sink} format=s16le rate=48000 channels=2 sink_properties=device.description=HermesMeetSink")
                self.modules.load("module-virtual-source", f"source_name={source} master={sink}.monitor")
            except BaseException:
                with contextlib.suppress(OSError):
                    self.close()
                raise
            return {"platform": "linux", "device_name": source, "sample_rate": 48000, "channels": 2,
                    "module_ids": list(self.modules.modules), "write_target": sink}
        if family == "darwin":
            uid = MacDevices().find("BlackHole 2ch")
            return {"platform": "darwin", "device_name": "BlackHole 2ch", "sample_rate": 48000, "channels": 2,
                    "module_ids": [], "write_target": "BlackHole 2ch", "write_uid": uid}
        raise AudioError("Meet virtual audio routing is supported on Linux and macOS only")

    def close(self):
        if self.modules is not None:
            modules, self.modules = self.modules, None
            modules.close()


class _NamedOutput:
    def __init__(self, info):
        self.info = info
        if info.get("platform") == "linux":
            self.backend = PulseAudio()
        elif info.get("platform") == "darwin":
            self.backend = MacDevices()
        else:
            raise AudioError("Meet PCM output has no supported native virtual device")

    def open(self, direction, rate, channels, blocksize, stop):
        if direction != "output":
            raise ValueError("Meet virtual routing opens output only")
        if self.info["platform"] == "linux":
            target = self.info.get("write_target")
            if not isinstance(target, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", target) is None:
                raise AudioError("Meet virtual sink is missing; default-device playback is refused")
            return self.backend.open(direction, rate, channels, blocksize, stop, device=target)
        uid = self.info.get("write_uid") or self.backend.find(self.info["write_target"])
        return self.backend.open(direction, rate, channels, blocksize, stop, uid=uid)


class NativePump:
    """Tail bounded raw-PCM blocks; aborting cancels the actual native write."""

    def __init__(self, info, path, state, stop_flag, *, provider=None):
        self.path, self.state, self.stop_flag = path, state, stop_flag
        self.stop_event = threading.Event()
        self.stream = OutputStream(provider or _NamedOutput(info), samplerate=24000, channels=1)
        self.stream.start()
        self.thread = threading.Thread(target=self._run, name="meet-native-pcm", daemon=True)
        try:
            self.thread.start()
        except BaseException:
            self.stream.close()
            raise

    def _run(self):
        pending = b""
        try:
            with open(self.path, "rb") as source:
                while not self.stop_event.is_set() and not self.stop_flag.get("stop"):
                    raw = source.read(65536)
                    if not raw:
                        self.stop_event.wait(.05)
                        continue
                    block = pending + raw
                    length = len(block) - len(block) % 2
                    if length:
                        self.stream.write(block[:length])
                    pending = block[length:]
        except Exception as error:
            if not self.stop_event.is_set():
                self.state.set(error="native Meet PCM pump failed: " + str(error), realtime_ready=False)
        finally:
            with contextlib.suppress(OSError):
                self.stream.close()

    def terminate(self):
        self.stop_event.set()
        self.stream.abort()

    def wait(self, timeout=3):
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise AudioError("Native Meet PCM worker did not stop within its deadline")
