"""Application-owned PCM input/output streams over native operating-system APIs.

No PortAudio or array-library namespace is provided. Only explicit Hermes PCM
operations are exposed; construction and imports never open a microphone.
"""

from __future__ import annotations

from contextlib import suppress
import threading
import time

from standalone.audio_samples import PCM16, as_pcm, format_size


class AudioError(OSError):
    pass


class AudioOverflowError(AudioError):
    pass


def _block_size(rate, channels, size):
    format_size(rate, channels)
    if type(size) is not int or size < 0 or size > 65536:
        raise ValueError("Audio block size must be between zero and 65536 frames")
    return size or max(128, rate // 20)


def _select_device(devices, selector, kind):
    if type(selector) not in (int, str) or (isinstance(selector, str) and (not selector.strip() or len(selector) > 4096 or "\0" in selector)):
        raise AudioError("Audio device selection requires an index or a bounded nonempty name")
    if len(devices) > 1024:
        raise AudioError("Audio device enumeration exceeds its bound")
    if type(selector) is int and 0 <= selector < len(devices):
        selected = [devices[selector]]
    else:
        selected = [entry for entry in devices if entry.get("id") == selector]
        if not selected and isinstance(selector, str):
            needle = selector.casefold()
            selected = [entry for entry in devices if needle == entry.get("name", "").casefold()]
            if not selected:
                terms = needle.split()
                selected = [entry for entry in devices
                            if all(term in (entry.get("name", "") + " " + entry.get("native_backend", "")).casefold()
                                   for term in terms)]
    selected = [entry for entry in selected if kind is None or kind in entry["directions"]]
    if len(selected) != 1:
        raise AudioError("Selected " + (kind or "audio") + " device is missing or ambiguous")
    return dict(selected[0])


class InputStream:
    def __init__(self, provider, *, samplerate=16000, channels=1, dtype="int16", blocksize=0, callback=None, device=None):
        if dtype != "int16":
            raise ValueError("Native capture accepts signed 16-bit PCM only")
        self.samplerate, self.channels = format_size(samplerate, channels)
        self.blocksize = _block_size(samplerate, channels, blocksize)
        self._provider, self._callback = provider, callback
        self._selector = device
        self._stop, self._lock = threading.Event(), threading.Lock()
        self._device, self._thread, self._active, self.error = None, None, False, None

    @property
    def active(self):
        return self._active and not self._stop.is_set()

    def start(self):
        if self._device is not None:
            raise AudioError("Audio input stream is already open")
        self._stop.clear()
        self.error = None
        options = {} if self._selector is None else {
            "device": _select_device(self._provider.query_devices(), self._selector, "input")}
        self._device = self._provider.open("input", self.samplerate, self.channels, self.blocksize, self._stop, **options)
        self._active = True
        if self._callback is not None:
            self._thread = threading.Thread(target=self._capture, name="hermes-pcm-input", daemon=True)
            self._thread.start()
        return self

    def _capture(self):
        try:
            while not self._stop.is_set():
                block, overflow = self.read(self.blocksize)
                if self._stop.is_set():
                    break
                self._callback(block, len(block), {"inputBufferAdcTime": time.monotonic()}, overflow)
        except Exception as error:
            if not self._stop.is_set():
                self.error = error
                with suppress(Exception):
                    self._callback(PCM16(channels=self.channels), 0, {}, error)
        finally:
            self._active = False
            with self._lock:
                if self._device is not None:
                    device, self._device = self._device, None
                    try:
                        device.close()
                    except Exception as error:
                        self.error = self.error or error

    def read(self, frames):
        if type(frames) is not int or not 1 <= frames <= 65536:
            raise ValueError("Audio read requires one to 65536 frames")
        with self._lock:
            if not self.active or self._device is None:
                raise self.error or AudioError("Audio input stream is not active")
            raw = self._device.read(frames)
        if len(raw) != frames * self.channels * 2:
            raise AudioError("Audio input returned an incomplete PCM block")
        return PCM16(raw, self.channels), False

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
            if thread.is_alive():
                raise AudioError("Audio input did not stop within its deadline")
        with self._lock:
            if self._device is not None:
                self._device.close()
                self._device = None
        self._active = False

    close = stop

    def __enter__(self):
        return self.start()

    def __exit__(self, *_args):
        self.close()


class OutputStream:
    def __init__(self, provider, *, samplerate=24000, channels=1, dtype="int16", blocksize=0, device=None):
        if dtype != "int16":
            raise ValueError("Native output accepts signed 16-bit PCM only")
        self.samplerate, self.channels = format_size(samplerate, channels)
        self.blocksize = _block_size(samplerate, channels, blocksize)
        self._provider, self._stop, self._lock = provider, threading.Event(), threading.Lock()
        self._selector = device
        self._device, self.error, self._drained = None, None, False

    @property
    def active(self):
        return self._device is not None and not self._stop.is_set()

    def start(self):
        if self._device is not None:
            raise AudioError("Audio output stream is already open")
        self._stop.clear()
        self.error, self._drained = None, False
        options = {} if self._selector is None else {
            "device": _select_device(self._provider.query_devices(), self._selector, "output")}
        self._device = self._provider.open("output", self.samplerate, self.channels, self.blocksize, self._stop, **options)
        return self

    def write(self, data):
        pcm = as_pcm(data, self.channels)
        with self._lock:
            if not self.active:
                raise self.error or AudioError("Audio output stream is not active")
            try:
                self._device.write(pcm.tobytes())
                self._drained = False
            except Exception as error:
                self.error = error
                raise
        return False

    def drain(self):
        with self._lock:
            if self.active and not self._drained:
                self._device.drain()
                self._drained = True

    def stop(self):
        try:
            if self.error is None:
                self.drain()
        finally:
            self.abort()

    def abort(self):
        self._stop.set()
        with self._lock:
            if self._device is not None:
                self._device.close()
                self._device = None

    close = abort

    def __enter__(self):
        return self.start()

    def __exit__(self, error_type, *_args):
        try:
            if error_type is None:
                self.drain()
        finally:
            self.close()


class _Playback:
    def __init__(self, owner, pcm, samplerate, blocksize):
        self.stream = owner.OutputStream(samplerate=samplerate, channels=pcm.channels, blocksize=blocksize)
        self.error, self._pcm = None, pcm
        self.stream.start()
        self._thread = threading.Thread(target=self._play, name="hermes-pcm-output", daemon=True)
        self._thread.start()

    @property
    def active(self):
        if self.error is not None:
            raise self.error
        return self._thread.is_alive()

    def _play(self):
        try:
            raw = self._pcm.tobytes()
            step = self.stream.blocksize * self.stream.channels * 2
            for offset in range(0, len(raw), step):
                if not self.stream.active:
                    return
                self.stream.write(raw[offset:offset + step])
            self.stream.drain()
        except Exception as error:
            if not self.stream._stop.is_set():
                self.error = error
        finally:
            self.stream.close()

    def stop(self):
        self.stream.abort()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise AudioError("Audio playback did not stop within its deadline")


class NativeAudioIO:
    """The explicit application PCM boundary, with an injectable OS provider."""

    def __init__(self, provider):
        self.provider, self._playback, self._play_lock = provider, None, threading.RLock()

    def query_devices(self, device=None, kind=None):
        devices = self.provider.query_devices()
        if len(devices) > 1024:
            raise AudioError("Audio device enumeration exceeds its bound")
        devices = [dict(entry, index=index) for index, entry in enumerate(devices)]
        if device is None and kind is None:
            return devices
        if device is not None:
            return _select_device(devices, device, kind)
        selected = [entry for entry in devices if kind is None or kind in entry["directions"]]
        if not selected:
            raise AudioError("No accessible " + (kind or "audio") + " device was found")
        return next((entry for entry in selected if entry.get("default_" + kind, entry.get("default"))), selected[0])

    def InputStream(self, **options):
        return InputStream(self.provider, **options)

    def OutputStream(self, **options):
        return OutputStream(self.provider, **options)

    def play(self, data, *, samplerate, blocksize=0):
        pcm = as_pcm(data)
        format_size(samplerate, pcm.channels)
        with self._play_lock:
            self.stop()
            if len(pcm):
                self._playback = _Playback(self, pcm, samplerate, blocksize)

    def get_stream(self):
        return self._playback

    def stop(self):
        with self._play_lock:
            playback, self._playback = self._playback, None
            if playback is not None:
                playback.stop()


_DEFAULT, _DEFAULT_LOCK = None, threading.Lock()


def native_audio_io():
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            from hermes_platform.host.facts import os_family
            family = os_family()
            if family.startswith("linux"):
                from standalone.audio_linux import LinuxAudio
                provider = LinuxAudio()
            elif family == "win32":
                from standalone.audio_windows import WindowsAudio
                provider = WindowsAudio()
            elif family == "darwin":
                from standalone.audio_macos import MacAudio
                provider = MacAudio()
            else:
                raise AudioError("This operating system has no supported native PCM backend")
            _DEFAULT = NativeAudioIO(provider)
        return _DEFAULT
