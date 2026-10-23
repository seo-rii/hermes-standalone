"""PCM oracle vectors, native stream lifetimes and portable operating-system ABIs."""

from __future__ import annotations

import base64
import ctypes
import json
import os
from pathlib import Path
import queue
import struct
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from standalone import audio_samples as pcm
from standalone.audio_device import AudioError, AudioOverflowError, NativeAudioIO


class ControlledDevice:
    def __init__(self, provider, direction, rate, channels, blocksize, stop):
        self.provider, self.direction, self.stop = provider, direction, stop
        self.rate, self.channels, self.blocksize = rate, channels, blocksize
        self.blocks, self.writes = queue.Queue(maxsize=16), []
        self.opened, self.written, self.closed = threading.Event(), threading.Event(), threading.Event()
        self.close_count, self.drain_count = 0, 0
        self.opened.set()

    def read(self, frames):
        while not self.stop.is_set():
            try:
                item = self.blocks.get(timeout=.05)
                if isinstance(item, Exception):
                    raise item
                if len(item) != frames * self.channels * 2:
                    raise AudioError("Controlled device received an incorrect fixture block")
                return item
            except queue.Empty:
                continue
        raise InterruptedError("Controlled capture cancelled")

    def write(self, raw):
        self.writes.append(bytes(raw))
        self.written.set()
        if self.provider.output_failure:
            raise self.provider.output_failure
        while self.provider.hold_output and not self.stop.wait(.01):
            pass
        if self.stop.is_set():
            raise InterruptedError("Controlled playback cancelled")

    def drain(self):
        self.drain_count += 1

    def close(self):
        self.close_count += 1
        self.closed.set()


class ControlledProvider:
    def __init__(self):
        self.devices = []
        self.output_failure, self.hold_output = None, False

    def query_devices(self):
        return [{"id": "fixture", "name": "Controlled PCM", "directions": ["input", "output"],
                 "default_samplerate": 16000, "native_backend": "controlled"}]

    def open(self, direction, rate, channels, blocksize, stop):
        device = ControlledDevice(self, direction, rate, channels, blocksize, stop)
        self.devices.append(device)
        return device


class PCMSampleTests(unittest.TestCase):
    def test_independent_numpy_vectors(self):
        fixture = json.loads((Path(__file__).parent / "fixtures" / "audio.json").read_text())
        for case in fixture["pcm"]:
            options = case["options"]
            if case["kind"] == "warmup":
                data = pcm.PCM16(base64.b64decode(case["input"]), case["channels"])
                actual = pcm.warmup(data, **options)
            else:
                actual = getattr(pcm, case["kind"])(**options)
            expected = pcm.PCM16(base64.b64decode(case["pcm"]), actual.channels)
            self.assertEqual(actual.shape, expected.shape)
            self.assertLessEqual(max(abs(a - b) for a, b in zip(actual.values(), expected.values())), 1,
                                 (case["kind"], options))
        for case in fixture["rms"]:
            self.assertAlmostEqual(pcm.rms(pcm.from_samples(case["samples"])), case["value"], places=10)
        for case in fixture["percentiles"]:
            self.assertAlmostEqual(pcm.percentile(case["samples"], case["percentage"]), case["value"], places=10)

    def test_pcm_frames_bounds_and_bad_inputs(self):
        block = pcm.from_samples([-40000, -1.9, 0, 1.9, 40000, 4], channels=2)
        self.assertEqual(list(block.values()), [-32768, -1, 0, 1, 32767, 4])
        self.assertEqual(block.shape, (3, 2))
        self.assertIs(block.copy(), block)
        self.assertEqual(pcm.PCM16(memoryview(bytearray(block.tobytes())).cast("h"), 2).tobytes(), block.tobytes())
        self.assertEqual(pcm.concatenate([block, pcm.silence(2, 2)]).shape, (5, 2))
        self.assertEqual(pcm.beep(duration=0).nbytes, 0)
        for data, channels in ((b"x", 1), (b"xx", 2), (b"", 0), (b"", 9)):
            with self.assertRaises(ValueError):
                pcm.PCM16(data, channels)
        for sample in (float("nan"), float("inf"), "1"):
            with self.assertRaises(ValueError):
                pcm.from_samples([sample])
        with patch.object(pcm, "MAX_BYTES", 8):
            with self.assertRaises(ValueError):
                pcm.concatenate([pcm.silence(3), pcm.silence(2)])
            with self.assertRaises(ValueError):
                pcm.silence(5)
        with self.assertRaises(ValueError):
            pcm.as_pcm(block, 1)
        with self.assertRaises(ValueError):
            pcm.percentile([], 90)
        with self.assertRaises(ValueError):
            pcm.format_size(16000, 20)


class NativeStreamTests(unittest.TestCase):
    def test_capture_callback_error_restart_and_manual_context(self):
        provider = ControlledProvider()
        audio, delivered, failures = NativeAudioIO(provider), threading.Event(), threading.Event()
        received = []

        def callback(data, frames, time_info, status):
            if status:
                received.append(status)
                failures.set()
            else:
                received.append((data.tobytes(), frames, time_info))
                delivered.set()

        stream = audio.InputStream(blocksize=160, callback=callback)
        self.assertFalse(provider.devices)
        stream.start()
        device = provider.devices[-1]
        raw = pcm.from_samples([1234, -1234] * 80).tobytes()
        device.blocks.put(raw)
        self.assertTrue(delivered.wait(3))
        self.assertEqual(received[0][:2], (raw, 160))
        self.assertIn("inputBufferAdcTime", received[0][2])
        device.blocks.put(AudioOverflowError("controlled overrun"))
        self.assertTrue(failures.wait(3))
        self.assertTrue(device.closed.wait(3))
        self.assertFalse(stream.active)
        self.assertIsInstance(stream.error, AudioOverflowError)
        stream.close()
        self.assertEqual(device.close_count, 1)
        stream.start()
        self.assertTrue(stream.active)
        stream.stop()
        self.assertEqual(provider.devices[-1].close_count, 1)
        with audio.InputStream(blocksize=160) as manual:
            provider.devices[-1].blocks.put(raw)
            block, overflow = manual.read(160)
            self.assertFalse(overflow)
            self.assertEqual(block.tobytes(), raw)
        self.assertEqual(provider.devices[-1].close_count, 1)

    def test_output_drain_abort_and_background_errors(self):
        provider = ControlledProvider()
        audio = NativeAudioIO(provider)
        raw = pcm.from_samples([0, 1000, -1000] * 100)
        with audio.OutputStream(samplerate=16000) as stream:
            stream.write(raw)
        device = provider.devices[-1]
        self.assertEqual(b"".join(device.writes), raw.tobytes())
        self.assertEqual((device.drain_count, device.close_count), (1, 1))
        audio.play(raw, samplerate=16000, blocksize=128)
        playback = audio.get_stream()
        playback._thread.join(3)
        self.assertFalse(playback.active)
        self.assertEqual(provider.devices[-1].drain_count, 1)
        audio.stop()
        self.assertEqual(provider.devices[-1].close_count, 1)
        provider.hold_output = True
        audio.play(raw, samplerate=16000)
        device = provider.devices[-1]
        self.assertTrue(device.written.wait(3))
        audio.stop()
        self.assertTrue(device.closed.is_set())
        self.assertEqual(device.drain_count, 0)
        provider.hold_output, provider.output_failure = False, AudioError("controlled output error")
        audio.play(raw, samplerate=16000)
        failed = audio.get_stream()
        failed._thread.join(3)
        with self.assertRaisesRegex(AudioError, "controlled output error"):
            _active = failed.active
        audio.stop()
        before = len(provider.devices)
        audio.play(pcm.PCM16(), samplerate=16000)
        self.assertEqual(len(provider.devices), before)
        self.assertIsNone(audio.get_stream())

    def test_invalid_format_and_device_absence_are_errors(self):
        audio = NativeAudioIO(ControlledProvider())
        with self.assertRaises(ValueError):
            audio.InputStream(dtype="float32")
        with self.assertRaises(ValueError):
            audio.OutputStream(blocksize=65537)
        with self.assertRaises(AudioError):
            audio.query_devices(device="nonexistent", kind="input")
        with self.assertRaises(AudioError):
            audio.OutputStream().write(b"xx")


class PortableAudioABITests(unittest.TestCase):
    def test_fixed_width_structures(self):
        from standalone.audio_windows import WaveFormat, WaveHeader, WaveInCaps, WaveOutCaps
        from standalone.audio_macos import PropertyAddress, QueueBuffer, StreamDescription, fourcc
        from standalone.audio_pulse import BufferAttr, DevicePrefix, SampleSpec
        pointer = ctypes.sizeof(ctypes.c_void_p)
        self.assertEqual(ctypes.sizeof(WaveFormat), 18)
        self.assertEqual(WaveFormat.rate.offset, 4)
        self.assertEqual(ctypes.sizeof(WaveHeader), 48 if pointer == 8 else 32)
        self.assertEqual(WaveHeader.flags.offset, 24 if pointer == 8 else 16)
        self.assertEqual(ctypes.sizeof(WaveInCaps), 80)
        self.assertEqual(ctypes.sizeof(WaveOutCaps), 84)
        self.assertEqual(ctypes.sizeof(StreamDescription), 40)
        self.assertEqual(StreamDescription.channels.offset, 28)
        self.assertEqual(ctypes.sizeof(QueueBuffer), 56 if pointer == 8 else 28)
        self.assertEqual(ctypes.sizeof(PropertyAddress), 12)
        self.assertEqual(fourcc("lpcm"), 0x6C70636D)
        self.assertEqual(ctypes.sizeof(SampleSpec), 12)
        self.assertEqual(ctypes.sizeof(BufferAttr), 20)
        self.assertEqual(DevicePrefix.sample_spec.offset, 24 if pointer == 8 else 12)

    def test_audioqueue_native_callback_copies_and_reports_overflow(self):
        from standalone.audio_macos import QueueBuffer, QueuePCM, _INPUT_CALLBACK
        device = object.__new__(QueuePCM)
        device._lock, device._input = threading.Lock(), queue.Queue(maxsize=1)
        device._closing, device._error, device.blockbytes, device.channels = False, None, 4, 1
        queued = []
        device.api = SimpleNamespace(library=SimpleNamespace(AudioQueueEnqueueBuffer=lambda *args: queued.append(args) or 0),
                                     check=lambda result, operation: self.assertEqual(result, 0, operation))
        raw = ctypes.create_string_buffer(b"\x01\0\x02\0")
        info = QueueBuffer(4, ctypes.cast(raw, ctypes.c_void_p), 4, None, 0, None, 0)
        callback = _INPUT_CALLBACK(device._input_ready)
        callback(None, None, ctypes.pointer(info), None, 0, None)
        self.assertEqual(device._input.get_nowait(), b"\x01\0\x02\0")
        callback(None, None, ctypes.pointer(info), None, 0, None)
        callback(None, None, ctypes.pointer(info), None, 0, None)
        self.assertIsInstance(device._error, AudioOverflowError)
        self.assertEqual(len(queued), 2)
        device._input.get_nowait()
        device._closing = True
        callback(None, None, ctypes.pointer(info), None, 0, None)
        self.assertTrue(device._input.empty())

    def test_winmm_pcm_native_buffer_lifetimes(self):
        from standalone.audio_windows import WaveHeader, WavePCM
        headers, writes, released = [], [], []

        def native_open(handle, *_args):
            ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(123)
            return 0

        def add(handle, header, size):
            item = ctypes.cast(header, ctypes.POINTER(WaveHeader)).contents
            ctypes.memset(item.data, 0, item.length)
            item.recorded, item.flags = item.length, item.flags | 1
            return 0

        def write(handle, header, size):
            item = ctypes.cast(header, ctypes.POINTER(WaveHeader)).contents
            writes.append(ctypes.string_at(item.data, item.length))
            item.flags |= 1
            return 0

        library = SimpleNamespace(waveInOpen=native_open, waveOutOpen=native_open,
            waveInPrepareHeader=lambda *args: headers.append(args) or 0,
            waveOutPrepareHeader=lambda *args: 0, waveInAddBuffer=add, waveInStart=lambda *_args: 0,
            waveOutWrite=write, waveOutReset=lambda *_args: 0, waveInReset=lambda *_args: 0,
            waveOutUnprepareHeader=lambda *_args: 0, waveInUnprepareHeader=lambda *_args: 0,
            waveOutClose=lambda *_args: released.append("output") or 0,
            waveInClose=lambda *_args: released.append("input") or 0)
        api = SimpleNamespace(library=library, _quarantined=[], check=lambda code, *_args: self.assertEqual(code, 0))
        source = WavePCM(api, "input", 16000, 1, 160, threading.Event())
        self.assertEqual(source.read(480), b"\0" * 960)
        self.assertEqual(len(headers), 4)
        source.close()
        source.close()
        target = WavePCM(api, "output", 16000, 1, 160, threading.Event())
        raw = pcm.from_samples([1000, -1000] * 200).tobytes()
        target.write(raw)
        self.assertEqual(b"".join(writes), raw)
        self.assertFalse(target.buffers)
        target.close()
        self.assertEqual(released, ["input", "output"])


class NativeHostAudioTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows winMM OS library")
    def test_actual_winmm_device_enumeration_without_capture(self):
        from standalone.audio_windows import WindowsAudio
        api = WindowsAudio()
        devices = api.query_devices()
        self.assertIsInstance(devices, list)
        self.assertTrue(all(entry["native_backend"] == "winmm" for entry in devices))
        self.assertFalse(api._quarantined)

    @unittest.skipUnless(sys.platform == "darwin", "macOS AudioQueue/CoreAudio OS libraries")
    def test_actual_coreaudio_device_enumeration_without_capture(self):
        from standalone.audio_macos import MacAudio
        api = MacAudio()
        devices = api.query_devices()
        self.assertIsInstance(devices, list)
        self.assertTrue(all(entry["native_backend"] == "audioqueue" for entry in devices))
        self.assertFalse(api._quarantined)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux ALSA virtual PCM")
    def test_actual_alsa_explicit_null_pcm(self):
        from standalone.audio_linux import ALSA
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "alsa.conf"
            config.write_text("pcm.null { type null }\n")
            with patch.dict(os.environ, {"HOME": directory, "ALSA_CONFIG_PATH": str(config)}):
                self._alsa_null_io(ALSA)

    def _alsa_null_io(self, ALSA):
        try:
            api = ALSA()
        except OSError as error:
            self.skipTest(str(error))
        stop = threading.Event()
        source = api.open("input", 16000, 1, 160, stop, device="null")
        try:
            self.assertEqual(len(source.read(160)), 320)
            stop.set()
            with self.assertRaises(InterruptedError):
                source.read(160)
        finally:
            source.close()
        target = api.open("output", 16000, 2, 160, threading.Event(), device="null")
        try:
            target.write(pcm.from_samples([1000, -1000] * 160, 2).tobytes())
            target.drain()
        finally:
            target.close()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux PulseAudio client library")
    def test_actual_pulse_explicit_missing_private_server(self):
        from standalone.audio_pulse import PulseAudio
        try:
            api = PulseAudio()
        except OSError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "client.conf"
            config.write_text("autospawn = no\n")
            with patch.dict(os.environ, {"PULSE_SERVER": "unix:" + directory + "/absent", "HOME": directory,
                                         "PULSE_CLIENTCONFIG": str(config), "PULSE_COOKIE": directory + "/absent-cookie"}):
                with self.assertRaisesRegex(AudioError, "PulseAudio"):
                    api.query_devices()


if __name__ == "__main__":
    unittest.main()
