"""Meet-owned virtual devices and PCM; controlled ABI peers never open hardware."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from standalone.audio_device import AudioError
from standalone.meet_audio import MacDevices, NativeBridge, NativePump, PulseModules, _NamedOutput
from standalone.tests.test_entrypoints import BundledSourceTestCase, support


class ModulePeer:
    """C callback ABI peer; its connection never contacts a sound server."""

    def __init__(self):
        self.operations, self.loaded, self.unloaded = [], [], []
        self.context, self.close_count, self.next_id = ctypes.c_void_p(41), 0, 70
        self.fail_load, self.fail_unload = False, set()
        self.library = SimpleNamespace(pa_context_load_module=self.load, pa_context_unload_module=self.unload)
        # bind() sets ctypes metadata, so retain plain function objects rather
        # than transient bound-method objects for the two native declarations.
        self.library.pa_context_load_module = lambda *args: self.load(*args)
        self.library.pa_context_unload_module = lambda *args: self.unload(*args)

    def load(self, context, name, arguments, callback, userdata):
        self.loaded.append((name.decode(), arguments.decode()))
        callback(context, 0xFFFFFFFF if self.fail_load else self.next_id, userdata)
        self.next_id += 1
        return len(self.loaded)

    def unload(self, context, identifier, callback, userdata):
        self.unloaded.append(identifier)
        callback(context, identifier not in self.fail_unload, userdata)
        return 100 + len(self.unloaded)

    def operation(self, operation):
        self.operations.append(operation)

    def close(self):
        self.close_count += 1


class ModuleOwnershipTests(unittest.TestCase):
    def test_callback_widths_owned_reverse_cleanup_and_no_reconnect(self):
        peer = ModulePeer()
        modules = PulseModules(api=peer, connection=peer)
        self.assertEqual(modules.load("module-null-sink", "sink_name=private"), 70)
        self.assertEqual(modules.load("module-virtual-source", "master=private.monitor"), 71)
        self.assertEqual(peer.library.pa_context_load_module.argtypes[3]._argtypes_[1], ctypes.c_uint32)
        peer.fail_unload.add(71)
        with self.assertRaisesRegex(AudioError, "cleanup failed"):
            modules.close()
        self.assertEqual(peer.unloaded, [71, 70])
        self.assertEqual(peer.close_count, 1)
        self.assertIsNone(modules.connection)
        modules.close()
        self.assertEqual(peer.close_count, 1)
        self.assertEqual(peer.operations, [1, 2, 101, 102])

    def test_failed_module_is_never_owned(self):
        peer = ModulePeer()
        modules = PulseModules(api=peer, connection=peer)
        modules.load("module-null-sink", "sink_name=private")
        peer.fail_load = True
        with self.assertRaisesRegex(AudioError, "could not create"):
            modules.load("module-virtual-source", "master=private.monitor")
        modules.close()
        self.assertEqual(peer.unloaded, [70])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux virtual routing selection")
    def test_owned_virtual_modules_roll_back_partial_setup(self):
        peer = ModulePeer()
        modules = PulseModules(api=peer, connection=peer)
        with patch("standalone.meet_audio.PulseModules", return_value=modules):
            bridge = NativeBridge("private_meeting")
            info = bridge.setup()
            self.assertEqual(info["write_target"], "private_meeting_sink")
            self.assertEqual(info["device_name"], "private_meeting_src")
            self.assertEqual(info["module_ids"], [70, 71])
            self.assertIn("master=private_meeting_sink.monitor", peer.loaded[1][1])
            bridge.close()
            bridge.close()
        self.assertEqual(peer.unloaded, [71, 70])
        self.assertEqual(peer.close_count, 1)
        peer = ModulePeer()
        modules = PulseModules(api=peer, connection=peer)
        original = peer.load

        def fail_source(context, name, *args):
            peer.fail_load = name == b"module-virtual-source"
            return original(context, name, *args)

        peer.library.pa_context_load_module = fail_source
        with patch("standalone.meet_audio.PulseModules", return_value=modules):
            bridge = NativeBridge("private_meeting")
            with self.assertRaises(AudioError):
                bridge.setup()
        self.assertEqual(peer.unloaded, [70])
        self.assertIsNone(bridge.modules)
        with self.assertRaises(ValueError):
            NativeBridge("bad module=arguments")


class MacDevicePeer:
    """CoreAudio CFString/property ABI peer; IDs and Unicode UIDs are independent."""

    def __init__(self):
        self.names = {10: ("Physical speakers", "physical"), 20: ("BlackHole 2ch", "BlackHole-가상-UID")}
        self.strings, self.released, self.queues, self.selected = {}, [], [], []
        self.property_failure = False
        self.core = SimpleNamespace(AudioObjectGetPropertyDataSize=lambda *args: self.size(*args),
                                    AudioObjectGetPropertyData=lambda *args: self.data(*args))
        self.library = SimpleNamespace(AudioQueueSetProperty=lambda *args: self.select(*args))
        self.foundation = SimpleNamespace(CFStringGetCString=lambda *args: self.string(*args),
                                         CFStringCreateWithCString=lambda *args: self.create(*args),
                                         CFRelease=lambda value: self.released.append(value.value if hasattr(value, "value") else value))

    def size(self, _identifier, _address, _length, _qualifier, size):
        ctypes.cast(size, ctypes.POINTER(ctypes.c_uint32))[0] = len(self.names) * 4
        return 0

    def data(self, identifier, address, _length, _qualifier, size, output):
        from standalone.audio_macos import PropertyAddress, fourcc
        selector = ctypes.cast(address, ctypes.POINTER(PropertyAddress)).contents.selector
        if selector == fourcc("dev#"):
            identifiers = (ctypes.c_uint32 * len(self.names))(*self.names)
            ctypes.memmove(output, identifiers, ctypes.sizeof(identifiers))
            ctypes.cast(size, ctypes.POINTER(ctypes.c_uint32))[0] = ctypes.sizeof(identifiers)
        else:
            index = 0 if selector == fourcc("lnam") else 1
            handle = 1000 + identifier * 2 + index
            self.strings[handle] = self.names[identifier][index]
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(handle)
            ctypes.cast(size, ctypes.POINTER(ctypes.c_uint32))[0] = ctypes.sizeof(ctypes.c_void_p)
        return 0

    def string(self, handle, buffer, capacity, encoding):
        self.check(encoding != 0x08000100, "UTF-8 encoding")
        value = self.strings[handle.value if hasattr(handle, "value") else handle].encode() + b"\0"
        if len(value) > capacity:
            return 0
        ctypes.memmove(buffer, value, len(value))
        return 1

    def create(self, _allocator, value, _encoding):
        handle = 9000 + len(self.strings)
        self.strings[handle] = value.decode()
        return handle

    def select(self, handle, selector, value, size):
        from standalone.audio_macos import fourcc
        self.check(selector != fourcc("aqcd") or size != ctypes.sizeof(ctypes.c_void_p), "AudioQueue device property")
        reference = ctypes.cast(value, ctypes.POINTER(ctypes.c_void_p)).contents.value
        self.selected.append((handle.value, self.strings[reference]))
        return -1 if self.property_failure else 0

    @staticmethod
    def check(status, operation):
        if status:
            raise AudioError("Controlled CoreAudio " + operation)

    def open(self, direction, rate, channels, blocksize, stop):
        device = SimpleNamespace(handle=ctypes.c_void_p(700 + len(self.queues)), closed=False)
        device.close = lambda: setattr(device, "closed", True)
        self.queues.append(device)
        return device


class ExplicitVirtualOutputTests(unittest.TestCase):
    def test_coreaudio_exact_name_uid_selection_release_and_failure_cleanup(self):
        peer = MacDevicePeer()
        devices = MacDevices(api=peer, foundation=peer.foundation)
        uid = devices.find("blackhole 2CH")
        self.assertEqual(uid, "BlackHole-가상-UID")
        self.assertEqual(len(peer.released), 3)
        queue = devices.open("output", 24000, 1, 1200, threading.Event(), uid=uid)
        self.assertEqual(peer.selected, [(700, uid)])
        self.assertFalse(queue.closed)
        self.assertEqual(len(peer.released), 4)
        peer.property_failure = True
        with self.assertRaises(AudioError):
            devices.open("output", 24000, 1, 1200, threading.Event(), uid=uid)
        self.assertTrue(peer.queues[-1].closed)
        peer.names[30] = ("BlackHole 2ch", "different-uid")
        with self.assertRaisesRegex(AudioError, "ambiguous"):
            devices.find("BlackHole 2ch")
        for uid in ("", "device\0hidden", "x" * 4097):
            with self.assertRaises(ValueError):
                devices.open("output", 24000, 1, 1200, threading.Event(), uid=uid)

    def test_pulse_output_never_uses_a_default_device(self):
        calls = []
        provider = SimpleNamespace(open=lambda *args, **kwargs: calls.append((args, kwargs)) or "stream")
        with patch("standalone.meet_audio.PulseAudio", return_value=provider):
            stream = _NamedOutput({"platform": "linux", "write_target": "private_sink"})
            self.assertEqual(stream.open("output", 24000, 1, 1200, threading.Event()), "stream")
            self.assertEqual(calls[0][1], {"device": "private_sink"})
            for target in (None, "", "bad\0sink"):
                with self.assertRaises(AudioError):
                    _NamedOutput({"platform": "linux", "write_target": target}).open("output", 24000, 1, 1200, threading.Event())
        self.assertEqual(len(calls), 1)


class PCMProvider:
    def __init__(self, *, fail=False, block=False):
        self.writes, self.close_count, self.fail, self.block = [], 0, fail, block
        self.changed = threading.Condition()
        self.entered = threading.Event()

    def open(self, direction, rate, channels, blocksize, stop):
        if (direction, rate, channels) != ("output", 24000, 1):
            raise AssertionError("Meet PCM must retain the upstream signed-16 mono format")
        self.stop = stop
        return self

    def write(self, raw):
        self.entered.set()
        with self.changed:
            self.writes.append(bytes(raw))
            self.changed.notify_all()
        if self.fail:
            raise AudioError("controlled virtual output failure")
        while self.block and not self.stop.wait(.01):
            pass
        if self.stop.is_set():
            raise InterruptedError("controlled virtual output cancelled")

    def close(self):
        self.close_count += 1


class PCMState:
    def __init__(self):
        self.values, self.changed = {}, threading.Event()

    def set(self, **values):
        self.values.update(values)
        self.changed.set()


class NativePumpTests(unittest.TestCase):
    def test_append_odd_byte_boundaries_and_cancel_a_blocked_native_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speaker.pcm"
            path.write_bytes(b"\x01\x02\x03")
            provider, state = PCMProvider(), PCMState()
            pump = NativePump({}, path, state, {"stop": False}, provider=provider)
            self.addCleanup(pump.terminate)
            with provider.changed:
                self.assertTrue(provider.changed.wait_for(lambda: len(b"".join(provider.writes)) >= 2, 3))
            with path.open("ab") as output:
                output.write(b"\x04\x05\x06")
            with provider.changed:
                self.assertTrue(provider.changed.wait_for(lambda: len(b"".join(provider.writes)) >= 6, 3))
            pump.terminate()
            pump.wait()
            self.assertEqual(b"".join(provider.writes), b"\x01\x02\x03\x04\x05\x06")
            self.assertEqual(provider.close_count, 1)
            self.assertFalse(state.values)
            provider = PCMProvider(block=True)
            pump = NativePump({}, path, state, {"stop": False}, provider=provider)
            self.assertTrue(provider.entered.wait(3))
            pump.terminate()
            pump.wait()
            self.assertEqual(provider.close_count, 1)

    def test_device_failure_is_visible_and_releases_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speaker.pcm"
            path.write_bytes(b"\x01\x02")
            provider, state = PCMProvider(fail=True), PCMState()
            pump = NativePump({}, path, state, {"stop": False}, provider=provider)
            self.assertTrue(state.changed.wait(3))
            pump.wait()
            self.assertFalse(state.values["realtime_ready"])
            self.assertIn("controlled virtual output failure", state.values["error"])
            self.assertEqual(provider.close_count, 1)
            pump.terminate()


_REALTIME_WORKER = r'''
import base64,contextlib,json,os,pathlib,sys,threading,time
from types import SimpleNamespace
context=json.loads(sys.argv[1]);scratch=pathlib.Path(sys.argv[2]);source=pathlib.Path(context['root'])/'app'
sys.path[:0]=[str(source),str(source.parent/'vendor')]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
from websockets.sync.server import serve
requests=[];peer_done=threading.Event();pcm=b'\x01\x02\x03\x04'*400
def peer(connection):
    try:
        assert connection.request.headers['Authorization']=='Bearer fixture-not-a-real-key'
        for kind in ('session.update','conversation.item.create','response.create'):
            message=json.loads(connection.recv());assert message['type']==kind;requests.append(message)
        assert requests[1]['item']['content'][0]['text']=='한글 realtime fixture'
        for chunk in (pcm[:3],pcm[3:1001],pcm[1001:]):
            connection.send(json.dumps({'type':'response.audio.delta','delta':base64.b64encode(chunk).decode()}))
        connection.send(json.dumps({'type':'response.done'}));peer_done.set()
        for message in connection:requests.append(json.loads(message))
    finally:peer_done.set()
server=serve(peer,'127.0.0.1',0);port=server.socket.getsockname()[1]
server_thread=threading.Thread(target=server.serve_forever,daemon=True);server_thread.start()
from standalone.agent_probe_worker import ProbeGuard
class RealtimeGuard(ProbeGuard):
    def __call__(self,event,args):
        if event.startswith('socket.'):
            if event in ('socket.getaddrinfo','socket.connect'):
                host,number=(args[0],args[1]) if event=='socket.getaddrinfo' else args[1][:2]
                assert host=='127.0.0.1' and number==port,(event,host,number)
                return
            if event=='socket.__new__':return
        return super().__call__(event,args)
guard=RealtimeGuard(source,source.parent/'vendor',scratch);sys.addaudithook(guard)
import standalone.meet_audio as audio
class VirtualPCM:
    def __init__(self):self.writes=[];self.changed=threading.Condition();self.close_count=0
    def open(self,direction,rate,channels,blocksize,stop):
        assert (direction,rate,channels)==('output',24000,1);return self
    def write(self,raw):
        with self.changed:self.writes.append(bytes(raw));self.changed.notify_all()
    def close(self):self.close_count+=1
provider=VirtualPCM();audio._NamedOutput=lambda _info:provider
from plugins.google_meet import meet_bot,process_manager as pm
from plugins.google_meet.realtime import openai_client
openai_client.REALTIME_URL='ws://127.0.0.1:'+str(port)+'/v1/realtime'
folder=scratch/'realtime';folder.mkdir()
cfg=SimpleNamespace(out_dir=folder,realtime_api_key='fixture-not-a-real-key',realtime_model='fixture',
                    realtime_voice='alloy',realtime_instructions='fixture-only')
state=meet_bot._BotState(folder,'abc-defg-hij','https://meet.google.com/abc-defg-hij')
rt={'bridge_info':{'platform':'linux','write_target':'private_sink'}};stop={'stop':False}
try:
    meet_bot._start_realtime_speaker(rt,cfg,stop,state)
    assert state.realtime_ready
    pm._write_active({'pid':0,'meeting_id':'abc-defg-hij','out_dir':str(folder),'mode':'realtime'})
    result=pm.enqueue_say('한글 realtime fixture');assert result['ok']
    assert peer_done.wait(5)
    with provider.changed:assert provider.changed.wait_for(lambda:len(b''.join(provider.writes))>=len(pcm),5)
    assert b''.join(provider.writes)==pcm
    deadline=time.monotonic()+5
    while not (folder/'say_processed.jsonl').exists():
        assert time.monotonic()<deadline
        threading.Event().wait(.01)
    processed=json.loads((folder/'say_processed.jsonl').read_text())
    assert processed['id']==result['enqueued_id'] and processed['result']['bytes_written']==len(pcm)
    assert rt['session'].audio_bytes_out==len(pcm)
finally:
    stop['stop']=True;meet_bot._teardown_realtime(rt);server.shutdown();server_thread.join(3)
assert not rt['speaker_thread'].is_alive() and not rt['native_pcm_pump'].thread.is_alive()
assert provider.close_count==1 and not guard.events,guard.events
assert not any(name in sys.modules for name in ('numpy','sounddevice','playwright'))
print(json.dumps({'original_realtime_speaker':True,'owned_pcm':True,'bytes':len(pcm)}))
'''


class OriginalRealtimeTests(BundledSourceTestCase):
    def test_original_queue_websocket_and_native_pcm_without_audio_hardware(self):
        with tempfile.TemporaryDirectory(prefix="realtime-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            process = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", _REALTIME_WORKER,
                                      json.dumps(self.context), str(scratch)], cwd=scratch / "work", env=environment,
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            self.assertTrue(json.loads(process.stdout)["original_realtime_speaker"])


if __name__ == "__main__":
    unittest.main()
