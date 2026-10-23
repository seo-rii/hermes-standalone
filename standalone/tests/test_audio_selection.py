"""Explicit microphone/output selection, native IDs and no default fallback."""

import ctypes as C
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from standalone.audio_device import AudioError, NativeAudioIO
from standalone.tests.test_audio import ControlledProvider


class SelectedProvider(ControlledProvider):
    def __init__(self):
        super().__init__()
        self.selected = []
        self.entries = [
            {"id": "source-a", "name": "USB Microphone", "native_backend": "controlled", "directions": ["input"]},
            {"id": "sink-a", "name": "USB Speaker", "native_backend": "controlled", "directions": ["output"]},
            {"id": "source-b", "name": "Conference Microphone", "native_backend": "controlled", "directions": ["input"], "default": True},
        ]

    def query_devices(self):
        return self.entries

    def open(self, *args, device=None):
        self.selected.append(device)
        return super().open(*args)


class AudioSelectionTests(unittest.TestCase):
    def test_index_id_exact_name_and_unique_substring_select_the_requested_device(self):
        provider = SelectedProvider()
        audio = NativeAudioIO(provider)
        self.assertEqual(audio.query_devices(kind="input")["id"], "source-b")
        self.assertEqual([row["index"] for row in audio.query_devices()], [0, 1, 2])
        for selector in (2, "source-b", "Conference Microphone", "conference controlled"):
            with self.subTest(selector=selector):
                with audio.InputStream(device=selector):
                    self.assertEqual(provider.selected[-1]["id"], "source-b")
        with audio.OutputStream(device=1):
            self.assertEqual(provider.selected[-1]["id"], "sink-a")
        with audio.InputStream():
            self.assertIsNone(provider.selected[-1])

    def test_missing_ambiguous_wrong_direction_and_invalid_selectors_never_open_default(self):
        provider = SelectedProvider()
        audio = NativeAudioIO(provider)
        for selector in ("Microphone", "unknown", 1, -1, 9999, " ", "\0", True, {}):
            with self.subTest(selector=selector), self.assertRaises(AudioError):
                audio.InputStream(device=selector).start()
        self.assertFalse(provider.selected)
        self.assertFalse(provider.devices)

    def test_linux_explicit_backend_error_cannot_fall_back_to_another_device(self):
        from standalone.audio_linux import LinuxAudio
        calls = []
        api = object.__new__(LinuxAudio)
        def pulse(*args, **options):
            calls.append(options)
            raise AudioError("Selected source is gone")
        api.pulse = SimpleNamespace(open=pulse)
        api.alsa = SimpleNamespace(open=lambda *args, **options: self.fail("Explicit selection fell back to ALSA"))
        descriptor = {"id": "selected-source", "native_backend": "pulse", "directions": ["input"]}
        with self.assertRaisesRegex(AudioError, "source is gone"):
            api.open("input",16000,1,160,threading.Event(),device=descriptor)
        self.assertEqual(calls, [{"device": "selected-source"}])

    def test_winmm_open_receives_the_exact_native_id(self):
        from standalone.audio_windows import WindowsAudio, WaveHeader
        opened = []
        def native_open(handle, identifier, *_args):
            opened.append(identifier)
            C.cast(handle, C.POINTER(C.c_void_p))[0] = 77
            return 0
        def add(_handle, header, _size):
            value = C.cast(header, C.POINTER(WaveHeader)).contents
            value.flags |= 1
            return 0
        library = SimpleNamespace(waveInOpen=native_open, waveInPrepareHeader=lambda *args:0,
            waveInAddBuffer=add,waveInStart=lambda *args:0,waveInReset=lambda *args:0,
            waveInUnprepareHeader=lambda *args:0,waveInClose=lambda *args:0)
        api = object.__new__(WindowsAudio)
        api.library, api._quarantined = library, []
        descriptor = {"id":"input:7","native_backend":"winmm","directions":["input"]}
        stream = api.open("input",16000,1,160,threading.Event(),device=descriptor)
        stream.close()
        self.assertEqual(opened, [7])
        descriptor["id"] = "output:7"
        with self.assertRaises(AudioError):
            api.open("input",16000,1,160,threading.Event(),device=descriptor)
        self.assertEqual(opened, [7])

    def test_pulse_enumeration_preserves_server_default_and_native_source_names(self):
        from standalone.audio_pulse import PulseAudio, DevicePrefix, SampleSpec, ServerInfoPrefix
        defaults = ServerInfoPrefix(b"user",b"host",b"version",b"server",SampleSpec(3,48000,2),b"sink-b",b"source-b")
        def server_info(context,callback,user):
            callback(context,C.pointer(defaults),user);return 1
        def device_info(direction):
            def found(context,callback,user):
                for index,suffix in enumerate((b"a",b"b")):
                    name=(b"source-" if direction=="input" else b"sink-")+suffix
                    item=DevicePrefix(name,index,b"Fixture "+name,SampleSpec(3,48000,2))
                    callback(context,C.pointer(item),0,user)
                callback(context,None,1,user);return 1
            return found
        api=object.__new__(PulseAudio)
        api.library=SimpleNamespace(pa_context_get_server_info=server_info,
            pa_context_get_source_info_list=device_info("input"),pa_context_get_sink_info_list=device_info("output"))
        connection=SimpleNamespace(context=123,operation=lambda *args:None)
        # Special methods must be defined on the context-manager type.
        class Connection:
            def __enter__(self):return connection
            def __exit__(self,*args):pass
        with patch("standalone.audio_pulse.PulseConnection",return_value=Connection()):
            audio=NativeAudioIO(api)
            self.assertEqual(audio.query_devices(kind="input")["id"],"source-b")
            self.assertEqual(audio.query_devices(kind="output")["id"],"sink-b")
            self.assertEqual(audio.query_devices("source-a","input")["max_input_channels"],2)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Private Linux ALSA null PCM")
    def test_actual_named_virtual_alsa_device_through_application_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=root/"alsa.conf"
            config.write_text('pcm.hermes_fixture { type null hint { show on description "Private Hermes fixture" } }\n')
            program=r'''
import json,pathlib,sys
sys.path.insert(0,sys.argv[1])
def guard(event,args):
 if event=='open' and isinstance(args[0],str) and args[0].startswith('/dev/snd'):
  raise AssertionError('The virtual test must never open real audio hardware')
sys.addaudithook(guard)
from standalone.audio_device import NativeAudioIO
from standalone.audio_linux import ALSA,LinuxAudio
api=object.__new__(LinuxAudio);api.pulse=None
try:api.alsa=ALSA()
except OSError:raise SystemExit(77)
audio=NativeAudioIO(api)
info=audio.query_devices('hermes_fixture','input');assert info['id']=='hermes_fixture',info
with audio.InputStream(device='hermes_fixture',samplerate=16000,blocksize=160) as stream:
 pcm,overflow=stream.read(160);assert not overflow and len(pcm)==160
with audio.OutputStream(device='hermes_fixture',samplerate=16000,blocksize=160) as stream:
 stream.write(b'\0'*320)
print(json.dumps({'ok':True}))
'''
            result=subprocess.run([sys.executable,"-I","-S","-B","-c",program,str(Path(__file__).resolve().parents[2])],
                env={"HOME":directory,"ALSA_CONFIG_PATH":str(config),"PATH":""},
                capture_output=True,text=True,timeout=15)
            if result.returncode==77:self.skipTest("OS ALSA library is unavailable")
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(json.loads(result.stdout)["ok"])


class CoreAudioPeer:
    """Raw property, CFString and AudioQueue callback peer, independent of host OS."""
    def __init__(self):
        from standalone.audio_macos import AudioBuffer, AudioBufferList, QueueBuffer, fourcc
        self.strings, self.released, self.events, self.buffers = {}, [], [], []
        self.properties, self.fail_selection = {}, False
        def prop(identifier, selector, value, scope="glob"):
            self.properties[identifier,fourcc(selector),fourcc(scope)] = value
        prop(1,"dIn ",struct.pack("=I",30));prop(1,"dOut",struct.pack("=I",10))
        prop(1,"dev#",struct.pack("=III",10,20,30))
        for identifier,name,uid,inputs,outputs in (
            (10,"Speakers","speakers",[],[2]),(20,"USB Mic","usb-마이크",[1,1],[]),
            (30,"Conference Mic","conference",[1],[2]),
        ):
            for selector,text in (("lnam",name),("uid ",uid)):
                handle=1000+len(self.strings);self.strings[handle]=text
                prop(identifier,selector,bytes(C.c_void_p(handle)))
            prop(identifier,"nsrt",struct.pack("=d",48000))
            for scope,channels in (("inpt",inputs),("outp",outputs)):
                raw=C.create_string_buffer(AudioBufferList.buffer.offset+len(channels)*C.sizeof(AudioBuffer))
                C.cast(raw,C.POINTER(C.c_uint32))[0]=len(channels)
                for index,count in enumerate(channels):
                    item=AudioBuffer(count,0,None)
                    C.memmove(C.addressof(raw)+AudioBufferList.buffer.offset+index*C.sizeof(item),C.byref(item),C.sizeof(item))
                prop(identifier,"slay",raw.raw,scope)
        self.core=SimpleNamespace(AudioObjectGetPropertyDataSize=self.size,AudioObjectGetPropertyData=self.data)
        self.foundation=SimpleNamespace(CFStringGetCString=self.string,CFStringCreateWithCString=self.create,CFRelease=self.release)
        def new(_description,_callback,_user,_loop,_mode,_flags,handle):
            self.events.append("new");C.cast(handle,C.POINTER(C.c_void_p))[0]=77;return 0
        def allocate(_handle,length,output):
            raw=C.create_string_buffer(length);buffer=C.pointer(QueueBuffer(length,C.cast(raw,C.c_void_p),0,None,0,None,0))
            self.buffers.append((raw,buffer));C.cast(output,C.POINTER(C.POINTER(QueueBuffer)))[0]=buffer
            self.events.append("allocate");return 0
        self.library=SimpleNamespace(AudioQueueNewInput=new,AudioQueueAllocateBuffer=allocate,
            AudioQueueEnqueueBuffer=lambda *args:self.events.append("enqueue") or 0,
            AudioQueueStart=lambda *args:self.events.append("start") or 0,
            AudioQueueStop=lambda *args:self.events.append("stop") or 0,
            AudioQueueDispose=lambda *args:self.events.append("dispose") or 0,
            AudioQueueSetProperty=self.select)

    def key(self,identifier,address):
        from standalone.audio_macos import PropertyAddress
        value=C.cast(address,C.POINTER(PropertyAddress)).contents
        return identifier,value.selector,value.scope

    def size(self,identifier,address,_length,_qualifier,size):
        C.cast(size,C.POINTER(C.c_uint32))[0]=len(self.properties[self.key(identifier,address)])
        return 0

    def data(self,identifier,address,_length,_qualifier,size,output):
        raw=self.properties[self.key(identifier,address)]
        if C.cast(size,C.POINTER(C.c_uint32))[0]<len(raw):return -1
        C.memmove(output,raw,len(raw));C.cast(size,C.POINTER(C.c_uint32))[0]=len(raw);return 0

    def string(self,value,output,capacity,_encoding):
        handle=value.value if hasattr(value,"value") else value
        raw=self.strings[handle].encode()+b"\0"
        if len(raw)>capacity:return 0
        C.memmove(output,raw,len(raw));return 1

    def create(self,_allocator,raw,_encoding):
        handle=1000+len(self.strings);self.strings[handle]=raw.decode();return handle

    def release(self,value):
        self.released.append(value.value if hasattr(value,"value") else value)

    def select(self,_handle,_selector,reference,size):
        if size!=C.sizeof(C.c_void_p):return -1
        handle=C.cast(reference,C.POINTER(C.c_void_p))[0]
        self.events.append(("select",self.strings[handle]))
        return -1 if self.fail_selection else 0


class CoreAudioSelectionTests(unittest.TestCase):
    def make_api(self):
        from standalone.audio_macos import MacAudio
        peer=CoreAudioPeer();api=object.__new__(MacAudio)
        api.core,api.library,api.foundation,api._quarantined=peer.core,peer.library,peer.foundation,[]
        return peer,api

    def test_all_devices_channels_default_and_uid_before_capture_start(self):
        peer,api=self.make_api();audio=NativeAudioIO(api)
        self.assertEqual(audio.query_devices(kind="input")["id"],30)
        self.assertEqual(audio.query_devices(kind="output")["id"],10)
        self.assertEqual(audio.query_devices("USB Mic","input")["max_input_channels"],2)
        self.assertFalse(peer.events)
        with audio.InputStream(device="USB Mic"):
            self.assertLess(peer.events.index(("select","usb-마이크")),peer.events.index("enqueue"))
            self.assertLess(peer.events.index(("select","usb-마이크")),peer.events.index("start"))
        self.assertEqual(peer.events[-1],"dispose")
        self.assertEqual(set(peer.released),set(peer.strings))

    def test_selected_device_failure_disposes_queue_without_start_or_default_fallback(self):
        peer,api=self.make_api();peer.fail_selection=True
        with self.assertRaisesRegex(AudioError,"select audio device"):
            NativeAudioIO(api).InputStream(device="USB Mic").start()
        self.assertNotIn("start",peer.events)
        self.assertNotIn("allocate",peer.events)
        self.assertEqual(peer.events[-1],"dispose")
