"""Original bundled voice/TTS consumers over a controlled native-device boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone.tests.test_entrypoints import BundledSourceTestCase, support


_WORKER = r'''
import contextlib,ctypes,io,json,os,pathlib,queue,struct,sys,threading,time,wave
context=json.loads(sys.argv[1]);scratch=pathlib.Path(sys.argv[2]);source=pathlib.Path(context['root'])/'app'
sys.path[:0]=[str(source),str(source.parent/'vendor')]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
def guard(event,args):
    if event.startswith('socket.') and event!='socket.__new__':
        raise AssertionError('Audio fixture cannot use network services')
    if event in ('subprocess.Popen','os.system'):
        raise AssertionError('Audio fixture cannot start external players or installers')
sys.addaudithook(guard)
from agent.sdk_boundary import bundled
from tools import voice_mode as voice,tts_tool as tts
from standalone import audio_samples as pcm
from standalone.audio_device import AudioError,AudioOverflowError,NativeAudioIO
audio,np=voice._import_audio()
assert bundled() and isinstance(audio,NativeAudioIO) and np is None
assert tts._import_sounddevice() is audio
assert 'sounddevice' not in sys.modules and 'numpy' not in sys.modules

class Device:
    def __init__(self,owner,direction,rate,channels,blocksize,stop):
        self.owner,self.direction,self.rate,self.channels,self.blocksize,self.stop=owner,direction,rate,channels,blocksize,stop
        self.blocks=queue.Queue(maxsize=32);self.writes=[];self.written=threading.Event();self.closed=threading.Event();self.drains=0
    def read(self,frames):
        while not self.stop.is_set():
            try:
                data=self.blocks.get(timeout=.02)
                if isinstance(data,Exception):raise data
                assert len(data)==frames*self.channels*2
                return data
            except queue.Empty:pass
        raise InterruptedError('fixture capture cancelled')
    def write(self,data):
        self.writes.append(bytes(data));self.written.set()
        while self.owner.hold and not self.stop.wait(.005):pass
        if self.stop.is_set():raise InterruptedError('fixture playback cancelled')
        if self.owner.failures:
            self.owner.failures-=1
            raise AudioError('fixture output failed')
    def drain(self):self.drains+=1
    def close(self):self.closed.set()
class Provider:
    def __init__(self):self.devices=[];self.scripts=[];self.hold=False;self.failures=0
    def query_devices(self):return [{'id':'private','name':'Fixture PCM','directions':['input','output'],'default_samplerate':16000}]
    def open(self,direction,rate,channels,blocksize,stop):
        result=Device(self,direction,rate,channels,blocksize,stop);self.devices.append(result)
        if direction=='input' and self.scripts:
            for raw in self.scripts.pop(0):result.blocks.put(raw)
        return result
provider=Provider();audio.provider=provider
def wait_for(predicate):
    deadline=time.monotonic()+3
    while not predicate():
        assert time.monotonic()<deadline,'Timed out waiting for original audio consumer'
        threading.Event().wait(.005)
def read_wav(path):
    with wave.open(path,'rb') as data:
        return data.getframerate(),data.getnchannels(),data.readframes(data.getnframes())
def outputs():return [device for device in provider.devices if device.direction=='output']
recorder=voice.AudioRecorder();recorder.start()
capture=provider.devices[-1];raw=pcm.from_samples([5000,-5000]*400).tobytes()
for _ in range(8):capture.blocks.put(raw)
wait_for(lambda:len(recorder._frames)==8)
assert recorder.current_rms==5000
path=recorder.stop();assert read_wav(path)==(16000,1,raw*8)
assert recorder._stream.active
recorder.start();capture.blocks.put(raw);wait_for(lambda:bool(recorder._frames));recorder.cancel()
assert not recorder.is_recording and not recorder._frames and recorder._stream.active
recorder.start();recorder._capture_bytes=pcm.MAX_BYTES-10;capture.blocks.put(raw)
wait_for(lambda:recorder._capture_error is not None)
try:recorder.stop()
except RuntimeError as error:assert '64 MiB' in str(error)
else:raise AssertionError('Capture limit must fail truthfully')
recorder._stream.stop();recorder.start();assert provider.devices[-1] is not capture
new_input=provider.devices[-1];new_input.blocks.put(AudioOverflowError('fixture input overrun'))
wait_for(lambda:recorder._capture_error is not None)
try:recorder.stop()
except RuntimeError as error:assert 'input overrun' in str(error)
else:raise AssertionError('Native capture errors must reach the original recorder')
recorder.shutdown();assert new_input.closed.is_set()

voice.play_beep(440,.025,2)
assert b''.join(outputs()[-1].writes)==pcm.beep(440,.025,2,volume=voice._get_beep_volume()).tobytes()
stereo=pcm.from_samples([1000,-1000]*1600,2);wav=scratch/'stereo.wav'
with wave.open(str(wav),'wb') as data:
    data.setnchannels(2);data.setsampwidth(2);data.setframerate(16000);data.writeframes(stereo.tobytes())
assert voice._play_wav_via_sounddevice(str(wav))
assert outputs()[-1].channels==2 and b''.join(outputs()[-1].writes)==stereo.tobytes()
with wave.open(str(scratch/'bad-width.wav'),'wb') as data:
    data.setnchannels(1);data.setsampwidth(1);data.setframerate(16000);data.writeframes(b'\x80'*100)
assert not voice._play_wav_via_sounddevice(str(scratch/'bad-width.wav'))
assert len(voice._synth_thinking_blip(None,392.0))==2560
assert abs(voice._percentile(None,[1,2,3,4,100],90)-61.6)<1e-10
quiet=pcm.silence(480).tobytes();loud=pcm.from_samples([6000]*480).tobytes()
provider.scripts.append([quiet]*3+[loud]*5+[quiet]*3)
triggered=[]
barge=voice.full_duplex_listen(lambda:False,is_playing=lambda:False,on_trigger=triggered.append,
    calibration_ms=60,sustained_ms=60,pre_roll_ms=120,endpoint_silence_ms=60,max_utterance_ms=600)
assert barge and read_wav(barge)[0]==16000 and triggered==['generation']
provider.scripts.append([quiet]*3+[loud]*4)
assert voice.listen_for_speech(lambda:False,threshold=500,calibration_ms=60,sustained_ms=60)
try:voice.full_duplex_listen(lambda:True,pre_roll_ms=60001)
except ValueError:pass
else:raise AssertionError('Unbounded pre-roll was accepted')

from tools.tts_tool_speaker import _StreamerPlayback,_align_int16_chunks,_SentenceChunks
assert list(_align_int16_chunks([b'\x01',b'\x02\x03'],threading.Event()))==[b'\x01\x02',b'\x03\0']
class Streamer:
    sample_rate=24000;channels=1
    def stream(self,text):
        self.sample_rate=44100
        yield b'\x01';yield b'\x02\x03';yield b'\x04'
player=_StreamerPlayback(Streamer(),threading.Event())
player.speak('One sentence.');player.speak('A second sentence.');player.finish()
assert outputs()[-1].rate==44100 and b''.join(outputs()[-1].writes)==b'\x01\x02\x03\x04'*2
assert outputs()[-1].closed.is_set() and outputs()[-1].drains==1
assert not voice.is_audio_output_active()
provider.failures=1
player=_StreamerPlayback(Streamer(),threading.Event());player.speak('Recover output.');player.finish()
assert player._reinit_count==1 and outputs()[-1].closed.is_set()
class LargeStreamer:
    sample_rate=16000;channels=1
    def stream(self,text):
        for _ in range(200):yield b'\0'*65536
provider.hold=True;stop=threading.Event();player=_StreamerPlayback(LargeStreamer(),stop)
player.speak('Cancel while output is full.');wait_for(lambda:outputs()[-1].written.is_set())
assert player._audio_queue.maxsize==3
stop.set();player.finish();provider.hold=False
assert outputs()[-1].closed.is_set() and not player._worker.is_alive()
for thread in player._prefetch_threads:assert not thread.is_alive()
assert not voice.is_audio_output_active()
q=_SentenceChunks(1,threading.Event());q.put(b'\0\0');q.complete.set()
assert q.get()==b'\0\0' and q.get() is None and q.consumed
assert 'numpy' not in sys.modules and 'sounddevice' not in sys.modules
assert all(device.closed.is_set() for device in provider.devices)
print(json.dumps({'recorder':True,'wav':True,'barge_in':True,'tts':True,'device_count':len(provider.devices)}))
'''


class OriginalAudioConsumerTests(BundledSourceTestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "Original Linux native PCM consumer policy")
    def test_original_recorder_playback_barge_in_and_tts(self):
        with tempfile.TemporaryDirectory(prefix="audio-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            profile = scratch / "home" / "hermes"
            profile.mkdir()
            (profile / "config.yaml").write_text(json.dumps({"stt": {"enabled": False}, "voice": {"beep_volume": .3}}))
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            process = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", _WORKER,
                 json.dumps(self.context), str(scratch)],
                env=environment, cwd=scratch / "work", capture_output=True, text=True, timeout=40,
            )
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            report = json.loads(process.stdout)
            self.assertTrue(all(report[name] for name in ("recorder", "wav", "barge_in", "tts")))


if __name__ == "__main__":
    unittest.main()
