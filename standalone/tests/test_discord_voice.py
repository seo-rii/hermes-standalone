"""Original Discord RTP/recorder boundaries with native ciphertext fixtures.

These checks verify transport and PCM consumers. DAVE MLS and the original
voice gateway and complete codec PCM media path are checked in test_dave_voice.
A deliberate decoder failure here checks that transport never fabricates PCM.
"""

import json
from pathlib import Path
import sys
import unittest

from standalone.tests import test_platform_sdks as platform_fixture


FIXTURES = Path(__file__).with_name("fixtures")


_VOICE = r'''
import asyncio,base64,socket,struct,traceback
from types import SimpleNamespace
from unittest.mock import patch
import discord
from discord.voice_state import VoiceConnectionState
from plugins.platforms.discord.adapter import VoiceReceiver
from standalone import discord_crypto
decode=lambda text:base64.b64decode(text)
data=json.loads(pathlib.Path(VOICE_FIXTURE).read_text())
report={}
def exercise():
    receiver_socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sender_socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    receiver_socket.bind(("127.0.0.1",0));receiver_socket.settimeout(3)
    sender_socket.connect(receiver_socket.getsockname())
    count=0
    recovered=[]
    class CodecBoundary:
        def decode(self,packet):
            recovered.append(packet)
            raise discord.opus.OpusNotLoaded("Injected decoder failure after authenticated transport")
    last=None
    try:
        for case in data["opus"]:
            header,key,payload=(decode(case[name]) for name in ("header","key","packet"))
            for mode,expected in case["transports"].items():
                client=object.__new__(discord.VoiceClient)
                client._connection=SimpleNamespace(dave_session=None,can_encrypt=False,ssrc=0x11223344,
                                                   mode=mode,secret_key=list(key))
                client.sequence=case["index"];client.timestamp=case["index"]*960;client._incr_nonce=case["counter"]
                with patch.object(discord_crypto,"random",return_value=decode(case["suffix"])):
                    packet=client._get_voice_packet(payload)
                assert packet==decode(expected),mode
                sender_socket.send(packet)
                wire,_=receiver_socket.recvfrom(65536)
                assert wire==decode(expected)
                count+=1
                if mode=="aead_xchacha20_poly1305_rtpsize":
                    connection=SimpleNamespace(secret_key=list(key),dave_session=None,dave_protocol_version=0,
                        dave_downgraded=False,dave_pending_transitions={},ssrc=0x55667788)
                    source=SimpleNamespace(_connection=connection,channel=None,user=SimpleNamespace(id=9))
                    receiver=VoiceReceiver(source,{"123"});receiver._resolve_credentials(connection);receiver._running=True
                    with patch.object(discord.opus,"Decoder",CodecBoundary):receiver._on_packet(wire)
                    assert recovered[-1]==payload and receiver._decode_failed==0
                    assert receiver._decode_ok==0 and not receiver._buffers,"Unavailable codec produced fake PCM"
                    last=(receiver,wire,payload,connection)
        receiver,wire,payload,connection=last
        corrupt=bytearray(wire);corrupt[15]^=1
        before=len(recovered)
        with patch.object(discord.opus,"Decoder",CodecBoundary):receiver._on_packet(corrupt)
        assert len(recovered)==before and receiver._decode_failed==1
        # A genuine key rotation is recovered through the original failure
        # streak and credential-generation swap, not a decryption fallback.
        case=data["opus"][0]
        connection.secret_key=list(decode(case["key"]))
        changed=decode(case["transports"]["aead_xchacha20_poly1305_rtpsize"])
        with patch.object(discord.opus,"Decoder",CodecBoundary):
            for _ in range(receiver.REKEY_FAILURE_STREAK):receiver._on_packet(changed)
            assert receiver._secret_key==decode(case["key"])
            receiver._on_packet(changed)
        assert recovered[-1]==decode(case["packet"]) and receiver._nacl_fail_streak==0
        # The recorder's post-codec buffer boundary retains the real user's
        # identity and drains exactly once. PCM originates in native libopus.
        receiver._ssrc_to_user[0x11223344]=123
        pcm=decode(case["decoded_pcm"])*25
        receiver._buffers[0x11223344]=bytearray(pcm)
        assert receiver.flush_pending()==[(123,pcm)]
        assert receiver.flush_pending()==[]
        receiver._buffers[0x11223344]=bytearray(pcm)
        receiver.discard_pending()
        assert not receiver._buffers
        # A qualified DAVE backend remains unready until its MLS handshake.
        # The complete original gateway handshake is checked separately.
        from standalone import dave
        state=object.__new__(VoiceConnectionState)
        state.dave_protocol_version=1;state.dave_session=None
        assert state.max_dave_protocol_version==1 and not state.can_encrypt
        pending=dave.DaveSession(1,9,10)
        try:pending.encrypt_opus(payload)
        except ValueError as error:assert "not ready" in str(error).lower()
        else:raise AssertionError("DAVE encrypted before the authenticated MLS handshake")
        assert not pending.ready and state.dave_session is None
        assert discord.opus.is_loaded() and discord.opus._lib is None
        assert discord.opus.Decoder().decode(discord.opus.OPUS_SILENCE)==bytes(3840)
        try:discord.opus.load_opus("opus")
        except discord.opus.OpusNotLoaded:pass
        else:raise AssertionError("An external native Opus codec was loaded")
        assert not any(name in sys.modules for name in ("nacl","_sodium","davey"))
        report.update(rtp_modes=True,udp_wire=True,receiver_authentication=True,credential_rotation=True,
            record_drain_identity=True,required_dave_readiness=True,native_opus_refused=True,
            native_crypto_packets=count,encoded_payloads=len(recovered),codec_available=True,dave_mls_available=True)
    finally:
        sender_socket.close();receiver_socket.close()
try:exercise()
except BaseException:report["error"]=traceback.format_exc()
report["blocked_events"]=guard.events
print(json.dumps(report),flush=True)
os._exit(0)
'''


_MIXER = r'''
import base64,struct,traceback,wave
from plugins.platforms.discord import voice_mixer
from plugins.platforms.discord.adapter import VoiceReceiver
data=json.loads(pathlib.Path(PCM_FIXTURE).read_text())
report={}
try:
    for case in data["vectors"]:
        mixer=voice_mixer.VoiceMixer()
        mixer.play_speech(base64.b64decode(case["first"]),gain=case["first_gain"],fade_in_ms=0)
        mixer.play_speech(base64.b64decode(case["second"]),gain=case["second_gain"],fade_in_ms=0)
        assert mixer.read()==base64.b64decode(case["expected"])
        assert mixer.read()==voice_mixer.SILENCE_FRAME and not mixer.speech_active
        mixer.cleanup();assert mixer.read()==voice_mixer.SILENCE_FRAME
    path=scratch/"work/voice.wav"
    with wave.open(str(path),"wb") as output:
        output.setparams((1,2,48000,0,"NONE","not compressed"))
        output.writeframes(struct.pack("<h",0)*4800)
    decoded=voice_mixer.decode_to_pcm(str(path))
    assert decoded==bytes(4800*4)
    output=scratch/"work/transcription.wav"
    VoiceReceiver.pcm_to_wav(decoded,str(output))
    with wave.open(str(output),"rb") as recording:
        assert (recording.getframerate(),recording.getnchannels(),recording.getnframes())==(16000,1,1600)
        assert recording.readframes(1600)==bytes(3200)
    assert len(voice_mixer.synth_ambient_pcm(.1))==4800*4
    assert "numpy" not in sys.modules
    report.update(original_mixer=True,native_float32=True,original_wav_decode=True,
                  original_recorder_wav=True,no_numpy=True,no_ffmpeg=True)
except BaseException:report["error"]=traceback.format_exc()
report["blocked_events"]=guard.events
print(json.dumps(report),flush=True)
os._exit(0)
'''


@unittest.skipUnless(sys.platform.startswith("linux") and sys.version_info >= (3, 14),
                     "Original Discord integration is verified on Linux/Python 3.14")
class OriginalDiscordVoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        platform_fixture.OriginalPlatformSDKTests.setUpClass.__func__(cls)
        cls.fixture_inputs = cls.bundle / "app" / "standalone" / "test-fixtures"
        cls.fixture_inputs.mkdir()
        for name in ("discord-voice-crypto.json", "discord-pcm.json"):
            (cls.fixture_inputs / name).write_bytes((FIXTURES / name).read_bytes())

    def test_original_sdk_rtp_and_recorder_keep_authentication_and_required_encryption_gates(self):
        program = _VOICE.replace("VOICE_FIXTURE", repr(str(self.fixture_inputs / "discord-voice-crypto.json")))
        report, _ = platform_fixture.OriginalPlatformSDKTests.invoke(self, program)
        self.assertEqual(report["native_crypto_packets"], 60)
        self.assertTrue(all(report[key] for key in ("rtp_modes", "udp_wire", "receiver_authentication",
            "credential_rotation", "record_drain_identity", "required_dave_readiness", "native_opus_refused")))
        self.assertTrue(report["codec_available"])
        self.assertTrue(report["dave_mls_available"])

    def test_original_pcm_mixer_and_recording_wav_need_no_numpy_or_subprocess(self):
        program = _MIXER.replace("PCM_FIXTURE", repr(str(self.fixture_inputs / "discord-pcm.json")))
        report, _ = platform_fixture.OriginalPlatformSDKTests.invoke(self, program)
        self.assertTrue(all(report[key] for key in ("original_mixer", "native_float32", "original_wav_decode",
            "original_recorder_wav", "no_numpy", "no_ffmpeg")))


if __name__ == "__main__":
    unittest.main()
