"""Original Discord voice gateway over TLS WebSocket and nested native RTP."""

import json
from pathlib import Path
import sys
import unittest

from standalone.tests import test_platform_sdks as platform_fixture


FIXTURES = Path(__file__).resolve().with_name("fixtures") / "dave"
OPUS_FIXTURE_INPUT = FIXTURES.parent/'opus_discord.json'


PROGRAM = r'''
import asyncio,ssl,socket,struct,threading,traceback
from types import SimpleNamespace
from unittest.mock import patch
import aiohttp,discord
from discord.gateway import DiscordVoiceWebSocket
from discord.voice_state import ConnectionFlowState
from websockets.asyncio.server import serve
from standalone import dave,mls_types
from standalone.mls_crypto import public_key
from standalone.mls_codec import opaque
from plugins.platforms.discord.adapter import VoiceReceiver
case=json.loads(pathlib.Path(DAVE_FIXTURE).read_text())['native_session']
media=case['media'];report={}
async def exercise():
    loop=asyncio.get_running_loop()
    peer_future=loop.create_future();outgoing=asyncio.Queue()
    async def accept(peer):
        peer_future.set_result(peer)
        try:
            async for raw in peer:await outgoing.put(raw)
        except Exception:pass
    tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(scratch/'cert.pem',scratch/'key.pem')
    client=discord.Client(intents=discord.Intents.none())
    await client._async_setup_hook()
    client._connection.user=SimpleNamespace(id=123456789012345678)
    channel=SimpleNamespace(id=918273645012345678,guild=SimpleNamespace(id=42),members=[])
    voice=discord.VoiceClient(client,channel)
    state=voice._connection
    state.ssrc=0x11223344;state.server_id=42;state.session_id='fixture-session';state.token='fixture-voice-token'
    rx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);rx.bind(('127.0.0.1',0));rx.setblocking(False)
    tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);tx.bind(('127.0.0.1',0));tx.connect(rx.getsockname())
    state.socket=rx
    receiver=None
    try:
        async with serve(accept,'127.0.0.1',0,ssl=tls) as server:
            port=server.sockets[0].getsockname()[1]
            async with aiohttp.ClientSession() as http:
                socket_ws=await http.ws_connect('wss://127.0.0.1:'+str(port),ssl=ssl.create_default_context())
                gateway=DiscordVoiceWebSocket(socket_ws,loop);gateway._connection=state;state.ws=gateway
                peer=await asyncio.wait_for(peer_future,5)
                await gateway.identify()
                identify=json.loads(await asyncio.wait_for(outgoing.get(),5))
                assert identify['op']==gateway.IDENTIFY
                assert identify['d']==dict(server_id='42',user_id='123456789012345678',session_id='fixture-session',token='fixture-voice-token',max_dave_protocol_version=1)
                await peer.send(json.dumps(dict(op=gateway.SESSION_DESCRIPTION,d=dict(mode='aead_xchacha20_poly1305_rtpsize',secret_key=list(bytes.fromhex(media['key'])),dave_protocol_version=1))))
                signer=bytes.fromhex(case['signer'])
                pairs=iter((bytes.fromhex(case['init_private']),bytes.fromhex(case['leaf_private'])))
                def fixed_key():
                    private=next(pairs);return private,public_key(private)
                with patch.object(dave,'generate_p256_keypair',return_value=dave.SigningKeyPair(signer,public_key(signer))),patch.object(mls_types,'generate_key',side_effect=fixed_key):
                    await gateway.poll_event()
                speaking=json.loads(await asyncio.wait_for(outgoing.get(),5))
                package=await asyncio.wait_for(outgoing.get(),5)
                assert speaking['op']==gateway.SPEAKING and package==bytes([gateway.MLS_KEY_PACKAGE])+bytes.fromhex(case['package'])
                assert state.dave_protocol_version==1 and not state.can_encrypt and not state.dave_session.ready
                try:voice._get_voice_packet(bytes.fromhex(media['plaintext']))
                except RuntimeError as error:assert 'not ready' in str(error)
                else:raise AssertionError('Negotiated DAVE sent plaintext before MLS readiness')
                await peer.send(struct.pack('>HB',1,gateway.MLS_EXTERNAL_SENDER)+bytes.fromhex(case['external_sender']))
                await gateway.poll_event()
                await peer.send(struct.pack('>HBH',2,gateway.MLS_WELCOME,9)+bytes.fromhex(case['initial_welcome']))
                await gateway.poll_event()
                ready=json.loads(await asyncio.wait_for(outgoing.get(),5))
                assert ready==dict(op=gateway.DAVE_TRANSITION_READY,d=dict(transition_id=9))
                assert state.can_encrypt and state.dave_session.get_epoch_authenticator().hex()==case['initial_authenticator']
                await peer.send(json.dumps(dict(op=gateway.DAVE_EXECUTE_TRANSITION,d=dict(transition_id=9))))
                await gateway.poll_event();assert not state.dave_pending_transitions
                # Actual original voice packet producer: inner DAVE AES-GCM and
                # outer XChaCha RTP both equal independent native outputs.
                voice.sequence=7;voice.timestamp=7*960;voice._incr_nonce=media['counter']
                assert voice._get_voice_packet(bytes.fromhex(media['plaintext'])).hex()==media['b_rtp']
                state.state=ConnectionFlowState.connected
                receiver=VoiceReceiver(voice,{'111111111111111111'});receiver.start()
                state._socket_reader.resume()
                await peer.send(json.dumps(dict(op=gateway.SPEAKING,d=dict(ssrc=0x55667788,user_id='111111111111111111'))))
                await gateway.poll_event()
                assert receiver._ssrc_to_user[0x55667788]==111111111111111111
                async def wait_decoded(count):
                    while receiver._decode_ok<count:await asyncio.sleep(.005)
                async def require_decoded(count):
                    await asyncio.wait_for(wait_decoded(count),5)
                def near_pcm(actual,expected):
                    assert len(actual)==len(expected)
                    left=struct.unpack('<'+'h'*(len(actual)//2),actual)
                    right=struct.unpack('<'+'h'*(len(expected)//2),expected)
                    assert max(abs(a-b) for a,b in zip(left,right))<=2
                # The actual original UDP reader reaches the qualified Python decoder.
                tx.send(bytes.fromhex(media['a_rtp']));await require_decoded(1)
                near_pcm(bytes(receiver._buffers[0x55667788]),bytes.fromhex(media['native_pcm']))
                for name in ('a_rtp','bad_dave_rtp','plaintext_rtp'):
                    receiver._on_packet(bytes.fromhex(media[name]))
                corrupt=bytearray.fromhex(media['a_rtp']);corrupt[15]^=1;receiver._on_packet(corrupt)
                assert receiver._decode_ok==1 and len(receiver._buffers[0x55667788])==3840
                # Only an authenticated server transition permits plaintext frames.
                await peer.send(json.dumps(dict(op=gateway.DAVE_PREPARE_TRANSITION,d=dict(transition_id=12,protocol_version=0))))
                await gateway.poll_event()
                assert json.loads(await asyncio.wait_for(outgoing.get(),5))==dict(op=gateway.DAVE_TRANSITION_READY,d=dict(transition_id=12))
                receiver._on_packet(bytes.fromhex(media['plaintext_rtp']))
                assert receiver._decode_ok==2 and len(receiver._buffers[0x55667788])==7680
                await peer.send(json.dumps(dict(op=gateway.DAVE_EXECUTE_TRANSITION,d=dict(transition_id=12))))
                await gateway.poll_event()
                assert state.dave_protocol_version==0 and state.dave_downgraded and not state.can_encrypt
                await peer.send(json.dumps(dict(op=gateway.DAVE_PREPARE_TRANSITION,d=dict(transition_id=13,protocol_version=1))))
                await gateway.poll_event()
                assert json.loads(await asyncio.wait_for(outgoing.get(),5))==dict(op=gateway.DAVE_TRANSITION_READY,d=dict(transition_id=13))
                await peer.send(json.dumps(dict(op=gateway.DAVE_EXECUTE_TRANSITION,d=dict(transition_id=13))))
                await gateway.poll_event()
                assert state.dave_protocol_version==1 and not state.dave_downgraded and state.can_encrypt
                # Original mixer -> AudioPlayer -> Encoder -> VoiceClient -> DAVE/RTP/UDP.
                import base64,hashlib,wave,zlib
                from plugins.platforms.discord import voice_mixer
                from standalone.dave_frames import FrameEncryptor,FrameDecryptor
                from standalone.discord_crypto import Aead
                fixture=json.loads(pathlib.Path(OPUS_FIXTURE).read_text())
                raw=zlib.decompress(base64.b64decode(fixture['data_zlib_base64']))
                assert hashlib.sha256(raw).hexdigest()==fixture['data_sha256']
                frames=json.loads(raw)['frames'];pcm=b''.join(base64.b64decode(frame['pcm16le']) for frame in frames)
                playback=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
                playback.bind(('127.0.0.1',0));playback.setblocking(False)
                state.ip,state.port=playback.getsockname()
                playback.connect(rx.getsockname());rx.connect(playback.getsockname())
                remote_decryptor=FrameDecryptor(bytes.fromhex(case['initial_exports']['123456789012345678']))
                remote_encryptor=FrameEncryptor(bytes.fromhex(case['initial_exports']['111111111111111111']))
                remote_encryptor.truncated_nonce=100
                remote_client=object.__new__(discord.VoiceClient)
                remote_client._connection=SimpleNamespace(dave_session=None,can_encrypt=False,ssrc=0x66778899,
                    mode='aead_xchacha20_poly1305_rtpsize',secret_key=state.secret_key)
                remote_client.sequence=0;remote_client.timestamp=0;remote_client._incr_nonce=9000
                await peer.send(json.dumps(dict(op=gateway.SPEAKING,d=dict(ssrc=0x66778899,user_id='111111111111111111'))))
                await gateway.poll_event()
                peer_decoder=discord.opus.Decoder();mixer=voice_mixer.VoiceMixer()
                mixer.play_speech(pcm,fade_in_ms=0)
                try:
                    voice.play(mixer)
                    assert voice.encoder.fec_requested and not voice.encoder.fec_enabled
                    for number,frame in enumerate(frames):
                        wire=await asyncio.wait_for(loop.sock_recv(playback,65536),5)
                        assert struct.unpack_from('>HII',wire,2)==(8+number,(7+number)*960,0x11223344)
                        nonce=wire[-4:]+bytes(20)
                        encrypted=Aead(bytes(state.secret_key)).decrypt(wire[12:-4],wire[:12],nonce)
                        packet=remote_decryptor.decrypt(encrypted)
                        assert packet==base64.b64decode(frame['packet'])
                        near_pcm(peer_decoder.decode(packet),base64.b64decode(frame['decoded_pcm16le']))
                        remote_client.sequence=100+number;remote_client.timestamp=number*960
                        await loop.sock_sendall(playback,remote_client._get_voice_packet(remote_encryptor.encrypt(packet)))
                    player=voice._player;voice.stop()
                    await asyncio.to_thread(player.join,3)
                    assert not player.is_alive()
                    await require_decoded(32)
                    utterances=receiver.flush_pending()
                    assert len(utterances)==1 and utterances[0][0]==111111111111111111
                    native_pcm=b''.join(base64.b64decode(frame['decoded_pcm16le']) for frame in frames)
                    near_pcm(utterances[0][1],native_pcm)
                    assert receiver.flush_pending()==[]
                    recording=scratch/'work/recorded.wav';receiver.pcm_to_wav(utterances[0][1],str(recording))
                    with wave.open(str(recording),'rb') as source:
                        assert (source.getframerate(),source.getnchannels(),source.getnframes())==(16000,1,9600)
                        assert any(source.readframes(9600))
                    replay_pcm=voice_mixer.decode_to_pcm(str(recording));assert len(replay_pcm)==len(pcm) and any(replay_pcm)
                    replay=voice_mixer.VoiceMixer();replay.play_speech(replay_pcm,fade_in_ms=0)
                    voice.send_audio_packet(replay.read())
                    while True:
                        wire=await asyncio.wait_for(loop.sock_recv(playback,65536),5)
                        encrypted=Aead(bytes(state.secret_key)).decrypt(wire[12:-4],wire[:12],wire[-4:]+bytes(20))
                        if encrypted==discord.opus.OPUS_SILENCE:continue
                        packet=remote_decryptor.decrypt(encrypted)
                        assert len(peer_decoder.decode(packet))==3840
                        break
                    replay.cleanup()
                finally:
                    voice.stop();rx.connect(tx.getsockname());playback.close()
                for _ in range(2):
                    speaking_event=json.loads(await asyncio.wait_for(outgoing.get(),5))
                    assert speaking_event['op']==gateway.SPEAKING
                assert discord.opus.is_loaded() and discord.opus._lib is None
                report.update(actual_pcm_receiver=True,original_mixer_player=True,actual_pcm_sender=True,
                    native_codec_interop=True,original_recording=True,original_recording_playback=True,
                    effective_send_fec=False,codec_packets=len(frames))
                # The original binary parser emits CommitWelcome from a
                # source-authenticated external Add, then merges its own echo.
                epoch=case['epochs'][0]
                blocks=iter(bytes.fromhex(value) for value in epoch['random_blocks'])
                await peer.send(struct.pack('>HBB',3,gateway.MLS_PROPOSALS,0)+opaque(bytes.fromhex(epoch['proposals'][0])))
                with patch('os.urandom',side_effect=lambda size:next(blocks)):
                    await gateway.poll_event()
                cw=await asyncio.wait_for(outgoing.get(),5)
                assert cw==bytes([gateway.MLS_COMMIT_WELCOME])+bytes.fromhex(epoch['commit']+epoch['welcome'])
                await peer.send(struct.pack('>HBH',4,gateway.MLS_ANNOUNCE_COMMIT_TRANSITION,10)+bytes.fromhex(epoch['commit']))
                await gateway.poll_event()
                assert json.loads(await asyncio.wait_for(outgoing.get(),5))==dict(op=gateway.DAVE_TRANSITION_READY,d=dict(transition_id=10))
                assert state.dave_session.epoch==2 and state.dave_session.get_epoch_authenticator().hex()==epoch['authenticator']
                # Corrupt Commit triggers the SDK's actual invalid-message
                # recovery and a fresh KeyPackage; it never reports readiness.
                malformed=bytearray.fromhex(case['epochs'][1]['commit']);malformed[-1]^=1
                await peer.send(struct.pack('>HBH',5,gateway.MLS_ANNOUNCE_COMMIT_TRANSITION,11)+malformed)
                await gateway.poll_event()
                invalid=json.loads(await asyncio.wait_for(outgoing.get(),5));fresh=await asyncio.wait_for(outgoing.get(),5)
                assert invalid==dict(op=gateway.MLS_INVALID_COMMIT_WELCOME,d=dict(transition_id=11))
                assert fresh[0]==gateway.MLS_KEY_PACKAGE and fresh!=package and not state.can_encrypt
                assert not state.dave_session.ready and state.dave_protocol_version==1
                assert not any(name in sys.modules for name in ('davey','nacl','_sodium','Crypto','cryptography'))
                report.update(original_voice_client=True,original_tls_websocket=True,negotiated_mls_v1=True,
                    native_welcome=True,native_nested_rtp=True,actual_udp_receiver=True,ssrc_identity=True,
                    dave_auth_replay_refusals=True,plaintext_before_ready_refused=True,original_proposal_commit=True,
                    invalid_commit_recovery=True,negotiated_passthrough=True,dave_available=True,codec_available=True)
                await gateway.close()
    finally:
        if receiver:receiver.stop()
        state._socket_reader.stop();tx.send(b'');state._socket_reader.join(3)
        tx.close();rx.close();await client.close()
try:asyncio.run(exercise())
except BaseException:report['error']=traceback.format_exc()
report['blocked_events']=guard.events
print(json.dumps(report),flush=True)
os._exit(0)
'''


@unittest.skipUnless(sys.platform.startswith("linux") and sys.version_info >= (3, 14), "Original voice gateway fixture uses Linux/Python 3.14")
class DaveVoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        platform_fixture.OriginalPlatformSDKTests.setUpClass.__func__(cls)
        dest=cls.bundle/'app/standalone/test-fixtures';dest.mkdir(exist_ok=True)
        cls.fixture=dest/'dave-mls.json'
        cls.fixture.write_bytes((FIXTURES/'dave-mls.json').read_bytes())
        cls.opus_fixture=dest/'opus_discord.json'
        cls.opus_fixture.write_bytes(OPUS_FIXTURE_INPUT.read_bytes())

    def test_original_tls_gateway_negotiates_mls_uses_nested_native_udp_and_recovers_invalid_commits(self):
        program=PROGRAM.replace('DAVE_FIXTURE',repr(str(self.fixture))).replace('OPUS_FIXTURE',repr(str(self.opus_fixture)))
        report,_=platform_fixture.OriginalPlatformSDKTests.invoke(self,program)
        self.assertTrue(all(report[name] for name in ('original_voice_client','original_tls_websocket',
            'negotiated_mls_v1','native_welcome','native_nested_rtp','actual_udp_receiver','ssrc_identity',
            'dave_auth_replay_refusals','plaintext_before_ready_refused','original_proposal_commit','invalid_commit_recovery','negotiated_passthrough','dave_available')))
        self.assertTrue(report['codec_available'])
        self.assertTrue(all(report[name] for name in ('actual_pcm_receiver','original_mixer_player',
            'actual_pcm_sender','native_codec_interop','original_recording','original_recording_playback')))
        self.assertFalse(report['effective_send_fec'])
        self.assertEqual(report['codec_packets'],30)


if __name__ == '__main__':
    unittest.main()
