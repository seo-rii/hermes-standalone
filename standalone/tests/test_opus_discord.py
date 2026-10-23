"""Original discord.py uses the qualified codec without loading native Opus."""
import json
from pathlib import Path
import sys
import unittest

from standalone.tests import test_platform_sdks as platform_fixture


FIXTURE=Path(__file__).resolve().with_name('fixtures')/'opus_discord.json'


PROGRAM=r'''
import base64,hashlib,struct,traceback,zlib
import discord
from discord.errors import DiscordException
wrapper=json.loads(pathlib.Path(OPUS_FIXTURE).read_text())
payload=zlib.decompress(base64.b64decode(wrapper['data_zlib_base64']))
assert hashlib.sha256(payload).hexdigest()==wrapper['data_sha256']
data=json.loads(payload);report={}
try:
    assert discord.opus.is_loaded() and discord.opus._lib is None
    encoder=discord.opus.Encoder();other=discord.opus.Encoder(fec=False)
    assert encoder.fec_requested and not encoder.fec_enabled and encoder.codec_mode=='celt'
    assert not other.fec_requested and not other.fec_enabled
    decoder=discord.opus.Decoder()
    for frame in data['frames']:
        pcm=base64.b64decode(frame['pcm16le']);expected=base64.b64decode(frame['packet'])
        packet=encoder.encode(pcm,960)
        assert packet==expected==other.encode(pcm,960)
        assert packet[0]&128,'CELT policy silently changed modes'
        actual=decoder.decode(packet);golden=base64.b64decode(frame['decoded_pcm16le'])
        values=struct.unpack('<1920h',actual);reference=struct.unpack('<1920h',golden)
        assert max(abs(a-b) for a,b in zip(values,reference))<=2
        assert decoder.final_range==frame['final_range'] and decoder._get_last_packet_duration()==960
    # Native forced-CELT ignores this SILK-only request too; these packets
    # come from independent C encoders with FEC=0 and FEC=1, not this adapter.
    for case in data['native_forced_celt_fec_pairs']:
        first=discord.opus.Decoder();second=discord.opus.Decoder()
        assert case['fec_off']==case['fec_on']
        packets=list(map(base64.b64decode,case['fec_on']))
        assert all(packet[0]&128 for packet in packets)
        first.decode(packets[0]);second.decode(packets[0])
        assert first.decode(packets[1],fec=True)==second.decode(None)
        assert first.decode(packets[1])==second.decode(packets[1])
    for construct in (lambda:discord.opus.Encoder(application='invalid'),
                      lambda:discord.opus.Encoder(bitrate=15),
                      lambda:discord.opus.Encoder(expected_packet_loss=0)):
        try:construct()
        except ValueError:pass
        else:raise AssertionError('Invalid original Encoder argument accepted')
    try:discord.opus.Decoder().decode(b'\xff')
    except discord.opus.OpusError as error:assert error.code==-4 and isinstance(error,DiscordException)
    else:raise AssertionError('Invalid packet did not use the original SDK error contract')
    try:encoder.encode(b'bad',960)
    except discord.opus.OpusError as error:assert error.code==-1
    else:raise AssertionError('Invalid PCM did not use the original SDK error contract')
    for loader in (discord.opus.load_opus,discord.opus.libopus_loader):
        try:loader('opus')
        except discord.opus.OpusNotLoaded:pass
        else:raise AssertionError('External native Opus loader enabled')
    assert not any(name in sys.modules for name in ('nacl','davey','_sodium'))
    report.update(original_sdk=True,actual_pcm=True,native_interoperability=True,
                  error_contract=True,native_load_refused=True,fec_requested=encoder.fec_requested,
                  send_fec=encoder.fec_enabled,celt_only=True,packets=len(data['frames']))
except BaseException:report['error']=traceback.format_exc()
report['blocked_events']=guard.events
print(json.dumps(report),flush=True)
os._exit(0)
'''


@unittest.skipUnless(sys.platform.startswith('linux') and sys.version_info>=(3,14),
                     'Original Discord SDK consumer fixture uses Linux/Python 3.14')
class OriginalOpusDiscordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        platform_fixture.OriginalPlatformSDKTests.setUpClass.__func__(cls)
        dest=cls.bundle/'app/standalone/test-fixtures';dest.mkdir(exist_ok=True)
        cls.fixture=dest/'opus_discord.json';cls.fixture.write_bytes(FIXTURE.read_bytes())

    def test_original_encoder_decoder_pcm_native_vectors_and_effective_fec_policy(self):
        report,_=platform_fixture.OriginalPlatformSDKTests.invoke(self,PROGRAM.replace('OPUS_FIXTURE',repr(str(self.fixture))))
        self.assertTrue(all(report[name] for name in ('original_sdk','actual_pcm','native_interoperability',
            'error_contract','native_load_refused','fec_requested','celt_only')))
        self.assertFalse(report['send_fec'])
        self.assertEqual(report['packets'],30)


if __name__=='__main__':
    unittest.main()
