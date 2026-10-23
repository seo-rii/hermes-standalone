"""Original transcription and QQ conversion paths use the qualified SILK codec."""

import base64
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

from standalone.tests import test_transport_agent as _agent_fixture


WORKER = r'''
import asyncio,importlib.util,json,logging,pathlib,shutil,sys,wave
logging.basicConfig(level=logging.DEBUG)
source,vendor,scratch=map(pathlib.Path,sys.argv[1:])
sys.path[:0]=[str(source),str(vendor)]
from standalone.runtime import initialize_bundle
initialize_bundle({'root':str(source.parent),'artifact':str(scratch/'fixture.py'),
 'bundle_sha256':source.parent.name,'manifest':{'project':'hermes-standalone'}})
from standalone.agent_probe_worker import ProbeGuard
guard=ProbeGuard(source,vendor,scratch);sys.addaudithook(guard)
from hermes_platform.audio import silk_codec
from pm import available
from tools import transcription_tools as stt
from tools.transcription_audio import _prepare_audio_for_transcription
from gateway.config import PlatformConfig
from plugins.platforms.qqbot.adapter import QQAdapter
assert available('silk') and available('audio-io')
assert stt._HAS_PILK and stt._safe_find_spec('pilk')
assert importlib.util.find_spec('pilk') is None
assert silk_codec().__name__=='standalone.silk'
input=scratch/'input.silk'
output,temporary,error=_prepare_audio_for_transcription(str(input))
assert error is None,(output,temporary,error)
try:
 with wave.open(output,'rb') as wav:
  assert (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())==(1,2,24000)
  assert wav.readframes(wav.getnframes())==(scratch/'expected-24000.pcm').read_bytes()
finally:shutil.rmtree(temporary)
# The adapter constructor and conversion retain their original runtime/config.
adapter=QQAdapter(PlatformConfig(enabled=True,extra={'app_id':'fixture','client_secret':'fixture'}))
qq_output=scratch/'qq.wav'
assert asyncio.run(adapter._convert_silk_to_wav(str(input),str(qq_output)))==str(qq_output)
with wave.open(str(qq_output),'rb') as wav:
 assert (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())==(1,2,16000)
 assert wav.readframes(wav.getnframes())==(scratch/'expected-16000.pcm').read_bytes()
input.write_bytes(b'#!SILK_V3\0\0')
output,temporary,error=_prepare_audio_for_transcription(str(input))
assert output is None and temporary is None and not error['success']
assert not list((scratch/'tmp').glob('hermes-silk-*'))
assert 'pilk' not in sys.modules and 'numpy' not in sys.modules and 'sounddevice' not in sys.modules
assert not guard.events,guard.events
print(json.dumps({'ok':True,'backend':silk_codec().__name__}))
'''


@unittest.skipIf(sys.version_info < (3, 14), "Original Hermes requires Python 3.14")
class SilkConsumerTests(unittest.TestCase):
    setUpClass = classmethod(_agent_fixture.AgentHTTPIntegrationTests.setUpClass.__func__)

    def test_original_transcription_and_qq_paths_produce_native_pcm(self):
        from standalone.agent_probe_worker import fixture_environment

        fixture = json.loads((Path(__file__).parent / "fixtures/silk.json").read_text())
        rows = {row["output_rate"]: row for row in fixture["vectors"]
                if row["input_rate"] == 16000 and row["packet_ms"] == 20 and not row["lost_packet_indices"]}
        with tempfile.TemporaryDirectory(prefix="silk-consumer-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            payload = bytearray(b"\x02#!SILK_V3")
            for value in rows[16000]["packets"]:
                packet = base64.b64decode(value)
                payload.extend(struct.pack("<h", len(packet)))
                payload.extend(packet)
            (scratch / "input.silk").write_bytes(payload)
            for rate in (16000, 24000):
                (scratch / f"expected-{rate}.pcm").write_bytes(
                    zlib.decompress(base64.b64decode(rows[rate]["pcm_zlib_base64"])))
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", WORKER,
                 str(self.source), str(self.vendor), str(scratch)],
                cwd=scratch / "work", env=fixture_environment(scratch),
                capture_output=True, text=True, encoding="utf-8", timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout)["ok"])
