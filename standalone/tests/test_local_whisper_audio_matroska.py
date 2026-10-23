"""Independent native Matroska packets, codec profiles and actual file frontend."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone.agent_probe_worker import fixture_environment


WORKER = r'''
import base64,hashlib,importlib.abc,io,json,pathlib,struct,sys,zlib
root,scratch,mode=sys.argv[1:];root=pathlib.Path(root);scratch=pathlib.Path(scratch)
sys.path.insert(0,str(root))
from standalone.agent_probe_worker import ProbeGuard
guard=ProbeGuard(root,root/'vendor',scratch);sys.addaudithook(guard)
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]in {'av','numpy','ctypes','cffi','subprocess','onnxruntime','torch','soundfile','audioread','pydub'}:
            raise AssertionError('Matroska runtime requested native or process dependency '+fullname)
sys.meta_path.insert(0,NoNative())
def offline(event,args):
    if event.startswith('socket.')or event in {'subprocess.Popen','os.system','ctypes.dlopen','ctypes.dlsym'}:
        raise AssertionError('Matroska runtime requested external execution '+event)
sys.addaudithook(offline)
from standalone.local_whisper_audio import decode_audio
from standalone.local_whisper_audio_matroska import decode_matroska
from standalone.local_whisper_audio_matroska_demux import demux
from standalone.local_whisper_audio_matroska_lzo import decode_lzo
from standalone.local_whisper_audio_resample import convert_audio
fixtures=root/'standalone/tests/fixtures'
review=json.loads((root/'standalone/local-whisper-matroska-provenance.json').read_bytes())

def fixture(name):
    path=fixtures/('local_whisper_matroska_'+name+'.json.zlib')
    raw=path.read_bytes();expected=review['fixtures'][path.name]
    assert len(raw)==expected['bytes']and hashlib.sha256(raw).hexdigest()==expected['sha256']
    return json.loads(zlib.decompress(raw))

def unpack(text,kind):
    return [row[0]for row in struct.iter_unpack('<'+kind,base64.b64decode(text,validate=True))]

def difference(rows,expected):
    assert[len(row)for row in rows]==[len(row)for row in expected]
    return max((abs(a-b)for row,wanted in zip(rows,expected,strict=True)for a,b in zip(row,wanted,strict=True)),default=0)

def packets_and_lzo():
    total,refused=0,0
    for group in ('remux','variants','lzo_containers'):
        source=fixture(group)
        for case in source['cases']:
            data=base64.b64decode(case['data'],validate=True)
            assert hashlib.sha256(data).hexdigest()==case['sha256']
            track,packets,scale=demux(data)
            assert track.private==base64.b64decode(case['stream']['extradata'])
            assert[packet.data for packet in packets]==[base64.b64decode(row['data'])for row in case['packets']]
            total+=len(packets)
        for case in source['errors']:
            data=base64.b64decode(case['data'],validate=True)
            assert hashlib.sha256(data).hexdigest()==case['sha256']
            try:decode_matroska(data)
            except ValueError:refused+=1
            else:raise AssertionError('Original unsupported compressed private header was accepted')
    original=fixture('lzo')['cases']
    for case in original:
        actual=decode_lzo(base64.b64decode(case['data'],validate=True))
        assert actual==base64.b64decode(case['native']['output'],validate=True)
        assert len(actual)==case['raw_bytes']and hashlib.sha256(actual).hexdigest()==case['raw_sha256']
    return {'packet_count':total,'lzo_cases':len(original),'original_refusals':refused}

def frontend(case,profile):
    data=base64.b64decode(case['data'],validate=True)
    path=scratch/(case['sha256']+'.webm');path.write_bytes(data)
    checked=0
    for target in profile['outputs']:
        expected=[unpack(row,'h')for row in target['codec_reference']]
        for original in (io.BytesIO(data),path):
            actual=decode_audio(original,sampling_rate=target['rate'],split_stereo=target['split'])
            if isinstance(original,io.BytesIO):assert not original.closed
            rows=actual if target['split']else(actual,)
            assert all(row.typecode=='f'for row in rows)
            integers=[[round(value*32768)for value in row]for row in rows]
            assert difference(integers,expected)<=1,(case['name'],target['rate'],target['split'])
            checked+=1
    return checked

def codec_group(group):
    source=fixture(group);reference=fixture(group+'_profiles')
    source_path=fixtures/('local_whisper_matroska_'+group+'.json.zlib')
    assert reference['source_sha256']==hashlib.sha256(source_path.read_bytes()).hexdigest()
    profiles={case['name']:case for case in reference['cases']}
    raw_error,max_pcm,max_generic,max_host,frontends=0,0,0,0,0
    for case in source['cases']:
        data=base64.b64decode(case['data'],validate=True);ref=profiles[case['name']]
        rate,pcm,fmt,mask=decode_matroska(data)
        error=difference(pcm,[unpack(row,'f')for row in ref['raw']])
        assert error<=(1/32768 if ref['codec']=='opus'else 0),(case['name'],error)
        raw_error=max(raw_error,error)
        for target,host in zip(ref['outputs'],case['outputs'],strict=True):
            result=convert_audio(pcm,rate,target['rate'],fmt,mask,target['split'])
            error=difference(result,[unpack(row,'h')for row in target['codec_reference']])
            assert error<=1,(case['name'],target['rate'],target['split'],error)
            max_pcm=max(max_pcm,error)
            max_generic=max(max_generic,difference(result,[unpack(row,'h')for row in target['original_generic']]))
            max_host=max(max_host,difference(result,[unpack(row,'h')for row in host['pcm']]))
        if group=='remux'and case['name']in ('jfk-vorbis-16000-1ch.ogg-webm','speech48k.opus-webm'):
            frontends+=frontend(case,ref)
    return {'cases':len(source['cases']),'outputs':len(source['cases'])*4,'max_codec_reference_lsb':max_pcm,
            'max_raw_float_error':raw_error,'max_original_generic_lsb':max_generic,
            'max_original_host_lsb':max_host,'actual_frontend_calls':frontends}

report=packets_and_lzo()if mode=='packets'else codec_group(mode)
assert not guard.events,guard.events
print(json.dumps({'mode':mode,'result':report,'blocked_events':guard.events}))
'''


class MatroskaTests(unittest.TestCase):
    def run_isolated(self, mode):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix='hermes-matroska-public-') as temporary:
            scratch = Path(temporary)
            for name in ('home', 'work', 'tmp'):
                (scratch / name).mkdir()
            child = subprocess.run(
                [sys.executable, '-I', '-S', '-B', '-X', 'utf8', '-c', WORKER,
                 str(root), str(scratch), mode], cwd=scratch / 'work',
                env=fixture_environment(scratch), capture_output=True,
                text=True, encoding='utf8', timeout=1200,
            )
            self.assertEqual(child.returncode, 0, child.stderr + child.stdout)
            report = json.loads(child.stdout)
            self.assertEqual(report['blocked_events'], [])
            return report['result']

    def test_original_demux_packets_content_encodings_and_lzo(self):
        self.assertEqual(self.run_isolated('packets'), {
            'packet_count': 2138, 'lzo_cases': 93, 'original_refusals': 5,
        })

    def test_real_muxed_opus_vorbis_pcm_and_common_file_frontend(self):
        report = self.run_isolated('remux')
        self.assertEqual((report['cases'], report['outputs'], report['actual_frontend_calls']), (33, 132, 16))
        self.assertLessEqual(report['max_codec_reference_lsb'], 1)

    def test_native_laces_delay_padding_multitrack_and_encoding_boundaries(self):
        report = self.run_isolated('variants')
        self.assertEqual((report['cases'], report['outputs']), (175, 700))
        self.assertLessEqual(report['max_codec_reference_lsb'], 1)

    def test_native_lzo_zlib_header_strip_in_both_content_scopes(self):
        report = self.run_isolated('lzo_containers')
        self.assertEqual((report['cases'], report['outputs']), (30, 120))
        self.assertLessEqual(report['max_codec_reference_lsb'], 1)
