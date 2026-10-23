"""Original trace uploader and Hub SDK over owned HTTP and LFS peers."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone.tests import test_transport_agent as _agent_fixture


WORKER = r'''
import base64,hashlib,json,os,pathlib,sys,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
source,vendor,scratch=map(pathlib.Path,sys.argv[1:])
sys.path[:0]=[str(source),str(vendor)]
from standalone.runtime import initialize_bundle
initialize_bundle({'root':str(source.parent),'artifact':str(scratch/'fixture.py'),
 'bundle_sha256':source.parent.name,'manifest':{'project':'hermes-standalone'}})
records=[]
class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args):pass
 def body(self):
  if self.headers.get('transfer-encoding','').lower()!='chunked':
   n=int(self.headers.get('content-length','0'));assert 0<=n<=4*1024*1024
   return self.rfile.read(n)
  out=bytearray()
  while True:
   n=int(self.rfile.readline(80).split(b';',1)[0],16)
   if not n:
    assert self.rfile.readline(80)==b'\r\n';return bytes(out)
   assert n+len(out)<=4*1024*1024
   out.extend(self.rfile.read(n));assert self.rfile.read(2)==b'\r\n'
 def respond(self,obj,status=200):
  data=json.dumps(obj).encode();self.send_response(status)
  self.send_header('content-type','application/json');self.send_header('content-length',str(len(data)))
  self.end_headers();self.wfile.write(data)
 def do_GET(self):
  records.append((self.command,self.path,dict(self.headers),b''))
  if self.path=='/api/whoami-v2':
   return self.respond({'name':'fixture-user'} if not self.server.reject else {'error':'invalid fixture token'},401 if self.server.reject else 200)
  self.respond({'error':'unhandled fixture endpoint'},404)
 def do_POST(self):
  data=self.body();records.append((self.command,self.path,dict(self.headers),data))
  if self.path=='/api/repos/create':return self.respond({'url':origin+'/datasets/fixture-user/hermes-traces'})
  if '/preupload/' in self.path:
   return self.respond({'files':[{'path':f['path'],'uploadMode':self.server.mode,'shouldIgnore':False} for f in json.loads(data)['files']]})
  if self.path.endswith('/info/lfs/objects/batch'):
   return self.respond({'transfer':'basic','objects':[{'oid':o['oid'],'size':o['size'],'actions':{
    'upload':{'href':origin+'/lfs/upload/'+o['oid']},
    'verify':{'href':origin+'/lfs/verify','header':{'X-Fixture-Verify':'verify-scope'}}}} for o in json.loads(data)['objects']]})
  if self.path=='/lfs/verify':return self.respond({})
  if '/commit/' in self.path:
   return self.respond({'commitUrl':origin+'/datasets/fixture-user/hermes-traces/commit/'+'a'*40,'commitOid':'a'*40})
  self.respond({'error':'unhandled fixture endpoint'},404)
 def do_PUT(self):
  data=self.body();records.append((self.command,self.path,dict(self.headers),data))
  assert self.path=='/lfs/upload/'+hashlib.sha256(data).hexdigest()
  self.respond({})
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.mode='regular';server.reject=False
origin='http://127.0.0.1:'+str(server.server_port)
os.environ['HF_ENDPOINT']=origin
os.environ['HF_HOME']=str(scratch/'home'/'hf')
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from standalone.agent_probe_worker import ProbeGuard
class Guard(ProbeGuard):
 def __call__(self,event,args):
  if event=='socket.connect' and args[1][:2]==('127.0.0.1',server.server_port):return
  if event=='socket.getaddrinfo' and args[0]=='127.0.0.1' and int(args[1])==server.server_port:return
  super().__call__(event,args)
guard=Guard(source,vendor,scratch);sys.addaudithook(guard)
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
 import importlib.util
 from huggingface_hub import constants
 from huggingface_hub.utils._runtime import is_xet_available
 assert constants.HF_HUB_DISABLE_XET and not is_xet_available()
 assert importlib.util.find_spec('hf_xet') is None
 from agent.trace_upload import _do_upload,build_trace_jsonl
 trace=build_trace_jsonl([{'role':'user','content':'token sk-proj-'+'a'*70},
  {'role':'assistant','content':'fixture trace'}],session_id='fixture-session')
 assert 'sk-proj-'+'a'*70 not in trace
 for mode in ('regular','lfs'):
  server.mode=mode;start=len(records)
  result=_do_upload(trace,token='hf_fixture_write_token',session_id='fixture-'+mode)
  assert result.startswith('Uploaded -> '),(result,[(r[0],r[1]) for r in records[start:]])
  entries=records[start:]
  create=next(json.loads(r[3]) for r in entries if r[1]=='/api/repos/create')
  assert create['visibility']=='private' and create['organization']=='fixture-user'
  for method,path,headers,data in entries:
   if path.startswith('/api/') or path.startswith('/datasets/'):
    assert headers.get('authorization')=='Bearer hf_fixture_write_token',headers
  commit=next(r for r in entries if '/commit/' in r[1]);lines=[json.loads(line) for line in commit[3].splitlines()]
  assert lines[0]['key']=='header' and lines[0]['value']['summary']=='add session trace fixture-'+mode
  file=next(line for line in lines if line['key'] in ('file','lfsFile'))
  assert file['value']['path']=='sessions/fixture-'+mode+'.jsonl'
  if mode=='regular':assert base64.b64decode(file['value']['content'])==trace.encode()
  else:
   put=next(r for r in entries if r[0]=='PUT');assert put[3]==trace.encode()
   assert 'authorization' not in {k.lower():v for k,v in put[2].items()}
   assert file['value']['oid']==hashlib.sha256(trace.encode()).hexdigest()
 server.reject=True;start=len(records)
 result=_do_upload(trace,token='hf_fixture_rejected',session_id='rejected')
 assert 'token was rejected' in result and len(records)==start+1
 assert not guard.events,guard.events
 print(json.dumps({'ok':True,'requests':len(records)}))
finally:
 from huggingface_hub.utils import close_session
 close_session();server.shutdown();server.server_close();thread.join(timeout=2)
'''


@unittest.skipIf(sys.version_info < (3, 14), "Original Hermes uploader requires Python 3.14")
class TraceUploadTests(unittest.TestCase):
    setUpClass = classmethod(_agent_fixture.AgentHTTPIntegrationTests.setUpClass.__func__)

    def test_original_trace_upload_uses_sdk_http_and_lfs_without_native_xet(self):
        from standalone.agent_probe_worker import fixture_environment

        with tempfile.TemporaryDirectory(prefix="trace-", dir=self.directory) as temporary:
            scratch = Path(temporary)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", WORKER,
                 str(self.source), str(self.vendor), str(scratch)],
                cwd=scratch / "work", env=fixture_environment(scratch),
                capture_output=True, text=True, encoding="utf-8", timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout)["ok"])
