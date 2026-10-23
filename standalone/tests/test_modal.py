"""Original Modal SDK and Hermes environment over real loopback gRPC/HTTP2."""

import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone.tests import test_transport_agent as _agent_fixture


class FileWatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_polling_reports_changes_filters_and_stops_without_external_tools(self):
        from standalone.file_watch import Change, DefaultFilter, awatch
        with tempfile.TemporaryDirectory(prefix="hermes-watch-") as temporary:
            root = Path(temporary)
            target = root / "example.py"
            target.write_text("before")
            stop = asyncio.Event()
            events = awatch(root, step=10, poll_delay_ms=5, debounce=30, stop_event=stop)
            pending = asyncio.create_task(anext(events))
            await asyncio.sleep(0.05)
            target.write_text("after, with different size")
            changes = await asyncio.wait_for(pending, 2)
            self.assertIn((Change.modified, str(target)), changes)
            pending = asyncio.create_task(anext(events))
            await asyncio.sleep(0.03)
            target.unlink()
            changes = await asyncio.wait_for(pending, 2)
            self.assertIn((Change.deleted, str(target)), changes)
            self.assertFalse(DefaultFilter()(Change.added, str(root / ".git" / "objects")))
            self.assertFalse(DefaultFilter()(Change.added, str(root / "example.pyc")))
            pending = asyncio.create_task(anext(events))
            await asyncio.sleep(0.02)
            stop.set()
            with self.assertRaises(StopAsyncIteration):
                await asyncio.wait_for(pending, 2)
            await events.aclose()

    async def test_cancel_and_missing_root_close_promptly(self):
        from standalone.file_watch import awatch
        with tempfile.TemporaryDirectory(prefix="hermes-watch-") as temporary:
            stream = awatch(temporary, step=10)
            task = asyncio.create_task(anext(stream))
            await asyncio.sleep(0.03)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await stream.aclose()
            with self.assertRaises(FileNotFoundError):
                await anext(awatch(Path(temporary) / "missing"))


WORKER = r'''
import asyncio,base64,contextlib,io,json,os,pathlib,re,sys,tarfile,time,traceback
source,vendor,scratch=map(pathlib.Path,sys.argv[1:])
sys.path[:0]=[str(source),str(vendor)]
from standalone.runtime import initialize_bundle
initialize_bundle({'root':str(source.parent),'artifact':str(scratch/'fixture.py'),
    'bundle_sha256':source.parent.name,'manifest':{'project':'hermes-standalone'}})
home=pathlib.Path(os.environ['HERMES_HOME']);home.mkdir(parents=True)
(home/'config.yaml').write_text(json.dumps({'terminal':{'backend':'modal'},'agent':{'environment_probe':False}}))
os.environ['MODAL_CONFIG_PATH']=str(scratch/'modal.toml')
os.environ['MODAL_TOKEN_ID']='fixture-modal-id'
os.environ['MODAL_TOKEN_SECRET']='fixture-modal-secret'
from standalone.agent_probe_worker import ProbeGuard
ports=set()
class Guard(ProbeGuard):
    def deny(self,*args,**kwargs):
        try:super().deny(*args,**kwargs)
        finally:(scratch/'modal-guard.json').write_text(json.dumps(self.events))
    def __call__(self,event,args):
        if event=='socket.bind' and args[1]==('127.0.0.1',0): return
        if event=='socket.connect' and isinstance(args[1],tuple) and args[1][0]=='127.0.0.1' and args[1][1] in ports: return
        if event=='socket.getaddrinfo' and args[0]=='127.0.0.1' and int(args[1]) in ports: return
        super().__call__(event,args)
guard=Guard(source,vendor,scratch);sys.addaudithook(guard)
import faulthandler
stack_log=(scratch/'modal-stack.log').open('w')
faulthandler.dump_traceback_later(25,file=stack_log,repeat=False)
import modal,cbor2
from modal.client import _Client
from modal_proto import api_pb2 as pb,task_command_router_pb2 as router
from google.protobuf import message_factory
from grpclib.server import Server
from grpclib.const import Cardinality,Handler,Status
from grpclib.exceptions import GRPCError
records,commands,inputs=[],{},{}
archive=io.BytesIO()
with tarfile.open(fileobj=archive,mode='w'): pass
empty_tar=archive.getvalue()
origin=''
router_token='eyJhbGciOiJub25lIn0.'+base64.urlsafe_b64encode(json.dumps({'exp':4102444800}).encode()).rstrip(b'=').decode()+'.fixture'

class Service:
    def __init__(self,descriptor):self.descriptor=descriptor
    def __mapping__(self):
        methods={}
        for method in self.descriptor.methods:
            async def call(stream,method=method):await self.call(method,stream)
            cardinality=(Cardinality.STREAM_STREAM if method.server_streaming else Cardinality.STREAM_UNARY) if method.client_streaming else (Cardinality.UNARY_STREAM if method.server_streaming else Cardinality.UNARY_UNARY)
            methods['/'+self.descriptor.full_name+'/'+method.name]=Handler(call,cardinality,
                message_factory.GetMessageClass(method.input_type),message_factory.GetMessageClass(method.output_type))
        return methods
    async def call(self,method,stream):
        request=await stream.recv_message();name=method.name
        records.append((name,request))
        with (scratch/'modal-rpc.txt').open('a') as progress:progress.write(name+'\n')
        response=message_factory.GetMessageClass(method.output_type)
        metadata=dict(stream.metadata)
        if self.descriptor.name=='TaskCommandRouter':
            assert metadata['authorization']=='Bearer '+router_token,metadata
        else:
            assert metadata['x-modal-token-id']=='fixture-modal-id',metadata
            assert metadata['x-modal-token-secret']=='fixture-modal-secret',metadata
        if name=='ClientHello': value=response()
        elif name=='AuthTokenGet': value=response(token=router_token)
        elif name=='AppGetOrCreate': value=response(app_id='ap-fixture')
        elif name=='ImageFromId': value=response(image_id=request.image_id)
        elif name=='ImageGetOrCreate': value=response(image_id='im-fixture',result={'status':pb.GenericResult.GENERIC_STATUS_SUCCESS})
        elif name=='ImageJoinStreaming': value=response(result={'status':pb.GenericResult.GENERIC_STATUS_SUCCESS},eof=True)
        elif name=='SandboxCreate':
            assert list(request.definition.entrypoint_args)==['sleep','infinity']
            value=response(sandbox_id='sb-'+'a'*22)
        elif name=='SandboxWait': value=response()
        elif name in ('SandboxGetTaskId','SandboxGetTaskIdV2'):value=response(task_id='ta-fixture')
        elif name=='TaskGetCommandRouterAccess':value=response(url=origin,jwt=router_token)
        elif name=='TaskExecStart':
            commands[request.exec_id]=list(request.command_args);value=response()
        elif name=='TaskExecStdioRead':
            command=' '.join(commands[request.exec_id])
            if request.file_descriptor==router.TASK_EXEC_STDIO_FILE_DESCRIPTOR_STDOUT:
                data=empty_tar if 'tar cf -' in command else ('원본 Modal 실행\n'.encode() if 'modal-fixture-output' in command else b'')
                for offset in range(0,len(data),7):await stream.send_message(response(data=data[offset:offset+7]))
            return
        elif name=='TaskExecStdinWrite':
            inputs.setdefault(request.exec_id,bytearray()).extend(request.data);value=response()
        elif name in ('TaskExecWait','TaskExecPoll'):value=response(code=0)
        elif name in ('SandboxTerminate','AppClientDisconnect'):value=response()
        else:raise GRPCError(Status.UNIMPLEMENTED,'Unexpected fixture RPC: '+name)
        await stream.send_message(value)

async def exercise():
    global origin
    service=Server([Service(pb.DESCRIPTOR.services_by_name['ModalClient']),
                    Service(router.DESCRIPTOR.services_by_name['TaskCommandRouter'])])
    await service.start('127.0.0.1',0)
    port=service._server.sockets[0].getsockname()[1];ports.add(port);origin='http://127.0.0.1:'+str(port)
    os.environ['MODAL_SERVER_URL']=origin
    from modal._serialization import serialize_data_format,deserialize_data_format
    from decimal import Decimal
    from datetime import datetime,timezone
    from uuid import UUID
    values={'text':'한글','binary':b'\0\xff','big':2**100,'negative':-2**100,'set':{1,2},
            'decimal':Decimal('1.25'),'uuid':UUID(int=42),'time':datetime(2026,1,1,tzinfo=timezone.utc)}
    wire=serialize_data_format(values,pb.DATA_FORMAT_CBOR)
    assert deserialize_data_format(wire,pb.DATA_FORMAT_CBOR,None)==values
    assert cbor2.loads(bytes.fromhex('9f018202039f0405ffff'))==[1,[2,3],[4,5]]
    assert '_cbor2' not in sys.modules
    from tools.environments.modal import ModalEnvironment
    env=None
    sdk_client=router_client=None
    try:
        env=await asyncio.to_thread(ModalEnvironment,'im-fixture',persistent_filesystem=False,task_id='modal-fixture')
        from modal._utils.async_utils import synchronizer
        original_sandbox=synchronizer._translate_in(env._sandbox)
        sdk_client=original_sandbox._client
        result=await asyncio.to_thread(env.execute,'printf modal-fixture-output',timeout=10)
        assert result['returncode']==0 and '원본 Modal 실행' in result['output'],{
            'result':result,'methods':[name for name,_ in records],
            'commands':commands,'errors':err.getvalue()[-4000:]}
        upload=scratch/'upload.txt';upload.write_text('전송 데이터')
        await asyncio.to_thread(env._modal_upload,str(upload),'/root/fixture.txt')
        assert any(base64.b64encode(upload.read_bytes()) in data for data in inputs.values())
        router_client=original_sandbox._command_router_client
        await asyncio.to_thread(env.cleanup)
        assert any(name=='SandboxTerminate' for name,_ in records)
        assert any(name=='SandboxCreate' for name,_ in records)
        assert any(name=='TaskExecStart' and list(req.command_args)[:2]==['bash','-c'] for name,req in records)
    except BaseException:
        (scratch/'modal-error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if env is not None:
            if env._sandbox is not None:await asyncio.to_thread(env.cleanup)
            if sdk_client is not None:
                async def close_sdk():
                    if router_client is not None:await router_client.close()
                    await sdk_client._close()
                owner=sdk_client._cancellation_context_event_loop
                if owner is asyncio.get_running_loop():await close_sdk()
                elif owner.is_running():await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(close_sdk(),owner))
            loop=env._worker._loop
            if loop and not loop.is_running() and not loop.is_closed():
                await asyncio.to_thread(loop.close)
        service.close();await service.wait_closed()
    assert not any(name in sys.modules for name in ('_cbor2','watchfiles','grpc','cryptography','pydantic'))
    return {'ok':True,'rpc_methods':sorted({name for name,_ in records}),'commands':len(commands),'cbor':True}
out,err=io.StringIO(),io.StringIO()
try:
    with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):result=asyncio.run(exercise())
except BaseException:
    print(traceback.format_exc(),file=sys.stderr);print(err.getvalue()[-3000:],file=sys.stderr);raise
print(json.dumps(result))
'''


@unittest.skipIf(sys.version_info < (3, 14), "Original Hermes environment requires Python 3.14")
class ModalEnvironmentTests(unittest.TestCase):
    setUpClass = classmethod(_agent_fixture.AgentHTTPIntegrationTests.setUpClass.__func__)

    def test_original_environment_and_sdk_use_real_grpc_and_pure_cbor(self):
        from standalone.agent_probe_worker import fixture_environment
        with tempfile.TemporaryDirectory(prefix="modal-", dir=self.directory) as temporary:
            scratch = Path(temporary)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            try:
                result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", WORKER,
                                         str(self.source), str(self.vendor), str(scratch)],
                                        cwd=scratch / "work", env=fixture_environment(scratch),
                                        capture_output=True, text=True, encoding="utf-8", timeout=60)
            except subprocess.TimeoutExpired:
                self.fail({p.name:p.read_text(errors="replace")[-8000:] for p in scratch.glob("modal-*") if p.is_file()})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout)["ok"])
