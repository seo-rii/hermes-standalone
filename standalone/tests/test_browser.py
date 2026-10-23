"""Original browser tool, bundled child/daemon, and real loopback CDP wire."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone.tests.test_entrypoints import BundledSourceTestCase, support


_GUARD = r'''
from standalone.agent_probe_worker import ProbeGuard
class BrowserGuard(ProbeGuard):
    def __call__(self,event,args):
        if event.startswith('socket.') and event not in {'socket.__new__','socket.gethostname'}:
            if event in {'socket.connect','socket.bind'}:
                address=args[1]
                if isinstance(address,str):
                    if pathlib.Path(address).resolve().is_relative_to(scratch): return
                elif isinstance(address,tuple) and address[0] in ('127.0.0.1','::1'):
                    if event=='socket.bind' and address[1]==0: return
                    if event=='socket.connect' and address[1]==int(os.environ['BROWSER_FIXTURE_PORT']): return
                    if event=='socket.connect' and address[1] in (9222,9223):
                        raise ConnectionRefusedError('Fixture refuses discovery of the user browser')
            if event=='socket.getaddrinfo' and args[0] in ('127.0.0.1','::1','localhost'): return
            return self.deny(event,'unexpected_browser_network')
        if event=='subprocess.Popen':
            command=args[1]
            prefix=[sys.executable,'-I','-S','-B','-X','utf8',context['artifact'],'--cache-dir',str(source.parent.parent)]
            # The verified runtime chooses its own extraction cache directory.
            if (isinstance(command,(list,tuple)) and list(command[:7])==prefix[:7]
                    and len(command)>10 and command[7]=='--cache-dir'
                    and command[9]=='--internal-module'
                    and command[10] in context['manifest']['internal_modules']): return
            return self.deny(event,'unexpected_browser_child')
        return super().__call__(event,args)
guard=BrowserGuard(source,source.parent/'vendor',scratch)
sys.addaudithook(guard)
'''

_ARTIFACT = r'''
import json,os,pathlib,runpy,sys
context=json.loads(CONTEXT_JSON)
source=pathlib.Path(context['root'])/'app'
scratch=pathlib.Path(os.environ['BROWSER_FIXTURE_SCRATCH'])
sys.path[:0]=[str(source),str(source.parent/'vendor')]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
GUARD_CODE
arguments=sys.argv[1:]
assert arguments[:1]==['--cache-dir'] and arguments[2:3]==['--internal-module']
module=arguments[3]
assert module in context['manifest']['internal_modules']
sys.argv=[module,*arguments[4:]]
runpy.run_module(module,run_name='__main__',alter_sys=True)
'''

_WORKER = r'''
import asyncio,base64,contextlib,io,json,os,pathlib,sys,traceback
context=json.loads(sys.argv[1]);scratch=pathlib.Path(sys.argv[2]);source=pathlib.Path(context['root'])/'app'
sys.path[:0]=[str(source),str(source.parent/'vendor')]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
os.chdir(scratch/'work')
from standalone import raster
image=raster.new('RGB',(80,40),'#204060')
image.paste((220,180,90),(40,0,80,40))
buffer=io.BytesIO();image.save(buffer,format='PNG');png=base64.b64encode(buffer.getvalue()).decode()
from websockets.asyncio.server import serve
requests=[];tabs={'page-fixture':{'targetId':'page-fixture','type':'page','title':'Fixture page','url':'https://fixture.invalid/'}}
current='page-fixture'
async def cdp(websocket):
    global current
    async for raw in websocket:
        message=json.loads(raw);method=message['method'];params=message.get('params',{})
        requests.append({'method':method,'params':params,'sessionId':message.get('sessionId')})
        result={}
        if method=='Target.getTargets':result={'targetInfos':list(tabs.values())}
        elif method=='Target.createTarget':
            current='page-'+str(len(tabs)+1);tabs[current]={'targetId':current,'type':'page','title':'Fixture created','url':params['url']};result={'targetId':current}
        elif method=='Target.attachToTarget':current=params['targetId'];result={'sessionId':'session-'+current}
        elif method=='Target.getTargetInfo':result={'targetInfo':tabs[params.get('targetId',current)]}
        elif method=='Target.closeTarget':tabs.pop(params['targetId'],None);result={'success':True}
        elif method=='Runtime.evaluate':
            expression=params['expression']
            if 'JSON.stringify({url:location.href' in expression:
                value=json.dumps({'url':tabs[current]['url'],'title':tabs[current]['title'],'w':80,'h':40,'sx':0,'sy':0,'pw':80,'ph':40})
            elif expression=='window.devicePixelRatio':value=1
            elif 'document.readyState' in expression:value='complete'
            elif expression=='1+2':value=3
            elif 'location.href' in expression:value=tabs[current]['url']
            else:value=True
            result={'result':{'type':'string' if isinstance(value,str) else 'number' if type(value) is int else 'boolean','value':value}}
        elif method=='Page.captureScreenshot':result={'data':png}
        elif method=='Page.navigate':tabs[current]['url']=params['url'];result={'frameId':'frame-fixture'}
        elif method in {'Page.enable','DOM.enable','Runtime.enable','Network.enable','Network.disable',
                         'Input.dispatchMouseEvent','Input.insertText','Target.activateTarget','Target.setAutoAttach'}:pass
        else:
            await websocket.send(json.dumps({'id':message['id'],'error':{'code':-32601,'message':'CDP fixture does not implement '+method}}));continue
        await websocket.send(json.dumps({'id':message['id'],'result':result}))

async def main():
    async with serve(cdp,'127.0.0.1',0) as server:
        port=server.sockets[0].getsockname()[1]
        os.environ.update(BROWSER_FIXTURE_PORT=str(port),BROWSER_FIXTURE_SCRATCH=str(scratch),
            BU_CDP_WS=f'ws://127.0.0.1:{port}/browser',BH_RUNTIME_DIR=str(scratch/'run'),
            BH_TMP_DIR=str(scratch/'shots'),BH_HOME=str(scratch/'browser-home'),
            BH_AGENT_WORKSPACE=str(scratch/'work'),BH_UPDATE_CHECK='0',BH_RECORD='0',BH_TAB_MARKER='0',
            BH_TELEMETRY='0',ANONYMIZED_TELEMETRY='false',BU_AUTOSPAWN='')
        GUARD_CODE
        try:
            from tools.browser_use_cli import browser_exec,stop_harness_daemons
            # URL safety is the original pre-spawn gate. A metadata URL must
            # never reach the CDP endpoint or start a daemon.
            denied=await asyncio.to_thread(browser_exec,"goto_url('http://169.254.169.254/latest/meta-data/')")
            denied=json.loads(denied) if isinstance(denied,str) else denied
            before=len(requests)
            code="""import json
print('fixture-evaluation',js('1+2'))
print(json.dumps(page_info()))
print(capture_screenshot(max_dim=20))
"""
            result=await asyncio.to_thread(browser_exec,code,timeout_s=30)
            result=json.loads(result) if isinstance(result,str) else result
            # Invoke the original CLI again against the same persistent daemon.
            from hermes_cli.runtime_module import python_module_command
            command=python_module_command('browser_harness.run',['--debug-clicks'])
            process=await asyncio.create_subprocess_exec(*command,stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,env=dict(os.environ),cwd=scratch/'work')
            stdout,stderr=await asyncio.wait_for(process.communicate(b"click_at_xy(20,20)\nprint('clicked')\n"),30)
            shot=raster.open(scratch/'shots'/'shot.png')
            debug=raster.open(scratch/'shots'/'debug_click_0.png')
            from browser_harness import video_render
            output=scratch/'work'/'contact.jpg'
            video_render.contact_sheet([{'path':str(scratch/'shots'/'shot.png'),'label':'frame','time':1.25}],output,'Hermes review')
            contact=raster.open(output)
            from browser_harness import _ipc
            daemon_pid=_ipc.identify('default')
            endpoint_exists=_ipc.ping('default')
            mcp_command=python_module_command('browser_harness.mcp_cli')
            mcp=await asyncio.create_subprocess_exec(*mcp_command,stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,env=dict(os.environ),cwd=scratch/'work')
            async def rpc(message,reply=True):
                mcp.stdin.write((json.dumps(message)+'\n').encode());await mcp.stdin.drain()
                if reply:
                    line=await asyncio.wait_for(mcp.stdout.readline(),15)
                    if not line:
                        error=(await mcp.stderr.read()).decode()
                        raise RuntimeError('Browser MCP child exited: '+error)
                    return json.loads(line)
            modern_metadata={'io.modelcontextprotocol/protocolVersion':'2026-07-28',
                             'io.modelcontextprotocol/clientCapabilities':{}}
            modern_discovery=await rpc({'jsonrpc':'2.0','id':90,'method':'server/discover',
                                       'params':{'_meta':modern_metadata}})
            modern_evaluated=await rpc({'jsonrpc':'2.0','id':91,'method':'tools/call','params':{
                '_meta':modern_metadata,'name':'browser_js','arguments':{'expression':'1+2'}}})
            initialized=await rpc({'jsonrpc':'2.0','id':1,'method':'initialize','params':{
                'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'fixture','version':'1'}}})
            await rpc({'jsonrpc':'2.0','method':'notifications/initialized'},False)
            listed=await rpc({'jsonrpc':'2.0','id':2,'method':'tools/list'})
            evaluated=await rpc({'jsonrpc':'2.0','id':3,'method':'tools/call','params':{
                'name':'browser_js','arguments':{'expression':'1+2'}}})
            clicks_before=sum(r['method']=='Input.dispatchMouseEvent'for r in requests)
            invalid=await rpc({'jsonrpc':'2.0','id':4,'method':'tools/call','params':{
                'name':'browser_click','arguments':{'x':True,'y':20}}})
            clicks_after=sum(r['method']=='Input.dispatchMouseEvent'for r in requests)
            mcp_shot=await rpc({'jsonrpc':'2.0','id':5,'method':'tools/call','params':{
                'name':'browser_screenshot','arguments':{'path':str(scratch/'shots'/'mcp.png'),'max_dim':20}}})
            mcp.stdin.close();await asyncio.wait_for(mcp.wait(),10)
            mcp_error=(await mcp.stderr.read()).decode()
            report={'denied':denied,'requests_before_allowed':before,'tool_result':result,
                    'cli_exit':process.returncode,'cli_stdout':stdout.decode(),'cli_stderr':stderr.decode(),
                    'screenshot_size':shot.size,'debug_size':debug.size,
                    'debug_center':debug.getpixel((20,20)),'contact_size':contact.size,
                    'daemon_pid':daemon_pid,'daemon_live':endpoint_exists,'methods':[r['method']for r in requests],
                    'mcp_initialize':initialized,'mcp_tools':listed,'mcp_evaluated':evaluated,
                    'mcp_modern_discovery':modern_discovery,'mcp_modern_evaluated':modern_evaluated,
                    'mcp_invalid':invalid,'mcp_invalid_sent_click':clicks_after!=clicks_before,
                    'mcp_screenshot':mcp_shot,'mcp_exit':mcp.returncode,'mcp_stderr':mcp_error,
                    'no_pil':not any(n=='PIL'or n.startswith('PIL.')for n in sys.modules),
                    'blocked_events':guard.events}
            await asyncio.to_thread(stop_harness_daemons)
            report['daemon_stopped']=not _ipc.ping('default')
            print(json.dumps(report),flush=True)
        except BaseException:
            print(traceback.format_exc(),file=sys.stderr)
            with contextlib.suppress(BaseException):
                from browser_harness import _ipc
                channel,token=_ipc.connect('default');_ipc.request(channel,token,{'meta':'shutdown'});channel.close()
            raise
asyncio.run(main())
'''


class BrowserRuntimeTests(BundledSourceTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.context["manifest"]["internal_modules"].extend([
            "browser_harness.run", "browser_harness.daemon", "browser_harness.mcp_cli",
        ])
        Path(cls.context["artifact"]).write_text(
            _ARTIFACT.replace("CONTEXT_JSON", repr(json.dumps(cls.context))).replace("GUARD_CODE", _GUARD),
            encoding="utf-8",
        )

    def test_original_browser_exec_and_detached_daemon_use_real_cdp_wire(self):
        with tempfile.TemporaryDirectory(prefix="browser-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp", "run", "shots", "browser-home"):
                (scratch / name).mkdir()
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            worker = _WORKER.replace("        GUARD_CODE", "\n".join("        " + line for line in _GUARD.splitlines()))
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", worker,
                 json.dumps(self.context), str(scratch)],
                cwd=scratch / "work", env=environment, capture_output=True, text=True, timeout=65,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout.splitlines()[-1])
            self.assertIn("error", report["denied"])
            self.assertEqual(report["requests_before_allowed"], 0)
            self.assertTrue(report["tool_result"].get("success"), report)
            self.assertIn("fixture-evaluation 3", report["tool_result"]["output"])
            self.assertEqual(report["screenshot_size"], [20, 10])
            self.assertEqual(report["debug_size"], [80, 40])
            self.assertEqual(report["debug_center"], [255, 0, 0])
            self.assertEqual(report["cli_exit"], 0, report)
            self.assertIn("clicked", report["cli_stdout"])
            self.assertTrue(report["daemon_live"] and report["daemon_stopped"], report)
            self.assertIsInstance(report["daemon_pid"], int)
            for method in ("Target.getTargets", "Target.attachToTarget", "Page.captureScreenshot", "Input.dispatchMouseEvent"):
                self.assertIn(method, report["methods"])
            self.assertTrue(report["no_pil"])
            self.assertEqual(report["blocked_events"], [])
            self.assertEqual(report["mcp_initialize"]["result"]["serverInfo"]["name"], "browser-harness")
            self.assertIn("2026-07-28", report["mcp_modern_discovery"]["result"]["supportedVersions"])
            modern = report["mcp_modern_evaluated"]["result"]
            self.assertEqual(modern["resultType"], "complete")
            self.assertEqual(json.loads(modern["content"][0]["text"]), 3)
            self.assertIn("browser_screenshot", {tool["name"] for tool in report["mcp_tools"]["result"]["tools"]})
            self.assertEqual(json.loads(report["mcp_evaluated"]["result"]["content"][0]["text"]), 3)
            self.assertEqual(report["mcp_invalid"]["error"]["code"], -32602)
            self.assertFalse(report["mcp_invalid_sent_click"])
            shot = json.loads(report["mcp_screenshot"]["result"]["content"][0]["text"])
            self.assertEqual((shot["width"], shot["height"]), (20, 10))
            self.assertEqual(report["mcp_exit"], 0, report)


if __name__ == "__main__":
    unittest.main()
