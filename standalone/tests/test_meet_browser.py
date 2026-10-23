"""Owned CDP pipes, original Meet bot admission/captions and bundled child re-entry."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import quote

from standalone.meet_browser_pipe import CDPError, PipeCDP
from standalone.tests.test_entrypoints import BundledSourceTestCase, support


# An independent engine protocol peer. It owns fd 3/4 like Chromium and models
# page/application state; it does not interpret JavaScript or claim Meet DOM conformance.
_PEER = r'''
import json,os,pathlib,signal,sys
trace={'argv':sys.argv[1:],'pid':os.getpid(),'requests':[],'closed':False}
mode=os.environ.get('MEET_FIXTURE_MODE','admitted');targets={};contexts={};sequence=0;navs={};buffer=bytearray()
def deny(event,args):
    if event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.posix_spawn'):
        raise AssertionError('CDP peer cannot contact a network or start a process')
sys.addaudithook(deny)
def emit(message):
    raw=json.dumps(message,ensure_ascii=False,separators=(',',':')).encode()+b'\0'
    # Fragment headers and Unicode independently of NUL framing.
    for start in range(0,len(raw),19):
        part=raw[start:start+19]
        while part:part=part[os.write(4,part):]
def event(method,params,session=None):
    value={'method':method,'params':params}
    if session:value['sessionId']=session
    emit(value)
def response(message,result=None,error=None):
    value={'id':message['id'],'result':result or {}}
    if error:value={'id':message['id'],'error':{'code':-32000,'message':error}}
    if message.get('sessionId'):value['sessionId']=message['sessionId']
    emit(value)
def loaded(message):
    target=targets[message['sessionId'][2:]];loader='loader-'+str(message['id']);target['url']=message['params']['url']
    from urllib.parse import urlsplit
    parsed=urlsplit(target['url']);target['origin']=parsed.scheme+'://'+parsed.netloc
    event('Page.frameNavigated',{'frame':{'id':'main','url':target['url']}},message['sessionId'])
    event('Page.lifecycleEvent',{'name':'DOMContentLoaded','loaderId':loader,'frameId':'main'},message['sessionId'])
    response(message,{'frameId':'main','loaderId':loader})
try:
    while True:
        while b'\0' not in buffer:
            chunk=os.read(3,65536)
            if not chunk:raise SystemExit(0)
            buffer.extend(chunk)
        end=buffer.index(0);message=json.loads(bytes(buffer[:end]));del buffer[:end+1]
        trace['requests'].append(message);method=message['method'];params=message.get('params',{});session=message.get('sessionId')
        target=targets.get(session[2:]) if session else None
        if method=='Browser.getVersion':response(message,{'product':'Chromium/fixture','protocolVersion':'1.3'})
        elif method=='Target.createBrowserContext':
            identifier='context-'+str(len(contexts));contexts[identifier]={'cookies':[],'storage':{}}
            response(message,{'browserContextId':identifier})
        elif method=='Target.createTarget':
            sequence+=1;identifier='target-'+str(sequence)
            targets[identifier]={'targetId':identifier,'browserContextId':params['browserContextId'],'url':params['url'],
                                 'origin':'null','join':False,'muted':True,'drained':False,'fetch':False}
            response(message,{'targetId':identifier})
        elif method=='Target.attachToTarget':response(message,{'sessionId':'s-'+params['targetId']})
        elif method=='Target.getTargetInfo':
            if params['targetId'] in targets:response(message,{'targetInfo':targets[params['targetId']]})
            else:response(message,error='target closed')
        elif method=='Target.closeTarget':
            targets.pop(params['targetId'],None);response(message,{'success':True})
        elif method=='Target.disposeBrowserContext':
            targets={key:value for key,value in targets.items() if value['browserContextId']!=params['browserContextId']}
            response(message)
        elif method=='Storage.setCookies':
            contexts[params['browserContextId']]['cookies']=[{**cookie,'expires':cookie.get('expires',-1),'priority':'Medium'} for cookie in params['cookies']]
            response(message)
        elif method=='Storage.getCookies':response(message,{'cookies':contexts[params['browserContextId']]['cookies']})
        elif method=='Fetch.enable':target['fetch']=True;response(message)
        elif method=='Page.navigate':
            if target['fetch']:
                request='request-'+str(message['id']);navs[request]=message
                event('Fetch.requestPaused',{'requestId':request,'resourceType':'Document','request':{'url':params['url']}},session)
            elif mode=='navigation_error':response(message,{'frameId':'main','errorText':'fixture navigation refused'})
            else:loaded(message)
        elif method=='Fetch.fulfillRequest':
            assert params['responseCode']==200
            response(message);loaded(navs.pop(params['requestId']))
        elif method=='Runtime.evaluate':
            expression=params['expression'];value=None
            assert params['returnByValue'] and params['awaitPromise']
            state=contexts[target['browserContextId']]
            if 'const spec=' in expression:
                spec=json.JSONDecoder().raw_decode(expression.split('const spec=',1)[1])[0]
                selector=spec.get('selector') or '';name=spec.get('name')
                present=(selector.startswith('input[') or name=='Ask to join' or
                         ('Turn on microphone' in selector and target['muted']) or
                         ('Turn off microphone' in selector and not target['muted']))
                if 'return matches.length;' in expression:value=int(present)
                elif 'return !!element&&visible(element);' in expression:value=present
                elif 'element.focus();element.select();' in expression:target['focused']=True;value=present
                elif 'const hit=' in expression:
                    target['action']='mic' if 'microphone' in selector else 'join'
                    value={'x':150,'y':100} if present else None
            elif 'for (const item of ' in expression:
                items=json.JSONDecoder().raw_decode(expression.split('for (const item of ',1)[1])[0]
                state['storage'][target['origin']]={item['name']:item['value'] for item in items}
            elif expression.startswith('Object.keys(localStorage)'):
                value=[{'name':name,'value':value} for name,value in state['storage'].get(target['origin'],{}).items()]
            elif expression=='location.origin':value=target['origin']
            elif "You can't join this video call" in expression:value=mode=='denied'
            elif 'window.__hermesMeetInstalled && document.querySelector' in expression:
                value=target['join'] and mode not in ('denied','waiting')
            elif 'KeyboardEvent' in expression:value=True
            elif 'MutationObserver' in expression:target['observer']=True
            elif expression=='window.__hermesMeetDrain && window.__hermesMeetDrain()':
                assert target.get('observer')
                if mode=='closed':
                    event('Target.targetDestroyed',{'targetId':target['targetId']});response(message,error='fixture page closed');continue
                value=[] if target['drained'] else [{'speaker':'Alice','text':'Hello fixture'},
                    {'speaker':'Alice','text':'Hello fixture'},{'speaker':'Bob','text':'한글 caption'}]
                target['drained']=True
            elif 'if (b) b.click();' in expression:
                assert expression.startswith('(() =>') and expression.endswith(')()'),'Arrow callback must execute'
                target['left']=True
            else:raise AssertionError('Unexpected JavaScript operation in independent CDP peer')
            response(message,{'result':{'type':'object' if isinstance(value,(list,dict)) else 'boolean' if isinstance(value,bool)
                                        else 'number' if isinstance(value,(int,float)) else 'string' if isinstance(value,str) else 'undefined',
                                        **({'value':value} if value is not None else {})}})
        elif method=='Input.insertText':assert target.get('focused');target['guest_name']=params['text'];response(message)
        elif method=='Input.dispatchMouseEvent':
            if params['type']=='mouseReleased':
                if target.get('action')=='join':target['join']=True
                if target.get('action')=='mic':target['muted']=False
            response(message)
        elif method=='Browser.close':
            response(message);trace['closed']=True;break
        elif method in ('Target.setDiscoverTargets','Browser.grantPermissions','Page.enable','Page.setLifecycleEventsEnabled',
                        'Emulation.setDeviceMetricsOverride','Emulation.setUserAgentOverride','Network.setBypassServiceWorker'):
            response(message)
        else:raise AssertionError('Unknown CDP command '+method)
finally:
    path=pathlib.Path(os.environ['MEET_FIXTURE_TRACE'])
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600),'w') as output:json.dump(trace,output,ensure_ascii=False)
'''

_GUARD = r'''
from standalone.agent_probe_worker import ProbeGuard
class MeetGuard(ProbeGuard):
    def __call__(self,event,args):
        if event=='os.posix_spawn' and pathlib.Path(args[0]).resolve()==pathlib.Path(os.environ['MEET_FIXTURE_PEER']).resolve():return
        if event=='subprocess.Popen':
            command=args[1];prefix=[sys.executable,'-I','-S','-B','-X','utf8',context['artifact']]
            if (isinstance(command,(list,tuple)) and list(command[:7])==prefix and len(command)>10 and
                    command[7]=='--cache-dir' and command[9]=='--internal-module' and command[10]=='plugins.google_meet.meet_bot'):return
        return super().__call__(event,args)
guard=MeetGuard(source,source.parent/'vendor',scratch);sys.addaudithook(guard)
'''

_ARTIFACT = r'''
import json,os,pathlib,runpy,sys
context=json.loads(CONTEXT_JSON);scratch=pathlib.Path(os.environ['MEET_FIXTURE_SCRATCH'])
source=pathlib.Path(context['root'])/'app';sys.path[:0]=[str(source),str(source.parent/'vendor')]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
GUARD_CODE
args=sys.argv[1:];assert args[:1]==['--cache-dir'] and args[2:4]==['--internal-module','plugins.google_meet.meet_bot']
sys.argv=['plugins.google_meet.meet_bot',*args[4:]]
runpy.run_module('plugins.google_meet.meet_bot',run_name='__main__',alter_sys=True)
'''

_WORKER = r'''
import contextlib,io,json,os,pathlib,subprocess,sys,threading,time
context=json.loads(sys.argv[1]);scratch=pathlib.Path(sys.argv[2]);source=pathlib.Path(context['root'])/'app'
sys.path[:0]=[str(source),str(source.parent/'vendor')]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
GUARD_CODE
from plugins.google_meet import meet_bot,process_manager as pm,tools,cli
from plugins.google_meet.browser_boundary import chromium
assert tools.check_meet_requirements()
with contextlib.redirect_stdout(io.StringIO()) as output:
    assert cli._cmd_setup()==0
    assert cli._cmd_install(realtime=False,assume_yes=True)==0
assert 'bundled CDP pipe' in output.getvalue()
profiles=[]
state={'cookies':[{'name':'fixture','value':'not-a-real-credential','domain':'.google.com','path':'/',
                  'expires':-1,'httpOnly':True,'secure':True,'sameSite':'Lax'}],
       'origins':[{'origin':'https://accounts.google.com','localStorage':[{'name':'fixture-key','value':'보존'}]}]}
with chromium() as engine:
    browser=engine.launch(channel='chromium',headless=True);profiles.append(browser.profile.name)
    context_browser=browser.new_context(viewport={'width':1280,'height':800},user_agent='fixture-agent',
        permissions=['microphone','camera'],storage_state=state)
    page=context_browser.new_page();page.goto('https://accounts.google.com/',wait_until='domcontentloaded')
    saved=scratch/'saved-state.json';restored=context_browser.storage_state(path=saved)
    assert restored==state,(restored,state)
    assert saved.stat().st_mode&0o777==0o600
    context_browser.close();browser.close();assert browser.child.pid is None
assert not any(pathlib.Path(path).exists() for path in profiles)
for mode,reason,code in [('admitted','duration_expired',0),('denied','denied',0),('closed','page_closed',0),
                         ('waiting','lobby_timeout',0),('navigation_error',None,4)]:
    folder=scratch/mode;folder.mkdir();trace=scratch/(mode+'-trace.json')
    os.environ.update(HERMES_MEET_URL='https://meet.google.com/abc-defg-hij',HERMES_MEET_OUT_DIR=str(folder),
                      HERMES_MEET_DURATION='.1s',MEET_FIXTURE_TRACE=str(trace),MEET_FIXTURE_MODE=mode,
                      HERMES_MEET_LOBBY_TIMEOUT='-1' if mode=='waiting' else '300')
    assert meet_bot.run_bot()==code,mode
    result=json.loads((folder/'status.json').read_text());assert result['exited']
    assert result['leaveReason']==reason,(mode,result)
    if mode=='admitted':
        text=(folder/'transcript.txt').read_text();assert text.count('Alice: Hello fixture')==1 and 'Bob: 한글 caption' in text
        assert result['transcriptLines']==2 and result['joinedAt'] and result['micState']=='unmuted_clicked'
    protocol=json.loads(trace.read_text());assert protocol['closed']
    for argument in protocol['argv']:
        assert not argument.startswith('--remote-debugging-port')
        if argument.startswith('--user-data-dir='):assert not pathlib.Path(argument.split('=',1)[1]).exists()
# A native virtual device is already owned when navigation fails. The original
# bot must release it on this early return, before a Realtime socket is opened.
from standalone import meet_audio
class OwnedVirtualDevice:
    def __init__(self):self.closed=0
    def setup(self):
        return {'platform':'fixture','device_name':'private','write_target':'private','module_ids':[]}
    def close(self):self.closed+=1
owned=OwnedVirtualDevice();original_bridge=meet_audio.NativeBridge
meet_audio.NativeBridge=lambda _prefix:owned
try:
    failed=scratch/'realtime-navigation-error';failed.mkdir()
    os.environ.update(HERMES_MEET_MODE='realtime',HERMES_MEET_REALTIME_KEY='fixture-not-a-real-key',
                      HERMES_MEET_OUT_DIR=str(failed),MEET_FIXTURE_MODE='navigation_error',
                      MEET_FIXTURE_TRACE=str(scratch/'realtime-navigation-error-trace.json'))
    assert meet_bot.run_bot()==4
    assert owned.closed==1
finally:
    meet_audio.NativeBridge=original_bridge
    os.environ['HERMES_MEET_MODE']='transcribe'
os.environ.update(MEET_FIXTURE_MODE='admitted',MEET_FIXTURE_TRACE=str(scratch/'detached-trace.json'),
                  HERMES_MEET_LOBBY_TIMEOUT='300')
out=scratch/'detached';result=pm.start('https://meet.google.com/abc-defg-hij',out_dir=out,duration='.1s',
                                      guest_name='Private fixture',realtime_api_key='fixture-not-a-real-key')
assert result['ok'];pid=result['pid'];deadline=time.monotonic()+15
while True:
    try:status=json.loads((out/'status.json').read_text())
    except (OSError,ValueError):status={}
    if status.get('exited'):break
    assert time.monotonic()<deadline,'Detached original bot did not finish'
    threading.Event().wait(.02)
waited,exitstatus=os.waitpid(pid,0);assert waited==pid and os.waitstatus_to_exitcode(exitstatus)==0
assert pm.transcript()['total']==2 and pm.stop()['ok']
assert not any(name=='playwright' or name.startswith('playwright.') for name in sys.modules)
assert not guard.events,guard.events
print(json.dumps({'original_bot':True,'storage':True,'detached_bundle_child':True,'cases':5}))
'''


class PipeProtocolTests(unittest.TestCase):
    def fixture(self):
        commands, replies = os.pipe(), os.pipe()
        child = SimpleNamespace(write_fd=commands[1], read_fd=replies[0])
        pipe = PipeCDP(child)
        self.addCleanup(pipe.close)
        self.addCleanup(os.close, commands[0])
        self.addCleanup(os.close, replies[1])
        return pipe, commands[0], replies[1]

    def test_out_of_order_responses_unicode_and_lifecycle_before_response(self):
        pipe, _commands, replies = self.fixture()
        first = pipe.send("Fixture.first")
        second = pipe.send("Fixture.second")
        messages = [{"method": "Page.lifecycleEvent", "sessionId": "page", "params": {"name": "DOMContentLoaded", "loaderId": "A"}},
                    {"id": second, "result": {"text": "한글"}}, {"id": first, "result": {"number": 42}}]
        for message in messages:
            os.write(replies, json.dumps(message, ensure_ascii=False).encode() + b"\0")
        self.assertEqual(pipe.result(first), {"number": 42})
        self.assertEqual(pipe.result(second), {"text": "한글"})
        self.assertEqual(pipe.event(lambda item: item.get("sessionId") == "page")["params"]["loaderId"], "A")

    def test_malformed_oversized_unknown_response_and_timeout_fail(self):
        for raw in (b"[]\0", b"{bad\0", b'{"id":999,"result":{}}\0'):
            with self.subTest(raw=raw):
                pipe, _commands, replies = self.fixture()
                identifier = pipe.send("Fixture.invalid")
                os.write(replies, raw)
                with self.assertRaises(CDPError):
                    pipe.result(identifier)
        pipe, _commands, replies = self.fixture()
        identifier = pipe.send("Fixture.large")
        with patch("standalone.meet_browser_pipe.MAX_MESSAGE", 64):
            os.write(replies, b"x" * 64)
            with self.assertRaisesRegex(CDPError, "byte limit"):
                pipe.result(identifier)
        pipe, _commands, _replies = self.fixture()
        with self.assertRaises(TimeoutError):
            pipe.result(pipe.send("Fixture.silent"), timeout=.001)


@unittest.skipUnless(os.name == "posix", "Meet Chromium pipe uses the plugin's POSIX boundary")
class OriginalMeetBrowserTests(BundledSourceTestCase):
    def test_original_bot_and_owned_pipe_engine(self):
        with tempfile.TemporaryDirectory(prefix="meet-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            peer = scratch / "chromium-peer"
            peer.write_text("#!" + sys.executable + "\n" + _PEER)
            peer.chmod(0o700)
            context = json.loads(json.dumps(self.context))
            context["manifest"]["internal_modules"].append("plugins.google_meet.meet_bot")
            Path(context["artifact"]).write_text(_ARTIFACT.replace("CONTEXT_JSON", repr(json.dumps(context))).replace("GUARD_CODE", _GUARD))
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            environment.update(AGENT_BROWSER_EXECUTABLE_PATH=str(peer), MEET_FIXTURE_PEER=str(peer),
                               MEET_FIXTURE_TRACE=str(scratch / "storage-trace.json"), MEET_FIXTURE_SCRATCH=str(scratch))
            process = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", _WORKER.replace("GUARD_CODE", _GUARD),
                 json.dumps(context), str(scratch)], cwd=scratch / "work", env=environment,
                capture_output=True, text=True, timeout=50,
            )
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            report = json.loads(process.stdout)
            self.assertTrue(report["original_bot"] and report["storage"] and report["detached_bundle_child"])


class NativeChromiumDOMTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "Meet browser engine is POSIX-only")
    def test_private_engine_dom_and_trusted_input_without_meet_account(self):
        from standalone.meet_browser import Chromium, find_executable
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "HOME": directory, "USERPROFILE": directory, "HERMES_HOME": directory + "/hermes",
                "XDG_CONFIG_HOME": directory + "/config", "XDG_CACHE_HOME": directory + "/cache"}):
            self._private_engine_dom(Chromium, find_executable)

    def _private_engine_dom(self, Chromium, find_executable):
        try:
            executable = find_executable()
        except CDPError as error:
            self.skipTest(str(error))
        html = ('<input aria-label="Your name"><button aria-label="Ask to join" '
                'onclick="window.trusted=event.isTrusted">Join</button>')
        with Chromium() as engine:
            browser = engine.launch(executable_path=executable)
            profile = browser.profile.name
            page = browser.new_context().new_page()
            page.goto("data:text/html," + quote(html))
            page.locator('input[aria-label*="name" i]').first.fill("이름")
            self.assertEqual(page.evaluate("document.querySelector('input').value"), "이름")
            page.get_by_role("button", name="join", exact=False).first.click()
            self.assertTrue(page.evaluate("window.trusted"))
            browser.close()
            self.assertIsNone(browser.child.pid)
        self.assertFalse(Path(profile).exists())


if __name__ == "__main__":
    unittest.main()
