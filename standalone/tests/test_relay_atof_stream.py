"""Native ATOF streaming records through actual owned WS and NDJSON peers."""
import base64
import json
from pathlib import Path
import unittest
import zlib

from standalone import relay_plugin
from standalone.tests import test_platform_sdks as platform_fixture


_WORKER=r'''
import asyncio,time,zlib
from standalone import relay,relay_plugin
from standalone.relay_types import Event
fixture=json.loads(zlib.decompress(base64.b85decode(FROZEN_ATOF_DATA)))
ports=set();guard.http_ports=ports
for index,case in enumerate(fixture['scenarios']):
    options={'refusals':case['initial_refusals'],'status':case['status']}
    if index%2==0:options.update(certificate=scratch/'cert.pem',key=scratch/'key.pem')
    with AtofPeer(**options) as peer:
        ports.add(peer.port)
        token=scratch/('header-'+str(index));token.write_text('file-one\n')
        os.environ['HERMES_ATOF_REFERENCE']='env-one'
        peer.on_refusal=lambda remaining:token.write_text('file-two\n') if remaining==1 else None
        config={'type':'stream','transport':case['transport'],'url':peer.url(case['transport']),
                'field_name_policy':case['policy'],'timeout_millis':1000,
                'headers':{'X-Literal':'literal'},'header_env':{'X-From-Env':'HERMES_ATOF_REFERENCE'},
                'header_file':{'X-From-File':str(token)}}
        sink=relay_plugin._AtofSink(config)
        os.environ['HERMES_ATOF_REFERENCE']='env-two'
        for record in case['input_records']:sink(Event(record))
        sink.flush()
        deadline=time.monotonic()+3
        while len(peer.records)<4 and time.monotonic()<deadline:
            time.sleep(.005)
        assert peer.records==case['wire_records'],(index,len(peer.records),len(case['wire_records']))
        if case['transport']=='ndjson':assert not peer.finished.is_set(),'NDJSON upload ended before exporter close'
        sink.close();sink.close()
        assert not sink.stream._thread.is_alive()
        headers=peer.headers[-1]
        assert headers['X-Literal']=='literal' and headers['X-From-Env']=='env-one'
        assert headers['X-From-File']==('file-two' if case['initial_refusals'] else 'file-one')
        assert peer.paths[-1]=='/owned/stream?fixture=atof'
        if case['initial_refusals']:assert len(peer.headers)==len(case['configured_headers'])==4
        if case['status']==500:assert any(row['code']=='atof.endpoint_delivery_failed' for row in sink.stream.runtime_diagnostics())
# Exercise the ordinary plugin/facade path, separately from exact frozen wire replay.
async def lifecycle():
    for transport in ('websocket','ndjson'):
        with AtofPeer(certificate=scratch/'cert.pem',key=scratch/'key.pem') as peer:
            ports.add(peer.port)
            config={'components':[{'kind':'observability','config':{'enable_full_payloads':True,
                'atof':{'enabled':True,'sinks':[{'type':'stream','transport':transport,'url':peer.url(transport)}]}}}]}
            assert not relay_plugin.validate_exact(config)['config']['diagnostics']
            active=await relay_plugin.initialize(config)
            handle=relay.scope.push('atof-original-path',relay.ScopeType.Agent)
            relay.scope.event('checkpoint',handle=handle,data={'visible':'owned'})
            relay.scope.pop(handle,output={'done':True})
            await active.close()
            assert [record['name'] for record in peer.records]==['atof-original-path','checkpoint','atof-original-path']
            assert peer.records[1]['data']=={'visible':'owned'}
            assert peer.records[2]['parent_uuid']==peer.records[0]['parent_uuid']
asyncio.run(lifecycle())
assert 'nemo_relay' not in sys.modules
print(json.dumps({'native_scenarios':len(fixture['scenarios']),'records':24,'tls':True,'plugin':True,
                  'reconnect':True,'long_lived_upload':True,'blocked_events':guard.events}),flush=True)
os._exit(0)
'''


class RelayAtofStreamTests(unittest.TestCase):
    setUpClass=classmethod(platform_fixture.OriginalPlatformSDKTests.setUpClass.__func__)

    def test_native_records_reconnect_headers_long_lived_upload_tls_and_plugin_lifecycle(self):
        fixture=Path(__file__).with_name('fixtures').joinpath('relay_atof_stream.json').read_bytes()
        peer=Path(__file__).with_name('relay_atof_peer.py').read_text()
        program=peer+'\n'+_WORKER.replace('FROZEN_ATOF_DATA',repr(base64.b85encode(zlib.compress(fixture))))
        report,_=platform_fixture.OriginalPlatformSDKTests.invoke(self,program)
        self.assertTrue(report['tls'] and report['plugin'] and report['reconnect'] and report['long_lived_upload'])
        self.assertEqual(report['records'],24)


class RelayAtofStreamValidationTests(unittest.TestCase):
    def test_transport_and_remote_credential_refusals_do_not_activate_workers(self):
        for sink in ({'transport':'websocket','url':'https://127.0.0.1/'},
                     {'transport':'ndjson','url':'ws://127.0.0.1/'},
                     {'transport':'other','url':'http://127.0.0.1/'},
                     {'transport':'websocket','url':'ws://remote.invalid/','headers':{'X-Fixed':'owned'}},
                     {'transport':'ndjson','url':'http://remote.invalid/','headers':{'X-Fixed':'owned'}},
                     {'transport':'ndjson','url':'http://127.0.0.1/','timeout_millis':0},
                     {'transport':'websocket','url':'ws://127.0.0.1/','timeout_millis':True}):
            config={'components':[{'kind':'observability','config':{'atof':{'enabled':True,'sinks':[{'type':'stream',**sink}]}}}]}
            with self.subTest(sink=sink):
                self.assertTrue(any(item['level']=='error' for item in relay_plugin.validate_exact(config)['config']['diagnostics']))


if __name__=='__main__':
    unittest.main()
