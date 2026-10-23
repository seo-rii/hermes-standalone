"""Pinned native OTLP records and actual private/shared file-handle contracts."""
import ctypes
import json
from pathlib import Path
import unittest

from standalone.tests import test_entrypoints as fixture


_WORKER = r'''
import base64,json,os,pathlib,struct,sys
context=json.loads(sys.argv[1]);scratch=pathlib.Path(sys.argv[2]);source=pathlib.Path(context['root'])/'app'
sys.path[:0]=[str(source),str(source.parent/'vendor')]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
from standalone.agent_probe_worker import ProbeGuard
guard=ProbeGuard(source,source.parent/'vendor',scratch);sys.addaudithook(guard)
from standalone.relay_files import FileDestination,encode_request,validate_config
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
reference=json.loads(pathlib.Path(FIXTURE).read_text())
native=base64.b64decode(reference['proto_base64'])
frames=[];offset=0
while offset<len(native):
    size=struct.unpack_from('>I',native,offset)[0];offset+=4
    frames.append(native[offset:offset+size]);offset+=size
assert offset==len(native)
def normalize(value):
    if isinstance(value,dict):
        return {key:sorted([normalize(item) for item in child],key=lambda item:item['key']) if key=='attributes'
                else normalize(child) for key,child in value.items()}
    if isinstance(value,list):return list(map(normalize,value))
    return value
expected=list(map(json.loads,reference['json_lines'].splitlines()))
assert [normalize(json.loads(encode_request(payload,'json_lines'))) for payload in frames]==list(map(normalize,expected))
assert b''.join(encode_request(payload,'proto') for payload in frames)==native
assert encode_request(b'','json_lines')==encode_request(b'','proto')==b''
payload=frames[0]
for format,filename in [('json_lines','actual.jsonl'),('proto','actual.pb')]:
    config={'output_directory':str(scratch/'work/new/nested'),'filename':filename,'format':format}
    first=FileDestination(config);first.write(payload)
    second=FileDestination(config)
    assert first.writer is second.writer
    for change in ({'format':'proto' if format=='json_lines' else 'json_lines'},{'mode':'append'}):
        try:FileDestination({**config,**change})
        except ValueError:pass
        else:raise AssertionError('Conflicting shared file settings were accepted')
    first.close();second.write(payload);second.write(b'');second.close()
    path=scratch/'work/new/nested'/filename
    record=encode_request(payload,format)
    assert path.read_bytes()==record*2
    third=FileDestination({**config,'mode':'append'});third.write(payload);third.close()
    assert path.read_bytes()==record*3
    fourth=FileDestination(config);fourth.write(payload);fourth.close()
    assert path.read_bytes()==record
    try:fourth.write(payload)
    except OSError:pass
    else:raise AssertionError('Closed destination accepted a publication')
    if os.name=='posix':
        assert path.stat().st_mode&0o777==0o600
        assert path.parent.stat().st_mode&0o777==0o700
        assert path.parent.parent.stat().st_mode&0o777==0o700
    if format=='proto':
        decoded=ExportTraceServiceRequest.FromString(path.read_bytes()[4:]);assert decoded.resource_spans
    else:
        assert len(path.read_text().splitlines())==1
        span=json.loads(path.read_text())['resourceSpans'][0]['scopeSpans'][0]['spans'][0]
        assert len(bytes.fromhex(span['traceId']))==16 and len(bytes.fromhex(span['spanId']))==8
for change in ({'output_directory':' '},{'filename':'../escaped'},{'filename':' padded'},
               {'filename':'..'},{'filename':''},{'format':'json'},{'mode':'truncate'},
               {'output_directory':str(scratch/'work/not-created/../escape')}):
    try:FileDestination({'output_directory':str(scratch/'work/not-created'),'filename':'test.jsonl',**change})
    except ValueError:pass
    else:raise AssertionError('Invalid file sink configuration accepted')
assert not (scratch/'work/not-created').exists() and not (scratch/'work/escape').exists()
if os.name=='posix':
    root=scratch/'work/real';root.mkdir()
    alias=scratch/'work/alias';alias.symlink_to(root,target_is_directory=True)
    base={'output_directory':str(root),'filename':'shared.pb','format':'proto'}
    first=FileDestination(base);first.write(payload)
    second=FileDestination({**base,'output_directory':str(alias)})
    assert first.writer is second.writer
    first.close();second.write(payload);second.close()
    assert (root/'shared.pb').read_bytes()==encode_request(payload,'proto')*2
    target=root/'untouched';target.write_bytes(b'untouched')
    (root/'symlink').symlink_to(target)
    (root/'directory').mkdir();os.mkfifo(root/'fifo')
    for name in ('symlink','directory','fifo'):
        try:FileDestination({**base,'filename':name})
        except OSError:pass
        else:raise AssertionError('Unsafe file type accepted')
    assert target.read_bytes()==b'untouched'

import asyncio,uuid
from standalone import relay,relay_plugin
destinations=[{'output_directory':str(scratch/'work/plugin-output'),'filename':filename,'format':format,
               'service_name':'actual-file-plugin'} for format,filename in [('json_lines','plugin.jsonl'),('proto','plugin.pb')]]
document={'version':1,'components':[{'kind':'observability','config':{'opentelemetry':{'enabled':True,'file_sinks':destinations}}}]}
report=relay_plugin.validate_exact(document)
assert not [item for item in report['config']['diagnostics'] if item['level']=='error'],report
assert not (scratch/'work/plugin-output').exists(),'Validation opened an exporter'
invalid={'version':1,'components':[{'kind':'observability','config':{'opentelemetry':{'enabled':True,'file_sinks':[destinations[0]]*2}}}]}
assert any(item['level']=='error' for item in relay_plugin.validate_exact(invalid)['config']['diagnostics'])
activation=asyncio.run(relay_plugin.initialize(document))
handle=relay.scope.push('file-plugin\n한글','agent',input={'fixture':'private owned input'})
relay.scope.event('file-plugin-mark',data={'fixture':'owned mark'})
relay.scope.pop(handle,output={'fixture':True});relay.subscribers.flush()
asyncio.run(activation.close())
binary=(scratch/'work/plugin-output/plugin.pb').read_bytes()
size=int.from_bytes(binary[:4],'big');assert size==len(binary)-4
message=ExportTraceServiceRequest.FromString(binary[4:])
span=message.resource_spans[0].scope_spans[0].spans[0]
assert span.name=='file-plugin\n한글' and span.trace_id==uuid.UUID(handle.uuid).bytes
assert span.events[0].name=='file-plugin-mark'
lines=(scratch/'work/plugin-output/plugin.jsonl').read_text().splitlines();assert len(lines)==1
assert json.loads(lines[0])['resourceSpans'][0]['scopeSpans'][0]['spans'][0]['name']==span.name
assert not relay_plugin._ACTIVE
print(json.dumps({'native_json':True,'native_proto':True,'private_files':True,'shared_handle':True,
                  'confined_output':True,'unsafe_types_refused':True,'plugin_events':True,'blocked_events':guard.events}),flush=True)
os._exit(0)
'''


class RelayFileTests(fixture.BundledSourceTestCase):
    invoke_worker = fixture.OriginalCommandTests.invoke_worker

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        target = cls.bundle/'app/standalone/test-fixtures'
        target.mkdir(exist_ok=True)
        cls.fixture = target/'relay_files.json'
        cls.fixture.write_bytes(Path(__file__).with_name('fixtures').joinpath('relay_files.json').read_bytes())

    def test_native_json_protobuf_and_actual_shared_private_file_lifecycles(self):
        report = self.invoke_worker(_WORKER.replace('FIXTURE', repr(str(self.fixture))))
        self.assertTrue(all(report[key] for key in ('native_json','native_proto','private_files','shared_handle',
                                                   'confined_output','unsafe_types_refused','plugin_events')))
        self.assertEqual(report['blocked_events'], [])


class WindowsRelayAbiTests(unittest.TestCase):
    def test_handle_relative_unicode_object_layout_and_component_refusals(self):
        from standalone.relay_files_windows import _object, OBJECT_ATTRIBUTES, IO_STATUS_BLOCK, UNICODE_STRING
        descriptor = ctypes.c_void_p(456)
        attributes, unicode, buffer = _object(123,'음성-🦀.pb',descriptor)
        encoded = '음성-🦀.pb'.encode('utf-16-le')
        self.assertEqual(ctypes.string_at(unicode.Buffer,unicode.Length),encoded)
        self.assertEqual(unicode.Length,len(encoded))
        self.assertEqual(unicode.MaximumLength,len(encoded))
        self.assertEqual(attributes.RootDirectory,123)
        self.assertEqual(attributes.SecurityDescriptor,456)
        self.assertEqual(attributes.Attributes,0x40)
        pointer = ctypes.sizeof(ctypes.c_void_p)
        self.assertEqual(ctypes.sizeof(OBJECT_ATTRIBUTES),48 if pointer==8 else 24)
        self.assertEqual(ctypes.sizeof(UNICODE_STRING),16 if pointer==8 else 8)
        self.assertEqual(ctypes.sizeof(IO_STATUS_BLOCK),pointer*2)
        for name in ('','..','a/b','a\\b','a\0b','x'*32768):
            with self.assertRaises(ValueError):_object(123,name,descriptor)


if __name__=='__main__':
    unittest.main()
