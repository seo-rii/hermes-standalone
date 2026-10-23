"""Real TLS HTTP/2 OTLP RPCs with official service types and private peers."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone.tests import test_platform_sdks as fixture


_WORKER=r'''
import asyncio,os,ssl
from grpclib.const import Cardinality,Handler,Status
from grpclib.exceptions import GRPCError
from grpclib.server import Server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest,ExportTraceServiceResponse
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest,ExportLogsServiceResponse
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans,ScopeSpans,Span
from standalone.relay_grpc import post
ports=set();guard.http_ports=ports
records=[];mode='ok'
class Service:
    def __mapping__(self):
        return {
            '/opentelemetry.proto.collector.trace.v1.TraceService/Export':Handler(self.trace,Cardinality.UNARY_UNARY,ExportTraceServiceRequest,ExportTraceServiceResponse),
            '/opentelemetry.proto.collector.logs.v1.LogsService/Export':Handler(self.logs,Cardinality.UNARY_UNARY,ExportLogsServiceRequest,ExportLogsServiceResponse)}
    async def trace(self,stream):
        request=await stream.recv_message();records.append(('traces',request,dict(stream.metadata)))
        if mode=='fail':raise GRPCError(Status.UNAVAILABLE,'private server details must stay private')
        if mode=='slow':await asyncio.sleep(.3)
        await stream.send_message(ExportTraceServiceResponse())
    async def logs(self,stream):
        request=await stream.recv_message();records.append(('logs',request,dict(stream.metadata)))
        await stream.send_message(ExportLogsServiceResponse())
async def exercise():
    global mode
    tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.load_cert_chain(scratch/'cert.pem',scratch/'key.pem');tls.set_alpn_protocols(['h2'])
    server=Server([Service()]);await server.start('127.0.0.1',0,ssl=tls)
    port=server._server.sockets[0].getsockname()[1];ports.add(port);endpoint='https://127.0.0.1:'+str(port)
    token_file=scratch/'export-token';token_file.write_text('fixture-header-one\n');token_file.chmod(0o600)
    config={'timeout_millis':1500,'headers':{'X-Fixture':'fixed'},'header_env':{'Authorization':'RELAY_GRPC_FIXTURE_TOKEN'},'header_file':{'X-Rotating':str(token_file)}}
    os.environ['RELAY_GRPC_FIXTURE_TOKEN']='Bearer fixture-scope-one'
    span=Span(trace_id=bytes(range(16)),span_id=bytes(range(8)),name='grpc-owned-span',start_time_unix_nano=123,end_time_unix_nano=456)
    message=ExportTraceServiceRequest(resource_spans=[ResourceSpans(scope_spans=[ScopeSpans(spans=[span])])])
    try:
        await asyncio.to_thread(post,endpoint,message.SerializeToString(),config,'traces')
        token_file.write_text('fixture-header-two\n');os.environ['RELAY_GRPC_FIXTURE_TOKEN']='Bearer fixture-scope-two'
        await asyncio.to_thread(post,endpoint,ExportLogsServiceRequest().SerializeToString(),config,'logs')
        assert records[0][1]==message
        assert records[0][2]['authorization']=='Bearer fixture-scope-one'
        assert records[1][2]['authorization']=='Bearer fixture-scope-two'
        assert records[0][2]['x-rotating']=='fixture-header-one' and records[1][2]['x-rotating']=='fixture-header-two'
        mode='fail'
        try:await asyncio.to_thread(post,endpoint,message.SerializeToString(),config,'traces')
        except OSError as error:
            assert 'UNAVAILABLE' in str(error) and 'private server details' not in str(error)
        else:raise AssertionError('Failed gRPC status was accepted')
        mode='slow'
        try:await asyncio.to_thread(post,endpoint,b'',{**config,'timeout_millis':20},'traces')
        except OSError:pass
        else:raise AssertionError('gRPC deadline was ignored')
        mode='ok';before=len(records)
        for invalid in ({'headers':{'x-key-bin':'raw-ascii'}},{'headers':{'x-key':'한글'}},{'timeout_millis':float('nan')}):
            try:await asyncio.to_thread(post,endpoint,b'',invalid,'traces')
            except ValueError:pass
            else:raise AssertionError('Invalid gRPC metadata or deadline reached the peer')
        assert len(records)==before
        for url,signal,options in [('http://remote.invalid','traces',{}),
                                    ('http://203.0.113.1','logs',config)]:
            try:await asyncio.to_thread(post,url,b'',options,signal)
            except ValueError:pass
            else:raise AssertionError('Native protected endpoint policy was bypassed')
        assert len(records)==before
        await asyncio.to_thread(post,endpoint,message.SerializeToString(),config,'traces')

        from standalone import relay,relay_observability as obs,relay_plugin
        from standalone.relay_otlp import OtlpSink
        from standalone.relay_types import Event
        settings={'version':1,'components':[obs.ComponentSpec(config=obs.ObservabilityConfig(
            opentelemetry=obs.OpenTelemetrySectionConfig(enabled=True,endpoints=[
                obs.OpenTelemetryEndpointConfig(type='full',endpoint=endpoint,transport='grpc',service_name='owned-grpc',
                    resource_attributes={'fixture':'actual-sink'})],
                logs=obs.OpenTelemetryLogSectionConfig(enabled=True)))).to_dict()]}
        report=relay_plugin.validate_exact(settings)
        assert not [item for item in report['config']['diagnostics'] if item['level']=='error'],report
        sink=OtlpSink(settings['components'][0]['config']['opentelemetry'])
        start={'kind':'scope','scope_category':'start','uuid':'00000000-0000-0000-0000-000000000001',
               'name':'retry-owned-span','timestamp':'2026-01-01T00:00:00Z','category':'function'}
        await asyncio.to_thread(sink,Event(start))
        mode='fail';before=len(records)
        try:await asyncio.to_thread(sink,Event({**start,'scope_category':'end','timestamp':'2026-01-01T00:00:01Z'}))
        except OSError:pass
        else:raise AssertionError('Failed publication was accepted')
        assert len(sink.pending)==1 and len(records)==before+1
        failed_message=records[-1][1]
        mode='ok';await asyncio.to_thread(sink.flush)
        assert not sink.pending and records[-1][1]==failed_message and len(records)==before+2
        await asyncio.to_thread(sink.close)

        before=len(records)
        activation=await relay_plugin.initialize(settings)
        def sanitize(event,fields):return {**fields,'data':None}
        for kind in ('scope_sanitize_start','scope_sanitize_end','mark_sanitize'):
            getattr(relay.guardrails,'register_'+kind)('grpc-owned-sanitizer',1,sanitize)
        try:
            root=relay.scope.push('grpc-owned-agent','agent',input={'private':'grpc-private-canary'})
            child=relay.scope.push('grpc-owned-child','function',input={'private':'grpc-private-canary'})
            relay.scope.event('grpc-owned-log',data={'private':'grpc-private-canary'},severity='warning')
            relay.scope.pop(child,output={'private':'grpc-private-canary'})
            relay.scope.pop(root)
            await relay.subscribers.flush_async()
        finally:
            await activation.close()
            for kind in ('scope_sanitize_start','scope_sanitize_end','mark_sanitize'):
                getattr(relay.guardrails,'deregister_'+kind)('grpc-owned-sanitizer')
        received=records[before:]
        assert len(received)==3 and not relay_plugin._ACTIVE
        spans={span.name:span for signal,request,_ in received if signal=='traces'
               for resource in request.resource_spans for scope in resource.scope_spans for span in scope.spans}
        import uuid
        assert spans['grpc-owned-agent'].trace_id==uuid.UUID(root.uuid).bytes
        assert spans['grpc-owned-child'].trace_id==uuid.UUID(root.uuid).bytes
        assert spans['grpc-owned-child'].parent_span_id==uuid.UUID(root.uuid).bytes[-8:]
        assert spans['grpc-owned-agent'].parent_span_id==b''
        logged=next(request.resource_logs[0].scope_logs[0].log_records[0] for signal,request,_ in received if signal=='logs')
        assert logged.body.string_value=='grpc-owned-log' and logged.severity_number==13
        for signal,request,_ in received:
            assert b'grpc-private-canary' not in request.SerializeToString()
            resource=(request.resource_spans if signal=='traces' else request.resource_logs)[0].resource
            attributes={item.key:item.value.string_value for item in resource.attributes}
            assert attributes['service.name']=='owned-grpc' and attributes['fixture']=='actual-sink'
    finally:server.close();await server.wait_closed()
    assert not any(name in sys.modules for name in ('grpc','cygrpc','nemo_relay'))
    return {'grpc_traces':True,'grpc_logs':True,'rotating_headers':True,'failed_status_refused':True,'deadline':True,
            'plugin_activation':True,'scope_parentage':True,'sanitized_wire':True,'retained_retry':True,'blocked_events':guard.events}
print(json.dumps(asyncio.run(exercise())),flush=True)
os._exit(0)
'''


@unittest.skipIf(sys.version_info<(3,14),'Hermes original consumer integration requires Python 3.14')
class RelayGrpcTests(unittest.TestCase):
    setUpClass=classmethod(fixture.OriginalPlatformSDKTests.setUpClass.__func__)

    def test_real_tls_http2_plugin_events_scope_parentage_sanitization_retry_and_deadline(self):
        report,_=fixture.OriginalPlatformSDKTests.invoke(self,_WORKER)
        self.assertTrue(all(report[name] for name in ('grpc_traces','grpc_logs','rotating_headers','failed_status_refused','deadline',
                                                   'plugin_activation','scope_parentage','sanitized_wire','retained_retry')))


if __name__=='__main__':
    unittest.main()
