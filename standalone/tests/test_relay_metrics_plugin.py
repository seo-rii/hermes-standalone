"""Actual qualified Relay plugin scopes and metrics over owned TLS collectors."""
import json
from pathlib import Path
import sys
import unittest

from standalone.tests import test_platform_sdks as fixture


_WORKER = r'''
import asyncio,socketserver,ssl,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from grpclib.const import Cardinality,Handler,Status
from grpclib.exceptions import GRPCError
from grpclib.server import Server
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest,ExportMetricsServiceResponse
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest,ExportTraceServiceResponse
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest,ExportLogsServiceResponse
from standalone import relay,relay_observability as obs,relay_plugin
transport=TRANSPORT_NAME
expected=json.loads(EXPECTED_NATIVE_JSON)
records=[];mode='ok';ports=set();guard.http_ports=ports
messages={'traces':(ExportTraceServiceRequest,ExportTraceServiceResponse),
          'logs':(ExportLogsServiceRequest,ExportLogsServiceResponse),
          'metrics':(ExportMetricsServiceRequest,ExportMetricsServiceResponse)}
class Peer(BaseHTTPRequestHandler):
    def do_POST(self):
        signal=self.path.rsplit('/',1)[-1]
        body=self.rfile.read(int(self.headers['Content-Length']))
        records.append((signal,messages[signal][0].FromString(body)))
        self.send_response(500 if mode=='fail' and signal=='metrics' else 200)
        self.send_header('Content-Length','0');self.end_headers()
    def log_message(self,*args):pass
class OwnedServer(ThreadingHTTPServer):
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name,self.server_port='127.0.0.1',self.server_address[1]
class Service:
    def __mapping__(self):
        return {'/opentelemetry.proto.collector.'+('trace' if signal=='traces' else signal)+'.v1.'+service+'/Export':
            Handler(callback,Cardinality.UNARY_UNARY,*messages[signal])
            for signal,service,callback in [('traces','TraceService',self.trace),('logs','LogsService',self.log),('metrics','MetricsService',self.metric)]}
    async def receive(self,stream,signal):
        records.append((signal,await stream.recv_message()))
        if mode=='fail' and signal=='metrics':raise GRPCError(Status.UNAVAILABLE,'private-metric-server-detail')
        await stream.send_message(messages[signal][1]())
    async def trace(self,stream):await self.receive(stream,'traces')
    async def log(self,stream):await self.receive(stream,'logs')
    async def metric(self,stream):await self.receive(stream,'metrics')
def measurements(value,private=False):
    return [relay.MetricMeasurement('Owned.Counter',relay.MetricKind.Counter,relay.MetricValueType.U64,value,
        unit='item',description='first descriptor',attributes={'private':'metric-private-canary'} if private else {'typed':[1,2]})]
def points(request):
    return [point for resource in request.resource_metrics for scope in resource.scope_metrics for metric in scope.metrics for point in metric.sum.data_points]
def any_value(value):
    field=value.WhichOneof('value')
    return [any_value(item) for item in value.array_value.values] if field=='array_value' else getattr(value,field)
async def normal(endpoint):
    before=len(records)
    config=obs.ObservabilityConfig(opentelemetry=obs.OpenTelemetrySectionConfig(enabled=True,endpoints=[
        obs.OpenTelemetryEndpointConfig(type='full',endpoint=endpoint,transport=transport,service_name='owned-metric-plugin',
            promote_resource_metadata_prefixes=['customer.'],headers={'X-Owned':'plugin'})],
        logs=obs.OpenTelemetryLogSectionConfig(enabled=True),
        metrics=obs.OpenTelemetryMetricSectionConfig(enabled=True,temporality='delta',export_interval_millis=3600000)))
    activation=await relay_plugin.initialize({'version':1,'components':[obs.ComponentSpec(config=config)]})
    assert activation.is_active
    def privacy(event,fields):
        return {**fields,'data':None} if event.name=='private.metric' else fields
    relay.guardrails.register_mark_sanitize('metric-owned-privacy',1,privacy)
    try:
        root=relay.scope.push('metric-owned-agent',relay.ScopeType.Agent,metadata={'customer.id':'owned-tenant'})
        child=relay.scope.push('metric-owned-child',relay.ScopeType.Function,metadata={'customer.id':'ignored-child'})
        relay.scope.metric('owned.metric',measurements(3))
        relay.scope.metric('owned.metric',measurements(4))
        relay.scope.metric('private.metric',measurements(99,True))
        relay.scope.event('owned.log',data={'body':'visible'})
        relay.scope.pop(child);relay.scope.pop(root)
        await relay.subscribers.flush_async()
        assert not [request for signal,request in records[before:] if signal=='metrics']
        await activation.close()
    finally:
        relay.guardrails.deregister_mark_sanitize('metric-owned-privacy')
    assert not activation.is_active and not relay_plugin._ACTIVE
    received=records[before:]
    assert [signal for signal,_ in received].count('traces')==2
    assert [signal for signal,_ in received].count('logs')==1
    assert [signal for signal,_ in received].count('metrics')==1
    traces=[span for signal,request in received if signal=='traces' for resource in request.resource_spans
            for scope in resource.scope_spans for span in scope.spans]
    assert {span.name for span in traces}=={'metric-owned-agent','metric-owned-child'}
    assert [event.name for span in traces for event in span.events]==['owned.log']
    logs=[row for signal,request in received if signal=='logs' for resource in request.resource_logs
          for scope in resource.scope_logs for row in scope.log_records]
    assert [row.body.string_value for row in logs]==['owned.log']
    request=next(request for signal,request in received if signal=='metrics')
    point=points(request)[0]
    assert point.as_int==7 and {item.key:any_value(item.value) for item in point.attributes}=={'typed':[1,2]}
    attrs={item.key:any_value(item.value) for item in request.resource_metrics[0].resource.attributes}
    assert attrs['customer.id']=='owned-tenant' and attrs['service.name']=='owned-metric-plugin'
    for _,request in received:assert b'metric-private-canary' not in request.SerializeToString()
    diagnostics=activation.report['config']['runtime_diagnostics']
    assert any(row['code']=='otel.metric_mark_invalid' and row['count']==1 and row['field']=='opentelemetry.metrics.endpoints[0].endpoint' for row in diagnostics)
    return True
async def historical(endpoint):
    global mode
    # Signal-specific endpoint paths are explicit; gRPC uses its service path.
    target=endpoint.rsplit('/v1/traces',1)[0]+'/v1/metrics' if transport=='http_binary' else endpoint
    settings={'version':1,'components':[{'kind':'observability','enabled':True,'config':{'opentelemetry':{
        'enabled':True,'metrics':{'enabled':True,'temporality':'delta','export_interval_millis':3600000,
        'endpoints':[{'endpoint':target,'transport':transport,'timeout_millis':1500}]}}}}]}
    before=len(records)
    activation=await relay_plugin.initialize(settings)
    relay.scope.metric('owned.failed',measurements(3))
    await relay.subscribers.flush_async();mode='fail'
    try:await activation.close()
    except RuntimeError as error:first=str(error)
    else:raise AssertionError('Historical delivery failure was not reported')
    assert activation.is_active==expected['active_after_error']
    assert 'otel.metrics_export_failed (1)' in first and 'private-metric-server-detail' not in first
    try:await activation.close()
    except RuntimeError as error:assert str(error)==first
    else:raise AssertionError('Historical close error was forgotten')
    assert activation.is_active==expected['active_after_second_close'] and not relay_plugin._ACTIVE
    diagnostic=next(row for row in activation.report['config']['runtime_diagnostics'] if row['code']=='otel.metrics_export_failed')
    assert diagnostic['count']==1 and diagnostic['field']=='opentelemetry.metrics.endpoints[0].endpoint'
    mode='ok';replacement=await relay_plugin.initialize(settings)
    assert replacement.is_active==expected['reactivated']
    relay.scope.metric('owned.reactivated',measurements(7));await relay.subscribers.flush_async()
    await replacement.close()
    assert replacement.is_active==expected['replacement_active_after_close']
    wire=[points(request)[0].as_int for signal,request in records[before:] if signal=='metrics']
    assert wire==expected['wire_values'],wire
    return True
async def exercise():
    if transport=='grpc':
        tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.load_cert_chain(scratch/'cert.pem',scratch/'key.pem');tls.set_alpn_protocols(['h2'])
        server=Server([Service()]);await server.start('127.0.0.1',0,ssl=tls)
        port=server._server.sockets[0].getsockname()[1];ports.add(port)
        endpoint='https://127.0.0.1:'+str(port)
        try:result={'scopes_and_signals':await normal(endpoint),'delivery_cleanup':await historical(endpoint)}
        finally:server.close();await server.wait_closed()
    else:
        server=OwnedServer(('127.0.0.1',0),Peer);server.daemon_threads=True
        tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.load_cert_chain(scratch/'cert.pem',scratch/'key.pem')
        server.socket=tls.wrap_socket(server.socket,server_side=True)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();ports.add(server.server_port)
        endpoint='https://127.0.0.1:'+str(server.server_port)+'/v1/traces'
        try:result={'scopes_and_signals':await normal(endpoint),'delivery_cleanup':await historical(endpoint)}
        finally:await asyncio.to_thread(server.shutdown);server.server_close();thread.join(3)
    assert not any(name in sys.modules for name in ('nemo_relay','grpc','cygrpc'))
    result['blocked_events']=guard.events
    return result
print(json.dumps(asyncio.run(exercise())),flush=True)
os._exit(0)
'''


@unittest.skipIf(sys.version_info < (3, 14), 'Original Hermes plugin integration uses Python 3.14')
class RelayMetricPluginTests(unittest.TestCase):
    setUpClass = classmethod(fixture.OriginalPlatformSDKTests.setUpClass.__func__)

    def invoke(self, transport):
        native = json.loads(Path(__file__).with_name('fixtures').joinpath('relay_metrics_plugin_failure.json').read_text())
        expected = {key:native[key] for key in ('active_after_error','active_after_second_close','reactivated','replacement_active_after_close')}
        expected['wire_values'] = [rows[0]['data_points'][0]['as_int'] for rows in native['messages']]
        program = _WORKER.replace('TRANSPORT_NAME', repr(transport)).replace('EXPECTED_NATIVE_JSON', repr(json.dumps(expected)))
        report, _ = fixture.OriginalPlatformSDKTests.invoke(self, program)
        self.assertTrue(report['scopes_and_signals'] and report['delivery_cleanup'])

    def test_actual_http_plugin_metric_scopes_privacy_reserved_exclusion_and_historical_cleanup(self):
        self.invoke('http_binary')

    def test_actual_grpc_plugin_metric_scopes_privacy_reserved_exclusion_and_historical_cleanup(self):
        self.invoke('grpc')


if __name__ == '__main__':
    unittest.main()
