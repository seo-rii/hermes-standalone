"""Real TLS HTTP and HTTP/2 metric export through official protobuf messages."""
import json
import base64
from pathlib import Path
import sys
import unittest
import zlib
from standalone.tests import test_platform_sdks as fixture

WORKER=r'''
import asyncio,base64,socketserver,ssl,threading,time,zlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest,ExportMetricsServiceResponse
from standalone.relay_metrics import MetricKind,MetricValueType,MetricMeasurement,metric_payload,METRIC_SCHEMA
from standalone.relay_metrics_otlp import OtlpMetricsSink,serialize_metrics,validate_metrics
from standalone.relay_types import Event
records=[];http_mode='ok';received=threading.Event()
ports=set();guard.http_ports=ports
class Peer(BaseHTTPRequestHandler):
    def do_POST(self):
        data=self.rfile.read(int(self.headers['Content-Length']))
        records.append((self.path,ExportMetricsServiceRequest.FromString(data),dict(self.headers)))
        self.send_response(500 if http_mode=='fail' else 200)
        self.send_header('Content-Length','0');self.end_headers();received.set()
    def log_message(self,*args):pass
class OwnedServer(ThreadingHTTPServer):
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name,self.server_port='127.0.0.1',self.server_address[1]
server=OwnedServer(('127.0.0.1',0),Peer);server.daemon_threads=True
tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.load_cert_chain(scratch/'cert.pem',scratch/'key.pem')
server.socket=tls.wrap_socket(server.socket,server_side=True)
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();ports.add(server.server_port)
endpoint='https://127.0.0.1:'+str(server.server_port)+'/team/v1/traces?fixture=metrics'
def mark(value=1,name='Counter',kind=MetricKind.Counter,value_type=MetricValueType.U64,**options):
    measurement=MetricMeasurement(name,kind,value_type,value,**options)
    return Event(dict(kind='mark',uuid='00000000-0000-0000-0000-000000000001',parent_uuid=None,name='owned.metric',
        timestamp='2026-01-02T00:00:00Z',data_schema=METRIC_SCHEMA,data=metric_payload([measurement])))
def metric(request):return request.resource_metrics[0].scope_metrics[0].metrics[0]
def value(request):return metric(request).sum.data_points[0].as_int
def settings(temporality='cumulative',interval=3600000):
    return {'endpoints':[
        {'endpoint':endpoint,'service_name':'owned-metrics','resource_attributes':{'fixture':'owned'},
         'instrumentation_scope':'owned.metrics','headers':{'X-Owned':'one'},'timeout_millis':1500}],
         'metrics':{'enabled':True,'temporality':temporality,'export_interval_millis':interval}}
try:
    for temporality in ('cumulative','delta','low_memory'):
        sink=OtlpMetricsSink(settings(temporality));before=len(records)
        sink(mark(3));sink.flush()
        path,message,headers=records[-1]
        assert path=='/team/v1/metrics?fixture=metrics' and headers['X-Owned']=='one'
        assert value(message)==3 and metric(message).sum.is_monotonic
        assert metric(message).sum.aggregation_temporality==(2 if temporality=='cumulative' else 1)
        attrs={item.key:item.value.string_value for item in message.resource_metrics[0].resource.attributes}
        assert attrs['service.name']=='owned-metrics' and attrs['telemetry.sdk.language']=='python'
        assert attrs['telemetry.sdk.name']=='hermes-standalone'
        assert message.resource_metrics[0].scope_metrics[0].scope.name=='owned.metrics'
        http_mode='fail';sink(mark(5))
        try:sink.flush()
        except OSError:pass
        else:raise AssertionError('Metrics HTTP failure was accepted')
        assert len(records)==before+2
        failed=value(records[-1][1]);assert failed==(8 if temporality=='cumulative' else 5)
        assert [row['code'] for row in sink.runtime_diagnostics()]==['otel.metrics_export_failed']
        http_mode='ok';sink.flush()
        assert len(records)==before+(3 if temporality=='cumulative' else 2)
        if temporality=='cumulative':assert value(records[-1][1])==8
        sink(mark(7));sink.flush();assert value(records[-1][1])==(15 if temporality=='cumulative' else 7)
        sink.close();assert not sink._worker.is_alive()
        count=len(records);sink.close();sink(mark(10));sink.flush();assert len(records)==count
    sink=OtlpMetricsSink(settings('delta',20));received.clear();sink(mark(11));assert received.wait(2)
    assert value(records[-1][1])==11;sink.close()
    sink=OtlpMetricsSink(settings('delta'))
    writers=[threading.Thread(target=lambda:[sink(mark(1)) for _ in range(100)]) for _ in range(8)]
    for writer in writers:writer.start()
    for writer in writers:writer.join()
    sink.flush();assert value(records[-1][1])==800;sink.close()
    # Validate every frozen native numeric/typed-array shape through actual official messages.
    def restore(value):
        if isinstance(value,dict):
            if set(value)=={'special_float'}:return float(value['special_float'])
            return {key:restore(item) for key,item in value.items()}
        return [restore(item) for item in value] if isinstance(value,list) else value
    frozen=restore(json.loads(zlib.decompress(base64.b85decode(FROZEN_METRICS_JSON))))
    serialized=0
    for scenario in frozen['scenarios']:
        for cycle in scenario['cycles']:
            for exported in cycle['messages']:
                rows=json.loads(json.dumps(exported['metrics']))
                for row in rows:
                    for point in row['data_points']:point.update(start_time_unix_nano=1,time_unix_nano=2)
                payload=serialize_metrics(rows,{'service.name':'numeric-fixture'}, {'instrumentation_scope':'owned.fixture'})
                decoded=ExportMetricsServiceRequest.FromString(payload)
                for wanted,actual in zip(rows,decoded.resource_metrics[0].scope_metrics[0].metrics):
                    assert actual.name==wanted['name'] and actual.WhichOneof('data')==wanted['type']
                    for expected,point in zip(wanted['data_points'],getattr(actual,wanted['type']).data_points):
                        assert point.start_time_unix_nano==1 and point.time_unix_nano==2
                        def decode(v):
                            k=v.WhichOneof('value')
                            return [decode(x) for x in v.array_value.values] if k=='array_value' else getattr(v,k)
                        assert {item.key:decode(item.value) for item in point.attributes}==expected['attributes']
                        if wanted['type']=='histogram':
                            assert list(point.bucket_counts)==expected['bucket_counts'] and point.sum==expected['sum']
                            assert point.HasField('min') and point.min==expected['min'] and point.HasField('max') and point.max==expected['max']
                        else:assert getattr(point,point.WhichOneof('value'))==expected[point.WhichOneof('value')]
                serialized+=1
    assert serialized==sum(len(cycle['messages']) for scenario in frozen['scenarios'] for cycle in scenario['cycles'])
    before=len(records)
    for change in ({'cardinality_limit':0},{'temporality':'invalid'},{'export_interval_millis':False},{'max_instruments':-1}):
        cfg=settings();cfg['metrics'].update(change)
        try:validate_metrics(cfg)
        except ValueError:pass
        else:raise AssertionError('Invalid metric configuration accepted')
    assert len(records)==before
finally:server.shutdown();server.server_close();thread.join(3)

from grpclib.const import Cardinality,Handler,Status
from grpclib.exceptions import GRPCError
from grpclib.server import Server
grpc_records=[];grpc_mode='ok'
class Service:
    def __mapping__(self):
        return {'/opentelemetry.proto.collector.metrics.v1.MetricsService/Export':
            Handler(self.export,Cardinality.UNARY_UNARY,ExportMetricsServiceRequest,ExportMetricsServiceResponse)}
    async def export(self,stream):
        request=await stream.recv_message();grpc_records.append((request,dict(stream.metadata)))
        if grpc_mode=='fail':raise GRPCError(Status.UNAVAILABLE,'private-metric-canary')
        await stream.send_message(ExportMetricsServiceResponse())
async def grpc_test():
    global grpc_mode
    tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.load_cert_chain(scratch/'cert.pem',scratch/'key.pem');tls.set_alpn_protocols(['h2'])
    server=Server([Service()]);await server.start('127.0.0.1',0,ssl=tls)
    port=server._server.sockets[0].getsockname()[1];ports.add(port)
    cfg=settings('delta');cfg['endpoints'][0].update(endpoint='https://127.0.0.1:'+str(port),transport='grpc')
    sink=OtlpMetricsSink(cfg)
    try:
        sink(mark(2));await asyncio.to_thread(sink.flush)
        assert value(grpc_records[-1][0])==2 and grpc_records[-1][1]['x-owned']=='one'
        grpc_mode='fail';sink(mark(4))
        try:await asyncio.to_thread(sink.flush)
        except OSError as error:assert 'private-metric-canary' not in str(error)
        else:raise AssertionError('Failed MetricsService RPC accepted')
        before=len(grpc_records);grpc_mode='ok';await asyncio.to_thread(sink.flush)
        assert len(grpc_records)==before
        sink(mark(8));await asyncio.to_thread(sink.close)
        assert value(grpc_records[-1][0])==8
    finally:server.close();await server.wait_closed()
asyncio.run(grpc_test())
assert not any(name in sys.modules for name in ('nemo_relay','grpc','cygrpc'))
print(json.dumps({'http':True,'grpc':True,'failed_delta_not_replayed':True,'periodic':True,'concurrent':True,
    'official_serializations':serialized,'blocked_events':guard.events}),flush=True)
os._exit(0)
'''


@unittest.skipIf(sys.version_info<(3,14),'Original Hermes bundled harness uses Python 3.14')
class RelayMetricTransportTests(unittest.TestCase):
    setUpClass = classmethod(fixture.OriginalPlatformSDKTests.setUpClass.__func__)

    def test_actual_tls_http_and_grpc_export_and_lifecycle(self):
        frozen=Path(__file__).with_name('fixtures').joinpath('relay_metrics_export.json').read_text()
        program=WORKER.replace('FROZEN_METRICS_JSON',repr(base64.b85encode(zlib.compress(frozen.encode()))))
        report,_=fixture.OriginalPlatformSDKTests.invoke(self,program)
        self.assertTrue(report['http'] and report['grpc'] and report['concurrent'] and report['periodic'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
