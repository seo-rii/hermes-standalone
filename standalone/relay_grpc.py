"""Qualified Relay OTLP/gRPC export through embedded pure Python HTTP/2.

Service paths and metadata contracts follow pinned NeMo Relay 0.10.0 and
the embedded official OpenTelemetry protobuf service definitions. No grpcio
native extension or telemetry process is required.
"""
import asyncio
import math
import re
import ssl
from urllib.parse import urlsplit

from standalone.relay_plugin import _header_destination, _headers, _url


async def _post(url,payload,config,signal):
    from grpclib.client import Channel
    from grpclib.const import Cardinality
    from grpclib.exceptions import GRPCError
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest,ExportTraceServiceResponse
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest,ExportLogsServiceResponse
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest,ExportMetricsServiceResponse
    if signal=='traces':
        service='opentelemetry.proto.collector.trace.v1.TraceService'
        request_type,response_type=ExportTraceServiceRequest,ExportTraceServiceResponse
    elif signal=='logs':
        service='opentelemetry.proto.collector.logs.v1.LogsService'
        request_type,response_type=ExportLogsServiceRequest,ExportLogsServiceResponse
    elif signal=='metrics':
        service='opentelemetry.proto.collector.metrics.v1.MetricsService'
        request_type,response_type=ExportMetricsServiceRequest,ExportMetricsServiceResponse
    else:raise ValueError('Unsupported Relay gRPC signal')
    endpoint=urlsplit(_url(url))
    _header_destination(url,config,signal=signal)
    milliseconds=config.get('timeout_millis',3000)
    if type(milliseconds) not in (int,float) or not math.isfinite(milliseconds) or milliseconds<=0:
        raise ValueError('Relay exporter timeout must be positive and finite')
    metadata={}
    for name,value in _headers(config).items():
        name=name.lower()
        if not re.fullmatch('[0-9a-z_.-]+',name) or name.endswith('-bin') or any(not 32<=ord(char)<127 for char in value):
            raise ValueError('Invalid Relay gRPC ASCII metadata')
        metadata[name]=value
    context=None
    if endpoint.scheme=='https':
        context=ssl.create_default_context()
        context.set_alpn_protocols(['h2'])
    channel=Channel(endpoint.hostname,endpoint.port or (443 if context else 80),ssl=context)
    try:
        async with channel.request('/'+service+'/Export',Cardinality.UNARY_UNARY,request_type,response_type,
                                   timeout=milliseconds/1000,metadata=metadata) as stream:
            await stream.send_message(request_type.FromString(payload),end=True)
            response=await stream.recv_message()
            if response is None:raise OSError('Relay gRPC exporter returned no response')
        return response
    except GRPCError as error:
        raise OSError('Relay gRPC exporter returned '+error.status.name) from None
    except (OSError,asyncio.TimeoutError) as error:
        raise OSError('Relay gRPC exporter transport failed ('+type(error).__name__+')') from None
    finally:
        channel.close()


def post(url,payload,config,signal):
    """Send on an isolated event loop; Relay retains a failed publication."""
    from standalone.relay import _resolve
    # The RPC deadline owns the request lifetime, including configurations over
    # the generic middleware callback's ten-second limit. Do not abandon a live
    # publication in another thread and allow its queue entry to be retried.
    return _resolve(_post(url,payload,config,signal),timeout=None)
