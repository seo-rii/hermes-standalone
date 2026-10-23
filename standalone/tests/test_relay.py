"""Qualified Relay contracts and original Hermes consumers over owned HTTP peers."""

import contextvars
import asyncio
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from standalone import relay_codecs, relay_plugin
from standalone.relay_types import LLMRequest
from standalone.tests import test_entrypoints as entry_tests


class RelayContractTests(unittest.TestCase):
    def test_signal_endpoint_suffix_retains_prefix_query_and_explicit_paths(self):
        from standalone.relay_otlp import _signal_url
        for signal in ('traces','logs','metrics'):
            with self.subTest(signal=signal):
                self.assertEqual(_signal_url('https://peer.invalid',signal),'https://peer.invalid/v1/'+signal)
                self.assertEqual(_signal_url('https://peer.invalid/',signal),
                                 'https://peer.invalid/' if signal=='traces' else 'https://peer.invalid/v1/'+signal)
                self.assertEqual(_signal_url('https://peer.invalid/team/v1/traces?fixture=1#part',signal),
                                 'https://peer.invalid/team/v1/'+signal+'?fixture=1#part')
                for path in ('/v1/logs','/v1/metrics','/team/','/team/v1/traces/','/v1/traces-extra'):
                    url='https://peer.invalid'+path
                    self.assertEqual(_signal_url(url,signal),url)

    def test_headers_match_native_exporter_activation_and_file_rotation_policy(self):
        fixture = json.loads(Path(__file__).with_name('fixtures').joinpath('relay_headers.json').read_text())
        with patch.dict(os.environ, fixture['environment'], clear=True):
            for case in fixture['cases']:
                with self.subTest(endpoint=case['endpoint'], options=case['options']):
                    if case['accepted']:
                        relay_plugin._header_destination(case['endpoint'], case['options'], signal='traces')
                        relay_plugin._headers(case['options'])
                    else:
                        with self.assertRaises(ValueError):
                            relay_plugin._header_destination(case['endpoint'], case['options'], signal='traces')
                            relay_plugin._headers(case['options'])
        with tempfile.TemporaryDirectory(prefix='relay-owned-header-') as directory:
            path = Path(directory)/'header'
            path.write_text(' leading-value\n \t')
            config = {'header_file': {'X-Key': str(path)}}
            self.assertEqual(relay_plugin._headers(config), {'X-Key': ' leading-value'})
            path.write_text('rotated-value\n')
            self.assertEqual(relay_plugin._headers(config), {'X-Key': 'rotated-value'})
            with self.assertRaises(ValueError):
                relay_plugin._headers({**config, 'headers': {'x-key': 'duplicate'}})
            path.write_text(' \n')
            with self.assertRaises(ValueError):relay_plugin._headers(config)
            if os.name == 'posix':
                fifo = Path(directory)/'fifo'
                os.mkfifo(fifo)
                with self.assertRaises(ValueError):relay_plugin._headers({'header_file': {'X-Key': str(fifo)}})

    def test_atof_field_rewriting_retains_collisions_and_failed_http_delivery_remains_retryable(self):
        from standalone.relay_types import Event
        self.assertEqual(relay_plugin._replace_dots({"a.b": 1, "a_b": 2, "nested": [{"x.y": 3}]}),
                         {"a_b": 1, "a_b_2": 2, "nested": [{"x_y": 3}]})
        sink = relay_plugin._AtofSink({"type": "stream", "url": "http://127.0.0.1:1/never-contacted"})
        event = Event({"kind": "mark", "name": "retry-fixture"})
        with patch.object(relay_plugin, "_post", side_effect=OSError("injected transport failure")):
            with self.assertRaises(OSError):
                sink(event)
        sent = []
        with patch.object(relay_plugin, "_post", side_effect=lambda url, content, config, mime: sent.append(content)):
            sink.flush()
            sink.close()
        self.assertEqual([json.loads(content) for content in sent], [{"kind": "mark", "name": "retry-fixture"}])

    def test_sanitizer_failure_clears_observability_fields_and_subscribers_cannot_mutate_events(self):
        from standalone import relay
        seen = []
        relay.subscribers.register("fixture-sanitize-proof", seen.append)
        relay.guardrails.register_mark_sanitize("fixture-refusal", 1, lambda event, fields: (_ for _ in ()).throw(ValueError("private")))
        try:
            contextvars.Context().run(relay.scope.event, "fixture", data={"private": "canary"}, metadata={"private": "canary"})
            relay.subscribers.flush()
        finally:
            relay.guardrails.deregister_mark_sanitize("fixture-refusal")
            relay.subscribers.deregister("fixture-sanitize-proof")
        self.assertIsNone(seen[0].data)
        self.assertIsNone(seen[0].metadata)
        with self.assertRaises(AttributeError):
            seen[0].data = {"replacement": True}

    def test_execution_interceptor_continuation_is_revoked_when_its_owner_returns(self):
        from standalone import relay
        saved, executions = [], []

        async def intercept(context, next_call):
            saved.append(next_call)
            result = await next_call(context.args)
            return relay.ToolExecutionInterceptOutcome(result.result, annotation={"private": "annotation"})

        relay.intercepts.register_tool_execution("fixture-continuation", 1, intercept)
        try:
            async def exercise():
                result = await relay.tools.execute("fixture", {"input": 1},
                    lambda arguments: executions.append(arguments) or relay.ToolExecutionResult({"ok": True}))
                self.assertEqual(result.result, {"ok": True})
                self.assertEqual(result.annotation, {"private": "annotation"})
                with self.assertRaisesRegex(RuntimeError, "no longer active"):
                    await saved[0]({"input": 2})
            contextvars.Context().run(asyncio.run, exercise())
        finally:
            relay.intercepts.deregister_tool_execution("fixture-continuation")
        self.assertEqual(executions, [{"input": 1}])

    def test_original_observability_dataclasses_and_exact_validation_never_activate(self):
        from standalone import relay_observability as obs
        payload = {"version": 1, "components": [obs.ComponentSpec(config=obs.ObservabilityConfig(
            atof=obs.AtofConfig(enabled=True, sinks=[obs.AtofFileSinkConfig(filename="events.jsonl")]),
            opentelemetry=obs.OpenTelemetrySectionConfig(enabled=False))).to_dict()]}
        with patch.object(relay_plugin, "_AtofSink", side_effect=AssertionError("validation activated exporter")):
            report = relay_plugin.validate_exact(payload)
        self.assertFalse([item for item in report["config"]["diagnostics"] if item["level"] == "error"])
        self.assertTrue(report["resolved_config"]["components"][0]["enabled"])
        native = relay_plugin.validate_exact({"version": 1, "components": [{"kind": "adaptive", "enabled": True}]})
        self.assertTrue(any(item["level"] == "error" for item in native["config"]["diagnostics"]))
        denied = relay_plugin.validate_exact({"version": 1, "plugins": {"dynamic": [{"manifest": "never-loaded.toml"}],
                    "policy": {"defaults": {"allowed": False}}}})
        self.assertFalse(denied["dynamic_plugins"][0]["selected"])
        selected = relay_plugin.validate_exact({"version": 1, "plugins": {"dynamic": [{"manifest": "never-loaded.toml"}]}})
        self.assertTrue(any(item["level"] == "error" for item in selected["config"]["diagnostics"]))

    def test_configuration_precedence_concatenates_exporters_and_clears_explicit_signal_endpoints(self):
        lower = {"version": 1, "components": [{"kind": "observability", "enabled": True, "config": {
            "atof": {"sinks": [{"type": "file", "filename": "lower"}]},
            "opentelemetry": {"logs": {"endpoints": [{"endpoint": "https://lower.invalid"}]}}}}]}
        higher = {"components": [{"kind": "observability", "config": {
            "atof": {"sinks": [{"type": "file", "filename": "higher"}]},
            "opentelemetry": {"logs": {"endpoints": []}}}}]}
        result = relay_plugin._merge(lower, higher)
        config = result["components"][0]["config"]
        self.assertEqual([item["filename"] for item in config["atof"]["sinks"]], ["higher", "lower"])
        self.assertEqual(config["opentelemetry"]["logs"]["endpoints"], [])
        self.assertEqual(lower["components"][0]["config"]["atof"]["sinks"][0]["filename"], "lower")
        with tempfile.TemporaryDirectory(prefix="relay layers ") as directory:
            user, system = Path(directory) / "user.toml", Path(directory) / "system.toml"
            user.write_text('version=1\n[[components]]\nkind="observability"\nenabled=true\n')
            system.write_text('version=1\n[[components]]\nkind="observability"\nenabled=false\n')
            with patch.object(relay_plugin, "_config_paths", return_value=[user, system]):
                report = relay_plugin.validate({})
            self.assertEqual(report["config_paths"], [str(user), str(system)])
            self.assertFalse(report["resolved_config"]["components"][0]["enabled"])

    def test_codecs_preserve_provider_extensions_and_apply_only_semantic_edits(self):
        for codec, body in (
            (relay_codecs.OpenAIChatCodec(), {"model": "m", "messages": [{"role": "user", "content": "hi", "future": [1]}],
                                             "temperature": 0.2, "max_completion_tokens": 16, "stop": "END"}),
            (relay_codecs.OpenAIResponsesCodec(), {"model": "m", "input": "hi", "instructions": "system", "max_output_tokens": 16}),
            (relay_codecs.AnthropicMessagesCodec(), {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                                                   "system": [{"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}], "max_tokens": 16}),
        ):
            with self.subTest(codec=type(codec).__name__):
                body["future_parameter"] = {"opaque": [1, 2, 3]}
                request = LLMRequest({"x-fixture": "yes"}, body)
                annotation = codec.decode(request)
                self.assertEqual(codec.encode(annotation, request).content, body)
                annotation.model = "rewritten"
                annotation.params = {**(annotation.params or {}), "max_tokens": 32}
                encoded = codec.encode(annotation, request)
                self.assertEqual(encoded.content["model"], "rewritten")
                self.assertEqual(encoded.content["future_parameter"], body["future_parameter"])
                self.assertEqual(encoded.headers, {"x-fixture": "yes"})
                self.assertEqual(request.content, body)


class RelayTransportPolicyTests(unittest.TestCase):
    def test_actual_http_signal_paths_and_otlp_redirect_refusal(self):
        from standalone.relay_otlp import _signal_url
        records = []

        class Peer(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                records.append(self.path)
                self.rfile.read(int(self.headers.get('Content-Length','0')))
                self.send_response(302 if self.path=='/redirect' else 200)
                if self.path=='/redirect':self.send_header('Location','/must-not-follow')
                self.send_header('Content-Length','0')
                self.end_headers()

            def do_GET(self):
                records.append(self.path)
                self.send_response(200)
                self.send_header('Content-Length','0')
                self.end_headers()

        server=ThreadingHTTPServer(('127.0.0.1',0),Peer)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        origin='http://127.0.0.1:'+str(server.server_port)
        try:
            with patch.dict(os.environ,{'NO_PROXY':'*'},clear=True):
                for suffix,signal in [('', 'traces'),('/', 'traces'),('/team/v1/traces?fixture=1','logs'),
                                      ('/v1/logs','metrics'),('/team/v1/traces','metrics')]:
                    relay_plugin._post(_signal_url(origin+suffix,signal),b'owned-protobuf',{},
                                       'application/x-protobuf',no_redirect=True)
                with self.assertRaisesRegex(OSError,'HTTP 302'):
                    relay_plugin._post(origin+'/redirect',b'owned-protobuf',{},
                                       'application/x-protobuf',no_redirect=True)
        finally:
            server.shutdown();server.server_close();thread.join(3)
        self.assertEqual(records,['/v1/traces','/','/team/v1/logs?fixture=1','/v1/logs',
                                  '/team/v1/metrics','/redirect'])


class _RelayPeer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def emit(self, body, content_type="application/json"):
        content = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path != "/captured":
            self.send_error(404)
            return
        self.emit(self.server.captured)

    def do_POST(self):
        content = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.server.captured.append({"path": self.path, "content_type": self.headers.get("Content-Type"), "hex": content.hex()})
        if self.path == "/model":
            self.emit({"id": "wire-response", "model": "fixture-model", "choices": [{"message": {"role": "assistant", "content": "original wire response"},
                       "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})
        elif self.path == "/stream":
            chunks = [{"choices": [{"delta": {"content": "한글 🦀"}, "finish_reason": None}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
            self.emit(b"".join(b"data: " + json.dumps(item, ensure_ascii=False).encode() + b"\n\n" for item in chunks) + b"data: [DONE]\n\n", "text/event-stream")
        elif self.path == "/tool":
            self.emit({"ok": True, "arguments": json.loads(content)})
        else:
            self.emit({})


_WORKER = r'''
import asyncio, contextvars, json, os, pathlib, socket, sqlite3, sys, urllib.request
from contextlib import closing
context, scratch, endpoint = json.loads(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
source = pathlib.Path(context["root"])/"app"
sys.path[:0] = [str(source),str(source.parent/"vendor")]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
from standalone.agent_probe_worker import ProbeGuard
class Guard(ProbeGuard):
    def __call__(self,event,args):
        if event in ("socket.connect","socket.getaddrinfo"):
            address=args[1] if event=="socket.connect" else args[:2]
            if isinstance(address,tuple) and address[0]=="127.0.0.1" and int(address[1])==int(endpoint.rsplit(":",1)[1]):return
        if event=="subprocess.Popen" and args[1]==["uname","-p"]:self.deny(event,"optional_uname",recoverable=True)
        return super().__call__(event,args)
guard=Guard(source,source.parent/"vendor",scratch)
sys.addaudithook(guard)
from agent import relay_runtime,relay_llm,relay_tools
from hermes_platform.relay import relay_api
from hermes_cli.observability import relay_shared_metrics as collection
home=pathlib.Path(os.environ["HERMES_HOME"])
(home/"config.yaml").write_text("telemetry:\n  shared_metrics:\n    enabled: false\n    send: false\n")
assert not collection.enabled()
collection.observe_lifecycle("pre_llm_call",session_id="opted-out",task_id="private")
assert not (home/"telemetry").exists()
relay=relay_api()
assert relay.__name__=="standalone.relay" and "nemo_relay" not in sys.modules
settings={"version":1,"components":[relay.observability.ComponentSpec(config=relay.observability.ObservabilityConfig(
    atof=relay.observability.AtofConfig(enabled=True,sinks=[
        relay.observability.AtofFileSinkConfig(output_directory=str(scratch),filename="events.jsonl",mode="overwrite"),
        relay.observability.AtofStreamSinkConfig(url=endpoint+"/atof")]),
    atif=relay.observability.AtifConfig(enabled=True,output_directory=str(scratch),filename_template="atif-{session_id}.json",
        storage=[relay.observability.HttpStorageConfig(endpoint=endpoint+"/atif")]),
    opentelemetry=relay.observability.OpenTelemetrySectionConfig(enabled=True,endpoints=[
        relay.observability.OpenTelemetryEndpointConfig(type="full",endpoint=endpoint)],
        logs=relay.observability.OpenTelemetryLogSectionConfig(enabled=True)))) .to_dict()]}
from hermes_cli.relay_plugin_migrate import dumps_toml,migrate_profile_relay_env
plugins=scratch/"plugins.toml";plugins.write_text(dumps_toml(settings));os.environ["HERMES_NEMO_RELAY_PLUGINS_TOML"]=str(plugins)
sources=relay_runtime.resolve_plugin_sources()
assert sources.enabled and not sources.errors and str(plugins) in sources.config_paths,sources
lease=relay_runtime.SESSION_COORDINATOR.acquire_conversation(profile_key=relay_runtime.current_profile_key(),session_id="fixture-session",platform="cli")
turn=relay_runtime.SESSION_COORDINATOR.begin_turn(lease,turn_id="fixture-turn",task_id="fixture-task")
assert lease.host.managed_execution_enabled()
def post(path,value):
    request=urllib.request.Request(endpoint+path,data=json.dumps(value).encode(),headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(request,timeout=10) as response:return json.load(response)
requests=[]
def provider(request):
    requests.append(request)
    return post("/model",request)
response=relay_llm.execute({"model":"fixture-model","messages":[{"role":"user","content":"local fixture"}],
    "extra_headers":{"Authorization":"fixture-provider-secret"},"timeout":12,"future_parameter":{"opaque":True}},provider,
    name="fixture-provider",model_name="fixture-model",session_id="fixture-session",metadata={"api_mode":"chat_completions"})
assert response["choices"][0]["message"]["content"]=="original wire response"
assert requests[0]["timeout"]==12 and requests[0]["future_parameter"]=={"opaque":True}
assert requests[0]["extra_headers"]["Authorization"]=="fixture-provider-secret"
observed=[]
relay.intercepts.register_tool_request("fixture-path",1,False,lambda name,args:{**args,"approved":True})
try:
    value,args=relay_tools.execute("terminal",{"private":"tool fixture"},lambda arguments:observed.append(arguments) or post("/tool",arguments),
        session_id="fixture-session",tool_call_id="fixture-call")
    assert args["approved"] and len(observed)==1 and value["ok"]
finally:relay.intercepts.deregister_tool_request("fixture-path")
def provider_stream(request):
    request=urllib.request.Request(endpoint+"/stream",data=json.dumps(request).encode(),method="POST")
    with urllib.request.urlopen(request,timeout=10) as response:
        for line in response:
            if line.startswith(b"data: ") and line.strip()!=b"data: [DONE]":yield json.loads(line[6:])
stream=relay_llm.stream({"model":"fixture-model","messages":[{"role":"user","content":"stream"}]},provider_stream,
    name="fixture-provider",model_name="fixture-model",session_id="fixture-session",metadata={"api_mode":"chat_completions"},
    finalizer=lambda:{"model":"fixture-model","choices":[{"message":{"role":"assistant","content":"한글 🦀"},"finish_reason":"stop"}]})
chunks=list(stream);assert len(chunks)==2 and chunks[0]["choices"][0]["delta"]["content"]=="한글 🦀"
from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.observability.shared_metrics_subscriber import SharedMetricsSubscriber
from hermes_cli.observability import shared_metrics_contract as contract
store=SharedMetricsStore(database_path=scratch/"metrics.sqlite3",outbox_directory=scratch/"outbox")
subscriber=SharedMetricsSubscriber(store,"fixture",runtime_id=lease.host.runtime_id)
relay.subscribers.register("fixture-bounded-metrics",subscriber)
metadata={contract.SCHEMA_KEY:contract.SCHEMA_VERSION,relay_runtime.RUNTIME_INSTANCE_KEY:lease.host.runtime_id}
relay.scope.event(contract.CLIENT_ACTIVE_MARK,handle=turn.handle,data={},metadata=metadata)
relay.scope.event(contract.CLIENT_ACTIVE_MARK,handle=turn.handle,data={"prompt":"DO_NOT_EXPORT"},metadata=metadata)
relay.scope.event(contract.CLIENT_ACTIVE_MARK,handle=turn.handle,data={},metadata={**metadata,"unknown":"DO_NOT_EXPORT"})
relay.subscribers.flush();subscriber.deactivate();relay.subscribers.deregister("fixture-bounded-metrics")
with closing(sqlite3.connect(store.database_path)) as db:rows=db.execute("SELECT metric_name,value,dimensions_json FROM counter_aggregates").fetchall()
assert rows==[(contract.CLIENT_ACTIVE_METRIC,1,"{}")],rows
assert b"DO_NOT_EXPORT" not in store.database_path.read_bytes()
relay.scope.event("fixture-log",handle=turn.handle,data={"safe":"event"})
relay_runtime.SESSION_COORDINATOR.end_turn(turn,outcome="success")
relay_runtime.SESSION_COORDINATOR.finalize_conversation(profile_key=relay_runtime.current_profile_key(),session_id="fixture-session")
relay_runtime._reset_for_tests()
events=[json.loads(line) for line in (scratch/"events.jsonl").read_text().splitlines()]
assert any(event["name"]=="hermes.session" and event.get("scope_category")=="end" for event in events)
assert "fixture-provider-secret" not in (scratch/"events.jsonl").read_text()
assert not list(scratch.glob("atif-*.json")), "Successful remote ATIF storage skips local output"
with urllib.request.urlopen(endpoint+"/captured",timeout=10) as response:captured=json.load(response)
trajectories=[json.loads(bytes.fromhex(record['hex'])) for record in captured if record['path']=='/atif']
assert trajectories and any(step['source']=='agent' for trajectory in trajectories for step in trajectory['steps'])
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
traces=[];logs=[]
for record in captured:
    if record["path"]=="/v1/traces":
        message=ExportTraceServiceRequest.FromString(bytes.fromhex(record["hex"]))
        traces.extend(span for resource in message.resource_spans for scope in resource.scope_spans for span in scope.spans)
    elif record["path"]=="/v1/logs":
        message=ExportLogsServiceRequest.FromString(bytes.fromhex(record["hex"]))
        logs.extend(log for resource in message.resource_logs for scope in resource.scope_logs for log in scope.log_records)
assert any(span.name=="hermes.session" and len(span.trace_id)==16 and len(span.span_id)==8 for span in traces)
assert any(log.body.string_value=="fixture-log" for log in logs)
assert all("fixture-provider-secret" not in str(span) for span in traces)
migration=home/"migration";migration.mkdir();(migration/".env").write_text("HERMES_NEMO_RELAY_ATOF_ENABLED=true\nHERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY="+str(migration)+"\n")
result=migrate_profile_relay_env(migration);assert result.migrated and not result.validation_error,result
assert "enabled = false" in result.toml_path.read_text()
assert all(event.get("recoverable_capability_probe") for event in guard.events),guard.events
print(json.dumps({"ok":True,"events":len(events),"traces":len(traces),"logs":len(logs)}),flush=True)
os._exit(0)
'''


class OriginalRelayConsumerTests(entry_tests.BundledSourceTestCase):
    def test_original_managed_calls_streams_metrics_migration_and_actual_http_exports(self):
        with tempfile.TemporaryDirectory(prefix="relay-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _RelayPeer)
            server.captured, server.daemon_threads = [], True
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            endpoint = f"http://127.0.0.1:{server.server_port}"
            environment = entry_tests.support._load_helper("agent_probe_worker").fixture_environment(scratch)
            environment.update(PATH="", XDG_CONFIG_HOME=str(scratch / "home/config"))
            Path(environment["HERMES_HOME"]).mkdir()
            try:
                result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", _WORKER,
                                         json.dumps(self.context), str(scratch), endpoint], cwd=scratch / "work",
                                        env=environment, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                report = json.loads(result.stdout)
                self.assertTrue(report["ok"])
                self.assertGreater(report["traces"], 0)
                self.assertTrue({"/model", "/tool", "/stream", "/atof", "/atif", "/v1/traces", "/v1/logs"}
                                <= {record["path"] for record in server.captured})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
