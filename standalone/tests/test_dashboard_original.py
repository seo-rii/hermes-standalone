"""Import and exercise original dashboard handlers in the verified runtime."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[2]

WORKER = r'''
import asyncio, contextlib, io, json, pathlib, sys, traceback
source, vendor, scratch = map(pathlib.Path, sys.argv[1:])
sys.path[:0] = [str(source), str(vendor)]
from standalone.runtime import initialize_bundle
initialize_bundle({"root": str(source.parent), "artifact": str(scratch / "fixture.py"),
                   "bundle_sha256": source.parent.name, "manifest": {"project": "hermes-standalone"}})
blocked = []
def audit(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system", "os.exec"):
        blocked.append(event)
        raise PermissionError("Dashboard fixture disallows external operations")
sys.addaudithook(audit)
out, err = io.StringIO(), io.StringIO()
report = {"ok": False, "blocked": blocked}
try:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        from hermes_cli import web_models as models
        from standalone.web_schema import RequestModel, ValidationError
        report["models"] = len([obj for obj in vars(models).values()
                                if isinstance(obj, type) and issubclass(obj, RequestModel) and obj is not RequestModel])
        data = models.MoaConfigPayload.model_validate({"reference_timeout": "12.5", "reference_models": [{"model": "fixture"}]})
        assert data.reference_timeout == 12.5 and data.reference_models[0].model == "fixture"
        for bad in (True, 0, -1, "nan", "inf"):
            try: models.MoaConfigPayload(reference_timeout=bad)
            except ValidationError: pass
            else: raise AssertionError("Invalid reference timeout accepted")
        from tui_gateway import contracts
        from tui_gateway.contracts.registry import validate_params, canonical_param_problems
        from tui_gateway.contracts.connectors import ConnectorsConnectParams, ConnectorsListParams, ConnectorToolsResult
        from tui_gateway.contracts.canonical_sessions import CanonicalSubmitResult
        from tui_gateway.contracts.common import OpenModel, TranscriptMessage
        from types import SimpleNamespace
        for table in (contracts.METHODS, contracts.CANONICAL_METHODS, contracts.SERVER_REQUESTS):
            for contract in table.values():
                contract.params._fields()
                contract.result._fields()
        report["contracts"] = len(contracts.METHODS) + len(contracts.CANONICAL_METHODS)
        owner = {"type": "session", "session_id": "fixture", "unexpected": True}
        _, refusal = validate_params(contracts.METHODS["connectors.list"], {"owner": owner})
        assert "owner.session.unexpected" in refusal, refusal
        for invalid in ({"owner": {"type": "invalid"}}, {"owner": {"type": "session"}}):
            try: ConnectorsListParams.model_validate(invalid)
            except ValidationError: pass
            else: raise AssertionError("Invalid discriminated owner accepted")
        for invalid in ({"owner": {"type": "account"}, "connectors": ["bad/slug"]},
                        {"owner": {"type": "account"}, "connectors": ["one", "two"], "alias": "named"}):
            try: ConnectorsConnectParams.model_validate(invalid)
            except ValidationError: pass
            else: raise AssertionError("Invalid connector operation accepted")
        stopped = CanonicalSubmitResult.model_validate({"voice_stopped": True})
        assert stopped.model_dump(mode="json") == {"voice_stopped": True}
        assert "profile" in ConnectorsListParams.model_fields
        opened = OpenModel.model_validate({"kept": {"null": None}})
        assert opened.model_copy(update={"other": 1}).model_dump() == {"kept": {"null": None}, "other": 1}
        assert opened.model_dump() == {"kept": {"null": None}}
        from tools.connectors.portal.wire import ConnectorTool
        from tools.connectors.gateway.wire import ConnectorSearchRequest, ConnectorExecuteSingleRequest, ConnectorExecuteCall
        tool_data = {"slug": "fixture", "name": "Read", "description": "Read fixture", "facet": "future-facet",
                     "hints": [], "categories": ["FILES"], "noAuth": False, "deprecated": False, "newField": 1}
        tool = ConnectorTool.model_validate(tool_data)
        assert tool.facet == "unclassified" and tool.categories == ["files"]
        assert tool.model_dump(by_alias=True)["noAuth"] is False
        try: ConnectorTool.model_validate({**tool_data, "noAuth": "false"})
        except ValidationError: pass
        else: raise AssertionError("Strict remote credential policy accepted a string boolean")
        search = ConnectorSearchRequest.model_validate({"queries": [{"useCase": "find", "knownFields": "name"}]})
        assert search.model_dump(by_alias=True) == {"queries": [{"useCase": "find", "knownFields": "name"}]}
        execution = ConnectorExecuteSingleRequest(call=ConnectorExecuteCall(connector="fixture", tool="read", arguments={"value": None}))
        assert execution.model_dump(exclude_none=True)["call"]["arguments"] == {"value": None}
        from hermes_cli import web_server as web
        import httpx
        report["routes"] = len(web.app.routes)
        headers = {web._SESSION_HEADER_NAME: web._SESSION_TOKEN}
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://127.0.0.1") as client:
                denied = await client.get("/api/config")
                assert denied.status_code in (401, 403), (denied.status_code, denied.text)
                allowed = await client.get("/api/config", headers=headers)
                assert allowed.status_code == 200, (allowed.status_code, allowed.text)
                invalid = await client.get("/api/sessions?limit=-1", headers=headers)
                assert invalid.status_code == 422, (invalid.status_code, invalid.text)
                stats = await client.get("/api/shared-metrics/consent", headers=headers)
                assert stats.status_code == 200, (stats.status_code, stats.text)
                assert isinstance(stats.json()["enabled"], bool)
                report["statuses"] = [denied.status_code, allowed.status_code, invalid.status_code, stats.status_code]
        asyncio.run(run())
        report["ok"] = True
except BaseException as exc:
    report.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
report.update(stdout=out.getvalue()[-1500:], stderr=err.getvalue()[-4000:])
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(not report["ok"] or bool(blocked))
'''


@unittest.skipIf(sys.version_info < (3, 14), "Original Hermes targets Python 3.14")
class OriginalDashboardTests(unittest.TestCase):
    def test_original_dashboard_auth_validation_and_config_routes(self):
        from standalone.offline_agent_probe import _source_snapshot, _vendor_sources
        from standalone.agent_probe_worker import fixture_environment
        with tempfile.TemporaryDirectory(prefix="hermes-original-dashboard-") as directory:
            scratch = Path(directory)
            for name in ("app", "vendor", "home", "work", "tmp"):
                (scratch / name).mkdir()
            count = _vendor_sources(PROJECT, PROJECT / "standalone/.cache/downloads", scratch / "vendor")
            _, digest = _source_snapshot(PROJECT, scratch / "app")
            bundle = scratch / digest
            bundle.mkdir()
            (scratch / "app").rename(bundle / "app")
            (scratch / "vendor").rename(bundle / "vendor")
            result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", WORKER,
                                     str(bundle / "app"), str(bundle / "vendor"), str(scratch)],
                                    env=fixture_environment(scratch), cwd=scratch / "work",
                                    capture_output=True, text=True, encoding="utf-8", timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["ok"], report)
            self.assertGreater(report["models"], 0)
            self.assertGreater(report["routes"], 0)
            self.assertGreater(report["contracts"], 0)
            self.assertEqual(report["blocked"], [])
