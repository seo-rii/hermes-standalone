"""Original voice-tool dispatch through real loopback multipart and binary HTTP."""

import base64
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from urllib.parse import parse_qs, urlsplit

from standalone.tests import test_transport_agent as _agent_fixture
from standalone.tests import test_transport as _wire_harness


AUDIO = b"ID3\x04\x00\x00\x00\x00\x00\x00standalone-voice-fixture"


class _VoiceHTTPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["content-length"]))
        content_type = self.headers.get("content-type", "")
        body, files = {}, {}
        if content_type.startswith("multipart/"):
            message = BytesParser(policy=policy.default).parsebytes(
                b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + raw)
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                data = part.get_payload(decode=True)
                if part.get_filename():
                    files[name] = {"filename": part.get_filename(), "data": data}
                else:
                    body[name] = data.decode("utf-8")
        else:
            body = json.loads(raw)
        path = urlsplit(self.path)
        self.server.requests.append({"path": path.path, "query": parse_qs(path.query), "body": body,
                                     "files": files, "headers": dict(self.headers.items())})
        status, mime = 200, "application/json"
        if body.get("model") == "reject-fixture" or body.get("input") == "Reject fixture.":
            status, response = 400, {"error": {"message": "fixture request rejected", "type": "invalid_request_error"}}
        elif path.path == "/v1/audio/transcriptions":
            response = {"text": "음성 fixture transcript", "language": "ko", "segments": []}
        elif path.path == "/v1/audio/speech":
            response = {"audio_data": ("invalid~base64" if body.get("input") == "Malformed fixture."
                                      else base64.b64encode(AUDIO).decode()), "format": "mp3"}
        elif path.path.startswith("/v1/text-to-speech/"):
            response, mime = AUDIO, "audio/mpeg"
        else:
            status, response = 404, {"error": {"message": "unknown fixture route"}}
        payload = response if isinstance(response, bytes) else json.dumps(response).encode()
        self.send_response(status)
        self.send_header("content-type", mime)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


_WORKER = r'''
import contextlib, io, json, os, pathlib, sys, traceback, wave
from urllib.parse import urlsplit
source, vendor, scratch, endpoint, mode = sys.argv[1:]
source, vendor, scratch = map(pathlib.Path, (source, vendor, scratch))
sys.path[:0] = [str(source), str(vendor)]
from standalone.runtime import initialize_bundle
initialize_bundle({"root": str(source.parent), "artifact": str(scratch / "fixture.py"),
                   "bundle_sha256": source.parent.name, "manifest": {"project": "hermes-standalone"}})
home = pathlib.Path(os.environ["HERMES_HOME"])
home.mkdir(parents=True)
(home / "config.yaml").write_text(json.dumps({
    "stt": {"provider": "mistral", "cloud_trim_silence": False, "prompt": "Hermes fixture terminology",
            "mistral": {"base_url": endpoint + "/v1", "model": "voxtral-fixture", "language": "ko"}},
    "tts": {"provider": "mistral", "mistral": {"base_url": endpoint, "model": "voxtral-tts-fixture", "voice_id": "fixture-voice"},
            "elevenlabs": {"base_url": endpoint, "model_id": "eleven-fixture", "voice_id": "fixture-voice"}},
}), encoding="utf-8")
(home / ".env").write_text("MISTRAL_API_KEY=local-mistral-fixture\nELEVENLABS_API_KEY=local-eleven-fixture\n", encoding="utf-8")
audio = scratch / "voice upload.wav"
with wave.open(str(audio), "wb") as output:
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(16000)
    output.writeframes(b"\x00\x00" * 160)
address = urlsplit(endpoint)
blocked = []
def audit(event, args):
    if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork"):
        blocked.append(event)
        raise PermissionError("Voice fixture disallows process execution")
    if event == "socket.getaddrinfo" and (args[0] != "127.0.0.1" or int(args[1]) != address.port):
        blocked.append(event)
        raise PermissionError("Voice fixture disallows external name resolution")
    if event == "socket.connect" and args[1] != ("127.0.0.1", address.port):
        blocked.append(event)
        raise PermissionError("Voice fixture allows only its local server")
    if event == "import" and args[0].split(".")[0] in {"pip", "uv", "ensurepip"}:
        blocked.append(event)
        raise PermissionError("Voice fixture disallows installers")
sys.addaudithook(audit)
output, errors = io.StringIO(), io.StringIO()
report = {"ok": False, "blocked": blocked}
try:
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        from tools.transcription_tools import transcribe_audio
        from tools.tts_tool import text_to_speech_tool, check_tts_requirements
        report["available"] = check_tts_requirements()
        report["transcription"] = transcribe_audio(str(audio), model="reject-fixture" if mode == "errors" else None)
        report["mistral"] = json.loads(text_to_speech_tool(
            "Reject fixture." if mode == "errors" else "Mistral fixture speech.",
            output_path=str(scratch / "mistral.mp3"), provider="mistral"))
        if mode == "success":
            config_path = home / "config.yaml"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["tts"]["provider"] = "elevenlabs"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            report["elevenlabs_available"] = check_tts_requirements()
            report["elevenlabs"] = json.loads(text_to_speech_tool(
                "ElevenLabs fixture speech.", output_path=str(scratch / "elevenlabs.mp3"), provider="elevenlabs"))
            report["audio_equal"] = ((scratch / "mistral.mp3").read_bytes() == (scratch / "elevenlabs.mp3").read_bytes())
            report["ok"] = (report["transcription"].get("success") and report["mistral"].get("success")
                            and report["elevenlabs"].get("success") and report["audio_equal"])
        else:
            report["ok"] = not report["transcription"].get("success") and not report["mistral"].get("success")
            report["leftover_audio"] = [path.name for path in scratch.glob("*.mp3")]
        report["sdk_modules"] = [name for name in ("mistralai", "elevenlabs", "pydantic") if name in sys.modules]
except BaseException as error:
    report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
report["stdout"] = output.getvalue()[-1000:]
report["stderr"] = errors.getvalue()[-3000:]
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(not report["ok"] or bool(blocked))
'''


@unittest.skipIf(sys.version_info < (3, 14), "The upstream runtime targets Python 3.14")
class VoiceToolIntegrationTests(unittest.TestCase):
    setUpClass = classmethod(_agent_fixture.AgentHTTPIntegrationTests.setUpClass.__func__)

    def run_voice_fixture(self, mode):
        from standalone.agent_probe_worker import fixture_environment
        scratch = self.directory / mode
        for name in ("home", "work", "tmp"):
            (scratch / name).mkdir(parents=True)
        worker = scratch / "worker.py"
        worker.write_text(_WORKER, encoding="utf-8")
        server = ThreadingHTTPServer(("127.0.0.1", 0), _VoiceHTTPHandler)
        server.daemon_threads = True
        server.requests = []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run([
                sys.executable, "-I", "-S", "-B", "-X", "utf8", str(worker), str(self.source), str(self.vendor),
                str(scratch), f"http://127.0.0.1:{server.server_port}", mode,
            ], cwd=scratch / "work", env=fixture_environment(scratch), capture_output=True,
               text=True, encoding="utf-8", timeout=30)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["available"])
        self.assertEqual(report["blocked"], [])
        self.assertEqual(report["sdk_modules"], [])
        return report, server.requests, scratch

    def test_original_mistral_transcription_and_both_speech_tools(self):
        report, requests, scratch = self.run_voice_fixture("success")
        self.assertEqual(report["transcription"]["transcript"], "음성 fixture transcript")
        self.assertEqual(report["transcription"]["provider"], "mistral")
        self.assertEqual(report["mistral"]["provider"], "mistral")
        self.assertEqual(report["elevenlabs"]["provider"], "elevenlabs")
        self.assertTrue(report["elevenlabs_available"])
        upload, mistral, eleven = requests
        self.assertEqual(upload["path"], "/v1/audio/transcriptions")
        self.assertEqual(upload["body"], {"model": "voxtral-fixture", "language": "ko", "prompt": "Hermes fixture terminology"})
        self.assertEqual(upload["files"]["file"]["filename"], "voice upload.wav")
        self.assertEqual(upload["files"]["file"]["data"], (scratch / "voice upload.wav").read_bytes())
        self.assertEqual(upload["headers"]["Authorization"], "Bearer local-mistral-fixture")
        self.assertEqual(mistral["body"], {"model": "voxtral-tts-fixture", "input": "Mistral fixture speech.",
                                           "voice_id": "fixture-voice", "response_format": "mp3"})
        self.assertEqual(eleven["path"], "/v1/text-to-speech/fixture-voice")
        self.assertEqual(eleven["query"], {"output_format": ["mp3_44100_128"]})
        self.assertEqual(eleven["body"], {"text": "ElevenLabs fixture speech.", "model_id": "eleven-fixture"})
        self.assertEqual(eleven["headers"]["xi-api-key"], "local-eleven-fixture")
        self.assertEqual((scratch / "mistral.mp3").read_bytes(), AUDIO)

    def test_original_error_envelopes_and_failed_file_cleanup(self):
        report, requests, _scratch = self.run_voice_fixture("errors")
        self.assertEqual(len(requests), 2)
        self.assertIn("BadRequestError", report["transcription"]["error"])
        self.assertIn("BadRequestError", report["mistral"]["error"])
        self.assertEqual(report["leftover_audio"], [])


class VoiceDependencyTests(_wire_harness.TransportDependencyTests):
    CASES = "standalone.tests.test_transport_voice.VoiceBoundaryWireTests"


@unittest.skipUnless(importlib.util.find_spec("httpx"), "Voice wire cases run in the isolated dependency subprocess")
class VoiceBoundaryWireTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _VoiceHTTPHandler)
        self.server.daemon_threads = True
        self.server.requests = []
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        def close():
            self.server.shutdown()
            self.server.server_close()
            thread.join(timeout=3)
        self.addCleanup(close)

    def test_audio_close_before_iteration_and_client_reuse(self):
        from standalone.transport_vendor_media import ElevenLabs
        with ElevenLabs(api_key="fixture", base_url=self.endpoint) as client:
            stream = client.text_to_speech.convert(voice_id="voice", text="unused")
            stream.close()
            self.assertTrue(stream.response.is_closed)
            self.assertFalse(client._client.is_closed)
            stream = client.text_to_speech.convert(voice_id="voice", text="kept", enable_logging=False,
                extra_body={"seed": 0, "nullable": None}, extra_headers={"X-Profile": "fixture"})
            self.assertEqual(b"".join(stream), AUDIO)
            self.assertTrue(stream.response.is_closed)
        request = self.server.requests[-1]
        self.assertEqual(request["query"], {"enable_logging": ["false"]})
        self.assertEqual(request["body"], {"text": "kept", "seed": 0, "nullable": None})
        self.assertEqual(request["headers"]["X-Profile"], "fixture")

    def test_malformed_base64_fails_before_audio_can_be_written(self):
        from standalone.transport_types import APIResponseValidationError
        from standalone.transport_vendor_media import Mistral
        with Mistral(api_key="fixture", server_url=self.endpoint) as client:
            with self.assertRaises(APIResponseValidationError) as failure:
                client.audio.speech.complete(model="fixture", input="Malformed fixture.")
            self.assertTrue(failure.exception.response.is_closed)
            self.assertEqual(client.audio.speech.complete(model="fixture", input="Valid fixture.").audio_data,
                             base64.b64encode(AUDIO).decode())


if __name__ == "__main__":
    unittest.main()
