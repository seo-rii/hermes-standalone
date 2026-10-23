"""Real google-auth signers, claims, token exchange and service discovery offline."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from standalone import vendor


PROJECT = Path(__file__).resolve().parents[2]
DEPENDENCIES = PROJECT / "standalone/dependencies.json"
DOWNLOADS = PROJECT / "standalone/.cache/downloads"


class GoogleAuthCryptoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        packages = {item["name"]: item for item in vendor.load_dependencies(DEPENDENCIES)["packages"]}
        if "google-auth" not in packages:
            raise unittest.SkipTest("Google authentication closure has not been reviewed yet")
        pending = list(getattr(cls, "DEPENDENCIES", ("google-api-python-client", "google-auth-oauthlib", "tlslite-ng")))
        selected = set()
        while pending:
            name = pending.pop()
            if name in selected:
                continue
            selected.add(name)
            pending.extend(set(packages[name]["dependencies"]) - set(packages[name].get("replaced_dependencies", {})))
        if any(not (DOWNLOADS / packages[name]["filename"]).is_file() for name in selected):
            raise unittest.SkipTest("Explicitly fetch pinned Google authentication dependencies first")
        cls.temporary = tempfile.TemporaryDirectory(prefix="hermes-google-auth-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.library = cls.directory / "vendor"
        paths = vendor.ArchivePaths()
        for name in sorted(selected):
            files, _ = vendor.read_wheel(packages[name], DOWNLOADS)
            for path, content in files.items():
                paths.add(path)
                target = cls.library / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        # Public test keys from the hash-verified upstream archive, never user keys.
        cls.fixtures = cls.directory / "fixtures"
        cls.fixtures.mkdir()
        content = vendor._verified_bytes(packages["tlslite-ng"], DOWNLOADS)
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
            for name in ("serverX509Key.pem", "serverX509Cert.pem", "serverECKey.pem", "serverECCert.pem",
                         "serverP384ECKey.pem", "serverP384ECCert.pem"):
                data = archive.extractfile("tlslite_ng-0.8.2/tests/" + name).read()
                (cls.fixtures / name).write_bytes(data)

    def execute(self, program):
        home = self.directory / self._testMethodName
        home.mkdir()
        environment = {"HOME": str(home), "USERPROFILE": str(home), "PATH": os.defpath,
                       "GOOGLE_API_PREVENT_AGENT_TOKEN_SHARING_FOR_GCP_SERVICES": "false"}
        for name in ("SYSTEMROOT", "WINDIR"):
            if name in os.environ:
                environment[name] = os.environ[name]
        prefix = f"import sys; sys.path[:0] = {[str(self.library), str(PROJECT)]!r}\n"
        prefix += f"from pathlib import Path\nfixtures = Path({str(self.fixtures)!r})\n"
        prefix += '''
def only_fixture_network(event, args):
    if event == "socket.connect" and (not isinstance(args[1], tuple) or args[1][0] != "127.0.0.1"):
        raise RuntimeError("Only loopback fixture connections are permitted")
    if event == "socket.getaddrinfo" and args[0] != "127.0.0.1":
        raise RuntimeError("Only loopback fixture name resolution is permitted")
    if event in ("subprocess.Popen", "os.system"):
        if event == "subprocess.Popen" and args[1] == ["uname", "-p"]:
            # platform.platform() treats this optional host-fact command as
            # absent. Refuse execution while preserving its normal fallback.
            raise FileNotFoundError("No external uname in authentication fixtures")
        raise RuntimeError("Google authentication fixtures cannot run external commands")
sys.addaudithook(only_fixture_network)
'''
        child = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", prefix + program],
                               cwd=home, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=45)
        self.assertEqual(child.returncode, 0, child.stderr + child.stdout)

    def test_rsa_signing_public_certificate_jwt_claim_checks_and_pickle(self):
        self.execute('''
import base64, hashlib, pickle, time
from google.auth import crypt, jwt
from standalone.crypto_google import RSASigner
key = (fixtures / "serverX509Key.pem").read_bytes()
certificate = (fixtures / "serverX509Cert.pem").read_bytes()
signer = crypt.RSASigner.from_string(key, key_id="fixture-key")
verifier = crypt.RSAVerifier.from_string(certificate)
message = b"offline signed message"
signature = signer.sign(message)
assert verifier.verify(message, signature)
assert not verifier.verify(message + b"changed", signature)
assert not verifier.verify(message, signature[:-1])
# Independent RFC 8017 encoded-message equation checks the exact PKCS1v1.5 SHA256 encoding.
public = verifier._impl._pubkey
encoded = pow(int.from_bytes(signature, "big"), public.e, public.n).to_bytes(len(signature), "big")
digest_info = bytes.fromhex("3031300d060960864801650304020105000420") + hashlib.sha256(message).digest()
assert encoded == b"\\x00\\x01" + b"\\xff" * (len(signature) - len(digest_info) - 3) + b"\\x00" + digest_info
restored = pickle.loads(pickle.dumps(signer))
assert restored.key_id == "fixture-key" and restored.sign(message) == signature
now = int(time.time())
token = jwt.encode(signer, {"iss": "fixture", "aud": "fixture-audience", "iat": now, "exp": now + 60})
assert jwt.decode(token, certs={"fixture-key": certificate}, audience="fixture-audience")["iss"] == "fixture"
for changed_audience in ("wrong-audience",):
    try: jwt.decode(token, certs={"fixture-key": certificate}, audience=changed_audience)
    except ValueError: pass
    else: raise AssertionError("JWT audience validation was bypassed")
corrupted = bytearray(signature); corrupted[-1] ^= 1
assert not verifier.verify(message, corrupted)
parsed = RSASigner.from_string(key)._key
parsed.dP += 1
try: RSASigner(parsed)
except ValueError: pass
else: raise AssertionError("Malformed RSA CRT key was accepted")
assert "cryptography" not in sys.modules
''')

    def test_es256_and_es384_real_keys_raw_signatures_and_jwt(self):
        self.execute('''
import pickle, time
from google.auth import crypt, jwt
for key_name, cert_name, algorithm, length in [
    ("serverECKey.pem", "serverECCert.pem", "ES256", 64),
    ("serverP384ECKey.pem", "serverP384ECCert.pem", "ES384", 96),
]:
    key = (fixtures / key_name).read_bytes(); cert = (fixtures / cert_name).read_bytes()
    signer = crypt.EsSigner.from_string(key, key_id="fixture-key")
    verifier = crypt.EsVerifier.from_string(cert)
    assert signer.algorithm == algorithm
    signature = signer.sign(b"fixture")
    assert len(signature) == length and verifier.verify(b"fixture", signature)
    assert not verifier.verify(b"changed", signature) and not verifier.verify(b"fixture", signature[:-1])
    assert pickle.loads(pickle.dumps(signer)).sign(b"fixture") == signature
    now = int(time.time()); token = jwt.encode(signer, {"iat": now, "exp": now+60, "aud": "fixture"})
    assert jwt.decode(token, certs={"fixture-key": cert}, audience="fixture")["aud"] == "fixture"
assert "cryptography" not in sys.modules
''')

    def test_original_service_account_refresh_uses_signed_assertion_at_loopback_token_endpoint(self):
        self.execute('''
import json, threading, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from google.auth import jwt
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import requests
certificate = (fixtures / "serverX509Cert.pem").read_bytes()
observed = []
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args): pass
    def do_POST(self):
        fields = urllib.parse.parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
        observed.append(fields)
        claims = jwt.decode(fields["assertion"][0], certs=certificate)
        assert claims["iss"] == "fixture@example.invalid" and claims["scope"] == "fixture.scope"
        content = json.dumps({"access_token": "fixture-access-token", "expires_in": 3600, "token_type": "Bearer"}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content))); self.end_headers(); self.wfile.write(content)
server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
session = requests.Session(); session.trust_env = False
try:
    info = {"type": "service_account", "client_email": "fixture@example.invalid", "private_key_id": "fixture-key",
            "private_key": (fixtures / "serverX509Key.pem").read_text(),
            "token_uri": f"http://127.0.0.1:{server.server_port}/token"}
    credentials = service_account.Credentials.from_service_account_info(info, scopes=["fixture.scope"])
    credentials.refresh(Request(session=session))
    assert credentials.token == "fixture-access-token" and credentials.valid
    assert observed[0]["grant_type"] == ["urn:ietf:params:oauth:grant-type:jwt-bearer"]
finally:
    session.close(); server.shutdown(); thread.join(); server.server_close()
assert "cryptography" not in sys.modules
''')

    def test_certificate_spiffe_binding_and_fingerprint_preserve_upstream_rules(self):
        self.execute('''
import base64, hashlib
from urllib.parse import quote
from google.auth import _agent_identity_utils as identity, crypt
from standalone.crypto_google import parse_certificate, _pem
from pyasn1.codec.der.encoder import encode
from pyasn1.type import univ
from pyasn1_modules import rfc5280
certificate = parse_certificate((fixtures / "serverX509Cert.pem").read_bytes())
assert not identity._is_agent_identity_certificate(certificate)
assert identity.calculate_certificate_fingerprint(certificate) == quote(base64.b64encode(hashlib.sha256(certificate.der).digest()).decode().rstrip("="))
san = rfc5280.SubjectAltName(); san[0]["uniformResourceIdentifier"] = "spiffe://agents.global.org-123.system.id.goog/fixture"
extension = rfc5280.Extension(); extension["extnID"] = "2.5.29.17"; extension["extnValue"] = encode(san)
parsed = certificate.parsed
parsed["tbsCertificate"]["extensions"].clear(); parsed["tbsCertificate"]["extensions"][0] = extension
parsed["tbsCertificate"]["signature"]["algorithm"] = "1.2.840.113549.1.1.11"
parsed["signatureAlgorithm"]["algorithm"] = "1.2.840.113549.1.1.11"
signature = crypt.RSASigner.from_string((fixtures / "serverX509Key.pem").read_bytes()).sign(encode(parsed["tbsCertificate"]))
parsed["signature"] = univ.BitString.fromOctetString(signature)
bound = identity.parse_certificate(_pem("CERTIFICATE", encode(parsed)))
assert identity._is_agent_identity_certificate(bound)
assert identity.get_cached_cert_fingerprint(_pem("CERTIFICATE", encode(parsed))) == identity.calculate_certificate_fingerprint(bound)
try: identity.parse_certificate(b"malformed certificate")
except ValueError: pass
else: raise AssertionError("Malformed certificate accepted")
''')

    def test_google_discovery_and_oauth_libraries_import_and_validate_real_request_shapes(self):
        self.execute('''
from google.auth.credentials import AnonymousCredentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from google.api_core import retry
service = build("drive", "v3", credentials=AnonymousCredentials(), static_discovery=True)
request = service.files().list(pageSize=7, fields="files(id,name)")
assert "pageSize=7" in request.uri and request.method == "GET"
flow = Flow.from_client_config({"installed": {"client_id": "fixture", "client_secret": "fixture-secret",
    "auth_uri": "https://fixture.invalid/auth", "token_uri": "https://fixture.invalid/token"}}, scopes=["fixture.scope"])
url, state = flow.authorization_url(state="fixture-state")
assert state == "fixture-state" and "client_id=fixture" in url and "scope=fixture.scope" in url
assert "cryptography" not in sys.modules
service.close()
''')


if __name__ == "__main__":
    unittest.main()
