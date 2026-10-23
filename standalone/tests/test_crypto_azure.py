"""Original Azure/MSAL authentication with Python crypto and local HTTPS only."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import unittest

from standalone import vendor
from standalone.tests import test_crypto_google as google_tests


class AzureCryptoTests(unittest.TestCase):
    DEPENDENCIES = ("azure-identity", "google-auth", "tlslite-ng")

    @classmethod
    def setUpClass(cls):
        names = {package["name"] for package in vendor.load_dependencies(google_tests.DEPENDENCIES)["packages"]}
        if "azure-identity" not in names:
            raise unittest.SkipTest("Azure authentication closure has not been reviewed yet")
        google_tests.GoogleAuthCryptoTests.setUpClass.__func__(cls)
        manifest = json.loads((Path(__file__).with_name("fixtures") / "azure_certificates.json").read_text())
        for name, entry in manifest["files"].items():
            data = base64.b64decode(entry["base64"], validate=True)
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise AssertionError("Azure certificate fixture content differs: " + name)
            (cls.fixtures / name).write_bytes(data)

    def execute(self, program):
        google_tests.GoogleAuthCryptoTests.execute(self, program)

    def test_rc2_known_vectors_encrypted_keys_pkcs12_integrity_and_rsa_pss(self):
        self.execute(r'''
import hashlib, jwt, time
from standalone import crypto_azure as crypto
from standalone.crypto_pem import rc2_cbc_decrypt
vectors = [
    ("0000000000000000",63,"0000000000000000","ebb773f993278eff"),
    ("ffffffffffffffff",64,"ffffffffffffffff","278b27e42e2f0d49"),
    ("3000000000000000",64,"1000000000000001","30649edf9be7d2c2"),
    ("88",64,"0000000000000000","61a8a244adacccf0"),
    ("88bca90e90875a",64,"0000000000000000","6ccf4308974c267f"),
    ("88bca90e90875a7f0f79c384627bafb2",64,"0000000000000000","1a807d272bbe5db1"),
    ("88bca90e90875a7f0f79c384627bafb2",128,"0000000000000000","2269552ab0f85ca6"),
]
for key, bits, plain, cipher in vectors:
    assert rc2_cbc_decrypt(bytes.fromhex(key), bytes(8), bytes.fromhex(cipher), bits) == bytes.fromhex(plain)
key = crypto.load_pem_private_key((fixtures / "serverX509Key.pem").read_bytes())
certificate = crypto.load_pem_x509_certificate((fixtures / "serverX509Cert.pem").read_bytes())
assert crypto.certificate_fingerprint(certificate, "sha256") == hashlib.sha256(certificate.der).digest()
for name in ("pkcs8_aes256.pem", "rsa_aes256.pem"):
    encrypted = (fixtures / name).read_bytes()
    assert crypto.load_pem_private_key(encrypted, b"fixture-password").n == key.n
    for password in (None, b"wrong-password"):
        try: crypto.load_pem_private_key(encrypted, password)
        except (ValueError, TypeError): pass
        else: raise AssertionError("Encrypted PEM accepted an incorrect or missing password")
for name in ("modern.pfx", "legacy.pfx", "legacy128.pfx", "unencrypted.pfx"):
    data = (fixtures / name).read_bytes()
    loaded, cert, chain = crypto.load_key_and_certificates(data, None if name == "unencrypted.pfx" else b"fixture-password")
    assert loaded.n == key.n and cert.der == certificate.der and chain == []
    if name != "unencrypted.pfx":
        damaged = bytearray(data); damaged[-1] ^= 1
        for value, password in ((data,b"wrong-password"),(damaged,b"fixture-password")):
            try: crypto.load_key_and_certificates(value,password)
            except ValueError: pass
            else: raise AssertionError("PKCS12 accepted an incorrect password or damaged MAC")
for algorithm in ("RS256", "PS256"):
    token = jwt.encode({"aud":"fixture","exp":int(time.time())+60},key,algorithm=algorithm)
    assert jwt.decode(token,crypto.certificate_pem(certificate),algorithms=[algorithm],audience="fixture")["aud"] == "fixture"
assert "cryptography" not in sys.modules
''')

    def test_original_certificate_parsers_and_aad_signature(self):
        self.execute(r'''
import base64, hashlib
from azure.identity._credentials.certificate import get_client_credential
from azure.identity._internal.aadclient_certificate import AadClientCertificate
from standalone import crypto_azure as crypto
certificate = (fixtures / "serverX509Cert.pem").read_bytes()
raw = (fixtures / "serverX509Key.pem").read_bytes()+certificate
expected = crypto.load_pem_x509_certificate(certificate)
for name in (None,"pkcs8_aes256.pem","rsa_aes256.pem","modern.pfx","legacy.pfx"):
    data = raw if name is None else (fixtures/name).read_bytes()+(certificate if name.endswith(".pem") else b"")
    credential = get_client_credential(certificate_data=data,password=None if name is None else "fixture-password",send_certificate_chain=True)
    assert credential["thumbprint"] == hashlib.sha1(expected.der).hexdigest()
    assert "BEGIN CERTIFICATE" in credential["public_certificate"]
    loaded = crypto.load_pem_private_key(credential["private_key"],credential.get("passphrase"))
    assert loaded.hasPrivateKey()
aad = AadClientCertificate(raw)
assert base64.urlsafe_b64decode(aad.thumbprint) == hashlib.sha1(expected.der).digest()
assert base64.urlsafe_b64decode(aad.sha256_thumbprint) == hashlib.sha256(expected.der).digest()
public = crypto.load_pem_private_key(raw)
for signature, scheme, salt in ((aad.sign_rs256(b"message"),"PKCS1",0),(aad.sign_ps256(b"message"),"PSS",32)):
    assert public.hashAndVerify(bytearray(signature),bytearray(b"message"),rsaScheme=scheme,hAlg="sha256",sLen=salt)
assert "cryptography" not in sys.modules
''')

    def test_original_secret_environment_workload_and_certificate_over_https(self):
        self.execute(r'''
import json, os, ssl, threading, time, urllib.parse, warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import jwt
from azure.identity import ClientSecretCredential, DefaultAzureCredential, EnvironmentCredential, WorkloadIdentityCredential, CertificateCredential
from azure.core.pipeline.transport import RequestsTransport
from standalone import crypto_azure as crypto
warnings.filterwarnings("ignore",message="Unverified HTTPS request")
tenant = "00000000-0000-0000-0000-000000000042"
client_id = "00000000-0000-0000-0000-000000000043"
scope = "https://ai.azure.com/.default"
certificate = (fixtures/"serverX509Cert.pem").read_bytes()
key_data = (fixtures/"serverX509Key.pem").read_bytes()+certificate
requests_seen = []
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def emit(self,value):
        content = json.dumps(value).encode(); self.send_response(200)
        self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(content)))
        self.end_headers(); self.wfile.write(content)
    def do_GET(self):
        assert self.path == "/"+tenant+"/v2.0/.well-known/openid-configuration", self.path
        self.emit({"authorization_endpoint":authority+"/"+tenant+"/oauth2/v2.0/authorize",
                   "token_endpoint":authority+"/"+tenant+"/oauth2/v2.0/token","issuer":authority+"/"+tenant+"/v2.0"})
    def do_POST(self):
        assert self.path == "/"+tenant+"/oauth2/v2.0/token", self.path
        body = urllib.parse.parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
        assert body["client_id"] == [client_id] and body["scope"] == [scope], body
        if "client_assertion" in body:
            assertion = body["client_assertion"][0]
            if assertion != "federated.fixture.assertion":
                claims = jwt.decode(assertion,certificate,algorithms=["RS256","PS256"],audience=authority+self.path)
                assert claims["iss"] == client_id and claims["sub"] == client_id
        else:
            assert body["client_secret"] == ["fixture-client-secret"], body
        requests_seen.append(body)
        self.emit({"access_token":"fixture-access-token","token_type":"Bearer","expires_in":3600})
server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
tls.load_cert_chain(fixtures/"serverX509Cert.pem",fixtures/"serverX509Key.pem")
server.socket = tls.wrap_socket(server.socket,server_side=True)
authority = "https://127.0.0.1:"+str(server.server_port)
thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
def options():
    return {"authority":authority,"disable_instance_discovery":True,"transport":RequestsTransport(connection_verify=False)}
try:
    os.environ.update(AZURE_TENANT_ID=tenant,AZURE_CLIENT_ID=client_id,AZURE_CLIENT_SECRET="fixture-client-secret",AZURE_AUTHORITY_HOST=authority)
    token_file = Path.cwd()/"federated-token"; token_file.write_text("federated.fixture.assertion")
    credentials = [ClientSecretCredential(tenant,client_id,"fixture-client-secret",**options()),
                   EnvironmentCredential(**options()),DefaultAzureCredential(**options()),
                   WorkloadIdentityCredential(tenant_id=tenant,client_id=client_id,token_file_path=str(token_file),**options()),
                   CertificateCredential(tenant,client_id,certificate_data=key_data,send_certificate_chain=True,**options()),
                   CertificateCredential(tenant,client_id,certificate_data=(fixtures/"modern.pfx").read_bytes(),password="fixture-password",**options())]
    for credential in credentials:
        before = len(requests_seen)
        token = credential.get_token(scope)
        assert token.token == "fixture-access-token" and token.expires_on > time.time()+3000
        assert len(requests_seen) == before+1
        assert credential.get_token(scope).token == token.token and len(requests_seen) == before+1
        credential.close()
    assert len(requests_seen) == 6 and sum("client_assertion" in body for body in requests_seen) == 3
    assert "cryptography" not in sys.modules
finally:
    server.shutdown();server.server_close();thread.join(timeout=5)
''')

    def test_original_profile_credentials_refuse_ambient_identity(self):
        self.execute(r'''
import os
from azure.identity import ClientSecretCredential, DefaultAzureCredential, WorkloadIdentityCredential, ManagedIdentityCredential
from agent import secret_scope
from agent.azure_identity_adapter import EntraIdentityConfig, build_credential, reset_credential_cache
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
os.environ.update(AZURE_TENANT_ID="launch-tenant",AZURE_CLIENT_ID="launch-client",AZURE_CLIENT_SECRET="launch-secret")
config = EntraIdentityConfig()
assert isinstance(build_credential(config),DefaultAzureCredential)
homes = [Path.cwd()/name for name in ("service-principal","empty","federated","managed")]
for home in homes: home.mkdir()
token_file = homes[2]/"assertion";token_file.write_text("federated.fixture.assertion")
def in_scope(index, scope):
    home_token = set_hermes_home_override(homes[index])
    secret_token = secret_scope.set_secret_scope(scope)
    try: return build_credential(config)
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
secret_scope.set_multiplex_active(True)
try:
    service = {"AZURE_TENANT_ID":"profile-tenant","AZURE_CLIENT_ID":"profile-client","AZURE_CLIENT_SECRET":"profile-secret"}
    credential = in_scope(0,service)
    assert isinstance(credential,ClientSecretCredential) and credential._tenant_id == "profile-tenant"
    assert in_scope(0,service) is credential
    try: in_scope(1,{})
    except RuntimeError as error: assert "refused for this profile" in str(error)
    else: raise AssertionError("A profile without credentials borrowed the launch identity")
    federated = in_scope(2,{"AZURE_TENANT_ID":"federated-tenant","AZURE_CLIENT_ID":"federated-client","AZURE_FEDERATED_TOKEN_FILE":str(token_file)})
    assert isinstance(federated,WorkloadIdentityCredential)
    assert isinstance(in_scope(3,{"AZURE_CLIENT_ID":"user-assigned-mi"}),ManagedIdentityCredential)
    assert in_scope(0,service) is credential
finally:
    secret_scope.set_multiplex_active(False)
    reset_credential_cache()
assert "cryptography" not in sys.modules
''')
