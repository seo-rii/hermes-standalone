"""Independent asymmetric JWT vectors and actual original JWKS consumers."""

from standalone.tests import test_crypto_google as _google_fixture
import unittest


class AsymmetricJWTTests(unittest.TestCase):
    DEPENDENCIES = ("google-api-python-client", "google-auth-oauthlib", "tlslite-ng", "pyjwt", "httpx", "ruamel-yaml")
    setUpClass = classmethod(_google_fixture.GoogleAuthCryptoTests.setUpClass.__func__)
    execute = _google_fixture.GoogleAuthCryptoTests.execute

    def test_independent_rsa_pss_ec_and_eddsa_tokens_preserve_claim_boundaries(self):
        self.execute('''
import base64, json
import jwt
document = json.loads((Path(__import__('standalone').__file__).parent / 'tests/fixtures/jwt.json').read_text())
for case in document['cases']:
    key = jwt.PyJWK.from_dict(case['jwk'])
    claims = jwt.decode(case['token'], key, algorithms=[case['algorithm']],
                        audience='fixture-audience', issuer='fixture-issuer')
    assert claims == case['claims'], case['algorithm']
    for kwargs in ({'audience':'other','issuer':'fixture-issuer'},
                   {'audience':'fixture-audience','issuer':'other'}):
        try: jwt.decode(case['token'], key, algorithms=[case['algorithm']], **kwargs)
        except jwt.InvalidTokenError: pass
        else: raise AssertionError('Unscoped JWT accepted')
    header, payload, signature = case['token'].split('.')
    altered = base64.urlsafe_b64encode(json.dumps({**claims, 'sub':'forged'}).encode()).rstrip(b'=').decode()
    try: jwt.decode(header+'.'+altered+'.'+signature, key, algorithms=[case['algorithm']], audience='fixture-audience')
    except jwt.InvalidSignatureError: pass
    else: raise AssertionError('Forged JWT accepted')
assert not any(name == 'cryptography' or name.startswith('cryptography.') for name in sys.modules)
''')

    def test_private_key_signing_jwk_roundtrip_and_algorithm_confusion_refusal(self):
        self.execute('''
import time
import jwt, ecdsa
from standalone.crypto_jwt import RSAAlgorithm
private = (fixtures/'serverX509Key.pem').read_bytes()
certificate = (fixtures/'serverX509Cert.pem').read_bytes()
for algorithm in ('RS256','RS384','RS512','PS256','PS384','PS512'):
    token = jwt.encode({'sub':'fixture','exp':int(time.time())+60}, private, algorithm=algorithm)
    assert jwt.decode(token, certificate, algorithms=[algorithm])['sub']=='fixture'
    try: jwt.decode(token, certificate, algorithms=['HS256'])
    except jwt.InvalidAlgorithmError: pass
    else: raise AssertionError('Algorithm confusion accepted')
    public = RSAAlgorithm().prepare_key(certificate)
    jwk = RSAAlgorithm.to_jwk(public, as_dict=True)
    assert RSAAlgorithm.from_jwk(jwk).n == public.n
for algorithm, curve in (('ES256',ecdsa.NIST256p),('ES384',ecdsa.NIST384p),
                         ('ES512',ecdsa.NIST521p),('ES256K',ecdsa.SECP256k1),
                         ('EdDSA',ecdsa.Ed25519),('EdDSA',ecdsa.Ed448)):
    key = (ecdsa.SigningKey.from_string(bytes(range(32 if curve is ecdsa.Ed25519 else 57)),curve=curve)
           if algorithm=='EdDSA' else ecdsa.SigningKey.from_secret_exponent(7,curve=curve))
    token = jwt.encode({'sub':'fixture'},key,algorithm=algorithm)
    assert jwt.decode(token,key.verifying_key,algorithms=[algorithm])['sub']=='fixture'
    provider = jwt.get_algorithm_by_name(algorithm)
    restored = provider.from_jwk(provider.to_jwk(key,as_dict=True))
    assert restored.to_string()==key.to_string()
    if algorithm=='ES256':
        try: jwt.encode({'sub':'fixture'},key,algorithm='ES384')
        except jwt.InvalidKeyError: pass
        else: raise AssertionError('Mismatched EC curve accepted')
for bad in ({'kty':'RSA','n':'Aw','e':'Aw'}, {'kty':'RSA','n':'not base64!','e':'AQAB'},
            {'kty':'EC','crv':'P-256','x':'AQ','y':'AQ'}):
    try: jwt.PyJWK.from_dict(bad)
    except jwt.InvalidKeyError: pass
    else: raise AssertionError('Invalid JWK accepted')
''')

    def test_original_dashboard_and_cron_verification_against_real_jwks_http(self):
        self.execute('''
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,threading,time
import jwt
from standalone.crypto_jwt import RSAAlgorithm
from plugins.dashboard_auth._shared import make_jwks_client,verify_jwt
from hermes_cli.dashboard_auth.base import InvalidCodeError
from plugins.cron_providers.chronos.verify import verify_nas_fire_token
private=(fixtures/'serverX509Key.pem').read_bytes()
public=(fixtures/'serverX509Cert.pem').read_bytes()
jwk=RSAAlgorithm.to_jwk(RSAAlgorithm().prepare_key(public),as_dict=True)
jwk.update(kid='fixture',alg='RS256',use='sig')
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        self.server.records.append(dict(self.headers))
        data=json.dumps({'keys':[jwk]}).encode()
        self.send_response(200);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.records=[]
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
    endpoint=f'http://127.0.0.1:{server.server_port}/jwks'
    now=int(time.time());claims={'sub':'fixture','aud':'fixture-audience','iss':'fixture-issuer',
        'iat':now,'exp':now+60,'purpose':'cron_fire'}
    token=jwt.encode(claims,private,algorithm='RS256',headers={'kid':'fixture'})
    client=make_jwks_client(endpoint)
    assert verify_jwt(token,client,algorithms=['RS256'],audience='fixture-audience',
                      issuer='fixture-issuer',label='fixture')==claims
    assert verify_nas_fire_token(token=token,expected_audience='fixture-audience',
                                 jwks_or_key=endpoint,issuer='fixture-issuer')==claims
    expired=jwt.encode({**claims,'exp':now-3600},private,algorithm='RS256',headers={'kid':'fixture'})
    try:verify_jwt(expired,client,algorithms=['RS256'],audience='fixture-audience',issuer='fixture-issuer',label='fixture')
    except InvalidCodeError:pass
    else:raise AssertionError('Expired dashboard credential accepted')
    assert verify_nas_fire_token(token=token,expected_audience='other',jwks_or_key=endpoint) is None
    assert all(item.get('Accept')=='application/json' and item.get('User-Agent')=='HermesAgent/1.0' for item in server.records)
finally:
    server.shutdown();server.server_close();thread.join(3)
''')
