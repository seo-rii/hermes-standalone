"""Known-answer primitives and real Hermes encrypted stores with no native wheels."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone import offline_agent_probe, vendor


PROJECT = Path(__file__).resolve().parents[2]
DOWNLOADS = PROJECT / "standalone/.cache/downloads"


def run_isolated(source, library, home, program, context=None):
    environment = {"HOME": str(home), "USERPROFILE": str(home), "HERMES_HOME": str(home / "hermes"),
                   "PATH": os.defpath, "LANG": "C.UTF-8"}
    for name in ("SYSTEMROOT", "WINDIR"):
        if name in os.environ:
            environment[name] = os.environ[name]
    prefix = f"import sys; sys.path[:0] = {[str(source), str(library)]!r}\n"
    if context:
        prefix += f"from standalone.runtime import initialize_bundle\ninitialize_bundle({context!r})\n"
    prefix += '''
def no_network_or_process(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"):
        raise RuntimeError("Crypto fixtures cannot access the network or execute external processes")
sys.addaudithook(no_network_or_process)
'''
    return subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", prefix + program],
                          cwd=home, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=45)


class PureCryptoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="hermes-crypto-vectors-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.library = cls.directory / "vendor"
        packages = [item for item in vendor.load_dependencies(PROJECT / "standalone/dependencies.json")["packages"]
                    if item["name"] in {"ecdsa", "six", "tlslite-ng"}]
        if any(not (DOWNLOADS / item["filename"]).is_file() for item in packages):
            raise unittest.SkipTest("Explicitly fetch the pinned crypto inputs before vector tests")
        for package in packages:
            files, _ = vendor.read_wheel(package, DOWNLOADS)
            for name, content in files.items():
                target = cls.library / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

    def execute(self, program):
        child = run_isolated(PROJECT, self.library, self.directory, program)
        self.assertEqual(child.returncode, 0, child.stderr + child.stdout)

    def test_nist_aes_cbc_and_ecb_vectors_and_invalid_lengths(self):
        self.execute('''
from standalone.crypto import aes_cbc_encrypt, aes_cbc_decrypt, aes_ecb_encrypt, aes_ecb_decrypt
# NIST SP 800-38A, F.1.1 and F.2.1.
key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
iv = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
plain = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a" "ae2d8a571e03ac9c9eb76fac45af8e51"
                      "30c81c46a35ce411e5fbc1191a0a52ef" "f69f2445df4f9b17ad2b417be66c3710")
cipher = bytes.fromhex("7649abac8119b246cee98e9b12e9197d" "5086cb9b507219ee95db113a917678b2"
                       "73bed6b8e3c1743b7116e69e22229516" "3ff1caa1681fac09120eca307586e1a7")
assert aes_cbc_encrypt(key, iv, plain) == cipher
assert aes_cbc_decrypt(key, iv, cipher) == plain
ecb = bytes.fromhex("3ad77bb40d7a3660a89ecaf32466ef97")
assert aes_ecb_encrypt(key, plain[:16]) == ecb
assert aes_ecb_decrypt(key, ecb) == plain[:16]
for operation, args in [(aes_cbc_encrypt, (key, iv[:-1], plain)),
                        (aes_cbc_decrypt, (key, iv, cipher[:-1])),
                        (aes_ecb_encrypt, (key[:-1], plain)),
                        (aes_ecb_decrypt, (key, cipher[:-1]))]:
    try: operation(*args)
    except ValueError: pass
    else: raise AssertionError("Malformed AES input was accepted")
''')

    def test_nist_gcm_vectors_and_authentication_before_plaintext(self):
        self.execute('''
from standalone.crypto import AESGCM, InvalidTag
# NIST GCM test cases: empty, one block, AES-256, and partial final block with AAD.
assert AESGCM(bytes(16)).encrypt(bytes(12), b"", None).hex() == "58e2fccefa7e3061367f1d57a4e7455a"
assert AESGCM(bytes(16)).encrypt(bytes(12), bytes(16), None).hex() == (
    "0388dace60b6a392f328c2b971b2fe78" "ab6e47d42cec13bdf53a67b21257bddf")
assert AESGCM(bytes(32)).encrypt(bytes(12), bytes(16), None).hex() == (
    "cea7403d4d606b6e074ec5d3baf39d18" "d0d1c8a799996bf0265b98b5d48ab919")
key = bytes.fromhex("feffe9928665731c6d6a8f9467308308")
nonce = bytes.fromhex("cafebabefacedbaddecaf888")
plain = bytes.fromhex("d9313225f88406e5a55909c5aff5269a" "86a7a9531534f7da2e4c303d8a318a72"
                      "1c3c0c95956809532fcf0e2449a6b525" "b16aedf5aa0de657ba637b39")
aad = bytes.fromhex("feedfacedeadbeeffeedfacedeadbeefabaddad2")
expected = bytes.fromhex("42831ec2217774244b7221b784d0d49c" "e3aa212f2c02a4e035c17e2329aca12e"
                         "21d514b25466931c7d8f6a5aac84aa05" "1ba30b396a0aac973d58e091"
                         "5bc94fbc3221a5db94fae95ae7121a47")
engine = AESGCM(key)
assert engine.encrypt(nonce, plain, aad) == expected
assert engine.decrypt(nonce, expected, aad) == plain
for index in range(len(expected)):
    changed = bytearray(expected); changed[index] ^= 1
    try: engine.decrypt(nonce, changed, aad)
    except InvalidTag: pass
    else: raise AssertionError("Unauthenticated plaintext was released")
for changed_nonce, changed_aad in [(bytes(12), aad), (nonce, aad + b"x")]:
    try: engine.decrypt(changed_nonce, expected, changed_aad)
    except InvalidTag: pass
    else: raise AssertionError("Nonce/AAD authentication was bypassed")
try: engine.decrypt(nonce, bytes(15), aad)
except InvalidTag: pass
else: raise AssertionError("Truncated GCM tag was accepted")
''')

    def test_rfc5869_hkdf_sha256_vectors(self):
        self.execute('''
from standalone.crypto import hkdf_sha256
material = bytes.fromhex("0b" * 22)
output = hkdf_sha256(material, salt=bytes.fromhex("000102030405060708090a0b0c"),
                     info=bytes.fromhex("f0f1f2f3f4f5f6f7f8f9"), length=42)
assert output.hex() == "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
assert hkdf_sha256(material, salt=None, info=b"", length=42).hex() == (
    "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8")
try: hkdf_sha256(material, salt=None, info=b"", length=8161)
except ValueError: pass
else: raise AssertionError("HKDF counter overflow was accepted")
''')

    def test_fernet_official_vector_authentication_expiry_and_padding(self):
        self.execute('''
import base64, datetime, hmac
from unittest.mock import patch
from standalone import crypto
import tlslite.utils.python_aes  # initialize its entropy self-test before fixing the fixture IV
# https://github.com/fernet/spec/blob/master/generate.json
key = b"cw_0x689RpI-jtRR7oE8h_eQsKImvJapLeSbXpwF4e4="
token = b"gAAAAAAdwJ6wAAECAwQFBgcICQoLDA0ODy021cpGVWKZ_eEwCGM4BLLF_5CV9dOPmrhuVUPgJobwOz7JcbmrR64jVmpU4IwqDA=="
timestamp = int(datetime.datetime.fromisoformat("1985-10-26T01:20:00-07:00").timestamp())
fernet = crypto.Fernet(key)
with patch.object(crypto.os, "urandom", return_value=bytes(range(16))):
    assert fernet.encrypt_at_time(b"hello", timestamp) == token
assert fernet.decrypt(token) == b"hello"
assert fernet.decrypt_at_time(token, 60, timestamp + 60) == b"hello"
for now in (timestamp + 61, timestamp - 61):
    try: fernet.decrypt_at_time(token, 60, now)
    except crypto.InvalidToken: pass
    else: raise AssertionError("Expired/future Fernet token accepted")
raw = base64.urlsafe_b64decode(token)
for index in range(len(raw)):
    changed = bytearray(raw); changed[index] ^= 1
    try: fernet.decrypt(base64.urlsafe_b64encode(changed))
    except crypto.InvalidToken: pass
    else: raise AssertionError("Fernet authentication was bypassed")
decoded_key = base64.urlsafe_b64decode(key)
bad_padding = raw[:25] + crypto.aes_cbc_encrypt(decoded_key[16:], raw[9:25], bytes(16))
bad_padding += hmac.digest(decoded_key[:16], bad_padding, "sha256")
try: fernet.decrypt(base64.urlsafe_b64encode(bad_padding))
except crypto.InvalidToken: pass
else: raise AssertionError("Authenticated malformed PKCS7 padding accepted")
for malformed in (b"", b"invalid", b"====", "한글"):
    try: fernet.decrypt(malformed)
    except crypto.InvalidToken: pass
    else: raise AssertionError("Malformed token accepted")
''')

    def test_unbound_source_install_does_not_silently_select_pure_backend(self):
        self.execute('''
from standalone.crypto_backend import fernet_types, aes_gcm_type
for factory in (fernet_types, aes_gcm_type):
    try: factory()
    except ModuleNotFoundError as error: assert error.name == "cryptography"
    else: raise AssertionError("Unbound source install changed its native backend")
''')


@unittest.skipUnless(sys.version_info >= (3, 14), "Original Hermes source requires Python 3.14")
class HermesCryptoIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="hermes-crypto-stores-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        manifest = vendor.load_dependencies(PROJECT / "standalone/dependencies.json")
        if any(not (DOWNLOADS / item["filename"]).is_file() for item in manifest["packages"]):
            raise unittest.SkipTest("Explicitly fetch runtime dependencies before integration tests")
        source = cls.directory / "source"
        source.mkdir()
        _, digest = offline_agent_probe._source_snapshot(PROJECT, source)
        cls.bundle = cls.directory / digest
        cls.bundle.mkdir()
        cls.source = cls.bundle / "app"
        source.rename(cls.source)
        cls.library = cls.bundle / "vendor"
        cls.library.mkdir()
        offline_agent_probe._vendor_sources(PROJECT, DOWNLOADS, cls.library)
        cls.context = {"root": str(cls.bundle), "artifact": str(cls.directory / "hermes.py"),
                       "bundle_sha256": digest, "manifest": {"project": "hermes-standalone"}}

    def execute(self, program):
        home = self.directory / self._testMethodName
        home.mkdir()
        child = run_isolated(self.source, self.library, home, program, self.context)
        self.assertEqual(child.returncode, 0, child.stderr + child.stdout)

    def test_existing_vault_store_encrypts_reopens_and_rejects_corruption(self):
        self.execute('''
import base64, os
from pathlib import Path
from agent.vault_store import VaultStore, VaultError
store = VaultStore(Path(os.environ["HERMES_HOME"]) / "vault")
item = store.add_item("login", "fixture", {"identifier_type": "username", "identifier": "fixture-user",
                                           "password": "local-vault-password"}, "https://fixture.invalid/path")
assert store.resolve_secret(item.id)["password"] == "local-vault-password"
assert "local-vault-password" not in repr(store.list_items())
assert b"local-vault-password" not in store._vault_path.read_bytes()
reopened = VaultStore(store._base)
assert reopened.resolve_secret(item.id)["password"] == "local-vault-password"
raw = bytearray(base64.urlsafe_b64decode(store._vault_path.read_bytes())); raw[-1] ^= 1
original = store._vault_path.read_bytes()
store._vault_path.write_bytes(base64.urlsafe_b64encode(raw))
try: reopened.list_items()
except VaultError: pass
else: raise AssertionError("Vault ciphertext corruption went unnoticed")
store._vault_path.write_bytes(original)
if os.name == "posix":
    assert store._key_path.stat().st_mode & 0o777 == 0o600
    assert store._vault_path.stat().st_mode & 0o777 == 0o600
assert reopened.remove_item(item.id) and not reopened.has_items()
assert "cryptography" not in sys.modules
''')

    def test_bitwarden_cache_format_roundtrip_authentication_and_staleness(self):
        self.execute('''
import base64, json, os, time
from pathlib import Path
from agent.secret_sources import bitwarden as bws
from agent.secret_sources._cache import CachedFetch
home = Path(os.environ["HERMES_HOME"])
cache_key = ("offline-fingerprint", "fixture-project", "https://fixture.invalid")
token = "offline-bootstrap-fixture"
entry = CachedFetch({"FIXTURE_SECRET": "cache-only-value"}, time.time())
bws._write_encrypted_disk_cache(cache_key=cache_key, access_token=token, entry=entry, home_path=home)
path = bws._encrypted_disk_cache_path(home)
original = path.read_bytes(); payload = json.loads(original)
assert set(payload) == {"version", "key", "salt", "nonce", "ciphertext"}
assert payload["version"] == 1 and len(base64.b64decode(payload["salt"])) == 16
assert len(base64.b64decode(payload["nonce"])) == 12 and b"cache-only-value" not in original
def read(access_token=token, key=cache_key, age=60):
    return bws._read_encrypted_disk_cache(cache_key=key, access_token=access_token,
                                         max_age_seconds=age, home_path=home)
assert read().secrets == entry.secrets
assert read(access_token="wrong-token") is None
assert read(key=("different-fingerprint", *cache_key[1:])) is None
cipher = bytearray(base64.b64decode(payload["ciphertext"])); cipher[-1] ^= 1
payload["ciphertext"] = base64.b64encode(cipher).decode(); path.write_text(json.dumps(payload))
assert read() is None
path.write_bytes(original)
bws._write_encrypted_disk_cache(cache_key=cache_key, access_token=token,
                                entry=CachedFetch(entry.secrets, time.time()-120), home_path=home)
assert read(age=60) is None and read(age=180).secrets == entry.secrets
assert "cryptography" not in sys.modules
''')

    def test_platform_callback_media_ecb_and_gcm_formats(self):
        self.execute('''
import base64, xml.etree.ElementTree as ET
from standalone.crypto import AESGCM, InvalidTag, aes_cbc_encrypt
from plugins.platforms.wecom.wecom_crypto import WXBizMsgCrypt, PKCS7Encoder, SignatureError
from plugins.platforms.wecom.media import WeComMediaMixin
from plugins.platforms.weixin.adapter import _aes128_ecb_encrypt, _aes128_ecb_decrypt, CRYPTO_AVAILABLE
from plugins.platforms.qqbot.crypto import decrypt_secret
key = bytes(range(32)); encoded_key = base64.b64encode(key).decode().rstrip("=")
wecom = WXBizMsgCrypt("fixture-token", encoded_key, "fixture-corporation")
message = "<xml><Content>offline callback</Content></xml>"
envelope = ET.fromstring(wecom.encrypt(message, nonce="fixture-nonce", timestamp="1700000000"))
args = [envelope.findtext(name) for name in ("MsgSignature", "TimeStamp", "Nonce", "Encrypt")]
assert wecom.decrypt(*args).decode() == message
for signature in ("0" * 40, "한글"):
    try: wecom.decrypt(signature, *args[1:])
    except SignatureError: pass
    else: raise AssertionError("WeCom callback signature was not checked")
media = b"offline attachment bytes"
encrypted = aes_cbc_encrypt(key, key[:16], PKCS7Encoder.encode(media))
assert WeComMediaMixin._decrypt_file_bytes(encrypted, encoded_key) == media
assert CRYPTO_AVAILABLE
assert _aes128_ecb_decrypt(_aes128_ecb_encrypt(media, key[:16]), key[:16]) == media
nonce = bytes(range(12)); ciphertext = AESGCM(key).encrypt(nonce, b"offline-client-secret", None)
assert decrypt_secret(base64.b64encode(nonce + ciphertext).decode(), base64.b64encode(key).decode()) == "offline-client-secret"
changed = bytearray(ciphertext); changed[-1] ^= 1
try: decrypt_secret(base64.b64encode(nonce + changed).decode(), base64.b64encode(key).decode())
except InvalidTag: pass
else: raise AssertionError("QQ bind secret authentication was bypassed")
assert "cryptography" not in sys.modules
''')


if __name__ == "__main__":
    unittest.main()
