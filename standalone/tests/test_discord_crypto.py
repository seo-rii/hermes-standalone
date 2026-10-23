"""Discord transport crypto matches independent native libsodium outputs."""

from pathlib import Path
import unittest

from standalone.tests import test_crypto_google as crypto_fixture


FIXTURE = Path(__file__).with_name("fixtures") / "discord-voice-crypto.json"


class DiscordTransportCryptoTests(unittest.TestCase):
    DEPENDENCIES = ("tlslite-ng",)

    @classmethod
    def setUpClass(cls):
        crypto_fixture.GoogleAuthCryptoTests.setUpClass.__func__(cls)

    def execute(self, program):
        prefix = f"fixture_path=Path({str(FIXTURE)!r})\n"
        prefix += "import base64,json\ndata=json.loads(fixture_path.read_text())\ndecode=lambda text:base64.b64decode(text)\n"
        crypto_fixture.GoogleAuthCryptoTests.execute(self, prefix + program)

    def test_hchacha_hsalsa_key_derivation_native_vectors(self):
        self.execute('''
from standalone.discord_crypto import hchacha20,hsalsa20
for case in data["derivations"]:
    derive=hchacha20 if case["mode"]=="hchacha20" else hsalsa20
    assert derive(decode(case["key"]),decode(case["nonce"]))==decode(case["expected"]),case["mode"]
''')

    def test_aead_and_secretbox_exact_ciphertexts_authentication_and_combined_format(self):
        self.execute('''
from standalone.discord_crypto import Aead,SecretBox,CryptoError
for case in data["boxes"]:
    key,nonce,plaintext,aad,expected=(decode(case[name]) for name in ("key","nonce","plaintext","aad","expected"))
    box=Aead(key) if case["mode"]=="aead" else SecretBox(key)
    encrypted=box.encrypt(plaintext,aad,nonce) if case["mode"]=="aead" else box.encrypt(plaintext,nonce)
    assert encrypted.ciphertext==expected and encrypted.nonce==nonce
    assert bytes(encrypted)==nonce+expected
    assert (box.decrypt(expected,aad,nonce) if case["mode"]=="aead" else box.decrypt(expected,nonce))==plaintext
    assert (box.decrypt(encrypted,aad) if case["mode"]=="aead" else box.decrypt(encrypted))==plaintext
    for position in (0,len(expected)-1):
        corrupt=bytearray(expected);corrupt[position]^=1
        try:
            if case["mode"]=="aead":box.decrypt(corrupt,aad,nonce)
            else:box.decrypt(corrupt,nonce)
        except CryptoError:pass
        else:raise AssertionError("Unauthenticated transport bytes were accepted")
    if case["mode"]=="aead":
        try:box.decrypt(expected,aad+b"changed",nonce)
        except CryptoError:pass
        else:raise AssertionError("RTP associated data was not authenticated")
assert not any(name in sys.modules for name in ("nacl","_sodium","Crypto","cryptography"))
''')


if __name__ == "__main__":
    unittest.main()
