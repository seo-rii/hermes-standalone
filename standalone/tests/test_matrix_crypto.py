"""Independent RFC/libolm vectors for the explicit pure Matrix crypto boundary."""

import unittest

from standalone.tests import test_crypto_google as crypto_fixture


class MatrixCryptoTests(unittest.TestCase):
    DEPENDENCIES = ("ecdsa", "tlslite-ng")

    @classmethod
    def setUpClass(cls):
        crypto_fixture.GoogleAuthCryptoTests.setUpClass.__func__(cls)

    def execute(self, program):
        crypto_fixture.GoogleAuthCryptoTests.execute(self, program)

    def test_rfc7748_x25519_and_rfc8032_expanded_ed25519_signatures(self):
        self.execute('''
from standalone.matrix_crypto import x25519, ed25519_key, ed25519_sign, verify_ed25519
# RFC 7748 section 6.1: independently published Alice/Bob keys and shared secret.
alice = bytes.fromhex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
bob = bytes.fromhex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb")
ap = bytes.fromhex("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")
bp = bytes.fromhex("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f")
shared = bytes.fromhex("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742")
assert x25519(alice) == ap and x25519(bob) == bp
assert x25519(alice,bp) == shared and x25519(bob,ap) == shared
# RFC 8032 section 7.1, test 1 (empty message). Libolm pickles store the
# expanded secret rather than the seed, so the expanded signing path matters.
seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
public, expanded = ed25519_key(seed)
assert public.hex() == "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
signature = ed25519_sign(public, expanded, b"")
assert signature.hex() == "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
verify_ed25519(public,b"",signature)
for invalid in (signature[:-1], signature[:30] + bytes([signature[30]^1]) + signature[31:]):
    try: verify_ed25519(public,b"",invalid)
    except ValueError: pass
    else: raise AssertionError("Invalid Ed25519 signature accepted")
assert not any(name in sys.modules for name in ("olm", "_libolm", "cryptography", "Crypto"))
''')

    def test_nist_aes256_ctr_streaming_and_inplace_boundaries(self):
        self.execute('''
from standalone.matrix_crypto_native import AES,Counter
# NIST SP800-38A F.5.5 AES-256 CTR, four independently published blocks.
key=bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
iv=int("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",16)
plain=bytes.fromhex("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710")
expected=bytes.fromhex("601ec313775789a5b7a7f504bbf3d228" "f443e3ca4d62b59aca84e990cacaf5c5"
                       "2b0930daa23de94ce87017ba2d84988d" "dfc9c58db67aada613c2dd08457941a6")
for boundaries in ((64,),(1,15,7,9,32)):
    cipher=AES.new(key,AES.MODE_CTR,counter=Counter.new(128,initial_value=iv))
    position=0;parts=[]
    for length in boundaries:
        parts.append(cipher.encrypt(plain[position:position+length]));position+=length
    assert b"".join(parts)==expected
buffer=bytearray(expected)
cipher=AES.new(key,AES.MODE_CTR,counter=Counter.new(128,initial_value=iv))
assert cipher.decrypt(buffer,buffer) is None and buffer==plain
prefix=bytes.fromhex("f0f1f2f3f4f5f6f7")
cipher=AES.new(key,AES.MODE_CTR,counter=Counter.new(64,prefix=prefix,initial_value=iv&((1<<64)-1)))
assert cipher.encrypt(plain)==expected
''')

    def test_upstream_megolm_ratchet_vectors_and_counter_wrap(self):
        self.execute('''
from standalone.matrix_crypto_group import Megolm
ratchet = Megolm(b"0123456789ABCDEF"*8)
ratchet.advance_to(1)
assert ratchet.data[:96] == b"0123456789ABCDEF"*6
assert ratchet.data[96:].hex() == "ba9cd955741d1c162323ec825e7c5ce889bbb423a18f23828fb2090d6e2af86a"
ratchet.advance_to(0x1000000)
assert ratchet.data.hex() == (
"54022d7dc0298e1637e21c97153092f933c056ff74fe1b922d971f2482c2859c"
"7004c01ee49bd6efe0073525af9b1632c5be726d12349cc5bd472bdc2df6540f"
"3112591194fda617e568c683101eaecd7eddd6de1fbc0767ae34da1a09a54eab"
"ba9cd955741d1c162323ec825e7c5ce889bbb423a18f23828fb2090d6e2af86a")
ratchet.advance_to(0x1041506)
assert ratchet.data.hex() == (
"54022d7dc0298e1637e21c97153092f933c056ff74fe1b922d971f2482c2859c"
"55588df5b7a48878428927868164589f3663447b51edc3595b036ca604c46dcd"
"5c54850bfa98a1fd79a9df1cbe8fc5681937d30c85c8c31f7bb828816cf9ff3b"
"956cbf807e65126a49558d45c84a2e4cd56f03e24416b98e1cfd97c206aa907a")
first, second = Megolm(b"0123456789ABCDEF"*8,0xffffffff), Megolm(b"0123456789ABCDEF"*8)
first.advance_to(0x1000000); second.advance_to(0x2000000)
assert first.data == second.data
''')

    def test_upstream_signed_megolm_message_and_native_pickle_restore(self):
        self.execute('''
from unittest.mock import patch
from standalone.matrix_crypto import OlmGroupSessionError, b64decode, b64encode, pickle_decode, pickle_encode
from standalone.matrix_crypto_group import InboundGroupSession, OutboundGroupSession
key = ("AgAAAAAwMTIzNDU2Nzg5QUJERUYwMTIzNDU2Nzg5QUJDREVGMDEyMzQ1Njc4OUFCREVGM"
"DEyMzQ1Njc4OUFCQ0RFRjAxMjM0NTY3ODlBQkRFRjAxMjM0NTY3ODlBQkNERUYwMTIzND"
"U2Nzg5QUJERUYwMTIzNDU2Nzg5QUJDREVGMDEyMw0bdg1BDq4Px/slBow06q8n/B9WBfw"
"WYyNOB8DlUmXGGwrFmaSb9bR/eY8xgERrxmP07hFmD9uqA2p8PMHdnV5ysmgufE6oLZ5+"
"8/mWQOW3VVTnDIlnwd8oHUYRuk8TCQ")
message = ("AwgAEhAcbh6UpbByoyZxufQ+h2B+8XHMjhR69G8F4+qjMaFlnIXusJZX3r8LnRORG9T3D"
"XFdbVuvIWrLyRfm4i8QRbe8VPwGRFG57B1CtmxanuP8bHtnnYqlwPsD")
inbound = InboundGroupSession(key)
assert inbound.decrypt(message) == ("Message",0)
exported = InboundGroupSession.import_session(inbound.export_session(0))
assert not exported._verified and exported.decrypt(message) == ("Message",0) and exported._verified
# The upstream deterministic initial seed reproduces the independently published
# session share and ciphertext byte for byte, including the Ed25519 signature.
random = (b"0123456789ABDEF0123456789ABCDEF"*6)[:160]
with patch("standalone.matrix_crypto_group.os.urandom",return_value=random):
    outbound = OutboundGroupSession()
assert outbound.session_key == key
assert outbound.encrypt("Message") == message
in_pickle, out_pickle = inbound.pickle("fixture"), outbound.pickle("fixture")
assert InboundGroupSession.from_pickle(in_pickle,"fixture").pickle("fixture") == in_pickle
restored = OutboundGroupSession.from_pickle(out_pickle,"fixture")
assert restored.pickle("fixture") == out_pickle and restored.id == inbound.id
for index in range(1,8):
    encrypted = restored.encrypt("text "+str(index))
    assert inbound.decrypt(encrypted) == ("text "+str(index),index)
assert inbound.decrypt(message) == ("Message",0)
for kind, value in ((InboundGroupSession,in_pickle),(OutboundGroupSession,out_pickle)):
    try: kind.from_pickle(value,"wrong")
    except OlmGroupSessionError: pass
    else: raise AssertionError("Incorrect pickle key accepted")
    raw = pickle_decode(value,"fixture",OlmGroupSessionError).data
    try: kind.from_pickle(pickle_encode(raw+b"junk","fixture"),"fixture")
    except OlmGroupSessionError as exc: assert str(exc)=="PICKLE_EXTRA_DATA"
    else: raise AssertionError("Trailing pickle bytes accepted")
raw = bytearray(b64decode(message));raw[-1] ^= 1
try: inbound.decrypt(b64encode(raw))
except OlmGroupSessionError as exc: assert str(exc)=="BAD_SIGNATURE"
else: raise AssertionError("Corrupted group signature accepted")
# Imported history never expands backwards across the exported index.
later = InboundGroupSession.import_session(inbound.export_session(5))
try: later.decrypt(message)
except OlmGroupSessionError as exc: assert str(exc)=="UNKNOWN_MESSAGE_INDEX"
else: raise AssertionError("Group history before shared index was decrypted")
''')

    def test_upstream_native_olm_pickles_versions_one_and_logging(self):
        self.execute('''
from standalone.matrix_olm import Session, OlmSessionError
legacy = ("wkEpwMgiAqD7B1/Lw2cKYYDcUZVOd9QHes7ZroWxr/Rp/nWEAySgRsIu/a54YhO67rwitr"
"Lpos7tFxxK9IZ7pKB1qrR1coVWIt78V9lp9WgmBAvxHBSY+tu1lkL/JjLi963/yFdPancZ"
"+WHMVfaKlV3gWGpo7EfNK6qAOxI1Ea/eCsE2sYrsHEDvLLGlKAA9E56rmmoe2w6TKzsQjs"
"ZM2/XT2eJ82EgMO9pL02iLElXWmGNv72Ut7DouR0pQIT50HIEEKcFxYcoTb3WCfJD76Coe"
"sE4kx+TA6d45Xu1bwQNNkTGF+nCCu/GmKY+sECXbz9U6WhxG0YdF9Z4T8YkWYAgpKNS0FW"
"RV")
logging_pickle = ("m+DS/q34MXpw2xp50ZD0B7val1mlMpQXo0mx+VPje0weFYRRuuZQBdJgcFPEpi2MVSpA4c"
"qgqHyj2/bU7/lz+BXkEBrCFVx0BJidxXfOLDW4TNtRhLS1YHJNGP8GvTg1+dCytBTLsCdm"
"5f945Eq1U/pY3Cg96YTUufFP6EYrfRoDbAsRHc+h+wKKftQv+W44yUmRhcCemGHtpxk3UQ"
"AMCI7EBv9BvveyZMy3p9qZ3xvFK34Hef+R7gjtFycz7Nk/4UF46sT3cTmUlXz9iFW4uz2F"
"rTI1Wjym+l0DadsbSpHSUjmp9zt4qRP2UjwfZ5QNLv+cdObIfqFsiThGu/PlKigdF4SLHr"
"nG")
for value in (legacy,logging_pickle):
    restored = Session.from_pickle(value,"secret_key")
    assert restored.pickle("secret_key") == legacy.encode()
    assert restored._root.hex() == "49d640dc96b80176694af69fc4b8ca9fac49aecbd697d01fd8bee1ed2693b6c9"
    assert restored._alice_identity.hex() == "7326b58623a3f7bd8da11a1bab51f432c02a7430241b326e9fc8916a21eb257e"
    assert restored._alice_base.hex() == "0ab4b30bde20bd374ceccc72861660f0fd046f7516900796c3e5de41c598316c"
    assert restored._bob_one_time.hex() == "585dba930b10d90d81702c715f4085d07c42b0cd2d676010bb6086c86c4cc618"
    assert restored._sender.public.hex() == "f77a03eaa9b301fa7d2a5aa6b50286906de12cc96044f526dbbcb12839ad7003"
    assert restored._sender.private.hex() == "d945c6ed4c7c277117adf11fb133a7936d287afe97c0b3ac989644b4490d4f31"
    assert restored._sender.index == 0 and not restored._received
    try: Session.from_pickle(value,"incorrect")
    except OlmSessionError: pass
    else: raise AssertionError("Wrong Olm pickle key accepted")
''')

    def test_olm_double_ratchet_reordering_replay_authentication_and_restart(self):
        self.execute('''
from standalone.matrix_olm import Account, Session, InboundSession, OutboundSession, OlmMessage, OlmPreKeyMessage, OlmSessionError
from standalone.matrix_crypto import b64decode,b64encode
alice, bob = Account(),Account()
bob.generate_one_time_keys(2)
one_time = next(iter(bob.one_time_keys["curve25519"].values()))
outbound = OutboundSession(alice,bob.identity_keys["curve25519"],one_time)
first = outbound.encrypt("Hello, World")
assert isinstance(first,OlmPreKeyMessage)
inbound = InboundSession(bob,first,alice.identity_keys["curve25519"])
assert inbound.matches(first) and inbound.matches(first,alice.identity_keys["curve25519"])
assert not inbound.matches(first,bob.identity_keys["curve25519"])
assert inbound.decrypt(first) == "Hello, World" and inbound.id == outbound.id
bob.remove_one_time_keys(inbound)
assert len(bob.one_time_keys["curve25519"]) == 1
reply = inbound.encrypt("Hello back")
assert isinstance(reply,OlmMessage) and not isinstance(reply,OlmPreKeyMessage)
assert outbound.decrypt(reply) == "Hello back"
for index in range(8):
    sent = outbound.encrypt("alice "+str(index))
    assert inbound.decrypt(sent) == "alice "+str(index)
    sent = inbound.encrypt("bob "+str(index))
    assert outbound.decrypt(sent) == "bob "+str(index)
pending = [outbound.encrypt("out-of-order "+str(index)) for index in range(10)]
state = inbound.pickle("checkpoint")
raw = bytearray(b64decode(pending[-1].ciphertext));raw[-1] ^= 1
try: inbound.decrypt(OlmMessage(b64encode(raw)))
except OlmSessionError as exc: assert str(exc)=="BAD_MESSAGE_MAC"
else: raise AssertionError("Tampered Olm message accepted")
assert inbound.pickle("checkpoint") == state, "Unauthenticated message advanced ratchet"
assert inbound.decrypt(pending[-1]) == "out-of-order 9"
inbound = Session.from_pickle(inbound.pickle("checkpoint"),"checkpoint")
outbound = Session.from_pickle(outbound.pickle("checkpoint"),"checkpoint")
for index in reversed(range(9)):
    assert inbound.decrypt(pending[index]) == "out-of-order "+str(index)
try: inbound.decrypt(pending[0])
except OlmSessionError as exc: assert str(exc)=="BAD_MESSAGE_MAC"
else: raise AssertionError("Olm replay accepted")
assert outbound.decrypt(inbound.encrypt("survives restart")) == "survives restart"
assert inbound.decrypt(outbound.encrypt("new chain")) == "new chain"
assert not any(name in sys.modules for name in ("olm","_libolm","Crypto","cryptography"))
''')

    def test_account_fallback_keys_published_state_and_legacy_pickle_rejection(self):
        self.execute('''
from standalone.matrix_olm import Account, InboundSession, OutboundSession, OlmAccountError, PkSigning, ed25519_verify
from standalone.matrix_crypto import pickle_decode,pickle_encode,u32
alice,bob = Account(),Account()
bob.generate_fallback_key()
fallback = next(iter(bob.fallback_key["curve25519"].values()))
bob.generate_one_time_keys(3)
bob.mark_keys_as_published()
assert bob.one_time_keys == {"curve25519":{}} and bob.fallback_key == {"curve25519":{}}
for text in ("first fallback","second fallback"):
    outbound = OutboundSession(alice,bob.identity_keys["curve25519"],fallback)
    initial = outbound.encrypt(text)
    inbound = InboundSession(bob,initial)
    assert inbound.decrypt(initial)==text
    bob.remove_one_time_keys(inbound)
pickled = bob.pickle("account-key")
restored = Account.from_pickle(pickled,"account-key")
assert restored.pickle("account-key")==pickled and restored.identity_keys==bob.identity_keys
assert len(restored._one_time)==3 and all(key.published for key in restored._one_time)
restored.generate_fallback_key();restored.forget_old_fallback_key()
assert len(restored._fallback)==1
signature = alice.sign("Matrix signed object")
ed25519_verify(alice.identity_keys["ed25519"],"Matrix signed object",signature)
signer = PkSigning(bytes(range(32)))
ed25519_verify(signer.public_key,"cross signing",signer.sign("cross signing"))
legacy = ("x3h9er86ygvq56pM1yesdAxZou4ResPQC9Rszk/fhEL9JY/umtZ2N/foL/SUgVXS"
"v0IxHHZTafYjDdzJU9xr8dQeBoOTGfV9E/lCqDGBnIlu7SZndqjEKXtzGyQr4sP4"
"K/A/8TOu9iK2hDFszy6xETiousHnHgh2ZGbRUh4pQx+YMm8ZdNZeRnwFGLnrWyf9"
"O5TmXua1FcU")
try: Account.from_pickle(legacy)
except OlmAccountError as exc: assert str(exc)=="BAD_LEGACY_ACCOUNT_PICKLE"
else: raise AssertionError("Compromised old Ed25519 account format accepted")
fresh = Account()
raw = pickle_decode(fresh.pickle("version2"),"version2",OlmAccountError).data
v2 = pickle_encode(u32(2)+raw[4:168]+raw[-4:],"version2")
assert Account.from_pickle(v2,"version2").identity_keys==fresh.identity_keys
''')


class OriginalMautrixCryptoTests(unittest.TestCase):
    DEPENDENCIES = ("mautrix", "aiosqlite", "base58", "unpaddedbase64", "ecdsa", "tlslite-ng", "typing-extensions")

    @classmethod
    def setUpClass(cls):
        crypto_fixture.GoogleAuthCryptoTests.setUpClass.__func__(cls)

    def execute(self, program):
        crypto_fixture.GoogleAuthCryptoTests.execute(self, program)

    def test_original_mautrix_signatures_secret_storage_and_streamed_attachments(self):
        self.execute('''
from mautrix.crypto.account import OlmAccount
from mautrix.crypto.signature import verify_signature_json
from mautrix.crypto.ssss.key import Key
from mautrix.crypto.attachments import encrypt_attachment,decrypt_attachment,encrypted_attachment_generator,inplace_encrypt_attachment
from mautrix.types import UserID,DeviceID
account=OlmAccount()
device=account.get_device_keys(UserID("@fixture:example.invalid"),DeviceID("FIXTURE"))
data=device.serialize()
assert verify_signature_json(data,UserID("@fixture:example.invalid"),DeviceID("FIXTURE"),account.signing_key)
data["user_id"]="@attacker:example.invalid"
assert not verify_signature_json(data,UserID("@fixture:example.invalid"),DeviceID("FIXTURE"),account.signing_key)
key=Key.generate()
assert key.metadata.verify_recovery_key(key.id,key.recovery_key).key==key.key
secret=key.encrypt("m.cross_signing.master",b"fixture cross-signing seed")
assert key.decrypt("m.cross_signing.master",secret)==b"fixture cross-signing seed"
try:key.decrypt("m.cross_signing.self_signing",secret)
except ValueError:pass
else:raise AssertionError("Secret-storage context substitution accepted")
data=b"native encrypted Matrix attachment"*100
encrypted,description=encrypt_attachment(data)
assert decrypt_attachment(encrypted,description.key.key,description.hashes["sha256"],description.iv)==data
values=list(encrypted_attachment_generator([data[:13],data[13:79],data[79:]]))
assert decrypt_attachment(b"".join(values[:-1]),values[-1].key.key,values[-1].hashes["sha256"],values[-1].iv)==data
buffer=bytearray(data)
description=inplace_encrypt_attachment(buffer)
assert bytes(buffer)!=data
assert decrypt_attachment(buffer,description.key.key,description.hashes["sha256"],description.iv,inplace=True)==data
assert bytes(buffer)==data
assert not any(name in sys.modules for name in ("olm","_libolm","Crypto","Cryptodome","cryptography","asyncpg"))
''')

    def test_original_sqlite_crypto_store_account_sessions_trust_and_replay_persist(self):
        self.execute('''
import asyncio
from mautrix.crypto import OlmAccount,InboundGroupSession,OutboundGroupSession,Session
from mautrix.crypto.store.asyncpg import PgCryptoStore
from mautrix.util.async_db import Database
from mautrix.errors import EncryptionError
from mautrix.types import UserID,DeviceID,RoomID,SessionID,IdentityKey,SigningKey,DeviceIdentity,TrustState,EventID
async def exercise():
    db=Database.create("sqlite:///"+str(fixtures/"crypto.db"),upgrade_table=PgCryptoStore.upgrade_table)
    await db.start()
    try:
        store=PgCryptoStore("fixture","fixture-pickle-key",db)
        await store.open()
        account=OlmAccount()
        await store.put_account(account)
        await store.put_device_id(DeviceID("FIXTURE"))
        outbound=OutboundGroupSession(RoomID("!fixture:example.invalid"))
        try:outbound.encrypt("unshared must fail")
        except EncryptionError:pass
        else:raise AssertionError("Unshared group session encrypted a message")
        outbound.shared=True
        inbound=InboundGroupSession(outbound.session_key,account.signing_key,account.identity_key,RoomID("!fixture:example.invalid"))
        await store.put_group_session(RoomID("!fixture:example.invalid"),account.identity_key,SessionID(inbound.id),inbound)
        await store.add_outbound_group_session(outbound)
        encrypted=outbound.encrypt("stored encrypted message")
        await store.update_outbound_group_session(outbound)
        assert inbound.decrypt(encrypted)==("stored encrypted message",0)
        assert await store.validate_message_index(account.identity_key,SessionID(inbound.id),EventID("$fixture"),0,123)
        assert await store.validate_message_index(account.identity_key,SessionID(inbound.id),EventID("$fixture"),0,123)
        assert not await store.validate_message_index(account.identity_key,SessionID(inbound.id),EventID("$replay"),0,124)
        device=DeviceIdentity(UserID("@fixture:example.invalid"),DeviceID("OTHER"),account.identity_key,account.signing_key,
                              TrustState.VERIFIED,False,"Fixture verified device")
        await store.put_devices(device.user_id,{device.device_id:device})
        restored=PgCryptoStore("fixture","fixture-pickle-key",db)
        await restored.open()
        assert (await restored.get_account()).identity_key==account.identity_key
        assert await restored.get_device_id()=="FIXTURE"
        group=await restored.get_group_session(RoomID("!fixture:example.invalid"),SessionID(inbound.id))
        assert group.decrypt(encrypted)==("stored encrypted message",0)
        sent=await restored.get_outbound_group_session(RoomID("!fixture:example.invalid"))
        assert sent.message_index==1
        devices=await restored.get_devices(device.user_id)
        assert devices[device.device_id].trust==TrustState.VERIFIED
        assert not await restored.validate_message_index(account.identity_key,SessionID(inbound.id),EventID("$different"),0,123)
    finally:
        await db.stop()
asyncio.run(exercise())
assert "asyncpg" not in sys.modules and "olm" not in sys.modules
''')


if __name__ == "__main__":
    unittest.main()
