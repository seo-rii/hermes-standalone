"""RFC 9180 P-256 HPKE and RFC 9420 suite 0x0002 cryptographic operations.

The existing embedded ecdsa and tlslite-ng packages provide scalar arithmetic
and AES. All protocol constructions are implemented here without native code.
"""

from __future__ import annotations

import hashlib
import hmac
import os

from standalone.crypto import AESGCM, InvalidTag
from .mls_codec import opaque, uint


HASH_LENGTH = 32
SUITE = 2
KEM_SUITE = b"KEM\x00\x10"
HPKE_SUITE = b"HPKE\x00\x10\x00\x01\x00\x01"
P256_ORDER = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551


def extract(salt, material):
    return hmac.digest(salt or bytes(32), material, "sha256")


def expand(secret, info, length):
    if type(length) is not int or not 0 <= length <= 8160:
        raise ValueError("HKDF-SHA256 output length exceeds its counter range")
    previous = b""
    output = bytearray()
    for counter in range(1, (length + 31) // 32 + 1):
        previous = hmac.digest(secret, previous + info + bytes([counter]), "sha256")
        output.extend(previous)
    return bytes(output[:length])


def expand_label(secret, label, context=b"", length=32):
    if isinstance(label, str):
        label = label.encode("ascii")
    return expand(secret, uint(length, 2) + opaque(b"MLS 1.0 " + label) + opaque(context), length)


def derive_secret(secret, label):
    return expand_label(secret, label)


def ref_hash(label, value):
    if isinstance(label, str):
        label = label.encode("ascii")
    return hashlib.sha256(opaque(label) + opaque(value)).digest()


def _private(value):
    from ecdsa import NIST256p, SigningKey
    value = bytes(value)
    if len(value) != 32 or not 0 < int.from_bytes(value, "big") < P256_ORDER:
        raise ValueError("Invalid P-256 private key")
    return SigningKey.from_string(value, curve=NIST256p, hashfunc=hashlib.sha256)


def _public(value):
    from ecdsa import NIST256p, VerifyingKey
    from ecdsa.errors import MalformedPointError
    value = bytes(value)
    if len(value) != 65 or value[0] != 4:
        raise ValueError("P-256 public keys require uncompressed SEC1 encoding")
    try:
        return VerifyingKey.from_string(value, curve=NIST256p, hashfunc=hashlib.sha256,
                                        validate_point=True, valid_encodings={"uncompressed"})
    except MalformedPointError as error:
        raise ValueError("P-256 public point is not on the curve") from error


def public_key(private):
    return _private(private).verifying_key.to_string("uncompressed")


def generate_key():
    while True:
        private = os.urandom(32)
        if 0 < int.from_bytes(private, "big") < P256_ORDER:
            return private, public_key(private)


def dh(private, public):
    point = _public(public).pubkey.point * _private(private).privkey.secret_multiplier
    if point.x() is None:
        raise ValueError("Invalid P-256 shared point")
    return int(point.x()).to_bytes(32, "big")


def sign(private, content):
    from ecdsa.util import sigencode_der
    return _private(private).sign_deterministic(content, hashfunc=hashlib.sha256, sigencode=sigencode_der)


def verify(public, content, signature):
    from ecdsa import BadSignatureError
    from ecdsa.util import sigdecode_der
    try:
        return _public(public).verify(signature, content, hashfunc=hashlib.sha256, sigdecode=sigdecode_der)
    except BadSignatureError:
        return False


def sign_label(private, label, content):
    return sign(private, opaque(b"MLS 1.0 " + label.encode("ascii")) + opaque(content))


def verify_label(public, label, content, signature):
    return verify(public, opaque(b"MLS 1.0 " + label.encode("ascii")) + opaque(content), signature)


def _labeled_extract(suite, salt, label, ikm):
    return extract(salt, b"HPKE-v1" + suite + label + ikm)


def _labeled_expand(suite, secret, label, info, length):
    return expand(secret, uint(length, 2) + b"HPKE-v1" + suite + label + info, length)


def derive_key_pair(ikm):
    secret = _labeled_extract(KEM_SUITE, b"", b"dkp_prk", ikm)
    for counter in range(256):
        candidate = _labeled_expand(KEM_SUITE, secret, b"candidate", bytes([counter]), 32)
        if 0 < int.from_bytes(candidate, "big") < P256_ORDER:
            return candidate, public_key(candidate)
    raise ValueError("P-256 HPKE key derivation exhausted its rejection counter")


def _kem_shared(dh_value, enc, recipient):
    secret = _labeled_extract(KEM_SUITE, b"", b"eae_prk", dh_value)
    return _labeled_expand(KEM_SUITE, secret, b"shared_secret", enc + recipient, 32)


class HPKE:
    def __init__(self, shared_secret, info):
        psk_id_hash = _labeled_extract(HPKE_SUITE, b"", b"psk_id_hash", b"")
        info_hash = _labeled_extract(HPKE_SUITE, b"", b"info_hash", info)
        context = b"\x00" + psk_id_hash + info_hash
        secret = _labeled_extract(HPKE_SUITE, shared_secret, b"secret", b"")
        self.key = _labeled_expand(HPKE_SUITE, secret, b"key", context, 16)
        self.base_nonce = _labeled_expand(HPKE_SUITE, secret, b"base_nonce", context, 12)
        self.exporter_secret = _labeled_expand(HPKE_SUITE, secret, b"exp", context, 32)
        self.sequence = 0

    def _nonce(self):
        if self.sequence >= (1 << 96) - 1:
            raise ValueError("HPKE message sequence is exhausted")
        return bytes(a ^ b for a, b in zip(self.base_nonce, uint(self.sequence, 12)))

    def seal(self, plaintext, aad=b""):
        result = AESGCM(self.key).encrypt(self._nonce(), plaintext, aad)
        self.sequence += 1
        return result

    def open(self, ciphertext, aad=b""):
        result = AESGCM(self.key).decrypt(self._nonce(), ciphertext, aad)
        self.sequence += 1
        return result

    def export(self, context, length):
        return _labeled_expand(HPKE_SUITE, self.exporter_secret, b"sec", context, length)


def setup_sender(recipient, info, *, private=None):
    _public(recipient)
    if private is None:
        private, enc = generate_key()
    else:
        enc = public_key(private)
    return enc, HPKE(_kem_shared(dh(private, recipient), enc, recipient), info)


def setup_receiver(private, enc, info):
    recipient = public_key(private)
    return HPKE(_kem_shared(dh(private, enc), enc, recipient), info)


def seal_label(recipient, label, context, plaintext):
    info = opaque(b"MLS 1.0 " + label.encode("ascii")) + opaque(context)
    enc, sender = setup_sender(recipient, info)
    return enc, sender.seal(plaintext)


def open_label(private, label, context, enc, ciphertext):
    info = opaque(b"MLS 1.0 " + label.encode("ascii")) + opaque(context)
    return setup_receiver(private, enc, info).open(ciphertext)


def gcm8_seal(key, nonce, plaintext, aad=b""):
    if len(key) != 16:
        raise ValueError("DAVE requires AES-128-GCM")
    result = AESGCM(key).encrypt(nonce, plaintext, aad)
    return result[:-16], result[-16:-8]


def gcm8_open(key, nonce, ciphertext, tag, aad=b""):
    from tlslite.utils.python_aesgcm import new
    if len(key) != 16 or len(nonce) != 12 or len(tag) != 8:
        raise ValueError("Malformed DAVE AES-GCM key, nonce, or tag")
    # The pinned TLSLite GCM primitive authenticates the ciphertext with GHASH.
    # Its public open() fixes a 16-byte tag, whereas DAVE specifies eight bytes.
    cipher = new(bytearray(key))
    counter = bytearray(nonce + b"\x00\x00\x00\x01")
    mask = cipher._rawAesEncrypt(counter)
    expected = bytes(cipher._auth(bytearray(ciphertext), bytearray(aad), mask))[:8]
    if not hmac.compare_digest(expected, tag):
        raise InvalidTag("DAVE AES-GCM authentication failed")
    counter[-1] = 2
    cipher._ctr.counter = counter
    return bytes(cipher._ctr.decrypt(bytearray(ciphertext)))
