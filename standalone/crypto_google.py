"""Explicit google-auth signing and certificate boundary using reviewed Python code.

The pinned google-auth sources select these concrete signer/verifier classes.
OAuth, credential refresh, claims validation and HTTP behavior remain upstream.
RSA is tlslite-ng, ECDSA is python-ecdsa, and certificate ASN.1 is pyasn1-modules.
These Python signing engines do not promise constant-time arithmetic.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import math
import re
import textwrap

import ecdsa
from google.auth import _helpers
from google.auth.crypt import base
from tlslite.utils.python_rsakey import Python_RSAKey


# google-auth's public wrapper dispatches accepted key objects to this backend.
RSAPrivateKey = RSAPublicKey = Python_RSAKey


def _pem(marker: str, content: bytes) -> bytes:
    encoded = "\n".join(textwrap.wrap(base64.b64encode(content).decode("ascii"), 64))
    return f"-----BEGIN {marker}-----\n{encoded}\n-----END {marker}-----\n".encode("ascii")


def _pem_bytes(value: bytes | str, marker: str) -> bytes:
    content = _helpers.to_bytes(value)
    expression = rb"-----BEGIN " + marker.encode("ascii") + rb"-----\s*([A-Za-z0-9+/=\s]+?)\s*-----END " + marker.encode("ascii") + rb"-----"
    match = re.search(expression, content)
    if match is None:
        raise ValueError(f"Expected a PEM {marker}")
    try:
        return base64.b64decode(re.sub(rb"\s", b"", match.group(1)), validate=True)
    except ValueError as exc:
        raise ValueError(f"Invalid PEM {marker}") from exc


@dataclass(frozen=True)
class Certificate:
    der: bytes
    parsed: object

    def public_key_der(self) -> bytes:
        from pyasn1.codec.der.encoder import encode
        return encode(self.parsed["tbsCertificate"]["subjectPublicKeyInfo"])


def parse_certificate(value: bytes | str) -> Certificate:
    from pyasn1.codec.der.decoder import decode
    from pyasn1.error import PyAsn1Error
    from pyasn1_modules.rfc5280 import Certificate as CertificateSchema

    der = _pem_bytes(value, "CERTIFICATE")
    try:
        parsed, trailing = decode(der, asn1Spec=CertificateSchema())
    except PyAsn1Error as exc:
        raise ValueError("Invalid X.509 certificate") from exc
    if trailing:
        raise ValueError("Trailing X.509 certificate data")
    return Certificate(der, parsed)


def certificate_uris(certificate: Certificate) -> list[str]:
    from pyasn1.codec.der.decoder import decode
    from pyasn1_modules.rfc5280 import SubjectAltName

    result = []
    seen = False
    for extension in certificate.parsed["tbsCertificate"]["extensions"]:
        if str(extension["extnID"]) != "2.5.29.17":
            continue
        if seen:
            raise ValueError("Duplicate certificate subject alternative name extension")
        seen = True
        names, trailing = decode(extension["extnValue"].asOctets(), asn1Spec=SubjectAltName())
        if trailing:
            raise ValueError("Trailing subject alternative name data")
        result.extend(str(name["uniformResourceIdentifier"]) for name in names
                      if name.getName() == "uniformResourceIdentifier")
    return result


def _public_key_bytes(value: bytes | str) -> bytes:
    content = _helpers.to_bytes(value)
    if b"-----BEGIN CERTIFICATE-----" in content:
        return _pem("PUBLIC KEY", parse_certificate(content).public_key_der())
    return content


def _validate_rsa_key(key: Python_RSAKey, *, private: bool = False) -> None:
    if not isinstance(key, Python_RSAKey) or not (3 <= key.e < key.n and key.e % 2 and key.n % 2):
        raise ValueError("Invalid RSA public key numbers")
    if not private:
        return
    from tlslite.utils.cryptomath import isPrime
    if (not key.hasPrivateKey() or key.p <= 2 or key.q <= 2 or key.p == key.q
            or key.n != key.p * key.q or not 0 < key.d < key.n
            or (key.e * key.d) % math.lcm(key.p - 1, key.q - 1) != 1
            or key.dP != key.d % (key.p - 1) or key.dQ != key.d % (key.q - 1)
            or (key.qInv * key.q) % key.p != 1 or not isPrime(key.p) or not isPrime(key.q)):
        raise ValueError("Invalid RSA private key numbers")


class RSASigner(base.Signer, base.FromServiceAccountMixin):
    def __init__(self, private_key: Python_RSAKey, key_id=None):
        _validate_rsa_key(private_key, private=True)
        self._key = private_key
        self._key_id = key_id

    @property
    def key_id(self):
        return self._key_id

    def sign(self, message):
        return bytes(self._key.hashAndSign(bytearray(_helpers.to_bytes(message)), rsaScheme="PKCS1", hAlg="sha256"))

    @classmethod
    def from_string(cls, key, key_id=None):
        text = _helpers.to_bytes(key).decode("utf-8")
        try:
            parsed = Python_RSAKey.parsePEM(text)
        except (SyntaxError, ValueError, AssertionError, IndexError) as exc:
            raise ValueError("Invalid RSA private key") from exc
        return cls(parsed, key_id=key_id)

    def __getstate__(self):
        from ecdsa.der import encode_integer, encode_sequence
        key = self._key
        data = encode_sequence(*(encode_integer(value) for value in
                                 (0, key.n, key.e, key.d, key.p, key.q, key.dP, key.dQ, key.qInv)))
        return {"_key": _pem("RSA PRIVATE KEY", data), "_key_id": self._key_id}

    def __setstate__(self, state):
        restored = self.from_string(state["_key"], key_id=state["_key_id"])
        self.__dict__.update(restored.__dict__)


class RSAVerifier(base.Verifier):
    def __init__(self, public_key: Python_RSAKey):
        _validate_rsa_key(public_key)
        self._pubkey = Python_RSAKey(n=public_key.n, e=public_key.e)

    def verify(self, message, signature):
        try:
            return bool(self._pubkey.hashAndVerify(bytearray(_helpers.to_bytes(signature)),
                                                  bytearray(_helpers.to_bytes(message)),
                                                  rsaScheme="PKCS1", hAlg="sha256"))
        except (ValueError, IndexError):
            return False

    @classmethod
    def from_string(cls, public_key):
        from pyasn1.error import PyAsn1Error

        content = _public_key_bytes(public_key)
        try:
            if b"-----BEGIN RSA PUBLIC KEY-----" in content:
                from pyasn1.codec.der.decoder import decode
                from pyasn1_modules.rfc8017 import RSAPublicKey
                key, trailing = decode(_pem_bytes(content, "RSA PUBLIC KEY"), asn1Spec=RSAPublicKey())
                if trailing:
                    raise ValueError("Trailing RSA public key data")
                parsed = Python_RSAKey(n=int(key["modulus"]), e=int(key["publicExponent"]))
            else:
                if b"-----BEGIN PUBLIC KEY-----" not in content:
                    raise ValueError("Expected a public key or certificate")
                parsed = Python_RSAKey.parsePEM(content.decode("utf-8"))
        except (SyntaxError, ValueError, AssertionError, IndexError, PyAsn1Error) as exc:
            raise ValueError("Invalid RSA public key") from exc
        return cls(parsed)


def _ec_parameters(key):
    if key.curve == ecdsa.NIST256p:
        return "ES256", hashlib.sha256, 64
    if key.curve == ecdsa.NIST384p:
        return "ES384", hashlib.sha384, 96
    raise ValueError("Google authentication requires P-256 or P-384 ECDSA keys")


class EsSigner(base.Signer, base.FromServiceAccountMixin):
    def __init__(self, private_key: ecdsa.SigningKey, key_id=None):
        if not isinstance(private_key, ecdsa.SigningKey):
            raise TypeError("Expected an ECDSA private key")
        self._algorithm, self._hash, _ = _ec_parameters(private_key)
        self._key = private_key
        self._key_id = key_id

    @property
    def algorithm(self):
        return self._algorithm

    @property
    def key_id(self):
        return self._key_id

    def sign(self, message):
        return self._key.sign_deterministic(_helpers.to_bytes(message), hashfunc=self._hash,
                                            sigencode=ecdsa.util.sigencode_string)

    @classmethod
    def from_string(cls, key, key_id=None):
        try:
            parsed = ecdsa.SigningKey.from_pem(_helpers.to_bytes(key))
        except (ecdsa.der.UnexpectedDER, ecdsa.MalformedPointError, ValueError) as exc:
            raise ValueError("Invalid ECDSA private key") from exc
        return cls(parsed, key_id=key_id)

    def __getstate__(self):
        return {"_key": self._key.to_pem(format="pkcs8"), "_key_id": self._key_id}

    def __setstate__(self, state):
        restored = self.from_string(state["_key"], key_id=state["_key_id"])
        self.__dict__.update(restored.__dict__)


class EsVerifier(base.Verifier):
    def __init__(self, public_key: ecdsa.VerifyingKey):
        if not isinstance(public_key, ecdsa.VerifyingKey):
            raise TypeError("Expected an ECDSA public key")
        _, self._hash, self._signature_size = _ec_parameters(public_key)
        self._pubkey = public_key

    def verify(self, message, signature):
        signature = _helpers.to_bytes(signature)
        if len(signature) != self._signature_size:
            return False
        try:
            return self._pubkey.verify(signature, _helpers.to_bytes(message), hashfunc=self._hash,
                                       sigdecode=ecdsa.util.sigdecode_string)
        except (ecdsa.BadSignatureError, ValueError):
            return False

    @classmethod
    def from_string(cls, public_key):
        try:
            parsed = ecdsa.VerifyingKey.from_pem(_public_key_bytes(public_key))
        except (ecdsa.der.UnexpectedDER, ecdsa.MalformedPointError, ValueError) as exc:
            raise ValueError("Invalid ECDSA public key") from exc
        return cls(parsed)
