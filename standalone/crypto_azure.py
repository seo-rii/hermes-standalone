"""Qualified Azure/MSAL certificate operations using the reviewed Python engines.

The pinned SDK calls these functions directly. Token requests, credential chains,
assertion claims, tenant checks and caches remain the original Azure/MSAL code.
Python RSA arithmetic does not provide a constant-time execution guarantee.
"""

from __future__ import annotations

import hashlib

from tlslite.utils.python_rsakey import Python_RSAKey as RSAPrivateKey

from standalone.crypto_pem import encode_pem, private_key_der, pkcs12_contents


def load_pem_private_key(data, password=None):
    # google-auth's package initializes the concrete backend classes; importing
    # the backend first would create a private-module initialization cycle.
    from google.auth import crypt
    from standalone.crypto_google import _validate_rsa_key
    content, marker = private_key_der(bytes(data), password)
    try:
        key = RSAPrivateKey.parsePEM(encode_pem(content, marker).decode("ascii"))
        _validate_rsa_key(key, private=True)
    except (ValueError, SyntaxError, AssertionError, IndexError) as exc:
        raise ValueError("The certificate must include a valid RSA private key") from exc
    return key


def load_pem_x509_certificate(data):
    from google.auth import crypt
    from standalone.crypto_google import parse_certificate
    return parse_certificate(data)


def certificate_fingerprint(certificate, hash_name):
    if hash_name not in ("sha1", "sha256"):
        raise ValueError("Unsupported Azure certificate fingerprint hash")
    return hashlib.new(hash_name, certificate.der).digest()


def certificate_pem(certificate):
    return encode_pem(certificate.der, "CERTIFICATE")


def private_key_pem(key):
    from ecdsa.der import encode_integer, encode_sequence, encode_oid, encode_octet_string
    numbers = (0, key.n, key.e, key.d, key.p, key.q, key.dP, key.dQ, key.qInv)
    rsa = encode_sequence(*(encode_integer(value) for value in numbers))
    algorithm = encode_sequence(encode_oid(1, 2, 840, 113549, 1, 1, 1), b"\x05\x00")
    return encode_pem(encode_sequence(encode_integer(0), algorithm, encode_octet_string(rsa)), "PRIVATE KEY")


def sign_rs256(key, plaintext):
    return bytes(key.hashAndSign(bytearray(plaintext), rsaScheme="PKCS1", hAlg="sha256"))


def sign_ps256(key, plaintext):
    return bytes(key.hashAndSign(bytearray(plaintext), rsaScheme="PSS", hAlg="sha256", sLen=32))


def load_key_and_certificates(data, password=None):
    keys, certs = pkcs12_contents(data, password)
    if len(keys) > 1:
        raise ValueError("PKCS12 certificate credentials must contain at most one private key")
    key = load_pem_private_key(encode_pem(keys[0], "PRIVATE KEY")) if keys else None
    certificates = [load_pem_x509_certificate(encode_pem(cert, "CERTIFICATE")) for cert in certs]
    if key is None:
        return None, None, certificates
    from google.auth import crypt
    from standalone.crypto_google import RSAVerifier
    for index, certificate in enumerate(certificates):
        try:
            public = RSAVerifier.from_string(certificate_pem(certificate))._pubkey
        except ValueError:
            continue
        if public.n == key.n and public.e == key.e:
            return key, certificate, certificates[:index] + certificates[index + 1:]
    if certificates:
        raise ValueError("PKCS12 private key does not match its certificate")
    return key, None, []
