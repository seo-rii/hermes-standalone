"""Pure Python primitives for the encrypted formats Hermes actually consumes.

AES comes from the pinned tlslite-ng implementation. This module implements
Fernet framing and RFC 5869 HKDF with stdlib HMAC, not a cryptography namespace
replacement. Authentication comparisons are constant time; the Python AES
implementation itself does not offer side-channel resistance.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import time


def _blocks(key: bytes, data: bytes) -> None:
    if len(key) not in (16, 24, 32):
        raise ValueError("AES key must contain 16, 24, or 32 bytes")
    if len(data) % 16:
        raise ValueError("AES data length must be a multiple of 16 bytes")


def aes_cbc_encrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    from tlslite.utils.python_aes import Python_AES

    _blocks(key, data)
    if len(iv) != 16:
        raise ValueError("AES-CBC IV must contain 16 bytes")
    return bytes(Python_AES(bytearray(key), 2, bytearray(iv)).encrypt(bytearray(data)))


def aes_cbc_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    from tlslite.utils.python_aes import Python_AES

    _blocks(key, data)
    if len(iv) != 16:
        raise ValueError("AES-CBC IV must contain 16 bytes")
    return bytes(Python_AES(bytearray(key), 2, bytearray(iv)).decrypt(bytearray(data)))


def aes_ecb_encrypt(key: bytes, data: bytes) -> bytes:
    from tlslite.utils.rijndael import Rijndael

    _blocks(key, data)
    cipher = Rijndael(bytearray(key), 16)
    return b"".join(bytes(cipher.encrypt(bytearray(data[start:start + 16]))) for start in range(0, len(data), 16))


def aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    from tlslite.utils.rijndael import Rijndael

    _blocks(key, data)
    cipher = Rijndael(bytearray(key), 16)
    return b"".join(bytes(cipher.decrypt(bytearray(data[start:start + 16]))) for start in range(0, len(data), 16))


class InvalidTag(Exception):
    """The ciphertext or associated data failed authentication."""


class AESGCM:
    """Hermes' AES-128/256-GCM boundary: 96-bit nonce and full 128-bit tag."""

    def __init__(self, key: bytes):
        if len(key) not in (16, 32):
            raise ValueError("This AES-GCM boundary requires a 128-bit or 256-bit key")
        self._key = bytes(key)

    def encrypt(self, nonce: bytes, data: bytes, associated_data: bytes | None) -> bytes:
        from tlslite.utils.python_aesgcm import new

        if len(nonce) != 12:
            raise ValueError("AES-GCM nonce must contain 12 bytes")
        # Fresh state per operation makes one configured object safe for reuse.
        return bytes(new(bytearray(self._key)).seal(bytearray(nonce), bytearray(data), bytearray(associated_data or b"")))

    def decrypt(self, nonce: bytes, data: bytes, associated_data: bytes | None) -> bytes:
        from tlslite.utils.python_aesgcm import new

        if len(nonce) != 12:
            raise ValueError("AES-GCM nonce must contain 12 bytes")
        plaintext = new(bytearray(self._key)).open(bytearray(nonce), bytearray(data), bytearray(associated_data or b""))
        if plaintext is None:
            raise InvalidTag("AES-GCM authentication failed")
        return bytes(plaintext)


def hkdf_sha256(key_material: bytes, *, salt: bytes | None, info: bytes, length: int) -> bytes:
    """RFC 5869 extract-and-expand; matches the BWS encrypted cache derivation."""
    if not isinstance(length, int) or not 0 <= length <= 255 * hashlib.sha256().digest_size:
        raise ValueError("Invalid HKDF-SHA256 output length")
    pseudorandom_key = hmac.digest(salt or bytes(32), key_material, "sha256")
    previous = b""
    output = bytearray()
    for counter in range(1, (length + 31) // 32 + 1):
        previous = hmac.digest(pseudorandom_key, previous + info + bytes([counter]), "sha256")
        output.extend(previous)
    return bytes(output[:length])


class InvalidToken(Exception):
    """A Fernet token is malformed, expired or unauthenticated."""


class Fernet:
    """Fernet v0x80 tokens compatible with Hermes' existing vault files."""

    def __init__(self, key: bytes | str):
        try:
            decoded = base64.urlsafe_b64decode(key)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Fernet key must be 32 url-safe base64-encoded bytes") from exc
        if len(decoded) != 32:
            raise ValueError("Fernet key must be 32 url-safe base64-encoded bytes")
        self._signing_key = decoded[:16]
        self._encryption_key = decoded[16:]

    @staticmethod
    def generate_key() -> bytes:
        return base64.urlsafe_b64encode(os.urandom(32))

    def encrypt(self, data: bytes) -> bytes:
        return self.encrypt_at_time(data, int(time.time()))

    def encrypt_at_time(self, data: bytes, current_time: int) -> bytes:
        if not isinstance(data, bytes):
            raise TypeError("Fernet plaintext must be bytes")
        iv = os.urandom(16)
        padding = 16 - len(data) % 16
        ciphertext = aes_cbc_encrypt(self._encryption_key, iv, data + bytes([padding]) * padding)
        payload = b"\x80" + current_time.to_bytes(8, "big") + iv + ciphertext
        return base64.urlsafe_b64encode(payload + hmac.digest(self._signing_key, payload, "sha256"))

    def decrypt(self, token: bytes | str, ttl: int | None = None) -> bytes:
        return self._decrypt(token, ttl, int(time.time()))

    def decrypt_at_time(self, token: bytes | str, ttl: int, current_time: int) -> bytes:
        if ttl is None:
            raise ValueError("decrypt_at_time requires a non-None ttl")
        return self._decrypt(token, ttl, current_time)

    def _decrypt(self, token: bytes | str, ttl: int | None, current_time: int) -> bytes:
        if not isinstance(token, (bytes, str)):
            raise TypeError("Fernet token must be bytes or str")
        try:
            decoded = base64.urlsafe_b64decode(token)
        except (binascii.Error, ValueError) as exc:
            raise InvalidToken() from exc
        if len(decoded) < 73 or decoded[0] != 0x80:
            raise InvalidToken()
        timestamp = int.from_bytes(decoded[1:9], "big")
        if ttl is not None and (timestamp + ttl < current_time or timestamp > current_time + 60):
            raise InvalidToken()
        payload, signature = decoded[:-32], decoded[-32:]
        if not hmac.compare_digest(signature, hmac.digest(self._signing_key, payload, "sha256")):
            raise InvalidToken()
        try:
            padded = aes_cbc_decrypt(self._encryption_key, decoded[9:25], decoded[25:-32])
        except ValueError as exc:
            raise InvalidToken() from exc
        padding = padded[-1]
        if not 1 <= padding <= 16 or not hmac.compare_digest(padded[-padding:], bytes([padding]) * padding):
            raise InvalidToken()
        return padded[:-padding]
