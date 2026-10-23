"""Primitives and exact libolm wire/pickle encoding for the Matrix boundary.

Protocol adaptation of libolm 3.2.16 (Apache-2.0), copyright 2015-2016
OpenMarket Ltd. See matrix-crypto-provenance.json and licenses/libolm-LICENSE.
AES and Edwards arithmetic use the reviewed embedded Python implementations.
Python integer arithmetic does not provide a constant-time execution guarantee.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac

from standalone.crypto import aes_cbc_decrypt, aes_cbc_encrypt, hkdf_sha256


class OlmAccountError(Exception):
    pass


class OlmGroupSessionError(Exception):
    pass


class OlmSessionError(Exception):
    pass


class OlmVerifyError(Exception):
    pass


class PkSigningError(Exception):
    pass


def as_bytes(value):
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError("Expected text or bytes")


def b64encode(value):
    return base64.b64encode(value).rstrip(b"=").decode("ascii")


def b64decode(value, error=ValueError):
    try:
        encoded = as_bytes(value)
        if len(encoded) > 32 * 1024 * 1024:
            raise ValueError("Encoded message is too large")
        return base64.b64decode(encoded + b"=" * (-len(encoded) % 4), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise error("INVALID_BASE64") from exc


def u32(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xffffffff:
        raise ValueError("Expected an unsigned 32-bit integer")
    return value.to_bytes(4, "big")


def mac(key, value):
    return hmac.new(key, value, hashlib.sha256).digest()


def derive(secret, info, length=64, salt=None):
    return hkdf_sha256(secret, salt=salt, info=info, length=length)


def encrypt_packet(plaintext, secret, info, header=lambda length: b""):
    derived = derive(secret, info, 80)
    count = 16 - len(plaintext) % 16
    ciphertext = aes_cbc_encrypt(derived[:32], derived[64:], plaintext + bytes([count]) * count)
    packet = header(len(ciphertext)) + ciphertext
    return packet + mac(derived[32:64], packet)[:8]


def decrypt_packet(packet, ciphertext, secret, info, error):
    derived = derive(secret, info, 80)
    if len(packet) < 8 or not hmac.compare_digest(packet[-8:], mac(derived[32:64], packet[:-8])[:8]):
        raise error("BAD_MESSAGE_MAC")
    if not ciphertext or len(ciphertext) % 16:
        raise error("BAD_MESSAGE_MAC")
    clear = aes_cbc_decrypt(derived[:32], derived[64:], ciphertext)
    count = clear[-1]
    if not 1 <= count <= 16 or clear[-count:] != bytes([count]) * count:
        raise error("BAD_MESSAGE_MAC")
    return clear[:-count]


def pickle_encode(raw, passphrase):
    return b64encode(encrypt_packet(raw, as_bytes(passphrase), b"Pickle")).encode("ascii")


def pickle_decode(value, passphrase, error):
    packet = b64decode(value, error)
    try:
        return Reader(decrypt_packet(packet, packet[:-8], as_bytes(passphrase), b"Pickle", error), error)
    except error as exc:
        raise error("BAD_ACCOUNT_KEY") from exc


class Reader:
    def __init__(self, data, error):
        if len(data) > 2 * 1024 * 1024:
            raise error("CORRUPTED_PICKLE")
        self.data, self.error, self.position = data, error, 0

    def take(self, length):
        end = self.position + length
        if end > len(self.data):
            raise self.error("CORRUPTED_PICKLE")
        value, self.position = self.data[self.position:end], end
        return value

    def integer(self):
        return int.from_bytes(self.take(4), "big")

    def boolean(self):
        return bool(self.take(1)[0])

    def done(self):
        if self.position != len(self.data):
            raise self.error("PICKLE_EXTRA_DATA")


def varint(value):
    u32(value)
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field_bytes(number, value):
    return varint(number * 8 + 2) + varint(len(value)) + value


def field_integer(number, value):
    return varint(number * 8) + varint(value)


def fields(packet, trailer, error):
    if len(packet) < trailer + 1:
        raise error("BAD_MESSAGE_FORMAT")
    if packet[0] != 3:
        raise error("BAD_MESSAGE_VERSION")
    end, position, result = len(packet) - trailer, 1, {}

    def integer():
        nonlocal position
        value = 0
        for shift in range(0, 35, 7):
            if position >= end:
                break
            byte = packet[position]
            position += 1
            value |= (byte & 127) << shift
            if byte < 128 and value <= 0xffffffff:
                return value
        raise error("BAD_MESSAGE_FORMAT")

    while position < end:
        tag = integer()
        if tag & 7 == 0:
            result[tag >> 3] = integer()
        elif tag & 7 == 2:
            length = integer()
            if length > end - position:
                raise error("BAD_MESSAGE_FORMAT")
            result[tag >> 3] = packet[position:position + length]
            position += length
        else:
            raise error("BAD_MESSAGE_FORMAT")
    return result


def x25519(private, public=bytes([9]) + bytes(31)):
    """RFC 7748 Montgomery ladder, including scalar/u-coordinate masking."""
    if len(private) != 32 or len(public) != 32:
        raise ValueError("X25519 keys must contain 32 bytes")
    scalar = int.from_bytes(private, "little") & ((1 << 254) - 8) | 1 << 254
    prime = 2**255 - 19
    x1 = int.from_bytes(public, "little") & ((1 << 255) - 1)
    x2, z2, x3, z3, swap = 1, 0, x1, 1, 0
    for bit in range(254, -1, -1):
        current = scalar >> bit & 1
        swap ^= current
        if swap:
            x2, x3, z2, z3 = x3, x2, z3, z2
        swap = current
        a, b = (x2 + z2) % prime, (x2 - z2) % prime
        aa, bb = a * a % prime, b * b % prime
        e = (aa - bb) % prime
        c, d = (x3 + z3) % prime, (x3 - z3) % prime
        da, cb = d * a % prime, c * b % prime
        x3, z3 = (da + cb)**2 % prime, x1 * (da - cb)**2 % prime
        x2, z2 = aa * bb % prime, e * (aa + 121665 * e) % prime
    if swap:
        x2, z2 = x3, z3
    return (x2 * pow(z2, prime - 2, prime) % prime).to_bytes(32, "little")


def ed25519_key(seed):
    import ecdsa
    if len(seed) != 32:
        raise ValueError("Ed25519 seed must contain 32 bytes")
    expanded = bytearray(hashlib.sha512(seed).digest())
    expanded[0] &= 248
    expanded[31] = expanded[31] & 63 | 64
    public = (ecdsa.Ed25519.generator * int.from_bytes(expanded[:32], "little")).to_bytes()
    return public, bytes(expanded)


def ed25519_sign(public, expanded, message):
    import ecdsa
    if len(public) != 32 or len(expanded) != 64:
        raise ValueError("Invalid Ed25519 expanded key")
    generator = ecdsa.Ed25519.generator
    order = generator.order()
    r = int.from_bytes(hashlib.sha512(expanded[32:] + message).digest(), "little") % order
    encoded = (generator * r).to_bytes()
    challenge = int.from_bytes(hashlib.sha512(encoded + public + message).digest(), "little") % order
    scalar = (r + challenge * int.from_bytes(expanded[:32], "little")) % order
    return encoded + scalar.to_bytes(32, "little")


def verify_ed25519(public, message, signature):
    import ecdsa
    try:
        ecdsa.VerifyingKey.from_string(public, curve=ecdsa.Ed25519).verify(signature, message)
    except (ecdsa.BadSignatureError, ValueError, AssertionError) as exc:
        raise ValueError("BAD_SIGNATURE") from exc

