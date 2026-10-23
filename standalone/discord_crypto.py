"""Qualified Discord RTP XChaCha20/Poly1305 and historical XSalsa20 boxes.

ChaCha20 and Poly1305 reuse the reviewed embedded tlslite-ng implementation.
The HChaCha20 and Salsa20/HSalsa20 constructions follow their specifications;
independent libsodium ciphertext vectors cover both boxes. No NaCl package or
native library is loaded. Python arithmetic makes no constant-time guarantee.
"""

import hmac
import os
import struct


MAX_BYTES = 16 * 1024 * 1024
random = os.urandom


class CryptoError(ValueError):
    pass


def _bytes(value):
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("Discord transport crypto requires sample bytes")
    size = value.nbytes if isinstance(value, memoryview) else len(value)
    if size > MAX_BYTES:
        raise ValueError("Discord transport crypto input exceeds its byte limit")
    return bytes(value)


def hchacha20(key, nonce):
    from tlslite.utils.chacha import ChaCha

    key, nonce = _bytes(key), _bytes(nonce)
    if len(key) != 32 or len(nonce) != 16:
        raise ValueError("HChaCha20 requires a 32-byte key and 16-byte nonce")
    state = ChaCha.constants + list(struct.unpack("<8I", key)) + list(struct.unpack("<4I", nonce))
    for _ in range(10):
        ChaCha.double_round(state)
    return struct.pack("<8I", *(state[index] for index in (0, 1, 2, 3, 12, 13, 14, 15)))


def _salsa_rounds(state):
    state = list(state)
    for _ in range(10):
        for a, b, c, d in ((0, 4, 8, 12), (5, 9, 13, 1), (10, 14, 2, 6), (15, 3, 7, 11),
                           (0, 1, 2, 3), (5, 6, 7, 4), (10, 11, 8, 9), (15, 12, 13, 14)):
            for target, left, right, shift in ((b, a, d, 7), (c, b, a, 9), (d, c, b, 13), (a, d, c, 18)):
                value = (state[left] + state[right]) & 0xffffffff
                state[target] ^= ((value << shift) | (value >> (32 - shift))) & 0xffffffff
    return state


def _salsa_state(key, nonce):
    constants = struct.unpack("<4I", b"expand 32-byte k")
    words, state = struct.unpack("<8I", key), [0] * 16
    for index, value in zip((0, 5, 10, 15), constants):
        state[index] = value
    state[1:5], state[11:15] = words[:4], words[4:]
    state[6:10] = struct.unpack("<4I", nonce)
    return state


def hsalsa20(key, nonce):
    key, nonce = _bytes(key), _bytes(nonce)
    if len(key) != 32 or len(nonce) != 16:
        raise ValueError("HSalsa20 requires a 32-byte key and 16-byte nonce")
    state = _salsa_rounds(_salsa_state(key, nonce))
    return struct.pack("<8I", *(state[index] for index in (0, 5, 10, 15, 6, 7, 8, 9)))


def _xsalsa20(key, nonce, length):
    key = hsalsa20(key, nonce[:16])
    state, output = _salsa_state(key, nonce[16:] + bytes(8)), bytearray()
    for counter in range((length + 63) // 64):
        state[8], state[9] = counter & 0xffffffff, counter >> 32
        result = _salsa_rounds(state)
        output.extend(struct.pack("<16I", *((value + initial) & 0xffffffff for value, initial in zip(result, state))))
    return bytes(output[:length])


class _EncryptedMessage(bytes):
    def __new__(cls, nonce, ciphertext):
        value = super().__new__(cls, nonce + ciphertext)
        value.nonce, value.ciphertext = nonce, ciphertext
        return value


class Aead:
    """The concrete 24-byte-nonce AEAD calls used by discord.py RTP mode."""
    KEY_SIZE, NONCE_SIZE, MACBYTES = 32, 24, 16

    def __init__(self, key):
        self._key = _bytes(key)
        if len(self._key) != self.KEY_SIZE:
            raise ValueError("Discord XChaCha20 key must contain 32 bytes")

    def _cipher(self, nonce):
        from tlslite.utils.chacha20_poly1305 import CHACHA20_POLY1305

        if len(nonce) != self.NONCE_SIZE:
            raise ValueError("Discord XChaCha20 nonce must contain 24 bytes")
        return CHACHA20_POLY1305(bytearray(hchacha20(self._key, nonce[:16])), "python")

    def encrypt(self, plaintext, aad=b"", nonce=None):
        nonce = random(self.NONCE_SIZE) if nonce is None else _bytes(nonce)
        ciphertext = self._cipher(nonce).seal(bytearray(bytes(4) + nonce[16:]),
            bytearray(_bytes(plaintext)), bytearray(_bytes(aad)))
        return _EncryptedMessage(nonce, bytes(ciphertext))

    def decrypt(self, ciphertext, aad=b"", nonce=None):
        ciphertext, aad = _bytes(ciphertext), _bytes(aad)
        if nonce is None:
            nonce, ciphertext = ciphertext[:self.NONCE_SIZE], ciphertext[self.NONCE_SIZE:]
        else:
            nonce = _bytes(nonce)
        plaintext = self._cipher(nonce).open(bytearray(bytes(4) + nonce[16:]), bytearray(ciphertext), bytearray(aad))
        if plaintext is None:
            raise CryptoError("Discord transport authentication failed")
        return bytes(plaintext)


class SecretBox:
    """Historical Discord XSalsa20/Poly1305 transport, with NaCl tag order."""
    KEY_SIZE, NONCE_SIZE, MACBYTES = 32, 24, 16

    def __init__(self, key):
        self._key = _bytes(key)
        if len(self._key) != self.KEY_SIZE:
            raise ValueError("Discord XSalsa20 key must contain 32 bytes")

    def _stream(self, nonce, length):
        if len(nonce) != self.NONCE_SIZE:
            raise ValueError("Discord XSalsa20 nonce must contain 24 bytes")
        return _xsalsa20(self._key, nonce, length + 32)

    def encrypt(self, plaintext, nonce=None):
        from tlslite.utils.poly1305 import Poly1305

        plaintext = _bytes(plaintext)
        nonce = random(self.NONCE_SIZE) if nonce is None else _bytes(nonce)
        stream = self._stream(nonce, len(plaintext))
        ciphertext = bytes(value ^ mask for value, mask in zip(plaintext, stream[32:]))
        tag = bytes(Poly1305(stream[:32]).create_tag(ciphertext))
        return _EncryptedMessage(nonce, tag + ciphertext)

    def decrypt(self, ciphertext, nonce=None):
        from tlslite.utils.poly1305 import Poly1305

        ciphertext = _bytes(ciphertext)
        if nonce is None:
            nonce, ciphertext = ciphertext[:self.NONCE_SIZE], ciphertext[self.NONCE_SIZE:]
        else:
            nonce = _bytes(nonce)
        if len(ciphertext) < self.MACBYTES:
            raise CryptoError("Discord transport authentication failed")
        stream = self._stream(nonce, len(ciphertext) - self.MACBYTES)
        tag, encrypted = ciphertext[:self.MACBYTES], ciphertext[self.MACBYTES:]
        expected = bytes(Poly1305(stream[:32]).create_tag(encrypted))
        if not hmac.compare_digest(tag, expected):
            raise CryptoError("Discord transport authentication failed")
        return bytes(value ^ mask for value, mask in zip(encrypted, stream[32:]))
