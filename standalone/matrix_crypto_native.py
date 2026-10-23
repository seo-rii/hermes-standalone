"""Qualified AES-CTR/hash/random/HKDF operations for original mautrix sources.

This module supplies only the concrete primitive calls used by Matrix encrypted
attachments and secret storage. It does not install a Crypto/Cryptodome package.
"""

from __future__ import annotations

import hashlib
import os

from standalone.crypto import hkdf_sha256


class _RandomReader:
    def read(self, length):
        return os.urandom(length)


class Random:
    @staticmethod
    def new():
        return _RandomReader()


class SHA256:
    digest_size = 32
    block_size = 64
    SHA256Hash = type(hashlib.sha256())
    new = staticmethod(hashlib.sha256)


def HKDF(master, key_len, salt, hashmod, num_keys=1, context=b""):
    if hashmod is not SHA256 or key_len < 1 or num_keys < 1:
        raise ValueError("Matrix HKDF requires SHA-256 and positive key lengths")
    derived = hkdf_sha256(bytes(master), salt=bytes(salt) if salt else None,
                          info=bytes(context), length=key_len * num_keys)
    return derived if num_keys == 1 else [derived[i:i + key_len] for i in range(0, len(derived), key_len)]


class Counter:
    @staticmethod
    def new(nbits, prefix=b"", suffix=b"", initial_value=1, little_endian=False, allow_wraparound=False):
        if isinstance(nbits, bool) or not isinstance(nbits, int) or not 8 <= nbits <= 128 or nbits % 8:
            raise ValueError("Invalid AES counter width")
        if not isinstance(initial_value, int) or not 0 <= initial_value < 1 << nbits:
            raise ValueError("Invalid AES initial counter")
        return {"counter_len": nbits // 8, "prefix": bytes(prefix), "suffix": bytes(suffix),
                "initial_value": initial_value, "little_endian": bool(little_endian)}


class _CTR:
    def __init__(self, key, counter):
        from tlslite.utils.rijndael import Rijndael
        if len(key) not in (16, 24, 32):
            raise ValueError("Invalid AES key length")
        self._width, self._prefix, self._suffix = counter["counter_len"], counter["prefix"], counter["suffix"]
        if self._width + len(self._prefix) + len(self._suffix) != 16:
            raise ValueError("AES counter must fill a 16-byte block")
        self._value, self._limit = counter["initial_value"], 1 << (self._width * 8)
        self._endian = "little" if counter["little_endian"] else "big"
        self._cipher, self._remaining = Rijndael(bytearray(key), 16), b""

    def encrypt(self, plaintext, output=None):
        data = bytes(plaintext)
        result = bytearray(len(data))
        position = 0
        while position < len(data):
            if not self._remaining:
                if self._value >= self._limit:
                    raise OverflowError("AES counter wrapped around")
                block = self._prefix + self._value.to_bytes(self._width, self._endian) + self._suffix
                self._remaining = bytes(self._cipher.encrypt(bytearray(block)))
                self._value += 1
            count = min(len(self._remaining), len(data) - position)
            for index in range(count):
                result[position + index] = data[position + index] ^ self._remaining[index]
            self._remaining = self._remaining[count:]
            position += count
        if output is None:
            return bytes(result)
        view = memoryview(output)
        if view.readonly or view.nbytes != len(result):
            raise ValueError("AES output must be writable and match the input length")
        view[:] = result
        return None

    def decrypt(self, ciphertext, output=None):
        return self.encrypt(ciphertext, output)


class AES:
    MODE_CTR = 6
    block_size = 16

    @staticmethod
    def new(key, mode, *, counter):
        if mode != AES.MODE_CTR:
            raise ValueError("This Matrix boundary requires AES-CTR")
        return _CTR(bytes(key), counter)

