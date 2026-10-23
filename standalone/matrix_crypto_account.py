"""Libolm-compatible identity accounts, one-time/fallback keys and signing.

Adapted from libolm 3.2.16, copyright 2015-2016 OpenMarket Ltd, Apache-2.0.
See matrix-crypto-provenance.json and licenses/libolm-LICENSE.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from standalone.matrix_crypto import (
    OlmAccountError, PkSigningError, as_bytes, b64encode, ed25519_key, ed25519_sign,
    pickle_decode, pickle_encode, u32, x25519,
)


@dataclass
class _OneTimeKey:
    identifier: int
    published: bool
    public: bytes
    private: bytes

    def pickle(self):
        return u32(self.identifier) + bytes([self.published]) + self.public + self.private

    @classmethod
    def read(cls, reader):
        return cls(reader.integer(), reader.boolean(), reader.take(32), reader.take(32))


class Account:
    def __init__(self):
        random = os.urandom(64)
        self._ed_public, self._ed_expanded = ed25519_key(random[:32])
        self._curve_private = random[32:]
        self._curve_public = x25519(self._curve_private)
        self._one_time, self._fallback, self._next_id = [], [], 0

    @property
    def identity_keys(self):
        return {"curve25519": b64encode(self._curve_public), "ed25519": b64encode(self._ed_public)}

    @property
    def max_one_time_keys(self):
        return 100

    @property
    def one_time_keys(self):
        return {"curve25519": {b64encode(u32(key.identifier)): b64encode(key.public)
                              for key in self._one_time if not key.published}}

    @property
    def fallback_key(self):
        return {"curve25519": {b64encode(u32(key.identifier)): b64encode(key.public)
                              for key in self._fallback[:1] if not key.published}}

    def sign(self, message):
        return b64encode(ed25519_sign(self._ed_public, self._ed_expanded, as_bytes(message)))

    def _new_key(self, private):
        self._next_id = (self._next_id + 1) & 0xffffffff
        return _OneTimeKey(self._next_id, False, x25519(private), private)

    def generate_one_time_keys(self, count):
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 100000:
            raise ValueError("Invalid one-time key count")
        random = os.urandom(32 * count)
        for index in range(count):
            self._one_time.insert(0, self._new_key(random[index * 32:(index + 1) * 32]))
            del self._one_time[100:]

    def generate_fallback_key(self):
        self._fallback.insert(0, self._new_key(os.urandom(32)))
        del self._fallback[2:]

    def forget_old_fallback_key(self):
        del self._fallback[1:]

    def mark_keys_as_published(self):
        for key in self._one_time + self._fallback[:1]:
            key.published = True

    def _lookup(self, public):
        for key in self._one_time + self._fallback:
            if key.public == public:
                return key
        raise OlmAccountError("BAD_MESSAGE_KEY_ID")

    def remove_one_time_keys(self, session):
        public = session._bob_one_time
        for index, key in enumerate(self._one_time):
            if key.public == public:
                del self._one_time[index]
                return
        self._lookup(public)  # Published fallback keys remain reusable.

    def pickle(self, passphrase=""):
        raw = (u32(4) + self._ed_public + self._ed_expanded + self._curve_public + self._curve_private
               + u32(len(self._one_time)) + b"".join(key.pickle() for key in self._one_time)
               + bytes([len(self._fallback)]) + b"".join(key.pickle() for key in self._fallback) + u32(self._next_id))
        return pickle_encode(raw, passphrase)

    @classmethod
    def from_pickle(cls, pickle, passphrase=""):
        reader = pickle_decode(pickle, passphrase, OlmAccountError)
        version = reader.integer()
        if version == 1:
            raise OlmAccountError("BAD_LEGACY_ACCOUNT_PICKLE")
        if version not in (2, 3, 4):
            raise OlmAccountError("UNKNOWN_PICKLE_VERSION")
        account = cls.__new__(cls)
        account._ed_public, account._ed_expanded = reader.take(32), reader.take(64)
        account._curve_public, account._curve_private = reader.take(32), reader.take(32)
        count = reader.integer()
        if count > 100:
            raise OlmAccountError("CORRUPTED_PICKLE")
        account._one_time = [_OneTimeKey.read(reader) for _ in range(count)]
        if version == 2:
            account._fallback = []
        elif version == 3:
            keys = [_OneTimeKey.read(reader), _OneTimeKey.read(reader)]
            account._fallback = keys if keys[0].published and keys[1].published else keys[:1] if keys[0].published else []
        else:
            count = reader.take(1)[0]
            if count > 2:
                raise OlmAccountError("CORRUPTED_PICKLE")
            account._fallback = [_OneTimeKey.read(reader) for _ in range(count)]
        account._next_id = reader.integer()
        reader.done()
        return account


class PkSigning:
    def __init__(self, seed):
        seed = as_bytes(seed)
        if len(seed) < 32:
            raise PkSigningError("NOT_ENOUGH_RANDOM")
        self._public, self._expanded = ed25519_key(seed[:32])
        self.public_key = b64encode(self._public)

    @classmethod
    def generate_seed(cls):
        return os.urandom(32)

    def sign(self, message):
        return b64encode(ed25519_sign(self._public, self._expanded, as_bytes(message)))

