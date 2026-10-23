"""Libolm-compatible triple-DH setup, Olm double ratchet and encrypted pickles.

Adapted from libolm 3.2.16, copyright 2015-2016 OpenMarket Ltd, Apache-2.0.
See matrix-crypto-provenance.json and licenses/libolm-LICENSE.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os

from standalone.matrix_crypto import (
    OlmAccountError, OlmSessionError, as_bytes, b64decode, b64encode, decrypt_packet,
    derive, encrypt_packet, field_bytes, field_integer, fields, mac,
    pickle_decode, pickle_encode, u32, varint, x25519,
)


class OlmMessage:
    message_type = 1

    def __init__(self, ciphertext):
        self.ciphertext = as_bytes(ciphertext).decode("ascii")

    def __str__(self):
        return self.ciphertext


class OlmPreKeyMessage(OlmMessage):
    message_type = 0


@dataclass
class _Chain:
    public: bytes
    key: bytes
    index: int = 0
    private: bytes | None = None

    def pickle(self):
        return self.public + (self.private or b"") + self.key + u32(self.index)

    @classmethod
    def read(cls, reader, *, sender=False):
        public = reader.take(32)
        private = reader.take(32) if sender else None
        return cls(public, reader.take(32), reader.integer(), private)


def _prekey(message):
    if not isinstance(message, OlmPreKeyMessage):
        raise TypeError("Expected an OlmPreKeyMessage")
    value = fields(b64decode(message.ciphertext, OlmSessionError), 0, OlmSessionError)
    if any(not isinstance(value.get(key), bytes) for key in (1, 2, 3, 4)):
        raise OlmSessionError("BAD_MESSAGE_FORMAT")
    if any(len(value[key]) != 32 for key in (1, 2, 3)):
        raise OlmSessionError("BAD_MESSAGE_FORMAT")
    return value


def _new_chain(root, private, public):
    derived = derive(x25519(private, public), b"OLM_RATCHET", salt=root)
    return derived[:32], derived[32:]


class Session:
    def __init__(self):
        self._alice_identity = self._alice_base = self._bob_one_time = self._root = bytes(32)
        self._received, self._sender, self._receivers, self._skipped = False, None, [], []

    @property
    def id(self):
        return b64encode(hashlib.sha256(self._alice_identity + self._alice_base + self._bob_one_time).digest())

    def matches(self, message, identity_key=None):
        value = _prekey(message)
        return (value[1] == self._bob_one_time and value[2] == self._alice_base
                and value[3] == self._alice_identity
                and (identity_key is None or b64decode(identity_key, OlmSessionError) == self._alice_identity))

    def encrypt(self, plaintext):
        if self._sender is None:
            if not self._receivers:
                raise OlmSessionError("BAD_MESSAGE_KEY_ID")
            private = os.urandom(32)
            root, key = _new_chain(self._root, private, self._receivers[0].public)
            self._root, self._sender = root, _Chain(x25519(private), key, private=private)
        chain = self._sender
        key = mac(chain.key, b"\x01")
        packet = encrypt_packet(as_bytes(plaintext), key, b"OLM_KEYS",
            lambda length: b"\x03" + field_bytes(1, chain.public) + field_integer(2, chain.index) + b"\x22" + varint(length))
        chain.key, chain.index = mac(chain.key, b"\x02"), (chain.index + 1) & 0xffffffff
        if self._received:
            return OlmMessage(b64encode(packet))
        outer = (b"\x03" + field_bytes(1, self._bob_one_time) + field_bytes(2, self._alice_base)
                 + field_bytes(3, self._alice_identity) + field_bytes(4, packet))
        return OlmPreKeyMessage(b64encode(outer))

    def _decrypt(self, packet):
        decoded = fields(packet, 8, OlmSessionError)
        public, index, encrypted = decoded.get(1), decoded.get(2), decoded.get(4)
        if not isinstance(public, bytes) or len(public) != 32 or not isinstance(index, int) or not isinstance(encrypted, bytes):
            raise OlmSessionError("BAD_MESSAGE_FORMAT")
        chain = next((value for value in self._receivers if value.public == public), None)
        if chain is not None and index < chain.index:
            for position, skipped in enumerate(self._skipped):
                if skipped.public == public and skipped.index == index:
                    clear = decrypt_packet(packet, encrypted, skipped.key, b"OLM_KEYS", OlmSessionError)
                    del self._skipped[position]
                    return clear
            raise OlmSessionError("BAD_MESSAGE_MAC")
        root = None
        if chain is None:
            if self._sender is None or index > 2000:
                raise OlmSessionError("BAD_MESSAGE_MAC")
            root, key = _new_chain(self._root, self._sender.private, public)
            chain = _Chain(public, key)
        if index - chain.index > 2000:
            raise OlmSessionError("BAD_MESSAGE_MAC")
        key = chain.key
        for _ in range(index - chain.index):
            key = mac(key, b"\x02")
        clear = decrypt_packet(packet, encrypted, mac(key, b"\x01"), b"OLM_KEYS", OlmSessionError)
        # Authentication precedes every state mutation, including accepting a
        # new remote ratchet key and discarding our previous sender chain.
        if root is not None:
            self._root, self._sender = root, None
            self._receivers.insert(0, chain)
            del self._receivers[5:]
        while chain.index < index:
            self._skipped.insert(0, _Chain(public, mac(chain.key, b"\x01"), chain.index))
            del self._skipped[40:]
            chain.key, chain.index = mac(chain.key, b"\x02"), chain.index + 1
        chain.key, chain.index = mac(chain.key, b"\x02"), (chain.index + 1) & 0xffffffff
        return clear

    def decrypt(self, message, unicode_errors="replace"):
        if not isinstance(message, OlmMessage):
            raise TypeError("Expected an Olm message")
        packet = _prekey(message)[4] if isinstance(message, OlmPreKeyMessage) else b64decode(message.ciphertext, OlmSessionError)
        clear = self._decrypt(packet)
        self._received = True
        return clear.decode("utf-8", unicode_errors)

    def pickle(self, passphrase=""):
        sender = [] if self._sender is None else [self._sender]
        raw = u32(1) + bytes([self._received]) + self._alice_identity + self._alice_base + self._bob_one_time + self._root
        for values in (sender, self._receivers, self._skipped):
            raw += u32(len(values)) + b"".join(chain.pickle() for chain in values)
        return pickle_encode(raw, passphrase)

    @classmethod
    def from_pickle(cls, pickle, passphrase=""):
        reader = pickle_decode(pickle, passphrase, OlmSessionError)
        version = reader.integer()
        if version not in (1, 0x80000001):
            raise OlmSessionError("UNKNOWN_PICKLE_VERSION")
        session = cls.__new__(cls)
        session._received = reader.boolean()
        session._alice_identity, session._alice_base, session._bob_one_time = reader.take(32), reader.take(32), reader.take(32)
        session._root = reader.take(32)
        lists = []
        for capacity, sender in ((1, True), (5, False), (40, False)):
            count = reader.integer()
            if count > capacity:
                raise OlmSessionError("CORRUPTED_PICKLE")
            lists.append([_Chain.read(reader, sender=sender) for _ in range(count)])
        session._sender = lists[0][0] if lists[0] else None
        session._receivers, session._skipped = lists[1], lists[2]
        if version == 0x80000001:
            reader.integer()
        reader.done()
        return session

    def describe(self, buffer_length=600):
        sender = self._sender.index if self._sender is not None else 0
        result = (f"sender chain index: {sender} receiver chain indices:"
                  + "".join(f" {chain.index}" for chain in self._receivers) + " skipped message keys:"
                  + "".join(f" {chain.index}" for chain in self._skipped))
        return result[:max(0, buffer_length - 1)]


class OutboundSession(Session):
    def __init__(self, account, identity_key, one_time_key):
        super().__init__()
        identity, one_time = b64decode(identity_key, OlmSessionError), b64decode(one_time_key, OlmSessionError)
        if len(identity) != 32 or len(one_time) != 32:
            raise OlmSessionError("INVALID_BASE64")
        random = os.urandom(64)
        base, private = random[:32], random[32:]
        self._alice_identity, self._alice_base, self._bob_one_time = account._curve_public, x25519(base), one_time
        secret = x25519(account._curve_private, one_time) + x25519(base, identity) + x25519(base, one_time)
        derived = derive(secret, b"OLM_ROOT")
        self._root, self._sender = derived[:32], _Chain(x25519(private), derived[32:], private=private)


class InboundSession(Session):
    def __init__(self, account, message, identity_key=None):
        super().__init__()
        value = _prekey(message)
        if identity_key is not None and b64decode(identity_key, OlmSessionError) != value[3]:
            raise OlmSessionError("BAD_MESSAGE_KEY_ID")
        decoded = fields(value[4], 8, OlmSessionError)
        if not isinstance(decoded.get(1), bytes) or len(decoded[1]) != 32:
            raise OlmSessionError("BAD_MESSAGE_FORMAT")
        try:
            key = account._lookup(value[1])
        except OlmAccountError as exc:
            raise OlmSessionError(str(exc)) from exc
        self._alice_identity, self._alice_base, self._bob_one_time = value[3], value[2], value[1]
        secret = x25519(key.private, value[3]) + x25519(account._curve_private, value[2]) + x25519(key.private, value[2])
        derived = derive(secret, b"OLM_ROOT")
        self._root, self._receivers = derived[:32], [_Chain(decoded[1], derived[32:])]

