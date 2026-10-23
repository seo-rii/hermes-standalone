"""Libolm-compatible Megolm ratchet, signed messages and encrypted pickles.

Adapted from libolm 3.2.16, copyright 2016 OpenMarket Ltd, Apache-2.0.
See matrix-crypto-provenance.json and licenses/libolm-LICENSE.
"""

from __future__ import annotations

import os

from standalone.matrix_crypto import (
    OlmGroupSessionError, as_bytes, b64decode, b64encode, decrypt_packet,
    ed25519_key, ed25519_sign, encrypt_packet, field_integer, fields,
    mac, pickle_decode, pickle_encode, u32, varint, verify_ed25519,
)


class Megolm:
    def __init__(self, data, counter=0):
        if len(data) != 128:
            raise ValueError("A Megolm ratchet contains 128 bytes")
        u32(counter)
        self.parts = [data[i:i + 32] for i in range(0, 128, 32)]
        self.counter = counter

    @property
    def data(self):
        return b"".join(self.parts)

    def copy(self):
        return Megolm(self.data, self.counter)

    def advance_to(self, target):
        u32(target)
        for part in range(4):
            shift = (3 - part) * 8
            steps = ((target >> shift) - (self.counter >> shift)) & 255
            if steps == 0:
                if target < self.counter:
                    steps = 256
                else:
                    continue
            for _ in range(steps - 1):
                self.parts[part] = mac(self.parts[part], bytes([part]))
            for destination in range(3, part - 1, -1):
                self.parts[destination] = mac(self.parts[part], bytes([destination]))
            self.counter = target & (0xffffffff << shift)

    def pickle(self):
        return self.data + u32(self.counter)

    @classmethod
    def read(cls, reader):
        return cls(reader.take(128), reader.integer())


class OutboundGroupSession:
    def __init__(self):
        random = os.urandom(160)
        self._ratchet = Megolm(random[:128])
        self._public, self._expanded = ed25519_key(random[128:])

    @property
    def id(self):
        return b64encode(self._public)

    @property
    def message_index(self):
        return self._ratchet.counter

    @property
    def session_key(self):
        payload = b"\x02" + u32(self.message_index) + self._ratchet.data + self._public
        return b64encode(payload + ed25519_sign(self._public, self._expanded, payload))

    def encrypt(self, plaintext):
        counter = self.message_index
        packet = encrypt_packet(as_bytes(plaintext), self._ratchet.data, b"MEGOLM_KEYS",
                                lambda length: b"\x03" + field_integer(1, counter) + b"\x12" + varint(length))
        signature = ed25519_sign(self._public, self._expanded, packet)
        self._ratchet.advance_to((counter + 1) & 0xffffffff)
        return b64encode(packet + signature)

    def pickle(self, passphrase=""):
        raw = u32(1) + self._ratchet.pickle() + self._public + self._expanded
        return pickle_encode(raw, passphrase)

    @classmethod
    def from_pickle(cls, pickle, passphrase=""):
        reader = pickle_decode(pickle, passphrase, OlmGroupSessionError)
        if reader.integer() != 1:
            raise OlmGroupSessionError("UNKNOWN_PICKLE_VERSION")
        session = cls.__new__(cls)
        session._ratchet = Megolm.read(reader)
        session._public, session._expanded = reader.take(32), reader.take(64)
        reader.done()
        return session


class InboundGroupSession:
    def __init__(self, session_key):
        self._load_key(session_key, exported=False)

    def _load_key(self, value, *, exported):
        raw = b64decode(value, OlmGroupSessionError)
        if len(raw) != (165 if exported else 229) or raw[0] != (1 if exported else 2):
            raise OlmGroupSessionError("BAD_SESSION_KEY")
        public = raw[133:165]
        if not exported:
            try:
                verify_ed25519(public, raw[:165], raw[165:])
            except ValueError as exc:
                raise OlmGroupSessionError("BAD_SIGNATURE") from exc
        self._initial = Megolm(raw[5:133], int.from_bytes(raw[1:5], "big"))
        self._latest = self._initial.copy()
        self._public, self._verified = public, not exported

    @property
    def id(self):
        return b64encode(self._public)

    @property
    def first_known_index(self):
        return self._initial.counter

    def _at(self, index):
        u32(index)
        if (index - self._latest.counter) & 0xffffffff < 1 << 31:
            self._latest.advance_to(index)
            return self._latest.copy()
        if (index - self._initial.counter) & 0xffffffff >= 1 << 31:
            raise OlmGroupSessionError("UNKNOWN_MESSAGE_INDEX")
        result = self._initial.copy()
        result.advance_to(index)
        return result

    def decrypt(self, ciphertext, unicode_errors="replace"):
        packet = b64decode(ciphertext, OlmGroupSessionError)
        decoded = fields(packet, 72, OlmGroupSessionError)
        index, encrypted = decoded.get(1), decoded.get(2)
        if not isinstance(index, int) or not isinstance(encrypted, bytes):
            raise OlmGroupSessionError("BAD_MESSAGE_FORMAT")
        try:
            verify_ed25519(self._public, packet[:-64], packet[-64:])
        except ValueError as exc:
            raise OlmGroupSessionError("BAD_SIGNATURE") from exc
        ratchet = self._at(index)
        clear = decrypt_packet(packet[:-64], encrypted, ratchet.data, b"MEGOLM_KEYS", OlmGroupSessionError)
        self._verified = True
        return clear.decode("utf-8", unicode_errors), index

    def export_session(self, message_index):
        ratchet = self._at(message_index)
        return b64encode(b"\x01" + u32(message_index) + ratchet.data + self._public)

    @classmethod
    def import_session(cls, session_key):
        session = cls.__new__(cls)
        session._load_key(session_key, exported=True)
        return session

    def pickle(self, passphrase=""):
        raw = u32(2) + self._initial.pickle() + self._latest.pickle() + self._public + bytes([self._verified])
        return pickle_encode(raw, passphrase)

    @classmethod
    def from_pickle(cls, pickle, passphrase=""):
        reader = pickle_decode(pickle, passphrase, OlmGroupSessionError)
        version = reader.integer()
        if version not in (1, 2):
            raise OlmGroupSessionError("UNKNOWN_PICKLE_VERSION")
        session = cls.__new__(cls)
        session._initial, session._latest = Megolm.read(reader), Megolm.read(reader)
        session._public = reader.take(32)
        session._verified = True if version == 1 else reader.boolean()
        reader.done()
        return session

