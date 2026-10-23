"""RFC 9420 epoch schedule, PSK combination, confirmation, and exporter."""

from __future__ import annotations

import hashlib
import hmac

from .mls_codec import uint
from .mls_crypto import derive_secret, expand_label, extract


def psk_secret(identities, available):
    combined = bytes(32)
    seen = set()
    for index, identity in enumerate(identities):
        encoded = identity.encode()
        if encoded in seen:
            raise ValueError("Duplicate MLS pre-shared key identifier")
        seen.add(encoded)
        key = available.get(identity.lookup())
        if key is None:
            raise ValueError("MLS pre-shared key is unavailable")
        extracted = extract(bytes(32), key)
        label = encoded + uint(index, 2) + uint(len(identities), 2)
        derived = expand_label(extracted, "derived psk", label)
        combined = extract(derived, combined)
    return combined


class EpochSecrets:
    LABELS = {
        "init_secret": "init", "sender_data_secret": "sender data", "encryption_secret": "encryption",
        "exporter_secret": "exporter", "external_secret": "external", "confirmation_key": "confirm",
        "membership_key": "membership", "resumption_psk": "resumption", "epoch_authenticator": "authentication",
    }

    def __init__(self, epoch_secret):
        if len(epoch_secret) != 32:
            raise ValueError("MLS suite 0x0002 epoch secret must contain 32 bytes")
        for name, label in self.LABELS.items():
            setattr(self, name, derive_secret(epoch_secret, label))

    @classmethod
    def from_joiner(cls, joiner_secret, context, psk=bytes(32)):
        intermediate = extract(joiner_secret, psk)
        return cls(expand_label(intermediate, "epoch", context.encode()))

    @staticmethod
    def joiner(init_secret, commit_secret, context):
        return expand_label(extract(init_secret, commit_secret), "joiner", context.encode())

    @staticmethod
    def welcome(joiner_secret, psk=bytes(32)):
        return derive_secret(extract(joiner_secret, psk), "welcome")

    def confirmation(self, confirmed):
        return hmac.digest(self.confirmation_key, confirmed, "sha256")

    def export(self, label, context, length):
        secret = derive_secret(self.exporter_secret, label)
        return expand_label(secret, "exported", hashlib.sha256(context).digest(), length)
