"""Explicit pure Olm/Megolm boundary consumed by the bundled mautrix machine.

The module is selected by reviewed, qualified source patches. It never creates
``olm`` or ``_libolm`` global packages. The original mautrix machine retains
device trust, replay detection, key sharing, rotation and persistent SQL stores.
"""

import hashlib

from standalone.matrix_crypto import (
    OlmAccountError, OlmGroupSessionError, OlmSessionError, OlmVerifyError, PkSigningError,
    as_bytes, b64decode, b64encode, verify_ed25519,
)
from standalone.matrix_crypto_account import Account, PkSigning
from standalone.matrix_crypto_group import InboundGroupSession, OutboundGroupSession
from standalone.matrix_crypto_session import InboundSession, OlmMessage, OlmPreKeyMessage, OutboundSession, Session


__all__ = ["Account", "Session", "InboundSession", "OutboundSession", "InboundGroupSession", "OutboundGroupSession",
           "OlmMessage", "OlmPreKeyMessage", "PkSigning", "OlmAccountError", "OlmGroupSessionError", "OlmSessionError",
           "OlmVerifyError", "PkSigningError", "ed25519_verify", "sha256"]


def ed25519_verify(key, message, signature):
    try:
        verify_ed25519(b64decode(key), as_bytes(message), b64decode(signature))
    except ValueError as exc:
        raise OlmVerifyError("BAD_MESSAGE_MAC") from exc


def sha256(value):
    return b64encode(hashlib.sha256(as_bytes(value)).digest())

