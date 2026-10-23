"""Bind Hermes crypto call sites to the selected distribution's implementation.

An ordinary source install continues to use cryptography. Only the initialized,
verified standalone source tree selects the embedded pure Python primitives.
"""

from pathlib import Path
import sys


def _bundled() -> bool:
    runtime = sys.modules.get("standalone.runtime")
    return runtime is not None and runtime.is_active_bundle(Path(__file__).resolve().parents[1])


def available() -> bool:
    try:
        if _bundled():
            from tlslite.utils import python_aes
        else:
            from cryptography.hazmat.primitives import ciphers
    except ImportError:
        return False
    return True


def fernet_types():
    if _bundled():
        from standalone.crypto import Fernet, InvalidToken
    else:
        from cryptography.fernet import Fernet, InvalidToken
    return Fernet, InvalidToken


def aes_gcm_type():
    if _bundled():
        from standalone.crypto import AESGCM
    else:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM


def hkdf_sha256(key_material: bytes, *, salt: bytes, info: bytes, length: int) -> bytes:
    if _bundled():
        from standalone.crypto import hkdf_sha256 as derive
        return derive(key_material, salt=salt, info=info, length=length)
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(key_material)


def _aes_operation(key: bytes, iv: bytes | None, data: bytes, *, encrypt: bool) -> bytes:
    if _bundled():
        from standalone import crypto
        if iv is None:
            operation = crypto.aes_ecb_encrypt if encrypt else crypto.aes_ecb_decrypt
            return operation(key, data)
        operation = crypto.aes_cbc_encrypt if encrypt else crypto.aes_cbc_decrypt
        return operation(key, iv, data)
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    cipher = Cipher(algorithms.AES(key), modes.ECB() if iv is None else modes.CBC(iv))
    operation = cipher.encryptor() if encrypt else cipher.decryptor()
    return operation.update(data) + operation.finalize()


def aes_cbc_encrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    return _aes_operation(key, iv, data, encrypt=True)


def aes_cbc_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    return _aes_operation(key, iv, data, encrypt=False)


def aes_ecb_encrypt(key: bytes, data: bytes) -> bytes:
    return _aes_operation(key, None, data, encrypt=True)


def aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    return _aes_operation(key, None, data, encrypt=False)
