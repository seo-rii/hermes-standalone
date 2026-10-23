"""pypdf's explicit AES boundary using reviewed pure Python primitives."""

import hmac
import secrets

from pypdf._crypt_providers._base import CryptBase
from pypdf._crypt_providers._fallback import CryptRC4, rc4_decrypt, rc4_encrypt
from pypdf._utils import logger_warning
from pypdf.errors import PdfStreamError

from standalone.crypto import aes_cbc_decrypt, aes_cbc_encrypt, aes_ecb_decrypt, aes_ecb_encrypt


crypt_provider = ("hermes-pure-python", "1")


class CryptAES(CryptBase):
    def __init__(self, key):
        if len(key) not in (16, 24, 32):
            raise ValueError("Invalid PDF AES key length")
        self.key = bytes(key)

    def encrypt(self, data):
        iv = secrets.token_bytes(16)
        count = 16 - len(data) % 16
        return iv + aes_cbc_encrypt(self.key, iv, data + bytes([count]) * count)

    def decrypt(self, data, *, strict=True):
        iv, ciphertext = data[:16], data[16:]
        if not ciphertext:
            return ciphertext
        if not strict and len(ciphertext) % 16:
            logger_warning("Adding missing padding.", source=__name__)
            count = 16 - len(ciphertext) % 16
            ciphertext += bytes([count]) * count
        try:
            plaintext = aes_cbc_decrypt(self.key, iv, ciphertext)
        except ValueError as exc:
            raise PdfStreamError(str(exc)) from exc
        count = plaintext[-1]
        if 1 <= count <= 16 and hmac.compare_digest(plaintext[-count:], bytes([count]) * count):
            return plaintext[:-count]
        if strict:
            raise PdfStreamError("Invalid PDF AES padding")
        logger_warning("Ignoring invalid PDF AES padding.", source=__name__)
        return plaintext[:-count]
