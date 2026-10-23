"""Password-protected PEM/PKCS8 and PKCS12 decoding for the Azure boundary.

ASN.1 schemas are the reviewed pyasn1-modules distribution. CBC/stream ciphers
are the reviewed tlslite-ng Python engines; RC2 decryption follows RFC 2268.
No external program or global cryptography compatibility module is involved.
PKCS12 integrity is verified before decrypting its authenticated contents.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import struct


def decode_der(data, schema):
    from pyasn1.codec.der.decoder import decode
    from pyasn1.error import PyAsn1Error
    try:
        value, trailing = decode(bytes(data), asn1Spec=schema)
    except (PyAsn1Error, ValueError, TypeError) as exc:
        raise ValueError("Invalid ASN.1 credential data") from exc
    if trailing:
        raise ValueError("Trailing ASN.1 credential data")
    return value


def pem_block(data, marker):
    expression = rb"-----BEGIN " + marker.encode("ascii") + rb"-----\s*(.*?)-----END " + marker.encode("ascii") + rb"-----"
    match = re.search(expression, bytes(data), re.DOTALL)
    if match is None:
        raise ValueError("Missing PEM " + marker)
    lines = match.group(1).strip().splitlines()
    headers = {}
    while lines and b":" in lines[0]:
        name, value = lines.pop(0).split(b":", 1)
        headers[name.strip()] = value.strip()
    try:
        content = base64.b64decode(b"".join(line.strip() for line in lines), validate=True)
    except ValueError as exc:
        raise ValueError("Invalid PEM " + marker) from exc
    return content, headers


def encode_pem(data, marker):
    content = base64.b64encode(data)
    return (b"-----BEGIN " + marker.encode("ascii") + b"-----\n" +
            b"\n".join(content[index:index + 64] for index in range(0, len(content), 64)) +
            b"\n-----END " + marker.encode("ascii") + b"-----\n")


def _rounds(value):
    value = int(value)
    if not 1 <= value <= 10_000_000:
        raise ValueError("Credential KDF iterations are outside supported bounds")
    return value


def pkcs12_kdf(password, salt, iterations, purpose, size, hash_name="sha1"):
    """RFC 7292 Appendix B.2, including the Unicode terminating code unit."""
    iterations = _rounds(iterations)
    algorithm = hashlib.new(hash_name)
    width, digest_size = algorithm.block_size, algorithm.digest_size
    password = b"" if password is None else bytes(password).decode("utf-8").encode("utf-16-be") + b"\x00\x00"
    def repeat(value):
        length = width * ((len(value) + width - 1) // width)
        return (value * ((length + len(value) - 1) // len(value)))[:length] if value else b""
    material = bytearray(repeat(bytes(salt)) + repeat(password))
    diversifier = bytes([purpose]) * width
    result = bytearray()
    for _ in range((size + digest_size - 1) // digest_size):
        block = hashlib.new(hash_name, diversifier + material).digest()
        for _ in range(iterations - 1):
            block = hashlib.new(hash_name, block).digest()
        result.extend(block)
        adjustment = int.from_bytes((block * ((width + len(block) - 1) // len(block)))[:width], "big") + 1
        modulus = 1 << (8 * width)
        for offset in range(0, len(material), width):
            value = (int.from_bytes(material[offset:offset + width], "big") + adjustment) % modulus
            material[offset:offset + width] = value.to_bytes(width, "big")
    return bytes(result[:size])


_RC2_PI = bytes.fromhex(
    "d978f9c419ddb5ed28e9fd794aa0d89dc67e37832b76538e624c6488448bfba2"
    "179a59f587b34f1361456d8d09817d32bd8f40eb86b77b0bf09521225c6b4e82"
    "54d66593ce60b21c7356c014a78cf1dc1275ca1f3bbee4d1423dd430a33cb626"
    "6fbf0eda4669075727f21d9bbc944303f811c7f690ef3ee706c3d52fc8661ed7"
    "08e8eade8052eef784aa72ac354d6a2a961ad2715a1549744b9fd05e0418a4ec"
    "c2e0416e0f51cbcc2491af50a1f47039997c3a8523b8b47afc02365b25559731"
    "2d5dfa98e38a92ae05df2910676cbac9d300e6cfe19ea82c6316013f58e289a9"
    "0d38341bab33ffb0bb480c5fb9b1cd2ec5f3db47e5a59c770aa62068fe7fc1ad")


def rc2_cbc_decrypt(key, iv, ciphertext, effective_bits):
    if not 1 <= len(key) <= 128 or not 1 <= effective_bits <= 1024 or len(iv) != 8 or len(ciphertext) % 8:
        raise ValueError("Invalid RC2 key or ciphertext")
    expanded = bytearray(key) + bytearray(128 - len(key))
    for index in range(len(key), 128):
        expanded[index] = _RC2_PI[(expanded[index - 1] + expanded[index - len(key)]) & 255]
    count = (effective_bits + 7) // 8
    expanded[128 - count] = _RC2_PI[expanded[128 - count] & (255 >> (8 * count - effective_bits))]
    for index in range(127 - count, -1, -1):
        expanded[index] = _RC2_PI[expanded[index + 1] ^ expanded[index + count]]
    schedule = struct.unpack("<64H", expanded)
    output, previous = bytearray(), bytes(iv)
    shifts = (1, 2, 3, 5)
    for offset in range(0, len(ciphertext), 8):
        block = bytes(ciphertext[offset:offset + 8])
        words = list(struct.unpack("<4H", block))
        position = 63
        for round_index in range(16):
            if round_index in (5, 11):
                for index in range(3, -1, -1):
                    words[index] = (words[index] - schedule[words[(index - 1) % 4] & 63]) & 65535
            for index in range(3, -1, -1):
                value = words[index]
                value = ((value >> shifts[index]) | (value << (16 - shifts[index]))) & 65535
                value -= schedule[position] + (words[(index - 1) % 4] & words[(index - 2) % 4])
                value -= (~words[(index - 1) % 4]) & words[(index - 3) % 4]
                words[index] = value & 65535
                position -= 1
        output.extend(a ^ b for a, b in zip(struct.pack("<4H", *words), previous))
        previous = block
    return bytes(output)


def _decrypt(cipher, key, iv, ciphertext, effective_bits=None):
    from tlslite.utils import python_aes, python_tripledes, python_rc4
    if cipher == "rc4":
        return bytes(python_rc4.new(bytearray(key)).decrypt(bytearray(ciphertext)))
    width = 16 if cipher == "aes" else 8
    if len(iv) != width or not ciphertext or len(ciphertext) % width:
        raise ValueError("Invalid encrypted credential block length")
    if cipher == "aes":
        plaintext = bytes(python_aes.new(bytearray(key), 2, bytearray(iv)).decrypt(bytearray(ciphertext)))
    elif cipher == "3des":
        plaintext = bytes(python_tripledes.new(bytes(key), bytes(iv)).decrypt(bytes(ciphertext)))
    elif cipher == "rc2":
        plaintext = rc2_cbc_decrypt(key, iv, ciphertext, effective_bits)
    else:
        raise ValueError("Unsupported credential cipher")
    padding = plaintext[-1]
    if not 1 <= padding <= width or not hmac.compare_digest(plaintext[-padding:], bytes([padding]) * padding):
        raise ValueError("Incorrect password or malformed encrypted credential")
    return plaintext[:-padding]


_HMAC_HASH = {"1.2.840.113549.2." + str(number): name for number, name in
              ((7, "sha1"), (8, "sha224"), (9, "sha256"), (10, "sha384"), (11, "sha512"))}
_DIGEST_HASH = {"1.3.14.3.2.26": "sha1", **{"2.16.840.1.101.3.4.2." + str(number): name for number, name in
                ((1, "sha256"), (2, "sha384"), (3, "sha512"), (4, "sha224"))}}


def decrypt_pbe(algorithm, ciphertext, password):
    from pyasn1.type import univ
    from pyasn1_modules import rfc8018, rfc7292
    oid = str(algorithm["algorithm"])
    if oid == "1.2.840.113549.1.5.13":
        params = decode_der(algorithm["parameters"], rfc8018.PBES2_params())
        kdf = params["keyDerivationFunc"]
        if str(kdf["algorithm"]) != "1.2.840.113549.1.5.12":
            raise ValueError("Unsupported credential key derivation function")
        pbkdf = decode_der(kdf["parameters"], rfc8018.PBKDF2_params())
        if pbkdf["salt"].getName() != "specified":
            raise ValueError("Unsupported credential KDF salt source")
        scheme = params["encryptionScheme"]
        scheme_oid = str(scheme["algorithm"])
        choices = {"2.16.840.1.101.3.4.1.2": ("aes", 16), "2.16.840.1.101.3.4.1.22": ("aes", 24),
                   "2.16.840.1.101.3.4.1.42": ("aes", 32), "1.2.840.113549.3.7": ("3des", 24)}
        if scheme_oid not in choices:
            raise ValueError("Unsupported PBES2 credential cipher")
        cipher, size = choices[scheme_oid]
        if pbkdf["keyLength"].hasValue() and int(pbkdf["keyLength"]) != size:
            raise ValueError("Credential KDF key length does not match its cipher")
        hash_name = _HMAC_HASH.get(str(pbkdf["prf"]["algorithm"]))
        if hash_name is None:
            raise ValueError("Unsupported credential KDF hash")
        iv = decode_der(scheme["parameters"], univ.OctetString()).asOctets()
        key = hashlib.pbkdf2_hmac(hash_name, password or b"", pbkdf["salt"]["specified"].asOctets(),
                                 _rounds(pbkdf["iterationCount"]), dklen=size)
        return _decrypt(cipher, key, iv, ciphertext)
    prefix = "1.2.840.113549.1.12.1."
    if oid.startswith(prefix) and oid[len(prefix):] in ("1", "2", "3", "4", "5", "6"):
        number = int(oid[len(prefix):])
        cipher, size = {1: ("rc4", 16), 2: ("rc4", 5), 3: ("3des", 24), 4: ("3des", 16),
                        5: ("rc2", 16), 6: ("rc2", 5)}[number]
        params = decode_der(algorithm["parameters"], rfc7292.Pkcs_12PbeParams())
        salt, rounds = params["salt"].asOctets(), _rounds(params["iterations"])
        key = pkcs12_kdf(password, salt, rounds, 1, size)
        iv = b"" if cipher == "rc4" else pkcs12_kdf(password, salt, rounds, 2, 8)
        return _decrypt(cipher, key, iv, ciphertext, 8 * size)
    raise ValueError("Unsupported encrypted credential algorithm " + oid)


def decrypt_pkcs8(data, password):
    from pyasn1_modules.rfc5208 import EncryptedPrivateKeyInfo
    value = decode_der(data, EncryptedPrivateKeyInfo())
    return decrypt_pbe(value["encryptionAlgorithm"], value["encryptedData"].asOctets(), password)


def private_key_der(data, password=None):
    """Return unencrypted PKCS8 or PKCS1 DER and its original PEM marker."""
    if password is not None and not isinstance(password, bytes):
        raise TypeError("Credential password must be bytes")
    if b"-----BEGIN ENCRYPTED PRIVATE KEY-----" in data:
        if password is None:
            raise TypeError("Password was not given but private key is encrypted")
        encrypted, _ = pem_block(data, "ENCRYPTED PRIVATE KEY")
        return decrypt_pkcs8(encrypted, password), "PRIVATE KEY"
    marker = "RSA PRIVATE KEY" if b"-----BEGIN RSA PRIVATE KEY-----" in data else "PRIVATE KEY"
    content, headers = pem_block(data, marker)
    if headers:
        if headers.get(b"Proc-Type") != b"4,ENCRYPTED" or b"DEK-Info" not in headers:
            raise ValueError("Unsupported PEM credential headers")
        if password is None:
            raise TypeError("Password was not given but private key is encrypted")
        try:
            cipher_name, iv_hex = headers[b"DEK-Info"].split(b",", 1)
            iv = bytes.fromhex(iv_hex.decode("ascii"))
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Invalid encrypted PEM cipher header") from exc
        choices = {b"AES-128-CBC": ("aes", 16), b"AES-192-CBC": ("aes", 24), b"AES-256-CBC": ("aes", 32),
                   b"DES-EDE3-CBC": ("3des", 24)}
        if cipher_name not in choices:
            raise ValueError("Unsupported encrypted PEM cipher")
        cipher, size = choices[cipher_name]
        key, block = b"", b""
        while len(key) < size:
            block = hashlib.md5(block + password + iv[:8], usedforsecurity=False).digest()
            key += block
        return _decrypt(cipher, key[:size], iv, content), marker
    if password is not None:
        raise TypeError("Password was given but private key is not encrypted")
    return content, marker


def pkcs12_contents(data, password):
    """Return private-key DERs and certificate DERs, after MAC verification."""
    from pyasn1.type import univ
    from pyasn1_modules import rfc7292, rfc5652
    pfx = decode_der(data, rfc7292.PFX())
    if int(pfx["version"]) != 3 or str(pfx["authSafe"]["contentType"]) != "1.2.840.113549.1.7.1":
        raise ValueError("Unsupported PKCS12 integrity mode")
    authenticated = decode_der(pfx["authSafe"]["content"], univ.OctetString()).asOctets()
    if pfx["macData"].hasValue():
        mac = pfx["macData"]
        hash_name = _DIGEST_HASH.get(str(mac["mac"]["digestAlgorithm"]["algorithm"]))
        if hash_name is None:
            raise ValueError("Unsupported PKCS12 integrity hash")
        candidates = (password, b"") if password is None else (password,)
        for candidate in candidates:
            key = pkcs12_kdf(candidate, mac["macSalt"].asOctets(), _rounds(mac["iterations"]),
                             3, hashlib.new(hash_name).digest_size, hash_name)
            if hmac.compare_digest(hmac.digest(key, authenticated, hash_name), mac["mac"]["digest"].asOctets()):
                password = candidate
                break
        else:
            raise ValueError("Incorrect PKCS12 password or invalid integrity MAC")
    keys, certificates = [], []
    def bags(content, depth=0):
        if depth > 16:
            raise ValueError("PKCS12 safe bag nesting is too deep")
        for bag in decode_der(content, rfc7292.SafeContents()):
            kind, value = str(bag["bagId"]), bag["bagValue"].asOctets()
            if kind == "1.2.840.113549.1.12.10.1.1":
                keys.append(value)
            elif kind == "1.2.840.113549.1.12.10.1.2":
                keys.append(decrypt_pkcs8(value, password))
            elif kind == "1.2.840.113549.1.12.10.1.3":
                cert = decode_der(value, rfc7292.CertBag())
                if str(cert["certId"]) != "1.2.840.113549.1.9.22.1":
                    raise ValueError("Unsupported PKCS12 certificate bag")
                certificates.append(decode_der(cert["certValue"], univ.OctetString()).asOctets())
            elif kind == "1.2.840.113549.1.12.10.1.6":
                bags(value, depth + 1)
            elif kind not in ("1.2.840.113549.1.12.10.1.4", "1.2.840.113549.1.12.10.1.5"):
                raise ValueError("Unsupported PKCS12 safe bag")
    for content in decode_der(authenticated, rfc7292.AuthenticatedSafe()):
        kind = str(content["contentType"])
        if kind == "1.2.840.113549.1.7.1":
            bags(decode_der(content["content"], univ.OctetString()).asOctets())
        elif kind == "1.2.840.113549.1.7.6":
            encrypted = decode_der(content["content"], rfc5652.EncryptedData())["encryptedContentInfo"]
            if str(encrypted["contentType"]) != "1.2.840.113549.1.7.1":
                raise ValueError("Unsupported PKCS12 encrypted content type")
            bags(decrypt_pbe(encrypted["contentEncryptionAlgorithm"], encrypted["encryptedContent"].asOctets(), password))
        else:
            raise ValueError("Unsupported PKCS12 privacy mode")
    return keys, certificates
