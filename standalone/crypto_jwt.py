"""Explicit PyJWT asymmetric algorithms backed by reviewed Python engines.

PyJWT retains its own JWS/JWT/JWK parsing, registered-claim checks and algorithm
allowlists. The pinned vendor source selects these engines without providing a
cryptography namespace. Python big-integer arithmetic is not constant time.
"""

import base64
import hashlib
import json
import math
import secrets

import ecdsa
from jwt.algorithms import Algorithm
from jwt.exceptions import InvalidKeyError
from tlslite.utils.python_rsakey import Python_RSAKey


def _integer(value):
    if not isinstance(value, str) or not value:
        raise InvalidKeyError("JWK integer is missing or invalid")
    try:
        data = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except ValueError as exc:
        raise InvalidKeyError("JWK integer is not base64url") from exc
    if not data:
        raise InvalidKeyError("JWK integer is empty")
    return int.from_bytes(data, "big")


def _encoded(value, width=None):
    if isinstance(value, int):
        value = value.to_bytes(width or max(1, (value.bit_length() + 7) // 8), "big")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _jwk(value, kind):
    try:
        result = json.loads(value) if isinstance(value, str) else value
    except ValueError as exc:
        raise InvalidKeyError("Invalid JWK JSON") from exc
    if not isinstance(result, dict) or result.get("kty") != kind:
        raise InvalidKeyError("JWK has the wrong key family")
    return result


def _rsa_factors(n, e, d):
    # RFC 8017 key completion: a valid private exponent factors n using
    # nontrivial square roots of one. Work is bounded for malformed input.
    product = d * e - 1
    if product <= 0 or product % 2:
        raise InvalidKeyError("Invalid RSA private exponent")
    odd, power = product, 0
    while odd % 2 == 0:
        odd //= 2
        power += 1
    for _ in range(64):
        value = pow(2 + secrets.randbelow(n - 3), odd, n)
        if value in (1, n - 1):
            continue
        for _ in range(power):
            next_value = value * value % n
            if next_value == 1:
                p = math.gcd(value - 1, n)
                if 1 < p < n and n % p == 0:
                    return p, n // p
            if next_value == n - 1:
                break
            value = next_value
    raise InvalidKeyError("RSA private exponent does not match the public key")


class RSAAlgorithm(Algorithm):
    def __init__(self, hash_name="sha256", *, pss=False):
        self.hash_name, self.hash_alg, self.pss = hash_name, getattr(hashlib, hash_name), pss

    def prepare_key(self, key):
        # google-auth owns these shared signer classes. Initialize its public
        # package before selecting its backend to avoid a partially initialized
        # es module importing this same backend during package initialization.
        from google.auth import crypt
        from standalone.crypto_google import RSASigner, RSAVerifier, _validate_rsa_key
        try:
            if isinstance(key, Python_RSAKey):
                parsed = key
                _validate_rsa_key(parsed, private=parsed.hasPrivateKey())
            elif isinstance(key, (str, bytes)):
                data = key.encode() if isinstance(key, str) else key
                if b"PRIVATE KEY-----" in data:
                    parsed = RSASigner.from_string(data)._key
                else:
                    parsed = RSAVerifier.from_string(data)._pubkey
            else:
                raise InvalidKeyError("Expected an RSA key object or PEM")
        except (ValueError, TypeError, SyntaxError) as exc:
            raise InvalidKeyError("Invalid RSA key") from exc
        return parsed

    def check_key_length(self, key):
        return "RSA key is shorter than the recommended 2048 bits" if key.n.bit_length() < 2048 else None

    def sign(self, msg, key):
        if not key.hasPrivateKey():
            raise InvalidKeyError("Signing requires an RSA private key")
        return bytes(key.hashAndSign(bytearray(msg), rsaScheme="PSS" if self.pss else "PKCS1",
                                     hAlg=self.hash_name, sLen=self.hash_alg().digest_size if self.pss else 0))

    def verify(self, msg, key, sig):
        try:
            return bool(key.hashAndVerify(bytearray(sig), bytearray(msg), rsaScheme="PSS" if self.pss else "PKCS1",
                                          hAlg=self.hash_name, sLen=self.hash_alg().digest_size if self.pss else 0))
        except (ValueError, TypeError, AssertionError):
            return False

    @staticmethod
    def to_jwk(key_obj, as_dict=False):
        key = RSAAlgorithm().prepare_key(key_obj)
        result = {"kty": "RSA", "n": _encoded(key.n), "e": _encoded(key.e)}
        if key.hasPrivateKey():
            result.update({name: _encoded(getattr(key, attribute)) for name, attribute in
                           (("d", "d"), ("p", "p"), ("q", "q"), ("dp", "dP"), ("dq", "dQ"), ("qi", "qInv"))})
        return result if as_dict else json.dumps(result)

    @staticmethod
    def from_jwk(value):
        data = _jwk(value, "RSA")
        try:
            n, e = _integer(data.get("n")), _integer(data.get("e"))
            if n <= 3 or not n % 2 or e < 3 or not e % 2 or e >= n or n.bit_length() > 16384:
                raise InvalidKeyError("Invalid RSA public numbers")
            if "oth" in data:
                raise InvalidKeyError("Multi-prime RSA is not supported")
            if "d" not in data:
                key = Python_RSAKey(n=n, e=e)
            else:
                d = _integer(data["d"])
                if not 0 < d < n:
                    raise InvalidKeyError("Invalid RSA private exponent")
                if "p" in data and "q" in data:
                    p, q = _integer(data["p"]), _integer(data["q"])
                elif "p" not in data and "q" not in data:
                    p, q = _rsa_factors(n, e, d)
                else:
                    raise InvalidKeyError("RSA private prime fields must be supplied together")
                if p <= 2 or q <= 2 or p * q != n:
                    raise InvalidKeyError("RSA private factors do not match")
                key = Python_RSAKey(n=n, e=e, d=d, p=p, q=q,
                                    dP=_integer(data["dp"]) if "dp" in data else d % (p - 1),
                                    dQ=_integer(data["dq"]) if "dq" in data else d % (q - 1),
                                    qInv=_integer(data["qi"]) if "qi" in data else pow(q, -1, p))
            return RSAAlgorithm().prepare_key(key)
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            raise InvalidKeyError("Invalid RSA JWK") from exc


_CURVES = {"P-256": ecdsa.NIST256p, "P-384": ecdsa.NIST384p, "P-521": ecdsa.NIST521p,
           "secp256k1": ecdsa.SECP256k1, "Ed25519": ecdsa.Ed25519, "Ed448": ecdsa.Ed448}


class ECAlgorithm(Algorithm):
    def __init__(self, curve="P-256", hash_name="sha256"):
        self.curve_name, self.curve, self.hash_alg = curve, _CURVES[curve], getattr(hashlib, hash_name)

    def prepare_key(self, key):
        try:
            if isinstance(key, (str, bytes)):
                data = key.encode() if isinstance(key, str) else key
                if b"CERTIFICATE-----" in data:
                    from google.auth import crypt
                    from standalone.crypto_google import _public_key_bytes
                    data = _public_key_bytes(data)
                key = (ecdsa.SigningKey.from_pem(data) if b"PRIVATE KEY-----" in data
                       else ecdsa.VerifyingKey.from_pem(data))
            if not isinstance(key, (ecdsa.SigningKey, ecdsa.VerifyingKey)) or key.curve is not self.curve:
                raise InvalidKeyError("Key curve does not match the JWT algorithm")
            return key
        except (ValueError, TypeError, ecdsa.der.UnexpectedDER) as exc:
            raise InvalidKeyError("Invalid EC key") from exc

    def sign(self, msg, key):
        if not isinstance(key, ecdsa.SigningKey):
            raise InvalidKeyError("Signing requires an EC private key")
        return key.sign_deterministic(msg, hashfunc=self.hash_alg, sigencode=ecdsa.util.sigencode_string)

    def verify(self, msg, key, sig):
        public = key.verifying_key if isinstance(key, ecdsa.SigningKey) else key
        try:
            return public.verify(sig, msg, hashfunc=self.hash_alg, sigdecode=ecdsa.util.sigdecode_string)
        except (ecdsa.BadSignatureError, ValueError, TypeError):
            return False

    @staticmethod
    def to_jwk(key_obj, as_dict=False):
        if isinstance(key_obj, ecdsa.SigningKey):
            public, private = key_obj.verifying_key, key_obj
        elif isinstance(key_obj, ecdsa.VerifyingKey):
            public, private = key_obj, None
        else:
            raise InvalidKeyError("Expected a parsed EC key")
        curve = next((name for name, value in _CURVES.items() if value is public.curve), None)
        if curve is None or curve.startswith("Ed"):
            raise InvalidKeyError("Expected a supported EC curve")
        width, point = public.curve.baselen, public.pubkey.point
        result = {"kty": "EC", "crv": curve, "x": _encoded(point.x(), width), "y": _encoded(point.y(), width)}
        if private is not None:
            result["d"] = _encoded(private.privkey.secret_multiplier, width)
        return result if as_dict else json.dumps(result)

    @staticmethod
    def from_jwk(value):
        data = _jwk(value, "EC")
        curve = _CURVES.get(data.get("crv"))
        if curve is None or data.get("crv", "").startswith("Ed"):
            raise InvalidKeyError("Unsupported EC JWK curve")
        x, y = _integer(data.get("x")), _integer(data.get("y"))
        try:
            encoded = x.to_bytes(curve.baselen, "big") + y.to_bytes(curve.baselen, "big")
            public = ecdsa.VerifyingKey.from_string(encoded, curve=curve, validate_point=True)
            if "d" not in data:
                return public
            private = ecdsa.SigningKey.from_secret_exponent(_integer(data["d"]), curve=curve)
            if private.verifying_key.to_string() != public.to_string():
                raise InvalidKeyError("EC JWK public and private keys do not match")
            return private
        except (ValueError, OverflowError, ecdsa.MalformedPointError) as exc:
            raise InvalidKeyError("Invalid EC JWK point") from exc


class OKPAlgorithm(Algorithm):
    def prepare_key(self, key):
        try:
            if isinstance(key, (str, bytes)):
                data = key.encode() if isinstance(key, str) else key
                key = (ecdsa.SigningKey.from_pem(data) if b"PRIVATE KEY-----" in data else ecdsa.VerifyingKey.from_pem(data))
            if not isinstance(key, (ecdsa.SigningKey, ecdsa.VerifyingKey)) or key.curve not in (ecdsa.Ed25519, ecdsa.Ed448):
                raise InvalidKeyError("EdDSA requires an Ed25519 or Ed448 key")
            return key
        except (ValueError, TypeError, ecdsa.der.UnexpectedDER) as exc:
            raise InvalidKeyError("Invalid EdDSA key") from exc

    def sign(self, msg, key):
        if not isinstance(key, ecdsa.SigningKey):
            raise InvalidKeyError("Signing requires an EdDSA private key")
        return key.sign_deterministic(msg)

    def verify(self, msg, key, sig):
        public = key.verifying_key if isinstance(key, ecdsa.SigningKey) else key
        try:
            return public.verify(sig, msg)
        except (ecdsa.BadSignatureError, ValueError, TypeError):
            return False

    @staticmethod
    def to_jwk(key_obj, as_dict=False):
        key = OKPAlgorithm().prepare_key(key_obj)
        public = key.verifying_key if isinstance(key, ecdsa.SigningKey) else key
        result = {"kty": "OKP", "crv": "Ed25519" if key.curve is ecdsa.Ed25519 else "Ed448", "x": _encoded(public.to_string())}
        if isinstance(key, ecdsa.SigningKey):
            result["d"] = _encoded(key.to_string())
        return result if as_dict else json.dumps(result)

    @staticmethod
    def from_jwk(value):
        data = _jwk(value, "OKP")
        curve = _CURVES.get(data.get("crv"))
        if curve not in (ecdsa.Ed25519, ecdsa.Ed448):
            raise InvalidKeyError("Unsupported EdDSA curve")
        try:
            width = 32 if curve is ecdsa.Ed25519 else 57
            public = ecdsa.VerifyingKey.from_string(_integer(data.get("x")).to_bytes(width, "big"), curve=curve)
            if "d" not in data:
                return public
            private = ecdsa.SigningKey.from_string(_integer(data["d"]).to_bytes(width, "big"), curve=curve)
            if private.verifying_key.to_string() != public.to_string():
                raise InvalidKeyError("EdDSA JWK public and private keys do not match")
            return private
        except (ValueError, OverflowError) as exc:
            raise InvalidKeyError("Invalid EdDSA JWK") from exc


def default_algorithms():
    result = {}
    for bits in (256, 384, 512):
        result["RS" + str(bits)] = RSAAlgorithm("sha" + str(bits))
        result["PS" + str(bits)] = RSAAlgorithm("sha" + str(bits), pss=True)
    result.update({"ES256": ECAlgorithm("P-256", "sha256"), "ES256K": ECAlgorithm("secp256k1", "sha256"),
                   "ES384": ECAlgorithm("P-384", "sha384"), "ES512": ECAlgorithm("P-521", "sha512"),
                   "ES521": ECAlgorithm("P-521", "sha512"), "EdDSA": OKPAlgorithm()})
    return result
