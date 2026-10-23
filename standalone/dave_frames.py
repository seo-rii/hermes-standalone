"""DAVE v1 Opus frame encryption and ratchet/replay transitions.

Ported from the pinned MIT davey 0.1.4 cryptor. MLS session authentication is
handled separately: the caller supplies an authenticated epoch export secret.
"""

from __future__ import annotations

from collections import deque
import time

from standalone.crypto import InvalidTag
from .mls_crypto import expand_label, gcm8_open, gcm8_seal


OPUS_SILENCE = b"\xf8\xff\xfe"
MARKER = b"\xfa\xfa"
MAX_FRAME = 1024 * 1024


class FrameError(ValueError):
    pass


def leb128(value):
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise FrameError("DAVE integer outside its wire range")
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def read_leb128(data, offset=0):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise FrameError("Truncated DAVE variable integer")
        byte = data[offset]
        offset += 1
        if shift == 63 and byte > 1:
            raise FrameError("DAVE variable integer overflow")
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise FrameError("DAVE variable integer overflow")


class HashRatchet:
    def __init__(self, secret):
        if len(secret) != 16:
            raise ValueError("DAVE epoch exporter secret must contain 16 bytes")
        self.secret = bytes(secret)
        self.next_generation = 0
        self.cache = {}

    def get(self, generation):
        if type(generation) is not int or not 0 <= generation < 1 << 32:
            raise FrameError("DAVE generation outside its wire range")
        if generation in self.cache:
            return self.cache[generation]
        if generation < self.next_generation:
            raise FrameError("DAVE ratchet generation has expired")
        if generation - self.next_generation > 1024:
            raise FrameError("DAVE ratchet derivation exceeds its work bound")
        while self.next_generation <= generation:
            current = self.next_generation
            context = current.to_bytes(4, "big")
            key = expand_label(self.secret, "key", context, 16)
            nonce = expand_label(self.secret, "nonce", context, 12)
            self.secret = expand_label(self.secret, "secret", context, 32)
            self.cache[current] = key, nonce
            self.next_generation += 1
        return self.cache[generation]

    def erase(self, generation):
        self.cache.pop(generation, None)


def wrapped_generation(oldest, generation):
    return (oldest // 256 + (generation < oldest % 256)) * 256 + generation


def _nonce(value):
    return bytes(8) + value.to_bytes(4, "little")


def parse_frame(frame):
    frame = bytes(frame)
    if len(frame) > MAX_FRAME:
        raise FrameError("DAVE frame exceeds the frame bound")
    if len(frame) < 11 or frame[-2:] != MARKER:
        raise FrameError("Missing DAVE supplemental section")
    size = frame[-3]
    if size < 11 or size > len(frame):
        raise FrameError("Invalid DAVE supplemental length")
    payload, supplemental = frame[:-size], frame[-size:]
    tag = supplemental[:8]
    values = supplemental[8:-3]
    nonce, position = read_leb128(values)
    if nonce > 0xFFFFFFFF:
        raise FrameError("DAVE nonce outside its 32-bit wire range")
    ranges = []
    previous = 0
    while position < len(values):
        offset, position = read_leb128(values, position)
        length, position = read_leb128(values, position)
        if offset < previous or offset + length > len(payload):
            raise FrameError("DAVE authenticated ranges overlap or exceed the frame")
        ranges.append((offset, length))
        previous = offset + length
    aad = bytearray()
    ciphertext = bytearray()
    position = 0
    for offset, length in ranges:
        ciphertext.extend(payload[position:offset])
        aad.extend(payload[offset:offset + length])
        position = offset + length
    ciphertext.extend(payload[position:])
    return nonce, bytes(ciphertext), tag, bytes(aad), tuple(ranges)


def _reconstruct(plaintext, aad, ranges):
    out = bytearray()
    plain_offset = aad_offset = 0
    for offset, length in ranges:
        size = offset - len(out)
        out.extend(plaintext[plain_offset:plain_offset + size])
        plain_offset += size
        out.extend(aad[aad_offset:aad_offset + length])
        aad_offset += length
    out.extend(plaintext[plain_offset:])
    return bytes(out)


class FrameEncryptor:
    def __init__(self, secret):
        self.transition(secret)

    def transition(self, secret):
        self.ratchet = HashRatchet(secret)
        self.generation = 0
        self.truncated_nonce = 0

    def encrypt(self, frame):
        frame = bytes(frame)
        if len(frame) > MAX_FRAME - 255:
            raise FrameError("DAVE Opus frame exceeds the frame bound")
        if frame == OPUS_SILENCE:
            return frame
        self.truncated_nonce = (self.truncated_nonce + 1) & 0xFFFFFFFF
        self.generation = wrapped_generation(self.generation, self.truncated_nonce >> 24)
        key, _ = self.ratchet.get(self.generation)
        ciphertext, tag = gcm8_seal(key, _nonce(self.truncated_nonce), frame)
        encoded_nonce = leb128(self.truncated_nonce)
        return ciphertext + tag + encoded_nonce + bytes([11 + len(encoded_nonce)]) + MARKER


class _CipherManager:
    def __init__(self, secret, clock):
        self.clock = clock
        self.created = clock()
        self.ratchet = HashRatchet(secret)
        self.expiry = None
        self.keys = {}
        self.oldest_generation = self.newest_generation = 0
        self.newest_nonce = None
        self.missing = set()

    def expired(self):
        return self.expiry is not None and self.clock() > self.expiry

    def _cleanup(self):
        now = self.clock()
        self.keys = {generation: value for generation, value in self.keys.items()
                     if value[1] is None or value[1] >= now}
        while self.oldest_generation < self.newest_generation and self.oldest_generation not in self.keys:
            self.ratchet.erase(self.oldest_generation)
            self.oldest_generation += 1

    def _key(self, generation):
        self._cleanup()
        if generation < self.oldest_generation or generation > self.newest_generation + 250:
            raise FrameError("DAVE generation is outside the accepted window")
        maximum = (170 * int(self.clock() - self.created)) >> 24
        if generation > maximum:
            raise FrameError("DAVE generation exceeds the ratchet lifetime")
        if generation not in self.keys:
            key, _ = self.ratchet.get(generation)
            self.keys[generation] = key, self.clock() + 10 if generation < self.newest_generation else None
        return self.keys[generation][0]

    def decrypt(self, nonce, ciphertext, tag, aad):
        generation = wrapped_generation(self.oldest_generation, nonce >> 24)
        large_nonce = (generation << 24) | (nonce & 0xFFFFFF)
        if self.newest_nonce is not None and large_nonce <= self.newest_nonce and large_nonce not in self.missing:
            raise FrameError("DAVE replayed or expired nonce")
        plaintext = gcm8_open(self._key(generation), _nonce(nonce), ciphertext, tag, aad)
        # A failed authenticator cannot advance replay state or expire old keys.
        if self.newest_nonce is None:
            self.newest_nonce = large_nonce
        elif large_nonce > self.newest_nonce:
            oldest = max(0, large_nonce - 1000)
            self.missing = {value for value in self.missing if value >= oldest}
            self.missing.update(range(max(oldest, self.newest_nonce + 1), large_nonce))
            self.newest_nonce = large_nonce
        else:
            self.missing.discard(large_nonce)
        if generation > self.newest_generation:
            self.newest_generation = generation
            self.keys = {index: (key, self.clock() + 10 if index < generation else expiry)
                         for index, (key, expiry) in self.keys.items()}
        return plaintext


class FrameDecryptor:
    def __init__(self, secret=None, *, clock=time.monotonic):
        self.clock = clock
        self.managers = deque()
        self.passthrough_until = 0.0
        self.last_attempts = 0
        self.last_passthrough = False
        if secret is not None:
            self.transition(secret)

    def transition(self, secret):
        for manager in self.managers:
            manager.expiry = self.clock() + 10
        self.managers.append(_CipherManager(secret, self.clock))

    def set_passthrough(self, enabled, transition_expiry=10):
        if enabled:
            self.passthrough_until = float("inf")
        else:
            self.passthrough_until = min(self.passthrough_until, self.clock() + transition_expiry)

    def decrypt(self, frame):
        frame = bytes(frame)
        self.last_attempts = 0
        self.last_passthrough = False
        if frame == OPUS_SILENCE:
            return frame
        while self.managers and self.managers[0].expired():
            self.managers.popleft()
        try:
            nonce, ciphertext, tag, aad, ranges = parse_frame(frame)
        except FrameError:
            if self.passthrough_until > self.clock() and len(frame) <= MAX_FRAME:
                self.last_passthrough = True
                return frame
            raise
        for manager in self.managers:
            self.last_attempts += 1
            try:
                plaintext = manager.decrypt(nonce, ciphertext, tag, aad)
            except (FrameError, InvalidTag):
                continue
            return _reconstruct(plaintext, aad, ranges)
        raise FrameError("No current DAVE ratchet authenticated this frame")
