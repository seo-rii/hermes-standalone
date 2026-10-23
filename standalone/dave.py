"""Qualified pure-Python DAVE v1 MLS sessions for Hermes Discord Opus audio.

This implements the concrete davey 0.1.6 interface consumed by discord.py.
It does not register a global davey module or silently negotiate plaintext.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import time

from .dave_frames import FrameDecryptor, FrameEncryptor, OPUS_SILENCE
from .mls_codec import Reader, opaque
from .mls_crypto import generate_key, public_key
from .mls_group import Group
from .mls_types import ExternalSender, KeyPackage, PublicMessage, Welcome


DAVE_PROTOCOL_VERSION = 1
DEBUG_BUILD = False
__version__ = "0.1.6+standalone"
__author__ = "Hermes standalone contributors; based on Snazzah davey"
__copyright__ = "Copyright 2025-present Snazzah"
__license__ = "MIT"


class MediaType(IntEnum):
    audio = 0
    video = 1


class Codec(IntEnum):
    unknown = 0
    opus = 1
    vp8 = 2
    vp9 = 3
    h264 = 4
    h265 = 5
    av1 = 6


class ProposalsOperationType(IntEnum):
    append = 0
    revoke = 1


class SessionStatus(IntEnum):
    inactive = 0
    pending = 1
    awaiting_response = 2
    active = 3


@dataclass(frozen=True)
class SigningKeyPair:
    private: bytes
    public: bytes


@dataclass(frozen=True)
class CommitWelcome:
    commit: bytes
    welcome: bytes | None


@dataclass
class EncryptionStats:
    successes: int = 0
    failures: int = 0
    duration: int = 0
    attempts: int = 0
    max_attempts: int = 0


@dataclass
class DecryptionStats:
    successes: int = 0
    failures: int = 0
    duration: int = 0
    attempts: int = 0
    passthroughs: int = 0


def generate_p256_keypair():
    return SigningKeyPair(*generate_key())


def generate_displayable_code(data, desired_length, group_size):
    if type(group_size) is not int or not 1 <= group_size <= 8:
        raise ValueError("DAVE display group size must be between one and eight")
    if type(desired_length) is not int or desired_length < 0 or desired_length % group_size or len(data) < desired_length:
        raise ValueError("Invalid DAVE display code length")
    modulus = 10 ** group_size
    return "".join(f"{int.from_bytes(data[index:index + group_size], 'big') % modulus:0{group_size}d}"
                   for index in range(0, desired_length, group_size))


def generate_key_fingerprint(version, key, user_id):
    if version != 0 or not key:
        raise ValueError("Unsupported DAVE fingerprint version or empty key")
    return b"\x00\x00" + bytes(key) + _snowflake(user_id)


def generate_pairwise_fingerprint(version, local_key, local_user_id, remote_key, remote_user_id):
    values = sorted((generate_key_fingerprint(version, local_key, local_user_id),
                     generate_key_fingerprint(version, remote_key, remote_user_id)))
    return hashlib.scrypt(b"".join(values), salt=bytes.fromhex("24cab17a7af8ec2b82b412b92dab192e"),
                          n=16384, r=8, p=2, dklen=64)


def _snowflake(value):
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise ValueError("DAVE user and channel identifiers require unsigned 64-bit values")
    return value.to_bytes(8, "big")


class DaveSession:
    def __init__(self, protocol_version, user_id, channel_id, key_pair=None):
        self.external_sender = None
        self.group = None
        self._initialize(protocol_version, user_id, channel_id, key_pair)

    def _initialize(self, protocol_version, user_id, channel_id, key_pair):
        if protocol_version != 1:
            raise ValueError("Unsupported DAVE protocol version")
        _snowflake(user_id)
        _snowflake(channel_id)
        if key_pair is None:
            key_pair = generate_p256_keypair()
        if public_key(key_pair.private) != key_pair.public:
            raise ValueError("DAVE signing private and public keys do not match")
        self.protocol_version = protocol_version
        self.user_id, self.channel_id = user_id, channel_id
        self.signer = key_pair
        self.ciphersuite = 2
        self.status = SessionStatus.inactive
        self.ready = False
        self.voice_privacy_code = None
        self.encryptor = None
        self.decryptors = {}
        self.packages = {}
        self.encryption_stats = {kind: EncryptionStats() for kind in MediaType}
        self.decryption_stats = {}

    @property
    def epoch(self):
        return None if self.group is None else self.group.context.epoch

    @property
    def own_leaf_index(self):
        return None if self.group is None else self.group.own_index

    def reset(self):
        self.group = None
        self.packages.clear()
        self.status = SessionStatus.inactive

    def reinit(self, protocol_version, user_id, channel_id, key_pair=None):
        self.reset()
        self._initialize(protocol_version, user_id, channel_id, key_pair)
        if self.external_sender is not None:
            self._pending_group()

    def _pending_group(self):
        extension = opaque(self.external_sender.encode())
        self.group = Group.create(_snowflake(self.channel_id), self.signer.private, _snowflake(self.user_id), ((5, extension),))
        self.status = SessionStatus.pending

    def set_external_sender(self, external_sender_data):
        if self.status in (SessionStatus.awaiting_response, SessionStatus.active):
            raise ValueError("DAVE cannot replace the external sender of an established group")
        sender = ExternalSender.parse(external_sender_data)
        # Validate the advertised signature point before creating pending state.
        from .mls_crypto import _public
        _public(sender.signature_key)
        self.external_sender = sender
        self._pending_group()

    def get_serialized_key_package(self):
        if len(self.packages) >= 128:
            raise ValueError("DAVE has reached its retained KeyPackage bound; reinitialize the session")
        package, init, leaf = KeyPackage.create(self.signer.private, _snowflake(self.user_id))
        self.packages[package.reference()] = package, init, leaf
        return package.encode()

    def process_proposals(self, operation_type, proposals, expected_user_ids=None):
        if self.group is None or self.group.removed:
            raise ValueError("DAVE has no MLS group")
        operation = ProposalsOperationType(operation_type)
        reader = Reader(proposals)
        body = reader.vector()
        reader.end()
        if operation == ProposalsOperationType.append:
            while body:
                if body.uint(2) != 1 or body.uint(2) != 1:
                    raise ValueError("DAVE proposals require public MLS v1 messages")
                message = PublicMessage.read(body)
                if message.content_type == 2 and message.content.kind == 1:
                    identity = message.content.value.leaf.credential.identity
                    if len(identity) != 8:
                        raise ValueError("DAVE Add credential must identify a 64-bit user")
                    if expected_user_ids is not None and int.from_bytes(identity, "big") not in expected_user_ids:
                        raise ValueError("DAVE Add proposal identifies an unexpected user")
                self.group.queue(message)
        else:
            while body:
                reference = body.opaque()
                if reference not in self.group.proposals:
                    raise ValueError("DAVE cannot revoke an unknown proposal reference")
                del self.group.proposals[reference]
        if not self.group.proposals:
            self.group.pending = None
            if self.status == SessionStatus.awaiting_response:
                self.status = SessionStatus.active if self.ready else SessionStatus.pending
            return None
        self.group.pending = None
        commit, welcome = self.group.create_commit()
        self.status = SessionStatus.awaiting_response
        return CommitWelcome(commit.message(), None if welcome is None else welcome.encode())

    def process_welcome(self, welcome):
        if self.group is not None and self.status == SessionStatus.active:
            raise ValueError("DAVE session is already in an established group")
        if self.external_sender is None:
            raise ValueError("DAVE Welcome requires the pinned external sender")
        reader = Reader(welcome)
        message = Welcome.read(reader)
        reader.end()
        entry = next((self.packages[reference] for reference, _, _ in message.secrets if reference in self.packages), None)
        if entry is None:
            raise ValueError("DAVE Welcome addresses no retained KeyPackage")
        package, init, leaf = entry
        group = Group.join(message, package, init, leaf, self.signer.private, external_sender=self.external_sender)
        if group.context.group_id != _snowflake(self.channel_id):
            raise ValueError("DAVE Welcome belongs to another voice channel")
        self.group = group
        self.packages.pop(package.reference())
        self.status = SessionStatus.active
        self._update_ratchets()

    def process_commit(self, commit):
        if self.group is None or self.status == SessionStatus.pending:
            raise ValueError("DAVE Commit requires an established or awaiting MLS group")
        self.group.merge_commit(commit)
        self.status = SessionStatus.active
        if self.group.removed:
            self.ready = False
            self.encryptor = None
            self.decryptors.clear()
            raise ValueError("DAVE member was removed from the MLS group; reinitialization is required")
        self._update_ratchets()

    def _update_ratchets(self):
        members = [int(value) for value in self.get_user_ids()]
        for uid in members:
            if uid == self.user_id:
                continue
            secret = self.group.export("Discord Secure Frames v0", uid.to_bytes(8, "little"), 16)
            if uid not in self.decryptors:
                self.decryptors[uid] = FrameDecryptor(secret)
                self.decryption_stats[uid] = {kind: DecryptionStats() for kind in MediaType}
            else:
                self.decryptors[uid].transition(secret)
        self.decryptors = {uid: value for uid, value in self.decryptors.items() if uid in members}
        self.decryption_stats = {uid: value for uid, value in self.decryption_stats.items() if uid in members}
        secret = self.group.export("Discord Secure Frames v0", self.user_id.to_bytes(8, "little"), 16)
        if self.encryptor is None:
            self.encryptor = FrameEncryptor(secret)
        else:
            self.encryptor.transition(secret)
        self.voice_privacy_code = generate_displayable_code(self.group.secrets.epoch_authenticator, 30, 5)
        self.ready = True

    def get_epoch_authenticator(self):
        return None if self.group is None else self.group.secrets.epoch_authenticator

    def get_user_ids(self):
        if self.group is None:
            return []
        values = []
        for _, leaf in self.group.tree.members():
            if len(leaf.credential.identity) != 8:
                raise ValueError("DAVE group has a malformed member identifier")
            values.append(str(int.from_bytes(leaf.credential.identity, "big")))
        return values

    def encrypt(self, media_type, codec, packet):
        if not self.ready:
            raise ValueError("DAVE session is not ready to encrypt")
        if MediaType(media_type) != MediaType.audio or Codec(codec) != Codec.opus:
            raise NotImplementedError("Hermes DAVE implements Opus audio frame encryption")
        if bytes(packet) == OPUS_SILENCE:
            return bytes(packet)
        stats, start = self.encryption_stats[MediaType.audio], time.perf_counter_ns()
        stats.attempts = (stats.attempts + 1) & 0xFFFFFFFF
        stats.max_attempts = max(stats.max_attempts, 1)
        try:
            result = self.encryptor.encrypt(packet)
        except Exception:
            stats.failures = (stats.failures + 1) & 0xFFFFFFFF
            raise
        else:
            stats.successes = (stats.successes + 1) & 0xFFFFFFFF
            return result
        finally:
            stats.duration = (stats.duration + (time.perf_counter_ns() - start) // 1000) & 0xFFFFFFFF

    def encrypt_opus(self, packet):
        return self.encrypt(MediaType.audio, Codec.opus, packet)

    def decrypt(self, user_id, media_type, packet):
        if MediaType(media_type) != MediaType.audio:
            raise NotImplementedError("Hermes DAVE implements Opus audio frame decryption")
        if user_id not in self.decryptors:
            raise ValueError("DAVE has no decryptor for this user")
        if bytes(packet) == OPUS_SILENCE:
            return bytes(packet)
        decryptor, stats = self.decryptors[user_id], self.decryption_stats[user_id][MediaType.audio]
        start = time.perf_counter_ns()
        try:
            result = decryptor.decrypt(packet)
        except Exception:
            stats.failures = (stats.failures + 1) & 0xFFFFFFFF
            raise
        else:
            if decryptor.last_passthrough:
                stats.passthroughs = (stats.passthroughs + 1) & 0xFFFFFFFF
            else:
                stats.successes = (stats.successes + 1) & 0xFFFFFFFF
            return result
        finally:
            stats.attempts = (stats.attempts + decryptor.last_attempts) & 0xFFFFFFFF
            stats.duration = (stats.duration + (time.perf_counter_ns() - start) // 1000) & 0xFFFFFFFF

    def get_encryption_stats(self, media_type=None):
        return self.encryption_stats[MediaType.audio if media_type is None else MediaType(media_type)]

    def get_decryption_stats(self, user_id, media_type=None):
        if user_id not in self.decryption_stats:
            raise ValueError("DAVE has no decryptor for this user")
        return self.decryption_stats[user_id][MediaType.audio if media_type is None else MediaType(media_type)]

    def set_passthrough_mode(self, passthrough_mode, transition_expiry=None):
        expiry = 10 if transition_expiry is None else transition_expiry
        if type(expiry) is not int or not 0 <= expiry <= 0xFFFFFFFF:
            raise ValueError("DAVE transition expiry requires unsigned seconds")
        for decryptor in self.decryptors.values():
            decryptor.set_passthrough(passthrough_mode, expiry)

    def can_passthrough(self, user_id):
        decryptor = self.decryptors.get(user_id)
        return decryptor is not None and decryptor.passthrough_until > decryptor.clock()

    def get_pairwise_fingerprint(self, version, user_id):
        if self.group is None or self.status == SessionStatus.pending:
            raise ValueError("DAVE verification requires an established group")
        member = next((leaf for _, leaf in self.group.tree.members() if leaf.credential.identity == _snowflake(user_id)), None)
        if member is None:
            raise ValueError("DAVE verification user is not in this group")
        return generate_pairwise_fingerprint(version, self.signer.public, self.user_id, member.signature_key, user_id)

    def get_verification_code(self, user_id):
        return generate_displayable_code(self.get_pairwise_fingerprint(0, user_id), 45, 5)

    def __repr__(self):
        return f"DaveSession(protocol_version={self.protocol_version}, user_id={self.user_id}, channel_id={self.channel_id}, ready={self.ready}, status={self.status.name})"
