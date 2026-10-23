"""RFC 9420 wire objects for DAVE's authenticated public-message workflow."""

from __future__ import annotations

from dataclasses import dataclass
import time

from .mls_codec import Reader, opaque, uint
from .mls_crypto import _public, generate_key, public_key, ref_hash, sign_label, verify_label


def extensions(reader):
    values = []
    seen = set()
    body = reader.vector()
    while body:
        kind, value = body.uint(2), body.opaque()
        if kind in seen:
            raise ValueError("Duplicate MLS extension")
        seen.add(kind)
        values.append((kind, value))
    return tuple(values)


def encode_extensions(values):
    return opaque(b"".join(uint(kind, 2) + opaque(value) for kind, value in values))


def _u16_vector(reader):
    body = reader.vector()
    if len(body.data) % 2:
        raise ValueError("Malformed MLS uint16 vector")
    return tuple(body.uint(2) for _ in range(len(body.data) // 2))


@dataclass(frozen=True)
class Credential:
    identity: bytes
    kind: int = 1

    @classmethod
    def read(cls, reader):
        kind = reader.uint(2)
        if kind != 1:
            raise ValueError("DAVE requires an MLS BasicCredential")
        return cls(reader.opaque(), kind)

    def encode(self):
        return uint(self.kind, 2) + opaque(self.identity)


@dataclass(frozen=True)
class ExternalSender:
    signature_key: bytes
    credential: Credential

    @classmethod
    def read(cls, reader):
        return cls(reader.opaque(), Credential.read(reader))

    @classmethod
    def parse(cls, data):
        reader = Reader(data)
        result = cls.read(reader)
        reader.end()
        return result

    def encode(self):
        return opaque(self.signature_key) + self.credential.encode()


def external_senders(data):
    reader = Reader(data)
    body = reader.vector()
    reader.end()
    values = []
    while body:
        values.append(ExternalSender.read(body))
    return tuple(values)


@dataclass(frozen=True)
class LeafNode:
    encryption_key: bytes
    signature_key: bytes
    credential: Credential
    capabilities: tuple
    source: int
    lifetime: tuple | None
    parent_hash: bytes | None
    extensions: tuple
    signature: bytes

    @classmethod
    def read(cls, reader):
        encryption_key, signature_key = reader.opaque(), reader.opaque()
        credential = Credential.read(reader)
        capabilities = tuple(_u16_vector(reader) for _ in range(5))
        source = reader.uint(1)
        if source not in (1, 2, 3):
            raise ValueError("Unknown MLS leaf source")
        lifetime = (reader.uint(8), reader.uint(8)) if source == 1 else None
        parent_hash = reader.opaque() if source == 3 else None
        ext = extensions(reader)
        return cls(encryption_key, signature_key, credential, capabilities, source,
                   lifetime, parent_hash, ext, reader.opaque())

    def unsigned(self):
        body = opaque(self.encryption_key) + opaque(self.signature_key) + self.credential.encode()
        body += b"".join(opaque(b"".join(uint(value, 2) for value in values)) for values in self.capabilities)
        body += uint(self.source, 1)
        if self.source == 1:
            body += uint(self.lifetime[0], 8) + uint(self.lifetime[1], 8)
        elif self.source == 3:
            body += opaque(self.parent_hash)
        return body + encode_extensions(self.extensions)

    def encode(self):
        return self.unsigned() + opaque(self.signature)

    def verify(self, group_id=None, index=None):
        _public(self.encryption_key)
        if 1 not in self.capabilities[0] or 2 not in self.capabilities[1] or 1 not in self.capabilities[4]:
            raise ValueError("MLS leaf capabilities do not support the DAVE suite")
        body = self.unsigned()
        if self.source != 1:
            if group_id is None or index is None:
                raise ValueError("MLS update and commit leaves require their group position")
            body += opaque(group_id) + uint(index, 4)
        if not verify_label(self.signature_key, "LeafNodeTBS", body, self.signature):
            raise ValueError("Invalid MLS leaf signature")
        return self

    @classmethod
    def create(cls, signer, identity, *, source=1, encryption_key=None, parent_hash=b"", group_id=None, index=None):
        if encryption_key is None:
            encryption_private, encryption_key = generate_key()
        else:
            encryption_private = None
        leaf = cls(encryption_key, public_key(signer), Credential(identity),
                   ((1,), (2,), (), (), (1,)), source, (0, 0xFFFFFFFFFFFFFFFF) if source == 1 else None,
                   parent_hash if source == 3 else None, (), b"")
        unsigned = leaf.unsigned()
        if source != 1:
            unsigned += opaque(group_id) + uint(index, 4)
        leaf = cls(**{**leaf.__dict__, "signature": sign_label(signer, "LeafNodeTBS", unsigned)})
        return leaf, encryption_private


@dataclass(frozen=True)
class KeyPackage:
    version: int
    suite: int
    init_key: bytes
    leaf: LeafNode
    extensions: tuple
    signature: bytes

    @classmethod
    def read(cls, reader):
        return cls(reader.uint(2), reader.uint(2), reader.opaque(), LeafNode.read(reader),
                   extensions(reader), reader.opaque())

    @classmethod
    def parse(cls, data):
        reader = Reader(data)
        result = cls.read(reader)
        reader.end()
        return result

    def unsigned(self):
        return uint(self.version, 2) + uint(self.suite, 2) + opaque(self.init_key) + self.leaf.encode() + encode_extensions(self.extensions)

    def encode(self):
        return self.unsigned() + opaque(self.signature)

    def verify(self, *, now=None):
        if self.version != 1 or self.suite != 2 or self.leaf.source != 1:
            raise ValueError("Unsupported DAVE MLS KeyPackage version, suite, or leaf source")
        self.leaf.verify()
        _public(self.init_key)
        if self.init_key == self.leaf.encryption_key:
            raise ValueError("MLS init and leaf encryption keys must be distinct")
        if now is None:
            now = int(time.time())
        if not self.leaf.lifetime[0] <= now <= self.leaf.lifetime[1]:
            raise ValueError("MLS KeyPackage is outside its lifetime")
        if not verify_label(self.leaf.signature_key, "KeyPackageTBS", self.unsigned(), self.signature):
            raise ValueError("Invalid MLS KeyPackage signature")
        return self

    def reference(self):
        return ref_hash("MLS 1.0 KeyPackage Reference", self.encode())

    @classmethod
    def create(cls, signer, identity):
        init_private, init_key = generate_key()
        leaf, leaf_private = LeafNode.create(signer, identity)
        result = cls(1, 2, init_key, leaf, (), b"")
        result = cls(**{**result.__dict__, "signature": sign_label(signer, "KeyPackageTBS", result.unsigned())})
        return result, init_private, leaf_private


@dataclass(frozen=True)
class GroupContext:
    version: int
    suite: int
    group_id: bytes
    epoch: int
    tree_hash: bytes
    confirmed_transcript_hash: bytes
    extensions: tuple

    @classmethod
    def read(cls, reader):
        return cls(reader.uint(2), reader.uint(2), reader.opaque(), reader.uint(8), reader.opaque(),
                   reader.opaque(), extensions(reader))

    def encode(self):
        return (uint(self.version, 2) + uint(self.suite, 2) + opaque(self.group_id) + uint(self.epoch, 8)
                + opaque(self.tree_hash) + opaque(self.confirmed_transcript_hash) + encode_extensions(self.extensions))


@dataclass(frozen=True)
class GroupInfo:
    context: GroupContext
    extensions: tuple
    confirmation_tag: bytes
    signer: int
    signature: bytes

    @classmethod
    def read(cls, reader):
        return cls(GroupContext.read(reader), extensions(reader), reader.opaque(), reader.uint(4), reader.opaque())

    def unsigned(self):
        return self.context.encode() + encode_extensions(self.extensions) + opaque(self.confirmation_tag) + uint(self.signer, 4)

    def encode(self):
        return self.unsigned() + opaque(self.signature)

    def verify(self, key):
        if not verify_label(key, "GroupInfoTBS", self.unsigned(), self.signature):
            raise ValueError("Invalid MLS GroupInfo signature")


@dataclass(frozen=True)
class ParentNode:
    encryption_key: bytes
    parent_hash: bytes
    unmerged_leaves: tuple

    @classmethod
    def read(cls, reader):
        encryption_key, parent_hash = reader.opaque(), reader.opaque()
        body = reader.vector()
        if len(body.data) % 4:
            raise ValueError("Malformed MLS unmerged leaf vector")
        leaves = tuple(body.uint(4) for _ in range(len(body.data) // 4))
        if leaves != tuple(sorted(set(leaves))):
            raise ValueError("MLS unmerged leaves must be sorted and unique")
        return cls(encryption_key, parent_hash, leaves)

    def encode(self):
        return opaque(self.encryption_key) + opaque(self.parent_hash) + opaque(b"".join(uint(value, 4) for value in self.unmerged_leaves))


@dataclass(frozen=True)
class UpdatePathNode:
    encryption_key: bytes
    encrypted_secrets: tuple

    @classmethod
    def read(cls, reader):
        key = reader.opaque()
        secrets = []
        body = reader.vector()
        while body:
            secrets.append((body.opaque(), body.opaque()))
        return cls(key, tuple(secrets))

    def encode(self):
        return opaque(self.encryption_key) + opaque(b"".join(opaque(enc) + opaque(ciphertext) for enc, ciphertext in self.encrypted_secrets))


@dataclass(frozen=True)
class UpdatePath:
    leaf: LeafNode
    nodes: tuple

    @classmethod
    def read(cls, reader):
        leaf, nodes = LeafNode.read(reader), []
        body = reader.vector()
        while body:
            nodes.append(UpdatePathNode.read(body))
        return cls(leaf, tuple(nodes))

    def encode(self):
        return self.leaf.encode() + opaque(b"".join(node.encode() for node in self.nodes))


@dataclass(frozen=True)
class Proposal:
    kind: int
    value: object

    @classmethod
    def read(cls, reader):
        kind = reader.uint(2)
        if kind == 1:
            value = KeyPackage.read(reader)
        elif kind == 2:
            value = LeafNode.read(reader)
        elif kind == 3:
            value = reader.uint(4)
        elif kind == 4:
            value = PreSharedKeyID.read(reader)
        elif kind == 5:
            value = (reader.opaque(), reader.uint(2), reader.uint(2), extensions(reader))
        elif kind == 6:
            value = reader.opaque()
        elif kind == 7:
            value = extensions(reader)
        else:
            raise ValueError("Unsupported DAVE MLS proposal type")
        return cls(kind, value)

    def encode(self):
        if self.kind in (1, 2):
            value = self.value.encode()
        elif self.kind == 3:
            value = uint(self.value, 4)
        elif self.kind == 4:
            value = self.value.encode()
        elif self.kind == 5:
            group_id, version, suite, ext = self.value
            value = opaque(group_id) + uint(version, 2) + uint(suite, 2) + encode_extensions(ext)
        elif self.kind == 6:
            value = opaque(self.value)
        elif self.kind == 7:
            value = encode_extensions(self.value)
        else:
            raise ValueError("Unsupported DAVE MLS proposal type")
        return uint(self.kind, 2) + value


@dataclass(frozen=True)
class PreSharedKeyID:
    kind: int
    identity: bytes
    nonce: bytes
    usage: int | None = None
    epoch: int | None = None

    @classmethod
    def read(cls, reader):
        kind = reader.uint(1)
        if kind == 1:
            identity, usage, epoch = reader.opaque(), None, None
        elif kind == 2:
            usage, identity, epoch = reader.uint(1), reader.opaque(), reader.uint(8)
            if usage not in (1, 2, 3):
                raise ValueError("Unknown MLS resumption PSK usage")
        else:
            raise ValueError("Unknown MLS PSK type")
        nonce = reader.opaque()
        if len(nonce) != 32:
            raise ValueError("MLS suite 0x0002 PSK nonces must contain 32 bytes")
        return cls(kind, identity, nonce, usage, epoch)

    def encode(self):
        body = uint(self.kind, 1)
        if self.kind == 2:
            body += uint(self.usage, 1)
        body += opaque(self.identity)
        if self.kind == 2:
            body += uint(self.epoch, 8)
        return body + opaque(self.nonce)

    def lookup(self):
        return (self.kind, self.identity, self.usage, self.epoch)


@dataclass(frozen=True)
class Commit:
    proposals: tuple
    path: UpdatePath | None

    @classmethod
    def read(cls, reader):
        proposals = []
        body = reader.vector()
        while body:
            kind = body.uint(1)
            if kind == 1:
                proposals.append(Proposal.read(body))
            elif kind == 2:
                proposals.append(body.opaque())
            else:
                raise ValueError("Unknown MLS proposal reference discriminant")
        return cls(tuple(proposals), reader.optional(UpdatePath.read))

    def encode(self):
        proposals = b"".join(b"\x01" + value.encode() if isinstance(value, Proposal) else b"\x02" + opaque(value) for value in self.proposals)
        path = b"\x00" if self.path is None else b"\x01" + self.path.encode()
        return opaque(proposals) + path


@dataclass(frozen=True)
class PublicMessage:
    group_id: bytes
    epoch: int
    sender_type: int
    sender: int | None
    authenticated_data: bytes
    content_type: int
    content: object
    signature: bytes
    confirmation_tag: bytes | None
    membership_tag: bytes | None

    @classmethod
    def read(cls, reader):
        group_id, epoch, sender_type = reader.opaque(), reader.uint(8), reader.uint(1)
        if sender_type not in (1, 2, 3, 4):
            raise ValueError("Unknown MLS sender type")
        sender = reader.uint(4) if sender_type in (1, 2) else None
        aad, content_type = reader.opaque(), reader.uint(1)
        if content_type == 1:
            content = reader.opaque()
        elif content_type == 2:
            content = Proposal.read(reader)
        elif content_type == 3:
            content = Commit.read(reader)
        else:
            raise ValueError("Unknown MLS content type")
        signature = reader.opaque()
        confirmation = reader.opaque() if content_type == 3 else None
        membership = reader.opaque() if sender_type == 1 else None
        return cls(group_id, epoch, sender_type, sender, aad, content_type, content, signature, confirmation, membership)

    def framed(self):
        body = opaque(self.group_id) + uint(self.epoch, 8) + uint(self.sender_type, 1)
        if self.sender_type in (1, 2):
            body += uint(self.sender, 4)
        body += opaque(self.authenticated_data) + uint(self.content_type, 1)
        return body + (opaque(self.content) if self.content_type == 1 else self.content.encode())

    def tbs(self, context=None):
        body = uint(1, 2) + uint(1, 2) + self.framed()
        if self.sender_type in (1, 4):
            if context is None:
                raise ValueError("MLS member signatures require the group context")
            body += context.encode()
        return body

    def auth(self):
        return opaque(self.signature) + (opaque(self.confirmation_tag) if self.content_type == 3 else b"")

    def authenticated(self):
        return uint(1, 2) + self.framed() + self.auth()

    def encode(self):
        return self.framed() + self.auth() + (opaque(self.membership_tag) if self.sender_type == 1 else b"")

    def message(self):
        return uint(1, 2) + uint(1, 2) + self.encode()

    def reference(self):
        return ref_hash("MLS 1.0 Proposal Reference", self.authenticated())


@dataclass(frozen=True)
class Welcome:
    suite: int
    secrets: tuple
    encrypted_group_info: bytes

    @classmethod
    def read(cls, reader):
        suite, secrets = reader.uint(2), []
        body = reader.vector()
        while body:
            secrets.append((body.opaque(), body.opaque(), body.opaque()))
        return cls(suite, tuple(secrets), reader.opaque())

    def encode(self):
        return uint(self.suite, 2) + opaque(b"".join(opaque(ref) + opaque(enc) + opaque(ciphertext) for ref, enc, ciphertext in self.secrets)) + opaque(self.encrypted_group_info)


def parse_message(data):
    reader = Reader(data)
    if reader.uint(2) != 1:
        raise ValueError("Unsupported MLS protocol version")
    kind = reader.uint(2)
    readers = {1: PublicMessage.read, 3: Welcome.read, 4: GroupInfo.read, 5: KeyPackage.read}
    if kind not in readers:
        raise ValueError("DAVE requires public MLS handshake messages")
    result = readers[kind](reader)
    reader.end()
    return result
