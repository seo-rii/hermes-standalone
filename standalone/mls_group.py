"""Authenticated MLS v1 public handshakes for DAVE's P-256 cipher suite."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import hmac
import os

from standalone.crypto import AESGCM
from .mls_codec import Reader, opaque, uint
from .mls_crypto import derive_key_pair, derive_secret, expand_label, generate_key, open_label, public_key, seal_label, sign_label, verify_label
from .mls_schedule import EpochSecrets, psk_secret
from .mls_tree import Tree, copath, descendants, direct_path, root
from .mls_types import Commit, Credential, ExternalSender, GroupContext, GroupInfo, KeyPackage, LeafNode, ParentNode, PreSharedKeyID, Proposal, PublicMessage, UpdatePath, UpdatePathNode, Welcome, external_senders, parse_message


def _interim(confirmed, confirmation):
    return hashlib.sha256(confirmed + opaque(confirmation)).digest()


def _path_key(secret):
    return derive_key_pair(derive_secret(secret, "node"))


def merge_path(tree, group_id, sender, path):
    previous = tree.leaf(sender)
    if path.leaf.source != 3 or path.leaf.credential != previous.credential:
        raise ValueError("MLS commit leaf changes the sender identity or has an invalid source")
    path.leaf.verify(group_id, sender)
    tree.blank_path(sender)
    pairs = tree.filtered_path(sender)
    if len(pairs) != len(path.nodes):
        raise ValueError("MLS UpdatePath has the wrong number of nodes")
    for (index, _), node in zip(pairs, path.nodes):
        tree.nodes[index] = ParentNode(node.encryption_key, b"", ())
    parent_hash = b""
    for index, other in reversed(pairs):
        tree.nodes[index] = replace(tree.nodes[index], parent_hash=parent_hash)
        parent_hash = tree.parent_hash(index, other)
    if path.leaf.parent_hash != parent_hash:
        raise ValueError("MLS UpdatePath leaf parent hash does not authenticate the path")
    tree.nodes[2 * sender] = path.leaf
    return pairs


def decrypt_path(tree, sender, path, private_keys, context, excluded=()):
    pairs = tree.filtered_path(sender)
    current = None
    added = {}
    for (index, other), node in zip(pairs, path.nodes):
        resolution = tree.resolution(other, excluded)
        if len(resolution) != len(node.encrypted_secrets):
            raise ValueError("MLS UpdatePath has the wrong number of encrypted secrets")
        if current is None:
            for target, (enc, ciphertext) in zip(resolution, node.encrypted_secrets):
                key = private_keys.get(tree.nodes[target].encryption_key)
                if key is not None:
                    current = open_label(key, "UpdatePathNode", context.encode(), enc, ciphertext)
                    if len(current) != 32:
                        raise ValueError("MLS path secret has the wrong length")
                    break
        if current is not None:
            private, public = _path_key(current)
            if public != node.encryption_key:
                raise ValueError("MLS path public key does not match the decrypted secret")
            added[public] = private
            current = derive_secret(current, "path")
    if current is None:
        raise ValueError("No MLS UpdatePath secret was encrypted for this member")
    return current, added


class Group:
    def __init__(self, context, tree, secrets, own_index, signer, private_keys, *, interim=None, psks=None, verification_time=None):
        if context.version != 1 or context.suite != 2:
            raise ValueError("Unsupported DAVE MLS version or cipher suite")
        self.context = context
        self.tree = tree
        self.secrets = secrets
        self.own_index = own_index
        self.signer = signer
        self.private_keys = dict(private_keys)
        self.interim = _interim(context.confirmed_transcript_hash, secrets.confirmation(context.confirmed_transcript_hash)) if interim is None else interim
        self.psks = {} if psks is None else dict(psks)
        self.psks[(2, context.group_id, 1, context.epoch)] = secrets.resumption_psk
        self.proposals = {}
        self.pending = None
        self.removed = False
        self.verification_time = verification_time

    @classmethod
    def create(cls, group_id, signer, identity, extensions=()):
        leaf, private = LeafNode.create(signer, identity)
        tree = Tree([leaf])
        context = GroupContext(1, 2, bytes(group_id), 0, tree.tree_hash(), b"", tuple(extensions))
        return cls(context, tree, EpochSecrets(os.urandom(32)), 0, signer, {leaf.encryption_key: private})

    @classmethod
    def join(cls, welcome, package, init_private, leaf_private, signer, *, tree_data=None, psks=None, external_sender=None):
        if isinstance(welcome, (bytes, bytearray, memoryview)):
            reader = Reader(welcome)
            welcome = Welcome.read(reader)
            reader.end()
        if welcome.suite != 2 or package.suite != 2:
            raise ValueError("Unsupported MLS Welcome cipher suite")
        if public_key(init_private) != package.init_key or public_key(leaf_private) != package.leaf.encryption_key or public_key(signer) != package.leaf.signature_key:
            raise ValueError("MLS Welcome private keys do not match their KeyPackage")
        reference = package.reference()
        entry = next((item for item in welcome.secrets if hmac.compare_digest(item[0], reference)), None)
        if entry is None:
            raise ValueError("MLS Welcome does not address this KeyPackage")
        plaintext = open_label(init_private, "Welcome", welcome.encrypted_group_info, entry[1], entry[2])
        reader = Reader(plaintext)
        joiner_secret = reader.opaque()
        path_secret = reader.optional(lambda body: body.opaque())
        psk_reader, psk_ids = reader.vector(), []
        while psk_reader:
            psk_ids.append(PreSharedKeyID.read(psk_reader))
        reader.end()
        if len(joiner_secret) != 32 or (path_secret is not None and len(path_secret) != 32):
            raise ValueError("MLS Welcome contains a malformed joiner or path secret")
        available = {} if psks is None else dict(psks)
        combined = psk_secret(psk_ids, available)
        welcome_secret = EpochSecrets.welcome(joiner_secret, combined)
        key = expand_label(welcome_secret, "key", b"", 16)
        nonce = expand_label(welcome_secret, "nonce", b"", 12)
        plaintext = AESGCM(key).decrypt(nonce, welcome.encrypted_group_info, b"")
        reader = Reader(plaintext)
        info = GroupInfo.read(reader)
        reader.end()
        tree_extension = dict(info.extensions).get(2)
        if tree_extension is not None:
            if tree_data is not None and tree_data != tree_extension:
                raise ValueError("MLS Welcome tree extension conflicts with the supplied tree")
            tree_data = tree_extension
        if tree_data is None:
            raise ValueError("MLS Welcome has no ratchet tree")
        tree = Tree.parse(tree_data)
        if info.context.tree_hash != tree.tree_hash():
            raise ValueError("MLS Welcome tree hash does not match its group context")
        tree.validate(info.context.group_id)
        info.verify(tree.leaf(info.signer).signature_key)
        own = [index for index, leaf in tree.members() if leaf == package.leaf]
        if len(own) != 1:
            raise ValueError("MLS Welcome tree does not contain this KeyPackage leaf exactly once")
        if external_sender is not None:
            ext = dict(info.context.extensions).get(5)
            if ext is None:
                raise ValueError("DAVE MLS group has no authenticated external sender")
            if external_senders(ext) != (external_sender,):
                raise ValueError("DAVE MLS group changed its pinned external sender")
        secrets = EpochSecrets.from_joiner(joiner_secret, info.context, combined)
        expected = secrets.confirmation(info.context.confirmed_transcript_hash)
        if not hmac.compare_digest(expected, info.confirmation_tag):
            raise ValueError("MLS Welcome confirmation tag failed")
        private_keys = {package.leaf.encryption_key: leaf_private}
        if path_secret is not None:
            own_path = set(direct_path(2 * own[0], tree.width)) | {2 * own[0]}
            sender_path = direct_path(2 * info.signer, tree.width)
            common = next((index for index in sender_path if index in own_path), None)
            if common is None:
                raise ValueError("MLS Welcome path secret has no shared tree ancestor")
            started = False
            for index in sender_path:
                if index == common:
                    started = True
                if started and tree.nodes[index] is not None:
                    private, public = _path_key(path_secret)
                    if public != tree.nodes[index].encryption_key:
                        raise ValueError("MLS Welcome path secret does not match its tree public key")
                    private_keys[public] = private
                    path_secret = derive_secret(path_secret, "path")
        return cls(info.context, tree, secrets, own[0], signer, private_keys,
                   interim=_interim(info.context.confirmed_transcript_hash, info.confirmation_tag), psks=available)

    def verify_message(self, message):
        if self.removed:
            raise ValueError("This member was removed from the MLS group")
        if message.group_id != self.context.group_id or message.epoch != self.context.epoch:
            raise ValueError("MLS message belongs to another group or epoch")
        if message.sender_type == 1:
            key = self.tree.leaf(message.sender).signature_key
            expected = hmac.digest(self.secrets.membership_key, message.tbs(self.context) + message.auth(), "sha256")
            if not hmac.compare_digest(expected, message.membership_tag):
                raise ValueError("MLS message membership tag failed")
        elif message.sender_type == 2:
            encoded = dict(self.context.extensions).get(5)
            if encoded is None:
                raise ValueError("MLS group has no external senders")
            senders = external_senders(encoded)
            if message.sender >= len(senders):
                raise ValueError("MLS external sender index is outside the group extension")
            if message.content_type != 2 or message.content.kind not in (1, 3, 4, 5, 7):
                raise ValueError("MLS external sender cannot send this content")
            key = senders[message.sender].signature_key
        elif message.sender_type == 3 and message.content_type == 2 and message.content.kind == 1:
            key = message.content.value.leaf.signature_key
        else:
            raise ValueError("Unsupported DAVE MLS sender type")
        if not verify_label(key, "FramedContentTBS", message.tbs(self.context), message.signature):
            raise ValueError("MLS message signature failed")
        return message

    def queue(self, message):
        if not isinstance(message, PublicMessage):
            message = parse_message(message)
        self.verify_message(message)
        if message.content_type != 2:
            raise ValueError("MLS proposal queue requires proposal content")
        self.proposals[message.reference()] = message
        return message.reference()

    def export(self, label, context, length):
        if self.removed:
            raise ValueError("This member was removed from the MLS group")
        return self.secrets.export(label, context, length)

    def _apply(self, commit, sender):
        proposals, seen = [], set()
        for item in commit.proposals:
            if isinstance(item, Proposal):
                proposals.append((item, sender))
            else:
                if item in seen or item not in self.proposals:
                    raise ValueError("MLS Commit has a repeated or unavailable proposal reference")
                seen.add(item)
                message = self.proposals[item]
                proposals.append((message.content, message.sender if message.sender_type == 1 else None))
        tree, ext = self.tree.clone(), self.context.extensions
        removals = [proposal.value for proposal, _ in proposals if proposal.kind == 3]
        if len(removals) != len(set(removals)) or sender in removals:
            raise ValueError("MLS Commit removes the committer or removes a member twice")
        for index in removals:
            tree.leaf(index)
        updates = set()
        for proposal, source in proposals:
            if proposal.kind != 2:
                continue
            if source is None or source == sender or source in updates:
                raise ValueError("MLS Commit contains an invalid or duplicate Update proposal")
            updates.add(source)
            leaf = proposal.value
            previous = tree.leaf(source)
            if leaf.source != 2 or leaf.credential != previous.credential:
                raise ValueError("MLS Update changes the member identity or has the wrong source")
            leaf.verify(self.context.group_id, source)
            if source not in removals:
                tree.nodes[2 * source] = leaf
                tree.blank_path(source)
        for index in removals:
            tree.remove(index)
        added, psks, extension_updates = [], [], 0
        for proposal, _ in proposals:
            if proposal.kind == 1:
                package = proposal.value.verify(now=self.verification_time)
                for _, leaf in tree.members():
                    if package.leaf.signature_key == leaf.signature_key or package.leaf.encryption_key == leaf.encryption_key:
                        raise ValueError("MLS Add reuses a group signature or encryption key")
                added.append((package, tree.add(package.leaf)))
            elif proposal.kind == 4:
                if proposal.value.kind == 2 and proposal.value.usage != 1:
                    raise ValueError("MLS Commit uses a branch or reinit PSK outside that transition")
                psks.append(proposal.value)
            elif proposal.kind == 7:
                extension_updates += 1
                ext = proposal.value
            elif proposal.kind not in (2, 3):
                raise ValueError("DAVE does not support this MLS proposal transition")
        if extension_updates > 1:
            raise ValueError("MLS Commit changes the group extensions more than once")
        for _, leaf in tree.members():
            if 1 not in leaf.capabilities[0] or 2 not in leaf.capabilities[1] or 1 not in leaf.capabilities[4]:
                raise ValueError("MLS group member does not support the DAVE suite")
        path_required = not proposals or any(proposal.kind in (2, 3, 7) for proposal, _ in proposals)
        return tree, ext, added, psks, path_required

    def _new_group(self, context, tree, secrets, private_keys, confirmation):
        expected = {tree.leaf(self.own_index).encryption_key}
        expected.update(tree.nodes[index].encryption_key for index in direct_path(2 * self.own_index, tree.width) if tree.nodes[index] is not None)
        keys = {public: private for public, private in private_keys.items() if public in expected}
        return Group(context, tree, secrets, self.own_index, self.signer, keys,
                     interim=_interim(context.confirmed_transcript_hash, confirmation), psks=self.psks,
                     verification_time=self.verification_time)

    def stage_commit(self, message):
        if not isinstance(message, PublicMessage):
            message = parse_message(message)
        self.verify_message(message)
        if message.content_type != 3 or message.sender_type != 1:
            raise ValueError("DAVE requires an authenticated MLS member Commit")
        if message.sender == self.own_index:
            if self.pending is None or self.pending[0] != message.message():
                raise ValueError("MLS own Commit does not match the pending transition")
            return self.pending[1]
        commit = message.content
        tree, ext, added, psks, path_required = self._apply(commit, message.sender)
        if commit.path is None and path_required:
            raise ValueError("MLS Commit is missing its required UpdatePath")
        if commit.path is not None:
            merge_path(tree, self.context.group_id, message.sender, commit.path)
        provisional = replace(self.context, epoch=self.context.epoch + 1, tree_hash=tree.tree_hash(), extensions=ext)
        confirmed = hashlib.sha256(self.interim + uint(1, 2) + message.framed() + opaque(message.signature)).digest()
        context = replace(provisional, confirmed_transcript_hash=confirmed)
        if self.own_index * 2 >= tree.width or tree.nodes[self.own_index * 2] is None:
            removed = Group(context, tree, self.secrets, self.own_index, self.signer, {},
                            interim=_interim(confirmed, message.confirmation_tag), psks={})
            removed.removed = True
            return removed
        private_keys = dict(self.private_keys)
        if commit.path is None:
            commit_secret = bytes(32)
        else:
            commit_secret, new_keys = decrypt_path(tree, message.sender, commit.path, private_keys, provisional, {index for _, index in added})
            private_keys.update(new_keys)
        joiner = EpochSecrets.joiner(self.secrets.init_secret, commit_secret, context)
        secrets = EpochSecrets.from_joiner(joiner, context, psk_secret(psks, self.psks))
        if not hmac.compare_digest(secrets.confirmation(confirmed), message.confirmation_tag):
            raise ValueError("MLS Commit confirmation tag failed")
        tree.validate(context.group_id)
        return self._new_group(context, tree, secrets, private_keys, message.confirmation_tag)

    def merge_commit(self, message):
        result = self.stage_commit(message)
        self.__dict__.update(result.__dict__)
        return self

    def create_commit(self, proposals=None, *, force_path=False):
        if self.removed:
            raise ValueError("This member was removed from the MLS group")
        references = tuple(self.proposals) if proposals is None else tuple(proposals)
        commit = Commit(references, None)
        tree, ext, added, psks, path_required = self._apply(commit, self.own_index)
        private_keys = dict(self.private_keys)
        path_secrets = {}
        if path_required or force_path:
            tree.blank_path(self.own_index)
            pairs = tree.filtered_path(self.own_index)
            secret = os.urandom(32)
            for index, _ in pairs:
                private, public = _path_key(secret)
                private_keys[public] = private
                path_secrets[index] = secret
                tree.nodes[index] = ParentNode(public, b"", ())
                secret = derive_secret(secret, "path")
            commit_secret = secret
            parent_hash = b""
            for index, other in reversed(pairs):
                tree.nodes[index] = replace(tree.nodes[index], parent_hash=parent_hash)
                parent_hash = tree.parent_hash(index, other)
            leaf, leaf_private = LeafNode.create(self.signer, tree.leaf(self.own_index).credential.identity,
                                                source=3, parent_hash=parent_hash,
                                                group_id=self.context.group_id, index=self.own_index)
            private_keys[leaf.encryption_key] = leaf_private
            tree.nodes[self.own_index * 2] = leaf
            provisional = replace(self.context, epoch=self.context.epoch + 1, tree_hash=tree.tree_hash(), extensions=ext)
            nodes = []
            excluded = {index for _, index in added}
            for index, other in pairs:
                encrypted = tuple(seal_label(tree.nodes[target].encryption_key, "UpdatePathNode", provisional.encode(), path_secrets[index])
                                  for target in tree.resolution(other, excluded))
                nodes.append(UpdatePathNode(tree.nodes[index].encryption_key, encrypted))
            commit = Commit(references, UpdatePath(leaf, tuple(nodes)))
        else:
            commit_secret = bytes(32)
            provisional = replace(self.context, epoch=self.context.epoch + 1, tree_hash=tree.tree_hash(), extensions=ext)
        message = PublicMessage(self.context.group_id, self.context.epoch, 1, self.own_index, b"", 3, commit, b"", b"", b"")
        signature = sign_label(self.signer, "FramedContentTBS", message.tbs(self.context))
        message = replace(message, signature=signature)
        confirmed = hashlib.sha256(self.interim + uint(1, 2) + message.framed() + opaque(signature)).digest()
        context = replace(provisional, confirmed_transcript_hash=confirmed)
        joiner = EpochSecrets.joiner(self.secrets.init_secret, commit_secret, context)
        combined = psk_secret(psks, self.psks)
        secrets = EpochSecrets.from_joiner(joiner, context, combined)
        confirmation = secrets.confirmation(confirmed)
        message = replace(message, confirmation_tag=confirmation)
        membership = hmac.digest(self.secrets.membership_key, message.tbs(self.context) + message.auth(), "sha256")
        message = replace(message, membership_tag=membership)
        tree.validate(context.group_id)
        staged = self._new_group(context, tree, secrets, private_keys, confirmation)
        welcome = None
        if added:
            info = GroupInfo(context, ((2, tree.encode()),), confirmation, self.own_index, b"")
            info = replace(info, signature=sign_label(self.signer, "GroupInfoTBS", info.unsigned()))
            welcome_secret = EpochSecrets.welcome(joiner, combined)
            key, nonce = expand_label(welcome_secret, "key", b"", 16), expand_label(welcome_secret, "nonce", b"", 12)
            encrypted_info = AESGCM(key).encrypt(nonce, info.encode(), b"")
            entries = []
            for package, index in added:
                shared = set(direct_path(index * 2, tree.width))
                common_secret = next((path_secrets[node] for node in direct_path(self.own_index * 2, tree.width) if node in shared and node in path_secrets), None)
                group_secrets = opaque(joiner) + (b"\x00" if common_secret is None else b"\x01" + opaque(common_secret))
                group_secrets += opaque(b"".join(identity.encode() for identity in psks))
                enc, ciphertext = seal_label(package.init_key, "Welcome", encrypted_info, group_secrets)
                entries.append((package.reference(), enc, ciphertext))
            welcome = Welcome(2, tuple(entries), encrypted_info)
        self.pending = message.message(), staged
        return message, welcome
