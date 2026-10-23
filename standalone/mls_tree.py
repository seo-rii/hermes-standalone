"""RFC 9420 left-balanced TreeKEM public trees and parent authentication."""

from __future__ import annotations

from dataclasses import replace
import hashlib

from .mls_codec import Reader, opaque, uint
from .mls_crypto import _public
from .mls_types import LeafNode, ParentNode


MAX_LEAVES = 4096


def level(index):
    return (index ^ (index + 1)).bit_length() - 1


def root(width):
    return (1 << (width.bit_length() - 1)) - 1


def left(index):
    return index ^ (1 << (level(index) - 1))


def right(index, width):
    candidate = index ^ (3 << (level(index) - 1))
    while candidate >= width:
        candidate = left(candidate)
    return candidate


def parent(index, width):
    if index == root(width):
        raise ValueError("The MLS root has no parent")
    while True:
        k = level(index)
        index = (index | (1 << k)) & ~(1 << (k + 1))
        if index < width:
            return index


def sibling(index, width):
    p = parent(index, width)
    return right(p, width) if index < p else left(p)


def direct_path(index, width):
    result = []
    while index != root(width):
        index = parent(index, width)
        result.append(index)
    return tuple(result)


def copath(index, width):
    result = []
    while index != root(width):
        result.append(sibling(index, width))
        index = parent(index, width)
    return tuple(result)


def descendants(index, width):
    if index % 2 == 0:
        return {index // 2}
    return descendants(left(index), width) | descendants(right(index, width), width)


class Tree:
    def __init__(self, nodes):
        self.nodes = list(nodes)
        if not self.nodes or len(self.nodes) % 2 == 0 or len(self.nodes) > 2 * MAX_LEAVES - 1:
            raise ValueError("MLS tree has an invalid node count")
        for index, node in enumerate(self.nodes):
            if node is not None and isinstance(node, LeafNode) != (index % 2 == 0):
                raise ValueError("MLS node type does not match its tree position")
        # MLS exports omit trailing blanks. Tree hashing includes the implicit
        # right-hand blank subtree up to the smallest complete binary tree.
        capacity = 1 << ((len(self.nodes) + 1) // 2 - 1).bit_length()
        self.nodes.extend([None] * (2 * capacity - 1 - len(self.nodes)))

    @classmethod
    def parse(cls, data):
        reader = Reader(data)
        body = reader.vector()
        nodes = []
        while body:
            present = body.uint(1)
            if present == 0:
                nodes.append(None)
            elif present == 1:
                kind = body.uint(1)
                if kind not in (1, 2):
                    raise ValueError("Unknown MLS tree node type")
                nodes.append((LeafNode if kind == 1 else ParentNode).read(body))
            else:
                raise ValueError("Invalid MLS optional node discriminant")
        reader.end()
        if not nodes or nodes[-1] is None:
            raise ValueError("Exported MLS trees cannot have trailing blank nodes")
        return cls(nodes)

    def encode(self):
        end = len(self.nodes)
        while end and self.nodes[end - 1] is None:
            end -= 1
        return opaque(b"".join(b"\x00" if node is None else b"\x01" + bytes([1 if isinstance(node, LeafNode) else 2]) + node.encode() for node in self.nodes[:end]))

    def clone(self):
        return Tree(self.nodes)

    @property
    def width(self):
        return len(self.nodes)

    def leaf(self, index):
        if type(index) is not int or index < 0 or 2 * index >= self.width:
            raise ValueError("MLS leaf index outside the tree")
        node = self.nodes[index * 2]
        if node is None:
            raise ValueError("MLS leaf is blank")
        return node

    def members(self):
        return tuple((index // 2, node) for index, node in enumerate(self.nodes) if isinstance(node, LeafNode))

    def resolution(self, index, excluded=()):
        node = self.nodes[index]
        if node is not None:
            if isinstance(node, LeafNode):
                return [] if index // 2 in excluded else [index]
            return [index] + [2 * leaf for leaf in node.unmerged_leaves if leaf not in excluded]
        if index % 2 == 0:
            return []
        return self.resolution(left(index), excluded) + self.resolution(right(index, self.width), excluded)

    def filtered_path(self, leaf_index):
        path, sides = direct_path(2 * leaf_index, self.width), copath(2 * leaf_index, self.width)
        return tuple((p, side) for p, side in zip(path, sides) if self.resolution(side))

    def tree_hash(self, index=None, excluded=frozenset()):
        if index is None:
            index = root(self.width)
        node = self.nodes[index]
        if index % 2 == 0:
            if index // 2 in excluded:
                node = None
            body = b"\x01" + uint(index // 2, 4) + (b"\x00" if node is None else b"\x01" + node.encode())
        else:
            if node is not None and excluded:
                node = replace(node, unmerged_leaves=tuple(leaf for leaf in node.unmerged_leaves if leaf not in excluded))
            body = b"\x02" + (b"\x00" if node is None else b"\x01" + node.encode())
            body += opaque(self.tree_hash(left(index), excluded)) + opaque(self.tree_hash(right(index, self.width), excluded))
        return hashlib.sha256(body).digest()

    def parent_hash(self, index, copath_index):
        node = self.nodes[index]
        if not isinstance(node, ParentNode):
            raise ValueError("MLS parent hash requires a nonblank parent")
        sibling_hash = self.tree_hash(copath_index, frozenset(node.unmerged_leaves))
        return hashlib.sha256(opaque(node.encryption_key) + opaque(node.parent_hash) + opaque(sibling_hash)).digest()

    def validate(self, group_id):
        encryption_keys, signature_keys = set(), set()
        for index, leaf in self.members():
            leaf.verify(group_id, index)
            if leaf.encryption_key in encryption_keys or leaf.signature_key in signature_keys:
                raise ValueError("Duplicate MLS leaf encryption or signature key")
            encryption_keys.add(leaf.encryption_key)
            signature_keys.add(leaf.signature_key)
        for index, node in enumerate(self.nodes):
            if not isinstance(node, ParentNode):
                continue
            _public(node.encryption_key)
            if node.encryption_key in encryption_keys:
                raise ValueError("Duplicate MLS tree encryption key")
            encryption_keys.add(node.encryption_key)
            if not set(node.unmerged_leaves) <= descendants(index, self.width):
                raise ValueError("MLS unmerged leaf is outside its parent subtree")
            for leaf in node.unmerged_leaves:
                self.leaf(leaf)
                for intermediate in direct_path(2 * leaf, self.width):
                    if intermediate == index:
                        break
                    ancestor = self.nodes[intermediate]
                    if ancestor is not None and leaf not in ancestor.unmerged_leaves:
                        raise ValueError("MLS unmerged leaf is absent from an intermediate parent")
            candidates = 0
            for child, other in ((left(index), right(index, self.width)), (right(index, self.width), left(index))):
                resolution = set(self.resolution(child))
                expected_unmerged = {2 * leaf for leaf in node.unmerged_leaves if leaf in descendants(child, self.width)}
                expected_hash = self.parent_hash(index, other)
                for candidate in resolution:
                    descendant = self.nodes[candidate]
                    if resolution - {candidate} == expected_unmerged and descendant.parent_hash == expected_hash:
                        candidates += 1
            if candidates != 1:
                raise ValueError("MLS parent node lacks a unique authenticated leaf chain")
        return self

    def blank_path(self, leaf):
        for index in direct_path(2 * leaf, self.width):
            self.nodes[index] = None

    def add(self, leaf):
        available = next((index for index in range(0, self.width, 2) if self.nodes[index] is None), None)
        if available is None:
            if self.width >= 2 * MAX_LEAVES - 1:
                raise ValueError("MLS group exceeds its leaf bound")
            available = self.width + 1
            self.nodes.extend([None] * (self.width + 1))
        self.nodes[available] = leaf
        leaf_index = available // 2
        for index in direct_path(available, self.width):
            node = self.nodes[index]
            if node is not None:
                self.nodes[index] = replace(node, unmerged_leaves=tuple(sorted(set(node.unmerged_leaves) | {leaf_index})))
        return leaf_index

    def remove(self, leaf):
        self.leaf(leaf)
        self.nodes[2 * leaf] = None
        self.blank_path(leaf)
        for index, node in enumerate(self.nodes):
            if isinstance(node, ParentNode) and leaf in node.unmerged_leaves:
                self.nodes[index] = replace(node, unmerged_leaves=tuple(value for value in node.unmerged_leaves if value != leaf))
        last = max((index for index, node in enumerate(self.nodes) if node is not None), default=0)
        capacity = 1 << (last // 2).bit_length()
        self.nodes = self.nodes[:2 * capacity - 1]
