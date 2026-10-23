# Publication review

This repository is a development fork. Public source availability does not
mean complete Hermes parity or a completed security audit. The current
capabilities and known validation gaps remain explicit in
[capabilities.json](capabilities.json) and [BUILD_REPORT.md](BUILD_REPORT.md).

The publication review checks the current source tree and changes from the
pinned public upstream baseline. It excludes generated distributions, local
download/authoring caches, credentials, local logs and private working notes.
Those files are not part of the public Git tree.

The initial privacy review scans 21,707 candidate files with credential/path
rules, parses 139 JSON files and 23 compressed JSON files, and checks 329,319
base64-shaped values. Identifying machine paths in 23 provenance/report files
are removed. Original metadata copies are retained outside Git; resource,
model and fixture hashes are unchanged. The additional Japanese data files
are checked against their complete pinned original size/SHA-256 manifest.

Credential-like strings in the added tests are offline fixtures or parser
markers. The encrypted RSA fixture is an encrypted copy of the pinned
upstream public TLS test key; all nine private integer parameters match that
public fixture. It is not a user or service credential. Tests use owned
temporary profiles and local peers, without the developer's live accounts.

This review does not contact providers to test credentials, inspect private
user state, or guarantee discovery of every possible secret representation.
Public upstream history is the supplied baseline, rather than newly audited
private content. Repository publication sends only the reviewed current
session changes and that public ancestry.

The implementation review also fixes dashboard asset staging: source/icon
receipts and installed dependency identities must match, and required runtime
license notices must be present. Oversized source models/dictionaries use
bounded, hash-checked lossless restoration from committed chunks. These
authoring checks run with the Python standard library alone.

The final static scan covers 21,783 candidate files (665,047,330 bytes), including
1,518 new or changed files. No unresolved service-credential or private local
project context is identified. Matches retained in the tree are reviewed public
upstream examples, offline test data or compressed-byte coincidences. Later
publication documentation is reviewed separately before committing.

The earlier compiled dashboard is not published: its package notices and an
optional build-backend dependency are incomplete. Source preparation now fails
closed for those inputs. A fresh dashboard producer run remains unqualified;
default source builds omit the browser assets.

The new source-only build and actual copied CLI/gateway checks are recorded in
[verification/current-source-20261023.json](verification/current-source-20261023.json).
The initial full copied-artifact run passes11of12 methods. The remaining
offline fixture is blocked because the authoring temporary directory inherits
a Git ancestor; its unchanged repeat outside any Git ancestor passes. These
are disjoint observations, and the original failed run is retained. They do
not establish a continuously green full suite or complete feature parity.

The new public fork keeps GitHub Actions disabled. Local validation receipts
do not represent a successful GitHub CI run.
