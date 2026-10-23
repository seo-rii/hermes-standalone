# Build and verification report — 2026-10-23

This is a development fork of Hermes at upstream commit
`bd3cbe91a90097f9804f16ea5c1768f5cbaaf0b7`. Runtime support targets CPython3.14;
authoring and focused compatibility checks also run on CPython3.11.
`full_parity_verified` remains false. Feature-level results and limitations
are recorded in [capabilities.json](capabilities.json) and component provenance.

## Reproduction

Restore the source assets, fetch the pinned dependency archives explicitly,
and use the standard-library builder and test runner:

```sh
python3.14 -I -S scripts/standalone_assets.py restore
python3.14 -I -S scripts/standalone_assets.py verify
python3.14 -I -S standalone/vendor.py fetch
python3.14 -I -S standalone/build.py
python3.14 -I -S standalone/check.py
python3.14 -I -S standalone/check.py --artifact dist/hermes.py --artifact-only
```

Web assets are a separate optional preparation step documented in the main
README. Runtime commands do not install packages. Models, resources and
third-party notices are preserved under their original source manifests.
Generated distributions, downloaded inputs, local logs and development state
are excluded from Git. Validation log references in component provenance are
run identifiers; the machine-local log directories are not published.

The previously staged dashboard is excluded: 20 package records have no notice,
including 11 in the reconstructed dependency closure. The old installation also
has an incomplete optional WASM backend. Fresh producer validation and complete
authentic notices remain required. CI prepares the dashboard only when its
manual `include_dashboard` input is selected; the provenance gate refuses
incomplete inputs. Default source builds contain the CLI and gateway without
dashboard browser assets.

## Deterministic source and archive checks

App-source minification re-emits the parsed AST deterministically, then checks
location-independent AST equality, recursive compiled-code fingerprints and
idempotence. A failed check or size regression keeps the original file.
Names, literals, docstrings, assertions, annotations and notices are preserved.
Vendor files and hash-pinned Python model configurations remain unchanged.
The AST proof is emitted separately and bound by its hash in the manifest.

The final current-source publication selection passes145 focused tests on
both Python3.11 and3.14, including packaging, minification, runtime boundaries,
the26 new source-resource restore contracts and seven dashboard provenance
contracts. This selection is separate from the earlier failed full-source run.

All three oversized original resources restore offline and match every
original byte in disposable source checkouts on both interpreters. Their
464,828,816 original bytes occupy174,915,160 compressed bytes in11 chunks of
at most16MiB. Cold/hit/verify operations and cache-hit mtimes are checked;
native imports, network and external processes are blocked during this proof.
The receipt is [verification/source-assets-20261023.json](verification/source-assets-20261023.json).

The frozen verification input contains12,997 files and869,925,077bytes. Its
sorted compact path-to-SHA256 map has digest
`2522163856f5e72469a01a414af4f1b4dc44c73ffe1b711ac1286593ae13a126`.
Two normal and two AST builds reproduce exactly, including AST proof reports.
Restoring the former temporary snapshot into a persistent verification cache
also matches every original file hash, total size and map digest.

## Current public-source snapshot

A Git-only export of21,783 files restores all three oversized assets offline.
Every one of the489 original resource manifest entries matches its size and
SHA256. The offline fresh build completes in517.75seconds with build network
and external process execution blocked. It contains no ignored dashboard
browser assets. The resulting unminified ZIP application is252,751,498bytes,
SHA256`5bc4caa2b56aa981bdb7cde53776b3934f868fa38d7cfd5b85c58e138ecc2dfa`.
This is a different input selection from the frozen dashboard-containing
checkpoint below, rather than a matched-input compression saving.

The original copied-artifact runner completes12 methods in646.59seconds:
11pass, including the real managed gateway/worker HTTP tool turn, and one
offline fixture-turn check is blocked. The authoring harness placed its empty
temporary profile beneath a Git checkout. Upstream workspace discovery found
that Git ancestor and attempted a Git configuration subprocess; the offline
guard correctly refused execution. No guard was widened and no process ran.

The unchanged failed method passes in56.25seconds when its owned temporary
profile is outside all Git ancestors. All source and artifact hashes and the
original300-second command bound remain unchanged. Thus every original method
has an observed pass across disjoint runs; this is not a continuously green
12-method run or a complete source-suite result. Both the first failure and
repeat are retained in [verification/current-source-20261023.json](verification/current-source-20261023.json).
The receipt also verifies every selected app source against the final worktree
and scans the full decoded archive for exact author home/workspace prefixes.
Subsequent report/receipt edits are outside the bundled source selection.

## Optional ZIP application checkpoint

The optional `--container zipapp` keeps the exact compressed payload in a
ZIP_STORED container with a small Python bootstrap. It supports `.pyz` output;
source `.py` remains the default. The loader verifies compressed/decoded hashes,
size bounds, exact XZ EOF, archive members, manifests and the complete cache.
Its outer import path is removed before stdlib imports. The original managed
child identity, module arguments and isolated re-entry behavior remain.

| Artifact | Bytes | Saving against matched format4 source |
| --- | ---: | ---: |
| `hermes-zipapp-20261023.pyz` | 254,066,246 | 3,063,055 |
| `hermes-zipapp-ast-20261023.pyz` | 254,057,162 | 3,063,862 |

The SHA256 values are respectively
`da7cd602a01e669f318e180447b6a42e1279810739a7d444d4c024f72f35fc71`
and `51ba02cf8597d0bb96822f62134b6421acfdc062d3545b868d44ad946c8cb900`.
These are two exact envelope renders per existing frozen payload, rather than
new source builds. Compressed payload and inner ZIP bytes are unchanged; the
AST proof is unchanged. No canonical artifact is automatically replaced.

Both complete original11-method copied-artifact suites pass in uninterrupted
repeats, with their original assertions and time limits: normal589.46seconds,
AST516.19seconds. This includes actual agent construction/fixture turns,
resource/profile/voice checks and relocated selftests. Every frozen input,
source and artifact hash remains unchanged before and after the repeat.
The portable receipt is [verification/zipapp-20261023.json](verification/zipapp-20261023.json).
The earlier incomplete process had no final exit record and is not counted as
a pass; its nine partial passes and selftest error remain historical evidence.

Single-process Linux measurements, including compilation and all cache checks:

| Envelope | Cold peak RSS, KiB | Warm peak RSS, KiB | Cold/warm seconds |
| --- | ---: | ---: | --- |
| Normal ZIP | 832,564 | 73,604 | 53.85 / 28.30 |
| AST ZIP | 832,792 | 73,820 | 25.87 / 16.26 |

These are observations under different host contention, not latency guarantees.
The prior source-text loader uses much more compiler memory. All111 focused
packaging/runtime/minify checks pass on both interpreters. Ten final size/ZIP
checks also pass on both after the member-validation refactor; these are
separate completed runs, not a claimed single combined suite.

## Preserved failures and limits

Format4 source-text outputs are257,129,301bytes normal and257,121,024bytes AST.
Their SHA256 values are
`894d88199e1a5754d95f08f00d2e5cb856d1aa973e286a8fe79ffb6b6184388e`
and `782ac3d978855506fe017e6078d3f61307a991d330e258e5d2a8f973dd81e8b3`.
The normal format4 copied suite passes all11 methods. The AST suite fails two
60-second probes and one300-second selftest deadline. Profiling measures about
30–33 CPU seconds compiling/loading its large source envelope. These failures
remain valid; the ZIP checkpoint does not erase them or qualify every host.

The earlier complete canonical source run finished686 methods:613 passes,
68 declared skips, one stale PINYIN-unavailable expectation and four HTTP
timeouts. The expectation is corrected, and the timed cases pass an unchanged
frozen-source repeat with the same30/60-second limits. The original continuous
run failed; there is no claim of a green full-source regression run.
Older format3 copied suites also have recorded startup/probe timeouts. A
separate normal gateway repeat passed with a300-second outer bound; it is
not evidence of passing the earlier120-second bound.

Chinese G2PW consumed labels, IDs and29 whole small-voice cases match their
references. Internal probabilities fail the declared1e-6 tolerance, with
maximum error0.0027914643. Japanese default processing uses the full original
MeCab/Sudachi resources;9,001 controller inputs,126,017 labels and58 voice cases
pass. These are not trained-language acoustic-quality claims. Opus/FFmpeg
profile differences and unsupported codec/platform paths remain documented.
Native Windows/macOS, hardware and live-service equivalence are unverified.

## Current private inference work

KittenTTS remains inactive in the public runtime. Its private default-plan
port matches the observed quantizers and source-order MLAS/Eigen primitives;
104,778 Exp values and the final27-node isolated segment are exact on both
interpreters. Segment tests use independent boundary values and do not prove
whole inference. Full live API/WAV qualification is separate and incomplete.
No prerecorded inference result or reduced-precision model is used as a runtime
fallback. The first live API case now passes, but the remaining cases and public
provider integration are still unqualified.

A resource-only lossless patch experiment restores every original byte, but
including the shared base with the actual solid-compression settings makes it
269,632bytes larger:7,344,844bytes for shared originals versus7,614,476 for
patches plus decoder/manifests. The smaller shared-original layout is retained.
These figures are not whole-artifact savings. No resource, dictionary entry,
model weight or precision is discarded.

## Further lossless resource packing

Three additional length-preserving transforms improve the existing XZ payload:
MeCab feature columns are grouped together, one G2PW byte embedding is
transposed, and TLTK protocol-3 LONG_BINPUT operands use reversible uint32
differences. The pickle is parsed with `pickletools`, never unpickled by the
packer or bootstrap. All original opcodes, model values and dictionary fields
are restored before the existing original-size and SHA256 checks.

The new ZIP application is 249,809,397 bytes, SHA256
`fc6ffade6cea2e768da7b38658f57bcbad1f0f465e7876d319b2f171ac7c0346`.
Against the matched 252,751,498-byte public-source ZIP baseline above, this
saves 2,942,101 bytes, or 1.1640%, including the added bootstrap decoders.
The offline build took 950.02 seconds with network and external processes
blocked. Its payload is 249,753,400 bytes with SHA256
`b7889e95fde10bedd61dbbe737c5148f072d687b4d8896e8c10cb90d66f49bf1`.

Every original path, size and SHA256 matches the baseline: 4,795 app files and
21,402 vendor files, totaling 830,645,471 bytes before archive/manifest overhead.
The sorted compact original-file map has digest
`c4c2aace601de73804cc15ed38721bbb2ec1f6768c36d04f3f6d2ceb220e82f6`.
Dependency records, capabilities, licenses, runtime sources and the AST
minifier's equivalence checks are unchanged. The same preset 8, 48 MiB dictionary,
`pb=0` and 64 MiB XZ decoder memory limit remain in use.

Rendering the same two frozen payloads as default format-4 text source gives
255,803,935 bytes before and 252,824,883 bytes after: a saving of 2,979,052 bytes
(1.1646%). The new source artifact has SHA256
`a83df45754162ab5848fe24bad3c6375d359d035e131103fa78e9e7e1349f662`.
No second payload compression is involved in this envelope comparison.

Isolated input comparisons explain the transforms; their sum is not the
whole-artifact saving:

| Input | Original XZ bytes | Transformed XZ bytes |
| --- | ---: | ---: |
| Complete MeCab dictionary | 15,055,268 | 13,629,140 |
| One G2PW 21,128 × 768 byte tensor | 9,938,072 | 9,582,580 |
| Complete TLTK trigram pickle | 3,315,752 | 2,063,420 |

The builder proves exact inverse equality for each transformed resource.
Complete resources also round-trip and repeat identically on Python 3.11
and 3.14. This full new compressed payload was built once; repeated helper,
small-build and envelope results do not constitute two full builds.

The final 117-method focused source selection passes in uninterrupted runs
on both interpreters: Python 3.11 in 94.09 seconds and Python 3.14 in 89.33 seconds.
It covers the five container formats, original-byte restoration, child reentry,
cache corruption, malformed/unauthorized resources and the existing minifier.
A missing-decoder-placeholder regression first failed, then passes after the
builder guard correction; rendering the valid full ZIP artifact remains byte
identical. Minimal templates without a decoder registry remain supported.

Independent extraction checks on both interpreters verify the size and SHA256
of all 26,197 restored files, reuse the validated cache without rewriting its
manifest, then reject a modified original-resource cache without repairing it.
The same checks pass for the baseline. Network and external processes are
blocked throughout these byte checks.

The additional inverses increase cold work. Single-process Linux observations
include loading the artifact, decoding XZ and the first complete extraction:

| Python / artifact | Cold CPU seconds | Cold wall seconds | Peak RSS, KiB |
| --- | ---: | ---: | ---: |
| 3.14 / baseline ZIP | 30.49 | 36.73 | 835,312 |
| 3.14 / new ZIP | 42.65 | 57.58 | 832,916 |
| 3.11 / new ZIP | 50.73 | 61.29 | 835,644 |

These are one sample per case under host contention, not latency guarantees.
Existing caches contain original bytes and skip the inverse transforms;
payload and cache verification still run. The separately recorded warm
`_extract` timings exclude payload decoding and interpreter startup.

Four unchanged copied-ZIP acceptance methods pass in one continuous
313.06-second run with their original assertions and deadlines. They exercise
the relocated selftest, real Thai resources/phonemization, Chinese and Japanese
resources/voice PCM, and the managed gateway/worker HTTP tool turn with durable
receipts. Their empty owned profiles are outside all Git ancestors. This is a
focused artifact selection, not a new full-source regression run.

The default source artifact also passes ordinary launch, isolated reentry and
`doctor --json --require-agent` in 142.75 seconds with the original 300-second
bound. All six serial qualification phases complete with exit status 0.
Portable hashes, measurements and the exact validation scope are recorded in
[verification/lossless-resource-packing-20261023.json](verification/lossless-resource-packing-20261023.json).
