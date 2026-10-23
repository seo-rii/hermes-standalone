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
