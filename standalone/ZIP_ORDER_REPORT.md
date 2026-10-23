# Lossless archive ordering and LZMA2 experiment

The useful result is an LZMA2 option change: retain preset 8, set `pb=0`, and
use a 48 MiB dictionary. On the complete measured archive it saved another
**294,416 bytes (0.5944%)** compared with unmodified preset 8. The resulting
XZ stream decoded byte for byte under the existing 64 MiB decoder memory
limit. XZ carries these filter parameters, so this option requires no new
payload format or inverse transform.

```python
lzma.compress(
    archive,
    format=lzma.FORMAT_XZ,
    filters=[{
        "id": lzma.FILTER_LZMA2,
        "preset": 8,
        "pb": 0,
        "dict_size": 48 << 20,
    }],
)
```

This report describes the existing candidate snapshot, not the final build
with later ports. The input was `dist/hermes-candidate.py` (63,123,804 bytes),
containing a **285,135,878-byte** stored ZIP with **25,093 entries**. Its SHA256
was `dbf510a70e2a12eca73730b0aeab2f7f66f09b5c5ea789cb5f01b1d8e5022af8`.
Measurements used CPython 3.14.7 on 2026-10-21, without importing bundled code.

| Complete archive encoding | Payload bytes | Encode seconds |
| --- | ---: | ---: |
| LZMA2 preset 6, prior measurement | 50,454,880 | 154.00 |
| LZMA2 preset 8, prior measurement | 49,533,348 | 153.15 |
| Preset 8, `pb=0`, 48 MiB dictionary | **49,238,932** | 157.57 |

The last payload SHA256 was
`0d82e8e24f8c6ebba57f77ec6ae2176b1fbc45107fd06c42d7422f0918f5e46a`.
Every decoded block was compared with the original ZIP, and the accumulated
SHA256 and total length also matched. The **64 MiB bound is the LZMA decoder
state limit**, not a bound on total Python process memory or the 285 MB ZIP.
The dictionary increases from 32 MiB to 48 MiB; encoding and decoding retain
the existing standard-library `lzma` dependency.

## Ordering results

Ordering was screened on a deterministic, path-hash-selected sample with
4,682 complete ZIP records: 46,137,013 body bytes and 46,549,481 bytes including
headers and the reversible index. The sample included 20 MiB JSON, 18 MiB
Python, 3 MiB gzip resources, 1 MiB Markdown and 2 MiB other content. No file
body, source, metadata, resource or license was omitted or normalized.

All variants used preset 8 and passed exact decompression with the same
64 MiB decoder limit. The sample baseline was repeated independently and
produced the same compressed SHA256 and byte count.

| Deterministic ordering / preprocessing | Sample compressed bytes |
| --- | ---: |
| Original record order | 8,905,580 |
| Original order, separate header plane | 8,901,236 |
| Extension, then path | 8,879,940 |
| Extension, then path; separate header plane | **8,872,664** |
| Extension, basename, then path | 8,934,060 |
| Extension, size bin, then size and path | 8,946,976 |
| Extension, normalized line-structure fingerprint, size bin and path | 8,920,868 |
| Same structure fingerprint; separate header plane | 8,908,092 |

The best ordering saved only 32,916 bytes (0.370%) on this sample. It needs
an additional inverse representation, unlike the measured LZMA2 option above.
It is **not recommended as the default** on this evidence, and no complete
archive ordering compression sweep was run.

Its inverse was nevertheless tested on the full archive. The representation
preserved every original local header, file body and central-directory byte,
plus the inverse permutation and header lengths. It occupied 285,286,454
bytes before compression. Reconstruction required no external original ZIP
or metadata and recovered the original **285,135,878 bytes and exact SHA256**.

The entire archive contained 648 exact-duplicate body groups, comprising
2,599 files and 2,849,361 duplicate bytes. Only 127,316 duplicate bytes were
more than 32 MiB from the previous identical body. Duplicate grouping alone
therefore has limited additional opportunity beyond the existing dictionary.

## Compression parameter screening

The same unchanged sample gave these results. The full-archive run above was
limited to the most useful option; no repeated large full-input sweep ran.

| LZMA2 option | Sample compressed bytes |
| --- | ---: |
| Preset 8 defaults | 8,905,580 |
| `pb=0`, default 32 MiB dictionary | 8,896,480 |
| `pb=0`, `lc=4`, default dictionary | 8,895,016 |
| `pb=1`, default dictionary | 8,900,508 |
| `pb=0`, 48 MiB dictionary | **8,889,584** |

Local experiment scripts and structured results are under
`standalone/.cache/size-*`; they are not runtime dependencies. The full-input
job exited 0 after 160.84 seconds; its private log is
`~/logs/hermes-size-order-full-params-kkifr4hl.log` and its completion record
is the adjacent `.status.json`. The sample ordering and parameter jobs exited
0 after 181.06 and 107.34 seconds respectively.
