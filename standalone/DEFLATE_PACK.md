# Gzip preprocessing experiment

This build-time prototype is rejected for the default bundle. Preserve the
existing solid LZMA payload: exposing inflated JSON plus a reversible DEFLATE
recipe made the measured archive larger and added Python reconstruction time.

`deflate_pack.pack(data)` returns a deterministic private representation, or
`None` for non-gzip input. `unpack(data, max_output=...)` recreates the original
gzip bytes directly, without any gzip/zlib encoder or decoder. The format magic
is `b"HERMES-GZ1\0"`. Gzip headers/trailers, original dynamic-tree encoding,
stored/fixed/dynamic blocks, exact LZ77 choices and final padding are retained.
CRC, SHA256, framing, history and output bounds are checked.

The complete pinned botocore 1.42.97 gzip resource set was measured on isolated
CPython 3.14.7. Its wheel SHA256 is
`77d2c8ce1bc592d3fbd7c01c35836f4a5b0cac2ca03ccdf6ffc60faa16b5fadc`.
The comparison frames the same sorted paths and byte lengths into solid LZMA6.

| Measurement | Bytes |
| --- | ---: |
| Original gzip resources (915 files) | 13,582,035 |
| Inflated JSON | 109,746,470 |
| Recipe and envelope | 16,849,727 |
| Total preprocessing representation | 126,596,197 |
| Solid LZMA6 of original gzip resources | 13,363,816 |
| Solid LZMA6 after preprocessing | 19,557,300 |

All 915 resources reconstructed byte for byte. The result is **46.35% larger**
after solid compression, with 113,014,162 extra decoded bytes. Reconstruction
took 21.91 seconds for the complete set; preprocessing took 13.26 seconds.
The separate codec profile peaked at 55.5 MiB RSS from a 36.9 MiB baseline;
the largest individual representation was 4,543,829 bytes, restoring a
491,336-byte gzip file. These timings describe this machine, not a runtime
guarantee.

Six independent tests cover native compression levels 0–9, multiple window and
memory settings, fixed/dynamic/stored and multiple blocks, history resets,
extended gzip headers/FHCRC, concatenated members, nonzero final padding,
corruption, truncation and bounds. Tests disable every encoder and decoder
during reconstruction. A bit-oriented match-recipe experiment also passed the
vector suite, but was not pursued after the first variant's runtime cost made
default integration impractical; the retained source is the measured variant.
