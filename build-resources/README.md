# Original source assets

These files preserve the complete, unchanged Chinese G2PW model and Japanese
Open JTalk and Sudachi dictionaries. Each original is one XZ stream split at
16 MiB boundaries for source distribution. No model values, dictionary entries
or precision are removed.

After cloning, restore and verify them using only the Python standard library:

```sh
python3 -I -S scripts/standalone_assets.py restore
python3 -I -S scripts/standalone_assets.py verify
```

The manifest records each chunk, complete compressed stream and original file's
size and SHA-256. Restoration checks the compressed data before decoding,
bounds decoded output, and checks the original bytes before publishing a file.
Existing valid files are preserved; different existing files cause an error.

These are build inputs. The single-file distribution contains the original
runtime resources through the normal reviewed manifest and packing checks;
it does not contain a second copy of these source archives. Runtime restoration
and all resource integrity checks remain unchanged.

The source compressor uses XZ/LZMA2 with preset 8, `pb=0` and a 48 MiB dictionary,
without timestamps or host paths. Regenerating with the same compression
implementation and parameters is deterministic. The original resource hashes,
upstream download pins and license notices are recorded in:

- [Chinese resource provenance](../standalone/piper-chinese-resources-provenance.json)
- [Japanese resource provenance](../standalone/piper-japanese-resources-provenance.json)
- [Source license notices](../standalone/licenses/)
