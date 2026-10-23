"""Exercise reversible resource packing through the real generated launcher."""

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from standalone import build
from standalone.mecab_pack import transform_mecab
from standalone.pickle_pack import transform_pickle_memo
from standalone.tests.test_bootstrap import fixture_archive, TEMPLATE, write_artifact
from standalone.tests.test_mecab_pack import fixture_mecab


MECAB = 'app/standalone/local_piper_japanese/data/openjtalk/dictionary/sys.dic'
G2PW = 'app/standalone/piper_chinese_data/model/g2pw.onnx'
THAI = 'vendor/hermes_tltk/sylseg.3g'


def resource_fixture():
    # Include opcode-looking bytes inside a length-delimited string. Neither
    # parsing nor restoration may treat these bytes as memo instructions.
    string = b'r\xff\xff\xff\xff\x80\x03.'
    thai = (b'\x80\x03]r' + struct.pack('<I', 0xfffffffe)
            + b'B' + struct.pack('<I', len(string)) + string
            + b'r' + struct.pack('<I', 1) + b'a.')
    originals = {MECAB: fixture_mecab(), THAI: thai}
    packed, encodings = build._pack_data_files(originals)
    return originals, packed, encodings


def packed_archive(originals, packed, encodings, *, digest=None):
    def describe(manifest):
        for row in manifest['files']:
            name = row['path']
            if name in encodings:
                row.update(encoding=encodings[name],
                           sha256=digest or hashlib.sha256(originals[name]).hexdigest())
    return fixture_archive(packed, manifest_edit=describe)


class ResourcePackingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='hermes-resource-pack-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.artifact = self.directory / 'resource fixture.py'
        self.cache = self.directory / 'private cache'

    def run_artifact(self, *args):
        return subprocess.run(
            [sys.executable, '-I', '-S', str(self.artifact),
             '--cache-dir', str(self.cache), *args],
            cwd=self.directory, env=dict(os.environ),
            capture_output=True, text=True, encoding='utf-8', timeout=30,
        )

    def test_original_bytes_restore_in_all_five_formats_and_managed_children(self):
        originals, packed, encodings = resource_fixture()
        archive = packed_archive(originals, packed, encodings)
        variants = [('deflate', 'base85'), ('lzma', 'base85'),
                    ('lzma', 'latin1'), ('lzma', 'latin1-chunks'), ('zipapp', None)]
        for number, (compression, encoding) in enumerate(variants, 1):
            with self.subTest(format=number):
                self.cache = self.directory / f'cache-{number}'
                if compression == 'zipapp':
                    self.artifact.write_bytes(build.render_zipapp(TEMPLATE, archive, 'fixture'))
                else:
                    source = build.render_artifact(TEMPLATE, archive, 'fixture',
                                                   compression=compression,
                                                   payload_encoding=encoding)
                    self.artifact.write_bytes(source.encode(
                        'latin-1' if encoding.startswith('latin1') else 'utf-8'))
                result = self.run_artifact()
                self.assertEqual(result.returncode, 0, result.stderr)
                root = Path(json.loads(result.stdout)['root'])
                for name, original in originals.items():
                    self.assertEqual((root / name).read_bytes(), original)
                before = (root / 'bundle-manifest.json').stat().st_mtime_ns
                child = self.run_artifact('child', 'two words')
                self.assertEqual(child.returncode, 0, child.stderr)
                self.assertEqual(json.loads(child.stdout)['argv'], ['two words'])
                self.assertEqual((root / 'bundle-manifest.json').stat().st_mtime_ns, before)
                target = root / THAI
                target.write_bytes(packed[THAI])
                failed = self.run_artifact()
                self.assertEqual(failed.returncode, 2, failed.stderr)
                self.assertIn('cache integrity', failed.stderr)
                self.assertEqual(target.read_bytes(), packed[THAI])

    def test_each_codec_rejects_unreviewed_paths_before_publishing_cache(self):
        _, packed, encodings = resource_fixture()
        cases = [(MECAB, encodings[MECAB], packed[MECAB]),
                 (G2PW, 'g2pw-embedding-v1', b'small-invalid-model'),
                 (THAI, encodings[THAI], packed[THAI])]
        for approved, encoding, content in cases:
            for target in ('app/standalone/encoded.py', 'vendor/other/resource.dat'):
                with self.subTest(codec=encoding, target=target):
                    archive = packed_archive({target: content}, {target: content},
                                             {target: encoding})
                    write_artifact(self.artifact, archive)
                    result = self.run_artifact()
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn('file encoding', result.stderr)
                    self.assertFalse(self.cache.exists() and any(self.cache.iterdir()))

    def test_corrupt_encodings_and_original_hash_mismatch_leave_no_cache(self):
        originals, packed, encodings = resource_fixture()
        broken_mecab = packed[MECAB][:-1] + b'x'
        for name, content, encoding, digest, message in (
            (MECAB, broken_mecab, encodings[MECAB], None, 'packed resource'),
            (THAI, b'\x80\x02N.', encodings[THAI], None, 'packed resource'),
            (G2PW, b'small-invalid-model', 'g2pw-embedding-v1', None, 'packed resource'),
            (MECAB, packed[MECAB], encodings[MECAB], '0' * 64, 'restored model integrity'),
            (THAI, packed[THAI], encodings[THAI], '0' * 64, 'restored model integrity'),
        ):
            with self.subTest(codec=encoding, message=message):
                original = originals.get(name, content)
                archive = packed_archive({name: original}, {name: content},
                                         {name: encoding}, digest=digest)
                write_artifact(self.artifact, archive)
                result = self.run_artifact()
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse(self.cache.exists() and any(self.cache.iterdir()))

    def test_unhashable_encoding_rejects_as_invalid_manifest(self):
        name = 'vendor/other/data'
        archive = packed_archive({name: b'data'}, {name: b'data'}, {name: []})
        write_artifact(self.artifact, archive)
        result = self.run_artifact()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('file encoding', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        self.assertFalse(self.cache.exists() and any(self.cache.iterdir()))

    def test_builder_preserves_original_mapping_and_ignores_other_paths(self):
        originals, packed, encodings = resource_fixture()
        self.assertEqual(encodings, {MECAB: 'mecab-columns-v1', THAI: 'pickle-memo-delta-v1'})
        self.assertEqual(transform_mecab(packed[MECAB], inverse=True), originals[MECAB])
        self.assertEqual(transform_pickle_memo(packed[THAI], inverse=True), originals[THAI])
        self.assertEqual(build._pack_data_files({'vendor/other/sys.dic': originals[MECAB]}), ({}, {}))
        for name in originals:
            self.assertEqual(len(originals[name]), len(packed[name]))
            self.assertNotEqual(originals[name], packed[name])
        for filename in ('mecab_pack.py', 'model_pack.py', 'pickle_pack.py'):
            self.assertIn(filename, build.EXCLUDED_STANDALONE_FILES)

    def test_builder_refuses_noninvertible_or_length_changing_transform(self):
        def changed_length(data, *, inverse=False):
            return data + b'x'

        def wrong_inverse(data, *, inverse=False):
            return b'y' * len(data)

        for transform in (changed_length, wrong_inverse):
            module = SimpleNamespace(transform=transform)
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda unused: None))
            with self.subTest(transform=transform.__name__), \
                    mock.patch.object(build, 'PACKED_RESOURCES',
                                      {'data': ('fixture.py', 'transform', 'fixture')}), \
                    mock.patch.object(build.importlib.util, 'spec_from_file_location', return_value=spec), \
                    mock.patch.object(build.importlib.util, 'module_from_spec', return_value=module):
                with self.assertRaisesRegex(build.BuildError, 'exact inverse'):
                    build._pack_data_files({'data': b'original'})

    def test_resource_marker_must_appear_once_and_rendering_is_deterministic(self):
        originals, packed, encodings = resource_fixture()
        archive = packed_archive(originals, packed, encodings)
        first = build.render_zipapp(TEMPLATE, archive, 'fixture')
        self.assertEqual(first, build.render_zipapp(TEMPLATE, archive, 'fixture'))
        for invalid in (TEMPLATE + '\n__HERMES_RESOURCE_PACK_SOURCE__\n',
                        TEMPLATE.replace('__HERMES_RESOURCE_PACK_SOURCE__', '')):
            with self.subTest(marker_count=invalid.count('__HERMES_RESOURCE_PACK_SOURCE__')):
                with self.assertRaisesRegex(build.BuildError, 'resource decoder placeholder'):
                    build.render_artifact(invalid, archive, 'fixture')
