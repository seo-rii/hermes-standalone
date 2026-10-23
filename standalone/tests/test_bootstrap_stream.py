"""Large stream boundaries, integrity order and decoded-file ownership."""
import copy
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import zipfile

from standalone.tests.test_bootstrap import fixture_archive, write_artifact


def stored_archive(extra):
    raw=fixture_archive(extra)
    output=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw))as source,zipfile.ZipFile(output,'w')as target:
        for info in source.infolist():
            data=source.read(info)
            info=copy.copy(info)
            info.compress_type=zipfile.ZIP_STORED
            target.writestr(info,data)
    return output.getvalue()


class BootstrapStreamTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='hermes bootstrap stream ')
        self.addCleanup(temporary.cleanup)
        self.artifact=Path(temporary.name)/'standalone fixture.py'

    def namespace(self,archive,**options):
        write_artifact(self.artifact,archive,**options)
        namespace={'__name__':'bootstrap_stream_fixture','__file__':str(self.artifact)}
        exec(compile(self.artifact.read_text(),str(self.artifact),'exec'),namespace)
        return namespace

    def test_base85_input_across_multiple_blocks_restores_exact_zip(self):
        resource=random.Random(20261023).randbytes(1250003)
        archive=fixture_archive({'vendor/fixturepkg/large.bin':resource})
        namespace=self.namespace(archive)
        with namespace['_decode_archive']()as decoded:
            self.assertEqual(decoded.read(),archive)

    def test_xz_drains_multiple_bounded_outputs_without_more_input(self):
        archive=stored_archive({'vendor/fixturepkg/large.bin':b'a'*(3*1024*1024+19)})
        namespace=self.namespace(archive,compression='lzma')
        factory=namespace['_decoder'];limits=[];sizes=[]
        class ObservedDecoder:
            def __init__(self,decoder):self.decoder=decoder
            def __getattr__(self,name):return getattr(self.decoder,name)
            def decompress(self,data,*,max_length):
                limits.append(max_length)
                result=self.decoder.decompress(data,max_length=max_length)
                sizes.append(len(result))
                return result
        namespace['_decoder']=lambda compression:ObservedDecoder(factory(compression))
        with namespace['_decode_archive']()as decoded:
            self.assertEqual(decoded.read(),archive)
        self.assertGreater(len(limits),3)
        self.assertTrue(all(0<value<=1024*1024 for value in limits))
        self.assertTrue(all(value<=1024*1024 for value in sizes))

    def test_checksum_failure_precedes_temporary_file_and_decoder(self):
        namespace=self.namespace(fixture_archive(),compression='lzma',
                                 info_edit=lambda info:info.update(payload_sha256='0'*64))
        with mock.patch.object(namespace['tempfile'],'TemporaryFile',side_effect=AssertionError('unverified input')):
            with self.assertRaisesRegex(namespace['BundleError'],'integrity check failed'):
                namespace['_decode_archive']()

    def test_success_and_validation_failure_close_the_owned_decoded_file(self):
        namespace=self.namespace(fixture_archive())
        archive,*_=namespace['_read_archive']()
        source=archive.fp
        self.assertFalse(source.closed)
        archive.close()
        self.assertTrue(source.closed)
        namespace=self.namespace(fixture_archive({'vendor/forbidden.so':b'not an executable'}))
        factory=namespace['tempfile'].TemporaryFile;opened=[]
        def record(*args,**kwargs):
            source=factory(*args,**kwargs);opened.append(source);return source
        with mock.patch.object(namespace['tempfile'],'TemporaryFile',side_effect=record):
            with self.assertRaises(namespace['BundleError']):namespace['_read_archive']()
        self.assertEqual(len(opened),1)
        self.assertTrue(opened[0].closed)
