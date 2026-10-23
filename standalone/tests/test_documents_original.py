"""Read documents through the original public read_file tool, without native readers."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone.tests import test_documents as _documents
from standalone.tests import test_transport_agent as _agent_fixture


WORKER = r'''
import contextlib,io,json,os,pathlib,sys
source,vendor,scratch=map(pathlib.Path,sys.argv[1:])
sys.path[:0]=[str(source),str(vendor)]
from standalone.runtime import initialize_bundle
initialize_bundle({"root":str(source.parent),"artifact":str(scratch/'fixture.py'),
    "bundle_sha256":source.parent.name,"manifest":{"project":"hermes-standalone"}})
home=pathlib.Path(os.environ['HERMES_HOME']);home.mkdir(parents=True)
(home/'config.yaml').write_text(json.dumps({'terminal':{'backend':'local','cwd':str(scratch/'work')},
    'file_tools':{'hosted_ocr':False},'agent':{'environment_probe':False}}))
def audit(event,args):
    if event.startswith('socket.') and event not in ('socket.__new__','socket.gethostname'):
        raise PermissionError('Document fixture forbids network access')
    if event in ('subprocess.Popen','os.system','os.exec'):
        raise PermissionError('Document fixture forbids external programs')
sys.addaudithook(audit)
from tools.read_extract import extract_document_text,extract_document_bytes,is_extractable_document
from tools.file_tools import read_file_tool
out,err=io.StringIO(),io.StringIO()
with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
    results=[]
    for name,expected in [('fixture.pdf','Original PDF fixture'),('fixture.odt','한글 문서'),('fixture.rtf','한글')]:
        path=scratch/'work'/name
        assert is_extractable_document(str(path))
        assert expected in extract_document_text(str(path))
        assert expected in extract_document_bytes(path.read_bytes(),str(path))
        result=json.loads(read_file_tool(str(path),task_id='document-'+name))
        assert result.get('extracted_document') is True,result
        assert expected in result['content'],result
        results.append(result['total_lines'])
    corrupt=scratch/'work'/'corrupt.pdf';corrupt.write_bytes(b'%PDF-1.4\ntruncated')
    refused=json.loads(read_file_tool(str(corrupt),task_id='corrupt-document'))
    assert refused.get('error') and 'extraction failed' in refused['error'],refused
    assert not any(name=='anydoc' or name=='cryptography' or name.startswith('cryptography.') for name in sys.modules)
print(json.dumps({'ok':True,'lines':results}))
'''


@unittest.skipIf(sys.version_info < (3, 14), "Original Hermes targets Python 3.14")
class OriginalDocumentTests(unittest.TestCase):
    setUpClass = classmethod(_agent_fixture.AgentHTTPIntegrationTests.setUpClass.__func__)

    def test_original_read_file_extracts_pdf_odf_rtf_and_rejects_corruption(self):
        from standalone.agent_probe_worker import fixture_environment
        with tempfile.TemporaryDirectory(prefix="documents-", dir=self.directory) as directory:
            scratch = Path(directory)
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            (scratch / "work/fixture.pdf").write_bytes(_documents.pdf())
            (scratch / "work/fixture.rtf").write_bytes(b"{\\rtf1\\ansi \\u54620?\\u44544?}")
            (scratch / "work/fixture.odt").write_bytes(_documents.archive({"content.xml":
                '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><text:p>한글 문서</text:p></office:body></office:document-content>'}))
            result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", WORKER,
                                     str(self.source), str(self.vendor), str(scratch)],
                                    env=fixture_environment(scratch), cwd=scratch / "work", capture_output=True,
                                    text=True, encoding="utf-8", timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout)["ok"])
