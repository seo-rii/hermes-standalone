"""Original Thai algorithms and consumed resources, with a native-free runtime."""
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zlib

from standalone import vendor

ROOT=Path(__file__).resolve().parents[2]


class ThaiPhonemizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='hermes-thai-tests-')
        cls.addClassCleanup(cls.temporary.cleanup);cls.directory=Path(cls.temporary.name)
        cls.library=cls.directory/'vendor';cache=ROOT/'standalone/.cache/downloads'
        packages=vendor.load_dependencies(ROOT/'standalone/dependencies.json')['packages']
        for package in packages:
            if package['name']not in {'tltk','unicode-rbnf'}:continue
            if not(cache/package['filename']).is_file():raise unittest.SkipTest('Explicitly fetch the reviewed Thai source archives first')
            for name,content in vendor.read_wheel(package,cache)[0].items():
                target=cls.library/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)

    def run_isolated(self,body):
        home=self.directory/self._testMethodName;home.mkdir()
        prefix=f'import sys;sys.path[:0]={[str(self.library),str(ROOT)]!r}\n'
        prefix+='''
import hashlib,importlib.abc,json
from pathlib import Path
import zlib
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]in {'numpy','pandas','nltk','sklearn','sklearn_crfsuite','gensim','ctypes','subprocess','onnxruntime','torch','piper','tltk'}:
            raise AssertionError('Thai phonemizer attempted unrelated/native dependency '+fullname)
sys.meta_path.insert(0,NoNative())
def no_external(event,args):
    if event.startswith('socket.')or event in {'subprocess.Popen','os.system'}:
        raise AssertionError('Thai fixture attempted a network or external program')
sys.addaudithook(no_external)
from hermes_tltk import nlp
from standalone.local_piper_thai import ThaiPhonemizer
'''
        prefix+=f"fixture=json.loads(zlib.decompress(Path({str(ROOT/'standalone/tests/fixtures/local_piper_thai.json.zlib')!r}).read_bytes()))\n"
        environment={'HOME':str(home),'USERPROFILE':str(home),'PATH':'','PYTHONDONTWRITEBYTECODE':'1'}
        for name in ('SYSTEMROOT','WINDIR'):
            if name in os.environ:environment[name]=os.environ[name]
        child=subprocess.run([sys.executable,'-I','-S','-B','-X','utf8','-c',prefix+body],cwd=home,env=environment,
                             capture_output=True,text=True,encoding='utf8',timeout=600)
        self.assertEqual(child.returncode,0,child.stderr+child.stdout)
        return json.loads(child.stdout)

    def test_all_679_original_th2ipa_and_incoming_g2p_transcripts(self):
        report=self.run_isolated('''
cases=fixture['engine']['cases'];assert len(cases)==679
profile=fixture['error_profiles'][str(sys.version_info.major)+str(sys.version_info.minor)]
errors={row['case']:row['error']for row in profile['cases']}
original=nlp.g2p;observed=[]
def capture(text):
    value=original(text);observed.append(value);return value
nlp.g2p=capture
for index,case in enumerate(cases):
    observed.clear()
    try:output=nlp.th2ipa(case['text']);error=None
    except Exception as exc:output=None;error={'type':type(exc).__name__,'message':str(exc)}
    assert observed==[case['g2p']],(index,case['text'],'g2p')
    assert output==case['output'],(index,case['text'],'phones')
    assert error==errors.get(index,case['error']),(index,case['text'],'error')
print(json.dumps({'cases':len(cases),'all_g2p_inputs_and_outputs':True}))
''')
        self.assertEqual(report,{'cases':679,'all_g2p_inputs_and_outputs':True})

    def test_all_310_original_piper_number_tone_and_normalization_consumers(self):
        report=self.run_isolated('''
cases=fixture['piper']['cases'];assert len(cases)==310
instances={expand:ThaiPhonemizer(expand_numbers=expand)for expand in (False,True)}
for index,case in enumerate(cases):
    instance=instances[case['expand_numbers']]
    assert instance._normalize(case['text'])==case['normalized'],(index,case['text'],'normalization')
    assert instance.phonemize(case['text'])==case['output'],(index,case['text'],'phonemes')
print(json.dumps({'cases':len(cases),'expand_numbers':[False,True]}))
''')
        self.assertEqual(report,{'cases':310,'expand_numbers':[False,True]})

    def test_exact_original_resources_and_distribution_metadata(self):
        from importlib.metadata import Distribution
        for name,version in (('tltk','1.10'),('unicode-rbnf','2.4.0')):
            paths=list(self.library.glob(name.replace('-','_')+'-'+version+'.dist-info'))
            self.assertEqual(len(paths),1)
            metadata=Distribution.at(paths[0]);self.assertEqual(metadata.version,version)
            self.assertTrue(any('LICENSE'in path.name.upper()for path in paths[0].rglob('*')))
        fixture=json.loads(zlib.decompress((ROOT/'standalone/tests/fixtures/local_piper_thai.json.zlib').read_bytes()))
        for name,row in fixture['consumed_resources'].items():
            content=(self.library/'hermes_tltk'/name).read_bytes()
            self.assertEqual(len(content),row['bytes'],name)
            self.assertEqual(hashlib.sha256(content).hexdigest(),row['sha256'],name)


if __name__=='__main__':unittest.main(verbosity=2)
