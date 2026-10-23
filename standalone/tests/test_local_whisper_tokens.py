"""Private original-Rust/faster-whisper tokenizer contract validation."""
import base64,importlib.abc,json,sys,unittest,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from standalone.local_whisper_tokenizer import ByteBPETokenizer
from standalone.local_whisper_tokens import Tokenizer
class NoNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.',1)[0] in {'tokenizers','numpy','ctranslate2','torch','onnxruntime'}:raise AssertionError(fullname)
class TokenizerTests(unittest.TestCase):
 def setUp(self):
  guard=NoNative();sys.meta_path.insert(0,guard);self.addCleanup(sys.meta_path.remove,guard)
 def test_original_tasks_languages_suppression_and_unicode_word_boundaries(self):
  fixture=json.loads((ROOT/'fixtures/local_whisper_tokens.json').read_text())
  for model in fixture['models']:
   graph=zlib.decompress(base64.b64decode(model['graph_zlib_base64'],validate=True));tokenizer=ByteBPETokenizer(graph)
   for record in model['wrappers']:
    wrapper=Tokenizer(tokenizer,model['model']=='tiny',record['task'],record['language'])
    for name,expected in record['properties'].items():
     with self.subTest(model=model['model'],language=record['language'],task=record['task'],property=name):self.assertEqual(json.loads(json.dumps(getattr(wrapper,name))),expected)
    for vector in record['vectors']:
     with self.subTest(model=model['model'],language=record['language'],task=record['task'],text=vector['text']):
      self.assertEqual(wrapper.encode(vector['text']),vector['encoded']);self.assertEqual(wrapper.decode(vector['tokens']),vector['decoded']);self.assertEqual(wrapper.decode_with_timestamps(vector['tokens']),vector['decoded_timestamps'])
      for method,key in [('split_tokens_on_unicode','unicode'),('split_tokens_on_spaces','spaces'),('split_to_word_tokens','words')]:self.assertEqual(json.loads(json.dumps(getattr(wrapper,method)(vector['tokens']))),vector[key])
 def test_invalid_multilingual_tasks_and_languages_refuse(self):
  model=json.loads((ROOT/'fixtures/local_whisper_tokens.json').read_text())['models'][1];tokenizer=ByteBPETokenizer(zlib.decompress(base64.b64decode(model['graph_zlib_base64'])))
  for task,language in [('unknown','en'),('transcribe','xx'),(None,'en'),('transcribe',None)]:
   with self.assertRaises(ValueError):Tokenizer(tokenizer,True,task,language)
if __name__ == '__main__':
 unittest.main(verbosity=2)
