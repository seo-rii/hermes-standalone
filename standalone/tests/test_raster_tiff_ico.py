"""Independent TIFF/BigTIFF, fax and ICO fixtures without installed image codecs."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone import vendor


PROJECT = Path(__file__).resolve().parents[2]


class TIFFIconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        packages = vendor.load_dependencies(PROJECT / "standalone/dependencies.json")["packages"]
        package = next(item for item in packages if item["name"] == "pyjpeg")
        cache = PROJECT / "standalone/.cache/downloads"
        if not (cache / package["filename"]).is_file():
            raise unittest.SkipTest("Explicitly fetch the pinned JPEG dependency first")
        cls.temporary = tempfile.TemporaryDirectory(prefix="hermes-tiff-ico-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.library = cls.directory / "vendor"
        for relative, content in vendor.read_wheel(package, cache)[0].items():
            target = cls.library / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def execute(self, program):
        profile = self.directory / self._testMethodName
        profile.mkdir()
        environment = {"HOME": str(profile), "USERPROFILE": str(profile)}
        for name in ("SYSTEMROOT", "WINDIR"):
            if name in os.environ:
                environment[name] = os.environ[name]
        prefix = f"import sys; sys.path[:0] = {[str(self.library), str(PROJECT)]!r}\n"
        prefix += "import base64, hashlib, io, json, struct, zlib\nfrom pathlib import Path\n"
        prefix += f"document = json.loads(Path({str(PROJECT / 'standalone/tests/fixtures/tiff-ico.json')!r}).read_text())\n"
        prefix += "from standalone import raster as Image, raster_tiff as tiff, raster_ico as ico\n"
        prefix += '''
def no_external_runtime(event, arguments):
    if event.startswith("socket.") or event in ("subprocess.Popen", "os.system"):
        raise RuntimeError("Image fixtures cannot access a network or external program")
sys.addaudithook(no_external_runtime)
'''
        child = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", prefix + program],
                               cwd=profile, env=environment, capture_output=True, text=True, timeout=45)
        self.assertEqual(child.returncode, 0, child.stderr + child.stdout)

    def test_native_oracle_pages_strips_tiles_predictors_and_icon_alternatives(self):
        self.execute('''
for case in document['cases']:
    data=base64.b64decode(case['data'])
    assert hashlib.sha256(data).hexdigest() == case['sha256']
    image=(tiff if case['format']=='TIFF' else ico).decode(data)
    assert image.format == case['format'],case['name']
    assert image.n_frames == len(case['frames']),case['name']
    image.verify()
    for index,frame in enumerate(case['frames']):
        image.seek(index)
        assert image.size == tuple(frame['size']),case['name']
        expected=zlib.decompress(base64.b64decode(frame['rgba']))
        actual=image.convert('RGBA').tobytes()
        assert len(actual) == len(expected),case['name']
        assert max(abs(a-b) for a,b in zip(actual,expected)) <= (3 if 'jpeg' in case['name'] else 0),case['name']
assert not any(name=='PIL' or name.startswith('PIL.') for name in sys.modules)
assert not any(name in ('ctypes','numpy','tifffile') for name in sys.modules)
''')

    def test_deterministic_writers_match_independent_native_decoder(self):
        self.execute('''
for case in document['encoding_cases']:
    raw=zlib.decompress(base64.b64decode(case['input']))
    source=Image.frombytes(case['mode'],tuple(case['size']),raw)
    codec=tiff if case['format']=='TIFF' else ico
    data=codec.encode(source,**case['options'])
    assert hashlib.sha256(data).hexdigest()==case['sha256'],case['mode']
    assert data==codec.encode(source,**case['options'])
    actual=codec.decode(data).convert('RGBA').tobytes()
    assert actual==zlib.decompress(base64.b64decode(case['rgba']))
    assert actual==source.convert('RGBA').tobytes()
for options in ({'sizes':[]},{'sizes':[(257,257)]}):
    try:ico.encode(source,**options)
    except ValueError:pass
    else:raise AssertionError('invalid ICO output size accepted')
''')

    def test_all_pages_alternatives_directory_bounds_and_cycles_are_validated(self):
        self.execute('''
cases={case['name']:base64.b64decode(case['data']) for case in document['cases']}
data=cases['multipage'];reader=tiff._Reader(data);first=reader.offset
count=reader.unpack('H',first)[0]
cyclic=bytearray(data);struct.pack_into('<I',cyclic,first+2+count*12,first)
oversized=bytearray(cases['classic-little'])
reader=tiff._Reader(oversized)
for n in range(reader.unpack('H',reader.offset)[0]):
    position=reader.offset+2+n*12
    if reader.unpack('H',position)[0]==256:struct.pack_into('<I',oversized,position+8,100000001)
for malformed in (b'',b'II*\\0',data[:7],data[:len(data)//2],bytes(cyclic),bytes(oversized)):
    try:tiff.decode(malformed)
    except (ValueError,OSError):pass
    else:raise AssertionError('malformed TIFF accepted')
icon=bytearray(cases['RGBA-bmp'])
offset=struct.unpack_from('<I',icon,18)[0]
icon[offset:offset+4]=b'BAD!'
for malformed in (b'',b'\\0\\0\\1\\0',cases['RGBA-bmp'][:-1],bytes(icon)):
    try:ico.decode(malformed)
    except (ValueError,OSError):pass
    else:raise AssertionError('malformed ICO alternative accepted')
''')

    def test_compression_bounds_and_fax_corruption_are_refused(self):
        self.execute('''
from standalone.raster_fax import decode as fax
for data,kind,expected in ((b'\\0',5,1),(b'\\x80',32773,1),(b'\\xff',32773,2),
                           (zlib.compress(b'x'*100000),8,1),(b'xx',1,1)):
    try:tiff._decompress(data,kind,expected)
    except (ValueError,zlib.error):pass
    else:raise AssertionError('invalid TIFF compressed size accepted')
for data,width,height,kind,t4,t6 in ((b'',13,7,4,0,0),(b'\\0',13,7,2,0,0),
                                    (b'\\x02',13,7,4,0,0),(b'\\x80',1,1,4,0,2),
                                    (b'\\x80',1,1,3,2,0)):
    try:fax(data,width,height,kind,t4,t6)
    except ValueError:pass
    else:raise AssertionError('malformed fax accepted')
''')


if __name__ == "__main__":
    unittest.main()
