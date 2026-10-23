"""JPEG fixture conformance without importing an installed image library."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone import vendor


PROJECT = Path(__file__).resolve().parents[2]


class JPEGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        packages = vendor.load_dependencies(PROJECT / "standalone/dependencies.json")["packages"]
        package = next(item for item in packages if item["name"] == "pyjpeg")
        cache = PROJECT / "standalone/.cache/downloads"
        if not (cache / package["filename"]).is_file():
            raise unittest.SkipTest("Explicitly fetch the reviewed JPEG codec first")
        cls.temporary = tempfile.TemporaryDirectory(prefix="hermes-jpeg-tests-")
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
        prefix += "import base64, hashlib, json, zlib\nfrom pathlib import Path\n"
        prefix += f"fixture_document = json.loads(Path({str(PROJECT / 'standalone/tests/fixtures/jpeg.json')!r}).read_text())\nfixtures = fixture_document['cases']\n"
        prefix += "from standalone import raster_jpeg as jpeg\n"
        prefix += '''
def no_external_runtime(event, arguments):
    if event.startswith("socket.") or event in ("subprocess.Popen", "os.system"):
        raise RuntimeError("JPEG fixtures cannot access a network or external program")
sys.addaudithook(no_external_runtime)
'''
        child = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", prefix + program],
                               cwd=profile, env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr + child.stdout)
        self.assertNotIn("PIL", child.stdout)

    def test_sequential_progressive_sampling_restart_and_color_fixture_pixels(self):
        self.execute('''
for case in fixtures:
    data = base64.b64decode(case["jpeg_b64"])
    assert hashlib.sha256(data).hexdigest() == case["jpeg_sha256"]
    width, height, mode, pixels = jpeg.decode(data)
    assert (width, height, mode) == (case["width"], case["height"], case["mode"]), case["name"]
    expected = zlib.decompress(base64.b64decode(case["pixels_zlib_b64"]))
    assert len(pixels) == len(expected)
    differences = [abs(a-b) for a,b in zip(pixels, expected)]
    # Floating-point DCT rounding can differ slightly from libjpeg's integer DCT.
    assert max(differences) <= 3, (case["name"], max(differences))
    assert sum(differences) / len(differences) < 0.3, case["name"]
assert not any(name == "PIL" or name.startswith("PIL.") for name in sys.modules)
''')

    def test_encoder_matches_committed_independent_decoder_results(self):
        self.execute('''
for case in fixture_document['encoding_cases']:
    pixels=zlib.decompress(base64.b64decode(case['input_zlib_b64']))
    encoded=jpeg.encode(case['width'],case['height'],case['mode'],pixels,case['quality'])
    assert hashlib.sha256(encoded).hexdigest() == case['jpeg_sha256']
    independently_decoded=zlib.decompress(base64.b64decode(case['pixels_zlib_b64']))
    decoded=jpeg.decode(encoded)[3]
    assert max(abs(a-b) for a,b in zip(decoded,independently_decoded)) <= 2
''')

    def test_single_component_sampling_factors_do_not_change_pixel_coordinates(self):
        self.execute('''
case=next(case for case in fixtures if case['name']=='gray')
data=bytearray(base64.b64decode(case['jpeg_b64']))
position=data.index(b"\\xff\\xc0"); data[position+11]=0x44
decoded=jpeg.decode(bytes(data))
assert decoded == jpeg.decode(base64.b64decode(case['jpeg_b64']))
''')

    def test_encoder_roundtrips_odd_edges_quality_and_deterministic_output(self):
        self.execute('''
for mode in ("L", "RGB"):
    channels = 1 if mode == "L" else 3
    width, height = 17, 9
    pixels = bytes(value for y in range(height) for x in range(width)
                   for value in ((x*7+y*3,) if channels == 1 else (x*7,y*13,(x+y)*5)))
    high = jpeg.encode(width,height,mode,pixels,quality=95)
    assert high == jpeg.encode(width,height,mode,pixels,quality=95)
    assert high.startswith(b"\\xff\\xd8") and high.endswith(b"\\xff\\xd9")
    decoded = jpeg.decode(high)
    assert decoded[:3] == (width,height,mode)
    error = sum(abs(a-b) for a,b in zip(decoded[3],pixels)) / len(pixels)
    assert error < 2, (mode,error)
    low = jpeg.encode(width,height,mode,pixels,quality=20)
    assert len(low) < len(high)
    assert jpeg.decode(jpeg.encode(1,1,mode,bytes([123]*channels)))[:3] == (1,1,mode)
''')

    def test_malformed_streams_and_pixel_limits_are_rejected(self):
        self.execute('''
original = base64.b64decode(next(case for case in fixtures if case['name']=='baseline420')['jpeg_b64'])
for malformed in (b"", b"not JPEG", original[:20], original[:-2], original[:-15]):
    try: jpeg.decode(malformed)
    except jpeg.JPEGError: pass
    else: raise AssertionError("Truncated JPEG accepted")
for limit in (0, -1, True, 10):
    try: jpeg.decode(original,max_pixels=limit)
    except jpeg.JPEGError: pass
    else: raise AssertionError("JPEG pixel bound bypassed")
oversized=bytearray(original); sof=oversized.index(b"\\xff\\xc0")
oversized[sof+5:sof+9]=b"\\xff\\xff\\xff\\xff"
try: jpeg.decode(bytes(oversized),max_pixels=1000)
except jpeg.JPEGError: pass
else: raise AssertionError("Oversized JPEG allocated")
restart=bytearray(base64.b64decode(next(case for case in fixtures if case['name']=='baseline-restart')['jpeg_b64']))
position=restart.index(b"\\xff\\xd0"); restart[position+1]=0xd3
try: jpeg.decode(bytes(restart))
except jpeg.JPEGError: pass
else: raise AssertionError("Out-of-order restart accepted")
progressive=bytearray(base64.b64decode(next(case for case in fixtures if case['name']=='progressive420')['jpeg_b64']))
position=progressive.index(b"\\xff\\xda"); length=int.from_bytes(progressive[position+2:position+4],'big')
progressive[position+1+length]=0x21
try: jpeg.decode(bytes(progressive))
except jpeg.JPEGError: pass
else: raise AssertionError("Refinement without initial coefficients accepted")
''')

    def test_invalid_encoder_arguments_and_unsupported_coding_fail_explicitly(self):
        self.execute('''
for args in [(0,1,"L",b"",75),(1,1,"RGBA",b"1234",75),(1,1,"RGB",b"12",75),
             (1,1,"L",b"x",0),(1,1,"L",b"x",101),(1,1,"L",b"x",True)]:
    try: jpeg.encode(*args)
    except jpeg.JPEGError: pass
    else: raise AssertionError("Invalid encoder argument accepted")
data=bytearray(base64.b64decode(fixtures[0]['jpeg_b64'])); marker=data.index(b"\\xff\\xc0");data[marker+1]=0xc9
try: jpeg.decode(bytes(data))
except jpeg.JPEGError as error: assert "Unsupported JPEG marker" in str(error)
else: raise AssertionError("Arithmetic coding silently misinterpreted")
''')


if __name__ == "__main__":
    unittest.main()
