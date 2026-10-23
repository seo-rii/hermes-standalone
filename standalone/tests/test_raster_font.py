"""Actual outline fonts, antialiasing, and browser contact-sheet operations."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from standalone import vendor


PROJECT = Path(__file__).resolve().parents[2]


class FontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        packages = vendor.load_dependencies(PROJECT / "standalone/dependencies.json")["packages"]
        package = next(package for package in packages if package["name"] == "fonttools")
        cache = PROJECT / "standalone/.cache/downloads"
        if not (cache / package["filename"]).is_file():
            raise unittest.SkipTest("Explicitly fetch the pinned fontTools source first")
        cls.temporary = tempfile.TemporaryDirectory(prefix="hermes-outline-font-tests-")
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
        prefix += "import base64, hashlib, io, json\nfrom pathlib import Path\n"
        prefix += f"fixture = json.loads(Path({str(PROJECT / 'standalone/tests/fixtures/font.json')!r}).read_text())\n"
        prefix += "from standalone import raster as Image, raster_font as ImageFont, raster_imageops as ImageOps\n"
        prefix += "from standalone.raster_draw import ImageDraw\n"
        prefix += '''
def no_external_runtime(event, arguments):
    if event.startswith("socket.") or event in ("subprocess.Popen", "os.system"):
        raise RuntimeError("Font fixtures cannot access a network or external program")
sys.addaudithook(no_external_runtime)
'''
        child = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", prefix + program],
                               cwd=profile, env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr + child.stdout)

    def test_pinned_default_font_metrics_and_visible_antialiased_glyphs(self):
        self.execute('''
from standalone.raster_fontdata import default_font_bytes
document=json.loads(Path(''' + repr(str(PROJECT / "standalone/font-provenance.json")) + ''').read_text())
assert hashlib.sha256(default_font_bytes()).hexdigest() == document['font_sha256']
font=ImageFont.load_default()
assert font.getname() == ('Aileron','Regular')
assert font.getmetrics() == (10,3)
assert 60 <= font.getlength('Hermes 00:01') <= 66
mask,offset=font.render('Hermes 00:01')
assert mask.mode == 'L' and mask.getbbox() is not None
assert any(0 < value < 255 for value in mask.tobytes())
assert font.getbbox('') == (0,0,0,0)
assert font.font_variant(size=20).getlength('Hermes') > font.getlength('Hermes')
assert not any(name == 'PIL' or name.startswith('PIL.') for name in sys.modules)
assert not any(name == 'freetype' or name.startswith('freetype.') for name in sys.modules)
''')

    def test_true_type_korean_cmap_quadratic_curves_and_contour_holes(self):
        self.execute('''
data=base64.b64decode(fixture['font_b64'])
path=Path('fixture.ttf');path.write_bytes(data)
font=ImageFont.truetype(path,40)
assert font.getname() == ('Hermes Fixture','Regular')
assert font.getmetrics() == (32,8)
assert abs(font.getlength('AV') - 48.4) < 0.001, 'GPOS must replace equivalent legacy kern pairs'
assert abs(font.getlength('OA') - 50.4) < 0.001
assert font.getlength('AV',features=['-kern']) == 52
mask,offset=font.render('O')
assert offset == (0,4)
assert mask.getpixel((1,10)) == 255
assert mask.getpixel((10,10)) == 0, 'Nonzero winding must preserve counter-shapes'
korean,_=font.render('한글')
unknown,_=font.render('??')
assert korean.tobytes() != unknown.tobytes()
assert korean.width == 52 and korean.getbbox() is not None
curve,_=font.render('e')
assert len(set(curve.tobytes())) > 4
assert font.getbbox('O',anchor='ls')[1] < 0
''')

    def test_postscript_cubic_outlines_and_composite_accent_glyphs(self):
        self.execute('''
font=ImageFont.truetype(io.BytesIO(base64.b64decode(fixture['otf_b64'])),36)
mask,_=font.render('O')
assert font.getname()[0] == 'Hermes Cubic Fixture'
assert mask.getbbox() is not None and len(set(mask.tobytes())) > 4
default=ImageFont.load_default(30)
plain,_=default.render('e')
accent,_=default.render('é')
assert accent.height > plain.height
assert accent.getbbox() is not None
''')

    def test_contact_sheet_title_masks_colors_and_aspect_preservation(self):
        self.execute('''
sheet=Image.new('RGB',(240,120),'#171a20')
draw=ImageDraw.Draw(sheet)
font=ImageFont.load_default()
draw.text((8,4),'Hermes 00:01',fill='#ffffff',font=font)
assert any(sheet.getpixel((x,y)) != (23,26,32) for y in range(5,17) for x in range(8,90))
box=draw.textbbox((8,4),'Hermes 00:01',font=font)
assert box[0] == 8 and 5 <= box[1] <= 7 and box[2] > 65
draw.rounded_rectangle((8,30,70,85),radius=12,fill='#000000',outline='#e2e7ec',width=2)
assert sheet.getpixel((8,30)) == (23,26,32)
assert sheet.getpixel((40,55)) == (0,0,0)
assert sheet.getpixel((40,30)) == (226,231,236)
preview=ImageOps.contain(sheet,(80,80),Image.Resampling.LANCZOS)
assert preview.size == (80,40)
assert ImageOps.fit(sheet,(50,50)).size == (50,50)
assert ImageOps.pad(sheet,(50,50),color='white').getpixel((0,0)) == (255,255,255)
draw.text((100,30),'ABC\\nxyz',font=ImageFont.load_default(18),fill='white')
assert draw.multiline_textbbox((100,30),'ABC\\nxyz',font=ImageFont.load_default(18))[3] > 60
assert Image.new('RGBA',(1,1),'#1234').getpixel((0,0)) == (17,34,51,68)
assert Image.new('LA',(1,1),'#fff8').getpixel((0,0)) == (255,136)
''')

    def test_font_validation_and_layout_limits_fail_explicitly(self):
        self.execute('''
for size in (0,-1,True,float('nan'),float('inf'),2049):
    try: ImageFont.load_default(size)
    except (ValueError,TypeError): pass
    else: raise AssertionError('Invalid font size accepted')
for data in (b'',b'not a font',b'wOF2'+b'0'*20):
    try: ImageFont.truetype(io.BytesIO(data),12)
    except OSError: pass
    else: raise AssertionError('Invalid font accepted')
font=ImageFont.load_default()
for options in ({'direction':'rtl'},{'language':'ar'},{'features':['liga']}):
    try: font.getbbox('text',**options)
    except NotImplementedError: pass
    else: raise AssertionError('Unsupported shaping silently accepted')
try: font.getbbox('x'*4097)
except ValueError: pass
else: raise AssertionError('Text length bound bypassed')
try: ImageFont.truetype(io.BytesIO(base64.b64decode(fixture['font_b64'])),12,layout_engine=ImageFont.Layout.RAQM)
except NotImplementedError: pass
else: raise AssertionError('Native shaping backend silently replaced')
''')


if __name__ == "__main__":
    unittest.main()
