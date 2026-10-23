"""Independent image fixtures and real Hermes vision/pet consumers."""

import base64
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

from standalone import raster as Image


PROJECT = Path(__file__).resolve().parents[2]
FIXTURES = PROJECT / "standalone/tests/fixtures/raster.json"


class RasterTests(unittest.TestCase):
    def test_independently_decoded_pixels_every_frame_and_interlacing(self):
        for case in json.loads(FIXTURES.read_text())["cases"]:
            with self.subTest(case=case["name"]):
                image = Image.open(io.BytesIO(base64.b64decode(case["data"])))
                self.assertEqual(image.size, tuple(case["size"]))
                self.assertEqual(image.format, case["format"])
                self.assertEqual(image.n_frames, len(case["frames"]))
                image.verify()
                for index, frame in enumerate(Image.ImageSequence.Iterator(image)):
                    expected = zlib.decompress(base64.b64decode(case["frames"][index]["rgba"]))
                    self.assertEqual(frame.convert("RGBA").tobytes(), expected)
                    image.seek(index)
                    self.assertEqual(image.convert("RGBA").tobytes(), expected)
                with self.assertRaises(EOFError):
                    image.seek(image.n_frames)

    def test_png_bmp_gif_output_and_bounded_palette_roundtrip(self):
        source = Image.frombytes("RGB", (19, 13), bytes(c for y in range(13) for x in range(19)
                                                    for c in ((x % 5) * 50, (y % 5) * 50, 17)))
        for kind in ("PNG", "BMP", "GIF"):
            with self.subTest(kind=kind):
                first, second = io.BytesIO(), io.BytesIO()
                source.save(first, format=kind)
                source.save(second, format=kind)
                self.assertEqual(first.getvalue(), second.getvalue())
                decoded = Image.open(io.BytesIO(first.getvalue())).convert("RGB")
                self.assertEqual(decoded.tobytes(), source.tobytes())
        palette = source.quantize(colors=3)
        self.assertLessEqual(len(set(palette.getdata())), 3)
        self.assertEqual(len(palette.getpalette()), 768)

    def test_compositing_crop_resample_and_channel_operations(self):
        source = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
        source.paste((20, 40, 80, 255), (2, 1, 6, 7))
        self.assertEqual(source.getbbox(), (2, 1, 6, 7))
        self.assertEqual(source.crop((-1, -1, 2, 2)).size, (3, 3))
        self.assertEqual(source.transpose(Image.Transpose.FLIP_LEFT_RIGHT).getpixel((2, 1)), (20, 40, 80, 255))
        for resample in Image.Resampling:
            result = source.resize((3, 3), resample)
            self.assertEqual(result.size, (3, 3))
            for red, green, blue, alpha in result.getdata():
                if alpha:
                    self.assertLessEqual(abs(red - 20), 1)
                    self.assertLessEqual(abs(green - 40), 1)
                    self.assertLessEqual(abs(blue - 80), 1)
        canvas = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
        canvas.alpha_composite(Image.new("RGBA", (1, 1), (0, 0, 255, 128)), (1, 1))
        self.assertEqual(canvas.getpixel((1, 1)), (127, 0, 128, 255))
        self.assertEqual(source.split()[3].histogram()[255], 24)

    def test_malformed_pixel_stream_checksums_and_animation_limits(self):
        from standalone.raster_png import SIGNATURE, chunk
        header = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
        for data in (SIGNATURE + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\0\1\2")) + chunk(b"IEND", b""),
                     SIGNATURE + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\0" * 10000)) + chunk(b"IEND", b""),
                     b"GIF89a\x01\0\x01\0\0\0\0\x3b", b"BM" + b"\0" * 40):
            with self.subTest(data=data[:16]), self.assertRaises((ValueError, OSError)):
                Image.open(io.BytesIO(data))
        valid = next(case for case in json.loads(FIXTURES.read_text())["cases"] if case["name"] == "apng-animation")
        data = base64.b64decode(valid["data"])
        location = data.index(b"acTL")
        modified = data[:location - 4] + chunk(b"acTL", struct.pack(">II", 101, 0)) + data[location + 12:]
        with self.assertRaises(ValueError):
            Image.open(io.BytesIO(modified))
        corrupt = bytearray(data)
        corrupt[corrupt.index(b"IDAT") + 5] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            Image.open(io.BytesIO(corrupt))


WORKER = r'''
import base64, io, json, pathlib, sys
source, vendor, scratch = map(pathlib.Path, sys.argv[1:])
sys.path[:0] = [str(source), str(vendor)]
from standalone.runtime import initialize_bundle
initialize_bundle({"root": str(source.parent), "artifact": str(scratch / "fixture.py"),
                   "bundle_sha256": source.parent.name, "manifest": {"project": "hermes-standalone"}})
def reject_external(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system", "os.exec"):
        raise PermissionError("Raster fixture forbids external operations")
sys.addaudithook(reject_external)
from hermes_platform.images import Image
from tools.vision_tools_image_prep import _validate_raster_image_decodable, _crop_image_region, _normalize_to_supported_image
image = Image.new("RGB", (40, 30), (20, 40, 80))
path = scratch / "fixture.bmp"
image.save(path)
assert _validate_raster_image_decodable(path) is None
converted, mime, error = _normalize_to_supported_image(path, "image/bmp")
assert error is None and mime == "image/png", error
assert Image.open(converted).getpixel((0, 0)) == (20, 40, 80)
offset = {}
cropped, mime, error = _crop_image_region(converted, [-5, 2, 15, 22], offset)
assert error is None and Image.open(cropped).size == (15, 20), error
assert offset == {"x": 0, "y": 2, "width": 15, "height": 20}
from agent.pet.generate import atlas
strip = Image.new("RGBA", (20, 18), (255, 0, 255, 255))
strip.paste((20, 40, 80, 255), (4, 3, 16, 15))
keyed = atlas.remove_background(strip, chroma_key=(255, 0, 255))
assert keyed.getpixel((0, 0))[3] == 0 and keyed.getpixel((8, 8)) == (20, 40, 80, 255)
from agent.pet import render
sheet = scratch / "sheet.png"
keyed.save(sheet)
frames = render._raw_frames(str(sheet), "idle", 20, 18, 1)
assert len(frames) == 1
assert base64.b64decode(render._png_b64(frames[0])).startswith(b"\x89PNG")
assert render._encode_sixel(frames[0]).startswith("\x1bP")
assert not any(name == "PIL" or name.startswith("PIL.") for name in sys.modules)
print(json.dumps({"ok": True, "vision": True, "pet": True}))
'''


@unittest.skipIf(sys.version_info < (3, 14), "Original Hermes targets Python 3.14")
class OriginalRasterTests(unittest.TestCase):
    def test_original_vision_crop_normalize_and_pet_renderer_without_pillow(self):
        from standalone.offline_agent_probe import _source_snapshot, _vendor_sources
        from standalone.agent_probe_worker import fixture_environment
        with tempfile.TemporaryDirectory(prefix="hermes-original-raster-") as directory:
            scratch = Path(directory)
            for name in ("app", "vendor", "home", "work", "tmp"):
                (scratch / name).mkdir()
            _vendor_sources(PROJECT, PROJECT / "standalone/.cache/downloads", scratch / "vendor")
            _, digest = _source_snapshot(PROJECT, scratch / "app")
            bundle = scratch / digest
            bundle.mkdir()
            (scratch / "app").rename(bundle / "app")
            (scratch / "vendor").rename(bundle / "vendor")
            result = subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", WORKER,
                                     str(bundle / "app"), str(bundle / "vendor"), str(scratch)],
                                    env=fixture_environment(scratch), cwd=scratch / "work",
                                    capture_output=True, text=True, encoding="utf-8", timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout)["ok"])
