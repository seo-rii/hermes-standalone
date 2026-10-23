"""Private authenticated X server, bundled sandbox wire and minified worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import tempfile
import unittest

from hermes_platform.host.facts import os_family
from hermes_platform.resolver import locate_command
from standalone import x11_capture
from standalone.minify import minify_source
from standalone.tests.test_entrypoints import BundledSourceTestCase, support


_WORKER = r'''
import io,json,os,pathlib,subprocess,sys
context=json.loads(sys.argv[1]);scratch=pathlib.Path(sys.argv[2]);display=sys.argv[3]
source=pathlib.Path(context['root'])/'app'
sys.path[:0]=[str(source),str(source.parent/'vendor')]
from standalone.runtime import initialize_bundle
initialize_bundle(context)
from hermes_platform.images import Image,ImageGrab
assert ImageGrab.__name__=='standalone.raster_grab'
for screen,size in ((0,(80,40)),(1,(19,13))):
    image=ImageGrab.grab(xdisplay=display+'.'+str(screen))
    assert image.size==size and image.mode=='RGB'
    assert image.tobytes()==b'\xff'*(size[0]*size[1]*3)
assert ImageGrab.grab((7,3,27,13),xdisplay=display).size==(20,10)
# Each capture closes its X client. Keep the private server generation stable
# across that disconnect and the following authenticated setup handshake.
for _ in range(24):
    assert ImageGrab.grab((7,3,27,13),xdisplay=display).tobytes()==b'\xff'*(20*10*3)
try:ImageGrab.grab((-1,0,20,10),xdisplay=display)
except ValueError:pass
else:raise AssertionError('out-of-screen screenshot accepted')
script=ImageGrab.worker_script()
original=(scratch/'original-x11.py').read_bytes()
minified=(scratch/'minified-x11.py').read_bytes()
minification=json.loads((scratch/'minification.json').read_text())
assert len(minified)<=len(original)
if minification['status']=='unchanged':assert minified==original
for text in (script,original.decode(),minified.decode()):
    checked=text+"\nassert not any(n=='PIL' or n.startswith('PIL.') or n in ('ctypes','Xlib','xcb','xcffib') for n in sys.modules)\n"
    process=subprocess.run([sys.executable,'-I','-S','-B','-c',checked,display],capture_output=True,timeout=15)
    assert process.returncode==0,process.stderr.decode()
    image=ImageGrab.from_worker(process.stdout)
    assert image.size==(80,40) and image.tobytes()==b'\xff'*(80*40*3)
from tools.bot_desktop import sandbox_host
from tools.environments.singularity import SingularityEnvironment
# A local exec transport fixture exercises the original Apptainer argv/stdio
# path. The independent X server and screenshot worker are real processes.
transport=scratch/'exec-transport'
transport.write_text('#!/bin/sh\n[ "$1" = exec ] && [ "$2" = instance://fixture ] || exit 91\nshift 2\nexec "$@"\n')
transport.chmod(0o700)
environment=object.__new__(SingularityEnvironment)
environment.executable=str(transport);environment.instance_id='fixture';environment._instance_started=True
environment._bd_desktop_user=None
encoded=sandbox_host.grab_jpeg(environment,{'DISPLAY':display,'XAUTHORITY':os.environ['XAUTHORITY']},(20,20),85)
assert encoded and encoded.startswith(b'\xff\xd8')
decoded=Image.open(io.BytesIO(encoded))
assert decoded.size==(20,10) and decoded.format=='JPEG'
assert all(value>=250 for value in decoded.convert('RGB').tobytes())
assert not any(name=='PIL' or name.startswith('PIL.') for name in sys.modules)
assert not any(name in ('Xlib','xcb','xcffib') for name in sys.modules)
print(json.dumps({'screen_sizes':[[80,40],[19,13]],'jpeg_size':list(decoded.size),'verified_worker':True,
                  'minifier_status':minification['status']}))
'''


class CaptureWireTests(unittest.TestCase):
    def test_truncation_extra_pixels_checksums_and_nonfinite_sizes_are_refused(self):
        import zlib

        valid = b"HX11" + struct.pack(">II", 2, 1) + zlib.compress(b"\x20\x40\x60" * 2)
        self.assertEqual(x11_capture.decode_wire(valid), (2, 1, b"\x20\x40\x60" * 2))
        malformed = [valid[:n] for n in range(len(valid))]
        malformed += [valid + b"x", b"HX11" + struct.pack(">II", 100_000_001, 1) + zlib.compress(b""),
                      b"HX11" + struct.pack(">II", 1, 1) + zlib.compress(b"x" * 100000),
                      b"HX11" + struct.pack(">II", 0, 1) + zlib.compress(b"")]
        for data in malformed:
            with self.subTest(size=len(data)), self.assertRaises((ValueError, zlib.error)):
                x11_capture.decode_wire(data)

    def test_truecolor_masks_padding_and_both_image_byte_orders(self):
        masks = (0xFF0000, 0xFF00, 0xFF)
        self.assertEqual(x11_capture._pixels(b"\x60\x40\x20\0", 1, 1, 32, 32, 0, masks), b"\x20\x40\x60")
        self.assertEqual(x11_capture._pixels(b"\0\x20\x40\x60", 1, 1, 32, 32, 1, masks), b"\x20\x40\x60")
        self.assertEqual(x11_capture._pixels(b"\x20\x40\x60\0", 1, 1, 24, 32, 1, masks), b"\x20\x40\x60")
        self.assertEqual(x11_capture._pixels(b"\0\xf8\0\0", 1, 1, 16, 32, 0, (0xF800, 0x7E0, 31)), b"\xff\0\0")
        for masks in ((0, 0xFF00, 255), (0xFFFF, 0xFF, 0xFF00), (0xF50000, 0xFF00, 255)):
            with self.assertRaises(OSError):
                x11_capture._pixels(b"\0" * 4, 1, 1, 32, 32, 0, masks)


@unittest.skipUnless(os_family().startswith("linux"), "Private Xvfb proof runs on a Linux host")
class BundledCaptureTests(BundledSourceTestCase):
    def test_private_xvfb_scoped_auth_sandbox_host_and_minified_worker(self):
        resolution = locate_command("Xvfb")
        if not resolution.found:
            self.skipTest("Private X server is unavailable; no capture of an existing display is permitted")
        with tempfile.TemporaryDirectory(prefix="x11-", dir=self.directory) as temporary:
            scratch = Path(temporary)
            for name in ("home", "tmp", "work"):
                (scratch / name).mkdir()
            authority = scratch / "authority"
            fields = (b"", b"", b"MIT-MAGIC-COOKIE-1", os.urandom(16))
            authority.write_bytes(struct.pack(">H", 65535) + b"".join(struct.pack(">H", len(field)) + field for field in fields))
            authority.chmod(0o600)
            original = Path(x11_capture.__file__).read_bytes()
            (scratch / "original-x11.py").write_bytes(original)
            minified, report = minify_source(original, "standalone/x11_capture.py")
            self.assertEqual((minified, report), minify_source(original, "standalone/x11_capture.py"))
            (scratch / "minification.json").write_text(json.dumps(report))
            (scratch / "minified-x11.py").write_bytes(minified)
            environment = support._load_helper("agent_probe_worker").fixture_environment(scratch)
            environment["XAUTHORITY"] = str(authority)
            with (scratch / "xvfb.log").open("wb") as log:
                process = subprocess.Popen([*resolution.command, "-noreset", "-displayfd", "1", "-screen", "0", "80x40x24",
                                            "-screen", "1", "19x13x16", "-nolisten", "tcp", "-auth", str(authority), "-wr"],
                                           stdout=subprocess.PIPE, stderr=log, env=environment)
                try:
                    self.assertTrue(select.select([process.stdout], [], [], 10)[0], "Private Xvfb startup timed out")
                    number = process.stdout.readline().decode().strip()
                    self.assertTrue(number.isdecimal(), "Private Xvfb did not allocate its own display")
                    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", _WORKER,
                                             json.dumps(self.context), str(scratch), ":" + number],
                                            capture_output=True, text=True, env=environment, cwd=scratch / "work", timeout=45)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                    report = json.loads(result.stdout)
                    self.assertEqual(report["jpeg_size"], [20, 10])
                    self.assertTrue(report["verified_worker"])
                finally:
                    process.terminate()
                    process.wait(timeout=10)
                    process.stdout.close()


class NativeCaptureABITests(unittest.TestCase):
    def test_windows_fixed_width_dib_header_and_top_down_bgra_pixels(self):
        import ctypes
        from standalone import raster_capture_native as native

        self.assertEqual(ctypes.sizeof(native.WinRect), 16)
        self.assertEqual(struct.unpack("<IiiHHIIiiII", native.bitmap_info(2, 1)),
                         (40, 2, -1, 1, 32, 0, 8, 0, 0, 0, 0))
        self.assertEqual(native.dib_pixels(bytes((60, 40, 20, 0, 90, 80, 70, 255)), 2, 1), bytes((20, 40, 60, 70, 80, 90)))
        with self.assertRaises(ValueError):
            native.dib_pixels(b"\0" * 7, 2, 1)
        for size in ((0, 1), (1, -1), (40_000_001, 1)):
            with self.assertRaises(OSError):
                native.bitmap_info(*size)

    def test_coregraphics_64_bit_rect_byte_orders_padding_and_associated_alpha(self):
        import ctypes
        from standalone import raster_capture_native as native

        self.assertEqual(ctypes.sizeof(native.CGPoint), 16)
        self.assertEqual(ctypes.sizeof(native.CGSize), 16)
        self.assertEqual(ctypes.sizeof(native.CGRect), 32)
        self.assertEqual(native.CGRect.size.offset, 16)
        for raw, info in ((bytes((128, 10, 20, 30)), 0x4002), (bytes((30, 20, 10, 128)), 0x2002),
                          (bytes((10, 20, 30, 128)), 0x4001), (bytes((128, 30, 20, 10)), 0x2001),
                          (bytes((99, 20, 40, 60)), 0x4006), (bytes((20, 40, 60, 99)), 0x4005)):
            with self.subTest(info=info):
                self.assertEqual(native.coregraphics_pixels(raw, 1, 1, 4, 32, info), bytes((20, 40, 60)))
        self.assertEqual(native.coregraphics_pixels(bytes((20, 40, 60, 99)), 1, 1, 4, 24, 0), bytes((20, 40, 60)))
        self.assertEqual(native.coregraphics_pixels(b"\0" * 4, 1, 1, 4, 32, 0x2002), b"\0" * 3)
        for raw, stride, bits, info in ((b"\0" * 3, 4, 32, 0x2002), (b"\0" * 4, 4, 32, 0x2102),
                                        (b"\0" * 4, 4, 32, 0x1002), (b"\0" * 4, 4, 16, 0)):
            with self.assertRaises(ValueError):
                native.coregraphics_pixels(raw, 1, 1, stride, bits, info)


if __name__ == "__main__":
    unittest.main()
