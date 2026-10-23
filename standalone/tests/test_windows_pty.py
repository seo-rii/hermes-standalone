"""Portable ConPTY wire/ABI contracts plus explicitly native Windows proofs.

ABI references: Microsoft STARTUPINFOEXW, PROCESS_INFORMATION and
JOBOBJECT_EXTENDED_LIMIT_INFORMATION. Host-independent tests never make the
interpreter believe it is Windows and never load a foreign operating-system DLL.
"""

import base64
import codecs
import ctypes as ct
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import unittest

from standalone import windows_pty as native


class PortableConPtyTests(unittest.TestCase):
    def test_fixed_windows_integer_widths_and_native_pointer_structures(self):
        self.assertEqual((ct.sizeof(native.DWORD), ct.sizeof(native.BOOL), ct.sizeof(native.WORD)), (4, 4, 2))
        self.assertEqual(ct.sizeof(native.COORD), 4)
        self.assertEqual(ct.sizeof(native.FILETIME), 8)
        self.assertEqual(ct.sizeof(native.IO_COUNTERS), 48)
        pointers = ct.sizeof(ct.c_void_p)
        self.assertEqual(native.PROCESS_INFORMATION.dwProcessId.offset, pointers * 2)
        self.assertEqual(ct.sizeof(native.PROCESS_INFORMATION), pointers * 2 + 8)
        if pointers == 8:
            self.assertEqual(native.STARTUPINFOW.lpReserved.offset, 8)
            self.assertEqual(native.STARTUPINFOW.hStdInput.offset, 80)
            self.assertEqual(ct.sizeof(native.STARTUPINFOW), 104)
            self.assertEqual(native.STARTUPINFOEXW.lpAttributeList.offset, 104)
            self.assertEqual(ct.sizeof(native.STARTUPINFOEXW), 112)
            self.assertEqual(native.JOBOBJECT_EXTENDED_LIMIT_INFORMATION.IoInfo.offset, 64)
            self.assertEqual(ct.sizeof(native.JOBOBJECT_EXTENDED_LIMIT_INFORMATION), 144)
        elif os.name == "nt":
            self.assertEqual(ct.sizeof(native.STARTUPINFOW), 68)
            self.assertEqual(ct.sizeof(native.STARTUPINFOEXW), 72)
            self.assertEqual(ct.sizeof(native.JOBOBJECT_EXTENDED_LIMIT_INFORMATION), 112)

    def test_crt_argument_quoting_preserves_spaces_quotes_and_trailing_backslashes(self):
        cases = [
            (["python.exe", ""], 'python.exe ""'),
            (["C:\\Program Files\\Python\\python.exe", "α β"], '"C:\\Program Files\\Python\\python.exe" "α β"'),
            (["python.exe", 'a"b'], 'python.exe a\\"b'),
            (["python.exe", "trailing space \\"], 'python.exe "trailing space \\\\"'),
            (["python.exe", "a&b", "$(inert)", "`inert`"], 'python.exe a&b $(inert) `inert`'),
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(native.command_line(arguments), expected)
        for invalid in (None, "python.exe", [], [""], ["python.exe", 3], ["py\0thon.exe"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                native.command_line(invalid)

    def test_environment_has_utf16_encoding_stable_order_and_double_termination(self):
        environment = {"z": "last", "연구": "🦀", "A": "a=b", "=C:": "C:\\private"}
        expected = "=C:=C:\\private\0A=a=b\0z=last\0연구=🦀\0\0".encode("utf-16-le")
        self.assertEqual(native.environment_block(environment), expected)
        self.assertEqual(native.environment_block({}), b"\0\0\0\0")
        for invalid in ({"": "x"}, {"a=b": "x"}, {"nul": "a\0b"}, {"x\0y": "value"},
                        {"x": None}, {"Path": "one", "PATH": "two"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                native.environment_block(invalid)

    def test_dimensions_preserve_console_axis_order_and_reject_out_of_abi_values(self):
        size = native._coord(24, 120)
        self.assertEqual((size.X, size.Y), (120, 24))
        for rows, columns in ((0, 80), (24, 32768), (True, 80), (24.0, 80), (24, -1)):
            with self.subTest(rows=rows, columns=columns), self.assertRaises(ValueError):
                native._coord(rows, columns)

    def test_dashboard_bridge_forwards_split_utf8_without_decoding_each_chunk(self):
        from hermes_cli.win_pty_bridge import WinPtyBridge

        text = "한글 🦀 split"
        encoded = text.encode("utf-8")

        class ByteConsole:
            supports_bytes = True
            pid = 1

            def __init__(self):
                self.incoming = iter(bytes([value]) for value in encoded)
                self.outgoing = []

            def read(self, size):
                try:
                    return next(self.incoming)
                except StopIteration:
                    raise EOFError from None

            def write(self, value):
                self.outgoing.append(value)

        console = ByteConsole()
        bridge = WinPtyBridge(console)
        decoder = codecs.getincrementaldecoder("utf-8")()
        pieces = []
        while (chunk := bridge.read()) is not None:
            self.assertIsInstance(chunk, bytes)
            pieces.append(decoder.decode(chunk))
            self.assertTrue(bridge._write_blocking(chunk))
        pieces.append(decoder.decode(b"", final=True))
        self.assertEqual("".join(pieces), text)
        self.assertEqual(b"".join(console.outgoing), encoded)

    def test_unavailable_host_refuses_before_loading_native_apis(self):
        if os.name == "nt":
            self.skipTest("This assertion applies to non-Windows hosts")
        self.assertFalse(native.PtyProcess.is_available())
        with self.assertRaises(OSError):
            native.PtyProcess.spawn(["fixture-no-native-executable"], env={})


@unittest.skipUnless(os.name == "nt", "Real Windows ConPTY/job execution; no emulated host evidence")
class NativeConPtyTests(unittest.TestCase):
    def setUp(self):
        if not native.PtyProcess.is_available():
            self.skipTest("This Windows host does not provide ConPTY")
        self.temporary = tempfile.TemporaryDirectory(prefix="hermes conpty 연구 ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.environment = {"HOME": str(self.directory), "USERPROFILE": str(self.directory),
                            "TEMP": str(self.directory), "TMP": str(self.directory), "PATH": "",
                            "PYTHONUTF8": "1", "SystemRoot": os.environ.get("SystemRoot", ""),
                            "HERMES_HOME": str(self.directory / "hermes"), "HERMES_FIXTURE": "연구 🦀"}

    def collect_until(self, process, marker, *, size=1, timeout=15):
        deadline, output = time.monotonic() + timeout, bytearray()
        while time.monotonic() < deadline:
            try:
                chunk = process.read(size)
            except EOFError:
                break
            output.extend(chunk)
            if marker in output:
                return bytes(output)
        self.fail(f"ConPTY did not produce {marker!r}: {bytes(output)!r}")

    def test_real_spawn_environment_arguments_resize_and_incremental_utf8(self):
        arguments = ["", "space value", 'quote"value', "trail\\", "한글 🦀"]
        script = (
            "import base64,json,os,sys; "
            "print('FACTS:' + base64.b64encode(json.dumps([sys.argv[1:],os.getcwd(),os.environ['HERMES_FIXTURE'],sys.stdin.isatty()]).encode()).decode(),flush=True); "
            "print('READY_NATIVE',flush=True); "
            "line=sys.stdin.readline().rstrip('\\r\\n'); "
            "print('ECHO:' + base64.b64encode(line.encode()).decode(),flush=True); "
            "print('DONE_한글🦀',flush=True)"
        )
        process = native.PtyProcess.spawn([sys.executable, "-I", "-S", "-u", "-c", script, *arguments],
                                          cwd=str(self.directory), env=self.environment, dimensions=(40, 2000))
        self.addCleanup(process.close)
        self.assertGreater(process.pid, 0)
        self.assertGreater(process.birth, 0)
        process.setwinsize(41, 1999)
        output = self.collect_until(process, b"READY_NATIVE", size=7)
        payload = re.search(rb"FACTS:([A-Za-z0-9+/=]+)", output)
        self.assertIsNotNone(payload, output)
        arguments_seen, cwd, value, tty = json.loads(base64.b64decode(payload.group(1)))
        self.assertEqual(arguments_seen, arguments)
        self.assertEqual(os.path.normcase(cwd), os.path.normcase(str(self.directory)))
        self.assertEqual(value, "연구 🦀")
        self.assertTrue(tty)
        text = "조각 🦀 UTF8"
        encoded = text.encode()
        for chunk in (encoded[:1], encoded[1:2], encoded[2:7], encoded[7:], b"\r"):
            self.assertEqual(process.write(chunk), len(chunk))
        output = self.collect_until(process, "DONE_한글🦀".encode(), size=1)
        self.assertIn(b"ECHO:" + base64.b64encode(encoded), output)
        decoder = codecs.getincrementaldecoder("utf-8")()
        decoded = "".join(decoder.decode(bytes([value])) for value in output) + decoder.decode(b"", final=True)
        self.assertIn("DONE_한글🦀", decoded)
        self.assertNotIn("\ufffd", decoded)
        self.assertEqual(process.wait(timeout=10), 0)
        process.close()
        self.assertFalse(process.isalive())

    def test_kernel_job_owns_descendant_after_original_parent_exits(self):
        script = (
            "import subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-I','-S','-c','import time;time.sleep(180)']); "
            "print('OWNED_DESCENDANT:%d' % child.pid,flush=True)"
        )
        process = native.PtyProcess.spawn([sys.executable, "-I", "-S", "-u", "-c", script],
                                          cwd=str(self.directory), env=self.environment)
        self.addCleanup(process.close)
        output = self.collect_until(process, b"OWNED_DESCENDANT:", size=4096)
        while re.search(rb"OWNED_DESCENDANT:(\d+)\r?\n", output) is None:
            output += self.collect_until(process, b"\n", size=4096)
        pid = int(re.search(rb"OWNED_DESCENDANT:(\d+)\r?\n", output).group(1))
        kernel = process._kernel
        kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [native.DWORD, native.BOOL, native.DWORD], native.HANDLE
        child = native._checked(kernel.OpenProcess(0x00100000 | 0x1000 | 0x1, False, pid))
        try:
            times = [native.FILETIME() for _ in range(4)]
            native._checked(kernel.GetProcessTimes(child, *(ct.byref(value) for value in times)))
            self.assertGreater((times[0].high << 32) | times[0].low, 0)
            self.assertEqual(kernel.WaitForSingleObject(process._process, 10000), 0)
            self.assertEqual(kernel.WaitForSingleObject(child, 0), 0x102)
            process.close()
            self.assertEqual(kernel.WaitForSingleObject(child, 10000), 0)
            self.assertFalse(process._reader.is_alive())
        finally:
            # The retained handle refers only to this fixture's child, even if
            # its PID has since been reused. No process-name or global cleanup.
            if kernel.WaitForSingleObject(child, 0) == 0x102:
                kernel.TerminateProcess(child, 1)
                kernel.WaitForSingleObject(child, 5000)
            kernel.CloseHandle(child)


if __name__ == "__main__":
    unittest.main()
