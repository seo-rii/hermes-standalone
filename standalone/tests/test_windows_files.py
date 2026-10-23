"""Windows SSH ACL/ABI contracts and explicitly native retained-handle tests.

Portable cases parse real Windows security layouts and exercise bounded I/O
against a controlled OS-call boundary. They never spoof the running platform.
"""

import ctypes as C
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from standalone import windows_files as files
from standalone.windows_acl import WindowsCacheACL


USER = b"\x01\x02\x00\x00\x00\x00\x00\x05" + struct.pack("<II", 21, 42)
ADMIN = b"\x01\x02\x00\x00\x00\x00\x00\x05" + struct.pack("<II", 32, 544)
EVERYONE = b"\x01\x01\x00\x00\x00\x00\x00\x01" + struct.pack("<I", 0)


def _ace(sid, kind=0, *, flags=0, mask=0x1f01ff, object_flags=0):
    body = struct.pack("<I", mask)
    if kind in (5, 6, 11, 12):
        body += struct.pack("<I", object_flags) + b"G" * (16 * object_flags.bit_count())
    body += sid
    if kind in (9, 10, 11, 12):
        body += b"artx"  # Callback/application data follows, never replaces, the SID.
    return struct.pack("<BBH", kind, flags, 4 + len(body)) + body


def _acl(*entries):
    body = b"".join(entries)
    return struct.pack("<BBHHH", 4, 0, 8 + len(body), len(entries), 0) + body


class PortablePrivateFileTests(unittest.TestCase):
    def test_windows_layouts_and_acl_policy_exclude_cache_administrators(self):
        self.assertEqual((C.sizeof(files.DWORD), C.sizeof(files.BOOL), C.sizeof(files.FILETIME)), (4, 4, 8))
        self.assertEqual(C.sizeof(files.BY_HANDLE_FILE_INFORMATION), 52)
        self.assertEqual(C.sizeof(files.SECURITY_ATTRIBUTES), 24 if C.sizeof(C.c_void_p) == 8 else 12)
        allowed = {USER, files.SYSTEM_SID}
        for kind in (0, 5, 9, 11):
            for object_flags in ((0, 1, 2, 3) if kind in (5, 11) else (0,)):
                files._check_ssh_acl(_acl(_ace(USER, kind, object_flags=object_flags),
                                          _ace(files.SYSTEM_SID)), allowed)
            for sid in (ADMIN, EVERYONE):
                for flags in (0, 3, 8):
                    with self.subTest(kind=kind, flags=flags), self.assertRaises(OSError):
                        files._check_ssh_acl(_acl(_ace(sid, kind, flags=flags)), allowed)
        for kind in (1, 6, 10, 12):
            files._check_ssh_acl(_acl(_ace(EVERYONE, kind)), allowed)
        valid = _acl(_ace(USER))
        invalid = [valid[:index] for index in (0, 7, 8, 15, len(valid) - 1)]
        for index, value in ((0, 3), (1, 1), (4, 2), (8, 255), (10, 4), (16, 2), (17, 16)):
            changed = bytearray(valid)
            changed[index] = value
            invalid.append(bytes(changed))
        for data in invalid:
            with self.subTest(data=data.hex()), self.assertRaises(OSError):
                files._check_ssh_acl(data, allowed)

    def test_absolute_paths_normalize_only_dos_and_unc_forms(self):
        self.assertEqual(files._normalized_path(r"C:\Runtime\private.token"), r"c:\runtime\private.token")
        self.assertEqual(files._normalized_path(r"\\?\C:\Runtime\private.token"), r"c:\runtime\private.token")
        self.assertEqual(files._normalized_path(r"\\?\UNC\Host\Share\private.token"),
                         r"\\host\share\private.token")
        for path in ("", "C:relative.token", r"\relative.token", "relative.token", "C:\\bad\0token",
                     r"\\.\C:\private.token", r"\\?\GLOBALROOT\Device\HarddiskVolume1\private.token"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                files._normalized_path(path)
        if os.name != "nt":
            with self.assertRaisesRegex(OSError, "require Windows"):
                files.WindowsPrivateFiles()

    def test_read_write_complete_short_transfers_and_preserve_the_limit(self):
        payload = bytes(range(256)) * 400
        pending, sent, read_sizes, write_sizes = bytearray(payload), bytearray(), [], []

        def read(handle, buffer, size, count, overlap):
            read_sizes.append(size)
            content = bytes(pending[:min(size, 137)])
            del pending[:len(content)]
            C.memmove(buffer, content, len(content))
            count._obj.value = len(content)
            return True

        def write(handle, buffer, size, count, overlap):
            write_sizes.append(size)
            amount = min(size, 181)
            sent.extend(C.string_at(buffer, amount))
            count._obj.value = amount
            return True

        api = SimpleNamespace(read=read, write=write)
        with patch.object(files, "_api", return_value=api):
            self.assertEqual(files.ReadFile(123, 777)[1], payload[:777])
            self.assertEqual(files.ReadFile(123, len(payload))[1], payload[777:])
            self.assertEqual(files.ReadFile(123, 0), (0, b""))
            self.assertEqual(files.WriteFile(123, payload), (0, len(payload)))
        self.assertEqual(sent, payload)
        self.assertLessEqual(max(read_sizes + write_sizes), 65536)
        for value in (-1, 2**32, 1.5, True):
            with self.assertRaises(ValueError):
                files.ReadFile(123, value)
        with self.assertRaises(ValueError):
            files.WriteFile(123, bytearray(b"not immutable"))

    def test_zero_and_oversized_os_transfer_results_fail_closed(self):
        for operation, count_value in (("read", 65537), ("write", 0), ("write", 65537)):
            def transfer(handle, buffer, size, count, overlap):
                count._obj.value = count_value
                return True
            with patch.object(files, "_api", return_value=SimpleNamespace(**{operation: transfer})):
                with self.subTest(operation=operation, count=count_value), self.assertRaises(OSError):
                    if operation == "read":
                        files.ReadFile(123, 65536)
                    else:
                        files.WriteFile(123, b"x" * 65536)

    def test_final_handle_path_and_reparse_refusals_close_the_opened_handle(self):
        for actual, attributes, security_error in (
            (r"\\?\C:\elsewhere\secret.token", 0, False),
            (r"\\?\C:\runtime\secret.token", 0x400, False),
            (r"\\?\C:\runtime\secret.token", 0, True),
        ):
            api = object.__new__(files.WindowsPrivateFiles)
            closed, freed = [], []
            api._descriptor = lambda: C.c_void_p(41)
            api.free = lambda descriptor: freed.append(descriptor.value)
            api.create_file = lambda *args: 123
            api.close = closed.append
            def file_info(handle, info):
                info._obj.dwFileAttributes = attributes
                return True
            def final_path(handle, buffer, size, flags):
                if buffer is not None:
                    buffer.value = actual
                    return len(actual)
                return len(actual) + 1
            def check(handle):
                self.assertEqual(handle, 123)
                if security_error:
                    raise OSError("untrusted ACL")
            api.file_info, api.final_path, api.check_handle = file_info, final_path, check
            with self.subTest(actual=actual, attributes=attributes), self.assertRaises(OSError):
                api.open_private(r"C:\runtime\secret.token", 0, files.constants.OPEN_EXISTING, 0)
            self.assertEqual(closed, [123])
            self.assertEqual(freed, [41])


@unittest.skipUnless(os.name == "nt", "Requires real Windows file handles and protected DACLs")
class NativePrivateFileTests(unittest.TestCase):
    def test_create_full_io_atomic_move_and_delete_on_close(self):
        api = files._api()
        with tempfile.TemporaryDirectory(prefix="hermes-ssh-files-") as temporary:
            directory = Path(temporary) / "private"
            files.ensure_directory(directory)
            source, target = directory / "new.json", directory / "state.json"
            handle = files.open_private(source, files.constants.GENERIC_WRITE | files.constants.READ_CONTROL,
                                        files.constants.CREATE_NEW, files.constants.FILE_ATTRIBUTE_NORMAL)
            try:
                api.check_handle(handle)
                payload = bytes(range(256)) * 1000
                self.assertEqual(files.WriteFile(handle, payload), (0, len(payload)))
                files.FlushFileBuffers(handle)
            finally:
                files.CloseHandle(handle)
            files.MoveFileEx(source, target, 1 | 8)
            self.assertFalse(source.exists())
            handle = files.open_private(target, files.constants.GENERIC_READ | files.constants.READ_CONTROL
                                        | files.constants.DELETE, files.constants.OPEN_EXISTING, 0x04000000)
            try:
                self.assertEqual(files.ReadFile(handle, len(payload) + 1), (0, payload))
            finally:
                files.CloseHandle(handle)
            self.assertFalse(target.exists())

    def test_cache_administrators_grant_is_not_accepted_for_ssh_credentials(self):
        with tempfile.TemporaryDirectory(prefix="hermes-ssh-policy-") as temporary:
            directory = Path(temporary) / "cache-policy"
            WindowsCacheACL().mkdir(directory)
            with self.assertRaisesRegex(OSError, "permissive DACL"):
                files.open_private(directory, files.constants.GENERIC_READ | files.constants.READ_CONTROL,
                                   files.constants.OPEN_EXISTING, files.constants.FILE_FLAG_BACKUP_SEMANTICS)

    def test_final_path_rejects_a_redirecting_parent_directory(self):
        with tempfile.TemporaryDirectory(prefix="hermes-ssh-reparse-") as temporary:
            directory, alias = Path(temporary) / "private", Path(temporary) / "alias"
            files.ensure_directory(directory)
            target = directory / "fixture"
            handle = files.open_private(target, files.constants.GENERIC_WRITE | files.constants.READ_CONTROL,
                                        files.constants.CREATE_NEW, 0)
            files.CloseHandle(handle)
            try:
                alias.symlink_to(directory, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"This Windows account cannot create a test symlink: {error.winerror}")
            with self.assertRaisesRegex(OSError, "escaped its expected path"):
                files.open_private(alias / "fixture", files.constants.GENERIC_READ | files.constants.READ_CONTROL,
                                   files.constants.OPEN_EXISTING, 0)
