"""Real ACL byte layouts and native Windows creation/privacy checks."""

import os
from pathlib import Path
import struct
import tempfile
import unittest

from standalone.windows_acl import WindowsCacheACL, _check_acl


USER = b"\x01\x02\x00\x00\x00\x00\x00\x05" + struct.pack("<II", 21, 42)
OTHER = b"\x01\x01\x00\x00\x00\x00\x00\x01" + struct.pack("<I", 0)


def acl(*entries, reserve=0):
    body = b"".join(struct.pack("<BBHI", kind, flags, 8 + len(sid), 0x1f01ff) + sid
                    for kind, flags, sid in entries)
    return struct.pack("<BBHHH", 2, 0, 8 + len(body) + reserve, len(entries), 0) + body + b"\0" * reserve


class ACLBufferTests(unittest.TestCase):
    def test_private_and_denied_aces_allow_only_the_expected_principal(self):
        _check_acl(acl((0, 3, USER), (1, 0, OTHER), reserve=16), {USER})
        for flags in (0, 1, 2, 8, 16):
            with self.assertRaises(OSError):
                _check_acl(acl((0, flags, OTHER)), {USER})
        for kind in (5, 9, 11, 255):
            with self.assertRaises(OSError):
                _check_acl(acl((kind, 0, USER)), {USER})

    def test_truncated_sid_ace_count_and_size_fail_closed(self):
        valid = acl((0, 0, USER))
        malformed = [valid[:n] for n in (0, 3, 7, 8, 15, len(valid) - 1)]
        for index, value in ((0, 3), (1, 1), (4, 2), (10, 0), (16, 2), (17, 16)):
            edited = bytearray(valid)
            edited[index] = value
            malformed.append(bytes(edited))
        for data in malformed:
            with self.subTest(data=data.hex()), self.assertRaises(OSError):
                _check_acl(data, {USER})


@unittest.skipUnless(os.name == "nt", "Requires an actual Windows owner and DACL")
class NativeWindowsACLTests(unittest.TestCase):
    def test_new_private_tree_and_open_handle_check_then_reject_everyone_grant(self):
        import ctypes as C
        api = WindowsCacheACL()
        with tempfile.TemporaryDirectory(prefix="hermes-acl-") as temporary:
            directory = Path(temporary) / "private"
            api.mkdir(directory)
            api.check(directory)
            target = directory / "fixture.txt"
            target.write_bytes(b"private fixture")
            with target.open("rb") as stream:
                api.check(target, stream.fileno())
            with self.assertRaises(FileExistsError):
                api.mkdir(directory)
            descriptor, dacl = C.c_void_p(), C.c_void_p()
            present, defaulted = C.c_int32(), C.c_int32()
            self.assertTrue(api.descriptor_from_string("D:P(A;;FA;;;WD)", 1, C.byref(descriptor), None))
            try:
                get_dacl = api.security.GetSecurityDescriptorDacl
                get_dacl.restype = C.c_int32
                get_dacl.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.POINTER(C.c_void_p), C.POINTER(C.c_int32)]
                self.assertTrue(get_dacl(descriptor, C.byref(present), C.byref(dacl), C.byref(defaulted)))
                set_info = api.security.SetNamedSecurityInfoW
                set_info.restype = C.c_uint32
                set_info.argtypes = [C.c_wchar_p, C.c_uint32, C.c_uint32] + [C.c_void_p] * 4
                self.assertEqual(set_info(str(target), 1, 0x80000004, None, None, dacl, None), 0)
                with self.assertRaises(OSError):
                    api.check(target)
            finally:
                api.free(descriptor)
