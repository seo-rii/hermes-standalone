"""Owner/SYSTEM-only Windows SSH files through documented Win32 APIs.

This is a qualified boundary for the original Desktop SSH runtime, not a global
replacement for pywin32. Unlike the bootstrap cache policy, SSH credentials do
not permit Administrators grants and may be owned by SYSTEM. Handles stay open
while their final path, reparse attribute, owner and DACL are checked.

https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-createfilew
https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-getfinalpathnamebyhandlew
https://learn.microsoft.com/windows/win32/api/aclapi/nf-aclapi-getsecurityinfo
"""

import ctypes as C
from functools import lru_cache
import ntpath
import os
from pathlib import Path
import struct
from types import SimpleNamespace

from standalone.windows_acl import WindowsCacheACL, _sid_length


DWORD, BOOL, HANDLE = C.c_uint32, C.c_int32, C.c_void_p
SYSTEM_SID = b"\x01\x01\x00\x00\x00\x00\x00\x05\x12\x00\x00\x00"
_IO_CHUNK = 64 * 1024

constants = SimpleNamespace(
    GENERIC_READ=0x80000000, GENERIC_WRITE=0x40000000, READ_CONTROL=0x00020000,
    DELETE=0x00010000, CREATE_NEW=1, OPEN_EXISTING=3,
    FILE_ATTRIBUTE_NORMAL=0x80, FILE_FLAG_BACKUP_SEMANTICS=0x02000000,
    FILE_SHARE_READ=1, FILE_SHARE_WRITE=2, FILE_SHARE_DELETE=4,
)


class SECURITY_ATTRIBUTES(C.Structure):
    _fields_ = [("nLength", DWORD), ("lpSecurityDescriptor", HANDLE), ("bInheritHandle", BOOL)]


class FILETIME(C.Structure):
    _fields_ = [("low", DWORD), ("high", DWORD)]


class BY_HANDLE_FILE_INFORMATION(C.Structure):
    _fields_ = [
        ("dwFileAttributes", DWORD), ("ftCreationTime", FILETIME),
        ("ftLastAccessTime", FILETIME), ("ftLastWriteTime", FILETIME),
        ("dwVolumeSerialNumber", DWORD), ("nFileSizeHigh", DWORD),
        ("nFileSizeLow", DWORD), ("nNumberOfLinks", DWORD),
        ("nFileIndexHigh", DWORD), ("nFileIndexLow", DWORD),
    ]


def _check_ssh_acl(data, allowed):
    """Check every grant form the upstream SSH boundary accepts.

    Object ACEs have optional GUIDs before the SID. Callback ACEs may contain an
    application condition after it. Conditions cannot broaden the SID's grant.
    Unknown ACEs fail closed; deny entries do not grant access to their SID.
    """
    if len(data) < 8:
        raise OSError("Truncated Windows SSH DACL")
    revision, reserved, size, count, reserved2 = struct.unpack_from("<BBHHH", data)
    if revision not in (2, 4) or reserved or reserved2 or size != len(data):
        raise OSError("Invalid Windows SSH DACL")
    offset = 8
    for _ in range(count):
        if offset + 8 > size:
            raise OSError("Truncated Windows SSH ACE")
        kind, flags, length, mask = struct.unpack_from("<BBHI", data, offset)
        if length < 8 or length % 4 or offset + length > size:
            raise OSError("Invalid Windows SSH ACE length")
        entry = data[offset:offset + length]
        if kind not in (0, 1, 5, 6, 9, 10, 11, 12):
            raise OSError("Unsupported Windows SSH ACE")
        sid_offset = 8
        if kind in (5, 6, 11, 12):
            if length < 12:
                raise OSError("Truncated Windows SSH object ACE")
            object_flags = struct.unpack_from("<I", entry, 8)[0]
            if object_flags & ~3:
                raise OSError("Invalid Windows SSH object ACE flags")
            sid_offset = 12 + 16 * bool(object_flags & 1) + 16 * bool(object_flags & 2)
        sid_length = _sid_length(entry, sid_offset)
        sid = entry[sid_offset:sid_offset + sid_length]
        if kind not in (9, 10, 11, 12) and sid_offset + sid_length != length:
            raise OSError("Invalid Windows SSH ACE extent")
        if kind in (0, 5, 9, 11) and mask and sid not in allowed:
            raise OSError("Windows SSH runtime object has a permissive DACL")
        offset += length


def _normalized_path(value):
    value = os.fspath(value)
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("Windows SSH path must be an absolute Unicode path")
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    drive, tail = ntpath.splitdrive(value)
    if not drive or not tail.startswith(("\\", "/")) or drive.startswith("\\\\."):
        raise ValueError("Windows SSH path must be absolute")
    return ntpath.normcase(ntpath.normpath(value))


class WindowsPrivateFiles(WindowsCacheACL):
    def __init__(self):
        super().__init__()
        # Do not inherit the broader cache allowlist or its owner-only rule.
        self.allowed = {self.user_sid, SYSTEM_SID}

        def bind(name, result, *arguments):
            function = getattr(self.kernel, name)
            function.restype, function.argtypes = result, list(arguments)
            return function

        pointer = C.c_void_p
        self.create_file = bind("CreateFileW", HANDLE, C.c_wchar_p, DWORD, DWORD,
                                C.POINTER(SECURITY_ATTRIBUTES), DWORD, DWORD, HANDLE)
        self.final_path = bind("GetFinalPathNameByHandleW", DWORD, HANDLE, C.c_wchar_p, DWORD, DWORD)
        self.file_info = bind("GetFileInformationByHandle", BOOL, HANDLE, C.POINTER(BY_HANDLE_FILE_INFORMATION))
        self.read = bind("ReadFile", BOOL, HANDLE, pointer, DWORD, C.POINTER(DWORD), pointer)
        self.write = bind("WriteFile", BOOL, HANDLE, pointer, DWORD, C.POINTER(DWORD), pointer)
        self.flush = bind("FlushFileBuffers", BOOL, HANDLE)
        self.move = bind("MoveFileExW", BOOL, C.c_wchar_p, C.c_wchar_p, DWORD)
        self.descriptor_control = self.security.GetSecurityDescriptorControl
        self.descriptor_control.restype = BOOL
        self.descriptor_control.argtypes = [HANDLE, C.POINTER(C.c_uint16), C.POINTER(DWORD)]

    @staticmethod
    def _error(operation, code=None):
        return C.WinError(C.get_last_error() if code is None else code, "Windows SSH runtime: " + operation)

    def _descriptor(self):
        descriptor = HANDLE()
        user = self.user_string
        if not self.descriptor_from_string(f"O:{user}D:P(A;;FA;;;{user})(A;;FA;;;SY)",
                                           1, C.byref(descriptor), None):
            raise self._error("ConvertStringSecurityDescriptorToSecurityDescriptorW")
        return descriptor

    def check_handle(self, handle):
        owner, dacl, descriptor = HANDLE(), HANDLE(), HANDLE()
        result = self.handle_info(handle, 1, 5, C.byref(owner), None,
                                  C.byref(dacl), None, C.byref(descriptor))
        try:
            if result:
                raise self._error("GetSecurityInfo", result)
            if self._copy_sid(owner) not in self.allowed:
                raise OSError("Windows SSH runtime object has the wrong owner")
            if not dacl:
                raise OSError("Windows SSH runtime object has a null DACL")
            control, revision = C.c_uint16(), DWORD()
            if not self.descriptor_control(descriptor, C.byref(control), C.byref(revision)):
                raise self._error("GetSecurityDescriptorControl")
            if not control.value & 0x1000:  # SE_DACL_PROTECTED
                raise OSError("Windows SSH runtime object has an unprotected DACL")
            length = C.c_uint16.from_address(dacl.value + 2).value
            if length < 8:
                raise OSError("Invalid Windows SSH DACL size")
            _check_ssh_acl(C.string_at(dacl, length), self.allowed)
        finally:
            if descriptor:
                self.free(descriptor)

    def open_private(self, path, access, creation, flags, share=0):
        expected = _normalized_path(path)
        descriptor = self._descriptor()
        try:
            attributes = SECURITY_ATTRIBUTES(C.sizeof(SECURITY_ATTRIBUTES), descriptor, False)
            handle = self.create_file(os.fspath(path), access, share, C.byref(attributes),
                                      creation, flags | 0x00200000, None)
            if handle in (None, C.c_void_p(-1).value):
                raise self._error("CreateFileW")
        finally:
            self.free(descriptor)
        try:
            info = BY_HANDLE_FILE_INFORMATION()
            if not self.file_info(handle, C.byref(info)):
                raise self._error("GetFileInformationByHandle")
            if info.dwFileAttributes & 0x400:
                raise OSError("Windows SSH runtime path contains a reparse point")
            needed = self.final_path(handle, None, 0, 0)
            if not 1 <= needed <= 32768:
                raise self._error("GetFinalPathNameByHandleW size")
            buffer = C.create_unicode_buffer(needed + 1)
            written = self.final_path(handle, buffer, len(buffer), 0)
            if not 0 < written < len(buffer):
                raise self._error("GetFinalPathNameByHandleW")
            if _normalized_path(buffer.value) != expected:
                raise OSError("Windows SSH runtime handle escaped its expected path")
            self.check_handle(handle)
            return handle
        except BaseException:
            self.close(handle)
            raise

    def ensure_directory(self, path):
        path = Path(path)
        if path.parent not in (Path(path.anchor), path) and not path.parent.exists():
            self.ensure_directory(path.parent)
        if not path.exists():
            descriptor = self._descriptor()
            try:
                attributes = SECURITY_ATTRIBUTES(C.sizeof(SECURITY_ATTRIBUTES), descriptor, False)
                if not self.create_directory(str(path), C.byref(attributes)):
                    code = C.get_last_error()
                    if code != 183:
                        raise self._error("CreateDirectoryW", code)
            finally:
                self.free(descriptor)
        handle = self.open_private(path, constants.GENERIC_READ | constants.READ_CONTROL,
                                   constants.OPEN_EXISTING, constants.FILE_FLAG_BACKUP_SEMANTICS,
                                   constants.FILE_SHARE_READ | constants.FILE_SHARE_WRITE | constants.FILE_SHARE_DELETE)
        self.close(handle)


@lru_cache(maxsize=1)
def _api():
    return WindowsPrivateFiles()


def open_private(path, access, creation, flags, share=0):
    return _api().open_private(path, access, creation, flags, share)


def ensure_directory(path):
    _api().ensure_directory(path)


def ReadFile(handle, limit):
    if type(limit) is not int or not 0 <= limit <= 0xFFFFFFFF:
        raise ValueError("Invalid Windows SSH read limit")
    api, chunks, remaining = _api(), [], limit
    while remaining:
        size = min(remaining, _IO_CHUNK)
        buffer, count = C.create_string_buffer(size), DWORD()
        if not api.read(handle, buffer, size, C.byref(count), None):
            raise api._error("ReadFile")
        if count.value > size:
            raise OSError("Windows SSH read exceeded its buffer")
        if count.value == 0:
            break
        chunks.append(buffer.raw[:count.value])
        remaining -= count.value
    return 0, b"".join(chunks)


def WriteFile(handle, data):
    if not isinstance(data, bytes) or len(data) > 0xFFFFFFFF:
        raise ValueError("Invalid Windows SSH write payload")
    api, written = _api(), 0
    buffer = C.create_string_buffer(data)
    while written < len(data):
        count = DWORD()
        size = min(len(data) - written, _IO_CHUNK)
        if not api.write(handle, C.byref(buffer, written), size, C.byref(count), None):
            raise api._error("WriteFile")
        if not 0 < count.value <= size:
            raise OSError("Windows SSH write did not advance within its buffer")
        written += count.value
    return 0, written


def FlushFileBuffers(handle):
    api = _api()
    if not api.flush(handle):
        raise api._error("FlushFileBuffers")


def CloseHandle(handle):
    api = _api()
    if not api.close(handle):
        raise api._error("CloseHandle")


def MoveFileEx(source, destination, flags):
    _normalized_path(source)
    _normalized_path(destination)
    api = _api()
    if not api.move(os.fspath(source), os.fspath(destination), flags):
        raise api._error("MoveFileExW")
