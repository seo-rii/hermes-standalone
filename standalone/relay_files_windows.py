"""Relay's handle-relative private file boundary on Windows.

NtCreateFile preserves the pinned native exporter's directory-handle and
reparse refusal rules. The existing qualified ACL implementation supplies the
current user's SID and the owner/SYSTEM protected descriptor. This module is
loaded only on Windows; Linux layout tests do not establish native OS parity.
"""
import ctypes as C
from functools import lru_cache
import os

from standalone.windows_files import BY_HANDLE_FILE_INFORMATION, _api as _private_api


HANDLE, DWORD = C.c_void_p, C.c_uint32


class UNICODE_STRING(C.Structure):
    _fields_ = [('Length', C.c_uint16), ('MaximumLength', C.c_uint16), ('Buffer', HANDLE)]


class OBJECT_ATTRIBUTES(C.Structure):
    _fields_ = [('Length', DWORD), ('RootDirectory', HANDLE), ('ObjectName', C.POINTER(UNICODE_STRING)),
                ('Attributes', DWORD), ('SecurityDescriptor', HANDLE), ('SecurityQualityOfService', HANDLE)]


class IO_STATUS_BLOCK(C.Structure):
    _fields_ = [('Status', HANDLE), ('Information', C.c_size_t)]


def _object(parent, name, descriptor):
    encoded = name.encode('utf-16-le', 'surrogatepass')
    if not encoded or len(encoded) > 65535 or '\0' in name or '/' in name or '\\' in name or name in {'.', '..'}:
        raise ValueError('Invalid Relay Windows path component')
    buffer = C.create_string_buffer(encoded)
    unicode = UNICODE_STRING(len(encoded), len(encoded), C.cast(buffer, HANDLE))
    attributes = OBJECT_ATTRIBUTES(C.sizeof(OBJECT_ATTRIBUTES), parent, C.pointer(unicode), 0x40, descriptor, None)
    return attributes, unicode, buffer


class WindowsRelayFiles:
    def __init__(self):
        self.private = _private_api()
        self.ntdll = C.WinDLL('ntdll', use_last_error=True)
        self.create = self.ntdll.NtCreateFile
        self.create.restype = C.c_int32
        self.create.argtypes = [C.POINTER(HANDLE), DWORD, C.POINTER(OBJECT_ATTRIBUTES), C.POINTER(IO_STATUS_BLOCK),
                                HANDLE, DWORD, DWORD, DWORD, DWORD, HANDLE, DWORD]
        self.status_error = self.ntdll.RtlNtStatusToDosError
        self.status_error.restype = DWORD
        self.status_error.argtypes = [C.c_int32]
        self.restrict = self.private.security.SetKernelObjectSecurity
        self.restrict.restype = C.c_int32
        self.restrict.argtypes = [HANDLE, DWORD, HANDLE]

    def validate(self, handle, directory):
        info = BY_HANDLE_FILE_INFORMATION()
        if not self.private.file_info(handle, C.byref(info)):
            raise C.WinError(C.get_last_error(), 'Relay GetFileInformationByHandle')
        if info.dwFileAttributes & 0x400 or bool(info.dwFileAttributes & 0x10) != directory:
            raise OSError('Relay output handle is a reparse point or has the wrong file type')

    def relative(self, parent, name, directory, append=False):
        descriptor = self.private._descriptor()
        try:
            attributes, unicode, buffer = _object(parent, name, descriptor)
            handle, status = HANDLE(), IO_STATUS_BLOCK()
            access = (0x00120089 | 2 | 4) if directory else ((4 | 0x80) if append else (0x00120116 | 0x80))
            result = self.create(C.byref(handle), access | 0x00140000, C.byref(attributes), C.byref(status),
                                 None, 0x10 if directory else 0x80, 7, 3 if directory or append else 5,
                                 (1 if directory else 0x40) | 0x00200020, None, 0)
            if result < 0:
                raise C.WinError(self.status_error(result), 'Relay NtCreateFile')
            try:
                self.validate(handle, directory)
                if not self.restrict(handle, 0x80000004, descriptor):
                    raise C.WinError(C.get_last_error(), 'Relay SetKernelObjectSecurity')
                self.private.check_handle(handle)
                return handle.value
            except BaseException:
                self.private.close(handle)
                raise
        finally:
            self.private.free(descriptor)

    def open(self, anchor, missing, filename, append):
        import msvcrt
        parent = self.private.create_file(os.fspath(anchor), 0x00120089 | 2 | 4, 7, None, 3, 0x02000000, None)
        if parent in (None, C.c_void_p(-1).value):
            raise C.WinError(C.get_last_error(), 'Relay CreateFileW anchor')
        try:
            self.validate(parent, True)
            for component in missing:
                child = self.relative(parent, component, True)
                self.private.close(parent)
                parent = child
            handle = self.relative(parent, filename, False, append)
            try:
                descriptor = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY | (os.O_APPEND if append else 0))
            except BaseException:
                self.private.close(handle)
                raise
            try:
                return os.fdopen(descriptor, 'wb')
            except BaseException:
                os.close(descriptor)
                raise
        finally:
            self.private.close(parent)


@lru_cache(maxsize=1)
def _api():
    return WindowsRelayFiles()


def open_private(anchor, missing, filename, append):
    return _api().open(anchor, missing, filename, append)
