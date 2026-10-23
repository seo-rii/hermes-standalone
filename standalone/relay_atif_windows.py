"""Handle-relative Windows ATIF atomic replacement through qualified OS APIs.

New children and temporary files receive the existing owner/SYSTEM protected
DACL. Every directory handle stays open without delete sharing until commit.
No descendant path is resolved into a new trusted anchor. Runtime Windows
proof is separate from the portable ABI and policy tests.

Native contracts:
https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntcreatefile
https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntsetinformationfile
https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_rename_information
"""
from contextlib import ExitStack
import ctypes as C
from functools import lru_cache
import os
import uuid

from standalone.relay_files_windows import (
    DWORD, HANDLE, IO_STATUS_BLOCK, WindowsRelayFiles, _object,
)
from standalone.windows_files import _normalized_path


FILE_OPEN, FILE_CREATE, FILE_OPEN_IF = 1, 2, 3
FILE_READ_ATTRIBUTES, FILE_TRAVERSE = 0x80, 0x20
DELETE, READ_CONTROL, WRITE_DAC, SYNCHRONIZE = 0x10000, 0x20000, 0x40000, 0x100000
FILE_DIRECTORY_FILE, FILE_NON_DIRECTORY_FILE = 1, 0x40
FILE_SYNCHRONOUS_IO_NONALERT, FILE_OPEN_REPARSE_POINT = 0x20, 0x00200000
SHARE_READ_WRITE, SHARE_ALL = 3, 7
FILE_RENAME_INFORMATION_CLASS, FILE_DISPOSITION_INFORMATION_CLASS = 10, 13
DIRECTORY_ACCESS = READ_CONTROL | WRITE_DAC | SYNCHRONIZE | FILE_READ_ATTRIBUTES | FILE_TRAVERSE | 1 | 2 | 4
TEMP_ACCESS = DELETE | READ_CONTROL | SYNCHRONIZE | FILE_READ_ATTRIBUTES | 2 | 0x10 | 0x100


class FILE_RENAME_INFORMATION(C.Structure):
    _fields_ = [('ReplaceIfExists', C.c_ubyte), ('RootDirectory', HANDLE),
                ('FileNameLength', DWORD), ('FileName', C.c_uint16 * 1)]


class FILE_DISPOSITION_INFORMATION(C.Structure):
    _fields_ = [('DeleteFile', C.c_ubyte)]


def _component(name):
    if not isinstance(name, str) or not name or name in {'.', '..'} or any(value in name for value in ('\0', '/', '\\', ':')):
        raise OSError('ATIF Windows output requires a single path component without alternate streams')
    encoded = name.encode('utf-16-le', 'surrogatepass')
    if len(encoded) > 65534:
        raise OSError('ATIF Windows output component exceeds the native name limit')
    return encoded


def _rename_information(parent, filename):
    encoded = _component(filename)
    # Keep the documented sizeof(struct)+name extent, including alignment.
    size = C.sizeof(FILE_RENAME_INFORMATION) + len(encoded)
    buffer = C.create_string_buffer(size)
    information = FILE_RENAME_INFORMATION.from_buffer(buffer)
    information.ReplaceIfExists = True
    information.RootDirectory = parent
    information.FileNameLength = len(encoded)
    C.memmove(C.addressof(buffer) + FILE_RENAME_INFORMATION.FileName.offset, encoded, len(encoded))
    return buffer


class WindowsAtifAtomic:
    def __init__(self):
        if os.name != 'nt':
            raise OSError('ATIF Windows atomic writer requires a Windows host')
        self.relay = WindowsRelayFiles()
        self.private = self.relay.private
        self.set_information = self.relay.ntdll.NtSetInformationFile
        self.set_information.restype = C.c_int32
        self.set_information.argtypes = [HANDLE, C.POINTER(IO_STATUS_BLOCK), C.c_void_p, DWORD, DWORD]
        self.file_type = self.private.kernel.GetFileType
        self.file_type.restype, self.file_type.argtypes = DWORD, [HANDLE]

    def _status_error(self, status, operation):
        code = int(self.relay.status_error(status))
        error = OSError('ATIF ' + operation + ' failed (Windows error ' + str(code) + ')')
        error.winerror = code
        return error

    def _close(self, handle):
        if not self.private.close(handle):
            raise self.private._error('ATIF CloseHandle')

    def _validate(self, handle, directory):
        self.relay.validate(handle, directory)
        if self.file_type(handle) != 1:
            raise OSError('ATIF output handle must be a regular disk file or directory')

    def _anchor(self, anchor):
        expected = _normalized_path(anchor)
        access = DIRECTORY_ACCESS & ~WRITE_DAC
        handle = self.private.create_file(os.fspath(anchor), access, SHARE_READ_WRITE, None, 3,
            0x02000000 | 0x00200000, None)
        if handle in (None, C.c_void_p(-1).value):
            raise self.private._error('ATIF CreateFileW anchor')
        try:
            self._validate(handle, True)
            length = self.private.final_path(handle, None, 0, 0)
            if not 1 <= length <= 32768:
                raise self.private._error('ATIF GetFinalPathNameByHandleW size')
            buffer = C.create_unicode_buffer(length + 1)
            written = self.private.final_path(handle, buffer, len(buffer), 0)
            if not 0 < written < len(buffer) or _normalized_path(buffer.value) != expected:
                raise OSError('ATIF anchor handle escaped its expected path')
            return handle
        except BaseException:
            self._close(handle)
            raise

    def _relative(self, parent, name, directory, disposition):
        _component(name)
        descriptor = self.private._descriptor()
        try:
            attributes, unicode, buffer = _object(parent, name, descriptor)
            handle, status = HANDLE(), IO_STATUS_BLOCK()
            options = FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT
            options |= FILE_DIRECTORY_FILE if directory else FILE_NON_DIRECTORY_FILE
            result = self.relay.create(C.byref(handle), DIRECTORY_ACCESS if directory else TEMP_ACCESS,
                C.byref(attributes), C.byref(status), None, 0x10 if directory else 0x80,
                SHARE_READ_WRITE if directory else 0, disposition, options, None, 0)
            if result < 0:
                raise self._status_error(result, 'NtCreateFile')
            try:
                self._validate(handle, directory)
                if directory and not self.relay.restrict(handle, 0x80000004, descriptor):
                    raise self.private._error('ATIF SetKernelObjectSecurity')
                self.private.check_handle(handle)
                return handle.value
            except BaseException:
                try:
                    if not directory and disposition == FILE_CREATE:
                        self._discard(handle)
                finally:
                    self._close(handle)
                raise
        finally:
            self.private.free(descriptor)

    def _reject_target(self, parent, filename):
        attributes, unicode, buffer = _object(parent, filename, None)
        handle, status = HANDLE(), IO_STATUS_BLOCK()
        result = self.relay.create(C.byref(handle), SYNCHRONIZE | FILE_READ_ATTRIBUTES,
            C.byref(attributes), C.byref(status), None, 0, SHARE_ALL, FILE_OPEN,
            FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT, None, 0)
        if result < 0:
            code = int(self.relay.status_error(result))
            if code == 2:  # Only a missing final name is a normal absent target.
                return
            raise self._status_error(result, 'NtCreateFile target inspection')
        try:
            self._validate(handle, False)
        finally:
            self._close(handle)

    def _write(self, handle, payload):
        buffer = C.create_string_buffer(payload)
        position = 0
        while position < len(payload):
            count = DWORD()
            size = min(len(payload) - position, 65536)
            if not self.private.write(handle, C.byref(buffer, position), size, C.byref(count), None):
                raise self.private._error('ATIF WriteFile')
            if not 0 < count.value <= size:
                raise OSError('ATIF WriteFile did not advance inside its buffer')
            position += count.value
        if not self.private.flush(handle):
            raise self.private._error('ATIF FlushFileBuffers')

    def _rename(self, handle, parent, filename):
        information = _rename_information(parent, filename)
        status = IO_STATUS_BLOCK()
        result = self.set_information(handle, C.byref(status), information, len(information), FILE_RENAME_INFORMATION_CLASS)
        if result < 0:
            raise self._status_error(result, 'NtSetInformationFile rename')

    def _discard(self, handle):
        information, status = FILE_DISPOSITION_INFORMATION(True), IO_STATUS_BLOCK()
        result = self.set_information(handle, C.byref(status), C.byref(information), C.sizeof(information), FILE_DISPOSITION_INFORMATION_CLASS)
        if result < 0:
            raise self._status_error(result, 'NtSetInformationFile temporary cleanup')

    def write(self, anchor, missing_components, filename, payload):
        if not isinstance(payload, bytes):
            raise TypeError('ATIF atomic payload must be bytes')
        _component(filename)
        components = tuple(missing_components)
        for component in components:
            _component(component)
        with ExitStack() as resources:
            parent = self._anchor(anchor)
            resources.callback(self._close, parent)
            for component in components:
                parent = self._relative(parent, component, True, FILE_OPEN_IF)
                resources.callback(self._close, parent)
            self._reject_target(parent, filename)
            for attempt in range(16):
                temporary = '.' + filename + '.' + uuid.uuid4().hex + '.tmp'
                try:
                    handle = self._relative(parent, temporary, False, FILE_CREATE)
                except OSError as error:
                    if getattr(error, 'winerror', None) in (80, 183):
                        continue
                    raise
                resources.callback(self._close, handle)
                try:
                    self._write(handle, payload)
                    self._reject_target(parent, filename)
                    self._rename(handle, parent, filename)
                except BaseException:
                    self._discard(handle)
                    raise
                return
            raise OSError('ATIF could not allocate a private temporary file')


@lru_cache(maxsize=1)
def _api():
    return WindowsAtifAtomic()


def atomic_write(anchor, missing_components, filename, payload):
    return _api().write(anchor, missing_components, filename, payload)
