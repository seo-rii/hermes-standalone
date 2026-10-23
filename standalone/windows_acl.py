"""Private Windows cache directories using only documented operating-system APIs.

The builder also embeds this source in the bootstrap, before any cache code runs.
GetNamedSecurityInfoW owns the returned descriptor until LocalFree; token SID
bytes and ACL bytes are copied while their owning buffers are still alive.
https://learn.microsoft.com/windows/win32/api/aclapi/nf-aclapi-getnamedsecurityinfow
https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-createdirectoryw
"""

import ctypes as C
import os
import struct


def _sid_length(data, offset=0):
    if offset < 0 or offset + 8 > len(data) or data[offset] != 1:
        raise OSError("Invalid Windows security identifier")
    count = data[offset + 1]
    if count > 15 or offset + 8 + 4 * count > len(data):
        raise OSError("Truncated Windows security identifier")
    return 8 + 4 * count


def _check_acl(data, allowed):
    """Fail closed on unfamiliar grants, including inheritable-only grants."""
    if len(data) < 8:
        raise OSError("Truncated Windows cache DACL")
    revision, reserved, size, count, reserved2 = struct.unpack_from("<BBHHH", data)
    if revision not in (2, 4) or reserved or reserved2 or size != len(data):
        raise OSError("Invalid Windows cache DACL")
    offset = 8
    for _ in range(count):
        if offset + 4 > size:
            raise OSError("Truncated Windows cache ACE")
        kind, flags, length = struct.unpack_from("<BBH", data, offset)
        if length < 4 or length % 4 or offset + length > size:
            raise OSError("Invalid Windows cache ACE length")
        if kind == 0:  # ACCESS_ALLOWED_ACE; other grant forms are unsupported.
            entry = data[offset:offset + length]
            sid_length = _sid_length(entry, 8)
            if 8 + sid_length != length or entry[8:] not in allowed:
                raise OSError("Windows cache DACL grants access to another account")
        elif kind != 1:  # ACCESS_DENIED_ACE grants no access.
            raise OSError("Windows cache DACL contains an unsupported ACE")
        offset += length
    # ACLs may reserve unused space after their final ACE.
    if offset > size:
        raise OSError("Invalid Windows cache DACL extent")


class WindowsCacheACL:
    def __init__(self):
        if os.name != "nt":
            raise OSError("Windows cache ACL checks require Windows")
        self.kernel = C.WinDLL("kernel32", use_last_error=True)
        self.security = C.WinDLL("advapi32", use_last_error=True)
        pointer, dword, boolean = C.c_void_p, C.c_uint32, C.c_int32

        def bind(library, name, result, *arguments):
            function = getattr(library, name)
            function.restype, function.argtypes = result, list(arguments)
            return function

        self.current_process = bind(self.kernel, "GetCurrentProcess", pointer)
        self.close = bind(self.kernel, "CloseHandle", boolean, pointer)
        self.free = bind(self.kernel, "LocalFree", pointer, pointer)
        self.create_directory = bind(self.kernel, "CreateDirectoryW", boolean, C.c_wchar_p, pointer)
        self.open_token = bind(self.security, "OpenProcessToken", boolean, pointer, dword, C.POINTER(pointer))
        self.token_information = bind(self.security, "GetTokenInformation", boolean,
                                      pointer, C.c_int32, pointer, dword, C.POINTER(dword))
        self.sid_valid = bind(self.security, "IsValidSid", boolean, pointer)
        self.sid_length = bind(self.security, "GetLengthSid", dword, pointer)
        self.sid_string = bind(self.security, "ConvertSidToStringSidW", boolean, pointer, C.POINTER(pointer))
        self.descriptor_from_string = bind(self.security, "ConvertStringSecurityDescriptorToSecurityDescriptorW",
                                           boolean, C.c_wchar_p, dword, C.POINTER(pointer), C.POINTER(dword))
        arguments = (dword, dword, C.POINTER(pointer), C.POINTER(pointer),
                     C.POINTER(pointer), C.POINTER(pointer), C.POINTER(pointer))
        self.named_info = bind(self.security, "GetNamedSecurityInfoW", dword, C.c_wchar_p, *arguments)
        self.handle_info = bind(self.security, "GetSecurityInfo", dword, pointer, *arguments)
        self.user_sid, self.user_string = self._current_user()
        self.allowed = {self.user_sid,
                        b"\x01\x01\x00\x00\x00\x00\x00\x05\x12\x00\x00\x00",  # SYSTEM
                        b"\x01\x02\x00\x00\x00\x00\x00\x05\x20\x00\x00\x00\x20\x02\x00\x00"}  # Administrators

    @staticmethod
    def _error(operation):
        return OSError(C.get_last_error(), "Windows cache security: " + operation)

    def _copy_sid(self, sid):
        if not sid or not self.sid_valid(sid):
            raise OSError("Windows cache security returned an invalid SID")
        length = self.sid_length(sid)
        if not 8 <= length <= 68:
            raise OSError("Windows cache security returned an oversized SID")
        data = C.string_at(sid, length)
        if _sid_length(data) != len(data):
            raise OSError("Windows cache security returned a malformed SID")
        return data

    def _current_user(self):
        token = C.c_void_p()
        if not self.open_token(self.current_process(), 8, C.byref(token)):  # TOKEN_QUERY
            raise self._error("OpenProcessToken")
        try:
            needed = C.c_uint32()
            self.token_information(token, 1, None, 0, C.byref(needed))  # TokenUser
            if not C.sizeof(C.c_void_p) <= needed.value <= 65536:
                raise self._error("GetTokenInformation size")
            buffer = C.create_string_buffer(needed.value)
            if not self.token_information(token, 1, buffer, needed.value, C.byref(needed)):
                raise self._error("GetTokenInformation")
            sid = C.c_void_p.from_buffer(buffer).value
            if not C.addressof(buffer) <= sid < C.addressof(buffer) + C.sizeof(buffer):
                raise OSError("Token SID is outside its response buffer")
            data = self._copy_sid(sid)
            text = C.c_void_p()
            if not self.sid_string(sid, C.byref(text)):
                raise self._error("ConvertSidToStringSidW")
            try:
                return data, C.wstring_at(text)
            finally:
                self.free(text)
        finally:
            self.close(token)

    def mkdir(self, path):
        """Create privately in one OS call, without an inherited-permission gap."""
        class Attributes(C.Structure):
            _fields_ = [("length", C.c_uint32), ("descriptor", C.c_void_p), ("inherit", C.c_int32)]

        descriptor = C.c_void_p()
        user = self.user_string
        sddl = f"O:{user}D:P(A;OICI;FA;;;{user})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"
        if not self.descriptor_from_string(sddl, 1, C.byref(descriptor), None):
            raise self._error("ConvertStringSecurityDescriptorToSecurityDescriptorW")
        try:
            attributes = Attributes(C.sizeof(Attributes), descriptor, False)
            if not self.create_directory(os.fspath(path), C.byref(attributes)):
                code = C.get_last_error()
                if code == 183:
                    raise FileExistsError(code, "Cache directory already exists", os.fspath(path))
                raise self._error("CreateDirectoryW")
        finally:
            self.free(descriptor)

    def check(self, path, descriptor=None):
        owner, dacl, security = C.c_void_p(), C.c_void_p(), C.c_void_p()
        if descriptor is None:
            result = self.named_info(os.fspath(path), 1, 5, C.byref(owner), None,
                                     C.byref(dacl), None, C.byref(security))
        else:
            import msvcrt
            result = self.handle_info(msvcrt.get_osfhandle(descriptor), 1, 5, C.byref(owner), None,
                                      C.byref(dacl), None, C.byref(security))
        try:
            if result:
                raise OSError(result, "Cannot inspect Windows cache owner and DACL", os.fspath(path))
            if self._copy_sid(owner) != self.user_sid:
                raise OSError("Windows cache must be owned by this user", os.fspath(path))
            if not dacl:
                raise OSError("Windows cache must have a non-null DACL", os.fspath(path))
            length = C.c_uint16.from_address(dacl.value + 2).value
            if length < 8:
                raise OSError("Invalid Windows cache DACL size")
            _check_acl(C.string_at(dacl, length), self.allowed)
        finally:
            if security:
                self.free(security)
