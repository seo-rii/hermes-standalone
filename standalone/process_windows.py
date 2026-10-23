"""Windows process ownership via kernel handles, Toolhelp and token SIDs.

Only operating-system DLLs are loaded. No executable or third-party DLL is
embedded. An opened handle's creation FILETIME is checked before mutation.
"""

from collections import namedtuple
from contextlib import contextmanager
import ctypes as C
from ctypes import wintypes as W
import os
import signal
import time

from standalone.process_errors import AccessDenied, NoSuchProcess

_kernel = C.WinDLL("kernel32", use_last_error=True)
_advapi = C.WinDLL("advapi32", use_last_error=True)
_ntdll = C.WinDLL("ntdll", use_last_error=True)


def _signature(library, name, restype, *argtypes):
    function = getattr(library, name)
    function.restype, function.argtypes = restype, list(argtypes)
    return function


_open = _signature(_kernel, "OpenProcess", W.HANDLE, W.DWORD, W.BOOL, W.DWORD)
_close = _signature(_kernel, "CloseHandle", W.BOOL, W.HANDLE)
_times = _signature(_kernel, "GetProcessTimes", W.BOOL, W.HANDLE, *([C.POINTER(W.FILETIME)] * 4))
_wait = _signature(_kernel, "WaitForSingleObject", W.DWORD, W.HANDLE, W.DWORD)
_terminate = _signature(_kernel, "TerminateProcess", W.BOOL, W.HANDLE, W.UINT)
_snapshot = _signature(_kernel, "CreateToolhelp32Snapshot", W.HANDLE, W.DWORD, W.DWORD)
_ticks = _signature(_kernel, "GetTickCount64", C.c_ulonglong)
_open_token = _signature(_advapi, "OpenProcessToken", W.BOOL, W.HANDLE, W.DWORD, C.POINTER(W.HANDLE))
_token_info = _signature(_advapi, "GetTokenInformation", W.BOOL, W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.POINTER(W.DWORD))
_lookup_sid = _signature(_advapi, "LookupAccountSidW", W.BOOL, W.LPCWSTR, C.c_void_p,
                         W.LPWSTR, C.POINTER(W.DWORD), W.LPWSTR, C.POINTER(W.DWORD), C.POINTER(W.DWORD))
_image = _signature(_kernel, "QueryFullProcessImageNameW", W.BOOL, W.HANDLE, W.DWORD, W.LPWSTR, C.POINTER(W.DWORD))
_query = _signature(_ntdll, "NtQueryInformationProcess", C.c_long, W.HANDLE, C.c_int,
                   C.c_void_p, W.ULONG, C.POINTER(W.ULONG))
_system_times = _signature(_kernel, "GetSystemTimes", W.BOOL, *([C.POINTER(W.FILETIME)] * 3))


class MemoryStatus(C.Structure):
    _fields_ = [("size", W.DWORD), ("load", W.DWORD)] + [
        (name, C.c_ulonglong) for name in
        ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")
    ]


class ProcessMemory(C.Structure):
    _fields_ = [("size", W.DWORD), ("faults", W.DWORD)] + [
        (name, C.c_size_t) for name in
        ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private")
    ]


_memory = _signature(_kernel, "GlobalMemoryStatusEx", W.BOOL, C.POINTER(MemoryStatus))
_process_memory = _signature(_kernel, "K32GetProcessMemoryInfo", W.BOOL, W.HANDLE, C.POINTER(ProcessMemory), W.DWORD)


class ProcessEntry(C.Structure):
    _fields_ = [("size", W.DWORD), ("usage", W.DWORD), ("pid", W.DWORD),
                ("heap", C.c_size_t), ("module", W.DWORD), ("threads", W.DWORD),
                ("ppid", W.DWORD), ("priority", W.LONG), ("flags", W.DWORD), ("name", W.WCHAR * 260)]


_first = _signature(_kernel, "Process32FirstW", W.BOOL, W.HANDLE, C.POINTER(ProcessEntry))
_next = _signature(_kernel, "Process32NextW", W.BOOL, W.HANDLE, C.POINTER(ProcessEntry))


def _failure(pid):
    code = C.get_last_error()
    if code in (87, 1168):  # invalid parameter / not found
        raise NoSuchProcess(pid)
    raise AccessDenied(pid, f"Windows process operation failed with error {code}")


@contextmanager
def _handle(pid, access=0x1000 | 0x100000):
    handle = _open(access, False, pid)
    if not handle:
        _failure(pid)
    try:
        yield handle
    finally:
        _close(handle)


def _birth(handle, pid):
    values = [W.FILETIME() for _ in range(4)]
    if not _times(handle, *(C.byref(value) for value in values)):
        _failure(pid)
    return (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime


def _entries():
    handle = _snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if handle == W.HANDLE(-1).value:
        _failure(None)
    try:
        entry = ProcessEntry()
        entry.size = C.sizeof(entry)
        more = _first(handle, C.byref(entry))
        while more:
            yield entry.pid, entry.ppid, entry.name, entry.threads
            more = _next(handle, C.byref(entry))
        if C.get_last_error() not in (0, 18):  # ERROR_NO_MORE_FILES
            _failure(None)
    finally:
        _close(handle)


def pids():
    return sorted(pid for pid, *_ in _entries() if pid > 0)


def boot_time():
    return time.time() - _ticks() / 1000


def memory():
    value = MemoryStatus()
    value.size = C.sizeof(value)
    if not _memory(C.byref(value)):
        _failure(None)
    return value.total, value.available, value.total - value.available


def cpu_times():
    idle, kernel, user = W.FILETIME(), W.FILETIME(), W.FILETIME()
    if not _system_times(C.byref(idle), C.byref(kernel), C.byref(user)):
        _failure(None)
    idle, kernel, user = [(value.dwHighDateTime << 32) | value.dwLowDateTime for value in (idle, kernel, user)]
    # Windows kernel time includes idle. On >64-CPU systems this is the calling processor group.
    return kernel + user, kernel + user - idle


def snapshot(pid):
    with _handle(pid) as handle:
        birth = _birth(handle, pid)
        state = _wait(handle, 0)
        if state == 0xFFFFFFFF:
            _failure(pid)
        entry = next((entry for entry in _entries() if entry[0] == pid), None)
        if entry is None:
            raise NoSuchProcess(pid)
        return {
            "identity": birth, "created": (birth - 116444736000000000) / 10_000_000,
            "ppid": entry[1], "name": entry[2], "threads": entry[3],
            "status": "dead" if state == 0 else "running",
        }


def _username(handle, pid):
    token = W.HANDLE()
    if not _open_token(handle, 8, C.byref(token)):
        _failure(pid)
    try:
        size = W.DWORD()
        _token_info(token, 1, None, 0, C.byref(size))  # TokenUser
        if size.value == 0:
            _failure(pid)
        buffer = C.create_string_buffer(size.value)
        if not _token_info(token, 1, buffer, size, C.byref(size)):
            _failure(pid)
        sid = C.cast(buffer, C.POINTER(C.c_void_p))[0]
        user_len, domain_len, kind = W.DWORD(), W.DWORD(), W.DWORD()
        _lookup_sid(None, sid, None, C.byref(user_len), None, C.byref(domain_len), C.byref(kind))
        user, domain = C.create_unicode_buffer(user_len.value), C.create_unicode_buffer(domain_len.value)
        if not _lookup_sid(None, sid, user, C.byref(user_len), domain, C.byref(domain_len), C.byref(kind)):
            _failure(pid)
        return f"{domain.value}\\{user.value}" if domain.value else user.value
    finally:
        _close(token)


def _cmdline(handle, pid):
    # ProcessCommandLineInformation returns an owned UNICODE_STRING buffer.
    # Unsupported Windows releases fail closed rather than reading guessed PEB offsets.
    size = W.ULONG()
    _query(handle, 60, None, 0, C.byref(size))
    if not 0 < size.value <= 16 * 1024 * 1024:
        raise AccessDenied(pid, "Windows process command line is unavailable")
    buffer = C.create_string_buffer(size.value)
    if _query(handle, 60, buffer, size.value, C.byref(size)) < 0:
        raise AccessDenied(pid, "Windows process command line query was denied")
    class UnicodeString(C.Structure):
        _fields_ = [("length", W.USHORT), ("maximum", W.USHORT), ("buffer", C.c_void_p)]
    text = UnicodeString.from_buffer(buffer)
    start, end = C.addressof(buffer), C.addressof(buffer) + len(buffer)
    if not text.buffer or not start <= text.buffer <= end or text.buffer + text.length > end or text.length % 2:
        raise AccessDenied(pid, "Windows process command line buffer was invalid")
    value = C.wstring_at(text.buffer, text.length // 2)
    shell = C.WinDLL("shell32", use_last_error=True)
    split = _signature(shell, "CommandLineToArgvW", C.POINTER(W.LPWSTR), W.LPCWSTR, C.POINTER(C.c_int))
    free = _signature(_kernel, "LocalFree", C.c_void_p, C.c_void_p)
    count = C.c_int()
    arguments = split(value, C.byref(count))
    if not arguments:
        _failure(pid)
    try:
        return [arguments[index] for index in range(count.value)]
    finally:
        free(arguments)


def data(pid, field):
    if field == "cpu_times":
        values = [W.FILETIME() for _ in range(4)]
        with _handle(pid) as handle:
            if not _times(handle, *(C.byref(value) for value in values)):
                _failure(pid)
        return tuple(((value.dwHighDateTime << 32) | value.dwLowDateTime) / 10_000_000
                     for value in (values[3], values[2]))
    if field == "environ" and pid == os.getpid():
        return dict(os.environ)
    if field == "cwd" and pid == os.getpid():
        return os.getcwd()
    if field == "memory_info":
        value = ProcessMemory()
        value.size = C.sizeof(value)
        with _handle(pid, 0x0400 | 0x0010) as handle:
            if not _process_memory(handle, C.byref(value), value.size):
                _failure(pid)
        return namedtuple("MemoryInfo", "rss vms")(value.rss, value.private)
    with _handle(pid) as handle:
        if field == "username":
            return _username(handle, pid)
        if field == "cmdline":
            return _cmdline(handle, pid)
        if field == "exe":
            buffer, size = C.create_unicode_buffer(32768), W.DWORD(32768)
            if not _image(handle, 0, buffer, C.byref(size)):
                _failure(pid)
            return buffer.value
    raise AccessDenied(pid, f"Windows process field {field!r} is not available through this adapter")


def send_signal(pid, identity, sig):
    if sig != signal.SIGTERM:
        raise ValueError("Windows process adapter supports terminate/kill only")
    with _handle(pid, 1 | 0x1000 | 0x100000) as handle:
        if _birth(handle, pid) != identity:
            raise NoSuchProcess(pid)
        if not _terminate(handle, 1):
            _failure(pid)


def suspend(pid, identity, resume):
    function = _signature(_ntdll, "NtResumeProcess" if resume else "NtSuspendProcess", C.c_long, W.HANDLE)
    with _handle(pid, 0x0800 | 0x1000 | 0x100000) as handle:
        if _birth(handle, pid) != identity:
            raise NoSuchProcess(pid)
        if function(handle) < 0:
            raise AccessDenied(pid, "Windows process suspend/resume was denied")
