"""Read Windows SCM service identity through documented operating-system APIs.

This module never starts, stops, installs or changes a service. SCM handles are
opened with only enumeration/query rights and are closed on every path.
Native execution requires Windows; buffer decoding and ABI layouts are portable.
"""

from contextlib import contextmanager
import ctypes as C
from functools import lru_cache
import os

from standalone.process_errors import AccessDenied, Error, NoSuchProcess


DWORD = C.c_uint32
HANDLE = C.c_void_p
_STATES = {1: "stopped", 2: "start_pending", 3: "stop_pending", 4: "running",
           5: "continue_pending", 6: "pause_pending", 7: "paused"}
_START_TYPES = {0: "boot", 1: "system", 2: "automatic", 3: "manual", 4: "disabled"}


class ServiceStatus(C.Structure):
    _fields_ = [(name, DWORD) for name in (
        "service_type", "state", "accepted", "exit_code", "service_exit_code",
        "checkpoint", "wait_hint", "pid", "flags",
    )]


class ServiceConfig(C.Structure):
    _fields_ = [("service_type", DWORD), ("start_type", DWORD), ("error_control", DWORD),
                ("binpath", C.c_void_p), ("group", C.c_void_p), ("tag", DWORD),
                ("dependencies", C.c_void_p), ("username", C.c_void_p), ("display_name", C.c_void_p)]


class ServiceEntry(C.Structure):
    _fields_ = [("name", C.c_void_p), ("display_name", C.c_void_p), ("status", ServiceStatus)]


def _buffer_text(buffer, address):
    if not address:
        return ""
    start, end = C.addressof(buffer), C.addressof(buffer) + C.sizeof(buffer)
    if address < start or address >= end or (address - start) % 2:
        raise Error("SCM returned a string outside its response buffer")
    data = C.string_at(address, end - address)
    for offset in range(0, len(data) - 1, 2):
        if data[offset:offset + 2] == b"\0\0":
            return data[:offset].decode("utf-16-le", errors="surrogatepass")
    raise Error("SCM returned an unterminated Unicode string")


def _failure(name=None, code=None):
    code = C.get_last_error() if code is None else code
    if code == 1060:  # ERROR_SERVICE_DOES_NOT_EXIST
        raise NoSuchProcess(name, "Windows service does not exist")
    raise AccessDenied(name, f"Windows SCM query failed with error {code}")


@lru_cache(maxsize=1)
def _api():
    if os.name != "nt":
        raise AttributeError("Windows SCM queries require Windows")
    library = C.WinDLL("advapi32", use_last_error=True)

    def signature(name, result, *arguments):
        function = getattr(library, name)
        function.restype, function.argtypes = result, list(arguments)
        return function

    return {
        "open_manager": signature("OpenSCManagerW", HANDLE, C.c_wchar_p, C.c_wchar_p, DWORD),
        "open_service": signature("OpenServiceW", HANDLE, HANDLE, C.c_wchar_p, DWORD),
        "close": signature("CloseServiceHandle", C.c_int32, HANDLE),
        "enumerate": signature("EnumServicesStatusExW", C.c_int32, HANDLE, C.c_int32, DWORD, DWORD,
                               C.c_void_p, DWORD, C.POINTER(DWORD), C.POINTER(DWORD), C.POINTER(DWORD), C.c_wchar_p),
        "status": signature("QueryServiceStatusEx", C.c_int32, HANDLE, C.c_int32, C.c_void_p, DWORD,
                            C.POINTER(DWORD)),
        "config": signature("QueryServiceConfigW", C.c_int32, HANDLE, C.c_void_p, DWORD, C.POINTER(DWORD)),
        "description": signature("QueryServiceConfig2W", C.c_int32, HANDLE, DWORD, C.c_void_p, DWORD,
                                 C.POINTER(DWORD)),
    }


@contextmanager
def _manager(access):
    api = _api()
    handle = api["open_manager"](None, None, access)
    if not handle:
        _failure()
    try:
        yield api, handle
    finally:
        api["close"](handle)


@contextmanager
def _service(name, access):
    with _manager(1) as (api, manager):  # SC_MANAGER_CONNECT
        handle = api["open_service"](manager, name, access)
        if not handle:
            _failure(name)
        try:
            yield api, handle
        finally:
            api["close"](handle)


def _status(name):
    with _service(name, 4) as (api, handle):  # SERVICE_QUERY_STATUS
        value, needed = ServiceStatus(), DWORD()
        if not api["status"](handle, 0, C.byref(value), C.sizeof(value), C.byref(needed)):
            _failure(name)
        if value.state not in _STATES:
            raise Error(f"SCM returned an unknown service state: {value.state}")
        return value


def _config(name):
    with _service(name, 1) as (api, handle):  # SERVICE_QUERY_CONFIG
        needed = DWORD()
        api["config"](handle, None, 0, C.byref(needed))
        if C.get_last_error() != 122 or not C.sizeof(ServiceConfig) <= needed.value <= 8192:
            _failure(name)
        buffer = C.create_string_buffer(needed.value)
        if not api["config"](handle, buffer, C.sizeof(buffer), C.byref(needed)):
            _failure(name)
        value = ServiceConfig.from_buffer(buffer)
        if value.start_type not in _START_TYPES:
            raise Error(f"SCM returned an unknown service start type: {value.start_type}")
        return {"binpath": _buffer_text(buffer, value.binpath), "username": _buffer_text(buffer, value.username),
                "display_name": _buffer_text(buffer, value.display_name), "start_type": _START_TYPES[value.start_type]}


class WindowsService:
    def __init__(self, name, display_name=None):
        if not isinstance(name, str) or not name or len(name) > 256 or "\0" in name:
            raise ValueError("A Windows service name must contain 1 to 256 non-null characters")
        self._name, self._display_name = name, display_name

    def name(self):
        return self._name

    def display_name(self):
        return self._display_name if self._display_name is not None else _config(self._name)["display_name"]

    def binpath(self):
        return _config(self._name)["binpath"]

    def username(self):
        return _config(self._name)["username"]

    def start_type(self):
        return _config(self._name)["start_type"]

    def status(self):
        return _STATES[_status(self._name).state]

    def pid(self):
        value = _status(self._name)
        # SCM explicitly provides no valid PID for a stopped service.
        return int(value.pid) if value.state != 1 and value.pid else None

    def description(self):
        with _service(self._name, 1) as (api, handle):
            needed = DWORD()
            api["description"](handle, 1, None, 0, C.byref(needed))
            if C.get_last_error() != 122 or not C.sizeof(C.c_void_p) <= needed.value <= 8192:
                _failure(self._name)
            buffer = C.create_string_buffer(needed.value)
            if not api["description"](handle, 1, buffer, C.sizeof(buffer), C.byref(needed)):
                _failure(self._name)
            return _buffer_text(buffer, C.c_void_p.from_buffer(buffer).value)

    def as_dict(self):
        config, status = _config(self._name), _status(self._name)
        return {"name": self._name, **config, "status": _STATES[status.state],
                "pid": int(status.pid) if status.state != 1 and status.pid else None,
                "description": self.description()}


def win_service_get(name):
    service = WindowsService(name)
    _status(name)  # Missing and inaccessible services keep their own error semantics.
    return service


def win_service_iter():
    with _manager(4) as (api, handle):  # SC_MANAGER_ENUMERATE_SERVICE
        resume, seen = DWORD(), set()
        for _ in range(64):
            buffer = C.create_string_buffer(256 * 1024)
            needed, count = DWORD(), DWORD()
            previous = resume.value
            complete = api["enumerate"](handle, 0, 0x30, 3, buffer, C.sizeof(buffer),
                                        C.byref(needed), C.byref(count), C.byref(resume), None)
            error = 0 if complete else C.get_last_error()
            if error not in (0, 234):  # ERROR_MORE_DATA
                _failure(code=error)
            if count.value > C.sizeof(buffer) // C.sizeof(ServiceEntry):
                raise Error("SCM service enumeration exceeded its response buffer")
            records = C.cast(buffer, C.POINTER(ServiceEntry))
            for index in range(count.value):
                name = _buffer_text(buffer, records[index].name)
                if name not in seen:
                    seen.add(name)
                    yield WindowsService(name, _buffer_text(buffer, records[index].display_name))
            if complete:
                return
            if resume.value == previous:
                raise Error("SCM service enumeration did not advance its cursor")
        raise Error("SCM service enumeration exceeded its bounded cursor budget")
