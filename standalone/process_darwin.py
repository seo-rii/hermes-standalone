"""macOS process identity from libproc, without a compiled Python extension.

Structure layout: Apple's xnu bsd/sys/proc_info.h (proc_bsdinfo/proc_taskinfo).
Process birth is the unadjusted kernel timeval used by Hermes worker receipts.
"""

from collections import namedtuple
import ctypes as C
import errno
import os
import signal
import struct

from standalone.process_errors import AccessDenied, NoSuchProcess

_libproc = C.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_libc = C.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
_libproc.proc_pidinfo.argtypes = [C.c_int, C.c_int, C.c_uint64, C.c_void_p, C.c_int]
_libproc.proc_pidinfo.restype = C.c_int
_libproc.proc_listallpids.argtypes = [C.c_void_p, C.c_int]
_libproc.proc_listallpids.restype = C.c_int
_libproc.proc_pidpath.argtypes = [C.c_int, C.c_void_p, C.c_uint32]
_libproc.proc_pidpath.restype = C.c_int
_libc.sysctl.argtypes = [C.POINTER(C.c_int), C.c_uint, C.c_void_p, C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
_libc.sysctl.restype = C.c_int
_libc.sysctlbyname.argtypes = [C.c_char_p, C.c_void_p, C.POINTER(C.c_size_t), C.c_void_p, C.c_size_t]
_libc.sysctlbyname.restype = C.c_int
_libc.mach_host_self.argtypes = []
_libc.mach_host_self.restype = C.c_uint32
_libc.host_statistics.argtypes = [C.c_uint32, C.c_int, C.c_void_p, C.POINTER(C.c_uint32)]
_libc.host_statistics.restype = C.c_int
_libc.mach_port_deallocate.argtypes = [C.c_uint32, C.c_uint32]
_libc.mach_port_deallocate.restype = C.c_int


class BSDInfo(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in (
        "flags", "status", "xstatus", "pid", "ppid", "uid", "gid", "ruid", "rgid", "svuid", "svgid", "reserved",
    )] + [("comm", C.c_char * 16), ("name", C.c_char * 32)] + [
        (name, C.c_uint32) for name in ("nfiles", "pgid", "jobc", "tdev", "tpgid")
    ] + [("nice", C.c_int32), ("start_sec", C.c_uint64), ("start_usec", C.c_uint64)]


class TaskInfo(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "virtual_size", "resident_size", "total_user", "total_system", "threads_user", "threads_system",
    )] + [(name, C.c_int32) for name in (
        "policy", "faults", "pageins", "cow_faults", "messages_sent", "messages_received",
        "syscalls_mach", "syscalls_unix", "csw", "threadnum", "numrunning", "priority",
    )]


def _failure(pid):
    code = C.get_errno()
    if code in (0, errno.ESRCH, errno.ENOENT):
        raise NoSuchProcess(pid)
    raise AccessDenied(pid, os.strerror(code))


def _info(pid, flavor, structure):
    value = structure()
    C.set_errno(0)
    count = _libproc.proc_pidinfo(pid, flavor, 0, C.byref(value), C.sizeof(value))
    if count != C.sizeof(value):
        _failure(pid)
    return value


def pids():
    count = _libproc.proc_listallpids(None, 0)
    if count < 0:
        _failure(None)
    for _ in range(4):
        capacity = count + 128
        values = (C.c_int * capacity)()
        count = _libproc.proc_listallpids(values, C.sizeof(values))
        if count < 0:
            _failure(None)
        if count < capacity:
            return sorted(pid for pid in values[:count] if pid > 0)
    raise AccessDenied(None, "process table grew while enumerating")


def _sysctl(mib):
    names = (C.c_int * len(mib))(*mib)
    length = C.c_size_t()
    if _libc.sysctl(names, len(mib), None, C.byref(length), None, 0) != 0:
        _failure(mib[-1])
    buffer = C.create_string_buffer(length.value)
    if _libc.sysctl(names, len(mib), buffer, C.byref(length), None, 0) != 0:
        _failure(mib[-1])
    return buffer.raw[:length.value]


def boot_time():
    raw = _sysctl([1, 21])  # CTL_KERN, KERN_BOOTTIME
    class Timeval(C.Structure):
        _fields_ = [("seconds", C.c_long), ("microseconds", C.c_int)]
    value = Timeval.from_buffer_copy(raw)
    return value.seconds + value.microseconds / 1_000_000


def _host_statistics(flavor, count):
    values = (C.c_uint32 * count)()
    length = C.c_uint32(count)
    host = _libc.mach_host_self()
    try:
        result = _libc.host_statistics(host, flavor, values, C.byref(length))
        if result or length.value != count:
            raise AccessDenied(None, f"macOS host statistics failed: {result}")
        return values
    finally:
        task = C.c_uint32.in_dll(_libc, "mach_task_self_").value
        _libc.mach_port_deallocate(task, host)


def memory():
    total, length = C.c_uint64(), C.c_size_t(C.sizeof(C.c_uint64))
    if _libc.sysctlbyname(b"hw.memsize", C.byref(total), C.byref(length), None, 0):
        _failure(None)
    values = _host_statistics(2, 15)  # HOST_VM_INFO, vm_statistics rev2
    page = os.sysconf("SC_PAGE_SIZE")
    available = min(total.value, (values[0] + values[2]) * page)
    used = (values[1] + values[3]) * page  # active + wired, as in upstream psutil
    return total.value, available, used


def cpu_times():
    values = _host_statistics(3, 4)  # HOST_CPU_LOAD_INFO: user, system, idle, nice
    total = sum(values)
    return total, total - values[2]


def snapshot(pid):
    value = _info(pid, 3, BSDInfo)  # PROC_PIDTBSDINFO
    if value.pid != pid:
        raise NoSuchProcess(pid)
    return {
        "identity": (value.start_sec, value.start_usec),
        "created": value.start_sec + value.start_usec / 1_000_000,
        "ppid": value.ppid, "name": os.fsdecode(value.name or value.comm),
        "uids": (value.ruid, value.uid, value.svuid), "threads": None,
        "status": {1: "idle", 2: "running", 3: "sleeping", 4: "stopped", 5: "zombie"}.get(value.status, "unknown"),
    }


def _arguments(pid):
    raw = _sysctl([1, 49, pid])  # CTL_KERN, KERN_PROCARGS2
    if len(raw) < 4:
        raise AccessDenied(pid, "kernel process arguments were truncated")
    count = struct.unpack_from("i", raw)[0]
    position = raw.index(b"\0", 4) + 1  # executable path, then alignment padding
    while position < len(raw) and raw[position] == 0:
        position += 1
    fields = raw[position:].split(b"\0")
    if count < 0 or count > len(fields):
        raise AccessDenied(pid, "kernel process argument count was invalid")
    return fields[:count], fields[count:]


def data(pid, field):
    if field == "num_fds":
        C.set_errno(0)
        size = _libproc.proc_pidinfo(pid, 1, 0, None, 0)  # PROC_PIDLISTFDS: proc_fdinfo is 8 bytes.
        if size < 0 or (size == 0 and C.get_errno()):
            _failure(pid)
        if size == 0:
            _info(pid, 3, BSDInfo)  # Distinguish no descriptors from a process that disappeared.
        return size // 8
    if field in {"cmdline", "environ"}:
        arguments, environment = _arguments(pid)
        if field == "cmdline":
            return [os.fsdecode(value) for value in arguments]
        return dict(os.fsdecode(value).split("=", 1) for value in environment if b"=" in value)
    if field == "exe":
        path = C.create_string_buffer(4096)
        if _libproc.proc_pidpath(pid, path, len(path)) <= 0:
            _failure(pid)
        return os.fsdecode(path.value)
    if field in {"memory_info", "num_threads", "cpu_times"}:
        task = _info(pid, 4, TaskInfo)
        if field == "cpu_times":
            return task.total_user / 1_000_000_000, task.total_system / 1_000_000_000
        if field == "num_threads":
            return task.threadnum
        return namedtuple("MemoryInfo", "rss vms")(task.resident_size, task.virtual_size)
    if field == "cwd" and pid == os.getpid():
        return os.getcwd()
    raise AccessDenied(pid, f"macOS process field {field!r} is not available through this adapter")


def send_signal(pid, identity, sig):
    if snapshot(pid)["identity"] != identity:
        raise NoSuchProcess(pid)
    os.kill(pid, sig)
