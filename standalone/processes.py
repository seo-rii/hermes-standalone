"""Hermes process operations implemented using the Python standard library.

This explicit dependency implements the process operations Hermes uses. It is
not installed as psutil. Every Process retains its kernel birth identity, and
mutation verifies that identity again before signalling. Missing permissions
raise AccessDenied rather than inventing ownership or liveness information.
"""

from collections import namedtuple
import math
import os
import shutil
import signal
import sys
import threading
import time

from standalone.process_errors import AccessDenied, Error, NoSuchProcess, TimeoutExpired, ZombieProcess

if sys.platform == "win32":
    from standalone import process_windows as _os_process
elif sys.platform == "darwin":
    from standalone import process_darwin as _os_process
elif sys.platform.startswith("linux"):
    from standalone import process_linux as _os_process
else:
    raise ImportError(f"Standalone process ownership is not implemented on {sys.platform}")

MACOS = sys.platform == "darwin"
WINDOWS = sys.platform == "win32"
LINUX = sys.platform.startswith("linux")
STATUS_ZOMBIE = "zombie"
STATUS_DEAD = "dead"
STATUS_STOPPED = "stopped"
STATUS_RUNNING = "running"
STATUS_SLEEPING = "sleeping"
_cpu_samples = threading.local()


def _call(pid, operation, *args):
    try:
        return operation(*args)
    except (FileNotFoundError, ProcessLookupError) as error:
        raise NoSuchProcess(pid) from error
    except PermissionError as error:
        raise AccessDenied(pid) from error
    except OSError as error:
        raise AccessDenied(pid, f"process query failed: {error.strerror or type(error).__name__}") from error


def pids():
    return _call(None, _os_process.pids)


def boot_time():
    return _call(None, _os_process.boot_time)


def sensors_battery():
    from standalone.power import sensors_battery as read_battery
    return read_battery()


def win_service_get(name):
    from standalone.windows_services import win_service_get as service
    return service(name)


def win_service_iter():
    from standalone.windows_services import win_service_iter as services
    return services()


def virtual_memory():
    total, available, used = _call(None, _os_process.memory)
    percent = round(100 * (total - available) / total, 1) if total else 0.0
    return namedtuple("VirtualMemory", "total available percent used")(total, available, percent, used)


def disk_usage(path):
    usage = shutil.disk_usage(path)
    # Match the user's usable capacity, excluding POSIX blocks reserved for root.
    usable = usage.used + usage.free
    percent = round(100 * usage.used / usable, 1) if usable else 0.0
    return namedtuple("DiskUsage", "total used free percent")(*usage, percent)


def cpu_percent(interval=None):
    """Sample aggregate kernel CPU counters; nonblocking calls retain a thread-local baseline."""
    if interval is not None and (not math.isfinite(interval) or interval < 0):
        raise ValueError("CPU sampling interval must be finite and nonnegative")
    current = _call(None, _os_process.cpu_times)
    if interval:
        previous = current
        time.sleep(interval)
        current = _call(None, _os_process.cpu_times)
    else:
        previous = getattr(_cpu_samples, "previous", current)
    _cpu_samples.previous = current
    elapsed = max(0, current[0] - previous[0])
    busy = max(0, current[1] - previous[1])
    return round(min(100.0, 100 * busy / elapsed), 1) if elapsed else 0.0


if hasattr(os, "getloadavg"):
    getloadavg = os.getloadavg


class Process:
    def __init__(self, pid=None):
        self.pid = os.getpid() if pid is None else pid
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("process PID must be a positive integer")
        if self.pid > 2**31 - 1:
            raise NoSuchProcess(self.pid)
        first = _call(self.pid, _os_process.snapshot, self.pid)
        self._identity = first["identity"]
        self._created = first["created"]
        self.info = {}
        self._waited = False
        self.returncode = None
        self._cpu_sample = None

    def _snapshot(self):
        current = _call(self.pid, _os_process.snapshot, self.pid)
        if current["identity"] != self._identity:
            raise NoSuchProcess(self.pid)
        return current

    def _data(self, field):
        self._snapshot()
        value = _call(self.pid, _os_process.data, self.pid, field)
        self._snapshot()
        return value

    def create_time(self):
        self._snapshot()
        return self._created

    def ppid(self):
        return self._snapshot()["ppid"]

    def name(self):
        return self._snapshot()["name"]

    def status(self):
        return self._snapshot()["status"]

    def is_running(self):
        try:
            return self._snapshot()["status"] != STATUS_DEAD
        except NoSuchProcess:
            return False

    def uids(self):
        record = self._snapshot()
        if "uids" not in record:
            raise AccessDenied(self.pid, "UIDs are not available on this operating system")
        return namedtuple("Uids", "real effective saved")(*record["uids"])

    def username(self):
        if WINDOWS:
            return self._data("username")
        import pwd
        uid = self.uids().real
        try:
            return pwd.getpwuid(uid).pw_name
        except KeyError:
            return str(uid)

    def parent(self):
        parent_pid = self.ppid()
        if parent_pid <= 0:
            return None
        try:
            parent = Process(parent_pid)
            if parent.create_time() > self.create_time() or self.ppid() != parent_pid:
                return None
            return parent
        except NoSuchProcess:
            return None

    def parents(self):
        result, seen = [], {self.pid}
        current = self
        while (current := current.parent()) is not None and current.pid not in seen:
            seen.add(current.pid)
            result.append(current)
        return result

    def children(self, recursive=False):
        self._snapshot()
        by_parent = {}
        for child in process_iter():
            try:
                by_parent.setdefault(child.ppid(), []).append(child)
            except (NoSuchProcess, AccessDenied):
                continue
        result, pending, seen = [], [self], {self.pid}
        while pending:
            parent = pending.pop()
            for child in by_parent.get(parent.pid, ()):
                try:
                    if child.pid in seen or child.create_time() < parent.create_time():
                        continue
                    if child.ppid() != parent.pid:
                        continue
                    seen.add(child.pid)
                    result.append(child)
                    if recursive:
                        pending.append(child)
                except (NoSuchProcess, AccessDenied):
                    continue
        self._snapshot()
        return result

    def cmdline(self):
        return self._data("cmdline")

    def environ(self):
        return self._data("environ")

    def cwd(self):
        return self._data("cwd")

    def exe(self):
        return self._data("exe")

    def open_files(self):
        return self._data("open_files")

    def memory_info(self):
        return self._data("memory_info")

    def num_threads(self):
        value = self._snapshot()["threads"]
        return value if value is not None else self._data("num_threads")

    def cpu_times(self):
        user, system = self._data("cpu_times")
        return namedtuple("CpuTimes", "user system")(user, system)

    def cpu_percent(self, interval=None):
        if interval is not None and (not math.isfinite(interval) or interval < 0):
            raise ValueError("CPU sampling interval must be finite and nonnegative")
        current = (time.monotonic(), sum(self.cpu_times()))
        if interval:
            previous = current
            time.sleep(interval)
            current = (time.monotonic(), sum(self.cpu_times()))
        else:
            previous = self._cpu_sample or current
        self._cpu_sample = current
        elapsed = current[0] - previous[0]
        return round(max(0, 100 * (current[1] - previous[1]) / elapsed), 1) if elapsed else 0.0

    def num_fds(self):
        if WINDOWS:
            raise AttributeError("num_fds is a POSIX-only process operation")
        return self._data("num_fds")

    def net_connections(self, kind="inet"):
        if not LINUX:
            raise AccessDenied(self.pid, "Network descriptor queries are implemented through Linux procfs")
        from standalone.process_network import connections
        self._snapshot()
        result = _call(self.pid, connections, self.pid, kind)
        self._snapshot()
        return result

    def send_signal(self, sig):
        _call(self.pid, _os_process.send_signal, self.pid, self._identity, sig)

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGTERM if WINDOWS else signal.SIGKILL)

    def suspend(self):
        if WINDOWS:
            return _call(self.pid, _os_process.suspend, self.pid, self._identity, False)
        self.send_signal(signal.SIGSTOP)

    def resume(self):
        if WINDOWS:
            return _call(self.pid, _os_process.suspend, self.pid, self._identity, True)
        self.send_signal(signal.SIGCONT)

    def wait(self, timeout=None):
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._waited:
            try:
                current = self._snapshot()
            except NoSuchProcess:
                self._waited = True
                break
            if not WINDOWS:
                try:
                    pid, status = os.waitpid(self.pid, os.WNOHANG)
                    if pid:
                        self.returncode = os.waitstatus_to_exitcode(status)
                        self._waited = True
                        break
                except ChildProcessError:
                    pass
            if current["status"] in {STATUS_ZOMBIE, STATUS_DEAD}:
                self._waited = True
            if self._waited:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutExpired(timeout, self.pid)
            time.sleep(0.01)
        return self.returncode

    def as_dict(self, attrs, ad_value=None):
        result = {}
        for name in attrs:
            if name == "pid":
                result[name] = self.pid
                continue
            if name.startswith("_") or name not in {
                "name", "ppid", "username", "uids", "status", "create_time", "cmdline",
                "environ", "cwd", "exe", "open_files", "memory_info", "num_threads",
                "cpu_percent", "cpu_times", "num_fds",
            }:
                raise ValueError(f"unsupported process attribute: {name}")
            try:
                result[name] = getattr(self, name)()
            except AccessDenied:
                result[name] = ad_value
        return result


def process_iter(attrs=None, ad_value=None):
    for pid in pids():
        if pid <= 0:
            continue
        try:
            process = Process(pid)
            if attrs is not None:
                process.info = process.as_dict(attrs, ad_value)
            yield process
        except (NoSuchProcess, AccessDenied):
            continue


def pid_exists(pid):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        Process(pid)
        return True
    except (NoSuchProcess, ValueError):
        return False
    except AccessDenied:
        # Permission denial cannot prove absence.
        return True


def wait_procs(procs, timeout=None, callback=None):
    pending, gone = list(procs), []
    deadline = None if timeout is None else time.monotonic() + max(0, timeout)
    while pending:
        for process in tuple(pending):
            try:
                process.wait(timeout=0)
            except (TimeoutExpired, AccessDenied):
                continue
            pending.remove(process)
            gone.append(process)
            if callback is not None:
                callback(process)
        if deadline is not None and time.monotonic() >= deadline:
            break
        if pending:
            time.sleep(0.01)
    return gone, pending
