"""Linux process facts from procfs; signalling uses pidfds when available."""

from collections import namedtuple
from functools import lru_cache
import errno
import os
from pathlib import Path
import signal

from standalone.process_errors import NoSuchProcess


@lru_cache(maxsize=1)
def boot_time():
    with open("/proc/stat", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("btime "):
                return float(line.split()[1])
    raise OSError("procfs does not expose the kernel boot time")


def pids():
    return sorted(int(name) for name in os.listdir("/proc") if name.isdecimal())


def memory():
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        name, _, value = line.partition(":")
        if value.split()[-1:] == ["kB"]:
            values[name] = int(value.split()[0]) * 1024
    total, free = values["MemTotal"], values["MemFree"]
    cached = values.get("Cached", 0) + values.get("SReclaimable", 0)
    reclaimable = free + values.get("Buffers", 0) + cached
    available = min(total, max(0, values.get("MemAvailable", reclaimable)))
    used = total - reclaimable
    return total, available, used if used >= 0 else total - free


def cpu_times():
    with open("/proc/stat", encoding="ascii") as stream:
        values = [int(value) for value in next(stream).split()[1:]]
    # guest/guest_nice are already included in user/nice. I/O wait is not CPU work.
    total = sum(values[:8])
    return total, total - values[3] - (values[4] if len(values) > 4 else 0)


def snapshot(pid):
    raw = Path(f"/proc/{pid}/stat").read_bytes()
    close = raw.rindex(b")")
    fields = raw[close + 2:].split()
    name = os.fsdecode(raw[raw.index(b"(") + 1:close])
    start = int(fields[19])
    values = {}
    for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="surrogateescape").splitlines():
        key, _, value = line.partition(":")
        values[key] = value.strip()
    uids = tuple(int(value) for value in values["Uid"].split()[:3])
    again = Path(f"/proc/{pid}/stat").read_bytes()
    if int(again[again.rindex(b")") + 2:].split()[19]) != start:
        raise NoSuchProcess(pid)
    return {
        "identity": start, "created": boot_time() + start / os.sysconf("SC_CLK_TCK"),
        "ppid": int(fields[1]), "name": name, "uids": uids,
        "status": {b"R": "running", b"S": "sleeping", b"D": "disk-sleep", b"T": "stopped",
                   b"t": "tracing-stop", b"Z": "zombie", b"X": "dead", b"I": "idle"}.get(fields[0], "unknown"),
        "threads": int(values["Threads"]),
    }


def data(pid, field):
    root = Path(f"/proc/{pid}")
    if field == "cpu_times":
        raw = (root / "stat").read_bytes()
        values = raw[raw.rindex(b")") + 2:].split()
        clock = os.sysconf("SC_CLK_TCK")
        return int(values[11]) / clock, int(values[12]) / clock
    if field == "num_fds":
        return len(list((root / "fd").iterdir()))
    if field == "cmdline":
        return [os.fsdecode(value) for value in (root / "cmdline").read_bytes().split(b"\0") if value]
    if field == "environ":
        return dict(os.fsdecode(value).split("=", 1) for value in
                    (root / "environ").read_bytes().split(b"\0") if b"=" in value)
    if field in {"cwd", "exe"}:
        return os.readlink(root / field)
    if field == "memory_info":
        values = (root / "statm").read_text(encoding="ascii").split()
        page = os.sysconf("SC_PAGE_SIZE")
        return namedtuple("MemoryInfo", "rss vms")(int(values[1]) * page, int(values[0]) * page)
    if field == "open_files":
        result = []
        record = namedtuple("OpenFile", "path fd")
        for item in (root / "fd").iterdir():
            try:
                path = os.readlink(item)
                if path.startswith("/") and Path(path).is_file():
                    result.append(record(path, int(item.name)))
            except FileNotFoundError:
                continue
        return result
    raise ValueError(f"unsupported Linux process field: {field}")


def send_signal(pid, identity, sig):
    descriptor = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            try:
                descriptor = os.pidfd_open(pid, 0)
            except OSError as error:
                if error.errno not in {errno.ENOSYS, errno.EINVAL}:
                    raise
        if snapshot(pid)["identity"] != identity:
            raise NoSuchProcess(pid)
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, sig)
        else:
            os.kill(pid, sig)
    finally:
        if descriptor is not None:
            os.close(descriptor)
