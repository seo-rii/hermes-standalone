"""Linux INET descriptors, matched by socket inode in the queried PID's namespace."""

from collections import namedtuple
import os
from pathlib import Path
import socket
import sys


_Address = namedtuple("Address", "ip port")
_Connection = namedtuple("Connection", "fd family type laddr raddr status")
_TCP_STATES = {
    "01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV", "04": "FIN_WAIT1",
    "05": "FIN_WAIT2", "06": "TIME_WAIT", "07": "CLOSE", "08": "CLOSE_WAIT",
    "09": "LAST_ACK", "0A": "LISTEN", "0B": "CLOSING", "0C": "NEW_SYN_RECV",
}
_KINDS = {
    "inet": ("tcp", "tcp6", "udp", "udp6"), "inet4": ("tcp", "udp"),
    "inet6": ("tcp6", "udp6"), "tcp": ("tcp", "tcp6"), "tcp4": ("tcp",),
    "tcp6": ("tcp6",), "udp": ("udp", "udp6"), "udp4": ("udp",), "udp6": ("udp6",),
}


def _address(value, family):
    host, port = value.split(":")
    packed = bytes.fromhex(host)
    if sys.byteorder == "little":
        packed = b"".join(packed[index:index + 4][::-1] for index in range(0, len(packed), 4))
    return _Address(socket.inet_ntop(family, packed), int(port, 16))


def connections(pid, kind):
    if kind not in _KINDS:
        raise ValueError(f"Unsupported INET connection kind: {kind}")
    root = Path(f"/proc/{pid}")
    descriptors = {}
    for item in (root / "fd").iterdir():
        try:
            target = os.readlink(item)
        except FileNotFoundError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            descriptors.setdefault(target[8:-1], []).append(int(item.name))
    result = []
    for protocol in _KINDS[kind]:
        family = socket.AF_INET6 if protocol.endswith("6") else socket.AF_INET
        transport = socket.SOCK_STREAM if protocol.startswith("tcp") else socket.SOCK_DGRAM
        try:
            stream = (root / "net" / protocol).open(encoding="ascii")
        except FileNotFoundError:
            if family == socket.AF_INET6:
                continue  # IPv6 can be disabled by the kernel.
            raise
        with stream:
            next(stream)
            for line in stream:
                fields = line.split()
                if len(fields) < 10:
                    raise ValueError("Kernel network descriptor row is incomplete")
                if fields[9] not in descriptors:
                    continue
                local, remote = _address(fields[1], family), _address(fields[2], family)
                status = _TCP_STATES[fields[3]] if transport == socket.SOCK_STREAM else "NONE"
                remote = remote if remote.port else ()
                result.extend(_Connection(fd, family, transport, local, remote, status)
                              for fd in descriptors[fields[9]])
    return result
