"""Bounded Chromium DevTools pipe and retained POSIX child ownership."""

from __future__ import annotations

from collections import deque
import contextlib
import fcntl
import json
import os
import selectors
import signal
import threading
import time


MAX_MESSAGE = 8 * 1024 * 1024


class CDPError(RuntimeError):
    pass


class BrowserChild:
    """Own a waitable PID until it is reaped; never identify a process by argv."""

    def __init__(self, executable, arguments, profile):
        incoming, outgoing = os.pipe(), os.pipe()
        self.pid, self.status = None, None
        self.read_fd, self.write_fd = outgoing[0], incoming[1]
        descriptors = [*incoming, *outgoing]
        try:
            # Sources above fd 4 cannot be overwritten by an earlier dup2 action.
            read_source = fcntl.fcntl(incoming[0], fcntl.F_DUPFD_CLOEXEC, 10)
            descriptors.append(read_source)
            write_source = fcntl.fcntl(outgoing[1], fcntl.F_DUPFD_CLOEXEC, 10)
            descriptors.append(write_source)
            null = os.open(os.devnull, os.O_RDWR)
            descriptors.append(null)
            log = os.open(str(profile / "chromium.log"), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            descriptors.append(log)
            actions = [(os.POSIX_SPAWN_DUP2, read_source, 3), (os.POSIX_SPAWN_DUP2, write_source, 4),
                       (os.POSIX_SPAWN_DUP2, null, 0), (os.POSIX_SPAWN_DUP2, null, 1),
                       (os.POSIX_SPAWN_DUP2, log, 2)]
            actions.extend((os.POSIX_SPAWN_CLOSE, fd) for fd in descriptors if fd not in (0, 1, 2, 3, 4))
            self.pid = os.posix_spawn(executable, [executable, *arguments], dict(os.environ), file_actions=actions)
        except BaseException:
            for fd in (self.read_fd, self.write_fd):
                with contextlib.suppress(OSError):
                    os.close(fd)
            raise
        finally:
            for fd in descriptors:
                if fd not in (self.read_fd, self.write_fd):
                    os.close(fd)

    def poll(self):
        if self.pid is not None:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.pid, self.status = None, os.waitstatus_to_exitcode(status)
        return self.status

    def _wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.poll() is None and time.monotonic() < deadline:
            threading.Event().wait(.01)
        return self.poll() is not None

    def close(self):
        if self._wait(2):
            return
        for sig, timeout in ((signal.SIGTERM, 1), (signal.SIGKILL, 2)):
            if self.pid is None:
                return
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.pid, sig)
            if self._wait(timeout):
                return
        raise CDPError("Owned Chromium child did not exit within its cleanup deadline")


class PipeCDP:
    def __init__(self, child):
        self.child, self._buffer = child, bytearray()
        self._sequence, self._pending, self._results = 0, set(), {}
        self._events, self._event_bytes = deque(), 0
        self.listeners, self.closed = [], False
        self._lock = threading.RLock()
        for fd in (child.read_fd, child.write_fd):
            os.set_blocking(fd, False)

    def _ready(self, fd, event, deadline):
        if self.closed:
            raise CDPError("Chromium DevTools pipe is closed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Chromium DevTools operation timed out")
        with selectors.DefaultSelector() as selector:
            selector.register(fd, event)
            if not selector.select(remaining):
                raise TimeoutError("Chromium DevTools operation timed out")

    def send(self, method, params=None, *, session=None, timeout=30):
        with self._lock:
            if len(self._pending) >= 32:
                raise CDPError("Chromium DevTools pending request bound exceeded")
            self._sequence += 1
            message = {"id": self._sequence, "method": method, "params": params or {}}
            if session is not None:
                message["sessionId"] = session
            raw = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode() + b"\0"
            if len(raw) > MAX_MESSAGE:
                raise CDPError("Chromium DevTools request exceeds its byte limit")
            self._pending.add(self._sequence)
            offset, deadline = 0, time.monotonic() + timeout
            while offset < len(raw):
                self._ready(self.child.write_fd, selectors.EVENT_WRITE, deadline)
                try:
                    amount = os.write(self.child.write_fd, raw[offset:])
                except BlockingIOError:
                    continue
                except BrokenPipeError as error:
                    raise CDPError("Chromium closed its command pipe") from error
                if not amount:
                    raise CDPError("Chromium DevTools command pipe made no progress")
                offset += amount
            return self._sequence

    def _read(self, deadline):
        while True:
            end = self._buffer.find(0)
            if end >= 0:
                raw = bytes(self._buffer[:end])
                del self._buffer[:end + 1]
                try:
                    message = json.loads(raw)
                except (UnicodeDecodeError, ValueError, RecursionError) as error:
                    raise CDPError("Chromium returned malformed DevTools JSON") from error
                if not isinstance(message, dict):
                    raise CDPError("Chromium returned a non-object DevTools message")
                identifier = message.get("id")
                if identifier is not None:
                    if type(identifier) is not int or identifier not in self._pending or identifier in self._results:
                        raise CDPError("Chromium returned an unexpected DevTools response ID")
                    self._results[identifier] = message
                elif isinstance(message.get("method"), str):
                    for listener in tuple(self.listeners):
                        listener(message)
                    retained = message["method"] == "Fetch.requestPaused" or (
                        message["method"] == "Page.lifecycleEvent" and message.get("params", {}).get("name") == "DOMContentLoaded")
                    if retained:
                        if len(self._events) >= 512 or self._event_bytes + len(raw) > MAX_MESSAGE * 2:
                            raise CDPError("Chromium DevTools event queue exceeded its bounded capacity")
                        self._events.append((message, len(raw)))
                        self._event_bytes += len(raw)
                else:
                    raise CDPError("Chromium returned an invalid DevTools event")
                return
            if len(self._buffer) >= MAX_MESSAGE:
                raise CDPError("Chromium DevTools message exceeds its byte limit")
            self._ready(self.child.read_fd, selectors.EVENT_READ, deadline)
            try:
                chunk = os.read(self.child.read_fd, min(65536, MAX_MESSAGE - len(self._buffer)))
            except BlockingIOError:
                continue
            if not chunk:
                raise CDPError("Chromium closed its DevTools pipe")
            self._buffer.extend(chunk)

    def result(self, identifier, *, timeout=30):
        with self._lock:
            deadline = time.monotonic() + timeout
            while identifier not in self._results:
                self._read(deadline)
            message = self._results.pop(identifier)
            self._pending.remove(identifier)
            if "error" in message:
                code = message["error"].get("code") if isinstance(message["error"], dict) else None
                raise CDPError(f"Chromium rejected DevTools request (code {code})")
            if not isinstance(message.get("result"), dict):
                raise CDPError("Chromium returned an invalid DevTools result")
            return message["result"]

    def request(self, method, params=None, *, session=None, timeout=30):
        with self._lock:
            return self.result(self.send(method, params, session=session, timeout=timeout), timeout=timeout)

    def event(self, predicate, *, timeout=30):
        with self._lock:
            deadline = time.monotonic() + timeout
            while True:
                for item, size in self._events:
                    if predicate(item):
                        self._events.remove((item, size))
                        self._event_bytes -= size
                        return item
                self._read(deadline)

    def discard_events(self, predicate):
        with self._lock:
            retained = [(event, size) for event, size in self._events if not predicate(event)]
            self._events = deque(retained)
            self._event_bytes = sum(size for _, size in retained)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for fd in (self.child.read_fd, self.child.write_fd):
            with contextlib.suppress(OSError):
                os.close(fd)
