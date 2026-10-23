"""Qualified stdlib polling for Modal's native watchfiles event boundary.

The original watchfiles filters are retained. Polling reports observed file
states; a change reverted entirely between polls cannot be observed, and native
notification ordering/coalescing is not reproduced.
"""

import asyncio
from enum import IntEnum
import os
from pathlib import Path
import stat
import time

from standalone.watch_filters import DefaultFilter


class Change(IntEnum):
    added = 1
    modified = 2
    deleted = 3


def _snapshot(paths, recursive, ignore_denied, *, initial=False):
    result = {}
    pending = list(paths)
    while pending:
        path = pending.pop()
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            if initial and path in paths:
                raise
            continue
        except PermissionError:
            if ignore_denied:
                continue
            raise
        result[path] = (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                        info.st_mtime_ns, info.st_ctime_ns)
        if len(result) > 1_000_000:
            raise RuntimeError("File watch exceeds one million observed entries")
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(path) as entries:
                    for entry in entries:
                        if recursive or not entry.is_dir(follow_symlinks=False):
                            pending.append(entry.path)
            except FileNotFoundError:
                continue
            except PermissionError:
                if not ignore_denied:
                    raise
    return result


async def awatch(*paths, watch_filter=DefaultFilter(), debounce=1600, step=50,
                 stop_event=None, rust_timeout=None, yield_on_timeout=False,
                 debug=None, raise_interrupt=None, force_polling=None,
                 poll_delay_ms=300, recursive=True, ignore_permission_denied=None):
    if not paths:
        raise ValueError("At least one watch path is required")
    paths = {os.path.abspath(os.fspath(path)) for path in paths}
    if ignore_permission_denied is None:
        ignore_permission_denied = os.getenv("WATCHFILES_IGNORE_PERMISSION_DENIED", "").lower() in ("1", "true")
    poll_ms = os.getenv("WATCHFILES_POLL_DELAY_MS", "")
    if poll_ms.isdecimal():
        poll_delay_ms = int(poll_ms)
    quiet_period = max(1, int(step)) / 1000
    max_delay = max(1, int(debounce)) / 1000
    interval = min(quiet_period, max(1, int(poll_delay_ms)) / 1000)
    previous = await asyncio.to_thread(_snapshot, paths, recursive, ignore_permission_denied, initial=True)
    pending = set()
    first_change = last_change = time.monotonic()
    last_yield = first_change
    while stop_event is None or not stop_event.is_set():
        await asyncio.sleep(interval)
        if stop_event is not None and stop_event.is_set():
            return
        current = await asyncio.to_thread(_snapshot, paths, recursive, ignore_permission_denied)
        changes = {(Change.added, path) for path in current.keys() - previous.keys()}
        changes.update((Change.deleted, path) for path in previous.keys() - current.keys())
        changes.update((Change.modified, path) for path in current.keys() & previous.keys()
                       if current[path] != previous[path])
        previous = current
        now = time.monotonic()
        if changes:
            if not pending:
                first_change = now
            pending.update(changes)
            last_change = now
        if pending and (now - last_change >= quiet_period or now - first_change >= max_delay):
            filtered = {(change, path) for change, path in pending
                        if watch_filter is None or watch_filter(change, path)}
            pending.clear()
            if filtered:
                last_yield = now
                yield filtered
        elif yield_on_timeout and rust_timeout and now - last_yield >= rust_timeout / 1000:
            last_yield = now
            yield set()
