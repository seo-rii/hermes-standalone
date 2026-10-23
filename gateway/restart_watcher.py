"""Detached Windows gateway restart watcher, runnable from source or a bundle."""

import os
import subprocess
import sys
import time


def _alive(pid: int) -> bool:
    # Windows os.kill(pid, 0) sends a console event. Keep the original
    # handle-based check and its conservative treatment of access failures.
    if os.name == "nt":
        import ctypes

        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.restype = ctypes.c_uint
        kernel.GetLastError.restype = ctypes.c_uint
        handle = kernel.OpenProcess(0x1000 | 0x100000, False, int(pid))
        if not handle:
            return kernel.GetLastError() != 87
        try:
            return kernel.WaitForSingleObject(handle, 0) == 0x102
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def main() -> None:
    from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway

    pid = int(sys.argv[1])
    deadline = time.monotonic() + float(sys.argv[2])
    while time.monotonic() < deadline:
        if not _alive(pid):
            break
        time.sleep(0.2)
    subprocess.Popen(
        sys.argv[3:], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=windows_detach_flags_without_breakaway(),
    )


if __name__ == "__main__":
    main()
