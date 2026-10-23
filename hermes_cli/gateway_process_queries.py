"""Process queries shared by the gateway command facade."""

from hermes_platform.processes import process_api


def _get_parent_pid(pid: int) -> int | None:
    """Parent PID for ``pid``, or None. psutil first (works on Windows, where ``ps`` doesn't)."""
    from hermes_cli.gateway import is_windows, shutil, subprocess, _CAPTURE_TEXT

    if pid <= 1:
        return None
    try:
        psutil = process_api()
        return psutil.Process(pid).ppid() or None
    except ImportError:
        pass
    except Exception:
        return None
    # ps fallback, POSIX only: Git Bash's ps.exe would flash a console from the windowless backend.
    if is_windows() or not shutil.which("ps"):
        return None
    try:
        result = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], timeout=5, **_CAPTURE_TEXT, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    raw = result.stdout.strip()
    if result.returncode != 0 or not raw:
        return None
    try:
        parent_pid = int(raw.splitlines()[-1].strip())
    except ValueError:
        return None
    return parent_pid if parent_pid > 0 else None
