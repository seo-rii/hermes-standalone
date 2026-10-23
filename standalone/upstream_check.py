#!/usr/bin/env python3
"""Run the canonical upstream test runner against an isolated source snapshot.

Runtime and test distributions must already exist in the verified download
cache. Nothing is installed, and the developer's HOME or Hermes state is never
passed to the test process. Test dependencies are not included in hermes.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT = Path(__file__).resolve().parents[1]


_NETWORK_GUARD = '''import ipaddress, sys
def _loopback(host):
    if isinstance(host, bytes):
        host = host.decode("ascii")
    if host is None or str(host).rstrip(".").lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or bool(getattr(address, "ipv4_mapped", None) and address.ipv4_mapped.is_loopback)
def _network_guard(event, args):
    if event == "socket.getaddrinfo" and not _loopback(args[0]):
        raise PermissionError("Isolated upstream checks allow only loopback DNS")
    if event == "socket.connect" and isinstance(args[1], tuple) and not _loopback(args[1][0]):
        raise PermissionError("Isolated upstream checks allow only loopback connections")
sys.addaudithook(_network_guard)
'''

_LAUNCHER = _NETWORK_GUARD + '''import json, os, pathlib, runpy, sys
context = json.loads(pathlib.Path(__file__).with_name("context.json").read_text())
sys.path[:0] = context["paths"]
sys.executable = str(pathlib.Path(__file__).with_name("python"))
from standalone.runtime import initialize_bundle
initialize_bundle(context["bundle"])
args = sys.argv[1:]
while args and args[0] in ("-u", "-B", "-I", "-S"):
    args.pop(0)
if not args:
    raise SystemExit("The isolated test interpreter requires -c, -m or a script")
if args[0] == "-c":
    sys.argv = ["-c", *args[2:]]
    exec(compile(args[1], "<string>", "exec"), {"__name__": "__main__"})
elif args[0] == "-m":
    sys.argv = [args[1], *args[2:]]
    runpy.run_module(args[1], run_name="__main__", alter_sys=True)
else:
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=PROJECT / "standalone/.cache/downloads")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for scripts/run_tests.sh")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 14):
        parser.error("The pinned upstream test target requires Python 3.14 or later")
    if os.name != "posix":
        parser.error("This wrapper requires the canonical Bash test runner")
    sys.path.insert(0, str(PROJECT))
    from standalone import offline_agent_probe, vendor

    with tempfile.TemporaryDirectory(prefix="hermes-upstream-tests-") as directory:
        scratch = Path(directory)
        source = scratch / "source"
        source.mkdir()
        _, digest = offline_agent_probe._source_snapshot(PROJECT, source)
        bundle = scratch / digest
        bundle.mkdir()
        source.rename(bundle / "app")
        source = bundle / "app"
        for name in ("tests", "scripts"):
            shutil.copytree(PROJECT / name, source / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules"))
        for name, manifest in (("vendor", "dependencies.json"), ("test-vendor", "test-dependencies.json")):
            paths = vendor.ArchivePaths()
            for package in vendor.load_dependencies(PROJECT / "standalone" / manifest)["packages"]:
                files, _ = vendor.read_wheel(package, args.downloads)
                for path, content in files.items():
                    paths.add(path)
                    target = bundle / name / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
        runtime = scratch / "interpreter"
        runtime.mkdir()
        context = {"paths": [str(source), str(bundle / "test-vendor"), str(bundle / "vendor")],
                   "bundle": {"root": str(bundle), "artifact": str(scratch / "fixture.py"),
                              "bundle_sha256": digest, "manifest": {"project": "hermes-standalone"}}}
        (runtime / "context.json").write_text(json.dumps(context), encoding="utf-8")
        (runtime / "launch.py").write_text(_LAUNCHER, encoding="utf-8")
        import shlex
        wrapper = runtime / "python"
        wrapper.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " -I -S -B "
                           + shlex.quote(str(runtime / "launch.py")) + ' "$@"\n', encoding="utf-8")
        wrapper.chmod(0o700)
        for name in ("home", "tmp"):
            (scratch / name).mkdir()
        env = {"PATH": os.defpath, "HOME": str(scratch / "home"), "TMPDIR": str(scratch / "tmp"),
               "HERMES_HOME": str(scratch / "home/hermes"), "HERMES_PYTHON": str(wrapper),
               "TZ": "UTC", "LANG": "C.UTF-8", "PYTHONUTF8": "1", "PYTHONHASHSEED": "0"}
        forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
        print(json.dumps({"source_snapshot": digest, "runtime_distributions": "reviewed cache",
                          "runner": "scripts/run_tests.sh", "python": sys.version.split()[0]}), flush=True)
        result = subprocess.run(["bash", str(source / "scripts/run_tests.sh"), *forwarded],
                                cwd=source, env=env)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
