#!/usr/bin/env python3
"""Probe the original AIAgent startup offline in a disposable, isolated process.

This is a diagnostic, not an SDK substitute or an agent implementation. It never
fetches packages. A blocked or failed import is reported as an unfinished port.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def _load_helper(name: str):
    spec = importlib.util.spec_from_file_location(
        "_offline_probe_" + name, Path(__file__).with_name(name + ".py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vendor_sources(root: Path, downloads: Path, destination: Path) -> int:
    vendor = _load_helper("vendor")
    manifest = vendor.load_dependencies(root / "standalone" / "dependencies.json")
    paths = vendor.ArchivePaths()
    for package in manifest["packages"]:
        files, _ = vendor.read_wheel(package, downloads)
        for name, content in files.items():
            paths.add(name)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return len(manifest["packages"])


def _source_snapshot(root: Path, destination: Path) -> tuple[int, str]:
    files = _load_helper("build").collect_sources(root)
    digest = hashlib.sha256()
    for name, content in sorted(files.items()):
        target = destination / name.removeprefix("app/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(content).digest())
    return len(files), digest.hexdigest()


def run_probe(
    root: Path, downloads: Path, *, stage: str = "import", timeout: float = 60,
    bootstrap_mode: str = "bundle",
) -> dict:
    root, downloads = root.resolve(), downloads.resolve()
    with tempfile.TemporaryDirectory(prefix="hermes-offline-agent-") as directory:
        scratch = Path(directory).resolve()
        for name in ("app", "vendor", "home", "work", "tmp"):
            (scratch / name).mkdir()
        count = _vendor_sources(root, downloads, scratch / "vendor")
        source_count, digest = _source_snapshot(root, scratch / "app")
        snapshot = scratch / digest
        snapshot.mkdir()
        (scratch / "app").rename(snapshot / "app")
        (scratch / "vendor").rename(snapshot / "vendor")
        environment = _load_helper("agent_probe_worker").fixture_environment(scratch)
        command = [
            sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
            "--worker", "--root", str(snapshot / "app"), "--vendor-root", str(snapshot / "vendor"),
            "--scratch-root", str(scratch), "--stage", stage,
            "--bootstrap-mode", bootstrap_mode,
        ]
        try:
            process = subprocess.run(
                command, env=environment, cwd=scratch / "work",
                capture_output=True, text=True, encoding="utf-8", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"schema_version": 1, "status": "timed_out", "timeout_seconds": timeout}
        try:
            report = json.loads(process.stdout)
        except ValueError:
            return {
                "schema_version": 1, "status": "worker_failed", "exit_code": process.returncode,
                "stderr": process.stderr[-3000:], "stdout": process.stdout[-1000:],
            }
        report.update({
            "probe_origin": "disposable_source_snapshot",
            "artifact_integrity_verified": False,
            "verified_distributions": count,
            "source_files": source_count,
            "source_snapshot_sha256": digest,
            "exit_code": process.returncode,
            "stderr": process.stderr[-3000:],
        })
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--stage", choices=("import", "construct", "fixture-turn"), default="import")
    parser.add_argument("--bootstrap-mode", choices=("bundle", "upstream"), default="bundle")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vendor-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--scratch-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.vendor_root is None or args.scratch_root is None:
            parser.error("worker needs vendor and scratch roots")
        sys.path[:0] = [str(args.root), str(args.vendor_root)]
        if args.bootstrap_mode == "bundle":
            from standalone.runtime import initialize_bundle
            initialize_bundle({
                "root": str(args.root.parent), "artifact": str(Path(__file__).resolve()),
                "bundle_sha256": args.root.parent.name,
                "manifest": {"project": "hermes-standalone"},
            })
        from standalone.agent_probe_worker import probe
        report = probe(
            args.root, args.vendor_root, args.scratch_root, args.stage,
        )
        print(json.dumps(report, sort_keys=True, indent=2), flush=True)
        # Do not run upstream atexit cleanup outside this bounded diagnostic.
        os._exit(0 if report["status"] == "passed" else 2)
    downloads = args.cache_dir or args.root / "standalone" / ".cache" / "downloads"
    report = run_probe(
        args.root, downloads, stage=args.stage, timeout=args.timeout,
        bootstrap_mode=args.bootstrap_mode,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
