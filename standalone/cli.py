"""Distribution commands and dispatch into the original Hermes CLI."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile


def _write_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _doctor(context: dict) -> dict:
    from standalone.variant import excluded_feature_details, manifest_variant

    manifest = context["manifest"]
    variant = manifest_variant(manifest)
    capabilities = manifest["capabilities"]
    target = tuple(int(part) for part in manifest["runtime"]["hermes_python"].split("."))
    python_supported = sys.version_info[:len(target)] >= target
    return {
        "project": manifest["project"],
        "version": manifest["version"],
        "build_variant": dict(variant),
        "excluded_feature_details": excluded_feature_details(variant),
        "stage": capabilities["stage"],
        "upstream": manifest["upstream"],
        "python": platform.python_version(),
        "runtime_policy": manifest["runtime"],
        "isolated": bool(sys.flags.isolated),
        "site_disabled": bool(sys.flags.no_site),
        "bundle_sha256": context["bundle_sha256"],
        "embedded_distributions": len(manifest["distributions"]),
        "embedded_files": len(manifest["files"]),
        "source_transform": {
            key: value for key, value in manifest.get(
                "source_transform", {"mode": "none", "byte_exact": True},
            ).items() if key != "files"
        },
        "python_supported": python_supported,
        "agent_ready": capabilities["agent_runtime_ready"] and python_supported,
        "full_parity_verified": capabilities.get("full_parity_verified", False),
        "features": capabilities["features"],
        "blockers": capabilities["blockers"],
    }


def _print_doctor(report: dict) -> None:
    print(f"Hermes standalone {report['version']} ({report['stage']})")
    print(f"Build: {report['build_variant']['name']}")
    if report["build_variant"]["excluded_features"]:
        for reason in report["excluded_feature_details"].values():
            print(f"  {reason}")
    print(f"Python: {report['python']}; agent target: {report['runtime_policy']['hermes_python']}")
    print(f"Isolated interpreter: {report['isolated']}; site disabled: {report['site_disabled']}")
    print(f"Embedded: {report['embedded_files']} files, {report['embedded_distributions']} distributions")
    print(f"Source transformation: {report['source_transform']['mode']}; "
          f"byte-exact: {report['source_transform']['byte_exact']}")
    print(f"Agent ready: {'yes' if report['agent_ready'] else 'no'}")
    for blocker in report["blockers"]:
        print(f"  {blocker['id']}: {blocker['title']}")


def _probe_agent(context: dict, stage: str, timeout: int) -> dict:
    from standalone.runtime import child_command

    failure = {
        "requested_stage": stage, "status": "failed",
        "agent_imported": False, "agent_constructed": False,
        "agent_turn_verified": False, "fixture_turn_verified": False,
        "original_constructor_verified": False, "live_provider_verified": False,
    }
    try:
        # The child retains its audit guard until exit. Cleanup belongs to this
        # parent, after the child has stopped (including on timeout).
        with tempfile.TemporaryDirectory(prefix="hermes-bundled-agent-probe-") as directory:
            scratch = Path(directory).resolve()
            for name in ("home", "work", "tmp"):
                (scratch / name).mkdir()
            command = child_command(
                context, "standalone.agent_probe_worker",
                ["--stage", stage, "--scratch-root", str(scratch)],
            )
            child = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8",
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        return {**failure, "error": f"Agent probe exceeded {timeout} seconds"}
    except OSError as error:
        return {**failure, "error": f"Could not start agent probe: {error}"}
    try:
        report = json.loads(child.stdout)
        if not isinstance(report, dict) or report.get("status") not in {
            "passed", "failed", "blocked",
        }:
            raise ValueError("invalid diagnostic report")
    except ValueError:
        return {
            **failure, "error": "Agent worker did not return a valid diagnostic report",
            "worker_exit_code": child.returncode,
            "diagnostic_stderr": child.stderr[-3000:],
        }
    if child.returncode != 0 and report["status"] == "passed":
        report.update(status="failed", error="Agent worker failed after reporting success")
    report["worker_exit_code"] = child.returncode
    report["probe_origin"] = "verified_artifact"
    report["artifact_integrity_verified"] = True
    report["bundle_sha256"] = context["bundle_sha256"]
    return report


def _license_report(context: dict, package: str | None) -> int:
    vendor = Path(context["root"]) / "vendor"
    distributions = sorted(importlib.metadata.distributions(path=[str(vendor)]),
                           key=lambda item: item.metadata.get("Name", "").lower())
    if package is None:
        for dist in distributions:
            print(f"{dist.metadata['Name']}=={dist.version}")
        print("Hermes: MIT (app/LICENSE)")
        return 0
    normalized = re.sub(r"[-_.]+", "-", package).lower()
    if normalized in {"hermes", "hermes-agent", "hermes-standalone"}:
        print((Path(context["root"]) / "app" / "LICENSE").read_text(encoding="utf-8"))
        return 0
    for dist in distributions:
        if re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower() != normalized:
            continue
        candidates = []
        for entry in dist.files or ():
            name = str(entry).lower()
            if ".dist-info/" in name and any(word in name for word in ("license", "copying", "notice")):
                candidate = Path(dist.locate_file(entry)).resolve()
                if candidate.is_relative_to(vendor.resolve()) and candidate.is_file():
                    candidates.append(candidate)
        if not candidates:
            print(f"No separate license notice found for {dist.metadata['Name']}.", file=sys.stderr)
            return 1
        for candidate in sorted(set(candidates)):
            print(f"--- {candidate.relative_to(vendor)} ---")
            print(candidate.read_text(encoding="utf-8"))
        return 0
    print(f"Not an embedded distribution: {package}", file=sys.stderr)
    return 2


def _run_original(argv: list[str], context: dict) -> int:
    from standalone.runtime import is_active_bundle

    if not is_active_bundle(Path(context["root"]) / "app"):
        print("Original Hermes commands require the verified standalone runtime.", file=sys.stderr)
        return 2
    target = tuple(int(part) for part in context["manifest"].get("runtime", {}).get("hermes_python", "3.14").split("."))
    if sys.version_info[:len(target)] < target:
        print(f"Hermes commands require Python {'.'.join(map(str, target))} or later; "
              f"this interpreter is {platform.python_version()}.", file=sys.stderr)
        return 2
    previous = sys.argv
    sys.argv = [context["artifact"], *argv]
    try:
        from hermes_cli.main import main as hermes_main

        hermes_main()
    except SystemExit as error:
        if isinstance(error.code, str):
            print(error.code, file=sys.stderr)
            return 1
        return error.code or 0
    finally:
        sys.argv = previous
    return 0


def main(argv: list[str], context: dict) -> int:
    if argv[:1] == ["--"]:
        return _run_original(argv[1:], context)
    if not argv or argv[0] not in {"doctor", "selftest", "probe-agent", "manifest", "licenses"}:
        return _run_original(argv, context)
    parser = argparse.ArgumentParser(
        prog="hermes.py",
        description="Hermes standalone distribution tooling; other commands use the original Hermes CLI.",
        epilog="Cache selection: python hermes.py --cache-dir PATH COMMAND. No runtime dependency installation.",
    )
    parser.add_argument("--version", action="version", version=context["manifest"]["version"])
    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser("doctor", help="Report packaging health and agent portability blockers")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--require-agent", action="store_true",
                        help="Exit with status 2 if the core agent runtime is unavailable")
    selftest = commands.add_parser("selftest", help="Run offline probes against embedded components")
    selftest.add_argument("--json", action="store_true")
    probe = commands.add_parser(
        "probe-agent", help="Exercise original AIAgent offline in a disposable profile",
    )
    probe.add_argument("--stage", choices=("import", "construct", "fixture-turn"), default="import")
    probe.add_argument("--timeout", type=int, default=60, help="Worker timeout in seconds (default: 60)")
    probe.add_argument("--json", action="store_true")
    manifest = commands.add_parser("manifest", help="Show embedded distribution provenance")
    manifest.add_argument("--full", action="store_true", help="Include every embedded file hash")
    licenses = commands.add_parser("licenses", help="List packages or display their embedded license")
    licenses.add_argument("package", nargs="?")
    args = parser.parse_args(argv)
    if args.command == "probe-agent" and args.timeout < 1:
        parser.error("--timeout must be positive")
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "doctor":
        report = _doctor(context)
        if args.json:
            _write_json(report)
        else:
            _print_doctor(report)
        return 2 if args.require_agent and not report["agent_ready"] else 0
    if args.command == "selftest":
        from standalone.selftest import run_checks
        report = run_checks(context)
        if args.json:
            _write_json(report)
        else:
            for check in report["checks"]:
                print(f"{'PASS' if check['ok'] else 'FAIL'} {check['name']}: {check['detail']}")
            print("Scope: offline component and packaging probes; full agent parity is not established.")
        return 0 if report["ok"] else 1
    if args.command == "probe-agent":
        report = _probe_agent(context, args.stage, args.timeout)
        if args.json:
            _write_json(report)
        else:
            print(f"Original AIAgent {args.stage}: {report['status']}")
            if report.get("error"):
                print(report["error"])
            print("Scope: offline diagnostic only; live provider and full agent parity are unverified.")
        return 0 if report["status"] == "passed" else 2
    if args.command == "manifest":
        report = dict(context["manifest"])
        if not args.full:
            report["file_count"] = len(report.pop("files"))
        _write_json(report)
        return 0
    if args.command == "licenses":
        return _license_report(context, args.package)
    return 2
