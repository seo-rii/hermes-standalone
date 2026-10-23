"""Internal offline probe for the original AIAgent, never a provider implementation."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import sys
import sysconfig
import traceback
from types import SimpleNamespace


_SECRET_FILES = {".env", "auth.json", "credentials.json", "credentials", "secrets.json"}
_PROCESS_EVENTS = {
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
    "os.fork", "os.forkpty", "os.spawn",
}
_MUTATION_EVENTS = {
    "os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.chown", "os.truncate",
    "os.utime", "shutil.rmtree",
}
_TWO_PATH_MUTATIONS = {"os.rename", "os.link", "os.symlink"}
_HOST_FACT_FILES = (
    "/proc/version", "/proc/1/cgroup", "/proc/self/cgroup", "/proc/self/mountinfo", "/proc/self/status",
    "/proc/stat", "/proc/meminfo", "/etc/os-release", "/usr/lib/os-release", "/etc/mime.types",
    "/proc/sys/kernel/random/boot_id",
    "/sys/fs/cgroup/memory.high", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)
_POWER_FACT_NAMES = frozenset({
    "type", "present", "capacity", "status", "online", "energy_now", "energy_full",
    "power_now", "charge_now", "charge_full", "current_now", "time_to_empty_now",
})


class ProbeBlocked(BaseException):
    """The diagnostic blocks side effects even when upstream catches Exception."""


class ProbeGuard:
    """A process-local policy; the child process exits instead of removing it."""

    def __init__(self, source: Path, vendor: Path, scratch: Path):
        self.source = source
        self.vendor = vendor
        self.scratch = scratch
        self.writable = (scratch,)
        self.readable = (
            source, vendor, scratch,
            Path(sysconfig.get_path("stdlib")).resolve(),
            Path(sysconfig.get_path("platstdlib")).resolve(),
            *(Path(entry).resolve() for entry in sys.path if entry),
        )
        self.events: list[dict] = []
        self.host_fact_files = {Path(path).resolve() for path in _HOST_FACT_FILES}
        cgroup_root = Path("/sys/fs/cgroup").resolve()
        try:
            groups = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
        except OSError:
            groups = ()
        for entry in groups:
            if entry.startswith("0::"):
                for name in ("memory.max", "memory.high", "memory.current"):
                    counter = (cgroup_root / entry[3:].lstrip("/") / name).resolve()
                    if counter.is_relative_to(cgroup_root):
                        self.host_fact_files.add(counter)
        # The kernel's public class symlinks may resolve into /sys/devices.
        # Bind exact known files now; a later symlink swap does not widen access.
        power_root = Path("/sys/class/power_supply")
        for supply in power_root.glob("*"):
            resolved_supply = supply.resolve()
            if (not resolved_supply.is_relative_to(Path("/sys/devices"))
                    or resolved_supply.parent.name != "power_supply" or not resolved_supply.is_dir()):
                continue
            for name in _POWER_FACT_NAMES:
                fact = resolved_supply / name
                if fact.resolve() == fact:
                    self.host_fact_files.add(fact)

    def deny(
        self, event: str, operation: str, path: Path | None = None, *, recoverable: bool = False,
    ) -> None:
        item = {"event": event, "operation": operation}
        if recoverable:
            item["recoverable_capability_probe"] = True
        if path is not None:
            if path.is_relative_to(self.source):
                item["path"] = "source/" + path.relative_to(self.source).as_posix()
            elif path.is_relative_to(self.scratch):
                item["path"] = "fixture/" + path.relative_to(self.scratch).as_posix()
            else:
                item["path"] = "<outside>/" + path.name
        self.events.append(item)
        if recoverable:
            # urllib3 probes IPv6 support with a loopback bind at import time.
            # Fail that capability check normally, without actually binding.
            raise OSError(f"Offline probe blocked {operation} ({event})")
        raise ProbeBlocked(f"Offline probe blocked {operation} ({event})")

    def check_path(self, event: str, value: object, *, write: bool) -> None:
        if isinstance(value, int) or value is None:
            return
        path = Path(os.fsdecode(value)).absolute()
        # resolve() only inspects links; access decisions use their final target.
        path = path.resolve()
        if path == Path(os.devnull).resolve():
            return
        if write and any(path.is_relative_to(root) for root in (self.source, self.vendor)):
            self.deny(event, "bundled_source_write", path)
        if (
            not write and {"site-packages", "dist-packages"}.intersection(path.parts)
            and not path.is_relative_to(self.vendor)
        ):
            self.deny(event, "external_package_read", path)
        if not write and path in self.host_fact_files:
            return
        # Worker admission reads kernel birth, parent and UID, never process
        # environment or command lines. These files contain no credentials.
        if not write and re.fullmatch(r"/proc/[0-9]+/(?:stat|status|statm|comm)", str(path)):
            return
        process_read = re.fullmatch(r"/proc/([0-9]+)/(?:cmdline|environ|net/(?:tcp6?|udp6?))", str(path))
        if not write and process_read:
            if int(process_read[1]) == os.getpid():
                return
            # Optional ancestry/ownership probes must take their normal denied
            # path. Process arguments and environments may contain credentials.
            raise PermissionError("The offline fixture does not inspect foreign process arguments or sockets")
        allowed = self.writable if write else self.readable
        if not any(path.is_relative_to(root) for root in allowed):
            self.deny(event, "write_outside_fixture" if write else "read_outside_fixture", path)
        if not write and path.is_relative_to(self.source) and (
            path.name in _SECRET_FILES or ".git" in path.relative_to(self.source).parts
        ):
            self.deny(event, "source_configuration_read", path)

    def __call__(self, event: str, args: tuple) -> None:
        if event.startswith("socket.") and event not in {"socket.__new__", "socket.gethostname"}:
            self.deny(event, "network", recoverable=event == "socket.bind")
        if event in _PROCESS_EVENTS:
            self.deny(event, "process_execution")
        if event == "import" and args[0].split(".", 1)[0] in {"pip", "ensurepip", "uv"}:
            self.deny(event, "package_installer_import")
        if event == "open":
            mode = args[1] or ""
            flags = args[2] or 0
            write = any(character in mode for character in "wax+") or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
            self.check_path(event, args[0], write=write)
        elif event in _MUTATION_EVENTS:
            self.check_path(event, args[0], write=True)
        elif event in {"shutil.copyfile", "shutil.copymode", "shutil.copystat"}:
            self.check_path(event, args[0], write=False)
            self.check_path(event, args[1], write=True)
        elif event in _TWO_PATH_MUTATIONS:
            for path in args[:2]:
                self.check_path(event, path, write=True)


def _display_frame(frame: traceback.FrameSummary, source: Path, vendor: Path) -> dict:
    path = Path(frame.filename)
    if path.is_relative_to(source):
        filename = "source/" + path.relative_to(source).as_posix()
    elif path.is_relative_to(vendor):
        filename = "vendor/" + path.relative_to(vendor).as_posix()
    else:
        filename = path.name
    return {"path": filename, "line": frame.lineno, "function": frame.name}


def fixture_environment(scratch: Path) -> dict[str, str]:
    """Only empty disposable profile paths and Windows runtime locations survive."""
    home = scratch / "home"
    environment = {
        "HOME": str(home), "USERPROFILE": str(home),
        "HERMES_HOME": str(home / "hermes"),
        "HERMES_RUNTIME_DIR": str(home / "runtime"),
        "XDG_CONFIG_HOME": str(home / "config"), "XDG_CACHE_HOME": str(home / "cache"),
        "XDG_DATA_HOME": str(home / "data"), "APPDATA": str(home / "appdata"),
        "LOCALAPPDATA": str(home / "localappdata"),
        "TMPDIR": str(scratch / "tmp"), "TMP": str(scratch / "tmp"), "TEMP": str(scratch / "tmp"),
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    }
    for name in ("SYSTEMROOT", "WINDIR"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def _agent_arguments(scratch: Path) -> dict:
    return {
        "model": "standalone-offline-fixture",
        "provider": "standalone-offline-fixture",
        "api_mode": "chat_completions",
        "base_url": "https://standalone.invalid/v1",
        "api_key": "offline-fixture-not-a-credential",
        "enabled_toolsets": [],
        "skip_context_files": True,
        "skip_memory": True,
        "skip_background_review": True,
        "quiet_mode": True,
        "save_trajectories": False,
        "max_iterations": 3,
        "cwd": str(scratch / "work"),
    }


def _capture_error(error: BaseException, source: Path, vendor: Path) -> dict:
    result = {
        "status": "blocked" if isinstance(error, ProbeBlocked) else "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "frames": [
            _display_frame(frame, source, vendor)
            for frame in traceback.StackSummary.extract(
                traceback.walk_tb(error.__traceback__), lookup_lines=False,
            )[-24:]
        ],
    }
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModuleNotFoundError):
            result["missing_module"] = current.name
            break
        current = current.__cause__ or current.__context__
    return result


def _prepare_fixture_profile(scratch: Path) -> None:
    """Use upstream configuration, scoped to the temporary offline fixture."""
    home = scratch / "home" / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    config = {
        "model": {
            "default": "standalone-offline-fixture",
            "provider": "standalone-offline-fixture",
            "base_url": "https://standalone.invalid/v1",
            "context_length": 131072,
        },
        "agent": {
            "environment_probe": False, "api_max_retries": 1, "auto_recovery_cycles": 0,
        },
        "compression": {"enabled": False},
        "tools": {"tool_search": {"enabled": False}},
    }
    # JSON is a YAML subset and needs no YAML dependency in this diagnostic.
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")


def probe(source: Path, vendor: Path, scratch: Path, stage: str) -> dict:
    """Run once in a disposable child; the caller binds the verified runtime first."""
    if stage not in ("import", "construct", "fixture-turn"):
        raise ValueError("unknown agent probe stage")
    source, vendor, scratch = (path.resolve() for path in (source, vendor, scratch))
    os.chdir(scratch / "work")
    sys.path[:0] = [str(source), str(vendor)]
    environment = fixture_environment(scratch)
    os.environ.clear()
    os.environ.update(environment)
    from standalone.runtime import is_active_bundle

    guard = ProbeGuard(source, vendor, scratch)
    sys.addaudithook(guard)
    result = {
        "schema_version": 1,
        "requested_stage": stage,
        "python": sys.version.split()[0],
        "isolated": bool(sys.flags.isolated and sys.flags.no_site),
        "agent_imported": False,
        "agent_constructed": False,
        "agent_turn_verified": False,
        "fixture_turn_verified": False,
        "original_constructor_verified": False,
        "live_provider_verified": False,
        "network_allowed": False,
        "process_execution_allowed": False,
        "installer_allowed": False,
        "runtime_binding": "bundle" if is_active_bundle(source) else "upstream",
    }
    output, errors = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            if stage in {"construct", "fixture-turn"}:
                _prepare_fixture_profile(scratch)
            module = importlib.import_module("run_agent")
            result["agent_imported"] = True
            if stage == "construct":
                agent = module.AIAgent(**_agent_arguments(scratch))
                result["agent_constructed"] = True
                result["original_constructor_verified"] = True
                agent.close()
            elif stage == "fixture-turn":
                _fixture_turn(module, scratch, result)
        result["status"] = "passed"
    except BaseException as error:
        result.update(_capture_error(error, source, vendor))
    result["blocked_events"] = guard.events
    if result["status"] == "passed" and any(
        not event.get("recoverable_capability_probe") for event in guard.events
    ):
        result.update({
            "status": "blocked", "error_type": "ProbeBlocked",
            "error": "An upstream background operation was blocked by the offline probe",
        })
    result["diagnostic_stdout"] = output.getvalue()[-2000:]
    result["diagnostic_stderr"] = errors.getvalue()[-3000:]
    return result


def _fixture_turn(module, scratch: Path, result: dict) -> None:
    """An instance-local test client exercises upstream code, never live providers."""
    calls: list[list[dict]] = []
    completed_todo = {"id": "offline-probe", "content": "Verify original tool execution", "status": "completed"}

    def respond(**kwargs):
        messages = kwargs["messages"]
        calls.append(messages)
        if len(calls) == 1:
            declared = {tool["function"]["name"] for tool in kwargs.get("tools", [])}
            if "todo_list" not in declared:
                raise AssertionError("The original agent did not expose the requested todo tool")
            tool = SimpleNamespace(
                id="offline-probe-tool", type="function",
                function=SimpleNamespace(name="todo_list", arguments=json.dumps({"todos": [completed_todo]})),
            )
            message = SimpleNamespace(role="assistant", content=None, tool_calls=[tool])
            reason = "tool_calls"
        elif len(calls) == 2:
            matches = [item for item in messages if item.get("role") == "tool"
                       and item.get("tool_call_id") == "offline-probe-tool"]
            if len(matches) != 1 or json.loads(matches[0]["content"]).get("todos") != [completed_todo]:
                raise AssertionError("The original todo result did not reach the next model request")
            result["fixture_tool_result_verified"] = True
            message = SimpleNamespace(role="assistant", content="offline fixture complete", tool_calls=None)
            reason = "stop"
        else:
            raise AssertionError("The offline fixture allows exactly two model responses")
        return SimpleNamespace(
            id=f"offline-response-{len(calls)}", model="standalone-offline-fixture",
            choices=[SimpleNamespace(message=message, finish_reason=reason)], usage=None,
        )

    class FixtureClient:
        def __init__(self):
            self.base_url = "https://standalone.invalid/v1"
            self.is_closed = False
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=respond))

        def close(self):
            self.is_closed = True

    class FixtureAgent(module.AIAgent):
        def _create_openai_client(self, client_kwargs, *, reason, shared):
            return FixtureClient()

    arguments = _agent_arguments(scratch)
    arguments["enabled_toolsets"] = ["todo"]
    agent = FixtureAgent(**arguments)
    agent._disable_streaming = True
    result["agent_constructed"] = True
    result["fixture_client_injected"] = True
    try:
        conversation = agent.run_conversation(
            "Record the offline probe task as completed, then confirm completion.",
            system_message="Use the todo_list tool once and then report completion.",
        )
        if conversation.get("final_response") != "offline fixture complete" or len(calls) != 2:
            raise AssertionError("The original conversation did not finish the two-response fixture")
        roles = [item.get("role") for item in conversation.get("messages", [])]
        if roles != ["user", "assistant", "tool", "assistant"]:
            raise AssertionError("The original conversation did not preserve tool-result message ordering")
        if agent._todo_store.read() != [completed_todo]:
            raise AssertionError("The original todo store did not retain the fixture's completed task")
        result["fixture_turn_verified"] = True
        result["fixture_todo_state_verified"] = True
        result["fixture_model_calls"] = len(calls)
        result["fixture_tool"] = "todo_list"
        result["fixture_conversation_roles"] = roles
    finally:
        agent.close()


def _validate_scratch(value: Path | None, source: Path) -> Path:
    """Require the parent's empty, disposable profile layout before any imports."""
    if value is None or not value.is_absolute() or value.is_symlink():
        raise ValueError("--scratch-root must name the parent's absolute temporary directory")
    scratch = value.resolve()
    protected_roots = (source, source.parent / "vendor")
    if any(scratch.is_relative_to(root) or root.is_relative_to(scratch) for root in protected_roots):
        raise ValueError("the temporary profile must be separate from bundled sources")
    if not scratch.is_dir() or {path.name for path in scratch.iterdir()} != {"home", "work", "tmp"}:
        raise ValueError("the temporary profile must contain exactly home, work and tmp directories")
    if hasattr(os, "getuid") and scratch.stat().st_uid != os.getuid():
        raise ValueError("the temporary profile must belong to the current user")
    for name in ("home", "work", "tmp"):
        child = scratch / name
        if child.is_symlink() or not child.is_dir() or any(child.iterdir()):
            raise ValueError("temporary home, work and tmp directories must be real and empty")
    return scratch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("import", "construct", "fixture-turn"), default="import")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parents[1]
    from standalone.runtime import is_active_bundle

    if not is_active_bundle(source):
        parser.error("agent probes require the verified standalone bundle runtime")
    try:
        scratch = _validate_scratch(args.scratch_root, source)
    except ValueError as error:
        parser.error(str(error))
    report = probe(source, source.parent / "vendor", scratch, args.stage)
    print(json.dumps(report, sort_keys=True, indent=2), flush=True)
    # The parent cleans this profile after completion or timeout. Keep the guard
    # active until exit; no upstream atexit hooks or worker-side rmtree can escape it.
    os._exit(0 if report["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
