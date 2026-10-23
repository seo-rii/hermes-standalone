"""Offline behavior probes of embedded upstream components, not agent conformance."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from typing import Callable


@contextmanager
def _temporary_profile():
    with tempfile.TemporaryDirectory(prefix="hermes-standalone-selftest-") as temporary:
        previous = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = temporary
        try:
            yield Path(temporary)
        finally:
            if previous is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = previous


def _yaml() -> str:
    from hermes_yaml import safe_dump, safe_load
    from ruamel.yaml.constructor import DuplicateKeyError

    expected = {"enabled": True, "label": "off", "unicode": "헤르메스"}
    if safe_load(safe_dump(expected)) != expected:
        raise AssertionError("YAML round-trip changed values")
    if safe_load("enabled: yes\n") != {"enabled": True}:
        raise AssertionError("Upstream YAML 1.1 boolean semantics changed")
    try:
        safe_load("key: first\nkey: second\n")
    except DuplicateKeyError:
        return "Original YAML policy, Unicode round-trip and duplicate-key rejection"
    raise AssertionError("Duplicate YAML keys were accepted")


def _output_safety() -> str:
    from tools.ansi_strip import sanitize_display_text, strip_ansi

    if strip_ansi("\x1b[31mtext\x1b[0m") != "text":
        raise AssertionError("ANSI sequence retained")
    if sanitize_display_text("\x1b]0;title\x07safe\x07") != "safe":
        raise AssertionError("Terminal control sequence retained")
    return "Original Hermes terminal-output sanitization"


def _budget() -> str:
    from agent.iteration_budget import IterationBudget

    budget = IterationBudget(2)
    if [budget.consume() for _ in range(3)] != [True, True, False]:
        raise AssertionError("Iteration limit changed")
    budget.refund()
    if budget.remaining != 1 or not budget.consume() or budget.remaining != 0:
        raise AssertionError("Iteration refund changed")
    return "Original iteration limit and refund behavior"


def _todo() -> str:
    from tools.todo_tool import TodoStore

    store = TodoStore()
    store.write([{"id": "one", "content": "Prepare bundle", "status": "pending"}])
    snapshot = store.snapshot()
    other = TodoStore()
    other.restore(snapshot["todos"], revision=snapshot["revision"])
    if other.snapshot() != snapshot:
        raise AssertionError("Todo snapshot did not round-trip")
    if "Prepare bundle" not in (other.format_for_injection() or ""):
        raise AssertionError("Todo context restoration lost content")
    other.write([{"id": "one", "status": "completed"}], merge=True)
    if other.snapshot()["revision"] <= snapshot["revision"]:
        raise AssertionError("Todo update did not advance its revision")
    return "Original revisioned task store, hydration and context restoration"


def _profiles() -> str:
    from hermes_constants import (
        get_hermes_home, reset_hermes_home_override, set_hermes_home_override,
    )

    with tempfile.TemporaryDirectory(prefix="hermes-standalone-profiles-") as temporary:
        first = Path(temporary) / "first"
        second = Path(temporary) / "second"
        first_token = set_hermes_home_override(first)
        try:
            if get_hermes_home() != first:
                raise AssertionError("First profile override was ignored")
            second_token = set_hermes_home_override(second)
            try:
                if get_hermes_home() != second:
                    raise AssertionError("Nested profile override was ignored")
            finally:
                reset_hermes_home_override(second_token)
            if get_hermes_home() != first:
                raise AssertionError("Profile scope did not restore")
        finally:
            reset_hermes_home_override(first_token)
    return "Original context-local profile resolution; no user profile opened"


def _transport_types() -> str:
    from agent.transports.types import NormalizedResponse, build_tool_call

    call = build_tool_call("call-1", "todo", {"value": "한글"}, call_id="provider-1")
    response = NormalizedResponse(None, [call], "tool_calls")
    if response.tool_calls[0].function.name != "todo":
        raise AssertionError("Tool accessor changed")
    if json.loads(call.arguments) != {"value": "한글"} or call.call_id != "provider-1":
        raise AssertionError("Tool-call arguments or provider identity lost")
    return "Original normalized response and tool-call types; no provider called"


def _sqlite() -> str:
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        connection.execute("INSERT INTO sample VALUES (?)", ("헤르메스",))
        if connection.execute("SELECT value FROM sample").fetchone() != ("헤르메스",):
            raise AssertionError("SQLite Unicode round-trip failed")
        try:
            connection.execute("CREATE VIRTUAL TABLE search USING fts5(text)")
        except sqlite3.OperationalError:
            return "Standard-library SQLite works; FTS5 is unavailable on this interpreter"
    return "Standard-library SQLite and FTS5 are available"


def _session_store() -> str:
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory(prefix="hermes-standalone-session-") as temporary:
        path = Path(temporary) / "state.db"
        session_id = "standalone-offline-probe"
        database = SessionDB(path)
        try:
            database.create_session(session_id, "cli")
            database.append_message(session_id, "user", "헤르메스 세션")
            database.append_message(session_id, "assistant", "Saved response")
            identifiers = [message["message_uid"] for message in database.get_messages(session_id)]
        finally:
            database.close()
        database = SessionDB(path)
        try:
            messages = database.get_messages(session_id)
            if [(item["role"], item["content"]) for item in messages] != [
                ("user", "헤르메스 세션"), ("assistant", "Saved response"),
            ]:
                raise AssertionError("Session messages changed after reopening the original store")
            if [message["message_uid"] for message in messages] != identifiers:
                raise AssertionError("Durable message identities changed after reopening")
        finally:
            database.close()
    return "Original SessionDB schema, Unicode transcript and message identities survive reopening"


def _http_transport() -> str:
    import httpx

    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"path": request.url.path}),
    )) as client:
        response = client.get("https://standalone.invalid/offline")
        if response.json() != {"path": "/offline"}:
            raise AssertionError("Embedded transport failed")
    return f"Embedded HTTPX {importlib.metadata.version('httpx')}, in-memory transport only"


def _resources(context: dict) -> str:
    root = Path(context["root"])
    if not (root / "app" / "LICENSE").is_file():
        raise AssertionError("Upstream license is absent")
    skills = list((root / "app" / "skills").rglob("SKILL.md"))
    if not skills:
        raise AssertionError("Bundled skill resources are absent")
    return f"{len(skills)} bundled skill documents accessible by ordinary paths"


def _child(context: dict) -> str:
    from standalone.runtime import child_command

    result = subprocess.run(
        child_command(context, "standalone.selftest_worker"),
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
    )
    if result.returncode:
        raise AssertionError(f"Internal worker exited {result.returncode}: {result.stderr[:500]}")
    report = json.loads(result.stdout)
    if not all(report[key] for key in ("isolated", "site_disabled", "yaml", "upstream_bootstrap")):
        raise AssertionError("Worker failed isolation or resource checks")
    if report["pm_loaded"]:
        raise AssertionError("Upstream bootstrap activated a package manager inside the bundle")
    if report["ansi"] != "worker":
        raise AssertionError("Worker could not use upstream source")
    return "Fresh isolated interpreter entered the original bootstrap without PM, networking or installers"


def _isolation(context: dict) -> str:
    if not sys.flags.isolated or not sys.flags.no_site:
        raise AssertionError("Interpreter is not running with -I -S")
    cache = Path(context["root"]).resolve()
    leaked = []
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename)
        if ("site-packages" in path.parts or "dist-packages" in path.parts) and not path.resolve().is_relative_to(cache):
            leaked.append(str(path))
    if leaked:
        raise AssertionError(f"External third-party imports: {leaked[:3]}")
    return "No imports from an external site-packages or dist-packages directory"


def run_checks(context: dict) -> dict:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("upstream-yaml", _yaml),
        ("upstream-output-safety", _output_safety),
        ("upstream-iteration-budget", _budget),
        ("upstream-todo-store", _todo),
        ("upstream-profile-scope", _profiles),
        ("upstream-transport-types", _transport_types),
        ("stdlib-sqlite", _sqlite),
        ("upstream-session-store", _session_store),
        ("embedded-http-transport", _http_transport),
        ("embedded-resources", lambda: _resources(context)),
        ("internal-worker", lambda: _child(context)),
        ("import-isolation", lambda: _isolation(context)),
    ]
    results = []
    with _temporary_profile():
        for name, check in checks:
            try:
                detail = check()
            except Exception as exc:
                results.append({"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
            else:
                results.append({"name": name, "ok": True, "detail": detail})
    return {
        "ok": all(item["ok"] for item in results),
        "scope": "offline packaging and selected upstream behavior probes",
        "agent_parity_verified": False,
        "python": sys.version.split()[0],
        "checks": results,
    }
