"""Publish the existing Python CLI's active session to the dashboard PTY owner."""

import os


def write_active_session(session_id):
    from agent.sdk_boundary import bundled

    marker = os.environ.get("HERMES_TUI_ACTIVE_SESSION_FILE")
    if not marker or not session_id or not bundled():
        return
    from utils import atomic_json_write

    atomic_json_write(marker, {"session_id": session_id}, mode=0o600)
