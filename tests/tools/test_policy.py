from __future__ import annotations

import stat

import pytest

from vela.memory import MemoryManager
from vela.policy.audit_log import AuditLog
from vela.policy.command_guard import CommandGuard, CommandPolicyError
from vela.policy.path_guard import PathGuard, PathPolicyError


def test_path_guard_rejects_escape(tmp_path):
    guard = PathGuard(tmp_path)
    assert guard.validate("inside.txt") == tmp_path / "inside.txt"
    with pytest.raises(PathPolicyError):
        guard.validate("../outside.txt")


def test_command_guard_rejects_destructive_command():
    with pytest.raises(CommandPolicyError):
        CommandGuard().validate("rm -rf /")


def test_audit_log_redacts_secret_values_inside_strings(tmp_path):
    log = AuditLog(tmp_path / "audit" / "audit.jsonl")

    log.record(
        tool_name="bash",
        input_data={
            "command": "curl -H 'Authorization: Bearer abcdef1234567890' https://x",
            "api_key": "plain-secret",
            "notes": ["export OPENAI_TOKEN=sk-abcdefghij1234567890"],
        },
        outcome="approved",
        approver="user",
        cwd=str(tmp_path),
    )

    recorded = log.tail()[0]["input"]
    assert recorded["api_key"] == "***"
    assert "abcdef1234567890" not in recorded["command"]
    assert "sk-abcdefghij1234567890" not in recorded["notes"][0]


def test_audit_log_file_and_directory_are_owner_only(tmp_path):
    path = tmp_path / "audit" / "audit.jsonl"
    log = AuditLog(path)

    log.record(
        tool_name="bash",
        input_data={"command": "ls"},
        outcome="approved",
        approver="user",
        cwd=str(tmp_path),
    )

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_memory_database_is_owner_only(tmp_path):
    db_path = tmp_path / "store" / "memory.db"
    manager = MemoryManager(db_path, scope="project")
    manager.save("remember this")

    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(db_path.parent.stat().st_mode) == 0o700
