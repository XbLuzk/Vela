from __future__ import annotations

import stat

import pytest

from vela.memory import MemoryManager
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


def test_memory_database_is_owner_only(tmp_path):
    db_path = tmp_path / "store" / "memory.db"
    manager = MemoryManager(db_path, scope="project")
    manager.save("remember this")

    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(db_path.parent.stat().st_mode) == 0o700
