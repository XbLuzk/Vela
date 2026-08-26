from __future__ import annotations

import os
import sqlite3

import pytest

from vela.storage import (
    PRIVATE_DIR_MODE,
    PRIVATE_FILE_MODE,
    apply_sqlite_pragmas,
    ensure_private_dir,
    ensure_private_file,
    exclusive_lock,
    set_private_mode,
    user_state_path,
    vela_dir,
    write_private_text,
)


def _mode(path) -> int:
    return path.stat().st_mode & 0o777


def test_ensure_private_dir_creates_owner_only_tree(tmp_path):
    directory = ensure_private_dir(tmp_path / "nested" / "state", verify=True)

    assert directory.is_dir()
    assert _mode(directory) == PRIVATE_DIR_MODE


def test_ensure_private_file_creates_file_and_parent(tmp_path):
    target = ensure_private_file(tmp_path / "state" / "db.sqlite", verify=True)

    assert target.is_file()
    assert _mode(target) == PRIVATE_FILE_MODE
    assert _mode(target.parent) == PRIVATE_DIR_MODE


def test_ensure_private_file_keeps_existing_content(tmp_path):
    target = tmp_path / "data.jsonl"
    target.write_text("keep\n", encoding="utf-8")

    ensure_private_file(target)

    assert target.read_text(encoding="utf-8") == "keep\n"


def test_set_private_mode_verify_reports_rejected_chmod(tmp_path, monkeypatch):
    target = tmp_path / "file"
    target.touch()
    monkeypatch.setattr("vela.storage.os.chmod", lambda *_args: None)

    with pytest.raises(PermissionError):
        set_private_mode(target, PRIVATE_FILE_MODE, verify=True)


def test_write_private_text_replaces_atomically(tmp_path):
    target = tmp_path / "trust.json"
    write_private_text(target, "first")
    write_private_text(target, "second")

    assert target.read_text(encoding="utf-8") == "second"
    assert _mode(target) == PRIVATE_FILE_MODE
    assert list(tmp_path.iterdir()) == [target]


def test_exclusive_lock_reports_contention_as_oserror(tmp_path):
    target = tmp_path / "runs.jsonl"

    with (
        exclusive_lock(target, busy_message="store is busy", timeout=0),
        pytest.raises(OSError, match="store is busy"),
        exclusive_lock(target, busy_message="store is busy", timeout=0),
    ):
        pass


def test_apply_sqlite_pragmas_enables_wal(tmp_path):
    connection = sqlite3.connect(tmp_path / "db.sqlite")
    try:
        apply_sqlite_pragmas(connection)
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        connection.close()

    assert mode.lower() == "wal"


def test_user_state_path_and_vela_dir_follow_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    if os.name == "nt":  # pragma: no cover - posix in CI
        monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert user_state_path("config.json") == tmp_path / ".vela" / "config.json"
    assert vela_dir(tmp_path / "project") == tmp_path / "project" / ".vela"
