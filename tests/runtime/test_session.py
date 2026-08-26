from __future__ import annotations

import json
import os
import sqlite3

import pytest

from vela.session import (
    ActiveSession,
    SessionStore,
    bounded_tool_transcript,
    finalize_interrupted_history,
)
from vela.types import Message


def test_session_store_round_trips_full_conversation(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    session = store.create(tmp_path / "project")
    messages = [
        Message(role="user", content="inspect the project"),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "name": "read_file", "input": {"path": "README.md"}}],
        ),
        Message(role="tool", content="project docs", tool_call_id="call_1"),
        Message(role="assistant", content="done"),
    ]

    saved = store.save(session.id, messages, title="Original user request")
    loaded = store.get(session.id, cwd=tmp_path / "project")

    assert loaded is not None
    assert loaded.id == session.id
    assert loaded.title == "Original user request"
    assert loaded.message_count == 4
    assert loaded.messages == messages
    assert saved.updated_at == loaded.updated_at


def test_active_session_degrades_to_memory_when_database_is_locked(tmp_path):
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    active = ActiveSession.open(tmp_path / "project", store=store)
    lock = sqlite3.connect(db_path)
    lock.execute("begin exclusive")
    try:
        active.save([Message(role="user", content="keep in memory")])
    finally:
        lock.rollback()
        lock.close()

    assert active.store is None
    assert active.current.messages[-1].content == "keep in memory"
    assert "Could not save session" in str(active.warning)


def test_interrupted_history_closes_pending_tool_call_for_resume():
    messages = [
        Message(role="user", content="run a command"),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "name": "bash", "input": {}}],
        ),
    ]

    finalized = finalize_interrupted_history(messages, status="cancelled")

    assert finalized[-2].role == "tool"
    assert finalized[-2].tool_call_id == "call_1"
    assert "cancelled" in str(finalized[-2].content)
    assert finalized[-1].role == "assistant"
    assert "cancelled" in str(finalized[-1].content)


def test_bounded_tool_transcript_keeps_recent_pairs_and_limits_payload():
    messages = []
    for index in range(30):
        call_id = f"call_{index}"
        messages.extend(
            [
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[{"id": call_id, "name": "bash", "input": {}}],
                ),
                Message(role="tool", content="x" * 100, tool_call_id=call_id),
            ]
        )

    transcript = bounded_tool_transcript(messages, max_calls=3, max_content_chars=10)

    assert len(transcript) == 6
    assert [message.tool_call_id for message in transcript if message.role == "tool"] == [
        "call_27",
        "call_28",
        "call_29",
    ]
    assert all(len(str(message.content)) < 60 for message in transcript)


def test_bounded_tool_transcript_caps_tool_call_arguments():
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_large",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "x" * 20_000},
                }
            ],
        )
    ]

    transcript = bounded_tool_transcript(messages, max_content_chars=100)

    serialized_call = json.dumps(transcript[0].tool_calls[0])
    assert len(serialized_call) < 250
    assert "_truncated" in serialized_call


def test_session_store_lists_and_resolves_sessions_within_project_scope(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    project = tmp_path / "project"
    first = store.create(project)
    store.save(first.id, [Message(role="user", content="first task")])
    second = store.create(project)
    store.save(second.id, [Message(role="user", content="second task")])
    store.create(tmp_path / "other-project")

    sessions = store.list(project)

    assert [item.id for item in sessions] == [second.id, first.id]
    assert all(item.messages == [] for item in sessions)
    assert store.resolve(project).id == second.id
    assert store.resolve(project).messages[0].content == "second task"
    assert store.resolve(project, "1").id == second.id
    assert store.resolve(project, "2").id == first.id
    assert store.resolve(project, first.id[:-1]).id == first.id
    assert store.resolve(project, exclude_id=second.id).id == first.id
    assert store.get(second.id, cwd=tmp_path / "other-project") is None


def test_session_store_returns_none_for_missing_reference(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")

    assert store.resolve(tmp_path / "project") is None
    assert store.resolve(tmp_path / "project", "missing") is None


def test_session_store_resolves_exact_id_older_than_list_limit(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    project = tmp_path / "project"
    oldest = store.create(project)
    store.save(oldest.id, [Message(role="user", content="oldest")])
    for index in range(20):
        record = store.create(project)
        store.save(record.id, [Message(role="user", content=f"newer {index}")])

    assert oldest.id not in {record.id for record in store.list(project, limit=20)}
    resolved = store.resolve(project, oldest.id)
    assert resolved is not None
    assert resolved.id == oldest.id
    assert resolved.messages[0].content == "oldest"


def test_implicit_resume_skips_newer_empty_session(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    project = tmp_path / "project"
    previous = store.create(project)
    store.save(previous.id, [Message(role="user", content="keep this")])
    empty = store.create(project)

    resolved = store.resolve(project)

    assert resolved is not None
    assert resolved.id == previous.id
    assert resolved.id != empty.id


def test_active_session_opens_latest_or_creates_new_session(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    project = tmp_path / "project"
    previous = store.create(project)
    store.save(previous.id, [Message(role="user", content="previous work")])

    resumed = ActiveSession.open(project, resume=True, store=store)
    fresh = ActiveSession.open(project, resume=False, store=store)

    assert resumed.resumed is True
    assert resumed.current.id == previous.id
    assert resumed.current.messages[0].content == "previous work"
    assert fresh.resumed is False
    assert fresh.current.id != previous.id


def test_active_session_keeps_the_first_user_prompt_as_its_title(tmp_path):
    active = ActiveSession.open(
        tmp_path / "project",
        store=SessionStore(tmp_path / "sessions.db"),
    )

    active.save([Message(role="user", content="injected first message")], title="First prompt")
    active.save(
        [
            Message(role="user", content="injected first message"),
            Message(role="assistant", content="answer"),
            Message(role="user", content="second prompt"),
        ],
        title="Second prompt",
    )

    assert active.current.title == "First prompt"


def test_active_session_switches_to_previous_and_discards_empty_current(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    project = tmp_path / "project"
    previous = store.create(project)
    store.save(previous.id, [Message(role="user", content="previous work")])
    active = ActiveSession.open(project, resume=False, store=store)
    empty_id = active.current.id

    switched = active.switch()

    assert switched is not None
    assert switched.id == previous.id
    assert active.current.id == previous.id
    assert store.get(empty_id) is None


def test_active_session_keeps_running_in_memory_when_save_fails(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path / "project", store=store)
    persisted_id = active.current.id

    def fail_save(*args, **kwargs):  # noqa: ARG001
        raise OSError("disk full")

    monkeypatch.setattr(store, "save", fail_save)
    active.save([Message(role="user", content="not lost")], title="not lost")

    assert active.current.messages[0].content == "not lost"
    assert active.store is None
    assert "continuing in memory" in (active.take_warning() or "")
    persisted = SessionStore(tmp_path / "sessions.db").get(persisted_id)
    assert persisted is not None
    assert persisted.message_count == 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits only")
def test_session_storage_uses_private_permissions(tmp_path):
    session_dir = tmp_path / "sessions"
    store = SessionStore(session_dir / "sessions.db")
    sidecar = session_dir / "sessions.db-wal"
    sidecar.write_text("temporary", encoding="utf-8")
    sidecar.chmod(0o666)

    store.list(tmp_path / "project")

    assert session_dir.stat().st_mode & 0o777 == 0o700
    assert store.db_path.stat().st_mode & 0o777 == 0o600
    assert sidecar.stat().st_mode & 0o777 == 0o600


def test_active_session_falls_back_to_memory_when_permissions_cannot_be_enforced(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def reject_chmod(path, mode):  # noqa: ARG001
        raise PermissionError("chmod rejected")

    monkeypatch.setattr("vela.storage.os.chmod", reject_chmod)

    active = ActiveSession.open(tmp_path / "project")

    assert active.store is None
    assert active.current.id.startswith("memory_")
    assert "continuing in memory" in (active.take_warning() or "")
