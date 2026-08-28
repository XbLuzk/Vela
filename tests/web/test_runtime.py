from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from vela.config import VelaConfig
from vela.plan.models import ExecutionPlan, Task, TaskType
from vela.session import ActiveSession, SessionStore
from vela.web import runtime as runtime_module
from vela.web.runtime import EventHub, RuntimeManager, _record_file_change, serialize_agent_event


def test_serialize_agent_event_converts_errors_and_plan_dataclasses():
    plan = ExecutionPlan(id="plan-1", goal="ship web", summary="Web only")
    plan.add_task(Task(id="T1", description="Build UI", type=TaskType.FILE_WRITE))

    payload = serialize_agent_event(
        {
            "type": "plan_created",
            "plan": plan,
            "error": RuntimeError("boom"),
        }
    )

    assert payload["type"] == "plan_created"
    assert payload["error"] == "boom"
    assert payload["plan"]["tasks"]["T1"]["type"] == "FILE_WRITE"


def test_file_change_tracking_emits_a_unified_diff(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("before\n", encoding="utf-8")
    snapshots = {}

    assert (
        _record_file_change(
            {
                "type": "tool_call",
                "tool_call_id": "write-1",
                "name": "write_file",
                "input": {"path": "notes.txt"},
            },
            cwd=tmp_path,
            snapshots=snapshots,
        )
        is None
    )
    target.write_text("after\n", encoding="utf-8")
    change = _record_file_change(
        {"type": "tool_result", "tool_call_id": "write-1", "is_error": False},
        cwd=tmp_path,
        snapshots=snapshots,
    )

    assert change is not None
    assert change["path"] == "notes.txt"
    assert "-before" in change["diff"]
    assert "+after" in change["diff"]


def test_event_hub_fans_out_to_connected_streams():
    async def scenario():
        hub = EventHub()
        stream = hub.stream()
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)

        await hub.publish({"type": "text_delta", "text": "hello"})

        assert await pending == {"type": "text_delta", "text": "hello"}
        await stream.aclose()

    asyncio.run(scenario())


def test_event_hub_close_unblocks_connected_streams():
    async def scenario():
        hub = EventHub()
        stream = hub.stream()
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)

        hub.close()

        try:
            await pending
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("closing the event hub must end connected streams")
        await stream.aclose()

    asyncio.run(scenario())


def test_pending_project_trust_starts_with_builtin_capabilities(monkeypatch, tmp_path):
    manager = RuntimeManager(tmp_path)
    manager.active_session = ActiveSession.open(
        tmp_path,
        store=SessionStore(tmp_path / "sessions.db"),
    )
    rebuild = AsyncMock()
    monkeypatch.setattr(runtime_module, "has_trust_sensitive_resources", lambda _cwd: True)
    monkeypatch.setattr(manager.trust_store, "get", lambda _cwd: None)
    monkeypatch.setattr(manager, "rebuild", rebuild)

    asyncio.run(manager.initialize())

    assert manager.project_extensions_pending is True
    assert manager.project_trusted is False
    rebuild.assert_awaited_once_with()


def test_sessions_are_available_without_model_configuration(monkeypatch, tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    active_session = ActiveSession.open(tmp_path, store=store)
    manager = RuntimeManager(tmp_path)
    monkeypatch.setattr(
        runtime_module.ActiveSession,
        "open",
        lambda *_args, **_kwargs: active_session,
    )
    monkeypatch.setattr(runtime_module, "load_config", lambda **_kwargs: VelaConfig())

    asyncio.run(manager.initialize())

    snapshot = manager.snapshot()
    assert snapshot["ready"] is False
    assert snapshot["session"]["id"] == active_session.current.id
    assert [item["id"] for item in snapshot["sessions"]] == [active_session.current.id]


def test_new_session_works_without_model_configuration(monkeypatch, tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    active_session = ActiveSession.open(tmp_path, store=store)
    first_id = active_session.current.id
    manager = RuntimeManager(tmp_path)
    monkeypatch.setattr(
        runtime_module.ActiveSession,
        "open",
        lambda *_args, **_kwargs: active_session,
    )
    monkeypatch.setattr(runtime_module, "load_config", lambda **_kwargs: VelaConfig())

    asyncio.run(manager.initialize())
    created = asyncio.run(manager.new_session())

    assert created["id"] != first_id
    assert store.get(first_id) is None
    assert [item["id"] for item in manager.list_sessions()] == [created["id"]]


def test_delete_session_returns_authoritative_replacement(monkeypatch, tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    active_session = ActiveSession.open(tmp_path, store=store)
    deleted_id = active_session.current.id
    manager = RuntimeManager(tmp_path)
    manager.active_session = active_session
    monkeypatch.setattr(runtime_module, "load_config", lambda **_kwargs: VelaConfig())

    snapshot = asyncio.run(manager.delete_session(deleted_id))

    assert snapshot["session"]["id"] != deleted_id
    assert deleted_id not in {item["id"] for item in snapshot["sessions"]}


def test_select_project_reopens_sessions_in_selected_directory(monkeypatch, tmp_path):
    old_project = tmp_path / "old"
    new_project = tmp_path / "new"
    old_project.mkdir()
    new_project.mkdir()
    store = SessionStore(tmp_path / "sessions.db")
    original_open = ActiveSession.open
    manager = RuntimeManager(old_project)
    manager.active_session = original_open(old_project, store=store)
    rebuild = AsyncMock()
    monkeypatch.setattr(
        runtime_module.ActiveSession,
        "open",
        lambda cwd, *, resume=False: original_open(cwd, resume=resume, store=store),
    )
    monkeypatch.setattr(runtime_module, "has_trust_sensitive_resources", lambda _cwd: False)
    monkeypatch.setattr(runtime_module, "load_config", lambda **_kwargs: VelaConfig())
    monkeypatch.setattr(manager, "rebuild", rebuild)

    snapshot = asyncio.run(manager.select_project(new_project))

    assert manager.cwd == new_project.resolve()
    assert snapshot["cwd"] == str(new_project.resolve())
    assert manager.active_session is not None
    assert manager.active_session.current.cwd == str(new_project.resolve())
    rebuild.assert_awaited_once_with()
