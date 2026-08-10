from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

import vela.agent.agent as agent_module
from vela.agent import Agent
from vela.config import load_config
from vela.entrypoints import repl
from vela.entrypoints.repl import (
    SLASH_COMMANDS,
    _handle_slash,
    _repl_loop,
    _run_agent_with_session,
    _run_delegated_with_session,
)
from vela.render.rich_renderer import RichRenderer
from vela.session import (
    ActiveSession,
    SessionStore,
    bounded_tool_transcript,
    finalize_interrupted_history,
)
from vela.task_control import InteractiveTaskController, TaskState
from vela.tools import ToolRegistry
from vela.tools.base import Tool, ToolResult, object_schema
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

    monkeypatch.setattr("vela.session.os.chmod", reject_chmod)

    active = ActiveSession.open(tmp_path / "project")

    assert active.store is None
    assert active.current.id.startswith("memory_")
    assert "continuing in memory" in (active.take_warning() or "")


def test_repl_lists_and_resumes_a_persisted_session(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    project = tmp_path / "project"
    previous = store.create(project)
    store.save(
        previous.id,
        [
            Message(role="user", content="continue this feature"),
            Message(role="assistant", content="previous answer"),
        ],
    )
    active = ActiveSession.open(project, resume=False, store=store)
    agent = _FakeAgent()
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=160)
    runtime = _repl_runtime(project, active, agent, console)

    asyncio.run(_handle_slash("/sessions", runtime))
    asyncio.run(_handle_slash("/resume", runtime))

    output = stream.getvalue()
    assert "/sessions" in SLASH_COMMANDS
    assert "/resume" in SLASH_COMMANDS
    assert previous.id in output
    assert "continue this feature" in output
    assert f"Resumed {previous.id}" in output
    assert [message.content for message in agent.history] == [
        "continue this feature",
        "previous answer",
    ]


@pytest.mark.parametrize("reference_kind", ["index", "id"])
def test_resume_command_accepts_explicit_index_or_id(tmp_path, reference_kind):
    store = SessionStore(tmp_path / "sessions.db")
    project = tmp_path / "project"
    target = store.create(project)
    store.save(target.id, [Message(role="user", content="target history")])
    active = ActiveSession.open(project, resume=False, store=store)
    reference = "2" if reference_kind == "index" else target.id
    agent, output = _run_session_command(f"/resume {reference}", project, active)

    assert active.current.id == target.id
    assert agent.history[0].content == "target history"
    assert f"Resumed {target.id}" in output


@pytest.mark.parametrize(
    ("reference", "expected"),
    [("missing", "No matching previous session"), ("session_", "Ambiguous session reference")],
)
def test_resume_command_reports_missing_or_ambiguous_reference(tmp_path, reference, expected):
    store = SessionStore(tmp_path / "sessions.db")
    project = tmp_path / "project"
    for content in ("first", "second"):
        record = store.create(project)
        store.save(record.id, [Message(role="user", content=content)])
    active = ActiveSession.open(project, resume=False, store=store)
    current_id = active.current.id

    agent, output = _run_session_command(f"/resume {reference}", project, active)

    assert active.current.id == current_id
    assert agent.history == []
    assert expected in output


def test_agent_retains_user_message_when_provider_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    agent = _agent_with_client(tmp_path, _ErrorClient())

    async def run():
        return [event async for event in agent.run("keep this request")]

    events = asyncio.run(run())

    assert any(event["type"] == "error" for event in events)
    assert agent.history[-1].role == "user"
    assert "keep this request" in str(agent.history[-1].content)


def test_agent_retains_user_message_when_cancelled(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    agent = _agent_with_client(tmp_path, _CancelledClient())

    async def run():
        async for _ in agent.run("keep cancelled request"):
            pass

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())

    assert agent.history[-1].role == "user"
    assert "keep cancelled request" in str(agent.history[-1].content)


@pytest.mark.parametrize(
    ("mode", "delegate_name"),
    [("plan", "LangGraphPlanAgent"), ("team", "AgentOrchestrator")],
)
def test_agent_delegated_modes_preserve_prior_history(tmp_path, monkeypatch, mode, delegate_name):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    delegated = _FakeDelegatedAgent()
    monkeypatch.setattr(agent_module, delegate_name, lambda **kwargs: delegated)
    agent = _agent_with_client(tmp_path, _ErrorClient())
    agent.mode = mode
    agent.history = [Message(role="assistant", content="earlier answer")]

    async def run():
        return [event async for event in agent.run("new task")]

    asyncio.run(run())

    assert [message.content for message in agent.history] == [
        "earlier answer",
        "new task",
        "delegated result",
    ]


@pytest.mark.parametrize(
    ("mode", "delegate_name"),
    [("plan", "LangGraphPlanAgent"), ("team", "AgentOrchestrator")],
)
def test_agent_forwards_plan_review_callback_to_delegated_modes(
    tmp_path, monkeypatch, mode, delegate_name
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    captured = {}
    delegated = _FakeDelegatedAgent()

    def create_delegate(**kwargs):
        captured.update(kwargs)
        return delegated

    monkeypatch.setattr(agent_module, delegate_name, create_delegate)
    callback = lambda plan: None  # noqa: E731, ARG005 - identity assertion only
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    agent = Agent(
        llm_client=_ErrorClient(),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
        mode=mode,
        plan_review_callback=callback,
    )

    async def run():
        return [event async for event in agent.run("new task")]

    asyncio.run(run())

    assert captured["plan_review_callback"] is callback


def test_repl_persists_incremental_history_when_run_is_cancelled(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path / "project", store=store)
    agent = _FakeAgent()
    console = Console(file=StringIO(), color_system=None)

    async def cancel_after_history_update(agent, renderer, message):  # noqa: ARG001
        agent.history = [Message(role="user", content=message)]
        raise asyncio.CancelledError

    monkeypatch.setattr(repl, "_run_agent", cancel_after_history_update)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_run_agent_with_session(agent, None, "persist me", active, console))

    persisted = store.get(active.current.id)
    assert persisted is not None
    assert persisted.messages[0].content == "persist me"


def test_repl_persists_and_resumes_cancelled_pending_tool_call(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions.db")
    project = tmp_path / "project"
    active = ActiveSession.open(project, store=store)
    agent = _FakeAgent()
    console = Console(file=StringIO(), color_system=None)

    async def cancel_with_pending_tool(agent, renderer, message):  # noqa: ARG001
        agent.history = [
            Message(role="user", content=message),
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "pending_1", "name": "bash", "input": {}}],
            ),
        ]
        raise asyncio.CancelledError

    monkeypatch.setattr(repl, "_run_agent", cancel_with_pending_tool)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_run_agent_with_session(agent, None, "resume me", active, console))

    resumed = ActiveSession.open(project, resume=True, store=store)
    tool_messages = [message for message in resumed.current.messages if message.role == "tool"]
    assert resumed.resumed
    assert tool_messages[-1].tool_call_id == "pending_1"
    assert "cancelled" in str(tool_messages[-1].content)
    assert "cancelled" in str(resumed.current.messages[-1].content)


def test_cancelled_run_keeps_completed_tool_result_and_closes_only_pending_call(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    second_started = asyncio.Event()

    async def first_tool(payload, context):  # noqa: ARG001
        return ToolResult("first completed")

    async def second_tool(payload, context):  # noqa: ARG001
        second_started.set()
        await asyncio.Event().wait()
        return ToolResult("unreachable")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="first_tool",
            description="first",
            parameters=object_schema({}),
            handler=first_tool,
            is_read_only=False,
        )
    )
    registry.register(
        Tool(
            name="second_tool",
            description="second",
            parameters=object_schema({}),
            handler=second_tool,
            is_read_only=False,
        )
    )
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.skill = False
    config.features.memory = False
    agent = Agent(
        llm_client=_TwoToolClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path, store=store)
    console = Console(file=StringIO(), color_system=None)

    async def run():
        task = asyncio.create_task(
            _run_agent_with_session(
                agent,
                RichRenderer(console),
                "run two tools",
                active,
                console,
            )
        )
        await second_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    persisted = store.get(active.current.id)
    assert persisted is not None
    tool_messages = {
        message.tool_call_id: message for message in persisted.messages if message.role == "tool"
    }
    assert tool_messages["call_1"].content == "first completed"
    assert "cancelled" in str(tool_messages["call_2"].content)


def test_ctrl_c_first_cancels_running_task_and_second_exits(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path / "project", store=store)
    controller = InteractiveTaskController()
    console = Console(file=StringIO(), color_system=None)

    class InterruptSession:
        def __init__(self):
            self.calls = 0

        async def prompt_async(self):
            self.calls += 1
            raise KeyboardInterrupt

    prompt_session = InterruptSession()

    async def never_finishes():
        await asyncio.Event().wait()

    async def run():
        controller.start(
            never_finishes(),
            initial_state=TaskState.RUNNING,
            label="long task",
        )
        runtime = _repl_runtime(
            tmp_path / "project",
            active,
            _FakeAgent(),
            console,
            task_controller=controller,
        )
        await _repl_loop(prompt_session, runtime)

    asyncio.run(run())

    assert prompt_session.calls == 2
    assert controller.state == TaskState.CANCELLED
    assert "再次 Ctrl+C" in console.file.getvalue()


def test_help_explains_cancel_plan_review_and_resume_contracts(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path / "project", store=store)
    agent, output = _run_session_command("/help", tmp_path / "project", active)

    assert agent is not None
    assert "/cancel" in output
    assert "modify <requirement>" in output
    assert "/resume [id|number]" in output
    assert "Ctrl+C" in output


def test_delegated_run_preserves_prior_history_and_persists_result(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path / "project", store=store)
    previous = Message(role="user", content="earlier request")
    active.save([previous], title="earlier request")
    agent = _FakeAgent()
    agent.history = [previous]
    delegated = _FakeDelegatedAgent()
    console = Console(file=StringIO(), color_system=None)

    asyncio.run(_run_delegated_with_session(delegated, "new task", agent, active, console))

    assert [message.content for message in agent.history] == [
        "earlier request",
        "new task",
        "delegated result",
    ]
    persisted = store.get(active.current.id)
    assert persisted is not None
    assert persisted.messages == agent.history


def _run_session_command(raw, project, active):
    agent = _FakeAgent()
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=160)
    asyncio.run(_handle_slash(raw, _repl_runtime(project, active, agent, console)))
    return agent, stream.getvalue()


def _repl_runtime(project, active, agent, console, *, task_controller=None):
    config = load_config(project_root=project)
    controller = task_controller or InteractiveTaskController()
    registry = ToolRegistry()
    return repl.ReplRuntime(
        console=console,
        cwd=str(project),
        config=config,
        agent=agent,
        registry=registry,
        permission_mode=repl.PermissionModeController(config),
        renderer=RichRenderer(console),
        active_session=active,
        task_controller=controller,
    )


def _agent_with_client(tmp_path, client):
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.skill = False
    config.features.memory = False
    return Agent(
        llm_client=client,
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
    )


class _ErrorClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1_000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {"type": "error", "error": RuntimeError("provider failed")}


class _CancelledClient(_ErrorClient):
    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        raise asyncio.CancelledError
        yield  # pragma: no cover


class _TwoToolClient(_ErrorClient):
    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        for index, name in enumerate(("first_tool", "second_tool")):
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": index,
                    "id": f"call_{index + 1}",
                    "function": {"name": name, "arguments": "{}"},
                },
            }
        yield {"type": "message_end", "stop_reason": "tool_use"}


class _FakeAgent:
    def __init__(self):
        self.history = []
        self.last_usage = None
        self.llm_client = SimpleNamespace(max_context_window=1_000)

    def clear_history(self):
        self.history = []


class _FakeDelegatedAgent:
    def __init__(self):
        self.history = []

    async def run(self, message):
        yield {"type": "text_delta", "text": "delegated result"}
        self.history = [
            Message(role="user", content=message),
            Message(role="assistant", content="delegated result"),
        ]
        yield {"type": "done", "messages": self.history}
