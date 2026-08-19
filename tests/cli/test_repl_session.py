from __future__ import annotations

import asyncio
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

import vela.agent.agent as agent_module
from vela.agent import Agent
from vela.config import load_config
from vela.entrypoints import repl, repl_tasks
from vela.entrypoints.repl import _repl_loop
from vela.entrypoints.repl_commands import SLASH_COMMANDS, handle_slash
from vela.entrypoints.repl_tasks import (
    run_agent_with_session,
    run_delegated_with_session,
    run_events,
)
from vela.render.rich_renderer import RichRenderer
from vela.run_trace import RunTrace, RunTracker
from vela.session import ActiveSession, SessionStore
from vela.task_control import InteractiveTaskController, TaskState
from vela.tools import ToolRegistry
from vela.tools.base import Tool, ToolResult, object_schema
from vela.types import Message


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

    asyncio.run(handle_slash("/sessions", runtime))
    asyncio.run(handle_slash("/resume", runtime))

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


def test_repl_consumes_run_finished_before_raising_agent_error(tmp_path) -> None:
    stream = StringIO()
    renderer = RichRenderer(Console(file=stream, color_system=None, width=160))
    tracker = RunTracker(
        mode="react",
        model="fake-model",
        provider="fake-provider",
        cwd=str(tmp_path),
    )

    async def failed_events():
        yield {"type": "error", "error": RuntimeError("provider failed")}

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(run_events(tracker.stream(failed_events()), renderer))

    output = stream.getvalue()
    assert "failed" in output
    assert tracker.trace.run_id.removeprefix("run_") in output


def test_repl_renders_cancelled_trace_before_propagating_cancel(tmp_path, monkeypatch) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path / "project", store=store)
    agent = _FakeAgent()
    agent.last_run_trace = RunTrace(
        run_id="run_cancelled123",
        status="cancelled",
        mode="react",
        model="fake-model",
        provider="fake-provider",
        cwd=str(tmp_path),
        session_id=active.current.id,
        started_at="2026-08-14T00:00:00+00:00",
        finished_at="2026-08-14T00:00:01+00:00",
        duration_ms=1_000,
    )
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=160)

    async def cancel_run(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(repl_tasks, "run_events", cancel_run)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_agent_with_session(
                agent,
                RichRenderer(console),
                "cancel me",
                active,
                console,
            )
        )

    assert "cancelled123" in stream.getvalue()
    assert "cancelled" in stream.getvalue()


def test_agent_plan_mode_preserves_prior_history(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    delegated = _FakeDelegatedAgent()
    monkeypatch.setattr(agent_module, "LangGraphPlanAgent", lambda **kwargs: delegated)
    agent = _agent_with_client(tmp_path, _ErrorClient())
    agent.mode = "plan"
    agent.history = [Message(role="assistant", content="earlier answer")]

    async def run():
        return [event async for event in agent.run("new task")]

    asyncio.run(run())

    assert [message.content for message in agent.history] == [
        "earlier answer",
        "new task",
        "delegated result",
    ]


def test_agent_forwards_plan_review_callback_to_plan_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    captured = {}
    delegated = _FakeDelegatedAgent()

    def create_delegate(**kwargs):
        captured.update(kwargs)
        return delegated

    monkeypatch.setattr(agent_module, "LangGraphPlanAgent", create_delegate)
    callback = lambda plan: None  # noqa: E731, ARG005 - identity assertion only
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    agent = Agent(
        llm_client=_ErrorClient(),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
        mode="plan",
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

    async def cancel_after_history_update(*_args, **_kwargs):
        message = "persist me"
        agent.history = [Message(role="user", content=message)]
        raise asyncio.CancelledError

    monkeypatch.setattr(repl_tasks, "run_events", cancel_after_history_update)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_agent_with_session(
                agent,
                RichRenderer(console),
                "persist me",
                active,
                console,
            )
        )

    persisted = store.get(active.current.id)
    assert persisted is not None
    assert persisted.messages[0].content == "persist me"


def test_repl_persists_and_resumes_cancelled_pending_tool_call(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions.db")
    project = tmp_path / "project"
    active = ActiveSession.open(project, store=store)
    agent = _FakeAgent()
    console = Console(file=StringIO(), color_system=None)

    async def cancel_with_pending_tool(*_args, **_kwargs):
        message = "resume me"
        agent.history = [
            Message(role="user", content=message),
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "pending_1", "name": "bash", "input": {}}],
            ),
        ]
        raise asyncio.CancelledError

    monkeypatch.setattr(repl_tasks, "run_events", cancel_with_pending_tool)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_agent_with_session(
                agent,
                RichRenderer(console),
                "resume me",
                active,
                console,
            )
        )

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
            run_agent_with_session(
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

        async def prompt_async(self, **_kwargs):
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


def test_repl_opens_the_next_draft_prompt_while_task_is_running(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path / "project", store=store)
    controller = InteractiveTaskController()
    console = Console(file=StringIO(), color_system=None)
    task_started = asyncio.Event()
    release_task = asyncio.Event()
    second_prompt_started = asyncio.Event()

    class RecordingSession:
        def __init__(self):
            self.calls = 0

        async def prompt_async(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return "hello"
            second_prompt_started.set()
            await release_task.wait()
            while controller.active:
                await asyncio.sleep(0)
            raise EOFError

    async def block_agent(*_args, **_kwargs):
        task_started.set()
        await release_task.wait()

    monkeypatch.setattr(repl, "run_agent_with_session", block_agent)

    async def run() -> tuple[bool, bool, int]:
        prompt_session = RecordingSession()
        runtime = _repl_runtime(
            tmp_path / "project",
            active,
            _FakeAgent(),
            console,
            task_controller=controller,
        )
        repl_task = asyncio.create_task(_repl_loop(prompt_session, runtime))
        await asyncio.sleep(0.05)
        observed = task_started.is_set(), second_prompt_started.is_set()
        release_task.set()
        await repl_task
        return observed[0], observed[1], prompt_session.calls

    started_before_release, prompted_before_release, prompt_calls = asyncio.run(run())

    assert started_before_release
    assert prompted_before_release
    assert prompt_calls == 2
    assert controller.state == TaskState.COMPLETED


def test_repl_prints_submitted_message_as_compact_history(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path / "project", store=store)
    console = Console(file=StringIO(), color_system=None)

    class OneMessageSession:
        submitted = False

        async def prompt_async(self, **_kwargs):
            if not self.submitted:
                self.submitted = True
                return "hello"
            raise EOFError

    async def complete_agent(*_args, **_kwargs):
        return None

    monkeypatch.setattr(repl, "run_agent_with_session", complete_agent)
    runtime = _repl_runtime(tmp_path / "project", active, _FakeAgent(), console)

    asyncio.run(_repl_loop(OneMessageSession(), runtime))

    assert "❯ hello" in console.file.getvalue()


def test_repl_keeps_typeahead_draft_when_two_messages_arrive_together(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path / "project", store=store)
    controller = InteractiveTaskController()
    console = Console(file=StringIO(), color_system=None)
    task_started = asyncio.Event()

    class TypeaheadSession:
        def __init__(self):
            self.calls = 0
            self.restored_draft = ""
            self.draft_restored = asyncio.Event()

        async def prompt_async(self, *, default=""):
            self.calls += 1
            if self.calls == 1:
                return "hello"
            if self.calls == 2:
                return "next task"
            self.restored_draft = default
            self.draft_restored.set()
            await asyncio.Event().wait()

    async def block_agent(*_args, **_kwargs):
        task_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(repl, "run_agent_with_session", block_agent)

    async def run() -> tuple[bool, str, str]:
        prompt_session = TypeaheadSession()
        runtime = _repl_runtime(
            tmp_path / "project",
            active,
            _FakeAgent(),
            console,
            task_controller=controller,
        )
        repl_task = asyncio.create_task(_repl_loop(prompt_session, runtime))
        await asyncio.wait_for(task_started.wait(), timeout=1)
        await asyncio.wait_for(prompt_session.draft_restored.wait(), timeout=1)
        result = controller.active, prompt_session.restored_draft, console.file.getvalue()
        controller.request_cancel()
        repl_task.cancel()
        await asyncio.gather(repl_task, return_exceptions=True)
        await controller.wait()
        return result

    active_while_drafting, draft, output = asyncio.run(run())

    assert active_while_drafting
    assert draft == "next task"
    assert "当前任务仍在运行" not in output


def test_help_explains_cancel_plan_review_and_resume_contracts(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path / "project", store=store)
    agent, output = _run_session_command("/help", tmp_path / "project", active)

    assert agent is not None
    assert "/cancel" in output
    assert "Draft while running; Enter unlocks after completion" in output
    assert "modify <requirement>" in output
    assert "/resume [id|number]" in output
    assert "Ctrl+C" in output


def test_running_task_rejects_another_message_without_starting_or_queueing_it(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path / "project", store=store)
    controller = InteractiveTaskController()
    console = Console(file=StringIO(), color_system=None)

    async def never_finishes():
        await asyncio.Event().wait()

    async def run():
        controller.start(
            never_finishes(),
            initial_state=TaskState.RUNNING,
            label="current task",
        )
        runtime = _repl_runtime(
            tmp_path / "project",
            active,
            _FakeAgent(),
            console,
            task_controller=controller,
        )
        assert not await repl._dispatch_message("next task", runtime)
        assert controller.active
        controller.request_cancel()
        await controller.wait()

    asyncio.run(run())

    assert "当前任务仍在运行" in console.file.getvalue()


def test_delegated_run_preserves_prior_history_and_persists_result(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    active = ActiveSession.open(tmp_path / "project", store=store)
    previous = Message(role="user", content="earlier request")
    active.save([previous], title="earlier request")
    agent = _FakeAgent()
    agent.history = [previous]
    delegated = _FakeDelegatedAgent()
    console = Console(file=StringIO(), color_system=None)

    asyncio.run(run_delegated_with_session(delegated, "new task", agent, active, console))

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
    asyncio.run(handle_slash(raw, _repl_runtime(project, active, agent, console)))
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
    max_context_window = 20_000

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
        self.llm_client = SimpleNamespace(max_context_window=20_000)

    def clear_history(self):
        self.history = []

    async def run(self, message):  # noqa: ARG002
        if False:  # pragma: no cover - gives tests an inert async event stream
            yield {}


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
