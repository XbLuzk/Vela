"""Run ReAct and Plan tasks while preserving the active REPL session."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from time import monotonic
from typing import TYPE_CHECKING

from rich.console import Console

from vela.agent import Agent, LangGraphPlanAgent
from vela.events import AgentEvent
from vela.render import RichRenderer
from vela.session import ActiveSession
from vela.session_history import finalize_interrupted_history
from vela.task_control import InteractiveTaskController, TaskState
from vela.types import Message

if TYPE_CHECKING:
    from vela.entrypoints.repl import ReplRuntime


async def run_agent_with_session(
    agent: Agent,
    renderer: RichRenderer,
    message: str,
    active_session: ActiveSession,
    console: Console,
    task_controller: InteractiveTaskController | None = None,
) -> None:
    """Run one ReAct request and always persist its final transcript."""
    try:
        await run_events(
            agent.run(message),
            renderer,
            agent.llm_client.max_context_window,
            task_controller,
        )
    except asyncio.CancelledError:
        _finalize_cancelled(agent, renderer)
        raise
    except BaseException as exc:
        _finalize_failed(agent, exc)
        raise
    finally:
        active_session.save(agent.history, title=message)
        print_session_warning(console, active_session)


async def run_events(
    events: AsyncIterable[AgentEvent],
    renderer: RichRenderer,
    context_window: int | None = None,
    task_controller: InteractiveTaskController | None = None,
) -> None:
    """Render one event stream, print a small ephemeral summary, then raise errors."""
    renderer.set_context_window(context_window)
    renderer.start_run()
    renderer.newline()
    started_at = monotonic()
    pending_error: BaseException | None = None
    completed = False
    turns = 0
    tool_calls = 0
    total_tokens = 0
    try:
        async for event in events:
            event_type = event.get("type")
            renderer.handle(event)
            if task_controller is not None and event_type == "plan_status":
                task_controller.set_phase(str(event.get("phase") or ""))
            if event_type == "turn_started":
                turns = max(turns, int(event.get("turn") or 0))
            elif event_type == "tool_call":
                tool_calls += 1
            elif event_type == "done":
                completed = True
                turns = max(turns, int(event.get("total_turns") or 0))
                total_tokens = int(event.get("total_tokens") or 0)
            elif event_type == "error" and pending_error is None:
                error = event.get("error")
                pending_error = (
                    error if isinstance(error, BaseException) else RuntimeError(str(error))
                )
    except asyncio.CancelledError:
        renderer.print_run_summary(
            status="cancelled",
            duration_ms=int((monotonic() - started_at) * 1_000),
            turns=turns,
            tool_calls=tool_calls,
            total_tokens=total_tokens,
        )
        renderer.newline()
        raise

    status = "completed" if completed and pending_error is None else "failed"
    renderer.print_run_summary(
        status=status,
        duration_ms=int((monotonic() - started_at) * 1_000),
        turns=turns,
        tool_calls=tool_calls,
        total_tokens=total_tokens,
    )
    renderer.newline()
    if pending_error is not None:
        raise pending_error
    if not completed:
        raise RuntimeError("Agent stream ended before completion.")


def start_plan(arg: str, runtime: ReplRuntime) -> None:
    """Start a new Plan or resume the active session's interrupted Plan."""
    resume_graph = arg in {"--resume", "resume", "继续"}
    plan_agent = LangGraphPlanAgent(
        llm_client=runtime.agent.llm_client,
        tool_registry=runtime.registry,
        config=runtime.config,
        cwd=runtime.cwd,
        approval_callback=runtime.agent.approval_callback,
        plan_review_callback=runtime.task_controller.request_plan_review,
        thread_id=runtime.active_session.current.id,
        resume=resume_graph,
    )
    run = run_delegated_with_session(
        plan_agent,
        "继续之前的计划" if resume_graph else arg,
        runtime.agent,
        runtime.active_session,
        runtime.console,
        runtime.task_controller,
    )
    runtime.task_controller.start(
        run,
        initial_state=TaskState.PLANNING,
        label=arg,
    )


async def run_delegated_with_session(
    delegated_agent: LangGraphPlanAgent,
    message: str,
    agent: Agent,
    active_session: ActiveSession,
    console: Console,
    task_controller: InteractiveTaskController | None = None,
) -> None:
    """Run a Plan agent while keeping the facade Agent history in sync."""
    previous_history = list(agent.history)
    agent.history = [*previous_history, Message(role="user", content=message)]
    events: AsyncIterable[AgentEvent] = delegated_agent.run(message)
    run_renderer = RichRenderer()
    try:
        await run_events(
            events,
            run_renderer,
            agent.llm_client.max_context_window,
            task_controller,
        )
        if delegated_agent.history:
            agent.history = [*previous_history, *delegated_agent.history]
    except asyncio.CancelledError:
        if delegated_agent.history:
            agent.history = [*previous_history, *delegated_agent.history]
        _finalize_cancelled(agent, run_renderer)
        raise
    except BaseException as exc:
        if delegated_agent.history:
            agent.history = [*previous_history, *delegated_agent.history]
        _finalize_failed(agent, exc)
        raise
    finally:
        active_session.save(agent.history, title=message)
        print_session_warning(console, active_session)


def _finalize_cancelled(agent: Agent, renderer: RichRenderer) -> None:
    """Close the cancelled transcript after the ephemeral summary was rendered."""
    agent.history = finalize_interrupted_history(agent.history, status="cancelled")


def _finalize_failed(agent: Agent, error: BaseException) -> None:
    agent.history = finalize_interrupted_history(
        agent.history,
        status="failed",
        detail=str(error),
    )


def print_session_warning(console: Console, active_session: ActiveSession) -> None:
    warning = active_session.take_warning()
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")
