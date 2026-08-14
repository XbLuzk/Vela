"""Run ReAct and Plan tasks while preserving the active REPL session."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING

from rich.console import Console

from vela.agent import Agent, LangGraphPlanAgent
from vela.events import AgentEvent
from vela.render import RichRenderer
from vela.run_trace import trace_finished_event
from vela.session import ActiveSession, finalize_interrupted_history
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
        agent.history = finalize_interrupted_history(agent.history, status="cancelled")
        trace = getattr(agent, "last_run_trace", None)
        if trace is not None:
            renderer.handle(
                trace_finished_event(
                    trace,
                    warning=getattr(agent, "last_run_trace_warning", None),
                )
            )
            renderer.newline()
        raise
    except BaseException as exc:
        agent.history = finalize_interrupted_history(
            agent.history,
            status="failed",
            detail=str(exc),
        )
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
    """Render one event stream and raise its first runtime error after Trace completion."""
    renderer.set_context_window(context_window)
    renderer.start_run()
    renderer.newline()
    pending_error: BaseException | None = None
    async for event in events:
        event_type = event.get("type")
        if pending_error is None or event_type == "run_finished":
            renderer.handle(event)
        if task_controller is not None and event_type == "plan_status":
            task_controller.set_phase(str(event.get("phase") or ""))
        if event_type == "error" and pending_error is None:
            error = event.get("error")
            pending_error = error if isinstance(error, BaseException) else RuntimeError(str(error))
    renderer.newline()
    if pending_error is not None:
        raise pending_error


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
        accepts_steering=False,
    )


async def run_delegated_with_session(
    delegated_agent: LangGraphPlanAgent,
    message: str,
    agent: Agent,
    active_session: ActiveSession,
    console: Console,
    task_controller: InteractiveTaskController | None = None,
) -> None:
    """Run a Plan agent while keeping the facade Agent history and Trace in sync."""
    previous_history = list(agent.history)
    agent.history = [*previous_history, Message(role="user", content=message)]
    events: AsyncIterable[AgentEvent] = delegated_agent.run(message)
    track_events = getattr(agent, "track_events", None)
    if callable(track_events):
        events = track_events(events, mode="plan")
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
        agent.history = finalize_interrupted_history(agent.history, status="cancelled")
        if getattr(agent, "last_run_trace", None) is not None:
            run_renderer.handle(
                trace_finished_event(
                    agent.last_run_trace,
                    warning=getattr(agent, "last_run_trace_warning", None),
                )
            )
            run_renderer.newline()
        raise
    except BaseException as exc:
        if delegated_agent.history:
            agent.history = [*previous_history, *delegated_agent.history]
        agent.history = finalize_interrupted_history(
            agent.history,
            status="failed",
            detail=str(exc),
        )
        raise
    finally:
        active_session.save(agent.history, title=message)
        print_session_warning(console, active_session)


def print_session_warning(console: Console, active_session: ActiveSession) -> None:
    warning = active_session.take_warning()
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")
