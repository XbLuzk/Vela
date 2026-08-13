"""Interactive startup, input loop, commands, and session persistence."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.table import Table

from vela import __version__
from vela.agent import Agent, LangGraphPlanAgent
from vela.bootstrap import build_tool_registry
from vela.branding import PRODUCT_NAME
from vela.config import VelaConfig
from vela.entrypoints.repl_commands import handle_context_command, handle_settings_command
from vela.entrypoints.repl_ui import (
    REPL_STYLE_RULES,
    BorderedPromptSession,
    PermissionModeController,
    permission_key_bindings,
    prompt_message,
)
from vela.events import AgentEvent
from vela.llm import create_llm_client
from vela.render import RichRenderer
from vela.run_trace import RunTraceStore, trace_finished_event
from vela.session import ActiveSession, finalize_interrupted_history
from vela.skill import SkillRegistry
from vela.task_control import InteractiveTaskController, TaskState
from vela.tools import ToolRegistry
from vela.types import Message

SLASH_COMMANDS = [
    "/help",
    "/exit",
    "/clear",
    "/cancel",
    "/sessions",
    "/resume",
    "/context",
    "/memory",
    "/save",
    "/config",
    "/tools",
    "/hitl",
    "/policy",
    "/audit",
    "/plan",
    "/model",
    "/usage",
    "/trace",
    "/skill",
    "/mcp",
]

INTERACTIVE_HELP = """\
Task controls
  /cancel              Cancel the current Agent, tool, or Plan task
  Esc                   Cancel the current task
  Ctrl+C                Cancel once; press again while cancelling to exit Vela
  Ctrl+V                Save a macOS clipboard image and insert an @image reference

Plan review
  execute               Confirm the displayed Plan
  modify <requirement>  Replan with your feedback
  cancel                Cancel before execution starts

Sessions
  /sessions             List sessions for the current project
  /resume [id|number]   Resume a previous session
  /plan --resume        Resume that session's interrupted LangGraph plan

Runs
  /trace                List recent Agent runs
  /trace <id|number>    Inspect one persisted run summary

Other commands
  {commands}
"""


@dataclass(slots=True)
class ReplRuntime:
    """Objects shared by the interactive input loop and slash commands."""

    console: Console
    cwd: str
    config: VelaConfig
    agent: Agent
    registry: ToolRegistry
    permission_mode: PermissionModeController
    renderer: RichRenderer
    active_session: ActiveSession
    task_controller: InteractiveTaskController


# Startup ---------------------------------------------------------------------


async def start_repl(cwd: str, config: VelaConfig, *, resume: bool = False) -> None:
    console = Console()
    permission_mode = PermissionModeController(config)
    registry, mcp_manager = await build_tool_registry(config=config, cwd=cwd)
    client = create_llm_client(config.llm)
    tool_count = len(registry.list_names())
    mcp_server_count = _count_mcp_servers(mcp_manager)
    skill_count = len(SkillRegistry(cwd).list())
    agents_file_count = _count_named_files(cwd, "AGENTS.md")
    renderer = RichRenderer(context_window=client.max_context_window)
    renderer.banner(
        version=__version__,
        api_key_configured=bool(config.llm.api_key),
    )
    task_controller = InteractiveTaskController(
        on_error=lambda exc: console.print(f"[red]Task failed:[/red] {exc}")
    )
    agent = Agent(
        llm_client=client,
        tool_registry=registry,
        cwd=cwd,
        config=config,
        plan_review_callback=task_controller.request_plan_review,
        approval_callback=lambda request: _approval_prompt(
            request,
            console,
            permission_mode,
            task_controller,
        ),
        trace_store=RunTraceStore(),
    )
    active_session = ActiveSession.open(cwd, resume=resume)
    agent.graph_thread_id = active_session.current.id
    agent.history = list(active_session.current.messages)
    _print_session_warning(console, active_session)
    if resume:
        if active_session.resumed:
            console.print(
                f"Resumed {active_session.current.id} "
                f"({active_session.current.message_count} messages)."
            )
        else:
            console.print(f"No previous session found. Started {active_session.current.id}.")

    history_path = Path.home() / ".vela" / "history" / "prompt_history.txt"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session = BorderedPromptSession(
        message=lambda: prompt_message(
            cwd=cwd,
            model=agent.llm_client.model_name,
            tools=tool_count,
            agents_files=agents_file_count,
            mcp_servers=mcp_server_count,
            skills=skill_count,
            stats=renderer.toolbar_status(),
            permission_mode=permission_mode.mode,
            task_state=task_controller.state,
        ),
        history=FileHistory(str(history_path)),
        completer=WordCompleter(SLASH_COMMANDS, ignore_case=True),
        placeholder=[("class:placeholder", "Type a message, @image:<path>, or Ctrl+V")],
        style=Style.from_dict(REPL_STYLE_RULES),
        key_bindings=permission_key_bindings(
            permission_mode,
            task_controller,
            console=console,
        ),
    )
    task_controller.set_callbacks(
        on_change=session.app.invalidate,
        on_error=lambda exc: console.print(f"[red]Task failed:[/red] {exc}"),
    )
    runtime = ReplRuntime(
        console=console,
        cwd=cwd,
        config=config,
        agent=agent,
        registry=registry,
        permission_mode=permission_mode,
        renderer=renderer,
        active_session=active_session,
        task_controller=task_controller,
    )

    with patch_stdout(raw=True):
        await _repl_loop(session, runtime)


# Input loop ------------------------------------------------------------------


async def _repl_loop(session: PromptSession, runtime: ReplRuntime) -> None:
    console = runtime.console
    active_session = runtime.active_session
    task_controller = runtime.task_controller
    while True:
        try:
            user_input = await session.prompt_async()
        except KeyboardInterrupt:
            if task_controller.cancelling:
                await task_controller.wait()
                active_session.close()
                _print_session_warning(console, active_session)
                console.print()
                return
            if task_controller.request_cancel():
                console.print("[yellow]正在取消当前任务；再次 Ctrl+C 将退出 Vela。[/yellow]")
                continue
            active_session.close()
            _print_session_warning(console, active_session)
            console.print()
            return
        except EOFError:
            if task_controller.active:
                task_controller.request_cancel()
                await task_controller.wait()
            active_session.close()
            _print_session_warning(console, active_session)
            console.print()
            return
        message = user_input.strip()
        if not message:
            continue
        if message == "/cancel":
            if task_controller.request_cancel():
                console.print("[yellow]正在取消当前任务……[/yellow]")
            else:
                console.print("[dim]当前没有正在运行的任务。[/dim]")
            continue
        if task_controller.awaiting_approval:
            console.print(task_controller.submit_approval(message))
            continue
        if task_controller.awaiting_plan_review:
            console.print(task_controller.submit_plan_review(message))
            continue
        if task_controller.active:
            console.print("[yellow]当前任务仍在运行；使用 /cancel、Esc 或 Ctrl+C 取消。[/yellow]")
            continue
        if message.startswith("/"):
            should_exit = await _handle_slash(message, runtime)
            if should_exit:
                active_session.close()
                _print_session_warning(console, active_session)
                return
            continue
        task_controller.start(
            _run_agent_with_session(
                runtime.agent,
                runtime.renderer,
                message,
                active_session,
                console,
                task_controller,
            ),
            initial_state=TaskState.RUNNING,
            label=message,
        )


# Agent execution and session persistence -------------------------------------


async def _run_agent(
    agent: Agent,
    renderer: RichRenderer,
    message: str,
    task_controller: InteractiveTaskController | None = None,
) -> None:
    await _run_events(
        agent.run(message),
        renderer,
        agent.llm_client.max_context_window,
        task_controller,
    )


async def _run_agent_with_session(
    agent: Agent,
    renderer: RichRenderer,
    message: str,
    active_session: ActiveSession,
    console: Console,
    task_controller: InteractiveTaskController | None = None,
) -> None:
    try:
        if task_controller is None:
            await _run_agent(agent, renderer, message)
        else:
            await _run_agent(agent, renderer, message, task_controller)
    except asyncio.CancelledError:
        agent.history = finalize_interrupted_history(agent.history, status="cancelled")
        trace = getattr(agent, "last_run_trace", None)
        if renderer is not None and trace is not None:
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
        _print_session_warning(console, active_session)


async def _run_events(
    events: AsyncIterable[AgentEvent],
    renderer: RichRenderer,
    context_window: int | None = None,
    task_controller: InteractiveTaskController | None = None,
) -> None:
    renderer.set_context_window(context_window)
    renderer.start_run()
    renderer.newline()
    pending_error: BaseException | None = None
    async for event in events:
        event_type = event.get("type")
        if pending_error is None or event_type == "run_finished":
            renderer.handle(event)
        if task_controller is not None and event.get("type") == "plan_status":
            task_controller.set_phase(str(event.get("phase") or ""))
        if event_type == "error" and pending_error is None:
            error = event.get("error")
            pending_error = error if isinstance(error, BaseException) else RuntimeError(str(error))
    renderer.newline()
    if pending_error is not None:
        raise pending_error


async def _handle_slash(raw: str, runtime: ReplRuntime) -> bool:
    command, _, rest = raw.partition(" ")
    arg = rest.strip()
    if command in {"/exit", "/quit"}:
        return True

    if command in {"/help", "/cancel", "/clear"}:
        _handle_repl_command(command, runtime)
    elif command in {"/sessions", "/resume"}:
        _handle_session_command(command, arg, runtime)
    elif command in {"/context", "/memory", "/save"}:
        handle_context_command(command, arg, runtime)
    elif command == "/plan":
        _handle_plan_command(arg, runtime)
    elif command in {
        "/config",
        "/tools",
        "/hitl",
        "/policy",
        "/audit",
        "/model",
        "/usage",
        "/trace",
        "/skill",
        "/mcp",
    }:
        await handle_settings_command(command, arg, runtime)
    else:
        runtime.console.print(f"[red]Unknown command:[/red] {command}")
    return False


def _handle_repl_command(command: str, runtime: ReplRuntime) -> None:
    if command == "/help":
        runtime.console.print(
            INTERACTIVE_HELP.format(commands="  ".join(SLASH_COMMANDS)),
            markup=False,
        )
    elif command == "/cancel":
        if runtime.task_controller.request_cancel():
            runtime.console.print("[yellow]正在取消当前任务……[/yellow]")
        else:
            runtime.console.print("[dim]当前没有正在运行的任务。[/dim]")
    else:
        runtime.agent.clear_history()
        runtime.active_session.save(runtime.agent.history)
        _print_session_warning(runtime.console, runtime.active_session)
        runtime.console.clear()


def _handle_session_command(command: str, arg: str, runtime: ReplRuntime) -> None:
    if command == "/sessions":
        _sessions_command(runtime.console, runtime.active_session)
    else:
        _resume_command(arg, runtime.console, runtime.agent, runtime.active_session)


def _handle_plan_command(arg: str, runtime: ReplRuntime) -> None:
    if not arg:
        runtime.console.print("[red]Usage:[/red] /plan <task>")
        return
    _start_plan(arg, runtime)


def _start_plan(arg: str, runtime: ReplRuntime) -> None:
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
    run = _run_delegated_with_session(
        plan_agent,
        "继续之前的计划" if resume_graph else arg,
        runtime.agent,
        runtime.active_session,
        runtime.console,
        runtime.task_controller,
    )
    runtime.task_controller.start(run, initial_state=TaskState.PLANNING, label=arg)


# Slash command helpers -------------------------------------------------------


def _sessions_command(console: Console, active_session: ActiveSession) -> None:
    sessions = active_session.list(limit=20)
    _print_session_warning(console, active_session)
    if not sessions:
        console.print("(no sessions)")
        return
    table = Table(title=f"{PRODUCT_NAME} Sessions")
    table.add_column("#", justify="right")
    table.add_column("Current")
    table.add_column("Updated")
    table.add_column("Messages", justify="right")
    table.add_column("Session")
    table.add_column("Title")
    for index, record in enumerate(sessions, start=1):
        table.add_row(
            str(index),
            "*" if record.id == active_session.current.id else "",
            record.updated_at.replace("T", " ")[:19],
            str(record.message_count),
            record.id,
            record.title,
        )
    console.print(table)


def _resume_command(
    reference: str,
    console: Console,
    agent: Agent,
    active_session: ActiveSession,
) -> None:
    try:
        record = active_session.switch(reference or None)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    _print_session_warning(console, active_session)
    if record is None:
        console.print("[yellow]No matching previous session.[/yellow]")
        return
    agent.clear_history()
    agent.history = list(record.messages)
    agent.graph_thread_id = record.id
    console.print(f"Resumed {record.id} ({record.message_count} messages).")


def _print_session_warning(console: Console, active_session: ActiveSession) -> None:
    warning = active_session.take_warning()
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")


async def _run_delegated_with_session(
    delegated_agent: LangGraphPlanAgent,
    message: str,
    agent: Agent,
    active_session: ActiveSession,
    console: Console,
    task_controller: InteractiveTaskController | None = None,
) -> None:
    previous_history = list(agent.history)
    agent.history = [*previous_history, Message(role="user", content=message)]
    events: AsyncIterable[AgentEvent] = delegated_agent.run(message)
    track_events = getattr(agent, "track_events", None)
    if callable(track_events):
        events = track_events(events, mode="plan")
    run_renderer = RichRenderer()
    try:
        await _run_events(
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
        _print_session_warning(console, active_session)


# Approval and small parsing helpers ------------------------------------------


async def _approval_prompt(
    request: dict[str, Any],
    console: Console,
    permission_mode: PermissionModeController,
    task_controller: InteractiveTaskController,
) -> str:
    console.print(
        f"[yellow]Approval required[/yellow] {request['tool_name']} "
        f"({request['danger_level']})\n{request['input']}\n"
        "[dim]输入 y 允许、n 拒绝、a 允许并切换 Auto、s 跳过；也可 /cancel。[/dim]"
    )
    answer = await task_controller.request_approval(request)
    if answer == "auto":
        permission_mode.set("auto")
        return "approve"
    return answer


def _count_mcp_servers(manager: Any) -> int:
    if manager is None:
        return 0
    return sum(1 for spec in manager.specs.values() if spec.enabled)


def _count_named_files(root: str, filename: str) -> int:
    excluded_dirs = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
    count = 0
    for _dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in excluded_dirs]
        if filename in filenames:
            count += 1
    return count
