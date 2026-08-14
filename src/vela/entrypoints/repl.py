"""Assemble the interactive runtime and route submitted prompt input."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console

from vela import __version__
from vela.agent import Agent
from vela.bootstrap import build_tool_registry
from vela.config import VelaConfig
from vela.entrypoints.repl_commands import SLASH_COMMANDS, handle_slash
from vela.entrypoints.repl_tasks import print_session_warning, run_agent_with_session
from vela.entrypoints.repl_ui import (
    REPL_STYLE_RULES,
    BorderedPromptSession,
    MessageDelivery,
    MessageDeliveryController,
    PermissionModeController,
    permission_key_bindings,
    prompt_message,
)
from vela.llm import create_llm_client
from vela.render import RichRenderer
from vela.run_trace import RunTraceStore
from vela.session import ActiveSession
from vela.skill import SkillRegistry
from vela.task_control import InteractiveTaskController, TaskState
from vela.tools import ToolRegistry


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
    message_delivery: MessageDeliveryController = field(default_factory=MessageDeliveryController)


# Startup ---------------------------------------------------------------------


async def start_repl(cwd: str, config: VelaConfig, *, resume: bool = False) -> None:
    console = Console()
    permission_mode = PermissionModeController(config)
    registry, mcp_manager = await build_tool_registry(config=config, cwd=cwd)
    client = create_llm_client(config.llm)
    tool_count = len(registry.list_names())
    mcp_server_count = _count_mcp_servers(mcp_manager)
    skill_count = len(SkillRegistry(cwd, include_project=config.project_trusted).list())
    agents_file_count = _count_named_files(cwd, "AGENTS.md")
    renderer = RichRenderer(context_window=client.max_context_window)
    renderer.banner(
        version=__version__,
        api_key_configured=bool(config.llm.api_key),
    )
    task_controller = InteractiveTaskController(
        on_error=lambda exc: console.print(f"[red]Task failed:[/red] {exc}")
    )
    message_delivery = MessageDeliveryController()
    agent = Agent(
        llm_client=client,
        tool_registry=registry,
        cwd=cwd,
        config=config,
        plan_review_callback=task_controller.request_plan_review,
        steering_callback=task_controller.take_steering_message,
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
    print_session_warning(console, active_session)
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
            message_delivery=message_delivery,
            console=console,
        ),
    )

    def refresh_prompt() -> None:
        if task_controller.state in {TaskState.CANCELLED, TaskState.FAILED}:
            pending = task_controller.take_pending_messages()
            if pending:
                restored = "\n\n".join(pending)
                buffer = session.default_buffer
                separator = "\n\n" if buffer.text and restored else ""
                buffer.text = f"{buffer.text}{separator}{restored}"
                buffer.cursor_position = len(buffer.text)
        session.app.invalidate()

    task_controller.set_callbacks(
        on_change=refresh_prompt,
        on_error=lambda exc: console.print(f"[red]Task failed:[/red] {exc}"),
    )
    task_controller.set_follow_up_runner(
        lambda message: run_agent_with_session(
            agent,
            renderer,
            message,
            active_session,
            console,
            task_controller,
        )
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
        message_delivery=message_delivery,
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
                print_session_warning(console, active_session)
                console.print()
                return
            if task_controller.request_cancel():
                console.print("[yellow]正在取消当前任务；再次 Ctrl+C 将退出 Vela。[/yellow]")
                continue
            active_session.close()
            print_session_warning(console, active_session)
            console.print()
            return
        except EOFError:
            if task_controller.active:
                task_controller.request_cancel()
                await task_controller.wait()
            active_session.close()
            print_session_warning(console, active_session)
            console.print()
            return
        delivery = runtime.message_delivery.consume()
        message = user_input.strip()
        if await _dispatch_message(message, delivery, runtime):
            active_session.close()
            print_session_warning(console, active_session)
            return


async def _dispatch_message(
    message: str,
    delivery: MessageDelivery,
    runtime: ReplRuntime,
) -> bool:
    """Route one submitted prompt and report whether the REPL should exit."""
    controller = runtime.task_controller
    if not message:
        return False
    if message == "/cancel":
        if controller.request_cancel():
            runtime.console.print("[yellow]正在取消当前任务……[/yellow]")
        else:
            runtime.console.print("[dim]当前没有正在运行的任务。[/dim]")
        return False
    if controller.awaiting_approval:
        runtime.console.print(controller.submit_approval(message))
        return False
    if controller.awaiting_plan_review:
        runtime.console.print(controller.submit_plan_review(message))
        return False
    if controller.active:
        queued_as = controller.queue_message(message, delivery=delivery)
        if queued_as == "steering":
            runtime.console.print("[dim]已排队：当前轮次的工具完成后送入 Agent。[/dim]")
        else:
            runtime.console.print("[dim]已排队：当前任务完成后继续执行。[/dim]")
        return False
    if message.startswith("/"):
        return await handle_slash(message, runtime)

    controller.start(
        run_agent_with_session(
            runtime.agent,
            runtime.renderer,
            message,
            runtime.active_session,
            runtime.console,
            controller,
        ),
        initial_state=TaskState.RUNNING,
        label=message,
    )
    return False


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
