"""Assemble the interactive runtime and route submitted prompt input."""

from __future__ import annotations

from dataclasses import dataclass
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
    ApprovalModeController,
    FixedComposerPromptSession,
    permission_key_bindings,
    prompt_message,
    prompt_placeholder,
    prompt_status,
    user_history_message,
)
from vela.llm import create_llm_client
from vela.render import RichRenderer
from vela.session import ActiveSession
from vela.storage import user_state_path
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
    approval_mode: ApprovalModeController
    renderer: RichRenderer
    active_session: ActiveSession
    task_controller: InteractiveTaskController
    mcp_manager: Any = None


# Startup ---------------------------------------------------------------------


async def start_repl(cwd: str, config: VelaConfig, *, resume: bool = False) -> None:
    console = Console()
    approval_mode = ApprovalModeController(config)
    registry, mcp_manager = await build_tool_registry(config=config, cwd=cwd)
    client = create_llm_client(config.llm)
    renderer = RichRenderer(context_window=client.max_context_window)
    renderer.banner(
        version=__version__,
        api_key_configured=bool(config.llm.api_key),
    )
    _report_mcp_problems(console, mcp_manager)
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
            approval_mode,
            task_controller,
        ),
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

    history_path = user_state_path("history", "prompt_history.txt")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    def status_provider() -> list[tuple[str, str]]:
        return prompt_status(
            model=agent.llm_client.model_name,
            stats=renderer.toolbar_status(),
            approval_mode=approval_mode.mode,
            task_state=task_controller.state,
        )

    repl_style = Style.from_dict(REPL_STYLE_RULES)
    session = FixedComposerPromptSession(
        message=prompt_message(),
        bottom_toolbar=status_provider,
        history=FileHistory(str(history_path)),
        completer=WordCompleter(SLASH_COMMANDS, ignore_case=True),
        placeholder=lambda: prompt_placeholder(task_controller),
        style=repl_style,
        key_bindings=permission_key_bindings(
            approval_mode,
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
        approval_mode=approval_mode,
        renderer=renderer,
        active_session=active_session,
        task_controller=task_controller,
        mcp_manager=mcp_manager,
    )

    with patch_stdout(raw=True):
        await _repl_loop(session, runtime)


# Input loop ------------------------------------------------------------------


async def _repl_loop(session: PromptSession, runtime: ReplRuntime) -> None:
    console = runtime.console
    active_session = runtime.active_session
    task_controller = runtime.task_controller
    draft = ""
    while True:
        try:
            user_input = await session.prompt_async(default=draft)
        except KeyboardInterrupt:
            draft = ""
            if task_controller.cancelling:
                console.print("[dim]取消仍在进行；Ctrl+D 或 /exit 可退出。[/dim]")
                continue
            if task_controller.request_cancel():
                console.print("[yellow]正在取消当前任务……[/yellow]")
                continue
            console.print("[dim]Ctrl+D 或 /exit 可退出 Vela。[/dim]")
            continue
        except EOFError:
            if task_controller.active:
                task_controller.request_cancel()
                await task_controller.wait()
            active_session.close()
            print_session_warning(console, active_session)
            console.print()
            return
        message = user_input.strip()
        if _should_hold_draft(message, task_controller):
            draft = user_input
            continue
        draft = ""
        if message:
            console.print(user_history_message(message))
        if await _dispatch_message(message, runtime):
            active_session.close()
            print_session_warning(console, active_session)
            return


def _should_hold_draft(message: str, controller: InteractiveTaskController) -> bool:
    return (
        controller.active
        and not controller.awaiting_approval
        and not controller.awaiting_plan_review
        and message != "/cancel"
    )


async def _dispatch_message(
    message: str,
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
        runtime.console.print(
            "[yellow]当前任务仍在运行；使用 /cancel、Esc 或 Ctrl+C 取消。[/yellow]"
        )
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
    approval_mode: ApprovalModeController,
    task_controller: InteractiveTaskController,
) -> str:
    console.print(
        f"[yellow]Approval required[/yellow] {request['tool_name']} "
        f"({request['danger_level']})\n{request['input']}\n"
        "[dim]输入 y 允许、n 拒绝、a 允许并切换 Auto、s 跳过；也可 /cancel。[/dim]"
    )
    answer = await task_controller.request_approval(request)
    if answer == "auto":
        approval_mode.set("auto")
        return "approve"
    return answer


def _report_mcp_problems(console: Console, manager: Any) -> None:
    """Show MCP config and startup failures that would otherwise be invisible here."""
    if manager is None:
        return
    for warning in getattr(manager, "config_warnings", []) or []:
        console.print(f"[yellow]{warning}[/yellow]")
    for name, error in (getattr(manager, "last_errors", None) or {}).items():
        console.print(f"[yellow]MCP server {name} failed to load:[/yellow] {error}")
