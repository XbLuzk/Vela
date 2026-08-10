"""Interactive startup, input loop, commands, and session persistence."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from vela import __version__
from vela.agent import Agent, AgentOrchestrator, LangGraphPlanAgent
from vela.bootstrap import build_tool_registry
from vela.branding import CLI_NAME, PRODUCT_NAME
from vela.config import VelaConfig, config_to_public_dict
from vela.entrypoints.model_selector import ModelSelectorState, run_model_selector
from vela.entrypoints.repl_ui import (
    REPL_STYLE_RULES,
    BorderedPromptSession,
    PermissionMode,
    PermissionModeController,
    permission_key_bindings,
    permission_mode_label,
    prompt_message,
)
from vela.llm import create_llm_client
from vela.llm.model_profiles import (
    DEFAULT_MODEL_PROFILES,
    PROVIDER_DEFAULTS,
    CustomModelStore,
    ModelProfile,
)
from vela.memory import MemoryManager
from vela.policy import AuditLog
from vela.prompt import PromptAssembler
from vela.render import RichRenderer
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
    "/team",
    "/model",
    "/usage",
    "/skill",
    "/mcp",
]

INTERACTIVE_HELP = """\
Task controls
  /cancel              Cancel the current Agent, tool, Plan, or Team task
  Esc                   Cancel the current task
  Ctrl+C                Cancel once; press again while cancelling to exit Vela
  Ctrl+V                Save a macOS clipboard image and insert an @image reference

Plan review
  execute               Confirm the displayed Plan/Team plan
  modify <requirement>  Replan with your feedback
  cancel                Cancel before execution starts

Sessions
  /sessions             List sessions for the current project
  /resume [id|number]   Resume a previous session
  /plan --resume        Resume that session's interrupted LangGraph plan

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
        model=client.model_name,
        provider=client.provider_name,
        cwd=cwd,
        tools=tool_count,
        version=__version__,
        api_key_configured=bool(config.llm.api_key),
        mcp_servers=mcp_server_count,
        skills=skill_count,
        agents_files=agents_file_count,
        hitl_mode=config.policy.hitl_mode,
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
    events,
    renderer: RichRenderer,
    context_window: int | None = None,
    task_controller: InteractiveTaskController | None = None,
) -> None:
    renderer.set_context_window(context_window)
    renderer.start_run()
    renderer.newline()
    async for event in events:
        renderer.handle(event)
        if task_controller is not None and event.get("type") == "plan_status":
            task_controller.set_phase(str(event.get("phase") or ""))
        if event.get("type") == "error":
            raise event["error"]
    renderer.newline()


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
        await _handle_context_command(command, arg, runtime)
    elif command in {"/plan", "/team"}:
        _handle_task_command(command, arg, runtime)
    elif command in {
        "/config",
        "/tools",
        "/hitl",
        "/policy",
        "/audit",
        "/model",
        "/usage",
        "/skill",
        "/mcp",
    }:
        await _handle_settings_command(command, arg, runtime)
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


async def _handle_context_command(command: str, arg: str, runtime: ReplRuntime) -> None:
    config = runtime.config
    if command == "/context":
        memories = MemoryManager(config.memory.long_term_db_path, scope=runtime.cwd).list(limit=5)
        table = Table(title=f"{PRODUCT_NAME} Context")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("cwd", runtime.cwd)
        table.add_row("model", f"{config.llm.model} ({config.llm.provider})")
        table.add_row("context window", str(runtime.agent.llm_client.max_context_window))
        table.add_row("memory", f"{len(memories)} recent entries")
        table.add_row("tools", str(len(runtime.registry.list_names())))
        runtime.console.print(table)
    elif command == "/memory":
        await _memory_command(arg, runtime.console, runtime.cwd, config)
    else:
        if not arg:
            runtime.console.print("[red]Usage:[/red] /save <fact>")
        else:
            memory_id = MemoryManager(
                config.memory.long_term_db_path,
                scope=runtime.cwd,
                max_entries=config.memory.max_long_term_entries,
                max_content_length=config.memory.max_memory_chars,
            ).save(arg, source="manual", importance=0.8)
            runtime.console.print(f"Saved memory #{memory_id}")


def _handle_task_command(command: str, arg: str, runtime: ReplRuntime) -> None:
    if not arg:
        runtime.console.print(f"[red]Usage:[/red] {command} <task>")
        return
    if command == "/plan":
        _start_plan(arg, runtime)
    else:
        _start_team(arg, runtime)


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


def _start_team(arg: str, runtime: ReplRuntime) -> None:
    try:
        worker_mode, team_task = _parse_mode_argument(arg)
    except ValueError as exc:
        runtime.console.print(f"[red]{exc}[/red]")
        return
    orchestrator = AgentOrchestrator(
        llm_client=runtime.agent.llm_client,
        tool_registry=runtime.registry,
        config=runtime.config,
        cwd=runtime.cwd,
        approval_callback=runtime.agent.approval_callback,
        default_worker_mode=worker_mode,
        plan_review_callback=runtime.task_controller.request_plan_review,
    )
    run = _run_delegated_with_session(
        orchestrator,
        team_task,
        runtime.agent,
        runtime.active_session,
        runtime.console,
        runtime.task_controller,
    )
    runtime.task_controller.start(
        run,
        initial_state=TaskState.PLANNING,
        label=team_task,
    )


async def _handle_settings_command(command: str, arg: str, runtime: ReplRuntime) -> None:
    config = runtime.config
    if command == "/config":
        runtime.console.print_json(json.dumps(config_to_public_dict(config), ensure_ascii=False))
    elif command == "/tools":
        runtime.console.print("\n".join(runtime.registry.list_names()))
    elif command == "/hitl":
        _hitl_command(arg, runtime.console, runtime.permission_mode)
    elif command == "/policy":
        runtime.console.print_json(
            json.dumps(config_to_public_dict(config)["policy"], ensure_ascii=False)
        )
    elif command == "/audit":
        limit = int(arg or "20") if (arg or "20").isdigit() else 20
        runtime.console.print_json(
            json.dumps(AuditLog(config.policy.audit_log_path).tail(limit), ensure_ascii=False)
        )
    elif command == "/model":
        await _model_command(
            arg,
            runtime.console,
            runtime.cwd,
            config,
            runtime.agent,
            runtime.registry,
            runtime.renderer,
        )
    elif command == "/usage":
        runtime.console.print_json(
            json.dumps(runtime.agent.last_usage.to_dict(), ensure_ascii=False)
        )
    elif command == "/skill":
        _skill_command(arg, runtime.console, runtime.cwd)
    else:
        runtime.console.print(f"Use `{CLI_NAME} mcp list` to inspect configured MCP servers.")


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
    delegated_agent: LangGraphPlanAgent | AgentOrchestrator,
    message: str,
    agent: Agent,
    active_session: ActiveSession,
    console: Console,
    task_controller: InteractiveTaskController | None = None,
) -> None:
    previous_history = list(agent.history)
    agent.history = [*previous_history, Message(role="user", content=message)]
    try:
        await _run_events(
            delegated_agent.run(message),
            RichRenderer(),
            agent.llm_client.max_context_window,
            task_controller,
        )
        if delegated_agent.history:
            agent.history = [*previous_history, *delegated_agent.history]
    except asyncio.CancelledError:
        if delegated_agent.history:
            agent.history = [*previous_history, *delegated_agent.history]
        agent.history = finalize_interrupted_history(agent.history, status="cancelled")
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


async def _memory_command(arg: str, console: Console, cwd: str, config: VelaConfig) -> None:
    manager = MemoryManager(
        config.memory.long_term_db_path,
        scope=cwd,
        max_entries=config.memory.max_long_term_entries,
        max_content_length=config.memory.max_memory_chars,
    )
    sub, _, rest = arg.partition(" ")
    if sub == "clear":
        count = manager.clear()
        console.print(f"Cleared {count} memories.")
    elif sub == "search":
        rows = manager.recall(rest, mark_access=False)
        console.print("\n".join(f"#{row.id} {row.content}" for row in rows) or "(no matches)")
    elif sub == "stats":
        console.print_json(json.dumps(manager.stats(), ensure_ascii=False))
    elif sub == "delete" and rest.strip().isdigit():
        console.print(f"Deleted: {manager.delete(int(rest.strip()))}")
    else:
        rows = manager.list()
        console.print("\n".join(f"#{row.id} {row.content}" for row in rows) or "(no memories)")


def _hitl_command(
    arg: str,
    console: Console,
    permission_mode: PermissionModeController,
) -> None:
    aliases: dict[str, PermissionMode] = {
        "default": "default",
        "on": "default",
        "auto": "auto",
        "off": "auto",
    }
    if arg in aliases:
        permission_mode.set(aliases[arg])
    elif arg:
        console.print("[red]Usage:[/red] /hitl default|auto")
        return
    console.print(f"Permission mode: {permission_mode_label(permission_mode.mode)}")


# Model selection -------------------------------------------------------------


async def _model_command(
    arg: str,
    console: Console,
    cwd: str,
    config: VelaConfig,
    agent: Agent,
    registry: ToolRegistry,
    renderer: RichRenderer,
) -> None:
    if arg:
        parts = arg.split(maxsplit=1)
        provider = config.llm.provider if len(parts) == 1 else parts[0]
        model = parts[0] if len(parts) == 1 else parts[1]
        profile = next(
            (
                item
                for item in DEFAULT_MODEL_PROFILES
                if item.provider == provider.lower() and item.model == model
            ),
            None,
        )
        if profile is None:
            base_url = (
                config.llm.base_url
                if len(parts) == 1 and config.llm.base_url
                else _provider_defaults(provider)[1]
            )
            profile = ModelProfile(
                id="command-line-selection",
                name=model,
                provider=provider,
                model=model,
                base_url=base_url,
                context_window=config.llm.context_window or 128_000,
                description="Selected from /model arguments",
                api_key_env=_provider_api_key_env(provider),
            )
        _activate_model(profile, config, agent, registry, renderer, cwd)
        console.print(f"[green]Switched model:[/green] {model} ({provider})")
        return

    store = CustomModelStore()
    while True:
        state = ModelSelectorState(
            defaults=list(DEFAULT_MODEL_PROFILES),
            custom=store.list(),
            current_provider=agent.llm_client.provider_name,
            current_model=agent.llm_client.model_name,
        )
        action = await run_model_selector(state)
        if action is None:
            return
        if action.kind == "add":
            profile = _prompt_custom_model(console)
            if profile is not None:
                store.add(profile)
                _activate_model(profile, config, agent, registry, renderer, cwd)
                console.print(
                    f"[green]Saved and switched to custom model:[/green] {profile.name} "
                    f"[dim]({store.path})[/dim]"
                )
                return
            continue
        if action.kind == "delete" and action.profile is not None:
            if store.delete(action.profile.id):
                console.print(f"Deleted custom model: {action.profile.name}")
            continue
        if action.profile is not None:
            _activate_model(action.profile, config, agent, registry, renderer, cwd)
            console.print(
                f"[green]Switched model:[/green] {action.profile.name} "
                f"[dim]({action.profile.provider}/{action.profile.model})[/dim]"
            )
            return


def _prompt_custom_model(console: Console) -> ModelProfile | None:
    console.print("\n[bold]Add custom model[/bold]")
    provider = Prompt.ask(
        "Provider",
        choices=list(PROVIDER_DEFAULTS),
        default="openai-compatible",
    )
    provider_label, default_base_url, default_context = _provider_defaults(provider)
    model = Prompt.ask("Model ID").strip()
    if not model:
        console.print("[red]Model ID is required.[/red]")
        return None
    name = Prompt.ask("Display name", default=f"{provider_label} · {model}").strip()
    base_url = Prompt.ask("Base URL", default=default_base_url).strip()
    api_key_env = Prompt.ask(
        "API key environment variable",
        default=_provider_api_key_env(provider),
    ).strip()
    api_key = Prompt.ask(
        f"API key (optional; leave blank to use ${api_key_env})",
        default="",
        password=True,
        show_default=False,
    )
    context_text = Prompt.ask("Context window", default=str(default_context)).replace(",", "")
    try:
        context_window = int(context_text)
        return ModelProfile.custom_profile(
            name=name,
            provider=provider,
            model=model,
            base_url=base_url,
            context_window=context_window,
            api_key=api_key,
            api_key_env=api_key_env,
        )
    except ValueError as exc:
        console.print(f"[red]Invalid custom model:[/red] {exc}")
        return None


def _activate_model(
    profile: ModelProfile,
    config: VelaConfig,
    agent: Agent,
    registry: ToolRegistry,
    renderer: RichRenderer,
    cwd: str,
) -> None:
    old_provider = config.llm.provider.lower()
    old_api_key = config.llm.api_key
    config.llm.provider = profile.provider
    config.llm.model = profile.model
    config.llm.base_url = profile.base_url
    config.llm.context_window = profile.context_window
    config.llm.api_key = profile.resolve_api_key(
        current_provider=old_provider,
        current_api_key=old_api_key,
    )
    client = create_llm_client(config.llm)
    agent.llm_client = client
    agent.system_prompt = PromptAssembler(
        config=config,
        cwd=cwd,
        tool_names=registry.list_names(),
        model=client.model_name,
        provider=client.provider_name,
    ).build_static()
    renderer.set_context_window(client.max_context_window)


def _provider_defaults(provider: str) -> tuple[str, str, int]:
    normalized = provider.lower()
    if normalized in PROVIDER_DEFAULTS:
        return PROVIDER_DEFAULTS[normalized]
    return (provider, config_base_url(provider), 128_000)


def config_base_url(provider: str) -> str:
    from vela.llm.factory import DEEPSEEK_BASE_URL, OPENAI_BASE_URL, PROVIDER_BASE_URLS

    normalized = provider.lower()
    if normalized == "deepseek":
        return DEEPSEEK_BASE_URL
    return PROVIDER_BASE_URLS.get(normalized, OPENAI_BASE_URL)


def _provider_api_key_env(provider: str) -> str:
    return {
        "deepseek": "DEEPSEEK_API_KEY",
        "glm": "ZAI_API_KEY",
        "zhipu": "ZAI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openai-compatible": "VELA_API_KEY",
    }.get(provider.lower(), "VELA_API_KEY")


def _skill_command(arg: str, console: Console, cwd: str) -> None:
    registry = SkillRegistry(cwd)
    sub, _, rest = arg.partition(" ")
    if sub == "show" and rest:
        skill = registry.load(rest.strip())
        if not skill:
            console.print(f'Skill "{rest.strip()}" not found.')
            return
        console.print(skill.content[:12_000])
        return
    if sub == "on" and rest:
        console.print("enabled" if registry.enable(rest.strip()) else "skill not found")
        return
    if sub == "off" and rest:
        console.print("disabled" if registry.disable(rest.strip()) else "skill not found")
        return
    if sub == "reload":
        registry.reload()
        console.print("skills reloaded")
        return
    rows = registry.all_skills()
    lines = [
        f"{item.name}\t{item.source}\t{'on' if item.enabled else 'off'}\t{item.description}"
        for item in rows
    ]
    console.print("\n".join(lines) or "(no skills)")


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


def _parse_mode_argument(
    value: str,
    *,
    allowed: set[str] | None = None,
) -> tuple[str, str]:
    modes = allowed or {"react", "plan"}
    parts = value.strip().split(maxsplit=2)
    if len(parts) >= 2 and parts[0] in {"--mode", "-m"}:
        mode = parts[1].lower()
        if mode not in modes:
            raise ValueError(f"mode must be one of: {', '.join(sorted(modes))}")
        prompt = parts[2].strip() if len(parts) == 3 else ""
        if not prompt:
            raise ValueError("task text is required after --mode")
        return mode, prompt
    if parts and parts[0] == "--plan":
        prompt = value.strip()[len("--plan") :].strip()
        if not prompt:
            raise ValueError("task text is required after --plan")
        return "plan", prompt
    return "react", value.strip()
