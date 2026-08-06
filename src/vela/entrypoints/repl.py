from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import Float, FloatContainer, Window
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
from vela.image import ClipboardImageResult, grab_clipboard_image
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
from vela.rag import CodeIndex
from vela.render import RichRenderer
from vela.runtime import DurableTaskManager
from vela.session import ActiveSession, finalize_interrupted_history
from vela.skill import SkillRegistry
from vela.snapshot import SnapshotService
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
    "/index",
    "/search",
    "/plan",
    "/team",
    "/model",
    "/usage",
    "/skill",
    "/mcp",
    "/task",
    "/snapshot",
    "/restore",
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

REPL_STYLE_RULES = {
    "prompt": "bold ansiblue",
    "placeholder": "italic ansibrightblack",
    "input.rule": "ansibrightblack",
    "prompt.dim": "ansibrightblack",
    "prompt.count.agents": "bold ansiblue",
    "prompt.count.mcp": "bold ansiblue",
    "prompt.count.skills": "bold ansiblue",
    "prompt.tools": "bold ansiblue",
    "toolbar.model": "bold",
    "toolbar.ctx.bar": "ansigreen",
    "toolbar.ctx.value": "",
    "toolbar.cwd.value": "ansiblue",
    "toolbar.mode.default": "bold ansigreen",
    "toolbar.mode.auto": "bold ansiyellow",
    "toolbar.task": "bold ansimagenta",
    "toolbar.gap": "",
}


PermissionMode = Literal["default", "auto"]


class _BorderedPromptSession(PromptSession):
    def _create_layout(self):
        layout = super()._create_layout()
        layout.container = FloatContainer(
            content=layout.container,
            floats=[_input_border_float()],
        )
        return layout


@dataclass
class PermissionModeController:
    """Apply one of the two interactive permission modes to the live config."""

    config: VelaConfig
    mode: PermissionMode = "default"

    def __post_init__(self) -> None:
        self._default_hitl_mode = self.config.policy.hitl_mode
        self._default_path_guard_enabled = self.config.policy.path_guard_enabled
        self._default_command_guard_enabled = self.config.policy.command_guard_enabled
        self.set(self.mode)

    def set(self, mode: PermissionMode) -> PermissionMode:
        self.mode = mode
        if mode == "auto":
            self.config.policy.hitl_mode = "never"
            self.config.policy.path_guard_enabled = False
            self.config.policy.command_guard_enabled = False
        else:
            self.config.policy.hitl_mode = self._default_hitl_mode
            self.config.policy.path_guard_enabled = self._default_path_guard_enabled
            self.config.policy.command_guard_enabled = self._default_command_guard_enabled
        return self.mode

    def toggle(self) -> PermissionMode:
        return self.set("auto" if self.mode == "default" else "default")


async def start_repl(cwd: str, config: VelaConfig, *, resume: bool = False) -> None:
    console = Console()
    permission_mode = PermissionModeController(config)
    registry, mcp_manager = await build_tool_registry(config=config, cwd=cwd)
    client = create_llm_client(config.llm)
    system_prompt = PromptAssembler(
        config=config,
        cwd=cwd,
        tool_names=registry.list_names(),
        model=client.model_name,
        provider=client.provider_name,
    ).build_static()
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
        system_prompt=system_prompt,
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
    session = _BorderedPromptSession(
        message=lambda: _prompt_message(
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
        key_bindings=_permission_key_bindings(
            permission_mode,
            task_controller,
            console=console,
        ),
    )
    task_controller.set_callbacks(
        on_change=session.app.invalidate,
        on_error=lambda exc: console.print(f"[red]Task failed:[/red] {exc}"),
    )

    with patch_stdout(raw=True):
        await _repl_loop(
            session=session,
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


async def _repl_loop(
    *,
    session: PromptSession,
    console: Console,
    cwd: str,
    config: VelaConfig,
    agent: Agent,
    registry: ToolRegistry,
    permission_mode: PermissionModeController,
    renderer: RichRenderer,
    active_session: ActiveSession,
    task_controller: InteractiveTaskController,
) -> None:
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
            should_exit = await _handle_slash(
                message,
                console,
                cwd,
                config,
                agent,
                registry,
                permission_mode,
                renderer,
                active_session,
                task_controller,
            )
            if should_exit:
                active_session.close()
                _print_session_warning(console, active_session)
                return
            continue
        task_controller.start(
            _run_agent_with_session(
                agent,
                renderer,
                message,
                active_session,
                console,
                task_controller,
            ),
            initial_state=TaskState.RUNNING,
            label=message,
        )


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


async def _handle_slash(
    raw: str,
    console: Console,
    cwd: str,
    config: VelaConfig,
    agent: Agent,
    registry: ToolRegistry,
    permission_mode: PermissionModeController,
    renderer: RichRenderer,
    active_session: ActiveSession,
    task_controller: InteractiveTaskController | None = None,
) -> bool:
    command, _, rest = raw.partition(" ")
    arg = rest.strip()
    if command in {"/exit", "/quit"}:
        return True
    if command == "/help":
        console.print(
            INTERACTIVE_HELP.format(commands="  ".join(SLASH_COMMANDS)),
            markup=False,
        )
    elif command == "/cancel":
        if task_controller is not None and task_controller.request_cancel():
            console.print("[yellow]正在取消当前任务……[/yellow]")
        else:
            console.print("[dim]当前没有正在运行的任务。[/dim]")
    elif command == "/clear":
        agent.clear_history()
        active_session.save(agent.history)
        _print_session_warning(console, active_session)
        console.clear()
    elif command == "/sessions":
        _sessions_command(console, active_session)
    elif command == "/resume":
        _resume_command(arg, console, agent, active_session)
    elif command == "/context":
        memories = MemoryManager(config.memory.long_term_db_path, scope=cwd).list(limit=5)
        table = Table(title=f"{PRODUCT_NAME} Context")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("cwd", cwd)
        table.add_row("model", f"{config.llm.model} ({config.llm.provider})")
        table.add_row("context window", str(agent.llm_client.max_context_window))
        table.add_row("render", config.render_mode)
        table.add_row("memory", f"{len(memories)} recent entries")
        table.add_row("tools", str(len(registry.list_names())))
        console.print(table)
    elif command == "/memory":
        await _memory_command(arg, console, cwd, config)
    elif command == "/save":
        if not arg:
            console.print("[red]Usage:[/red] /save <fact>")
        else:
            memory_id = MemoryManager(
                config.memory.long_term_db_path,
                scope=cwd,
                max_entries=config.memory.max_long_term_entries,
                max_content_length=config.memory.max_memory_chars,
            ).save(arg, source="manual", importance=0.8)
            console.print(f"Saved memory #{memory_id}")
    elif command == "/config":
        console.print_json(json.dumps(config_to_public_dict(config), ensure_ascii=False))
    elif command == "/tools":
        console.print("\n".join(registry.list_names()))
    elif command == "/hitl":
        _hitl_command(arg, console, permission_mode)
    elif command == "/policy":
        console.print_json(json.dumps(config_to_public_dict(config)["policy"], ensure_ascii=False))
    elif command == "/audit":
        limit = int(arg or "20") if (arg or "20").isdigit() else 20
        console.print_json(
            json.dumps(AuditLog(config.policy.audit_log_path).tail(limit), ensure_ascii=False)
        )
    elif command == "/index":
        count = CodeIndex(cwd).rebuild(arg or ".")
        console.print(f"Indexed {count} code lines.")
    elif command == "/search":
        results = CodeIndex(cwd).search(arg, limit=20)
        output = "\n".join(f"{r.path}:{r.line}: {r.snippet}" for r in results)
        console.print(output or "(no matches)")
    elif command == "/plan":
        if not arg:
            console.print("[red]Usage:[/red] /plan <task>")
        else:
            resume_graph = arg in {"--resume", "resume", "继续"}
            plan_agent = LangGraphPlanAgent(
                llm_client=agent.llm_client,
                tool_registry=registry,
                config=config,
                cwd=cwd,
                approval_callback=agent.approval_callback,
                plan_review_callback=(
                    task_controller.request_plan_review if task_controller is not None else None
                ),
                thread_id=active_session.current.id,
                resume=resume_graph,
            )
            run = _run_delegated_with_session(
                plan_agent,
                "继续之前的计划" if resume_graph else arg,
                agent,
                active_session,
                console,
                task_controller,
            )
            if task_controller is None:
                await run
            else:
                task_controller.start(run, initial_state=TaskState.PLANNING, label=arg)
    elif command == "/team":
        if not arg:
            console.print("[red]Usage:[/red] /team <task>")
        else:
            try:
                worker_mode, team_task = _parse_mode_argument(arg)
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                return False
            orchestrator = AgentOrchestrator(
                llm_client=agent.llm_client,
                tool_registry=registry,
                config=config,
                cwd=cwd,
                approval_callback=agent.approval_callback,
                default_worker_mode=worker_mode,
                plan_review_callback=(
                    task_controller.request_plan_review if task_controller is not None else None
                ),
            )
            run = _run_delegated_with_session(
                orchestrator,
                team_task,
                agent,
                active_session,
                console,
                task_controller,
            )
            if task_controller is None:
                await run
            else:
                task_controller.start(
                    run,
                    initial_state=TaskState.PLANNING,
                    label=team_task,
                )
    elif command == "/model":
        await _model_command(arg, console, cwd, config, agent, registry, renderer)
    elif command == "/usage":
        payload = {
            "usage": agent.last_usage.to_dict(),
            "cost": agent.last_cost,
            "pricing_note": "Built-in provider prices are dated defaults and may change.",
        }
        console.print_json(json.dumps(payload, ensure_ascii=False))
    elif command == "/skill":
        _skill_command(arg, console, cwd)
    elif command == "/mcp":
        console.print(
            f"Use `{CLI_NAME} mcp serve --transport stdio|http --port 3000` to expose tools."
        )
    elif command == "/task":
        _task_command(arg, console, cwd)
    elif command == "/snapshot":
        _snapshot_command(arg, console, cwd)
    elif command == "/restore":
        if not arg:
            console.print("[red]Usage:[/red] /restore <snapshot-id-or-index>")
        else:
            record = SnapshotService(cwd).restore(arg)
            console.print(f"Restored {record.id}")
    else:
        console.print(f"[red]Unknown command:[/red] {command}")
    return False


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
        rows = manager.search(rest)
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
    console.print(f"Permission mode: {_permission_mode_label(permission_mode.mode)}")


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


def _task_command(arg: str, console: Console, cwd: str) -> None:
    manager = DurableTaskManager(Path.home() / ".vela" / "tasks" / "tasks.db", scope=cwd)
    sub, _, rest = arg.partition(" ")
    if sub == "add" and rest:
        try:
            mode, prompt = _parse_mode_argument(rest, allowed={"react", "plan", "team"})
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        task_id = manager.add(prompt, mode=mode)
        consume_hint = f"Run `{CLI_NAME} worker` or `{CLI_NAME} serve` to consume it."
        console.print(f"Queued {task_id} ({mode}). {consume_hint}")
    elif sub == "cancel" and rest:
        console.print(f"Canceled: {manager.cancel(rest.strip())}")
    elif sub == "log" and rest:
        task = manager.get(rest.strip())
        if not task:
            console.print("(task not found)")
        else:
            console.print(task.result or task.error or f"Task {task.id} is {task.status}")
    else:
        rows = manager.list(limit=20)
        console.print(
            "\n".join(
                f"{task.id} {task.status} {task.mode} attempts={task.attempts} {task.prompt[:80]}"
                for task in rows
            )
            or "(no tasks)"
        )


def _snapshot_command(arg: str, console: Console, cwd: str) -> None:
    service = SnapshotService(cwd)
    if arg == "clean":
        console.print(f"Cleaned {service.clean()} snapshots.")
        return
    rows = service.list(limit=20)
    output = "\n".join(
        f"{index}. {row.id} {row.phase} {row.created_at}" for index, row in enumerate(rows, 1)
    )
    console.print(output or "(no snapshots)")


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


def _permission_key_bindings(
    permission_mode: PermissionModeController,
    task_controller: InteractiveTaskController | None = None,
    *,
    console: Console | None = None,
    clipboard_grabber: Callable[[], ClipboardImageResult] = grab_clipboard_image,
) -> KeyBindings:
    bindings = KeyBindings()
    clipboard_jobs: set[asyncio.Task[None]] = set()

    @bindings.add(Keys.BackTab)
    def _toggle_permission_mode(event) -> None:
        permission_mode.toggle()
        event.app.invalidate()

    @bindings.add(Keys.Escape)
    def _cancel_running_task(event) -> None:
        if task_controller is not None and task_controller.request_cancel():
            event.app.current_buffer.reset()
            event.app.invalidate()

    @bindings.add(Keys.ControlV)
    def _paste_clipboard_image(event):
        async def capture() -> None:
            buffer = event.app.current_buffer
            grabbed = await asyncio.to_thread(clipboard_grabber)
            if event.app.is_done:
                return
            if grabbed.ok and grabbed.path is not None:
                buffer.insert_text(f"@image:<{grabbed.path.resolve()}> ")
            elif console is not None:
                console.print(f"[yellow]Ctrl+V 抓图失败:[/yellow] {grabbed.error}")
            event.app.invalidate()

        job = asyncio.create_task(capture())
        clipboard_jobs.add(job)
        job.add_done_callback(clipboard_jobs.discard)
        return job

    @bindings.add(Keys.Enter)
    async def _submit_after_clipboard(event) -> None:
        if clipboard_jobs:
            await asyncio.gather(*tuple(clipboard_jobs), return_exceptions=True)
        if not event.app.is_done:
            event.app.current_buffer.validate_and_handle()

    return bindings


def _permission_mode_label(mode: PermissionMode) -> str:
    return "Auto (full access)" if mode == "auto" else "Default"


def _prompt_message(
    *,
    cwd: str,
    model: str,
    tools: int,
    agents_files: int,
    mcp_servers: int,
    skills: int,
    stats: dict[str, Any] | None = None,
    permission_mode: PermissionMode = "default",
    task_state: TaskState | None = None,
) -> list[tuple[str, str]]:
    return [
        ("class:prompt.count.agents", str(agents_files)),
        ("class:prompt.dim", f" {_plural_label(agents_files, 'AGENTS.md file')} · "),
        ("class:prompt.count.mcp", str(mcp_servers)),
        ("class:prompt.dim", f" {_plural_label(mcp_servers, 'MCP server')} · "),
        ("class:prompt.count.skills", str(skills)),
        ("class:prompt.dim", f" {_plural_label(skills, 'skill')} · Tools "),
        ("class:prompt.tools", str(tools)),
        ("class:prompt.dim", "\n"),
        *(
            _bottom_toolbar(
                cwd,
                model,
                stats,
                permission_mode=permission_mode,
                task_state=task_state,
            )
        ),
        ("class:prompt.dim", "\n\n"),
        ("class:prompt", "* "),
    ]


def _bottom_toolbar(
    cwd: str,
    model: str,
    stats: dict[str, Any] | None = None,
    *,
    permission_mode: PermissionMode = "default",
    task_state: TaskState | None = None,
) -> list[tuple[str, str]]:
    stats = stats or {}
    has_usage = bool(stats.get("has_usage"))
    context_ratio = float(stats.get("context_ratio") or 0)
    context_text = _format_toolbar_percent(context_ratio) if has_usage else "0%"
    toolbar = [
        ("class:toolbar.model", model),
        ("class:toolbar.gap", "    "),
        ("class:toolbar.ctx.bar", _format_toolbar_bar(context_ratio if has_usage else 0)),
        ("class:toolbar.gap", " "),
        ("class:toolbar.ctx.value", context_text),
        ("class:toolbar.gap", "  "),
        ("class:toolbar.cwd.value", _shorten_home(cwd)),
        ("class:toolbar.gap", "  "),
        (
            f"class:toolbar.mode.{permission_mode}",
            _permission_mode_label(permission_mode),
        ),
        ("class:toolbar.gap", "  Shift+Tab"),
    ]
    if task_state is not None:
        toolbar.extend(
            [
                ("class:toolbar.gap", "  Task "),
                ("class:toolbar.task", task_state.value),
            ]
        )
    return toolbar


def _input_border_float() -> Float:
    return Float(
        content=Window(char="─", style="class:input.rule"),
        left=2,
        right=1,
        height=1,
        ycursor=True,
    )


def _plural_label(count: int, singular: str) -> str:
    return singular if count == 1 else singular + "s"


def _shorten_home(path: str) -> str:
    home = str(Path.home())
    if path == home:
        return "~"
    prefix = home + os.sep
    if path.startswith(prefix):
        return "~/" + path[len(prefix) :]
    return path


def _format_toolbar_bar(value: float, *, width: int = 12) -> str:
    bounded = max(0.0, min(value, 1.0))
    filled = round(bounded * width)
    if bounded > 0 and filled == 0:
        filled = 1
    return "█" * filled + "░" * (width - filled)


def _format_toolbar_percent(value: float) -> str:
    if value <= 0:
        return "0%"
    if value < 0.01:
        return "<1%"
    return f"{value:.0%}"
