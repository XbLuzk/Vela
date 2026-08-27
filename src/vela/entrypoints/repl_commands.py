"""Slash commands that inspect or update REPL settings and context."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from vela.branding import CLI_NAME, PRODUCT_NAME
from vela.config import config_to_public_dict
from vela.entrypoints.model_command import handle_model_command
from vela.entrypoints.repl_tasks import print_session_warning, start_plan
from vela.entrypoints.repl_ui import ApprovalMode, approval_mode_label
from vela.memory import MemoryManager, memory_manager_for
from vela.session import ActiveSession
from vela.skill import SkillRegistry
from vela.trust import ProjectTrustStore

if TYPE_CHECKING:
    from vela.entrypoints.repl import ReplRuntime

SLASH_COMMANDS = [
    "/help",
    "/exit",
    "/clear",
    "/cancel",
    "/session",
    "/context",
    "/memory",
    "/save",
    "/status",
    "/hitl",
    "/plan",
    "/model",
    "/skill",
    "/trust",
]

INTERACTIVE_HELP = """\
Task controls
  /cancel              Cancel the current Agent, tool, or Plan task
  New task             Draft while running; Enter unlocks after completion
  Esc                   Cancel the current task
  Ctrl+C                Cancel the current task; never exits Vela
  Ctrl+D                Exit Vela
  Ctrl+V                Save a macOS clipboard image and insert an @image reference

Plan review
  execute               Confirm the displayed Plan
  modify <requirement>  Replan with your feedback
  cancel                Cancel before execution starts

Sessions
  /session              List sessions for the current project
  /session current      Show the active session
  /session resume <ref> Resume by session ID or list number
  /plan --resume        Resume that session's interrupted LangGraph plan

Project trust
  /trust                Trust project config, MCP, and Skills after restart
  /trust deny           Revoke project trust after restart

Other commands
  {commands}
"""

MOVED_COMMANDS = {
    "/config": "/status config",
    "/policy": "/status policy",
    "/tools": "/status tools",
    "/usage": "/status usage",
    "/mcp": "/status mcp",
    "/sessions": "/session list",
    "/resume": "/session resume <id|number>",
}


async def handle_slash(raw: str, runtime: ReplRuntime) -> bool:
    """Dispatch one slash command and return whether the REPL should exit."""
    command, _, rest = raw.partition(" ")
    arg = rest.strip()
    if command in {"/exit", "/quit"}:
        return True

    if command in MOVED_COMMANDS:
        runtime.console.print(f"[yellow]Moved:[/yellow] use {MOVED_COMMANDS[command]}")
    elif command in {"/help", "/cancel", "/clear"}:
        _handle_repl_command(command, runtime)
    elif command == "/session":
        _handle_session_command(arg, runtime)
    elif command in {"/context", "/memory", "/save"}:
        handle_context_command(command, arg, runtime)
    elif command == "/plan":
        if arg:
            start_plan(arg, runtime)
        else:
            runtime.console.print("[red]Usage:[/red] /plan <task>")
    elif command == "/trust":
        handle_trust_command(arg, runtime)
    elif command in {
        "/status",
        "/hitl",
        "/model",
        "/skill",
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
        print_session_warning(runtime.console, runtime.active_session)
        runtime.console.clear()


def _handle_session_command(arg: str, runtime: ReplRuntime) -> None:
    subcommand, _, reference = arg.partition(" ")
    if subcommand in {"", "list"}:
        _sessions_command(runtime.console, runtime.active_session)
    elif subcommand == "current":
        record = runtime.active_session.current
        runtime.console.print(
            f"Current {record.id} ({record.message_count} messages) · "
            f"{record.title or '(untitled)'}"
        )
    elif subcommand == "resume" and reference.strip():
        _resume_command(reference.strip(), runtime)
    else:
        runtime.console.print("[red]Usage:[/red] /session [list|current|resume <id|number>]")


def handle_trust_command(arg: str, runtime: ReplRuntime) -> None:
    normalized = arg.strip().lower()
    if normalized not in {"", "allow", "trust", "deny", "revoke"}:
        runtime.console.print("[red]Usage:[/red] /trust [deny]")
        return
    trusted = normalized not in {"deny", "revoke"}
    try:
        ProjectTrustStore().set(runtime.cwd, trusted)
    except OSError as exc:
        runtime.console.print(f"[red]Project trust could not be saved:[/red] {exc}")
        return
    state = "trusted" if trusted else "untrusted"
    runtime.console.print(f"Project marked {state}. Restart Vela to reload project resources.")


def _sessions_command(console: Console, active_session: ActiveSession) -> None:
    sessions = active_session.list(limit=20)
    print_session_warning(console, active_session)
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


def _resume_command(reference: str, runtime: ReplRuntime) -> None:
    try:
        record = runtime.active_session.switch(reference or None)
    except ValueError as exc:
        runtime.console.print(f"[red]{exc}[/red]")
        return
    print_session_warning(runtime.console, runtime.active_session)
    if record is None:
        runtime.console.print("[yellow]No matching previous session.[/yellow]")
        return
    runtime.agent.clear_history()
    runtime.agent.history = list(record.messages)
    runtime.agent.graph_thread_id = record.id
    runtime.console.print(f"Resumed {record.id} ({record.message_count} messages).")


def handle_context_command(command: str, arg: str, runtime: ReplRuntime) -> None:
    config = runtime.config
    try:
        manager = memory_manager_for(config, runtime.cwd)
    except RuntimeError as exc:
        runtime.console.print(f"[red]Memory unavailable:[/red] {exc}")
        return
    if command == "/context":
        _show_context(runtime, len(manager.list(limit=5)))
    elif command == "/memory":
        _handle_memory(arg, runtime, manager)
    elif not arg:
        runtime.console.print("[red]Usage:[/red] /save <fact>")
    else:
        memory_id = manager.save(arg, source="manual", importance=0.8)
        runtime.console.print(f"Saved memory #{memory_id}")


async def handle_settings_command(command: str, arg: str, runtime: ReplRuntime) -> None:
    if command == "/status":
        _handle_status(arg, runtime)
    elif command == "/hitl":
        _handle_hitl(arg, runtime)
    elif command == "/model":
        await handle_model_command(
            arg,
            runtime.console,
            runtime.agent,
            runtime.renderer,
        )
    else:
        _handle_skill(arg, runtime)


def _handle_status(section: str, runtime: ReplRuntime) -> None:
    section = section.strip().lower()
    if section == "":
        table = Table(title=f"{PRODUCT_NAME} Status")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("model", runtime.agent.llm_client.model_name)
        table.add_row("approval", approval_mode_label(runtime.approval_mode.mode))
        table.add_row("task", runtime.task_controller.state.value)
        table.add_row("session", runtime.active_session.current.id)
        table.add_row("tools", str(len(runtime.registry.list_names())))
        table.add_row(
            "skills",
            str(
                len(
                    SkillRegistry(
                        runtime.cwd,
                        include_project=runtime.config.project_trusted,
                    ).list()
                )
            ),
        )
        manager = runtime.mcp_manager
        mcp_count = (
            sum(1 for spec in manager.specs.values() if spec.enabled) if manager is not None else 0
        )
        table.add_row("mcp", str(mcp_count) if manager is not None else "unavailable")
        runtime.console.print(table)
    elif section == "config":
        runtime.console.print_json(
            json.dumps(config_to_public_dict(runtime.config), ensure_ascii=False)
        )
    elif section == "policy":
        policy = runtime.config.policy
        runtime.console.print_json(
            json.dumps(
                {
                    "approval_mode": policy.approval_mode,
                    "path_guard_enabled": policy.path_guard_enabled,
                    "command_guard_enabled": policy.command_guard_enabled,
                },
                ensure_ascii=False,
            )
        )
    elif section == "tools":
        runtime.console.print("\n".join(runtime.registry.list_names()) or "(no tools)")
    elif section == "usage":
        runtime.console.print_json(
            json.dumps(runtime.agent.last_usage.to_dict(), ensure_ascii=False)
        )
    elif section == "mcp":
        _handle_mcp(runtime)
    else:
        runtime.console.print(
            "[red]Unknown status section.[/red] Use /status [config|policy|tools|usage|mcp]"
        )


def _handle_mcp(runtime: ReplRuntime) -> None:
    """Show configured MCP servers together with config and load failures."""
    manager = runtime.mcp_manager
    if manager is None:
        runtime.console.print(f"MCP is disabled. Use `{CLI_NAME} mcp list` to inspect config.")
        return
    for spec in manager.specs.values():
        target = spec.url or f"{spec.command} {' '.join(spec.args)}".strip()
        state = "enabled" if spec.enabled else "disabled"
        runtime.console.print(f"{spec.name}\t{spec.type}\t{state}\t{target}")
    for warning in getattr(manager, "config_warnings", []) or []:
        runtime.console.print(f"[yellow]{warning}[/yellow]")
    for name, error in (manager.last_errors or {}).items():
        runtime.console.print(f"[yellow]MCP server {name} failed to load:[/yellow] {error}")


def _show_context(runtime: ReplRuntime, memory_count: int) -> None:
    config = runtime.config
    table = Table(title=f"{PRODUCT_NAME} Context")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("cwd", runtime.cwd)
    table.add_row("model", f"{config.llm.model} ({config.llm.provider})")
    table.add_row("context window", str(runtime.agent.llm_client.max_context_window))
    table.add_row("memory", f"{memory_count} recent entries")
    table.add_row("tools", str(len(runtime.registry.list_names())))
    runtime.console.print(table)


def _handle_memory(arg: str, runtime: ReplRuntime, manager: MemoryManager) -> None:
    sub, _, rest = arg.partition(" ")
    if sub == "clear":
        runtime.console.print(f"Cleared {manager.clear()} memories.")
    elif sub == "search":
        rows = manager.recall(rest, mark_access=False)
        runtime.console.print(
            "\n".join(f"#{row.id} {row.content}" for row in rows) or "(no matches)"
        )
    elif sub == "stats":
        runtime.console.print_json(json.dumps(manager.stats(), ensure_ascii=False))
    elif sub == "delete" and rest.strip().isdigit():
        runtime.console.print(f"Deleted: {manager.delete(int(rest.strip()))}")
    else:
        rows = manager.list()
        runtime.console.print(
            "\n".join(f"#{row.id} {row.content}" for row in rows) or "(no memories)"
        )


def _handle_hitl(arg: str, runtime: ReplRuntime) -> None:
    aliases: dict[str, ApprovalMode] = {
        "ask": "ask",
        "on": "ask",
        "auto": "auto",
        "off": "auto",
    }
    if arg in aliases:
        runtime.approval_mode.set(aliases[arg])
    elif arg:
        runtime.console.print("[red]Usage:[/red] /hitl ask|auto")
        return
    runtime.console.print(f"Approval mode: {approval_mode_label(runtime.approval_mode.mode)}")


def _handle_skill(arg: str, runtime: ReplRuntime) -> None:
    registry = SkillRegistry(
        runtime.cwd,
        include_project=runtime.config.project_trusted,
    )
    sub, _, rest = arg.partition(" ")
    if sub == "show" and rest:
        skill = registry.load(rest.strip())
        if not skill:
            runtime.console.print(f'Skill "{rest.strip()}" not found.')
            return
        runtime.console.print(skill.content[:12_000])
        return
    if sub and sub != "list":
        runtime.console.print("[red]Usage:[/red] /skill [list|show <name>]")
        return
    lines = [f"{item.name}\t{item.source}\t{item.description}" for item in registry.list()]
    runtime.console.print("\n".join(lines) or "(no skills)")
