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
from vela.entrypoints.repl_ui import PermissionMode, permission_mode_label
from vela.entrypoints.trace_command import parse_trace_args, show_run_traces
from vela.memory import MemoryManager
from vela.policy import AuditLog
from vela.run_trace import RunTraceStore
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
    "/trust",
]

INTERACTIVE_HELP = """\
Task controls
  /cancel              Cancel the current Agent, tool, or Plan task
  Esc                   Cancel the current task
  Ctrl+C                Cancel once; press again while cancelling to exit Vela
  Ctrl+V                Save a macOS clipboard image and insert an @image reference
  Enter                 Queue steering input while a ReAct task is running
  Alt+Enter             Queue a follow-up after the current task finishes

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

Project trust
  /trust                Trust project config, MCP, and Skills after restart
  /trust deny           Revoke project trust after restart

Other commands
  {commands}
"""


async def handle_slash(raw: str, runtime: ReplRuntime) -> bool:
    """Dispatch one slash command and return whether the REPL should exit."""
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
        if arg:
            start_plan(arg, runtime)
        else:
            runtime.console.print("[red]Usage:[/red] /plan <task>")
    elif command == "/trust":
        handle_trust_command(arg, runtime)
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
        print_session_warning(runtime.console, runtime.active_session)
        runtime.console.clear()


def _handle_session_command(command: str, arg: str, runtime: ReplRuntime) -> None:
    if command == "/sessions":
        _sessions_command(runtime.console, runtime.active_session)
    else:
        _resume_command(arg, runtime)


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
        manager = MemoryManager(
            config.memory.long_term_db_path,
            scope=runtime.cwd,
            max_entries=config.memory.max_long_term_entries,
            max_content_length=config.memory.max_memory_chars,
        )
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
    config = runtime.config
    if command == "/config":
        runtime.console.print_json(json.dumps(config_to_public_dict(config), ensure_ascii=False))
    elif command == "/tools":
        runtime.console.print("\n".join(runtime.registry.list_names()))
    elif command == "/hitl":
        _handle_hitl(arg, runtime)
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
        await handle_model_command(
            arg,
            runtime.console,
            runtime.agent,
            runtime.renderer,
        )
    elif command == "/usage":
        runtime.console.print_json(
            json.dumps(runtime.agent.last_usage.to_dict(), ensure_ascii=False)
        )
    elif command == "/trace":
        reference, json_output = parse_trace_args(arg)
        store = getattr(runtime.agent, "trace_store", None) or RunTraceStore()
        show_run_traces(
            runtime.console,
            store,
            reference=reference,
            json_output=json_output,
        )
    elif command == "/skill":
        _handle_skill(arg, runtime)
    else:
        runtime.console.print(f"Use `{CLI_NAME} mcp list` to inspect configured MCP servers.")


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
    aliases: dict[str, PermissionMode] = {
        "default": "default",
        "on": "default",
        "auto": "auto",
        "off": "auto",
    }
    if arg in aliases:
        runtime.permission_mode.set(aliases[arg])
    elif arg:
        runtime.console.print("[red]Usage:[/red] /hitl default|auto")
        return
    runtime.console.print(f"Permission mode: {permission_mode_label(runtime.permission_mode.mode)}")


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
