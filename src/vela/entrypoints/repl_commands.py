"""Slash commands that inspect or update REPL settings and context."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rich.table import Table

from vela.branding import CLI_NAME, PRODUCT_NAME
from vela.config import config_to_public_dict
from vela.entrypoints.model_command import handle_model_command
from vela.entrypoints.repl_ui import PermissionMode, permission_mode_label
from vela.memory import MemoryManager
from vela.policy import AuditLog
from vela.skill import SkillRegistry

if TYPE_CHECKING:
    from vela.entrypoints.repl import ReplRuntime


def handle_context_command(command: str, arg: str, runtime: ReplRuntime) -> None:
    config = runtime.config
    manager = MemoryManager(
        config.memory.long_term_db_path,
        scope=runtime.cwd,
        max_entries=config.memory.max_long_term_entries,
        max_content_length=config.memory.max_memory_chars,
    )
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
    registry = SkillRegistry(runtime.cwd)
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
