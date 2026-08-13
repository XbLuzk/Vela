from __future__ import annotations

import asyncio
import os
from typing import Any

from vela.memory import MemoryManager
from vela.policy import CommandGuard
from vela.skill import SkillRegistry
from vela.tools import file_ops as fops
from vela.tools.base import Tool, ToolContext, ToolResult, object_schema
from vela.tools.file_ops import FileOpResult
from vela.tools.process import stop_subprocess


def get_builtin_tools() -> list[Tool]:
    """Return the built-in tools in the order shown to the model."""

    return [
        *_workspace_tools(),
        *_shell_tools(),
        *_memory_and_skill_tools(),
    ]


def _workspace_tools() -> list[Tool]:
    return [
        Tool(
            name="read_file",
            description="Read a text file from the current workspace.",
            parameters=object_schema(
                {
                    "path": {"type": "string", "description": "Path to read"},
                    "offset": {"type": "number", "description": "Start line, 1-based"},
                    "limit": {"type": "number", "description": "Maximum number of lines"},
                },
                ["path"],
            ),
            required_keys=["path"],
            handler=_read_file,
        ),
        Tool(
            name="write_file",
            description="Write a UTF-8 text file inside the current workspace.",
            parameters=object_schema(
                {
                    "path": {"type": "string", "description": "Path to write"},
                    "content": {"type": "string", "description": "File content"},
                    "append": {"type": "boolean", "description": "Append instead of overwrite"},
                },
                ["path", "content"],
            ),
            required_keys=["path", "content"],
            handler=_write_file,
            reconcile=_reconcile_write_file,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="medium",
        ),
        Tool(
            name="edit_file",
            description=(
                "Make line-based edits to a text file. Each edit replaces exact line sequences "
                "with new content. Returns a git-style diff showing the changes made."
            ),
            parameters=object_schema(
                {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_text": {
                        "type": "string",
                        "description": "Text to search for — must match exactly",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Text to replace with",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Preview changes without modifying the file",
                    },
                },
                ["path", "old_text", "new_text"],
            ),
            required_keys=["path", "old_text", "new_text"],
            handler=_edit_file,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="medium",
        ),
        Tool(
            name="list_dir",
            description="List entries in a directory inside the current workspace.",
            parameters=object_schema(
                {"path": {"type": "string", "description": "Directory path"}},
                ["path"],
            ),
            required_keys=["path"],
            handler=_list_dir,
        ),
        Tool(
            name="glob",
            description="Find files by glob pattern inside the current workspace.",
            parameters=object_schema(
                {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "limit": {"type": "number", "description": "Maximum results"},
                },
                ["pattern"],
            ),
            required_keys=["pattern"],
            handler=_glob_files,
        ),
        Tool(
            name="grep",
            description="Search text in workspace files.",
            parameters=object_schema(
                {
                    "pattern": {"type": "string", "description": "Regex or plain text pattern"},
                    "path": {"type": "string", "description": "Optional path to search"},
                    "regex": {"type": "boolean", "description": "Treat pattern as regex"},
                    "limit": {"type": "number", "description": "Maximum matches"},
                },
                ["pattern"],
            ),
            required_keys=["pattern"],
            handler=_grep,
        ),
        Tool(
            name="directory_tree",
            description=(
                "Get a recursive tree view of files and directories as indented text. "
                "Each entry shows the name and type. Files have no children, "
                "while directories always show their contents."
            ),
            parameters=object_schema(
                {
                    "path": {
                        "type": "string",
                        "description": "Directory path (default: workspace root)",
                    },
                    "max_depth": {
                        "type": "number",
                        "description": "Maximum recursion depth (default: 3)",
                    },
                    "exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Directory names to exclude from the tree",
                    },
                },
            ),
            required_keys=[],
            handler=_directory_tree,
        ),
        Tool(
            name="get_file_info",
            description=(
                "Retrieve detailed metadata about a file or directory — size, "
                "modification time, permissions, and type."
            ),
            parameters=object_schema(
                {"path": {"type": "string", "description": "Path to inspect"}},
                ["path"],
            ),
            required_keys=["path"],
            handler=_get_file_info,
        ),
    ]


def _shell_tools() -> list[Tool]:
    return [
        Tool(
            name="bash",
            description="Execute a shell command in the current workspace.",
            parameters=object_schema(
                {
                    "command": {"type": "string", "description": "Shell command"},
                    "timeout": {"type": "number", "description": "Timeout seconds"},
                },
                ["command"],
            ),
            required_keys=["command"],
            handler=_bash,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="high",
            requires_approval=True,
        ),
    ]


def _memory_and_skill_tools() -> list[Tool]:
    return [
        Tool(
            name="save_memory",
            description=(
                "Save an explicit or durable fact to long-term project memory. Use only for stable "
                "preferences, project constraints, user corrections, or reusable decisions; never "
                "store secrets, temporary task state, raw logs, or uncertain claims."
            ),
            parameters=object_schema(
                {
                    "content": {"type": "string", "description": "Durable fact to remember"},
                    "kind": {
                        "type": "string",
                        "enum": ["fact", "preference", "constraint", "correction", "decision"],
                    },
                    "importance": {"type": "number", "description": "Score from 0 to 1"},
                    "confidence": {"type": "number", "description": "Score from 0 to 1"},
                    "expires_at": {
                        "type": "string",
                        "description": "Optional ISO-8601 expiry for time-sensitive memory",
                    },
                },
                ["content"],
            ),
            required_keys=["content"],
            handler=_save_memory,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="medium",
        ),
        Tool(
            name="search_memory",
            description=(
                "Search relevance-ranked long-term project memory when prior preferences, "
                "corrections, constraints, or decisions may matter."
            ),
            parameters=object_schema(
                {
                    "query": {"type": "string", "description": "What to recall"},
                    "limit": {"type": "number", "description": "Maximum results"},
                    "kinds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional memory kinds",
                    },
                },
                ["query"],
            ),
            required_keys=["query"],
            handler=_search_memory,
        ),
        Tool(
            name="load_skill",
            description="Load a named Vela skill manual from user/project skill directories.",
            parameters=object_schema(
                {"name": {"type": "string", "description": "Skill name"}},
                ["name"],
            ),
            required_keys=["name"],
            handler=_load_skill,
        ),
    ]


# ---------------------------------------------------------------------------
# Handler: file operations (thin wrappers over file_ops)
# ---------------------------------------------------------------------------


async def _read_file(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    result: FileOpResult = fops.read_file(
        context.cwd,
        str(payload["path"]),
        offset=int(payload.get("offset") or 1),
        limit=int(payload.get("limit") or 500),
        path_guard_enabled=context.config.policy.path_guard_enabled,
    )
    return _to_tool_result(result)


async def _write_file(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    result: FileOpResult = fops.write_file(
        context.cwd,
        str(payload["path"]),
        str(payload["content"]),
        append=bool(payload.get("append")),
        path_guard_enabled=context.config.policy.path_guard_enabled,
    )
    return _to_tool_result(result)


async def _reconcile_write_file(payload: dict[str, Any], context: ToolContext) -> ToolResult | None:
    if bool(payload.get("append")):
        return None
    resolved = fops.resolve_path(
        context.cwd,
        str(payload["path"]),
        context.config.policy.path_guard_enabled,
    )
    try:
        existing = resolved.read_text(encoding="utf-8")
    except OSError:
        return None
    if existing != str(payload["content"]):
        return None
    relative = (
        resolved.relative_to(context.cwd) if resolved.is_relative_to(context.cwd) else resolved
    )
    return ToolResult(
        content=f"Wrote {relative} (reconciled existing content)",
        display_summary=f"Reconciled {relative}",
        recovery_status="reconciled",
    )


async def _edit_file(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    result: FileOpResult = fops.edit_file(
        context.cwd,
        str(payload["path"]),
        str(payload["old_text"]),
        str(payload["new_text"]),
        path_guard_enabled=context.config.policy.path_guard_enabled,
        dry_run=bool(payload.get("dry_run")),
    )
    return _to_tool_result(result)


async def _list_dir(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    result: FileOpResult = fops.list_directory(
        context.cwd,
        str(payload["path"]),
        path_guard_enabled=context.config.policy.path_guard_enabled,
    )
    return _to_tool_result(result)


async def _glob_files(payload: dict[str, Any], _context: ToolContext) -> ToolResult:
    result: FileOpResult = fops.glob_files(
        _context.cwd,
        str(payload["pattern"]),
        limit=int(payload.get("limit") or 100),
    )
    return _to_tool_result(result)


async def _grep(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    result: FileOpResult = fops.grep(
        context.cwd,
        str(payload["pattern"]),
        path=str(payload.get("path") or "."),
        limit=int(payload.get("limit") or 100),
        use_regex=bool(payload.get("regex", True)),
        path_guard_enabled=context.config.policy.path_guard_enabled,
    )
    return _to_tool_result(result)


async def _directory_tree(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    result: FileOpResult = fops.directory_tree(
        context.cwd,
        str(payload.get("path", ".")),
        max_depth=int(payload.get("max_depth") or 3),
        path_guard_enabled=context.config.policy.path_guard_enabled,
        exclude_patterns=tuple(payload.get("exclude_patterns") or ()),
    )
    return _to_tool_result(result)


async def _get_file_info(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    result: FileOpResult = fops.get_file_info(
        context.cwd,
        str(payload["path"]),
        path_guard_enabled=context.config.policy.path_guard_enabled,
    )
    return _to_tool_result(result)


# ---------------------------------------------------------------------------
# Handler: bash
# ---------------------------------------------------------------------------


async def _bash(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    command = str(payload["command"])
    if context.config.policy.command_guard_enabled:
        CommandGuard(context.config.policy.command_blacklist).validate(command)
    timeout = float(payload.get("timeout") or context.config.tools.timeout)
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=context.cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.CancelledError:
        await stop_subprocess(proc)
        raise
    except TimeoutError:
        await stop_subprocess(proc)
        return ToolResult(f"Command timed out after {timeout:.0f}s", is_error=True)
    output = (stdout + stderr).decode("utf-8", errors="replace")
    if len(output) > 20_000:
        output = output[:20_000] + "\n... [truncated]"
    return ToolResult(
        output or f"(exit {proc.returncode}, no output)",
        is_error=proc.returncode != 0,
    )


# ---------------------------------------------------------------------------
# Handler: memory
# ---------------------------------------------------------------------------


async def _save_memory(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    if not context.config.features.memory:
        return ToolResult("Long-term memory is disabled.", is_error=True)
    manager = MemoryManager(
        context.config.memory.long_term_db_path,
        scope=context.cwd,
        max_entries=context.config.memory.max_long_term_entries,
        max_content_length=context.config.memory.max_memory_chars,
    )
    memory_id = manager.save(
        str(payload["content"]),
        kind=str(payload.get("kind") or "fact"),
        source="agent",
        importance=float(payload.get("importance", 0.5)),
        confidence=float(payload.get("confidence", 1.0)),
        expires_at=payload.get("expires_at"),
    )
    return ToolResult(f"Saved memory #{memory_id}")


async def _search_memory(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    if not context.config.features.memory:
        return ToolResult("Long-term memory is disabled.", is_error=True)
    raw_kinds = payload.get("kinds")
    if raw_kinds is not None and not isinstance(raw_kinds, list):
        return ToolResult("search_memory kinds must be an array of strings.", is_error=True)
    manager = MemoryManager(
        context.config.memory.long_term_db_path,
        scope=context.cwd,
        max_entries=context.config.memory.max_long_term_entries,
        max_content_length=context.config.memory.max_memory_chars,
    )
    rows = manager.recall(
        str(payload["query"]),
        limit=int(payload.get("limit") or context.config.memory.recall_limit),
        kinds=[str(kind) for kind in raw_kinds] if raw_kinds else None,
        min_score=context.config.memory.recall_min_score,
    )
    if not rows:
        return ToolResult("(no relevant long-term memory)")
    content = "\n".join(
        f"#{row.id} [{row.kind}, importance={row.importance:.2f}] {row.content}" for row in rows
    )
    return ToolResult(content, display_summary=f"Recalled {len(rows)} memories")


# ---------------------------------------------------------------------------
# Handler: skill
# ---------------------------------------------------------------------------


async def _load_skill(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    if not context.config.features.skill:
        return ToolResult("Skills are disabled.", is_error=True)
    skill = SkillRegistry(context.cwd).load(str(payload["name"]))
    if not skill:
        return ToolResult(f'Skill "{payload["name"]}" not found.', is_error=True)
    content = skill.body or skill.content
    if len(content) > 5_000:
        content = content[:5_000] + "\n... [truncated; use /skill show for the full skill]"
    if context.skill_context_buffer:
        context.skill_context_buffer.push(skill.name, content)
        return ToolResult(
            f'Loaded skill "{skill.name}" instructions for the next model turn.',
            display_summary=f"Loaded skill {skill.name}",
        )
    return ToolResult(content, display_summary=f"Loaded skill {skill.name}")


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------


def _to_tool_result(result: FileOpResult) -> ToolResult:
    """Convert a FileOpResult to a ToolResult."""
    return ToolResult(
        content=result.content,
        is_error=result.is_error,
        display_summary=result.display_summary,
    )
