from __future__ import annotations

import asyncio
import sqlite3
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from vela.config import load_config
from vela.entrypoints import repl_commands
from vela.entrypoints.repl import ReplRuntime
from vela.entrypoints.repl_commands import handle_slash, handle_trust_command
from vela.entrypoints.repl_ui import ApprovalModeController
from vela.render import RichRenderer
from vela.session import ActiveSession, SessionStore
from vela.task_control import InteractiveTaskController
from vela.tools import ToolRegistry
from vela.types import Usage


def test_context_and_settings_commands_use_shared_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    config = load_config(project_root=project)
    active = ActiveSession.open(project, store=SessionStore(tmp_path / "sessions.db"))
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=160)
    runtime = ReplRuntime(
        console=console,
        cwd=str(project),
        config=config,
        agent=SimpleNamespace(
            llm_client=SimpleNamespace(max_context_window=20_000),
            last_usage=Usage(),
        ),
        registry=ToolRegistry(),
        approval_mode=ApprovalModeController(config),
        renderer=RichRenderer(console),
        active_session=active,
        task_controller=InteractiveTaskController(),
    )

    for command in (
        "/save remember this",
        "/memory search remember",
        "/context",
        "/hitl auto",
        "/status config",
        "/status policy",
        "/status usage",
        "/skill list",
        "/status mcp",
    ):
        asyncio.run(handle_slash(command, runtime))

    output = stream.getvalue()
    assert "Saved memory" in output
    assert "remember this" in output
    assert "Vela Context" in output
    assert "Approval mode: Auto" in output
    assert "vela mcp list" in output
    assert runtime.approval_mode.mode == "auto"


def test_memory_command_reports_legacy_database_without_ending_command_dispatch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    runtime, stream = _trust_runtime(tmp_path)
    db_path = Path(runtime.config.memory.long_term_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            create table memories (
                id integer primary key autoincrement,
                scope text not null,
                content text not null,
                created_at text not null
            )
            """
        )

    asyncio.run(handle_slash("/memory", runtime))
    asyncio.run(handle_slash("/help", runtime))

    output = stream.getvalue()
    assert "Memory unavailable" in output
    assert "Move or delete the old database" in output
    assert "Task controls" in output


def test_trust_command_persists_allow_and_deny_decisions(tmp_path, monkeypatch):
    saved: list[tuple[str, bool]] = []

    class Store:
        def set(self, project_root, trusted):
            saved.append((str(project_root), trusted))

    runtime, stream = _trust_runtime(tmp_path)
    monkeypatch.setattr(repl_commands, "ProjectTrustStore", Store)

    handle_trust_command("allow", runtime)
    handle_trust_command("deny", runtime)

    assert saved == [(str(tmp_path / "project"), True), (str(tmp_path / "project"), False)]
    assert "Project marked trusted" in stream.getvalue()
    assert "Project marked untrusted" in stream.getvalue()


def test_trust_command_rejects_invalid_arguments_without_persisting(tmp_path, monkeypatch):
    called = False

    class Store:
        def set(self, project_root, trusted):  # noqa: ARG002
            nonlocal called
            called = True

    runtime, stream = _trust_runtime(tmp_path)
    monkeypatch.setattr(repl_commands, "ProjectTrustStore", Store)

    handle_trust_command("maybe", runtime)

    assert not called
    assert "Usage:" in stream.getvalue()


def test_trust_command_reports_store_errors(tmp_path, monkeypatch):
    class Store:
        def set(self, project_root, trusted):  # noqa: ARG002
            raise OSError("locked")

    runtime, stream = _trust_runtime(tmp_path)
    monkeypatch.setattr(repl_commands, "ProjectTrustStore", Store)

    handle_trust_command("", runtime)

    assert "could not be saved" in stream.getvalue()
    assert "locked" in stream.getvalue()


def _trust_runtime(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    config = load_config(project_root=project)
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=160)
    return (
        ReplRuntime(
            console=console,
            cwd=str(project),
            config=config,
            agent=SimpleNamespace(
                llm_client=SimpleNamespace(max_context_window=20_000),
                last_usage=Usage(),
            ),
            registry=ToolRegistry(),
            approval_mode=ApprovalModeController(config),
            renderer=RichRenderer(console),
            active_session=ActiveSession.open(
                project,
                store=SessionStore(tmp_path / "sessions.db"),
            ),
            task_controller=InteractiveTaskController(),
        ),
        stream,
    )
