from __future__ import annotations

import asyncio
from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from vela.config import load_config
from vela.entrypoints.repl import ReplRuntime, _handle_slash
from vela.entrypoints.repl_ui import PermissionModeController
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
            llm_client=SimpleNamespace(max_context_window=1_000),
            last_usage=Usage(),
        ),
        registry=ToolRegistry(),
        permission_mode=PermissionModeController(config),
        renderer=RichRenderer(console),
        active_session=active,
        task_controller=InteractiveTaskController(),
    )

    for command in (
        "/save remember this",
        "/memory search remember",
        "/context",
        "/hitl auto",
        "/config",
        "/policy",
        "/usage",
        "/skill list",
        "/mcp",
    ):
        asyncio.run(_handle_slash(command, runtime))

    output = stream.getvalue()
    assert "Saved memory" in output
    assert "remember this" in output
    assert "Vela Context" in output
    assert "Permission mode: Auto" in output
    assert "vela mcp list" in output
    assert runtime.permission_mode.mode == "auto"
