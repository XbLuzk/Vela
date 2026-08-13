from __future__ import annotations

import asyncio
import json

from typer.testing import CliRunner

from vela.config import load_config
from vela.entrypoints import cli
from vela.mcp import McpClientManager, load_mcp_server_specs
from vela.tools.base import ToolContext


def test_init_rag_writes_project_mcp_configuration(tmp_path) -> None:
    result = CliRunner().invoke(cli.app, ["mcp", "init-rag", "--cwd", str(tmp_path)])

    assert result.exit_code == 0
    specs = load_mcp_server_specs(tmp_path)
    assert specs["code-rag"].command == "vela-rag"
    assert specs["code-rag"].args == ["--root", str(tmp_path.resolve())]


def test_code_rag_runs_through_the_real_stdio_mcp_boundary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "service.py").write_text(
        "def resume_session(session_id):\n    return store.load(session_id)\n",
        encoding="utf-8",
    )
    CliRunner().invoke(cli.app, ["mcp", "init-rag", "--cwd", str(tmp_path)])

    async def run() -> tuple[str, str]:
        manager = McpClientManager(tmp_path)
        tools = await manager.load_tools()
        by_name = {tool.name: tool for tool in tools}
        config = load_config(project_root=tmp_path)
        config.policy.hitl_mode = "never"
        context = ToolContext(cwd=str(tmp_path), config=config)
        indexed = await by_name["mcp__code-rag__index_repository"].execute({}, context)
        searched = await by_name["mcp__code-rag__search_code"].execute(
            {"query": "resume session"}, context
        )
        return indexed.content, searched.content

    indexed, searched = asyncio.run(run())

    assert json.loads(indexed)["files"] == 1
    assert json.loads(searched)["results"][0]["path"] == "service.py"
