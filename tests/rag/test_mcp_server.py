from __future__ import annotations

import asyncio
import json
import sys

from vela.config import load_config
from vela.mcp import McpClientManager
from vela.tools.base import ToolContext


def test_code_rag_is_registered_automatically_for_trusted_projects(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VELA_RAG_EMBEDDING_API_KEY", "rag-secret")
    monkeypatch.setenv("VELA_RAG_EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("GITHUB_TOKEN", "unrelated-secret")
    manager = McpClientManager(tmp_path)

    assert manager.specs["code-rag"].command == sys.executable
    assert manager.specs["code-rag"].args == [
        "-m",
        "vela_rag.server",
        "--root",
        str(tmp_path.resolve()),
    ]
    assert manager.specs["code-rag"].env == {
        "VELA_RAG_EMBEDDING_API_KEY": "rag-secret",
        "VELA_RAG_EMBEDDING_MODEL": "embedding-model",
    }
    assert manager.specs["code-rag"].timeout == 300.0
    assert not (tmp_path / ".vela" / "mcp.json").exists()


def test_code_rag_runs_through_the_real_stdio_mcp_boundary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "service.py").write_text(
        "def resume_session(session_id):\n    return store.load(session_id)\n",
        encoding="utf-8",
    )

    async def run() -> tuple[list[str], float, str, str]:
        manager = McpClientManager(tmp_path)
        tools = await manager.load_tools()
        by_name = {tool.name: tool for tool in tools}
        config = load_config(project_root=tmp_path)
        config.policy.approval_mode = "auto"
        context = ToolContext(cwd=str(tmp_path), config=config)
        searched = await by_name["mcp__code-rag__search_code"].execute(
            {"query": "resume session"}, context
        )
        (tmp_path / "service.py").write_text(
            "def fork_session(session_id):\n    return store.copy(session_id)\n",
            encoding="utf-8",
        )
        refreshed = await by_name["mcp__code-rag__search_code"].execute(
            {"query": "fork session"}, context
        )
        return (
            list(by_name),
            by_name["mcp__code-rag__search_code"].timeout,
            searched.content,
            refreshed.content,
        )

    names, search_timeout, searched, refreshed = asyncio.run(run())

    assert set(names) >= {
        "mcp__code-rag__search_code",
        "mcp__code-rag__rag_status",
    }
    assert "mcp__code-rag__index_repository" not in names
    assert search_timeout == 300.0
    assert json.loads(searched)["results"][0]["path"] == "service.py"
    assert "fork_session" in json.loads(refreshed)["results"][0]["content"]
