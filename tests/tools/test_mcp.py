from __future__ import annotations

import asyncio
import json

from vela.config import load_config
from vela.mcp import McpClientManager, load_mcp_server_specs
from vela.mcp.client import _stdio_environment
from vela.tools.base import ToolContext


def test_mcp_client_registers_and_calls_stdio_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    server = tmp_path / "fake_mcp_server.py"
    server.write_text(
        """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake")

@mcp.tool()
def echo(text: str) -> str:
    return "echo:" + text

if __name__ == "__main__":
    mcp.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / ".vela").mkdir()
    (tmp_path / ".vela" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fake": {
                        "type": "stdio",
                        "command": "python",
                        "args": [str(server)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    async def run():
        manager = McpClientManager(tmp_path)
        tools = await manager.load_tools()
        names = [tool.name for tool in tools]
        tool = next(item for item in tools if item.name == "mcp__fake__echo")
        config = load_config(project_root=tmp_path)
        config.policy.hitl_mode = "never"
        result = await tool.execute({"text": "ok"}, ToolContext(cwd=str(tmp_path), config=config))
        return names, result

    names, result = asyncio.run(run())
    assert "mcp__fake__echo" in names
    assert result.content == "echo:ok"


def test_mcp_client_suppresses_stdio_server_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    server = tmp_path / "noisy_mcp_server.py"
    server.write_text(
        """
import sys
from mcp.server.fastmcp import FastMCP

sys.stderr.write("NOISY_MCP_STARTUP\\n")
sys.stderr.flush()

mcp = FastMCP("noisy")

@mcp.tool()
def echo(text: str) -> str:
    return text

if __name__ == "__main__":
    mcp.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / ".vela").mkdir()
    (tmp_path / ".vela" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "noisy": {
                        "type": "stdio",
                        "command": "python",
                        "args": [str(server)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    async def run():
        manager = McpClientManager(tmp_path)
        return await manager.load_tools()

    tools = asyncio.run(run())

    assert any(tool.name == "mcp__noisy__echo" for tool in tools)
    captured = capsys.readouterr()
    assert "NOISY_MCP_STARTUP" not in captured.err


def test_mcp_stdio_environment_does_not_inherit_parent_secrets(monkeypatch):
    monkeypatch.setenv("VELA_API_KEY", "model-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "repository-secret")
    monkeypatch.setenv("DB_PASSWORD", "database-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "cloud-secret")
    monkeypatch.setenv("SAFE_SETTING", "not-automatically-safe")
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")

    env = _stdio_environment({"GITHUB_TOKEN": "explicit-mcp-token"})

    assert "VELA_API_KEY" not in env
    assert "DB_PASSWORD" not in env
    assert "AWS_ACCESS_KEY_ID" not in env
    assert "SAFE_SETTING" not in env
    assert env["GITHUB_TOKEN"] == "explicit-mcp-token"
    assert env["LANG"] == "zh_CN.UTF-8"


def test_untrusted_mcp_loading_keeps_user_servers_and_ignores_project(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".vela").mkdir(parents=True)
    (project / ".vela").mkdir(parents=True)
    (home / ".vela" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"user": {"command": "user-server"}}}),
        encoding="utf-8",
    )
    (project / ".vela" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"project": {"command": "project-server"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    specs = load_mcp_server_specs(project, include_project=False)

    assert set(specs) == {"user"}


def test_untrusted_mcp_manager_does_not_register_builtin_code_rag(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    manager = McpClientManager(tmp_path, include_project=False)

    assert "code-rag" not in manager.specs
