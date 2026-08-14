from __future__ import annotations

from vela.config import VelaConfig
from vela.mcp import McpClientManager
from vela.tools import ToolRegistry, get_builtin_tools


async def build_tool_registry(
    *,
    config: VelaConfig,
    cwd: str,
) -> tuple[ToolRegistry, McpClientManager | None]:
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    manager: McpClientManager | None = None
    if config.features.mcp:
        manager = McpClientManager(cwd, include_project=config.project_trusted)
        registry.register_all(await manager.load_tools())
    return registry, manager
