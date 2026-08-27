from __future__ import annotations

import asyncio

from vela.config import load_config
from vela.entrypoints.repl_ui import ApprovalModeController
from vela.tools.base import Tool, ToolContext, ToolResult, object_schema
from vela.tools.executor import ToolExecutor
from vela.tools.registry import ToolRegistry


async def _execute_all(executor, calls, context):
    return [result async for result in executor.execute_stream(calls, context)]


def test_auto_mode_runs_approval_required_tool_without_callback(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config()
    controller = ApprovalModeController(config)
    executions: list[str] = []

    async def mutate(payload, _context):
        executions.append(str(payload["value"]))
        return ToolResult("done")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="mutate",
            description="Mutate test state",
            parameters=object_schema({"value": {"type": "string"}}, ["value"]),
            required_keys=["value"],
            handler=mutate,
            is_read_only=False,
            requires_approval=True,
        )
    )
    executor = ToolExecutor(registry)
    call = {"id": "call-1", "name": "mutate", "arguments": {"value": "ok"}}
    context = ToolContext(cwd=str(tmp_path), config=config)

    denied = asyncio.run(_execute_all(executor, [call], context))[0]
    assert denied.is_error
    assert executions == []

    controller.set("auto")
    approved = asyncio.run(_execute_all(executor, [call], context))[0]
    assert not approved.is_error
    assert executions == ["ok"]
