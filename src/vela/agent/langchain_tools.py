"""Vela tool scheduling and persistence inside LangChain's agent graph."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from vela.agent.langchain_model import to_openai_tool_call
from vela.tools.base import ToolContext, ToolResult
from vela.tools.executor import ToolExecutor
from vela.tools.registry import ToolRegistry
from vela.types import Message


@dataclass(slots=True)
class _ToolBatch:
    calls: list[dict[str, Any]]
    task: asyncio.Task[list[ToolMessage]] | None = None
    assigned_indices: set[int] = field(default_factory=set)
    remaining_requests: int = 0


class VelaToolMiddleware(AgentMiddleware):
    """Route LangChain tool calls through Vela's scheduler, policy, and journal."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        context: ToolContext,
        transcript: list[Message],
    ) -> None:
        super().__init__()
        self.registry = registry
        self.context = context
        self.transcript = transcript
        self.executor = ToolExecutor(registry)
        self.batches: dict[tuple[str, ...], _ToolBatch] = {}

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        del handler
        batch_key, index, batch = self._batch_for(request)
        if batch.task is None:
            batch.task = asyncio.create_task(self._execute_batch(batch.calls, request.runtime))
        try:
            return (await batch.task)[index]
        finally:
            batch.remaining_requests -= 1
            if batch.remaining_requests == 0:
                self.batches.pop(batch_key, None)

    def _batch_for(self, request: ToolCallRequest) -> tuple[tuple[str, ...], int, _ToolBatch]:
        messages = list(request.state.get("messages") or [])
        assistant = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        calls = list(assistant.tool_calls if assistant else [request.tool_call])
        key = tuple(str(call.get("id") or index) for index, call in enumerate(calls))
        batch = self.batches.get(key)
        if batch is None:
            batch = _ToolBatch(calls, remaining_requests=len(calls))
            self.batches[key] = batch
        index = _claim_call_index(calls, request.tool_call, batch.assigned_indices)
        batch.assigned_indices.add(index)
        return key, index, batch

    async def _execute_batch(self, calls: list[dict[str, Any]], runtime: Any) -> list[ToolMessage]:
        indexed_calls: list[dict[str, Any]] = []
        for index, call in enumerate(calls):
            converted = to_openai_tool_call(call)
            converted["id"] = str(index)
            indexed_calls.append(converted)

        messages: list[ToolMessage | None] = [None] * len(calls)
        async for result in self.executor.execute_stream(indexed_calls, self.context):
            index = int(result.tool_use_id)
            messages[index] = self._result_message(calls[index], result, runtime)
        return cast("list[ToolMessage]", messages)

    def _result_message(
        self,
        call: dict[str, Any],
        result: ToolResult,
        runtime: Any,
    ) -> ToolMessage:
        name = str(call.get("name") or "unknown")
        content = result.content
        if name == "load_skill":
            loaded = _drain_skill_context(self.context.skill_context_buffer)
            if loaded:
                content = f"{content}\n\n{loaded}"

        event = _tool_result_event(name, content, result)
        runtime.stream_writer(event)
        tool_call_id = str(call.get("id") or "")
        self.transcript.append(Message(role="tool", content=content, tool_call_id=tool_call_id))
        return ToolMessage(
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            status="error" if result.is_error else "success",
        )


def langchain_tools(registry: ToolRegistry) -> list[StructuredTool]:
    async def unreachable(**_kwargs: Any) -> str:
        raise RuntimeError("Vela tool middleware was bypassed.")

    return [
        StructuredTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.parameters,
            coroutine=unreachable,
        )
        for name in registry.list_names()
        if (tool := registry.get(name)) is not None
    ]


def _tool_result_event(name: str, content: str, result: ToolResult) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "name": name,
        "result": content,
        "is_error": result.is_error,
        "replayed": result.replayed,
        "execution_key": result.execution_key,
        "recovery_status": result.recovery_status,
    }


def _drain_skill_context(skill_context_buffer: Any) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return ""
    return skill_context_buffer.drain()


def _claim_call_index(
    calls: list[dict[str, Any]],
    requested: dict[str, Any],
    assigned: set[int],
) -> int:
    """Identify this invocation even when a provider repeats tool-call IDs."""

    available = [index for index in range(len(calls)) if index not in assigned]
    for index in available:
        if calls[index] is requested or calls[index] == requested:
            return index
    return available[0] if available else 0
