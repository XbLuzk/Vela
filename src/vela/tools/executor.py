from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from vela.tools.base import Tool, ToolContext, ToolDecision, ToolResult
from vela.tools.calls import tool_call_arguments, tool_call_name
from vela.tools.journal import ToolExecutionJournal, execution_identity
from vela.tools.registry import ToolRegistry


@dataclass(slots=True)
class _MutationExecution:
    """Durable identity for one state-changing tool call."""

    journal: ToolExecutionJournal
    key: str
    input_hash: str


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._journals: dict[str, ToolExecutionJournal] = {}

    async def execute_all(
        self,
        calls: list[dict[str, Any]],
        context: ToolContext,
    ) -> list[ToolResult]:
        return [result async for result in self.execute_stream(calls, context)]

    async def execute_stream(
        self,
        calls: list[dict[str, Any]],
        context: ToolContext,
    ) -> AsyncIterator[ToolResult]:
        """Yield results as each tool finishes so cancellation cannot hide completed work."""

        read_calls: list[tuple[dict[str, Any], Tool, int]] = []
        sequential_calls: list[tuple[dict[str, Any], Tool | None, int]] = []

        for call in calls:
            name = tool_call_name(call)
            tool = self.registry.get(name)
            sequence = -1
            if tool is not None and not tool.is_read_only:
                sequence = context.tool_sequence
                context.tool_sequence += 1
            if tool and tool.is_read_only and tool.is_concurrency_safe:
                read_calls.append((call, tool, sequence))
            else:
                sequential_calls.append((call, tool, sequence))

        if read_calls:
            semaphore = asyncio.Semaphore(context.config.tools.max_concurrent_read)

            async def run_read(call: dict[str, Any], tool: Tool, sequence: int) -> ToolResult:
                async with semaphore:
                    return await self._execute_single(call, tool, context, sequence)

            tasks = [
                asyncio.create_task(run_read(call, tool, sequence))
                for call, tool, sequence in read_calls
            ]
            try:
                for completed in asyncio.as_completed(tasks):
                    yield await completed
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        for call, tool, sequence in sequential_calls:
            yield await self._execute_single(call, tool, context, sequence)

    async def _execute_single(
        self,
        call: dict[str, Any],
        tool: Tool | None,
        context: ToolContext,
        sequence: int,
    ) -> ToolResult:
        tool_call_id = str(call.get("id") or "")
        name = tool_call_name(call)
        payload = tool_call_arguments(call)

        if not tool:
            return ToolResult(
                tool_use_id=tool_call_id,
                content=(
                    f'Tool "{name}" not found. Available tools: '
                    f"{', '.join(self.registry.list_names())}"
                ),
                is_error=True,
            )

        mutation: _MutationExecution | None = None
        execution_claimed = False
        try:
            data = tool.validate(payload)
            mutation, recovered = await self._prepare_mutation(
                tool,
                data,
                context,
                sequence,
                tool_call_id,
            )
            if recovered is not None:
                return recovered

            denied = await self._authorize(
                tool,
                data,
                context,
                tool_call_id,
            )
            if denied is not None:
                return denied

            if mutation is not None:
                blocked = self._claim_mutation(
                    mutation,
                    tool,
                    context,
                    sequence,
                    tool_call_id,
                )
                if blocked is not None:
                    return blocked
                execution_claimed = True

            return await self._run_tool(tool, data, context, tool_call_id, mutation)
        except Exception as exc:  # noqa: BLE001 - tool errors must flow back to the model
            result = ToolResult(
                tool_use_id=tool_call_id,
                content=f'Tool "{name}" execution error: {exc}',
                is_error=True,
                execution_key=mutation.key if mutation is not None else None,
                recovery_status="uncertain" if execution_claimed else None,
            )
            if mutation is not None and execution_claimed:
                mutation.journal.mark_uncertain(mutation.key)
            return result

    async def _prepare_mutation(
        self,
        tool: Tool,
        payload: dict[str, Any],
        context: ToolContext,
        sequence: int,
        tool_call_id: str,
    ) -> tuple[_MutationExecution | None, ToolResult | None]:
        if not context.execution_scope or tool.is_read_only:
            return None, None

        key, input_hash = execution_identity(
            context.execution_scope,
            sequence,
            tool.name,
            payload,
        )
        journal = self._journal(context.config.tools.execution_journal_path)
        mutation = _MutationExecution(journal=journal, key=key, input_hash=input_hash)
        existing = journal.get(key)
        if existing is not None and existing.status == "completed":
            return mutation, _replayed_result(existing.result, tool_call_id, key)
        if existing is None or existing.status not in {"running", "uncertain"}:
            return mutation, None

        reconciled = await _reconcile(tool, payload, context)
        if reconciled is None:
            return mutation, None
        reconciled.tool_use_id = tool_call_id
        reconciled.replayed = True
        reconciled.execution_key = key
        reconciled.recovery_status = reconciled.recovery_status or "reconciled"
        journal.complete(key, reconciled)
        return mutation, reconciled

    def _claim_mutation(
        self,
        mutation: _MutationExecution,
        tool: Tool,
        context: ToolContext,
        sequence: int,
        tool_call_id: str,
    ) -> ToolResult | None:
        claim = mutation.journal.claim(
            execution_key=mutation.key,
            scope=str(context.execution_scope),
            sequence=sequence,
            tool_name=tool.name,
            input_hash=mutation.input_hash,
            allow_uncertain_retry=context.allow_uncertain_retry,
        )
        if claim.action == "replay":
            return _replayed_result(claim.record.result, tool_call_id, mutation.key)
        if claim.action == "uncertain":
            return ToolResult(
                tool_use_id=tool_call_id,
                content=(
                    f'Tool "{tool.name}" has an uncertain previous execution. '
                    "Resume the Plan and explicitly confirm retry before running it again."
                ),
                is_error=True,
                execution_key=mutation.key,
                recovery_status="uncertain",
            )
        return None

    async def _run_tool(
        self,
        tool: Tool,
        payload: dict[str, Any],
        context: ToolContext,
        tool_call_id: str,
        mutation: _MutationExecution | None,
    ) -> ToolResult:
        try:
            result = await tool.execute(payload, context)
        except asyncio.CancelledError:
            if mutation is not None:
                mutation.journal.mark_uncertain(mutation.key)
            raise
        result.tool_use_id = tool_call_id
        result.execution_key = mutation.key if mutation is not None else None
        if mutation is not None:
            mutation.journal.complete(mutation.key, result)
        return result

    def _journal(self, path: str) -> ToolExecutionJournal:
        journal = self._journals.get(path)
        if journal is None:
            journal = ToolExecutionJournal(path)
            self._journals[path] = journal
        return journal

    async def _authorize(
        self,
        tool: Tool,
        payload: dict[str, Any],
        context: ToolContext,
        tool_call_id: str,
    ) -> ToolResult | None:
        decision = await self._approval_decision(tool, payload, context)
        if decision not in {"deny", "skip"}:
            return None

        return ToolResult(
            tool_use_id=tool_call_id,
            content=f'Tool "{tool.name}" was {decision}ed by approval policy.',
            is_error=True,
        )

    async def _approval_decision(
        self,
        tool: Tool,
        payload: dict[str, Any],
        context: ToolContext,
    ) -> ToolDecision:
        mode = context.config.policy.approval_mode
        if mode == "auto":
            return "approve"
        if not tool.requires_approval:
            return "approve"
        if not context.approval_callback:
            return "deny"
        result = context.approval_callback(
            {
                "tool_name": tool.name,
                "input": payload,
                "danger_level": tool.danger_level,
                "description": tool.description,
            }
        )
        if asyncio.iscoroutine(result):
            result = await result
        return result


async def _reconcile(
    tool: Tool,
    payload: dict[str, Any],
    context: ToolContext,
) -> ToolResult | None:
    if tool.reconcile is None:
        return None
    return await tool.reconcile(payload, context)


def _replayed_result(
    result: ToolResult | None,
    tool_call_id: str,
    execution_key: str,
) -> ToolResult:
    if result is None:
        raise RuntimeError("completed tool journal entry has no result")
    return ToolResult(
        content=result.content,
        is_error=result.is_error,
        display_summary=result.display_summary,
        tool_use_id=tool_call_id,
        replayed=True,
        execution_key=execution_key,
        recovery_status="replayed",
    )
