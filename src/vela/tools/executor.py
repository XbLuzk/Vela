from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from vela.policy import AuditLog
from vela.tools.base import Tool, ToolContext, ToolDecision, ToolResult
from vela.tools.journal import ToolExecutionJournal, execution_identity
from vela.tools.registry import ToolRegistry


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
            name = _tool_call_name(call)
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
        name = _tool_call_name(call)
        payload = _tool_call_arguments(call)

        if not tool:
            return ToolResult(
                tool_use_id=tool_call_id,
                content=(
                    f'Tool "{name}" not found. Available tools: '
                    f"{', '.join(self.registry.list_names())}"
                ),
                is_error=True,
            )

        audit = AuditLog(context.config.policy.audit_log_path)
        approver = "none"
        journal: ToolExecutionJournal | None = None
        execution_key = ""
        execution_claimed = False
        try:
            # 1. Validate input and recover any durable result from an earlier run.
            data = tool.validate(payload)
            if context.execution_scope and not tool.is_read_only:
                execution_key, input_hash = execution_identity(
                    context.execution_scope,
                    sequence,
                    tool.name,
                    data,
                )
                journal_path = context.config.tools.execution_journal_path
                journal = self._journals.get(journal_path)
                if journal is None:
                    journal = ToolExecutionJournal(journal_path)
                    self._journals[journal_path] = journal
                existing = journal.get(execution_key)
                if existing is not None and existing.status == "completed":
                    return _replayed_result(existing.result, tool_call_id, execution_key)
                if existing is not None and existing.status in {"running", "uncertain"}:
                    reconciled = await _reconcile(tool, data, context)
                    if reconciled is not None:
                        reconciled.tool_use_id = tool_call_id
                        reconciled.replayed = True
                        reconciled.execution_key = execution_key
                        reconciled.recovery_status = reconciled.recovery_status or "reconciled"
                        journal.complete(execution_key, reconciled)
                        return reconciled

            # 2. Apply the current approval policy before claiming a new mutation.
            decision = await self._approval_decision(tool, data, context)
            if decision in {"deny", "skip"}:
                approver = "hitl"
                audit.record(
                    tool_name=tool.name,
                    input_data=data,
                    outcome=decision,
                    approver=approver,
                    cwd=context.cwd,
                )
                return ToolResult(
                    tool_use_id=tool_call_id,
                    content=f'Tool "{tool.name}" was {decision}ed by approval policy.',
                    is_error=True,
                )
            if tool.requires_approval or context.config.policy.hitl_mode == "always":
                approver = "hitl"

            # 3. Claim this mutation so resume cannot silently execute it twice.
            if journal is not None:
                claim = journal.claim(
                    execution_key=execution_key,
                    scope=str(context.execution_scope),
                    sequence=sequence,
                    tool_name=tool.name,
                    input_hash=input_hash,
                    allow_uncertain_retry=context.allow_uncertain_retry,
                )
                if claim.action == "replay":
                    return _replayed_result(claim.record.result, tool_call_id, execution_key)
                if claim.action == "uncertain":
                    return ToolResult(
                        tool_use_id=tool_call_id,
                        content=(
                            f'Tool "{tool.name}" has an uncertain previous execution. '
                            "Resume the Plan and explicitly confirm retry before running it again."
                        ),
                        is_error=True,
                        execution_key=execution_key,
                        recovery_status="uncertain",
                    )
                execution_claimed = True

            # 4. Execute the tool, then persist and audit its final outcome.
            try:
                result = await tool.execute(data, context)
            except asyncio.CancelledError:
                if journal is not None:
                    journal.mark_uncertain(execution_key)
                raise
            result.tool_use_id = tool_call_id
            result.execution_key = execution_key or None
            if journal is not None:
                journal.complete(execution_key, result)
            if not tool.is_read_only and context.config.features.audit_log:
                audit.record(
                    tool_name=tool.name,
                    input_data=data,
                    outcome="allow" if not result.is_error else "error",
                    approver=approver,
                    cwd=context.cwd,
                )
            return result
        except Exception as exc:  # noqa: BLE001 - tool errors must flow back to the model
            if context.config.features.audit_log and tool and not tool.is_read_only:
                audit.record(
                    tool_name=tool.name,
                    input_data=payload,
                    outcome="error",
                    approver=approver,
                    cwd=context.cwd,
                )
            result = ToolResult(
                tool_use_id=tool_call_id,
                content=f'Tool "{name}" execution error: {exc}',
                is_error=True,
                execution_key=execution_key or None,
                recovery_status="uncertain" if execution_claimed else None,
            )
            if journal is not None and execution_claimed:
                journal.mark_uncertain(execution_key)
            return result

    async def _approval_decision(
        self,
        tool: Tool,
        payload: dict[str, Any],
        context: ToolContext,
    ) -> ToolDecision:
        mode = context.config.policy.hitl_mode
        if mode == "never":
            return "approve"
        if mode == "auto" and not tool.requires_approval:
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


def _tool_call_name(call: dict[str, Any]) -> str:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    return str(function.get("name") or call.get("name") or "")


def _tool_call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    arguments = function.get("arguments", call.get("arguments", {}))
    if isinstance(arguments, str):
        import json

        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            parsed = {"raw": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return arguments if isinstance(arguments, dict) else {}


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
