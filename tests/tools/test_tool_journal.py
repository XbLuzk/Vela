from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from vela.config import load_config
from vela.tools import ToolRegistry, get_builtin_tools
from vela.tools.base import Tool, ToolContext, ToolResult, object_schema
from vela.tools.executor import ToolExecutor
from vela.tools.journal import ToolExecutionJournal, execution_identity


def _config(tmp_path):
    config = load_config(project_root=tmp_path)
    config.policy.approval_mode = "auto"
    config.tools.execution_journal_path = str(tmp_path / "state" / "tool-executions.sqlite")
    return config


def _mutating_registry(handler) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="mutate",
            description="Mutate test state",
            parameters=object_schema({"value": {"type": "string"}}, ["value"]),
            required_keys=["value"],
            handler=handler,
            is_read_only=False,
            is_concurrency_safe=False,
        )
    )
    return registry


def test_completed_mutating_tool_is_replayed_without_second_execution(tmp_path):
    executions: list[str] = []

    async def mutate(payload, _context):
        executions.append(str(payload["value"]))
        return ToolResult("mutation completed")

    config = _config(tmp_path)
    executor = ToolExecutor(_mutating_registry(mutate))
    call = {"id": "first-id", "name": "mutate", "arguments": {"value": "one"}}

    first = asyncio.run(
        executor.execute_all(
            [call],
            ToolContext(cwd=str(tmp_path), config=config, execution_scope="session:plan:task"),
        )
    )[0]
    replay_call = {**call, "id": "resumed-id"}
    replayed = asyncio.run(
        executor.execute_all(
            [replay_call],
            ToolContext(cwd=str(tmp_path), config=config, execution_scope="session:plan:task"),
        )
    )[0]

    assert executions == ["one"]
    assert not first.replayed
    assert replayed.replayed
    assert replayed.recovery_status == "replayed"
    assert replayed.tool_use_id == "resumed-id"
    assert replayed.content == "mutation completed"
    assert (tmp_path / "state" / "tool-executions.sqlite").stat().st_mode & 0o777 == 0o600


def test_identical_calls_keep_distinct_sequence_keys_and_replay_in_order(tmp_path):
    executions: list[str] = []

    async def mutate(payload, _context):
        executions.append(str(payload["value"]))
        return ToolResult(f"completed {len(executions)}")

    config = _config(tmp_path)
    executor = ToolExecutor(_mutating_registry(mutate))
    calls = [
        {"id": "call-1", "name": "mutate", "arguments": {"value": "same"}},
        {"id": "call-2", "name": "mutate", "arguments": {"value": "same"}},
    ]

    first = asyncio.run(
        executor.execute_all(
            calls,
            ToolContext(cwd=str(tmp_path), config=config, execution_scope="duplicate-scope"),
        )
    )
    second = asyncio.run(
        executor.execute_all(
            calls,
            ToolContext(cwd=str(tmp_path), config=config, execution_scope="duplicate-scope"),
        )
    )

    assert executions == ["same", "same"]
    assert first[0].execution_key != first[1].execution_key
    assert [result.replayed for result in second] == [True, True]
    assert [result.content for result in second] == ["completed 1", "completed 2"]


def test_read_only_variation_does_not_shift_mutation_identity(tmp_path):
    executions = 0

    async def inspect(_payload, _context):
        return ToolResult("read completed")

    async def mutate(_payload, _context):
        nonlocal executions
        executions += 1
        return ToolResult("mutation completed")

    config = _config(tmp_path)
    registry = _mutating_registry(mutate)
    registry.register(
        Tool(
            name="inspect",
            description="Read state",
            parameters=object_schema({}),
            handler=inspect,
        )
    )
    executor = ToolExecutor(registry)
    first = [
        {"id": "read-1", "name": "inspect", "arguments": {}},
        {"id": "write-1", "name": "mutate", "arguments": {"value": "one"}},
    ]
    resumed = [
        {"id": "write-2", "name": "mutate", "arguments": {"value": "one"}},
    ]

    asyncio.run(
        executor.execute_all(
            first,
            ToolContext(cwd=str(tmp_path), config=config, execution_scope="read-variation"),
        )
    )
    replayed = asyncio.run(
        executor.execute_all(
            resumed,
            ToolContext(cwd=str(tmp_path), config=config, execution_scope="read-variation"),
        )
    )[0]

    assert executions == 1
    assert replayed.replayed


def test_cancelled_tool_becomes_uncertain_and_requires_explicit_retry(tmp_path):
    started = asyncio.Event()
    executions = 0

    async def mutate(_payload, _context):
        nonlocal executions
        executions += 1
        if executions == 1:
            started.set()
            await asyncio.Event().wait()
        return ToolResult("retry completed")

    config = _config(tmp_path)
    executor = ToolExecutor(_mutating_registry(mutate))
    call = {"id": "call-1", "name": "mutate", "arguments": {"value": "one"}}

    async def cancel_first_run():
        runner = asyncio.create_task(
            executor.execute_all(
                [call],
                ToolContext(cwd=str(tmp_path), config=config, execution_scope="uncertain-scope"),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

    asyncio.run(cancel_first_run())

    blocked = asyncio.run(
        executor.execute_all(
            [call],
            ToolContext(cwd=str(tmp_path), config=config, execution_scope="uncertain-scope"),
        )
    )[0]
    assert executions == 1
    assert blocked.is_error
    assert blocked.recovery_status == "uncertain"

    retried = asyncio.run(
        executor.execute_all(
            [call],
            ToolContext(
                cwd=str(tmp_path),
                config=config,
                execution_scope="uncertain-scope",
                allow_uncertain_retry=True,
            ),
        )
    )[0]
    assert executions == 2
    assert not retried.is_error
    assert retried.content == "retry completed"


def test_tool_exception_remains_uncertain_instead_of_being_cached_as_completed(tmp_path):
    executions = 0

    async def mutate(_payload, _context):
        nonlocal executions
        executions += 1
        if executions == 1:
            raise TimeoutError("downstream outcome unknown")
        return ToolResult("retry completed")

    config = _config(tmp_path)
    executor = ToolExecutor(_mutating_registry(mutate))
    call = {"id": "call-1", "name": "mutate", "arguments": {"value": "one"}}

    first = asyncio.run(
        executor.execute_all(
            [call],
            ToolContext(cwd=str(tmp_path), config=config, execution_scope="timeout-scope"),
        )
    )[0]
    blocked = asyncio.run(
        executor.execute_all(
            [call],
            ToolContext(cwd=str(tmp_path), config=config, execution_scope="timeout-scope"),
        )
    )[0]
    retried = asyncio.run(
        executor.execute_all(
            [call],
            ToolContext(
                cwd=str(tmp_path),
                config=config,
                execution_scope="timeout-scope",
                allow_uncertain_retry=True,
            ),
        )
    )[0]

    assert first.is_error and first.recovery_status == "uncertain"
    assert blocked.is_error and blocked.recovery_status == "uncertain"
    assert executions == 2
    assert retried.content == "retry completed"


def test_write_file_reconciles_uncertain_overwrite_without_reexecuting(tmp_path):
    config = _config(tmp_path)
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    executor = ToolExecutor(registry)
    scope = "write-reconciliation"
    payload = {"path": "result.txt", "content": "already written"}
    execution_key, input_hash = execution_identity(scope, 0, "write_file", payload)
    journal = ToolExecutionJournal(config.tools.execution_journal_path)
    journal.claim(
        execution_key=execution_key,
        scope=scope,
        sequence=0,
        tool_name="write_file",
        input_hash=input_hash,
        allow_uncertain_retry=False,
    )
    journal.mark_uncertain(execution_key)
    target = tmp_path / "result.txt"
    target.write_text("already written", encoding="utf-8")

    result = asyncio.run(
        executor.execute_all(
            [{"id": "write-1", "name": "write_file", "arguments": payload}],
            ToolContext(cwd=str(tmp_path), config=config, execution_scope=scope),
        )
    )[0]

    assert result.replayed
    assert result.recovery_status == "reconciled"
    assert target.read_text(encoding="utf-8") == "already written"
    assert journal.get(execution_key).status == "completed"  # type: ignore[union-attr]


def test_concurrent_claim_allows_only_one_executor(tmp_path):
    path = tmp_path / "tool-executions.sqlite"
    ToolExecutionJournal(path)
    execution_key, input_hash = execution_identity("shared", 0, "mutate", {"value": "x"})
    barrier = Barrier(2)

    def claim_once():
        journal = ToolExecutionJournal(path)
        barrier.wait()
        return journal.claim(
            execution_key=execution_key,
            scope="shared",
            sequence=0,
            tool_name="mutate",
            input_hash=input_hash,
            allow_uncertain_retry=False,
        ).action

    with ThreadPoolExecutor(max_workers=2) as pool:
        actions = sorted(pool.map(lambda _index: claim_once(), range(2)))

    assert actions == ["execute", "uncertain"]


def test_terminal_plan_cleanup_deletes_only_matching_scope_prefix(tmp_path):
    journal = ToolExecutionJournal(tmp_path / "tool-executions.sqlite")
    for scope, sequence in (("session:plan-a:task-1", 0), ("session:plan-b:task-1", 0)):
        execution_key, input_hash = execution_identity(scope, sequence, "mutate", {})
        journal.claim(
            execution_key=execution_key,
            scope=scope,
            sequence=sequence,
            tool_name="mutate",
            input_hash=input_hash,
            allow_uncertain_retry=False,
        )

    deleted = journal.delete_scope_prefix("session:plan-a:")

    assert deleted == 1
    remaining_key, _input_hash = execution_identity("session:plan-b:task-1", 0, "mutate", {})
    assert journal.get(remaining_key) is not None
