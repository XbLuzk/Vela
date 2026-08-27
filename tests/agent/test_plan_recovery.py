from __future__ import annotations

import asyncio

import pytest

from tests.agent.plan_support import (
    FailingTaskClient,
    JournalResumeClient,
    ParallelCheckpointClient,
    ResumePlanClient,
    ReviewPlanClient,
    SingleTaskClient,
    TwoToolClient,
    _collect,
    _consume,
    _two_tool_registry,
)
from vela.agent import LangGraphPlanAgent, plan_graph
from vela.config import load_config
from vela.task_control import PlanReviewDecision
from vela.tools import ToolRegistry
from vela.tools.base import Tool, ToolResult, object_schema


def test_cancelled_plan_task_retains_incremental_tool_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    second_started = asyncio.Event()
    registry = _two_tool_registry(second_started)
    config = load_config(project_root=tmp_path)
    config.policy.approval_mode = "auto"
    agent = LangGraphPlanAgent(
        llm_client=TwoToolClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def run():
        runner = asyncio.create_task(_consume(agent.run("执行工具")))
        await second_started.wait()
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

    asyncio.run(run())

    tool_messages = [message for message in agent.history if message.role == "tool"]
    assert [(message.tool_call_id, message.content) for message in tool_messages] == [
        ("call_1", "first completed")
    ]


def test_cancelled_graph_resumes_from_last_completed_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    second_started = asyncio.Event()
    checkpoint_path = tmp_path / "graph" / "checkpoints.sqlite"
    thread_id = "session-resume"
    first_client = ResumePlanClient(second_started=second_started, block_second=True)
    first_agent = LangGraphPlanAgent(
        llm_client=first_client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )

    async def cancel_during_second_task() -> None:
        runner = asyncio.create_task(_consume(first_agent.run("先执行任务一，然后执行任务二")))
        await asyncio.wait_for(second_started.wait(), timeout=2)
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

    asyncio.run(cancel_during_second_task())

    resumed_client = ResumePlanClient(block_second=False)
    resumed_agent = LangGraphPlanAgent(
        llm_client=resumed_client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
        plan_review_callback=lambda _plan: PlanReviewDecision.execute(),
        resume=True,
    )

    async def resume_graph():
        return [event async for event in resumed_agent.run("")]

    events = asyncio.run(resume_graph())

    assert first_client.task_requests == ["任务一", "任务二"]
    assert resumed_client.plan_requests == 0
    assert resumed_client.task_requests == ["任务二"]
    assert any(event.get("type") == "done" for event in events)
    assert any("计划执行完成" in str(event.get("text") or "") for event in events)
    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_mode & 0o777 == 0o600


def test_parallel_resume_keeps_successful_pending_write(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    checkpoint_path = tmp_path / "graph" / "parallel.sqlite"
    thread_id = "parallel-resume"
    second_started = asyncio.Event()
    first_client = ParallelCheckpointClient(
        block_second=True,
        second_started=second_started,
    )
    first_agent = LangGraphPlanAgent(
        llm_client=first_client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )

    async def cancel_after_first_finishes() -> None:
        first_done = asyncio.Event()

        async def consume() -> None:
            async for event in first_agent.run("同时执行任务一和任务二"):
                if event.get("type") == "plan_task_done" and event.get("task_id") == "task_1":
                    first_done.set()

        runner = asyncio.create_task(consume())
        await asyncio.wait_for(first_done.wait(), timeout=2)
        await asyncio.wait_for(second_started.wait(), timeout=2)
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

    asyncio.run(cancel_after_first_finishes())

    resumed_client = ParallelCheckpointClient(block_second=False)
    resumed_agent = LangGraphPlanAgent(
        llm_client=resumed_client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
        plan_review_callback=lambda _plan: PlanReviewDecision.execute(),
        resume=True,
    )

    events = asyncio.run(_collect(resumed_agent.run("")))

    assert first_client.task_requests == ["任务一", "任务二"]
    assert resumed_client.plan_requests == 0
    assert resumed_client.task_requests == ["任务二"]
    assert any(event.get("type") == "plan_resume_warning" for event in events)


def test_plan_review_interrupt_resumes_from_persisted_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    checkpoint_path = tmp_path / "graph" / "review.sqlite"
    thread_id = "review-resume"
    review_started = asyncio.Event()

    async def blocked_review(_plan):
        review_started.set()
        await asyncio.Event().wait()

    first_client = ReviewPlanClient()
    first_agent = LangGraphPlanAgent(
        llm_client=first_client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        plan_review_callback=blocked_review,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )

    async def cancel_during_review():
        runner = asyncio.create_task(_consume(first_agent.run("先分析代码，然后验证结果")))
        await asyncio.wait_for(review_started.wait(), timeout=2)
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

    asyncio.run(cancel_during_review())

    resumed_client = ReviewPlanClient()
    resumed_agent = LangGraphPlanAgent(
        llm_client=resumed_client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        plan_review_callback=lambda _plan: PlanReviewDecision.execute(),
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
        resume=True,
    )
    events = asyncio.run(_collect(resumed_agent.run("")))

    assert first_client.plan_requests == 1
    assert first_client.task_requests == 0
    assert resumed_client.plan_requests == 0
    assert resumed_client.task_requests == 1
    assert any(event.get("type") == "done" for event in events)


def test_plan_review_resume_without_callback_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    checkpoint_path = tmp_path / "graph" / "review-no-callback.sqlite"
    thread_id = "review-no-callback"
    review_started = asyncio.Event()

    async def blocked_review(_plan):
        review_started.set()
        await asyncio.Event().wait()

    first_client = ReviewPlanClient()
    first_agent = LangGraphPlanAgent(
        llm_client=first_client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        plan_review_callback=blocked_review,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )

    async def cancel_during_review():
        runner = asyncio.create_task(_consume(first_agent.run("先分析代码，然后验证结果")))
        await asyncio.wait_for(review_started.wait(), timeout=2)
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

    asyncio.run(cancel_during_review())

    resumed_client = ReviewPlanClient()
    resumed_agent = LangGraphPlanAgent(
        llm_client=resumed_client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
        resume=True,
    )
    events = asyncio.run(_collect(resumed_agent.run("")))

    error = next(event["error"] for event in events if event.get("type") == "error")
    assert "显式确认回调" in str(error)
    assert resumed_client.task_requests == 0


def test_fresh_plan_replaces_completed_state_for_same_thread(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    checkpoint_path = tmp_path / "graph" / "fresh.sqlite"
    thread_id = "same-session-new-plan"
    first_client = ReviewPlanClient()
    first_agent = LangGraphPlanAgent(
        llm_client=first_client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )
    asyncio.run(_consume(first_agent.run("先执行第一个任务，然后验证结果")))

    second_client = SingleTaskClient("全新的任务")
    second_agent = LangGraphPlanAgent(
        llm_client=second_client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )
    events = asyncio.run(_collect(second_agent.run("先执行全新的任务，然后总结结果")))

    assert second_client.plan_requests == 1
    assert second_client.task_requests == ["全新的任务"]
    final_text = "".join(
        str(event.get("text") or "") for event in events if event.get("type") == "text_delta"
    )
    assert "全新的任务" in final_text
    assert "验证任务" not in final_text


def test_failed_task_sets_graph_failed_terminal_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = FailingTaskClient()
    agent = LangGraphPlanAgent(
        llm_client=client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
    )

    events = asyncio.run(_collect(agent.run("执行会失败的任务")))

    assert any("任务执行失败" in str(event.get("text") or "") for event in events)
    done = next(event for event in events if event.get("type") == "done")
    assert done["langgraph"]["status"] == "failed"


def test_plan_resume_replays_completed_tool_and_retries_only_uncertain_call(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    checkpoint_path = tmp_path / "graph" / "tool-granularity.sqlite"
    thread_id = "tool-granularity"
    second_started = asyncio.Event()
    executions = {"first": 0, "second": 0}

    async def first_tool(_payload, _context):
        executions["first"] += 1
        return ToolResult("first completed")

    async def second_tool(_payload, _context):
        executions["second"] += 1
        if executions["second"] == 1:
            second_started.set()
            await asyncio.Event().wait()
        return ToolResult("second completed")

    registry = ToolRegistry()
    for name, handler in (("first_tool", first_tool), ("second_tool", second_tool)):
        registry.register(
            Tool(
                name=name,
                description=name,
                parameters=object_schema({}),
                handler=handler,
                is_read_only=False,
                is_concurrency_safe=False,
            )
        )

    config = load_config(project_root=tmp_path)
    config.policy.approval_mode = "auto"
    config.tools.execution_journal_path = str(tmp_path / "tool-executions.sqlite")
    first_agent = LangGraphPlanAgent(
        llm_client=JournalResumeClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )

    async def cancel_during_second_tool():
        runner = asyncio.create_task(_consume(first_agent.run("执行可恢复工具")))
        await asyncio.wait_for(second_started.wait(), timeout=2)
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

    asyncio.run(cancel_during_second_tool())

    resumed_agent = LangGraphPlanAgent(
        llm_client=JournalResumeClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
        plan_review_callback=lambda _plan: PlanReviewDecision.execute(),
        resume=True,
    )
    events = asyncio.run(_collect(resumed_agent.run("")))

    assert executions == {"first": 1, "second": 2}
    replayed = [event for event in events if event.get("type") == "tool_result"]
    assert any(event.get("name") == "first_tool" and event.get("replayed") for event in replayed)
    assert any(event.get("type") == "done" for event in events)


def test_one_shot_plan_does_not_create_orphan_tool_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    async def complete(_payload, _context):
        return ToolResult("completed")

    registry = ToolRegistry()
    for name in ("first_tool", "second_tool"):
        registry.register(
            Tool(
                name=name,
                description=name,
                parameters=object_schema({}),
                handler=complete,
                is_read_only=False,
                is_concurrency_safe=False,
            )
        )
    config = load_config(project_root=tmp_path)
    config.policy.approval_mode = "auto"
    journal_path = tmp_path / "tool-executions.sqlite"
    config.tools.execution_journal_path = str(journal_path)
    agent = LangGraphPlanAgent(
        llm_client=JournalResumeClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    events = asyncio.run(_collect(agent.run("执行一次性工具")))

    assert any(event.get("type") == "done" for event in events)
    assert not journal_path.exists()


def test_journal_cleanup_failure_is_reported_in_done_event(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    journal_path = tmp_path / "tool-executions.sqlite"
    journal_path.write_bytes(b"")
    config.tools.execution_journal_path = str(journal_path)
    agent = LangGraphPlanAgent(
        llm_client=SingleTaskClient("任务"),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
        thread_id="cleanup-session",
        checkpoint_path=tmp_path / "graph" / "cleanup.sqlite",
    )

    def fail(self, prefix):  # noqa: ANN001, ANN202, ARG001
        raise OSError("journal is locked")

    monkeypatch.setattr(plan_graph.ToolExecutionJournal, "delete_scope_prefix", fail)

    event = agent._finish_graph(  # noqa: SLF001
        {"status": "completed", "plan": {"id": "plan-1"}, "final_text": "done"}
    )

    assert "Tool journal cleanup failed for plan plan-1" in str(event["warning"])
    assert "journal is locked" in str(event["warning"])
