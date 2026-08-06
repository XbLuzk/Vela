from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vela.agent import LangGraphPlanAgent
from vela.config import load_config
from vela.plan import ExecutionPlan, Planner, Task, TaskType
from vela.task_control import PlanReviewDecision, TaskCancelledError
from vela.tools import ToolRegistry, get_builtin_tools
from vela.tools.base import Tool, ToolResult, object_schema


def test_execution_plan_exposes_dag_batches():
    plan = ExecutionPlan(id="plan_1", goal="demo")
    task_1 = Task("task_1", "read a", TaskType.FILE_READ)
    task_2 = Task("task_2", "read b", TaskType.FILE_READ)
    task_3 = Task("task_3", "summarize", TaskType.ANALYSIS, ["task_1", "task_2"])

    plan.add_task(task_1)
    plan.add_task(task_2)
    plan.add_task(task_3)

    assert plan.execution_order() == ["task_1", "task_2", "task_3"]
    assert plan.execution_batches() == [[task_1, task_2], [task_3]]
    assert plan.executable_tasks() == [task_1, task_2]
    task_1.mark_completed("done")
    assert plan.executable_tasks() == [task_2]


def test_execution_plan_summary_uses_chinese_labels():
    plan = ExecutionPlan(id="plan_1", goal="检查项目")
    plan.summary = "先检查再汇总"
    plan.add_task(Task("task_1", "检查文件", TaskType.FILE_READ))

    summary = plan.summarize()

    assert "计划 plan_1：先检查再汇总" in summary
    assert "任务数：1" in summary
    assert "当前可执行：1" in summary


def test_planner_parses_tasks_and_dependencies():
    planner = Planner(FakeClient())

    plan = planner.parse_plan(
        "demo",
        """
        ```json
        {
          "summary": "demo plan",
          "tasks": [
            {"id": "a", "description": "A", "type": "COMMAND", "dependencies": []},
            {"id": "b", "description": "B", "type": "VERIFICATION", "dependencies": ["a"]}
          ]
        }
        ```
        """,
    )

    assert plan.summary == "demo plan"
    assert plan.get_task("task_2").dependencies == ["task_1"]
    assert plan.get_task("task_2").type == TaskType.VERIFICATION


def test_plan_execute_runs_independent_tasks_in_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = ParallelPlanClient()
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = "never"
    agent = LangGraphPlanAgent(
        llm_client=client,
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def run():
        text = ""
        events = []
        async for event in agent.run("先做 A 和 B，然后汇总"):
            events.append(event)
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
        return text, events

    result, events = asyncio.run(run())

    assert "正在规划任务：" in result
    assert "开始执行计划" in result
    assert "已完成 [task_1]" in result
    assert "已完成 [task_2]" in result
    assert "计划执行完成" in result
    assert "Planning task" not in result
    assert "Completed [" not in result
    assert client.task_system_prompts
    assert all(
        "所有进度说明、分析和最终结果都必须使用中文" in prompt
        for prompt in client.task_system_prompts
    )
    assert client.peak_concurrency == 2
    assert any(
        event.get("type") == "thinking_delta" and event.get("phase") == "planning"
        for event in events
    )
    assert {
        event.get("task_id")
        for event in events
        if event.get("type") == "thinking_delta" and event.get("phase") == "execution"
    } == {"task_1", "task_2"}
    assert {
        event.get("task_id") for event in events if event.get("type") == "plan_task_started"
    } == {"task_1", "task_2"}


def test_plan_review_can_modify_then_execute(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = ReviewPlanClient()
    decisions = iter(
        [
            PlanReviewDecision.modify("只保留一个验证任务"),
            PlanReviewDecision.execute(),
        ]
    )
    agent = LangGraphPlanAgent(
        llm_client=client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        plan_review_callback=lambda _plan: next(decisions),
    )

    async def run():
        return [event async for event in agent.run("先分析，然后验证结果")]

    events = asyncio.run(run())

    assert client.plan_requests == 2
    review_events = [event for event in events if event.get("type") == "plan_review"]
    status_events = [event for event in events if event.get("type") == "plan_status"]
    assert len(review_events) == 2
    assert all(event["interrupt"]["kind"] == "plan_review" for event in review_events)
    assert all(event["interrupt"]["thread_id"] == agent.thread_id for event in review_events)
    assert status_events[-1] == {"type": "plan_status", "phase": "execution"}
    assert any("正在按要求修改计划" in str(event.get("text")) for event in events)
    assert agent.history[-1].role == "assistant"


def test_plan_review_cancel_stops_before_task_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = ReviewPlanClient()
    agent = LangGraphPlanAgent(
        llm_client=client,
        tool_registry=ToolRegistry(),
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        plan_review_callback=lambda _plan: PlanReviewDecision.cancel(),
    )

    async def run():
        return [event async for event in agent.run("先分析，然后验证结果")]

    with pytest.raises(TaskCancelledError):
        asyncio.run(run())

    assert client.task_requests == 0
    assert "计划已取消" in str(agent.history[-1].content)


def test_cancelled_plan_task_retains_incremental_tool_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    second_started = asyncio.Event()
    registry = _two_tool_registry(second_started)
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = "never"
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
    config.policy.hitl_mode = "never"
    config.features.audit_log = False
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
    config.policy.hitl_mode = "never"
    config.features.audit_log = False
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


class FakeClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {"type": "text_delta", "text": "{}"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class TwoToolClient(FakeClient):
    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        for index, name in enumerate(("first_tool", "second_tool")):
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": index,
                    "id": f"call_{index + 1}",
                    "function": {"name": name, "arguments": "{}"},
                },
            }
        yield {"type": "message_end", "stop_reason": "tool_use"}


def _two_tool_registry(second_started: asyncio.Event) -> ToolRegistry:
    async def first_tool(payload, context):  # noqa: ARG001
        return ToolResult("first completed")

    async def second_tool(payload, context):  # noqa: ARG001
        second_started.set()
        await asyncio.Event().wait()
        return ToolResult("unreachable")

    registry = ToolRegistry()
    for name, handler in (("first_tool", first_tool), ("second_tool", second_tool)):
        registry.register(
            Tool(
                name=name,
                description=name,
                parameters=object_schema({}),
                handler=handler,
                is_read_only=False,
            )
        )
    return registry


async def _consume(events) -> None:
    async for _ in events:
        pass


async def _collect(events) -> list[dict[str, Any]]:
    return [event async for event in events]


class ParallelPlanClient(FakeClient):
    def __init__(self):
        self.current_concurrency = 0
        self.peak_concurrency = 0
        self.ready = asyncio.Event()
        self.task_system_prompts: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            yield {"type": "thinking_delta", "thinking": "先拆分可并行任务"}
            yield {
                "type": "text_delta",
                "text": (
                    '{"summary":"parallel","tasks":['
                    '{"id":"a","description":"任务 A","type":"ANALYSIS","dependencies":[]},'
                    '{"id":"b","description":"任务 B","type":"ANALYSIS","dependencies":[]}'
                    "]}"
                ),
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        if "任务 A" in body or "任务 B" in body:
            self.task_system_prompts.append(system_prompt)
            self.current_concurrency += 1
            self.peak_concurrency = max(self.peak_concurrency, self.current_concurrency)
            if self.current_concurrency == 2:
                self.ready.set()
            await asyncio.wait_for(self.ready.wait(), timeout=2)
            self.current_concurrency -= 1
            text = "A 的结果" if "任务 A" in body else "B 的结果"
            yield {"type": "thinking_delta", "thinking": f"分析{text}"}
            yield {"type": "text_delta", "text": text}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        yield {"type": "text_delta", "text": "fallback"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class ReviewPlanClient(ParallelPlanClient):
    def __init__(self):
        super().__init__()
        self.plan_requests = 0
        self.task_requests = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            self.plan_requests += 1
            yield {
                "type": "text_delta",
                "text": (
                    '{"summary":"review","tasks":['
                    '{"id":"a","description":"验证任务","type":"VERIFICATION",'
                    '"dependencies":[]}]}'
                ),
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        self.task_requests += 1
        yield {"type": "text_delta", "text": "验证完成"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class ResumePlanClient(FakeClient):
    def __init__(
        self,
        *,
        second_started: asyncio.Event | None = None,
        block_second: bool = False,
    ) -> None:
        self.second_started = second_started
        self.block_second = block_second
        self.plan_requests = 0
        self.task_requests: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            self.plan_requests += 1
            yield {
                "type": "text_delta",
                "text": (
                    '{"summary":"resume","tasks":['
                    '{"id":"first","description":"任务一","type":"ANALYSIS",'
                    '"dependencies":[]},'
                    '{"id":"second","description":"任务二","type":"VERIFICATION",'
                    '"dependencies":["first"]}]}'
                ),
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        task_name = "任务一" if "当前任务 [task_1]" in body else "任务二"
        self.task_requests.append(task_name)
        if task_name == "任务二" and self.second_started is not None:
            self.second_started.set()
        if task_name == "任务二" and self.block_second:
            await asyncio.Event().wait()
        yield {"type": "text_delta", "text": f"{task_name}完成"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class ParallelCheckpointClient(FakeClient):
    def __init__(
        self,
        *,
        block_second: bool,
        second_started: asyncio.Event | None = None,
    ) -> None:
        self.block_second = block_second
        self.second_started = second_started
        self.plan_requests = 0
        self.task_requests: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            self.plan_requests += 1
            yield {
                "type": "text_delta",
                "text": (
                    '{"summary":"parallel checkpoint","tasks":['
                    '{"id":"first","description":"任务一","type":"ANALYSIS",'
                    '"dependencies":[]},'
                    '{"id":"second","description":"任务二","type":"ANALYSIS",'
                    '"dependencies":[]}]}'
                ),
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        task_name = "任务一" if "当前任务 [task_1]" in body else "任务二"
        self.task_requests.append(task_name)
        if task_name == "任务二" and self.second_started is not None:
            self.second_started.set()
        if task_name == "任务二" and self.block_second:
            await asyncio.Event().wait()
        yield {"type": "text_delta", "text": f"{task_name}完成"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class SingleTaskClient(FakeClient):
    def __init__(self, description: str) -> None:
        self.description = description
        self.plan_requests = 0
        self.task_requests: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            self.plan_requests += 1
            yield {
                "type": "text_delta",
                "text": (
                    '{"summary":"fresh","tasks":['
                    f'{{"id":"only","description":"{self.description}",'
                    '"type":"ANALYSIS","dependencies":[]}]} '
                ),
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        self.task_requests.append(self.description)
        yield {"type": "text_delta", "text": f"{self.description}完成"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class FailingTaskClient(SingleTaskClient):
    def __init__(self) -> None:
        super().__init__("失败任务")

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            async for event in super().chat(messages, tools, system_prompt=system_prompt):
                yield event
            return
        self.task_requests.append(self.description)
        raise RuntimeError("simulated task failure")


class JournalResumeClient(FakeClient):
    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        if messages[-1].role == "tool":
            yield {"type": "text_delta", "text": "两个工具均已完成"}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        for index, name in enumerate(("first_tool", "second_tool")):
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": index,
                    "id": f"call_{index + 1}",
                    "function": {"name": name, "arguments": "{}"},
                },
            }
        yield {"type": "message_end", "stop_reason": "tool_use"}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)
