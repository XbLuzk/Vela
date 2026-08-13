from __future__ import annotations

import asyncio

import pytest

from tests.agent.plan_support import FakeClient, ParallelPlanClient, ReviewPlanClient
from vela.agent import LangGraphPlanAgent
from vela.config import load_config
from vela.plan import ExecutionPlan, Planner, Task, TaskType
from vela.task_control import PlanReviewDecision, TaskCancelledError
from vela.tools import ToolRegistry, get_builtin_tools


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


def test_planner_uses_model_for_simple_goal():
    client = ReviewPlanClient()
    planner = Planner(client)

    async def run():
        return [event async for event in planner.stream_plan("查看文件")]

    events = asyncio.run(run())

    assert client.plan_requests == 1
    assert events[-1]["type"] == "plan_created"
    assert events[-1]["plan"].summary == "review"


def test_plan_agent_rejects_non_positive_task_turn_limit(tmp_path):
    config = load_config(project_root=tmp_path)

    with pytest.raises(ValueError, match="max_task_turns must be at least 1"):
        LangGraphPlanAgent(
            llm_client=FakeClient(),
            tool_registry=ToolRegistry(),
            config=config,
            cwd=str(tmp_path),
            max_task_turns=0,
        )


def test_closing_plan_run_closes_langgraph_stream(tmp_path, monkeypatch):
    class BlockingGraph:
        def __init__(self):
            self.closed = False

        async def astream(self, graph_input, graph_config, *, stream_mode):  # noqa: ARG002
            try:
                yield "custom", {"type": "text_delta", "text": "partial"}
                await asyncio.Event().wait()
            finally:
                self.closed = True

    config = load_config(project_root=tmp_path)
    graph = BlockingGraph()
    agent = LangGraphPlanAgent(
        llm_client=FakeClient(),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
    )
    monkeypatch.setattr(agent, "_build_graph", lambda _checkpointer: graph)

    async def close_after_first_event():
        stream = agent.run("plan something")
        assert (await anext(stream))["type"] == "text_delta"
        await stream.aclose()
        assert graph.closed

    asyncio.run(close_after_first_event())


def test_plan_cleanup_failure_does_not_replace_task_cancellation(tmp_path, monkeypatch):
    class FailingCleanupGraph:
        def __init__(self):
            self.started = asyncio.Event()

        async def astream(self, graph_input, graph_config, *, stream_mode):  # noqa: ARG002
            try:
                self.started.set()
                yield "custom", {"type": "text_delta", "text": "partial"}
                await asyncio.Event().wait()
            finally:
                raise RuntimeError("plan cleanup failed")

    config = load_config(project_root=tmp_path)
    graph = FailingCleanupGraph()
    agent = LangGraphPlanAgent(
        llm_client=FakeClient(),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
    )
    monkeypatch.setattr(agent, "_build_graph", lambda _checkpointer: graph)

    async def cancel_during_graph_stream():
        async def consume():
            return [event async for event in agent.run("plan something")]

        task = asyncio.create_task(consume())
        await graph.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as error:
            await task
        assert isinstance(error.value.__cause__, RuntimeError)

    asyncio.run(cancel_during_graph_stream())


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
