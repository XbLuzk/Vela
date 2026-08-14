from __future__ import annotations

import asyncio

from vela.task_control import (
    InteractiveTaskController,
    PlanReviewAction,
    TaskState,
)


def test_controller_tracks_successful_task_lifecycle():
    states: list[TaskState | None] = []
    controller = InteractiveTaskController(on_change=lambda: states.append(controller.state))

    async def run():
        controller.start(_complete(), initial_state=TaskState.RUNNING, label="demo")
        await controller.wait()

    asyncio.run(run())

    assert states[0] == TaskState.RUNNING
    assert controller.state == TaskState.COMPLETED
    assert controller.label == "demo"


def test_controller_cancels_running_task_and_reaches_cancelled():
    started = asyncio.Event()
    cleaned_up = False
    controller = InteractiveTaskController()

    async def work():
        nonlocal cleaned_up
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up = True

    async def run():
        controller.start(work(), initial_state=TaskState.PLANNING, label="long task")
        await started.wait()
        assert controller.request_cancel()
        assert controller.state == TaskState.CANCELLING
        assert not controller.request_cancel()
        await controller.wait()

    asyncio.run(run())

    assert cleaned_up
    assert controller.state == TaskState.CANCELLED


def test_controller_collects_plan_modify_feedback_then_executes():
    controller = InteractiveTaskController()
    decisions = []

    async def work():
        first = await controller.request_plan_review(object())
        decisions.append(first)
        second = await controller.request_plan_review(object())
        decisions.append(second)

    async def run():
        controller.start(work(), initial_state=TaskState.PLANNING, label="plan")
        await _wait_until(lambda: controller.awaiting_plan_review)
        assert controller.submit_plan_review("modify") == "请输入具体的计划修改要求。"
        assert controller.review_feedback_pending
        assert "重新规划" in controller.submit_plan_review("先加测试，再修改代码")
        await _wait_until(lambda: controller.awaiting_plan_review)
        assert "开始执行" in controller.submit_plan_review("execute")
        await controller.wait()

    asyncio.run(run())

    assert decisions[0].action == PlanReviewAction.MODIFY
    assert decisions[0].feedback == "先加测试，再修改代码"
    assert decisions[1].action == PlanReviewAction.EXECUTE
    assert controller.state == TaskState.COMPLETED


def test_controller_records_failed_state_and_notifies_error_callback():
    errors: list[BaseException] = []
    controller = InteractiveTaskController(on_error=errors.append)

    async def fail():
        raise RuntimeError("boom")

    async def run():
        controller.start(fail(), initial_state=TaskState.RUNNING, label="failure")
        await controller.wait()

    asyncio.run(run())

    assert controller.state == TaskState.FAILED
    assert isinstance(controller.error, RuntimeError)
    assert errors == [controller.error]


def test_controller_plan_cancel_prevents_execution():
    controller = InteractiveTaskController()
    executed = False

    async def work():
        nonlocal executed
        decision = await controller.request_plan_review(object())
        if decision.action == PlanReviewAction.CANCEL:
            from vela.task_control import TaskCancelledError

            raise TaskCancelledError
        executed = True

    async def run():
        controller.start(work(), initial_state=TaskState.PLANNING, label="plan")
        await _wait_until(lambda: controller.awaiting_plan_review)
        assert "已取消" in controller.submit_plan_review("cancel")
        await controller.wait()

    asyncio.run(run())

    assert not executed
    assert controller.state == TaskState.CANCELLED


def test_controller_cancels_task_while_tool_approval_is_pending():
    controller = InteractiveTaskController()

    async def work():
        await controller.request_approval({"tool_name": "bash"})

    async def run():
        controller.start(work(), initial_state=TaskState.RUNNING, label="approval")
        await _wait_until(lambda: controller.awaiting_approval)
        assert controller.request_cancel()
        await controller.wait()

    asyncio.run(run())

    assert controller.state == TaskState.CANCELLED
    assert not controller.awaiting_approval


def test_controller_collects_async_tool_approval():
    controller = InteractiveTaskController()
    decisions = []

    async def work():
        decisions.append(await controller.request_approval({"tool_name": "write_file"}))

    async def run():
        controller.start(work(), initial_state=TaskState.RUNNING, label="approval")
        await _wait_until(lambda: controller.awaiting_approval)
        assert "切换为 Auto" in controller.submit_approval("a")
        await controller.wait()

    asyncio.run(run())

    assert decisions == ["auto"]


def test_controller_queues_parallel_tool_approvals_fifo():
    controller = InteractiveTaskController()
    decisions: list[tuple[str, str]] = []

    async def work():
        first, second = await asyncio.gather(
            controller.request_approval({"tool_name": "first"}),
            controller.request_approval({"tool_name": "second"}),
        )
        decisions.extend([("first", first), ("second", second)])

    async def run():
        controller.start(work(), initial_state=TaskState.RUNNING, label="parallel approvals")
        await _wait_until(lambda: controller.approval_request == {"tool_name": "first"})
        assert "已允许" in controller.submit_approval("y")
        await _wait_until(lambda: controller.approval_request == {"tool_name": "second"})
        assert "已拒绝" in controller.submit_approval("n")
        await controller.wait()

    asyncio.run(run())

    assert decisions == [("first", "approve"), ("second", "deny")]
    assert controller.state == TaskState.COMPLETED
    assert not controller.awaiting_approval


def test_controller_cancels_all_queued_tool_approvals():
    controller = InteractiveTaskController()

    async def work():
        await asyncio.gather(
            controller.request_approval({"tool_name": "first"}),
            controller.request_approval({"tool_name": "second"}),
        )

    async def run():
        controller.start(work(), initial_state=TaskState.RUNNING, label="parallel approvals")
        await _wait_until(lambda: controller.awaiting_approval)
        assert controller.request_cancel()
        await controller.wait()

    asyncio.run(run())

    assert controller.state == TaskState.CANCELLED
    assert not controller.awaiting_approval
    assert controller.approval_request is None


def test_controller_runs_late_steering_and_followups_serially():
    controller = InteractiveTaskController()
    started = asyncio.Event()
    release = asyncio.Event()
    executed: list[str] = []

    async def first():
        started.set()
        await release.wait()

    async def follow_up(message: str):
        executed.append(message)

    async def run():
        controller.set_follow_up_runner(follow_up)
        controller.start(first(), initial_state=TaskState.RUNNING, label="first")
        await started.wait()
        assert controller.queue_message("adjust this") == "steering"
        assert controller.queue_message("then summarize", delivery="follow_up") == "follow_up"
        release.set()
        await controller.wait()

    asyncio.run(run())

    assert executed == ["adjust this", "then summarize"]
    assert controller.state == TaskState.COMPLETED
    assert controller.queued_messages == 0


def test_plan_task_converts_steering_input_to_followup():
    controller = InteractiveTaskController()

    async def run():
        controller.start(
            asyncio.Event().wait(),
            initial_state=TaskState.PLANNING,
            label="plan",
            accepts_steering=False,
        )
        assert controller.queue_message("change direction") == "follow_up"
        assert controller.queued_messages == 1
        assert controller.request_cancel()
        await controller.wait()

    asyncio.run(run())

    assert controller.take_pending_messages() == ["change direction"]


def test_cancellation_keeps_undelivered_messages_for_input_restore():
    controller = InteractiveTaskController()
    started = asyncio.Event()

    async def work():
        started.set()
        await asyncio.Event().wait()

    async def run():
        controller.start(work(), initial_state=TaskState.RUNNING, label="work")
        await started.wait()
        controller.queue_message("steer")
        controller.queue_message("follow", delivery="follow_up")
        controller.request_cancel()
        await controller.wait()

    asyncio.run(run())

    assert controller.take_pending_messages() == ["steer", "follow"]


async def _complete() -> None:
    await asyncio.sleep(0)


async def _wait_until(predicate, attempts: int = 20) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")
