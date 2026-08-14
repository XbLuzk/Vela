from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal


class TaskState(StrEnum):
    PLANNING = "planning"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanReviewAction(StrEnum):
    EXECUTE = "execute"
    MODIFY = "modify"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class PlanReviewDecision:
    action: PlanReviewAction
    feedback: str = ""

    @classmethod
    def execute(cls) -> PlanReviewDecision:
        return cls(PlanReviewAction.EXECUTE)

    @classmethod
    def modify(cls, feedback: str) -> PlanReviewDecision:
        return cls(PlanReviewAction.MODIFY, feedback.strip())

    @classmethod
    def cancel(cls) -> PlanReviewDecision:
        return cls(PlanReviewAction.CANCEL)


class TaskCancelledError(asyncio.CancelledError):
    """A cooperative cancellation that should end the current interactive run."""


async def resolve_plan_review(
    callback: Callable[[Any], PlanReviewDecision | Awaitable[PlanReviewDecision]] | None,
    plan: Any,
) -> PlanReviewDecision:
    """Resolve and validate the Plan review contract."""

    if callback is None:
        return PlanReviewDecision.execute()
    decision = callback(plan)
    if isinstance(decision, Awaitable):
        decision = await decision
    if not isinstance(decision, PlanReviewDecision):
        raise TypeError("plan review callback must return PlanReviewDecision")
    if decision.action == PlanReviewAction.MODIFY and not decision.feedback:
        raise ValueError("plan modification requires feedback")
    return decision


class InteractiveTaskController:
    """Own one foreground Agent run and its interactive lifecycle."""

    def __init__(
        self,
        *,
        on_change: Callable[[], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.state: TaskState | None = None
        self.label = ""
        self.error: BaseException | None = None
        self._task: asyncio.Task[None] | None = None
        self._review_future: asyncio.Future[PlanReviewDecision] | None = None
        self._approval_future: asyncio.Future[str] | None = None
        self._approval_request: dict[str, Any] | None = None
        self._approval_queue: deque[tuple[asyncio.Future[str], dict[str, Any]]] = deque()
        self._steering_messages: deque[str] = deque()
        self._follow_up_messages: deque[str] = deque()
        self._follow_up_runner: Callable[[str], Awaitable[None]] | None = None
        self._accepts_steering = True
        self._review_feedback_pending = False
        self._watch_started = False
        self._cancel_requested = False
        self._on_change = on_change
        self._on_error = on_error

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def cancelling(self) -> bool:
        return self.active and self.state == TaskState.CANCELLING

    @property
    def awaiting_plan_review(self) -> bool:
        return self._review_future is not None and not self._review_future.done()

    @property
    def review_feedback_pending(self) -> bool:
        return self.awaiting_plan_review and self._review_feedback_pending

    @property
    def awaiting_approval(self) -> bool:
        return self._approval_future is not None and not self._approval_future.done()

    @property
    def approval_request(self) -> dict[str, Any] | None:
        return self._approval_request

    @property
    def queued_messages(self) -> int:
        return len(self._steering_messages) + len(self._follow_up_messages)

    def set_callbacks(
        self,
        *,
        on_change: Callable[[], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._on_change = on_change
        self._on_error = on_error

    def set_follow_up_runner(self, runner: Callable[[str], Awaitable[None]]) -> None:
        """Set the REPL-owned runner used after the current request finishes."""
        self._follow_up_runner = runner

    def start(
        self,
        awaitable: Awaitable[None],
        *,
        initial_state: TaskState,
        label: str,
        accepts_steering: bool = True,
    ) -> None:
        if self.active:
            if hasattr(awaitable, "close"):
                awaitable.close()  # type: ignore[attr-defined]
            raise RuntimeError("A task is already running")
        self.label = label
        self.error = None
        self._watch_started = False
        self._cancel_requested = False
        self._accepts_steering = accepts_steering
        self._set_state(initial_state)
        self._task = asyncio.create_task(self._watch(awaitable))

    def queue_message(
        self,
        message: str,
        *,
        delivery: Literal["steering", "follow_up"] = "steering",
    ) -> Literal["steering", "follow_up"]:
        """Queue input without starting a second concurrent Agent run."""
        text = message.strip()
        if not text:
            raise ValueError("queued message must not be empty")
        if delivery == "steering" and self._accepts_steering:
            self._steering_messages.append(text)
            queued_as: Literal["steering", "follow_up"] = "steering"
        else:
            self._follow_up_messages.append(text)
            queued_as = "follow_up"
        self._notify()
        return queued_as

    def take_steering_message(self) -> str | None:
        """Return one message at the next safe ReAct turn boundary."""
        if not self._steering_messages:
            return None
        message = self._steering_messages.popleft()
        self._notify()
        return message

    def take_pending_messages(self) -> list[str]:
        """Return undelivered input so the REPL can restore it after failure/cancel."""
        messages = [*self._steering_messages, *self._follow_up_messages]
        self._steering_messages.clear()
        self._follow_up_messages.clear()
        return messages

    def set_phase(self, phase: str) -> None:
        if self.cancelling:
            return
        if phase == "planning":
            self._set_state(TaskState.PLANNING)
        elif phase in {"execution", "running"}:
            self._set_state(TaskState.RUNNING)

    def request_cancel(self) -> bool:
        if not self.active:
            return False
        if self.state == TaskState.CANCELLING:
            return False
        self._set_state(TaskState.CANCELLING)
        self._cancel_requested = True
        if self._task is not None and self._watch_started:
            self._task.cancel()
        return True

    async def wait(self) -> TaskState | None:
        task = self._task
        if task is not None:
            await task
        return self.state

    async def request_plan_review(self, _plan: Any) -> PlanReviewDecision:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[PlanReviewDecision] = loop.create_future()
        self._review_future = future
        self._review_feedback_pending = False
        self._set_state(TaskState.PLANNING)
        try:
            return await future
        finally:
            if self._review_future is future:
                self._review_future = None
                self._review_feedback_pending = False
                self._notify()

    def submit_plan_review(self, value: str) -> str:
        future = self._review_future
        if future is None or future.done():
            return "当前没有等待确认的计划。"

        text = value.strip()
        if self._review_feedback_pending:
            if not text:
                return "请输入具体的计划修改要求。"
            self._review_feedback_pending = False
            future.set_result(PlanReviewDecision.modify(text))
            return "正在按修改要求重新规划……"

        command, _, feedback = text.partition(" ")
        normalized = command.lower()
        if normalized in {"execute", "run", "y", "yes", "执行", "确认"}:
            future.set_result(PlanReviewDecision.execute())
            return "计划已确认，开始执行。"
        if normalized in {"cancel", "n", "no", "取消"}:
            future.set_result(PlanReviewDecision.cancel())
            return "计划已取消。"
        if normalized in {"modify", "edit", "m", "修改"}:
            if feedback.strip():
                future.set_result(PlanReviewDecision.modify(feedback))
                return "正在按修改要求重新规划……"
            self._review_feedback_pending = True
            self._notify()
            return "请输入具体的计划修改要求。"
        return "请选择 execute、modify 或 cancel。"

    async def request_approval(self, request: dict[str, Any]) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._approval_queue.append((future, request))
        self._promote_next_approval()
        try:
            return await future
        finally:
            if self._approval_future is future:
                self._advance_approval(future)
            else:
                self._approval_queue = deque(
                    item for item in self._approval_queue if item[0] is not future
                )

    def submit_approval(self, value: str) -> str:
        future = self._approval_future
        if future is None or future.done():
            return "当前没有等待确认的工具调用。"
        normalized = value.strip().lower()
        choices = {
            "y": "approve",
            "yes": "approve",
            "允许": "approve",
            "n": "deny",
            "no": "deny",
            "拒绝": "deny",
            "a": "auto",
            "auto": "auto",
            "自动": "auto",
            "s": "skip",
            "skip": "skip",
            "跳过": "skip",
        }
        decision = choices.get(normalized)
        if decision is None:
            return "请选择 y（允许）、n（拒绝）、a（允许并切换 Auto）或 s（跳过）。"
        future.set_result(decision)
        self._advance_approval(future)
        return {
            "approve": "已允许工具调用。",
            "deny": "已拒绝工具调用。",
            "auto": "已允许，并切换为 Auto 模式。",
            "skip": "已跳过工具调用。",
        }[decision]

    async def _watch(self, awaitable: Awaitable[None]) -> None:
        self._watch_started = True
        try:
            if self._cancel_requested:
                if isinstance(awaitable, asyncio.Future):
                    awaitable.cancel()
                elif hasattr(awaitable, "close"):
                    awaitable.close()  # type: ignore[attr-defined]
                raise TaskCancelledError
            await self._run_sequence(awaitable)
        except (TaskCancelledError, asyncio.CancelledError):
            self._set_state(TaskState.CANCELLED)
        except Exception as exc:  # noqa: BLE001 - task boundary retains ordinary failures
            self.error = exc
            self._set_state(TaskState.FAILED)
            if self._on_error is not None:
                self._on_error(exc)
        else:
            self._set_state(TaskState.COMPLETED)
        finally:
            review = self._review_future
            if review is not None and not review.done():
                review.cancel()
            self._review_future = None
            self._review_feedback_pending = False
            approval = self._approval_future
            if approval is not None and not approval.done():
                approval.cancel()
            for queued, _request in self._approval_queue:
                if not queued.done():
                    queued.cancel()
            self._approval_queue.clear()
            self._approval_future = None
            self._approval_request = None
            self._notify()

    async def _run_sequence(self, first: Awaitable[None]) -> None:
        """Run one foreground request, then queued follow-ups serially."""
        await first
        while not self._cancel_requested:
            if self._follow_up_runner is None:
                return
            message = self._next_follow_up()
            if message is None:
                return
            self.label = message
            self._accepts_steering = True
            self._set_state(TaskState.RUNNING)
            await self._follow_up_runner(message)

    def _next_follow_up(self) -> str | None:
        # A steering message can arrive just after the final safe turn boundary.
        # Preserve it by executing it before explicit follow-ups.
        if self._steering_messages:
            return self._steering_messages.popleft()
        if self._follow_up_messages:
            return self._follow_up_messages.popleft()
        return None

    def _promote_next_approval(self) -> None:
        if self._approval_future is not None and not self._approval_future.done():
            return
        self._approval_future = None
        self._approval_request = None
        while self._approval_queue:
            future, request = self._approval_queue.popleft()
            if future.done():
                continue
            self._approval_future = future
            self._approval_request = request
            break
        self._notify()

    def _advance_approval(self, completed: asyncio.Future[str]) -> None:
        if self._approval_future is not completed:
            return
        self._approval_future = None
        self._approval_request = None
        self._promote_next_approval()

    def _set_state(self, state: TaskState) -> None:
        self.state = state
        self._notify()

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()
