"""Typed streaming event contracts shared across Vela runtime layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

from vela.types import Message, UsagePayload

if TYPE_CHECKING:
    from vela.plan.models import ExecutionPlan

    PlanEventValue = ExecutionPlan
else:
    PlanEventValue = Any

AgentEventType = Literal[
    "context_compressed",
    "done",
    "error",
    "model_response_complete",
    "plan_created",
    "plan_resume_warning",
    "plan_review",
    "plan_status",
    "plan_task_done",
    "plan_task_started",
    "text_delta",
    "thinking_delta",
    "tool_call",
    "tool_result",
    "turn_complete",
    "turn_started",
    "usage",
]

AgentPhase = Literal["planning", "execution"]


class _AgentEventBase(TypedDict):
    type: AgentEventType


class AgentEvent(_AgentEventBase, total=False):
    """One event emitted by an Agent run and consumed by CLI renderers.

    ``type`` is always present and acts as the discriminator. The remaining
    fields belong only to the matching event type; keeping them in one compact
    contract makes the streaming protocol easy to discover without adding one
    class per small terminal event.
    """

    text: str
    thinking: str
    phase: AgentPhase
    error: BaseException
    usage: UsagePayload
    total_tokens: int
    total_turns: int
    turn: int
    tool_call_id: str
    stop_reason: str
    messages: list[Message]
    langgraph: dict[str, Any]
    name: str
    input: dict[str, Any]
    result: str
    is_error: bool
    replayed: bool
    execution_key: str | None
    recovery_status: str | None
    before_tokens: int
    after_tokens: int
    summarized_messages: int
    truncated_tool_results: int
    omitted_tool_characters: int
    recovered_from_overflow: bool
    interrupt: dict[str, Any]
    pending_tasks: int
    task_id: str
    task_description: str
    task_status: str
    turns: int
    tokens: int
    plan: PlanEventValue
    warning: str


LlmEventType = Literal[
    "error",
    "message_end",
    "message_start",
    "text_delta",
    "thinking_delta",
    "tool_call_delta",
    "usage",
]


class _LlmEventBase(TypedDict):
    type: LlmEventType


class LlmEvent(_LlmEventBase, total=False):
    """Provider-neutral events emitted by ``LlmClient.chat``."""

    model: str
    text: str
    thinking: str
    tool_call: dict[str, Any]
    stop_reason: str
    usage: UsagePayload
    error: BaseException
