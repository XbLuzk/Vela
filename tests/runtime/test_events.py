from __future__ import annotations

from typing import get_args, get_type_hints

from vela.events import AgentEvent, AgentEventType, LlmEvent, LlmEventType


def test_stream_event_contracts_require_a_closed_type_discriminator() -> None:
    assert AgentEvent.__required_keys__ == {"type"}
    assert LlmEvent.__required_keys__ == {"type"}
    assert set(get_args(AgentEventType)) == {
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
    }
    assert set(get_args(LlmEventType)) == {
        "error",
        "message_end",
        "message_start",
        "text_delta",
        "thinking_delta",
        "tool_call_delta",
        "usage",
    }
    assert get_type_hints(AgentEvent)["plan"] is not None
