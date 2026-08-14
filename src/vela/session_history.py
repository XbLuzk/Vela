"""Helpers for keeping Session history compact and resumable."""

from __future__ import annotations

import json
from typing import Any

from vela.types import Message


def bounded_tool_transcript(
    messages: list[Message],
    *,
    max_calls: int = 24,
    max_content_chars: int = 4_000,
) -> list[Message]:
    """Keep recent tool-call evidence without copying an unbounded worker chat."""

    tool_results = {
        str(message.tool_call_id): message
        for message in messages
        if message.role == "tool" and message.tool_call_id
    }
    call_groups = [
        message for message in messages if message.role == "assistant" and message.tool_calls
    ]
    selected: list[tuple[Message, list[dict[str, Any]]]] = []
    remaining = max_calls
    for assistant in reversed(call_groups):
        if remaining <= 0:
            break
        calls = assistant.tool_calls[-remaining:]
        selected.append((assistant, calls))
        remaining -= len(calls)

    transcript: list[Message] = []
    for assistant, calls in reversed(selected):
        bounded_calls = [_bounded_tool_call(call, max_content_chars) for call in calls]
        transcript.append(
            Message(
                role="assistant",
                content=_truncate_message_content(assistant.content, max_content_chars),
                tool_calls=bounded_calls,
            )
        )
        for call in calls:
            call_id = str(call.get("id") or "")
            result = tool_results.get(call_id)
            if result is not None:
                transcript.append(
                    Message(
                        role="tool",
                        content=_truncate_message_content(result.content, max_content_chars),
                        tool_call_id=call_id,
                    )
                )
    return transcript


def finalize_interrupted_history(
    messages: list[Message],
    *,
    status: str,
    detail: str = "",
) -> list[Message]:
    """Close incomplete tool calls and record a resumable interruption boundary."""

    finalized = list(messages)
    completed_tool_ids = {
        str(message.tool_call_id)
        for message in finalized
        if message.role == "tool" and message.tool_call_id
    }
    pending_tool_ids: list[str] = []
    for message in finalized:
        if message.role != "assistant":
            continue
        for call in message.tool_calls:
            call_id = str(call.get("id") or "")
            if call_id and call_id not in completed_tool_ids:
                pending_tool_ids.append(call_id)
    for call_id in pending_tool_ids:
        finalized.append(
            Message(
                role="tool",
                content=f"Tool execution {status} before producing a result.",
                tool_call_id=call_id,
            )
        )

    label = "Task cancelled by the user." if status == "cancelled" else "Task failed."
    if detail:
        label = f"{label} {detail}".strip()
    if not finalized or finalized[-1].role != "assistant" or finalized[-1].content != label:
        finalized.append(Message(role="assistant", content=label))
    return finalized


def _truncate_message_content(content: str | list[dict[str, Any]], limit: int) -> str:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [session transcript truncated]"


def _bounded_tool_call(call: dict[str, Any], limit: int) -> dict[str, Any]:
    if len(json.dumps(call, ensure_ascii=False)) <= limit:
        return dict(call)
    call_id = str(call.get("id") or "")
    function = call.get("function")
    if isinstance(function, dict):
        return {
            "id": call_id,
            "type": str(call.get("type") or "function"),
            "function": {
                "name": str(function.get("name") or "unknown"),
                "arguments": json.dumps({"_truncated": True, "reason": "session transcript limit"}),
            },
        }
    return {
        "id": call_id,
        "name": str(call.get("name") or "unknown"),
        "input": {"_truncated": True, "reason": "session transcript limit"},
    }
