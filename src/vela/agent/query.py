"""Shared ReAct loop for Agent, Plan workers, and Team workers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from vela.config import VelaConfig
from vela.context import ContextBudget, ContextWindowManager
from vela.image import parse_image_references
from vela.llm.base import LlmClient
from vela.prompt import PromptAssembler
from vela.skill import SkillRegistry
from vela.tools.base import ToolContext
from vela.tools.executor import ToolExecutor
from vela.tools.registry import ToolRegistry
from vela.types import Message, Usage


@dataclass(slots=True)
class _ModelTurn:
    """State collected while one streamed model response is arriving."""

    text: str = ""
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    tool_states: dict[int, dict[str, Any]] = field(default_factory=dict)
    failed: bool = False

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return _finalize_tool_calls(self.tool_states)


async def run_react_loop(
    *,
    llm_client: LlmClient,
    tool_registry: ToolRegistry,
    system_prompt: str,
    user_message: str,
    history: list[Message] | None,
    cwd: str,
    config: VelaConfig,
    approval_callback=None,
    skill_context_buffer=None,
    history_sink: list[Message] | None = None,
    tool_execution_scope: str | None = None,
    allow_uncertain_tool_retry: bool = False,
    max_turns: int = 20,
) -> AsyncIterator[dict[str, Any]]:
    """Run model -> tool -> model turns and stream progress events."""

    original_user_message = user_message
    user_message = _prepend_skill_candidates(user_message, cwd, config)
    user_message = _prepend_skill_context(user_message, skill_context_buffer)
    messages = [
        *(history or []),
        Message(role="user", content=parse_image_references(user_message, cwd)),
    ]
    _sync_history(history_sink, messages)
    tool_definitions = tool_registry.definitions()
    executor = ToolExecutor(tool_registry)
    tool_context = ToolContext(
        cwd=cwd,
        config=config,
        approval_callback=approval_callback,
        skill_context_buffer=skill_context_buffer,
        execution_scope=tool_execution_scope,
        allow_uncertain_retry=allow_uncertain_tool_retry,
    )
    dynamic_prompt = PromptAssembler(
        config=config,
        cwd=cwd,
        tool_names=tool_registry.list_names(),
        model=llm_client.model_name,
        provider=llm_client.provider_name,
    ).build_dynamic(original_user_message)
    effective_system_prompt = f"{system_prompt}\n\n{dynamic_prompt}".strip()
    window_manager = ContextWindowManager(
        ContextBudget(
            context_window=llm_client.max_context_window,
            max_output_tokens=config.llm.max_tokens,
            compression_threshold=config.memory.compression_threshold,
            compression_target=config.memory.compression_target,
            reserve_tokens=config.memory.compression_reserve_tokens,
        ),
        max_history_messages=config.memory.max_conversation_history,
        min_recent_messages=config.memory.min_recent_messages,
        summary_max_chars=config.memory.summary_max_chars,
    )

    total_usage = Usage()
    total_turns = 0

    for turn in range(1, max_turns + 1):
        total_turns = turn
        messages, compression_event = _prepare_turn_messages(
            messages,
            window_manager=window_manager,
            system_prompt=effective_system_prompt,
            tool_definitions=tool_definitions,
            history_sink=history_sink,
            enabled=config.features.context_compression,
        )
        if compression_event is not None:
            yield compression_event

        model_turn = _ModelTurn()
        async for event in _stream_model_turn(
            llm_client,
            messages,
            tool_definitions,
            effective_system_prompt,
            model_turn,
        ):
            yield event
        if model_turn.failed:
            return

        total_usage = total_usage + model_turn.usage
        tool_calls = model_turn.tool_calls
        assistant_message = Message(
            role="assistant",
            content=model_turn.text,
            tool_calls=tool_calls,
        )
        messages.append(assistant_message)
        _sync_history(history_sink, messages)
        yield {
            "type": "turn_complete",
            "turn": turn,
            "stop_reason": model_turn.stop_reason,
        }

        if model_turn.stop_reason != "tool_use" and not tool_calls:
            break

        async for event in _execute_tool_round(
            tool_calls,
            executor=executor,
            tool_context=tool_context,
            messages=messages,
            history_sink=history_sink,
            skill_context_buffer=skill_context_buffer,
        ):
            yield event

    done_event: dict[str, Any] = {
        "type": "done",
        "total_turns": total_turns,
        "total_tokens": total_usage.total_tokens,
        "usage": total_usage.to_dict(),
        "messages": messages,
    }
    yield done_event


def _prepare_turn_messages(
    messages: list[Message],
    *,
    window_manager: ContextWindowManager,
    system_prompt: str,
    tool_definitions: list[dict[str, Any]],
    history_sink: list[Message] | None,
    enabled: bool,
) -> tuple[list[Message], dict[str, Any] | None]:
    if not enabled:
        return messages, None
    compression = window_manager.prepare(
        messages,
        system_prompt=system_prompt,
        tool_definitions=tool_definitions,
    )
    messages = compression.messages
    _sync_history(history_sink, messages)
    if not compression.compressed:
        return messages, None
    return messages, {
        "type": "context_compressed",
        "before_tokens": compression.estimated_tokens_before,
        "after_tokens": compression.estimated_tokens_after,
        "summarized_messages": compression.summarized_messages,
    }


async def _stream_model_turn(
    llm_client: LlmClient,
    messages: list[Message],
    tool_definitions: list[dict[str, Any]],
    system_prompt: str,
    turn: _ModelTurn,
) -> AsyncIterator[dict[str, Any]]:
    async for event in llm_client.chat(
        messages,
        tool_definitions,
        system_prompt=system_prompt,
    ):
        event_type = event.get("type")
        if event_type == "text_delta":
            delta = str(event.get("text") or "")
            turn.text += delta
            yield {"type": "text_delta", "text": delta}
        elif event_type == "thinking_delta":
            delta = str(event.get("thinking") or "")
            yield {"type": "thinking_delta", "thinking": delta}
        elif event_type == "tool_call_delta":
            _merge_tool_delta(turn.tool_states, event["tool_call"])
        elif event_type == "message_end":
            turn.stop_reason = str(event.get("stop_reason") or "end_turn")
        elif event_type == "usage":
            usage = Usage.from_mapping(event.get("usage") or {})
            turn.usage = turn.usage + usage
            yield {"type": "usage", "usage": usage.to_dict()}
        elif event_type == "error":
            turn.failed = True
            yield {"type": "error", "error": event["error"]}
            return


async def _execute_tool_round(
    tool_calls: list[dict[str, Any]],
    *,
    executor: ToolExecutor,
    tool_context: ToolContext,
    messages: list[Message],
    history_sink: list[Message] | None,
    skill_context_buffer,
) -> AsyncIterator[dict[str, Any]]:
    for call in tool_calls:
        name = call.get("function", {}).get("name", "unknown")
        yield {"type": "tool_call", "name": name, "input": _tool_input(call)}

    load_skill_ids = [
        str(call.get("id") or "")
        for call in tool_calls
        if str(call.get("function", {}).get("name") or "") == "load_skill"
    ]
    injection_target = load_skill_ids[-1] if load_skill_ids else ""
    result_messages: dict[str, Message] = {}
    async for result in executor.execute_stream(tool_calls, tool_context):
        yield {
            "type": "tool_result",
            "name": _tool_name_by_id(tool_calls, result.tool_use_id or ""),
            "result": result.content,
            "is_error": result.is_error,
            "replayed": result.replayed,
            "execution_key": result.execution_key,
            "recovery_status": result.recovery_status,
        }
        result_message = Message(
            role="tool",
            content=result.content,
            tool_call_id=result.tool_use_id,
        )
        messages.append(result_message)
        result_messages[str(result.tool_use_id or "")] = result_message
        _sync_history(history_sink, messages)

    loaded_skill_context = _drain_skill_context(skill_context_buffer)
    injection_message = result_messages.get(injection_target)
    if loaded_skill_context and injection_message is not None:
        injection_message.content = f"{injection_message.content}\n\n{loaded_skill_context}"
    elif loaded_skill_context:
        messages.append(
            Message(
                role="user",
                content=(
                    f"{loaded_skill_context}\n\n"
                    "Use these loaded instructions to continue the current request."
                ),
            )
        )
    _sync_history(history_sink, messages)


def _sync_history(target: list[Message] | None, messages: list[Message]) -> None:
    if target is not None:
        target[:] = messages


def _merge_tool_delta(tool_states: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    index = int(delta.get("index") or 0)
    state = tool_states.setdefault(
        index,
        {
            "id": delta.get("id") or f"tool_{index}",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if delta.get("id"):
        state["id"] = delta["id"]
    function = delta.get("function") or {}
    if function.get("name"):
        state["function"]["name"] = function["name"]
    if function.get("arguments"):
        state["function"]["arguments"] += function["arguments"]


def _finalize_tool_calls(tool_states: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for index in sorted(tool_states):
        state = tool_states[index]
        if state["function"]["name"]:
            calls.append(state)
    return calls


def _tool_input(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("function", {}).get("arguments") or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _tool_name_by_id(calls: list[dict[str, Any]], tool_call_id: str) -> str:
    for call in calls:
        if call.get("id") == tool_call_id:
            return str(call.get("function", {}).get("name") or "unknown")
    return "unknown"


def _prepend_skill_context(user_message: str, skill_context_buffer) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return user_message
    drained = skill_context_buffer.drain()
    if not drained:
        return user_message
    return f"{drained}\n\n---\nUser request:\n{user_message}"


def _drain_skill_context(skill_context_buffer) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return ""
    return skill_context_buffer.drain()


def _prepend_skill_candidates(user_message: str, cwd: str, config: VelaConfig) -> str:
    if not config.features.skill:
        return user_message
    candidates = SkillRegistry(cwd).match(user_message, top_k=5)
    if not candidates:
        return user_message
    lines = [
        "Relevant skill candidates for this request:",
        "Call load_skill(name) before proceeding when a candidate applies.",
    ]
    for skill in candidates:
        description = " ".join(skill.description.split())
        if len(description) > 300:
            description = description[:297] + "..."
        tags = f" [tags: {', '.join(skill.tags)}]" if skill.tags else ""
        lines.append(f"- {skill.name}: {description}{tags}")
    candidate_text = "\n".join(lines)
    return f"{candidate_text}\n\n---\nUser request:\n{user_message}"
