"""Vela's small, explicit model -> tool -> model ReAct loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from vela.config import VelaConfig
from vela.context import ContextBudget, ContextEngine, ContextOverflowError, ContextResult
from vela.events import AgentEvent
from vela.image import parse_image_references
from vela.llm.base import LlmClient
from vela.prompt import PromptAssembler
from vela.skill import SkillContextBuffer, SkillRegistry
from vela.tools.base import ToolContext, ToolDecision, ToolResult
from vela.tools.calls import tool_call_arguments, tool_call_name
from vela.tools.executor import ToolExecutor
from vela.tools.registry import ToolRegistry
from vela.types import Message, Usage


@dataclass(slots=True)
class _ModelTurn:
    """One streamed model response collected into a complete assistant turn."""

    text: str = ""
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    tool_fragments: dict[int, dict[str, Any]] = field(default_factory=dict)
    failed: bool = False
    error: BaseException | None = None

    def tool_calls(self) -> list[dict[str, Any]]:
        return _complete_tool_calls(self.tool_fragments)


async def run_react_agent(
    *,
    llm_client: LlmClient,
    tool_registry: ToolRegistry,
    system_prompt: str,
    user_message: str,
    history: list[Message] | None,
    cwd: str,
    config: VelaConfig,
    approval_callback: (
        Callable[[dict[str, Any]], Awaitable[ToolDecision] | ToolDecision] | None
    ) = None,
    skill_context_buffer: SkillContextBuffer | None = None,
    tool_execution_scope: str | None = None,
    allow_uncertain_tool_retry: bool = False,
    steering_callback: Callable[[], str | None] | None = None,
    max_turns: int = 20,
) -> AsyncIterator[AgentEvent]:
    """Run one request and update a supplied ``history`` transcript in place.

    In-place updates happen before progress events are yielded, so callers can
    safely persist completed model and tool work after cancellation or failure.
    """

    if max_turns < 1:
        yield {"type": "error", "error": ValueError("max_turns must be at least 1")}
        return

    original_user_message = user_message
    user_message = _prepend_skill_candidates(user_message, cwd, config)
    user_message = _prepend_skill_context(user_message, skill_context_buffer)

    transcript = history if history is not None else []
    transcript.append(Message(role="user", content=parse_image_references(user_message, cwd)))

    tool_definitions = tool_registry.definitions()
    effective_system_prompt = _build_system_prompt(
        base=system_prompt,
        user_message=original_user_message,
        llm_client=llm_client,
        tool_registry=tool_registry,
        cwd=cwd,
        config=config,
    )
    context_engine = _context_engine(llm_client, config)
    tool_executor = ToolExecutor(tool_registry)
    tool_context = ToolContext(
        cwd=cwd,
        config=config,
        approval_callback=approval_callback,
        skill_context_buffer=skill_context_buffer,
        execution_scope=tool_execution_scope,
        allow_uncertain_retry=allow_uncertain_tool_retry,
    )

    total_usage = Usage()
    completed_turns = 0
    try:
        for turn_number in range(1, max_turns + 1):
            overflow_retried = False
            compression_event = _compress_context(
                transcript,
                context_engine=context_engine,
                system_prompt=effective_system_prompt,
                tool_definitions=tool_definitions,
                enabled=config.features.context_compression,
            )
            if compression_event is not None:
                yield compression_event

            yield {"type": "turn_started", "turn": turn_number}

            while True:
                model_turn = _ModelTurn()
                model_stream = _stream_model_turn(
                    llm_client,
                    transcript,
                    tool_definitions,
                    effective_system_prompt,
                    model_turn,
                )
                try:
                    async for event in model_stream:
                        yield event
                finally:
                    await model_stream.aclose()
                if not model_turn.failed:
                    break
                error = model_turn.error or RuntimeError("Model request failed")
                if (
                    isinstance(error, ContextOverflowError)
                    and config.features.context_compression
                    and not overflow_retried
                ):
                    try:
                        recovered = context_engine.recover_from_overflow(
                            transcript,
                            system_prompt=effective_system_prompt,
                            tool_definitions=tool_definitions,
                        )
                    except ContextOverflowError as recovery_error:
                        yield {"type": "error", "error": recovery_error}
                        return
                    transcript[:] = recovered.messages
                    overflow_retried = True
                    yield _context_compressed_event(recovered, recovered_from_overflow=True)
                    continue
                yield {"type": "error", "error": error}
                return

            completed_turns = turn_number
            total_usage = total_usage + model_turn.usage
            tool_calls = model_turn.tool_calls()
            if _has_incomplete_tool_request(model_turn, tool_calls):
                raise RuntimeError("Model returned an incomplete tool-call stream.")
            transcript.append(
                Message(role="assistant", content=model_turn.text, tool_calls=tool_calls)
            )
            yield {
                "type": "model_response_complete",
                "turn": turn_number,
                "stop_reason": model_turn.stop_reason,
            }

            if tool_calls and turn_number == max_turns:
                _close_skipped_tool_calls(transcript, tool_calls, max_turns)
                raise RuntimeError(f"Agent reached the model turn limit ({max_turns}).")
            if tool_calls:
                tool_stream = _execute_tool_round(
                    tool_calls,
                    executor=tool_executor,
                    context=tool_context,
                    transcript=transcript,
                    skill_context_buffer=skill_context_buffer,
                )
                try:
                    async for event in tool_stream:
                        yield event
                finally:
                    await tool_stream.aclose()
            yield {
                "type": "turn_complete",
                "turn": turn_number,
                "stop_reason": model_turn.stop_reason,
            }
            steering = (
                steering_callback() if steering_callback and turn_number < max_turns else None
            )
            if steering:
                _append_steering_message(
                    transcript,
                    steering,
                    cwd=cwd,
                    config=config,
                    skill_context_buffer=skill_context_buffer,
                )
                yield {"type": "steering_applied"}
                continue
            if not tool_calls:
                break
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - failures are part of the Agent event protocol
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError from exc
        yield {"type": "error", "error": exc}
        return

    yield {
        "type": "done",
        "total_turns": completed_turns,
        "total_tokens": total_usage.total_tokens,
        "usage": total_usage.to_dict(),
        "messages": list(transcript),
    }


async def _stream_model_turn(
    client: LlmClient,
    transcript: list[Message],
    tool_definitions: list[dict[str, Any]],
    system_prompt: str,
    turn: _ModelTurn,
) -> AsyncIterator[AgentEvent]:
    events = client.chat(
        transcript,
        tool_definitions,
        system_prompt=system_prompt,
    )
    try:
        async for event in events:
            event_type = event.get("type")
            if event_type == "text_delta":
                text = str(event.get("text") or "")
                turn.text += text
                yield {"type": "text_delta", "text": text}
            elif event_type == "thinking_delta":
                yield {"type": "thinking_delta", "thinking": str(event.get("thinking") or "")}
            elif event_type == "tool_call_delta":
                fragment = event.get("tool_call")
                if isinstance(fragment, dict):
                    _merge_tool_fragment(turn.tool_fragments, fragment)
            elif event_type == "message_end":
                turn.stop_reason = str(event.get("stop_reason") or "end_turn")
            elif event_type == "usage":
                usage = Usage.from_mapping(event.get("usage") or {})
                turn.usage = turn.usage + usage
                yield {"type": "usage", "usage": usage.to_dict()}
            elif event_type == "error":
                error = event.get("error")
                turn.failed = True
                turn.error = error if isinstance(error, BaseException) else RuntimeError(str(error))
                return
    finally:
        close = getattr(events, "aclose", None)
        if close is not None:
            await close()


async def _execute_tool_round(
    tool_calls: list[dict[str, Any]],
    *,
    executor: ToolExecutor,
    context: ToolContext,
    transcript: list[Message],
    skill_context_buffer: SkillContextBuffer | None,
) -> AsyncIterator[AgentEvent]:
    calls_by_id = {str(call["id"]): call for call in tool_calls}
    for call in tool_calls:
        yield {
            "type": "tool_call",
            "tool_call_id": str(call["id"]),
            "name": tool_call_name(call) or "unknown",
            "input": tool_call_arguments(call),
        }

    results = executor.execute_stream(tool_calls, context)
    try:
        async for result in results:
            call_id = str(result.tool_use_id or "")
            call = calls_by_id.get(call_id, {})
            name = tool_call_name(call) or "unknown"
            if name == "load_skill":
                loaded_context = _drain_skill_context(skill_context_buffer)
                if loaded_context:
                    result.content = f"{result.content}\n\n{loaded_context}"

            message = Message(role="tool", content=result.content, tool_call_id=call_id)
            transcript.append(message)
            yield _tool_result_event(call_id, name, result)
    finally:
        await results.aclose()


def _merge_tool_fragment(states: dict[int, dict[str, Any]], fragment: dict[str, Any]) -> None:
    index = _tool_fragment_index(states, fragment)
    state = states.setdefault(
        index,
        {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if fragment.get("id"):
        state["id"] = str(fragment["id"])
    function = fragment.get("function")
    if not isinstance(function, dict):
        return
    if function.get("name"):
        state["function"]["name"] = _merge_text_fragment(
            str(state["function"]["name"]),
            str(function["name"]),
        )
    if function.get("arguments") is not None:
        state["function"]["arguments"] += str(function["arguments"])


def _merge_text_fragment(current: str, incoming: str) -> str:
    """Accept both incremental deltas and providers that repeat the full value."""
    if not current or not incoming:
        return current or incoming
    if incoming == current:
        return current
    if incoming.startswith(current):
        return incoming
    return current + incoming


def _complete_tool_calls(states: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index in sorted(states):
        call = states[index]
        if not tool_call_name(call):
            continue
        call["id"] = _unique_tool_id(str(call.get("id") or f"tool_{index}"), used_ids)
        calls.append(call)
    return calls


def _close_skipped_tool_calls(
    transcript: list[Message],
    tool_calls: list[dict[str, Any]],
    max_turns: int,
) -> None:
    """Keep persisted provider history valid when the final turn asks for tools."""
    for call in tool_calls:
        name = tool_call_name(call) or "unknown"
        transcript.append(
            Message(
                role="tool",
                tool_call_id=str(call["id"]),
                content=(
                    f'Tool "{name}" was not executed because the Agent reached '
                    f"the model turn limit ({max_turns})."
                ),
            )
        )


def _tool_fragment_index(
    states: dict[int, dict[str, Any]],
    fragment: dict[str, Any],
) -> int:
    raw_index = fragment.get("index")
    if raw_index is not None:
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid tool-call fragment index: {raw_index!r}") from exc
        if index < 0:
            raise RuntimeError(f"Invalid tool-call fragment index: {index}")
        return index

    fragment_id = str(fragment.get("id") or "")
    matching = [index for index, state in states.items() if state.get("id") == fragment_id]
    if fragment_id and matching:
        if len(matching) == 1:
            return matching[0]
        raise RuntimeError(f"Ambiguous tool-call fragment with duplicate id: {fragment_id!r}")
    if not states:
        return 0
    if len(states) == 1 and not fragment_id:
        return next(iter(states))

    function = fragment.get("function")
    name = str(function.get("name") or "") if isinstance(function, dict) else ""
    if fragment_id and name:
        return max(states) + 1
    raise RuntimeError("Ambiguous tool-call fragment without a stable index or id.")


def _has_incomplete_tool_request(
    turn: _ModelTurn,
    tool_calls: list[dict[str, Any]],
) -> bool:
    return len(tool_calls) != len(turn.tool_fragments) or (
        turn.stop_reason == "tool_use" and not tool_calls
    )


def _unique_tool_id(candidate: str, used_ids: set[str]) -> str:
    if candidate not in used_ids:
        used_ids.add(candidate)
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used_ids:
        suffix += 1
    unique = f"{candidate}_{suffix}"
    used_ids.add(unique)
    return unique


def _tool_result_event(tool_call_id: str, name: str, result: ToolResult) -> AgentEvent:
    return {
        "type": "tool_result",
        "tool_call_id": tool_call_id,
        "name": name,
        "result": result.content,
        "is_error": result.is_error,
        "replayed": result.replayed,
        "execution_key": result.execution_key,
        "recovery_status": result.recovery_status,
    }


def _compress_context(
    transcript: list[Message],
    *,
    context_engine: ContextEngine,
    system_prompt: str,
    tool_definitions: list[dict[str, Any]],
    enabled: bool,
) -> AgentEvent | None:
    if not enabled:
        return None
    compression = context_engine.prepare(
        transcript,
        system_prompt=system_prompt,
        tool_definitions=tool_definitions,
    )
    transcript[:] = compression.messages
    if not compression.compressed:
        return None
    return _context_compressed_event(compression)


def _context_compressed_event(
    compression: ContextResult,
    *,
    recovered_from_overflow: bool = False,
) -> AgentEvent:
    event: AgentEvent = {
        "type": "context_compressed",
        "before_tokens": compression.estimated_tokens_before,
        "after_tokens": compression.estimated_tokens_after,
        "summarized_messages": compression.summarized_messages,
        "truncated_tool_results": compression.truncated_tool_results,
        "omitted_tool_characters": compression.omitted_tool_characters,
    }
    if recovered_from_overflow:
        event["recovered_from_overflow"] = True
    return event


def _build_system_prompt(
    *,
    base: str,
    user_message: str,
    llm_client: LlmClient,
    tool_registry: ToolRegistry,
    cwd: str,
    config: VelaConfig,
) -> str:
    dynamic = PromptAssembler(
        config=config,
        cwd=cwd,
        tool_names=tool_registry.list_names(),
        model=llm_client.model_name,
        provider=llm_client.provider_name,
    ).build_dynamic(user_message)
    return f"{base}\n\n{dynamic}".strip()


def _context_engine(llm_client: LlmClient, config: VelaConfig) -> ContextEngine:
    return ContextEngine(
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


def _prepend_skill_context(
    user_message: str,
    skill_context_buffer: SkillContextBuffer | None,
) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return user_message
    loaded = skill_context_buffer.drain()
    if not loaded:
        return user_message
    return f"{loaded}\n\n---\nUser request:\n{user_message}"


def _append_steering_message(
    transcript: list[Message],
    message: str,
    *,
    cwd: str,
    config: VelaConfig,
    skill_context_buffer: SkillContextBuffer | None,
) -> None:
    enriched = _prepend_skill_candidates(message, cwd, config)
    enriched = _prepend_skill_context(enriched, skill_context_buffer)
    transcript.append(Message(role="user", content=parse_image_references(enriched, cwd)))


def _drain_skill_context(skill_context_buffer: SkillContextBuffer | None) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return ""
    return skill_context_buffer.drain()


def _prepend_skill_candidates(user_message: str, cwd: str, config: VelaConfig) -> str:
    if not config.features.skill:
        return user_message
    candidates = SkillRegistry(cwd, include_project=config.project_trusted).match(
        user_message,
        top_k=5,
    )
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
