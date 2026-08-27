"""Vela's small, explicit model -> tool -> model ReAct loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from vela.agent.model_turn import ModelTurn, stream_model_turn
from vela.config import VelaConfig
from vela.context import ContextBudget, ContextEngine, ContextOverflowError, ContextResult
from vela.events import AgentEvent
from vela.image import parse_image_references
from vela.llm.base import LlmClient
from vela.prompt import PromptAssembler
from vela.skill import SkillContextBuffer, SkillRegistry
from vela.tools.base import ToolContext, ToolResult
from vela.tools.calls import tool_call_arguments, tool_call_name
from vela.tools.executor import ToolExecutor
from vela.tools.registry import ToolRegistry
from vela.types import Message, Usage


@dataclass(slots=True)
class ReactRuntime:
    """Stable dependencies for one or more ordinary ReAct requests."""

    llm_client: LlmClient
    tool_registry: ToolRegistry
    system_prompt: str
    tool_context: ToolContext
    max_turns: int = 20


@dataclass(slots=True)
class _PreparedReactRun:
    llm_client: LlmClient
    config: VelaConfig
    cwd: str
    transcript: list[Message]
    tool_definitions: list[dict[str, Any]]
    system_prompt: str
    context_engine: ContextEngine
    tool_executor: ToolExecutor
    tool_context: ToolContext
    skill_context_buffer: SkillContextBuffer | None


async def run_react_agent(
    user_message: str,
    history: list[Message] | None,
    runtime: ReactRuntime,
) -> AsyncIterator[AgentEvent]:
    """Run one request and update a supplied ``history`` transcript in place.

    In-place updates happen before progress events are yielded, so callers can
    safely persist completed model and tool work after cancellation or failure.
    """

    if runtime.max_turns < 1:
        yield {"type": "error", "error": ValueError("max_turns must be at least 1")}
        return

    cwd = runtime.tool_context.cwd
    config = runtime.tool_context.config
    skill_context_buffer = runtime.tool_context.skill_context_buffer
    original_user_message = user_message
    user_message = _prepend_skill_candidates(user_message, cwd, config)
    user_message = _prepend_skill_context(user_message, skill_context_buffer)

    transcript = history if history is not None else []
    transcript.append(Message(role="user", content=parse_image_references(user_message, cwd)))

    prepared = _PreparedReactRun(
        llm_client=runtime.llm_client,
        config=config,
        cwd=cwd,
        transcript=transcript,
        tool_definitions=runtime.tool_registry.definitions(),
        system_prompt=_build_system_prompt(
            base=runtime.system_prompt,
            user_message=original_user_message,
            llm_client=runtime.llm_client,
            tool_registry=runtime.tool_registry,
            cwd=cwd,
            config=config,
        ),
        context_engine=_context_engine(runtime.llm_client, config),
        tool_executor=ToolExecutor(runtime.tool_registry),
        tool_context=runtime.tool_context,
        skill_context_buffer=skill_context_buffer,
    )

    total_usage = Usage()
    completed_turns = 0
    try:
        for turn_number in range(1, runtime.max_turns + 1):
            model_turn = ModelTurn()
            turn_stream = _stream_react_turn(
                prepared,
                turn_number=turn_number,
                max_turns=runtime.max_turns,
                turn=model_turn,
            )
            try:
                async for event in turn_stream:
                    yield event
            finally:
                await turn_stream.aclose()
            if model_turn.failed:
                return

            completed_turns = turn_number
            total_usage = total_usage + model_turn.usage
            if not model_turn.tool_fragments:
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


async def _stream_react_turn(
    runtime: _PreparedReactRun,
    *,
    turn_number: int,
    max_turns: int,
    turn: ModelTurn,
) -> AsyncIterator[AgentEvent]:
    compression_event = _compress_context(
        runtime.transcript,
        context_engine=runtime.context_engine,
        system_prompt=runtime.system_prompt,
        tool_definitions=runtime.tool_definitions,
        enabled=runtime.config.features.context_compression,
    )
    if compression_event is not None:
        yield compression_event
    yield {"type": "turn_started", "turn": turn_number}

    model_stream = _stream_model_with_overflow_recovery(runtime=runtime, turn=turn)
    try:
        async for event in model_stream:
            yield event
    finally:
        await model_stream.aclose()
    if turn.failed:
        return

    tool_calls = turn.tool_calls()
    if turn.has_incomplete_tool_request(tool_calls):
        raise RuntimeError("Model returned an incomplete tool-call stream.")
    runtime.transcript.append(Message(role="assistant", content=turn.text, tool_calls=tool_calls))
    yield {
        "type": "model_response_complete",
        "turn": turn_number,
        "stop_reason": turn.stop_reason,
    }

    if tool_calls and turn_number == max_turns:
        _close_skipped_tool_calls(runtime.transcript, tool_calls, max_turns)
        raise RuntimeError(f"Agent reached the model turn limit ({max_turns}).")
    if tool_calls:
        tool_stream = _execute_tool_round(
            tool_calls,
            executor=runtime.tool_executor,
            context=runtime.tool_context,
            transcript=runtime.transcript,
            skill_context_buffer=runtime.skill_context_buffer,
        )
        try:
            async for event in tool_stream:
                yield event
        finally:
            await tool_stream.aclose()
    yield {
        "type": "turn_complete",
        "turn": turn_number,
        "stop_reason": turn.stop_reason,
    }


async def _stream_model_with_overflow_recovery(
    *,
    runtime: _PreparedReactRun,
    turn: ModelTurn,
) -> AsyncIterator[AgentEvent]:
    overflow_retried = False
    while True:
        turn.reset()
        events = stream_model_turn(
            runtime.llm_client,
            runtime.transcript,
            runtime.tool_definitions,
            runtime.system_prompt,
            turn,
        )
        try:
            async for event in events:
                yield event
        finally:
            await events.aclose()
        if not turn.failed:
            return

        error = turn.error or RuntimeError("Model request failed")
        if not (
            isinstance(error, ContextOverflowError)
            and runtime.config.features.context_compression
            and not overflow_retried
        ):
            yield {"type": "error", "error": error}
            return
        try:
            recovered = runtime.context_engine.recover_from_overflow(
                runtime.transcript,
                system_prompt=runtime.system_prompt,
                tool_definitions=runtime.tool_definitions,
            )
        except ContextOverflowError as recovery_error:
            turn.error = recovery_error
            yield {"type": "error", "error": recovery_error}
            return
        runtime.transcript[:] = recovered.messages
        overflow_retried = True
        yield _context_compressed_event(recovered, recovered_from_overflow=True)


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
        ),
        max_history_messages=config.memory.max_conversation_history,
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
