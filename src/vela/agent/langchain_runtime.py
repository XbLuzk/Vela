"""LangChain-backed ReAct runtime with Vela's model and tool semantics."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from vela.agent.langchain_model import (
    ModelCancelledError,
    VelaChatModel,
    from_ai_message,
    from_langchain_messages,
    text_content,
    to_langchain_messages,
)
from vela.agent.langchain_tools import VelaToolMiddleware, langchain_tools
from vela.config import VelaConfig
from vela.context import ContextBudget, ContextWindowManager
from vela.image import parse_image_references
from vela.llm.base import LlmClient
from vela.prompt import PromptAssembler
from vela.skill import SkillRegistry
from vela.tools.base import ToolContext
from vela.tools.registry import ToolRegistry
from vela.types import Message, Usage


class _ContextCompressionMiddleware(AgentMiddleware):
    """Keep LangChain message state inside Vela's deterministic context budget."""

    def __init__(
        self,
        *,
        manager: ContextWindowManager,
        system_prompt: str,
        tool_definitions: list[dict[str, Any]],
        transcript: list[Message],
        enabled: bool,
    ) -> None:
        super().__init__()
        self.manager = manager
        self.system_prompt = system_prompt
        self.tool_definitions = tool_definitions
        self.transcript = transcript
        self.enabled = enabled

    async def abefore_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        return self._prepare(state, runtime)

    def _prepare(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        _, messages = from_langchain_messages(list(state.get("messages") or []))
        compression = self.manager.prepare(
            messages,
            system_prompt=self.system_prompt,
            tool_definitions=self.tool_definitions,
        )
        if not compression.compressed:
            return None

        self.transcript[:] = compression.messages
        runtime.stream_writer(
            {
                "type": "context_compressed",
                "before_tokens": compression.estimated_tokens_before,
                "after_tokens": compression.estimated_tokens_after,
                "summarized_messages": compression.summarized_messages,
            }
        )
        replacement: list[BaseMessage] = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        replacement.extend(to_langchain_messages(compression.messages))
        return {"messages": replacement}


async def run_langchain_agent(
    *,
    llm_client: LlmClient,
    tool_registry: ToolRegistry,
    system_prompt: str,
    user_message: str,
    history: list[Message] | None,
    cwd: str,
    config: VelaConfig,
    approval_callback: Any = None,
    skill_context_buffer: Any = None,
    tool_execution_scope: str | None = None,
    allow_uncertain_tool_retry: bool = False,
    max_turns: int = 20,
) -> AsyncIterator[dict[str, Any]]:
    """Run one Vela request through LangChain's standard agent graph."""

    original_user_message = user_message
    user_message = _prepend_skill_candidates(user_message, cwd, config)
    user_message = _prepend_skill_context(user_message, skill_context_buffer)
    transcript = history if history is not None else []
    transcript.append(Message(role="user", content=parse_image_references(user_message, cwd)))

    effective_system_prompt = _build_system_prompt(
        base=system_prompt,
        user_message=original_user_message,
        llm_client=llm_client,
        tool_registry=tool_registry,
        cwd=cwd,
        config=config,
    )
    tool_definitions = tool_registry.definitions()
    tool_context = ToolContext(
        cwd=cwd,
        config=config,
        approval_callback=approval_callback,
        skill_context_buffer=skill_context_buffer,
        execution_scope=tool_execution_scope,
        allow_uncertain_retry=allow_uncertain_tool_retry,
    )
    middleware = [
        _ContextCompressionMiddleware(
            manager=_context_manager(llm_client, config),
            system_prompt=effective_system_prompt,
            tool_definitions=tool_definitions,
            transcript=transcript,
            enabled=config.features.context_compression,
        ),
        ModelCallLimitMiddleware(run_limit=max_turns, exit_behavior="end"),
        VelaToolMiddleware(
            registry=tool_registry,
            context=tool_context,
            transcript=transcript,
        ),
    ]
    graph = create_agent(
        model=VelaChatModel(client=llm_client),
        tools=langchain_tools(tool_registry),
        system_prompt=effective_system_prompt,
        middleware=middleware,
        name="vela_react_agent",
    )

    usage = Usage()
    turns = 0
    try:
        async for mode, payload in graph.astream(
            {"messages": to_langchain_messages(transcript)},
            stream_mode=["messages", "updates", "custom"],
            config={"recursion_limit": max(25, max_turns * 8)},
        ):
            if mode == "messages":
                event, current_usage = _streamed_message_event(payload)
                if current_usage is not None:
                    usage = usage + current_usage
                if event is not None:
                    yield event
            elif mode == "custom" and isinstance(payload, dict):
                yield payload
            elif mode == "updates" and isinstance(payload, dict) and "model" in payload:
                update_messages = payload["model"].get("messages") or []
                for model_message in update_messages:
                    if not isinstance(model_message, AIMessage):
                        continue
                    turns += 1
                    transcript.append(from_ai_message(model_message))
                    yield _turn_complete_event(model_message, turns)
                    for call in model_message.tool_calls:
                        yield {
                            "type": "tool_call",
                            "name": str(call.get("name") or "unknown"),
                            "input": call.get("args") if isinstance(call.get("args"), dict) else {},
                        }
    except ModelCancelledError as exc:
        raise asyncio.CancelledError from exc
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - errors remain part of Vela's event protocol
        yield {"type": "error", "error": exc}
        return

    yield {
        "type": "done",
        "total_turns": turns,
        "total_tokens": usage.total_tokens,
        "usage": usage.to_dict(),
        "messages": list(transcript),
    }


def _streamed_message_event(payload: Any) -> tuple[dict[str, Any] | None, Usage | None]:
    chunk, metadata = payload
    if metadata.get("langgraph_node") != "model" or not isinstance(chunk, AIMessageChunk):
        return None, None
    text = text_content(chunk.content)
    if text:
        return {"type": "text_delta", "text": text}, None
    thinking = str(chunk.additional_kwargs.get("reasoning_content") or "")
    if thinking:
        return {"type": "thinking_delta", "thinking": thinking}, None
    raw_usage = chunk.response_metadata.get("vela_usage")
    if isinstance(raw_usage, dict):
        usage = Usage.from_mapping(raw_usage)
        return {"type": "usage", "usage": usage.to_dict()}, usage
    return None, None


def _turn_complete_event(message: AIMessage, turn: int) -> dict[str, Any]:
    stop_reason = str(
        message.response_metadata.get("stop_reason")
        or ("tool_use" if message.tool_calls else "end_turn")
    )
    return {"type": "turn_complete", "turn": turn, "stop_reason": stop_reason}


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


def _context_manager(llm_client: LlmClient, config: VelaConfig) -> ContextWindowManager:
    return ContextWindowManager(
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


def _prepend_skill_context(user_message: str, skill_context_buffer: Any) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return user_message
    drained = skill_context_buffer.drain()
    if not drained:
        return user_message
    return f"{drained}\n\n---\nUser request:\n{user_message}"


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
