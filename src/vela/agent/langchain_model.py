"""LangChain chat-model adapter for Vela's provider client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel, agenerate_from_stream
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from vela.types import Message, Usage


class ModelCancelledError(Exception):
    """Carry provider cancellation through LangGraph's internal task boundary."""


class VelaChatModel(BaseChatModel):
    """Expose Vela's streaming LLM client through LangChain's chat-model API."""

    client: Any = Field(exclude=True)
    tool_definitions: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "vela_openai_compatible"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.client.model_name,
            "provider": self.client.provider_name,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        del tool_choice, kwargs
        definitions = [convert_to_openai_tool(tool) for tool in tools]
        return self.model_copy(update={"tool_definitions": definitions})

    def _generate(self, *_args: Any, **_kwargs: Any) -> ChatResult:
        raise RuntimeError("VelaChatModel supports asynchronous execution only.")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await agenerate_from_stream(
            self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del stop, run_manager, kwargs
        system_prompt, vela_messages = from_langchain_messages(messages)
        streamed_tool_indices: set[int] = set()
        try:
            async for event in self.client.chat(
                vela_messages,
                self.tool_definitions,
                system_prompt=system_prompt,
            ):
                event_type = event.get("type")
                if event_type == "text_delta":
                    yield _chat_chunk(content=str(event.get("text") or ""))
                elif event_type == "thinking_delta":
                    yield _chat_chunk(
                        additional_kwargs={"reasoning_content": str(event.get("thinking") or "")}
                    )
                elif event_type == "tool_call_delta":
                    yield _tool_call_chunk(event.get("tool_call") or {}, streamed_tool_indices)
                elif event_type == "message_end":
                    yield _chat_chunk(
                        response_metadata={
                            "stop_reason": str(event.get("stop_reason") or "end_turn")
                        }
                    )
                elif event_type == "usage":
                    usage = Usage.from_mapping(event.get("usage") or {})
                    yield _chat_chunk(
                        usage=usage,
                        response_metadata={"vela_usage": usage.to_dict()},
                    )
                elif event_type == "error":
                    error = event.get("error")
                    raise error if isinstance(error, Exception) else RuntimeError(str(error))
        except asyncio.CancelledError as exc:
            raise ModelCancelledError from exc


def to_langchain_messages(messages: list[Message]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        if message.role == "system":
            converted.append(SystemMessage(content=message.content))
        elif message.role == "user":
            converted.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            converted.append(
                AIMessage(
                    content=message.content,
                    tool_calls=[_from_openai_tool_call(call) for call in message.tool_calls],
                )
            )
        elif message.role == "tool":
            converted.append(
                ToolMessage(
                    content=message.content,
                    tool_call_id=message.tool_call_id or "",
                )
            )
    return converted


def from_langchain_messages(messages: list[BaseMessage]) -> tuple[str, list[Message]]:
    system_parts: list[str] = []
    converted: list[Message] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system_parts.append(text_content(message.content))
        elif isinstance(message, HumanMessage):
            converted.append(Message(role="user", content=message.content))
        elif isinstance(message, AIMessage):
            converted.append(from_ai_message(message))
        elif isinstance(message, ToolMessage):
            converted.append(
                Message(
                    role="tool",
                    content=message.content,
                    tool_call_id=message.tool_call_id,
                )
            )
    return "\n\n".join(part for part in system_parts if part), converted


def from_ai_message(message: AIMessage) -> Message:
    return Message(
        role="assistant",
        content=message.content,
        tool_calls=[to_openai_tool_call(call) for call in message.tool_calls],
    )


def to_openai_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(call.get("id") or ""),
        "type": "function",
        "function": {
            "name": str(call.get("name") or "unknown"),
            "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False),
        },
    }


def text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text") or ""))
    return "".join(parts)


def _chat_chunk(
    *,
    content: str = "",
    additional_kwargs: dict[str, Any] | None = None,
    response_metadata: dict[str, Any] | None = None,
    usage: Usage | None = None,
) -> ChatGenerationChunk:
    usage_metadata = None
    if usage is not None:
        usage_metadata = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs or {},
            response_metadata=response_metadata or {},
            usage_metadata=usage_metadata,
        )
    )


def _tool_call_chunk(
    delta: dict[str, Any],
    streamed_indices: set[int],
) -> ChatGenerationChunk:
    function = delta.get("function") if isinstance(delta.get("function"), dict) else {}
    index = int(delta.get("index") or 0)
    first_chunk = index not in streamed_indices
    if first_chunk:
        streamed_indices.add(index)
    raw_id = str(delta.get("id") or f"tool_{index}")
    chunk = {
        "name": function.get("name"),
        "args": function.get("arguments"),
        "id": raw_id if first_chunk else None,
        "index": index,
        "type": "tool_call_chunk",
    }
    return ChatGenerationChunk(
        message=AIMessageChunk(content="", tool_call_chunks=[chunk])  # type: ignore[list-item]
    )


def _from_openai_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    raw_arguments = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        arguments = {"raw": raw_arguments}
    return {
        "name": str(function.get("name") or "unknown"),
        "args": arguments if isinstance(arguments, dict) else {"value": arguments},
        "id": str(call.get("id") or ""),
        "type": "tool_call",
    }
