"""Collect one streamed provider response into a complete model turn."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, TypedDict

from vela.events import AgentEvent
from vela.llm.base import LlmClient
from vela.tools.calls import tool_call_name
from vela.types import Message, Usage


class _FunctionFragment(TypedDict):
    name: str
    arguments: str


class _ToolFragment(TypedDict):
    id: str
    type: str
    function: _FunctionFragment


@dataclass(slots=True)
class ModelTurn:
    """Mutable result populated while a provider response is streaming."""

    text: str = ""
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    tool_fragments: dict[int, _ToolFragment] = field(default_factory=dict)
    failed: bool = False
    error: BaseException | None = None

    def reset(self) -> None:
        """Discard a failed partial response before retrying the same turn."""
        self.text = ""
        self.stop_reason = "end_turn"
        self.usage = Usage()
        self.tool_fragments.clear()
        self.failed = False
        self.error = None

    def tool_calls(self) -> list[dict[str, Any]]:
        return _complete_tool_calls(self.tool_fragments)

    def has_incomplete_tool_request(self, tool_calls: list[dict[str, Any]]) -> bool:
        return len(tool_calls) != len(self.tool_fragments) or (
            self.stop_reason == "tool_use" and not tool_calls
        )


async def stream_model_turn(
    client: LlmClient,
    transcript: list[Message],
    tool_definitions: list[dict[str, Any]],
    system_prompt: str,
    turn: ModelTurn,
) -> AsyncIterator[AgentEvent]:
    """Forward provider events while assembling the complete response."""
    events = client.chat(transcript, tool_definitions, system_prompt=system_prompt)
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


def _merge_tool_fragment(states: dict[int, _ToolFragment], fragment: dict[str, Any]) -> None:
    index = _tool_fragment_index(states, fragment)
    state = states.setdefault(
        index,
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
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


def _complete_tool_calls(states: dict[int, _ToolFragment]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index in sorted(states):
        call = states[index]
        if not tool_call_name(call):
            continue
        call["id"] = _unique_tool_id(str(call.get("id") or f"tool_{index}"), used_ids)
        calls.append(call)
    return calls


def _tool_fragment_index(
    states: dict[int, _ToolFragment],
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
