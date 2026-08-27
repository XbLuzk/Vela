from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from vela.branding import USER_AGENT
from vela.context import ContextOverflowError
from vela.events import LlmEvent
from vela.types import Message, Usage


@dataclass(slots=True)
class OpenAICompatibleClient:
    provider_name: str
    model: str
    api_key: str
    base_url: str
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: float = 120.0
    max_context_window: int = 128_000

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def supports_images(self) -> bool:
        model = self.model.lower()
        return any(marker in model for marker in ("vision", "image", "5v", "vl"))

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        system_prompt: str,
    ) -> AsyncIterator[LlmEvent]:
        if not self.api_key:
            yield {
                "type": "error",
                "error": RuntimeError(
                    "VELA_API_KEY is not configured. Set it in env or ~/.vela/config.json."
                ),
            }
            return

        if not _is_secure_endpoint(self.base_url):
            yield {
                "type": "error",
                "error": RuntimeError(
                    f"Refusing to send the API key to {self.base_url} over cleartext HTTP. "
                    "Use an https:// base URL, or a loopback host for local proxies."
                ),
            }
            return

        payload = self._build_payload(messages, tools, system_prompt=system_prompt)

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
            "user-agent": USER_AGENT,
        }
        url = self.base_url.rstrip("/") + "/chat/completions"

        yield {"type": "message_start", "model": self.model}
        try:
            async with (
                httpx.AsyncClient(timeout=self.timeout, http2=False) as client,
                client.stream("POST", url, headers=headers, json=payload) as response,
            ):
                if response.is_error:
                    await response.aread()
                response.raise_for_status()
                produced = False
                async for event in _iter_sse(response):
                    if event == "[DONE]":
                        break
                    try:
                        chunk = json.loads(event)
                    except json.JSONDecodeError:
                        if not _looks_like_json(event):
                            # Providers interleave non-JSON keep-alive payloads.
                            continue
                        yield {
                            "type": "error",
                            "error": RuntimeError(
                                f"{self.provider_name} sent a malformed streaming payload; "
                                "part of the response was lost. Retry the request."
                            ),
                        }
                        return
                    stream_error = _stream_error_message(chunk)
                    if stream_error:
                        yield {
                            "type": "error",
                            "error": RuntimeError(
                                f"{self.provider_name} reported a streaming error: {stream_error}"
                            ),
                        }
                        return
                    async for parsed in self._parse_chunk(chunk):
                        produced = True
                        yield parsed
                if not produced:
                    yield {
                        "type": "error",
                        "error": RuntimeError(
                            f"{self.provider_name} returned an empty response stream for model "
                            f"{self.model}. Check model access and provider status, then retry."
                        ),
                    }
        except httpx.TimeoutException:
            yield {
                "type": "error",
                "error": RuntimeError(
                    f"{self.provider_name} request timed out after {self.timeout:g}s. "
                    "Check the network and retry."
                ),
            }
        except httpx.HTTPStatusError as exc:
            if _is_context_overflow(exc.response):
                yield {
                    "type": "error",
                    "error": ContextOverflowError(
                        f"{self.provider_name} rejected the request because its context limit "
                        "was exceeded."
                    ),
                }
                return
            detail = _response_detail(exc.response)
            yield {
                "type": "error",
                "error": RuntimeError(
                    f"{self.provider_name} API returned HTTP {exc.response.status_code}. "
                    "Check the API key, model access, account balance, and provider status."
                    + (f" Provider said: {detail}" if detail else "")
                ),
            }
        except httpx.RequestError as exc:
            yield {
                "type": "error",
                "error": RuntimeError(
                    f"Could not connect to {self.provider_name} at {self.base_url}. "
                    "Check the network, VPN/proxy, and provider status, then retry. "
                    f"({type(exc).__name__}: {exc})"
                ),
            }

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        system_prompt: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages, system_prompt),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def _format_messages(self, messages: list[Message], system_prompt: str) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in messages:
            if message.role == "tool":
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id or "",
                        "content": str(message.content),
                    }
                )
            elif message.role == "assistant":
                item: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
                if message.tool_calls:
                    item["tool_calls"] = message.tool_calls
                formatted.append(item)
            else:
                formatted.append(
                    {"role": message.role, "content": self._format_content(message.content)}
                )
        return formatted

    def _format_content(self, content: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        if self.supports_images:
            cleaned = []
            for part in content:
                item = {key: value for key, value in part.items() if key != "metadata"}
                cleaned.append(item)
            return cleaned
        text_parts = []
        for part in content:
            if part.get("type") == "text":
                text_parts.append(str(part.get("text") or ""))
            elif part.get("type") == "image_url":
                metadata = part.get("metadata") or {}
                source = metadata.get("source", "remote image")
                width = metadata.get("width", "?")
                height = metadata.get("height", "?")
                text_parts.append(f"[Image omitted: {source}, {width}x{height}]")
        return "\n".join(text_parts)

    async def _parse_chunk(self, chunk: dict[str, Any]) -> AsyncIterator[LlmEvent]:
        choices = chunk.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                yield {"type": "thinking_delta", "thinking": reasoning}

            content = delta.get("content")
            if isinstance(content, str) and content:
                yield {"type": "text_delta", "text": content}

            tool_calls = delta.get("tool_calls") or []
            for tool_call in tool_calls:
                yield {"type": "tool_call_delta", "tool_call": tool_call}

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                yield {
                    "type": "message_end",
                    "stop_reason": _map_finish_reason(str(finish_reason)),
                }

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            yield {"type": "usage", "usage": Usage.from_mapping(usage).to_dict()}


def _is_secure_endpoint(base_url: str) -> bool:
    """Allow https, plus http only for loopback hosts used by local proxies."""
    parsed = urlparse(base_url)
    if parsed.scheme == "https":
        return True
    if parsed.scheme != "http":
        return False
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _looks_like_json(payload: str) -> bool:
    return payload.lstrip()[:1] in {"{", "["}


def _stream_error_message(chunk: Any) -> str:
    """Return a human-readable message when a stream chunk carries a provider error."""
    if not isinstance(chunk, dict):
        return ""
    error = chunk.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        code = str(error.get("code") or error.get("type") or "").strip()
        if message and code:
            return f"{message} (code {code})"
        return message or code or json.dumps(error, ensure_ascii=False)[:200]
    if isinstance(error, str) and error.strip():
        return error.strip()
    return ""


def _response_detail(response: httpx.Response, limit: int = 200) -> str:
    try:
        text = response.text
    except httpx.ResponseNotRead:
        return ""
    detail = " ".join(text.split())
    return detail[:limit] if detail else ""


def _is_context_overflow(response: httpx.Response) -> bool:
    if response.status_code not in {400, 413, 422}:
        return False
    text = response.text.lower()
    markers = (
        "context_length_exceeded",
        "context length",
        "context window",
        "maximum context",
        "prompt is too long",
        "too many tokens",
    )
    return any(marker in text for marker in markers)


async def _iter_sse(response: httpx.Response) -> AsyncIterator[str]:
    buffer = ""
    async for text in response.aiter_text():
        buffer += text
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            payload = _sse_payload(event)
            if payload is not None:
                yield payload
    if buffer.strip():
        payload = _sse_payload(buffer)
        if payload is not None:
            yield payload


def _sse_payload(event: str) -> str | None:
    """Join the ``data:`` lines of one SSE event block, or ``None`` when it has none."""
    data_lines = [
        stripped[5:].strip()
        for stripped in (line.strip() for line in event.splitlines())
        if stripped.startswith("data:")
    ]
    return "\n".join(data_lines) if data_lines else None


def _map_finish_reason(reason: str) -> str:
    if reason in {"tool_calls", "tool_use"}:
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    if reason == "content_filter":
        return "stop_sequence"
    return "end_turn"
