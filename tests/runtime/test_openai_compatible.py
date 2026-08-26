from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from vela.branding import USER_AGENT
from vela.llm.openai_compatible import (
    OpenAICompatibleClient,
    _is_context_overflow,
    _iter_sse,
    _map_finish_reason,
)
from vela.types import Message


def _client(**overrides) -> OpenAICompatibleClient:
    defaults = {
        "provider_name": "deepseek",
        "model": "deepseek-chat",
        "api_key": "key",
        "base_url": "https://api.deepseek.com/v1/",
    }
    return OpenAICompatibleClient(**{**defaults, **overrides})


class _FakeStream:
    """Minimal stand-in for the async context manager returned by httpx stream()."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.is_error = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):  # noqa: ANN002
        return None

    def raise_for_status(self):
        return None

    async def aiter_text(self):
        for chunk in self._chunks:
            yield chunk


def _chat_events(client: OpenAICompatibleClient, messages=None, tools=None) -> list[dict]:
    async def collect() -> list[dict]:
        return [
            event
            async for event in client.chat(
                messages or [Message(role="user", content="hello")],
                tools or [],
                system_prompt="system",
            )
        ]

    return asyncio.run(collect())


# ---------------------------------------------------------------------------
# Client metadata
# ---------------------------------------------------------------------------


def test_model_name_exposes_the_configured_model():
    assert _client(model="deepseek-v4").model_name == "deepseek-v4"


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("deepseek", "deepseek-vision", True),
        ("deepseek", "image-model", True),
        ("openai-compatible", "qwen2.5-VL", True),
        ("glm", "glm-4.5v", True),
        ("zhipu", "GLM-4.5V", True),
        ("deepseek", "deepseek-chat", False),
        ("glm", "glm-4.6", False),
    ],
)
def test_supports_images_detects_multimodal_models(provider, model, expected):
    assert _client(provider_name=provider, model=model).supports_images is expected


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def test_payload_includes_tools_only_when_tools_are_available():
    client = _client(max_tokens=256, temperature=0.1)
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    without_tools = client._build_payload([], [], system_prompt="system")
    with_tools = client._build_payload([], tools, system_prompt="system")

    assert without_tools["max_tokens"] == 256
    assert without_tools["temperature"] == 0.1
    assert "tools" not in without_tools
    assert with_tools["tools"] == tools
    assert with_tools["tool_choice"] == "auto"


def test_messages_are_formatted_per_role():
    formatted = _client()._format_messages(
        [
            Message(role="user", content="hello"),
            Message(role="assistant", content="", tool_calls=[{"id": "call_1"}]),
            Message(role="assistant", content="done"),
            Message(role="tool", content="result", tool_call_id="call_1"),
            Message(role="tool", content="orphan"),
        ],
        "system",
    )

    assert formatted[0] == {"role": "system", "content": "system"}
    assert formatted[1] == {"role": "user", "content": "hello"}
    assert formatted[2] == {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]}
    assert formatted[3] == {"role": "assistant", "content": "done"}
    assert formatted[4] == {"role": "tool", "tool_call_id": "call_1", "content": "result"}
    assert formatted[5]["tool_call_id"] == ""


def test_multimodal_content_drops_metadata_for_vision_models():
    content = [
        {"type": "text", "text": "look"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
            "metadata": {"source": "page.png", "width": 10, "height": 20},
        },
    ]

    formatted = _client(model="deepseek-vision")._format_content(content)

    assert formatted == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]


def test_multimodal_content_degrades_to_text_for_text_only_models():
    content = [
        {"type": "text", "text": "look"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/a.png"},
            "metadata": {"source": "page.png", "width": 10, "height": 20},
        },
        {"type": "image_url", "image_url": {"url": "https://example.com/b.png"}},
        {"type": "audio", "data": "ignored"},
    ]

    formatted = _client()._format_content(content)

    assert formatted == (
        "look\n[Image omitted: page.png, 10x20]\n[Image omitted: remote image, ?x?]"
    )
    assert _client()._format_content("plain") == "plain"


# ---------------------------------------------------------------------------
# Chunk parsing
# ---------------------------------------------------------------------------


def test_parse_chunk_emits_thinking_text_tool_calls_and_stop_reason():
    chunk = {
        "choices": [
            {
                "delta": {
                    "reasoning_content": "thinking",
                    "content": "answer",
                    "tool_calls": [{"index": 0, "id": "call_1"}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    async def collect():
        return [event async for event in _client()._parse_chunk(chunk)]

    events = asyncio.run(collect())

    assert events == [
        {"type": "thinking_delta", "thinking": "thinking"},
        {"type": "text_delta", "text": "answer"},
        {"type": "tool_call_delta", "tool_call": {"index": 0, "id": "call_1"}},
        {"type": "message_end", "stop_reason": "tool_use"},
    ]


def test_parse_chunk_ignores_empty_and_non_string_deltas():
    chunk = {"choices": [{"delta": {"reasoning_content": "", "content": None}}]}

    async def collect():
        return [event async for event in _client()._parse_chunk(chunk)]

    assert asyncio.run(collect()) == []


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("tool_calls", "tool_use"),
        ("tool_use", "tool_use"),
        ("length", "max_tokens"),
        ("content_filter", "stop_sequence"),
        ("stop", "end_turn"),
        ("unknown", "end_turn"),
    ],
)
def test_finish_reasons_are_mapped_to_vela_stop_reasons(reason, expected):
    assert _map_finish_reason(reason) == expected


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_chat_streams_sse_chunks_until_done(monkeypatch):
    captured: dict[str, object] = {}

    def stream(self, method, url, **kwargs):  # noqa: ARG001
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        return _FakeStream(
            [
                'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
                "data: not-json\n\n",
                'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n',
                'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n',
                "data: [DONE]\n\n",
                'data: {"choices":[{"delta":{"content":"ignored"}}]}\n\n',
            ]
        )

    monkeypatch.setattr(httpx.AsyncClient, "stream", stream)

    events = _chat_events(_client())

    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer key"
    assert captured["headers"]["user-agent"] == USER_AGENT
    assert [event["type"] for event in events] == [
        "message_start",
        "text_delta",
        "text_delta",
        "message_end",
        "usage",
    ]
    assert events[0]["model"] == "deepseek-chat"
    assert events[-1]["usage"]["total_tokens"] == 5


def test_chat_without_api_key_fails_before_any_request(monkeypatch):
    def unexpected_stream(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("no request should be sent without an API key")

    monkeypatch.setattr(httpx.AsyncClient, "stream", unexpected_stream)

    events = _chat_events(_client(api_key=""))

    assert [event["type"] for event in events] == ["error"]
    assert "VELA_API_KEY is not configured" in str(events[0]["error"])


def test_chat_reports_non_context_http_errors_with_provider_guidance(monkeypatch):
    def failing_stream(*args, **kwargs):  # noqa: ARG001
        request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
        response = httpx.Response(401, request=request, json={"error": "invalid key"})
        raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setattr(httpx.AsyncClient, "stream", failing_stream)

    events = _chat_events(_client())

    assert [event["type"] for event in events] == ["message_start", "error"]
    assert "returned HTTP 401" in str(events[-1]["error"])


# ---------------------------------------------------------------------------
# SSE parsing and overflow detection
# ---------------------------------------------------------------------------


def test_iter_sse_joins_multiline_events_and_flushes_the_tail():
    class Response:
        async def aiter_text(self):
            yield 'data: {"a":1}\n'
            yield "\n: comment\ndata: line1\ndata: line2\n\n"
            yield "event: ping\n\n"
            yield "data: tail"

    async def collect():
        return [event async for event in _iter_sse(Response())]

    assert asyncio.run(collect()) == [
        json.dumps({"a": 1}, separators=(",", ":")),
        "line1\nline2",
        "tail",
    ]


def test_iter_sse_ignores_a_tail_without_data_lines():
    class Response:
        async def aiter_text(self):
            yield "event: ping\n"

    async def collect():
        return [event async for event in _iter_sse(Response())]

    assert asyncio.run(collect()) == []


@pytest.mark.parametrize(
    ("status_code", "body", "expected"),
    [
        (400, "context_length_exceeded", True),
        (413, "Prompt is too long", True),
        (422, "maximum context reached", True),
        (400, "invalid api key", False),
        (500, "context length exceeded", False),
    ],
)
def test_context_overflow_detection_requires_status_and_marker(status_code, body, expected):
    request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request, text=body)

    assert _is_context_overflow(response) is expected
