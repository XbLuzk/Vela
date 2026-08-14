from __future__ import annotations

import asyncio

import httpx
import pytest

from vela.agent import Agent
from vela.config import VelaConfig
from vela.context import ContextOverflowError
from vela.llm.openai_compatible import OpenAICompatibleClient
from vela.tools import ToolRegistry
from vela.types import Message, Usage


def test_streaming_payload_requests_usage() -> None:
    client = _client()

    payload = client._build_payload(
        [Message(role="user", content="hello")],
        [],
        system_prompt="system",
    )

    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_connection_error_becomes_recoverable_error_event(monkeypatch) -> None:
    def fail_stream(*args, **kwargs):
        request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
        raise httpx.ConnectError("temporary connection failure", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fail_stream)

    events = asyncio.run(_collect_chat_events(_client()))

    assert [event["type"] for event in events] == ["message_start", "error"]
    assert "Could not connect to deepseek" in str(events[-1]["error"])
    assert "VPN/proxy" in str(events[-1]["error"])


def test_timeout_becomes_recoverable_error_event(monkeypatch) -> None:
    def fail_stream(*args, **kwargs):
        request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
        raise httpx.ReadTimeout("temporary timeout", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fail_stream)

    events = asyncio.run(_collect_chat_events(_client()))

    assert [event["type"] for event in events] == ["message_start", "error"]
    assert "timed out after 120s" in str(events[-1]["error"])


def test_provider_context_limit_is_classified_for_agent_recovery(monkeypatch) -> None:
    def fail_stream(*args, **kwargs):
        request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
        response = httpx.Response(
            400,
            request=request,
            json={"error": {"code": "context_length_exceeded", "message": "too many tokens"}},
        )
        raise httpx.HTTPStatusError("context overflow", request=request, response=response)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fail_stream)

    events = asyncio.run(_collect_chat_events(_client()))

    assert isinstance(events[-1]["error"], ContextOverflowError)


@pytest.mark.parametrize("status_code", [400, 413])
def test_unread_streaming_context_limit_body_is_classified(monkeypatch, status_code) -> None:
    request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
    response = httpx.Response(
        status_code,
        request=request,
        stream=httpx.ByteStream(b'{"error":{"message":"context length exceeded"}}'),
    )

    class StreamingResponse:
        async def __aenter__(self):
            return response

        async def __aexit__(self, *args):  # noqa: ANN002
            return None

    def stream(*args, **kwargs):  # noqa: ARG001
        return StreamingResponse()

    monkeypatch.setattr(httpx.AsyncClient, "stream", stream)

    events = asyncio.run(_collect_chat_events(_client()))

    assert isinstance(events[-1]["error"], ContextOverflowError)


def test_agent_propagates_connection_error_without_raising(tmp_path, monkeypatch) -> None:
    def fail_stream(*args, **kwargs):
        request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
        raise httpx.ConnectError("temporary connection failure", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fail_stream)
    config = VelaConfig()
    config.llm.api_key = "key"
    config.memory.long_term_db_path = str(tmp_path / "memory.db")
    agent = Agent(
        llm_client=_client(),
        tool_registry=ToolRegistry(),
        system_prompt="system",
        cwd=str(tmp_path),
        config=config,
    )

    events = asyncio.run(_collect_agent_events(agent))

    assert [event["type"] for event in events] == [
        "run_started",
        "turn_started",
        "error",
        "run_finished",
    ]
    assert "Could not connect to deepseek" in str(events[2]["error"])
    assert events[-1]["status"] == "failed"


def test_usage_only_chunk_without_choices_is_parsed() -> None:
    chunk = {
        "choices": [],
        "usage": {
            "prompt_tokens": 1_000,
            "completion_tokens": 200,
            "prompt_cache_hit_tokens": 600,
            "prompt_cache_miss_tokens": 400,
            "completion_tokens_details": {"reasoning_tokens": 50},
            "total_tokens": 1_200,
        },
    }

    events = asyncio.run(_collect_events(_client(), chunk))

    assert events == [
        {
            "type": "usage",
            "usage": {
                "input_tokens": 1_000,
                "output_tokens": 200,
                "cache_hit_tokens": 600,
                "cache_miss_tokens": 400,
                "reasoning_tokens": 50,
                "total_tokens": 1_200,
            },
        }
    ]


def test_usage_normalizes_old_and_provider_fields() -> None:
    legacy = Usage.from_mapping({"input_tokens": 10, "output_tokens": 5})
    provider = Usage.from_mapping(
        {
            "prompt_cache_hit_tokens": 7,
            "prompt_cache_miss_tokens": 3,
            "completion_tokens": 4,
            "reasoning_tokens": 2,
        }
    )

    assert legacy.total_tokens == 15
    assert legacy.to_dict()["input_tokens"] == 10
    assert provider.input_tokens == 10
    assert provider.output_tokens == 4
    assert provider.reasoning_tokens == 2
    assert provider.total_tokens == 14


async def _collect_events(
    client: OpenAICompatibleClient,
    chunk: dict,
) -> list[dict]:
    return [event async for event in client._parse_chunk(chunk)]


async def _collect_chat_events(client: OpenAICompatibleClient) -> list[dict]:
    return [
        event
        async for event in client.chat(
            [Message(role="user", content="hello")],
            [],
            system_prompt="system",
        )
    ]


async def _collect_agent_events(agent: Agent) -> list[dict]:
    return [event async for event in agent.run("hello")]


def _client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        api_key="key",
        base_url="https://api.deepseek.com/v1",
    )
