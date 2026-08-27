from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from vela.config import load_config
from vela.llm.openai_compatible import OpenAICompatibleClient
from vela.mcp.config import load_mcp_server_specs
from vela.tools.file_ops import grep
from vela.types import Message

# Config ----------------------------------------------------------------------


def test_malformed_config_file_is_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    (project / ".vela").mkdir(parents=True)
    (project / ".vela" / "config.json").write_text("{not json", encoding="utf-8")

    warnings: list[str] = []
    config = load_config(project_root=project, env={}, warnings=warnings)

    assert config.llm.provider
    assert any("invalid JSON" in warning for warning in warnings)


def test_unknown_and_invalid_config_values_are_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    (project / ".vela").mkdir(parents=True)
    (project / ".vela" / "config.json").write_text(
        json.dumps({"llm": {"nope": 1}, "policy": "not-an-object"}),
        encoding="utf-8",
    )

    warnings: list[str] = []
    config = load_config(
        project_root=project,
        env={
            "VELA_TEMPERATURE": "warm",
            "VELA_HITL": "sometimes",
            "VELA_MCP": "maybe",
        },
        warnings=warnings,
    )

    assert config.policy.approval_mode != "sometimes"
    joined = " | ".join(warnings)
    assert "unknown llm config keys: nope" in joined
    assert "'policy'" in joined
    assert "VELA_TEMPERATURE='warm'" in joined
    assert "VELA_HITL='sometimes'" in joined
    assert "VELA_MCP='maybe'" in joined


def test_config_warnings_are_optional(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert load_config(project_root=tmp_path, env={}).llm.provider


def test_legacy_hitl_config_migrates_and_new_environment_value_wins(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    (project / ".vela").mkdir(parents=True)
    (project / ".vela" / "config.json").write_text(
        json.dumps({"policy": {"hitl_mode": "never"}}),
        encoding="utf-8",
    )

    warnings: list[str] = []
    config = load_config(
        project_root=project,
        env={"VELA_HITL": "never", "VELA_APPROVAL_MODE": "ask"},
        warnings=warnings,
    )

    assert config.policy.approval_mode == "ask"
    assert config.policy.path_guard_enabled is True
    assert config.policy.command_guard_enabled is True
    assert any("Migrated policy.hitl_mode" in warning for warning in warnings)
    assert any("Migrated legacy VELA_HITL" in warning for warning in warnings)


def test_removed_context_tuning_keys_are_reported_and_ignored(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    (project / ".vela").mkdir(parents=True)
    (project / ".vela" / "config.json").write_text(
        json.dumps(
            {
                "memory": {
                    "max_conversation_history": 42,
                    "compression_threshold": 0.5,
                    "summary_max_chars": 1234,
                }
            }
        ),
        encoding="utf-8",
    )

    warnings: list[str] = []
    config = load_config(project_root=project, env={}, warnings=warnings)

    assert config.memory.max_conversation_history == 42
    assert not hasattr(config.memory, "compression_threshold")
    assert any("unknown memory config keys" in warning for warning in warnings)


# MCP config ------------------------------------------------------------------


def test_malformed_mcp_config_and_entries_are_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    project = tmp_path / "project"
    (project / ".vela").mkdir(parents=True)
    (project / ".vela" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "broken": "not-an-object",
                    "good": {"command": "echo", "args": ["hi"]},
                }
            }
        ),
        encoding="utf-8",
    )

    warnings: list[str] = []
    specs = load_mcp_server_specs(project, warnings=warnings)

    assert set(specs) == {"good"}
    assert any("Ignored MCP server broken" in warning for warning in warnings)


def test_invalid_mcp_json_is_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    project = tmp_path / "project"
    (project / ".vela").mkdir(parents=True)
    (project / ".vela" / "mcp.json").write_text("[]", encoding="utf-8")

    warnings: list[str] = []

    assert load_mcp_server_specs(project, warnings=warnings) == {}
    assert any("expected a JSON object" in warning for warning in warnings)


# File tools ------------------------------------------------------------------


def test_grep_reports_unreadable_files(tmp_path, monkeypatch) -> None:
    (tmp_path / "readable.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "locked.txt").write_text("needle\n", encoding="utf-8")
    original = Path.read_text

    def maybe_fail(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if self.name == "locked.txt":
            raise PermissionError(13, "Permission denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", maybe_fail)

    result = grep(str(tmp_path), "needle")

    assert not result.is_error
    assert "readable.txt:1: needle" in result.content
    assert "skipped 1 unreadable file" in result.content
    assert "locked.txt" in result.content


# Provider streams ------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('data: {"error": {"message": "rate limited", "code": "429"}}\n\n', "rate limited"),
        ('data: {"choices": [broken\n\n', "malformed streaming payload"),
        ("data: [DONE]\n\n", "empty response stream"),
    ],
)
def test_stream_failures_become_error_events(monkeypatch, body, expected) -> None:
    monkeypatch.setattr(httpx.AsyncClient, "stream", _fake_stream(body))

    events = asyncio.run(_collect(_client()))

    assert events[-1]["type"] == "error"
    assert expected in str(events[-1]["error"])


def test_non_json_keep_alive_events_are_tolerated(monkeypatch) -> None:
    chunk = json.dumps({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})
    body = f"data: ping\n\ndata: {chunk}\n\ndata: [DONE]\n\n"
    monkeypatch.setattr(httpx.AsyncClient, "stream", _fake_stream(body))

    events = asyncio.run(_collect(_client()))

    assert [event["type"] for event in events] == ["message_start", "text_delta", "message_end"]


def _fake_stream(body: str):
    request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")

    class _Stream:
        async def __aenter__(self) -> httpx.Response:
            return httpx.Response(200, request=request, content=body.encode("utf-8"))

        async def __aexit__(self, *args) -> None:  # noqa: ANN002
            return None

    def stream(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return _Stream()

    return stream


async def _collect(client: OpenAICompatibleClient) -> list[dict]:
    return [
        event
        async for event in client.chat(
            [Message(role="user", content="hello")],
            [],
            system_prompt="system",
        )
    ]


def _client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        api_key="key",
        base_url="https://api.deepseek.com/v1",
    )
