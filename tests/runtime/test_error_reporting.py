from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from vela.config import load_config
from vela.llm.openai_compatible import OpenAICompatibleClient
from vela.mcp.config import load_mcp_server_specs
from vela.policy.audit_log import AuditLog
from vela.run_trace.models import RunTrace
from vela.run_trace.store import RunTraceStore
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

    assert config.policy.hitl_mode != "sometimes"
    joined = " | ".join(warnings)
    assert "unknown llm config keys: nope" in joined
    assert "'policy'" in joined
    assert "VELA_TEMPERATURE='warm'" in joined
    assert "VELA_HITL='sometimes'" in joined
    assert "VELA_MCP='maybe'" in joined


def test_config_warnings_are_optional(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert load_config(project_root=tmp_path, env={}).llm.provider


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


# Audit log -------------------------------------------------------------------


def test_audit_tail_reports_corrupt_entries(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text('{"tool": "read"}\nnot json\n\n', encoding="utf-8")
    log = AuditLog(path)

    events = log.tail()

    assert events == [{"tool": "read"}]
    assert log.last_warning == "Skipped 1 corrupt audit log entry"


def test_audit_tail_reports_read_failure(tmp_path, monkeypatch) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    log = AuditLog(path)

    def fail_read(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", fail_read)

    assert log.tail() == []
    assert "Audit log could not be read" in str(log.last_warning)


def test_audit_tail_warning_resets_between_calls(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    log = AuditLog(path)
    log.tail()
    path.write_text('{"tool": "read"}\n', encoding="utf-8")

    assert log.tail() == [{"tool": "read"}]
    assert log.last_warning is None


# Run traces ------------------------------------------------------------------


def test_run_trace_list_reports_corrupt_records(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    store.append(
        RunTrace(
            run_id="run_1",
            status="completed",
            mode="react",
            model="m",
            provider="p",
            cwd=".",
            session_id=None,
            started_at="2026-01-01T00:00:00Z",
        )
    )
    with store.path.open("ab") as handle:
        handle.write(b"not json\n")
        handle.write(b'{"run_id":"run_corrupt","usage":"bad"}\n')

    traces = store.list(limit=10)

    assert [item["run_id"] for item in traces] == ["run_1"]
    assert store.last_warning == "Skipped 2 corrupt run trace records"


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
