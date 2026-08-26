from __future__ import annotations

import json

from vela.policy.audit_log import AuditLog
from vela.run_trace.context import bind_run_id, reset_run_id


def test_record_creates_parent_directories_and_appends_jsonl(tmp_path):
    log = AuditLog(tmp_path / "nested" / "audit.jsonl")

    log.record(
        tool_name="write_file",
        input_data={"path": "a.txt"},
        outcome="approved",
        approver="user",
        cwd=str(tmp_path),
    )
    log.record(
        tool_name="bash",
        input_data={"command": "ls"},
        outcome="denied",
        approver="policy",
        cwd=str(tmp_path),
    )

    lines = (tmp_path / "nested" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]

    assert [event["tool_name"] for event in events] == ["write_file", "bash"]
    assert [event["outcome"] for event in events] == ["approved", "denied"]
    assert events[0]["approver"] == "user"
    assert events[0]["cwd"] == str(tmp_path)
    assert events[0]["timestamp"].endswith("+00:00")
    assert events[0]["run_id"] is None


def test_record_expands_user_home_in_the_log_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    log = AuditLog("~/.vela/audit.jsonl")

    log.record(
        tool_name="bash",
        input_data={},
        outcome="approved",
        approver="user",
        cwd=str(tmp_path),
    )

    assert (tmp_path / ".vela" / "audit.jsonl").exists()


def test_record_captures_the_current_run_id(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    token = bind_run_id("run_abc123")
    try:
        log.record(
            tool_name="bash",
            input_data={},
            outcome="approved",
            approver="user",
            cwd=str(tmp_path),
        )
    finally:
        reset_run_id(token)

    assert log.tail()[0]["run_id"] == "run_abc123"


def test_record_redacts_sensitive_values_at_every_nesting_level(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")

    log.record(
        tool_name="mcp__remote__call",
        input_data={
            "API_KEY": "secret-value",
            "headers": {"Authorization": "Bearer abc", "accept": "application/json"},
            "items": [{"password": "hunter2", "name": "keep"}, "plain"],
            "path": "a.txt",
        },
        outcome="approved",
        approver="user",
        cwd=str(tmp_path),
    )

    recorded = log.tail()[0]["input"]

    assert recorded["API_KEY"] == "***"
    assert recorded["headers"]["Authorization"] == "***"
    assert recorded["headers"]["accept"] == "application/json"
    assert recorded["items"][0]["password"] == "***"
    assert recorded["items"][0]["name"] == "keep"
    assert recorded["items"][1] == "plain"
    assert recorded["path"] == "a.txt"


def test_record_keeps_non_ascii_content_readable(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")

    log.record(
        tool_name="write_file",
        input_data={"content": "中文内容"},
        outcome="approved",
        approver="user",
        cwd=str(tmp_path),
    )

    assert "中文内容" in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")


def test_tail_returns_the_most_recent_events_and_skips_corrupt_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)

    assert log.tail() == []

    for index in range(3):
        log.record(
            tool_name=f"tool_{index}",
            input_data={},
            outcome="approved",
            approver="user",
            cwd=str(tmp_path),
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")

    assert [event["tool_name"] for event in log.tail()] == ["tool_0", "tool_1", "tool_2"]
    assert [event["tool_name"] for event in log.tail(limit=2)] == ["tool_2"]
