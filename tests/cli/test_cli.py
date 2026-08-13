from __future__ import annotations

import json

from typer.testing import CliRunner

from vela.entrypoints import cli
from vela.run_trace import RunTrace, RunTraceStore
from vela.types import Usage


def test_resume_flag_starts_interactive_repl_in_resume_mode(tmp_path, monkeypatch):
    called = {}

    async def fake_start_repl(cwd, config, *, resume=False):
        called.update(cwd=cwd, config=config, resume=resume)

    monkeypatch.setattr(cli, "start_repl", fake_start_repl)

    result = CliRunner().invoke(cli.app, ["--resume", "--cwd", str(tmp_path)])

    assert result.exit_code == 0
    assert called["cwd"] == str(tmp_path.resolve())
    assert called["resume"] is True


def test_resume_flag_is_rejected_for_single_prompt(tmp_path):
    result = CliRunner().invoke(
        cli.app,
        ["--resume", "--prompt", "hello", "--cwd", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "available only for interactive" in result.output
    assert "sessions" in result.output


def test_team_mode_is_rejected(tmp_path):
    result = CliRunner().invoke(
        cli.app,
        ["--prompt", "hello", "--mode", "team", "--cwd", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "mode must be react or plan" in result.output


def test_trace_command_lists_and_inspects_persisted_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = RunTraceStore()
    store.append(
        RunTrace(
            run_id="run_123456789abc",
            status="completed",
            mode="react",
            model="fake-model",
            provider="fake-provider",
            cwd=str(tmp_path),
            session_id="session_1",
            started_at="2026-08-14T00:00:00+00:00",
            finished_at="2026-08-14T00:00:01+00:00",
            duration_ms=1_000,
            turns=2,
            usage=Usage(input_tokens=20, output_tokens=10),
            tool_calls=1,
        )
    )

    listed = CliRunner().invoke(cli.app, ["trace"])
    inspected = CliRunner().invoke(cli.app, ["trace", "123456789abc", "--json"])

    assert listed.exit_code == 0
    assert "123456789abc" in listed.output
    assert "completed" in listed.output
    assert inspected.exit_code == 0
    assert '"tool_calls": 1' in inspected.output

    inspected = CliRunner().invoke(cli.app, ["trace", "run_1234", "--json"])
    assert inspected.exit_code == 0


def test_trace_json_keeps_warning_on_stderr(tmp_path, monkeypatch):
    class UnreadableStore(RunTraceStore):
        def list(self, *, limit=20):  # noqa: ARG002
            self.last_warning = "Run traces could not be read: denied"
            return []

    store = UnreadableStore(tmp_path / "runs.jsonl")
    monkeypatch.setattr(cli, "RunTraceStore", lambda: store)

    result = CliRunner().invoke(cli.app, ["trace", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == []
    assert "could not be read" in result.stderr
