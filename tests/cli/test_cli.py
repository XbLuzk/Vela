from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from typer.testing import CliRunner

from vela.entrypoints import cli
from vela.prompt import PromptAssembler
from vela.tools import ToolRegistry
from vela.trust import ProjectTrustStore
from vela.types import QueryResult, Usage


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


def test_project_trust_controls_instructions_but_never_changes_llm_config(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    (project / ".vela").mkdir(parents=True)
    (project / ".vela" / "config.json").write_text(
        json.dumps({"llm": {"model": "project-model"}}),
        encoding="utf-8",
    )
    (project / "AGENTS.md").write_text("untrusted instruction", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VELA_API_KEY", "test")
    seen: list[tuple[str, bool, str]] = []

    async def fake_run_prompt(prompt, cwd, config, **kwargs):  # noqa: ARG001
        static = PromptAssembler(config, cwd, [], "model", "provider").build_static()
        seen.append((config.llm.model, config.project_trusted, static))

    monkeypatch.setattr(cli, "_run_prompt", fake_run_prompt)

    denied = CliRunner().invoke(
        cli.app,
        ["--prompt", "hello", "--cwd", str(project)],
    )
    trusted = CliRunner().invoke(
        cli.app,
        ["--trust-project", "--prompt", "hello", "--cwd", str(project)],
    )

    assert denied.exit_code == 0
    assert trusted.exit_code == 0
    assert [(model, trusted) for model, trusted, _ in seen] == [
        ("deepseek-v4-flash", False),
        ("deepseek-v4-flash", True),
    ]
    assert "untrusted instruction" not in seen[0][2]
    assert "untrusted instruction" in seen[1][2]


def test_saved_trust_denial_is_respected_for_plain_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ProjectTrustStore().set(project, False)
    seen: list[bool] = []

    async def fake_start_repl(cwd, config, *, resume=False):  # noqa: ARG001
        seen.append(config.project_trusted)

    monkeypatch.setattr(cli, "start_repl", fake_start_repl)

    result = CliRunner().invoke(cli.app, ["--cwd", str(project)])

    assert result.exit_code == 0
    assert seen == [False]


def test_single_prompt_json_uses_ephemeral_status_without_run_id(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = cli.load_config(env={"VELA_API_KEY": "test"})

    async def fake_registry(**kwargs):  # noqa: ARG001
        return ToolRegistry(), SimpleNamespace(last_errors=[])

    class FakeAgent:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        async def run_complete(self, prompt):  # noqa: ARG002
            return QueryResult(
                text="answer",
                total_tokens=12,
                turns=1,
                usage=Usage(input_tokens=10, output_tokens=2, total_tokens=12),
            )

    monkeypatch.setattr(cli, "build_tool_registry", fake_registry)
    monkeypatch.setattr(cli, "create_llm_client", lambda config: object())
    monkeypatch.setattr(cli, "Agent", FakeAgent)

    asyncio.run(cli._run_prompt("hello", str(tmp_path), config, json_output=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["text"] == "answer"
    assert "run_id" not in payload
