from __future__ import annotations

from typer.testing import CliRunner

from vela.entrypoints import cli


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
