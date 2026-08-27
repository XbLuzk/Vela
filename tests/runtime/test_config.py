from __future__ import annotations

import json

from vela.config import config_to_public_dict, load_config


def test_config_precedence(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".vela").mkdir(parents=True)
    (project / ".vela").mkdir(parents=True)
    (home / ".vela" / "config.json").write_text(
        json.dumps({"llm": {"provider": "home", "model": "home-model"}}),
        encoding="utf-8",
    )
    (project / ".vela" / "config.json").write_text(
        json.dumps({"llm": {"provider": "project", "model": "project-model"}}),
        encoding="utf-8",
    )
    (project / ".env").write_text("VELA_MODEL=env-file-model\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("VELA_PROVIDER", "process")
    monkeypatch.setenv("VELA_MODEL", "process-model")

    config = load_config(
        project_root=project,
        overrides={"llm": {"model": "cli-model"}},
    )

    assert config.llm.provider == "process"
    assert config.llm.model == "process-model"


def test_provider_specific_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VELA_PROVIDER", "deepseek")
    monkeypatch.delenv("VELA_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    config = load_config(project_root=tmp_path)

    assert config.llm.api_key == "deepseek-key"


def test_untrusted_config_ignores_project_json_and_dotenv(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".vela").mkdir(parents=True)
    (project / ".vela").mkdir(parents=True)
    (home / ".vela" / "config.json").write_text(
        json.dumps({"llm": {"model": "user-model"}}),
        encoding="utf-8",
    )
    (project / ".vela" / "config.json").write_text(
        json.dumps({"llm": {"model": "project-model"}}),
        encoding="utf-8",
    )
    (project / ".env").write_text("VELA_MODEL=dotenv-model\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    config = load_config(project_root=project, include_project=False)

    assert config.llm.model == "user-model"
    assert not config.project_trusted


def test_runtime_project_trust_is_not_exposed_as_persisted_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path, include_project=False)

    assert "project_trusted" not in config_to_public_dict(config)
