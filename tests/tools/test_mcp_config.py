from __future__ import annotations

import json

from vela.mcp.config import load_mcp_server_specs, write_chrome_devtools_config


def test_project_config_overrides_user_config_and_can_be_excluded(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".vela").mkdir(parents=True)
    (home / ".vela" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "user-only": {"command": "user-cmd"},
                    "shared": {"command": "user-shared"},
                }
            }
        ),
        encoding="utf-8",
    )
    project = tmp_path / "project"
    (project / ".vela").mkdir(parents=True)
    (project / ".vela" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"shared": {"command": "project-shared"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    merged = load_mcp_server_specs(project)
    user_only = load_mcp_server_specs(project, include_project=False)

    assert set(merged) == {"user-only", "shared"}
    assert merged["shared"].command == "project-shared"
    assert user_only["shared"].command == "user-shared"


def test_specs_accept_a_bare_server_mapping_and_skip_invalid_entries(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".vela").mkdir(parents=True)
    (home / ".vela" / "mcp.json").write_text(
        json.dumps({"good": {"command": "cmd"}, "bad": "not-a-mapping"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    specs = load_mcp_server_specs(tmp_path / "project")

    assert set(specs) == {"good"}


def test_missing_and_malformed_config_files_are_ignored(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".vela").mkdir(parents=True)
    (home / ".vela" / "mcp.json").write_text("{not json", encoding="utf-8")
    project = tmp_path / "project"
    (project / ".vela").mkdir(parents=True)
    (project / ".vela" / "mcp.json").write_text(json.dumps(["a", "list"]), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    assert load_mcp_server_specs(project) == {}
    assert load_mcp_server_specs(tmp_path / "empty") == {}


def test_stdio_spec_defaults_and_placeholder_expansion(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / ".vela").mkdir()
    (project / ".vela" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {
                        "command": "${PROJECT_DIR}/bin/server",
                        "args": ["--root", "${PROJECT_DIR}", "--home", "${HOME}"],
                        "env": {"TOKEN": "${SECRET_TOKEN}", "MISSING": "${NOT_SET}"},
                        "cwd": "${PROJECT_DIR}/work",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SECRET_TOKEN", "token-value")
    monkeypatch.delenv("NOT_SET", raising=False)

    spec = load_mcp_server_specs(project)["local"]

    assert spec.type == "stdio"
    assert spec.enabled is True
    assert spec.timeout == 30.0
    assert spec.command == f"{project.resolve()}/bin/server"
    assert spec.args == ["--root", str(project.resolve()), "--home", str(home)]
    assert spec.env == {"TOKEN": "token-value", "MISSING": ""}
    assert spec.cwd == f"{project.resolve()}/work"
    assert spec.url is None


def test_url_servers_default_to_streamable_http_and_read_transport_aliases(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".vela").mkdir(parents=True)
    (home / ".vela" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {
                        "url": "https://example.com/mcp",
                        "headers": {"Authorization": "Bearer ${SECRET_TOKEN}"},
                        "enabled": False,
                        "startup_timeout": 12,
                    },
                    "aliased": {"transport": "sse", "url": "https://example.com/sse"},
                    "zero-timeout": {"url": "https://example.com/mcp", "timeout": 0},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SECRET_TOKEN", "token-value")

    specs = load_mcp_server_specs(tmp_path / "project")

    assert specs["remote"].type == "streamable_http"
    assert specs["remote"].headers == {"Authorization": "Bearer token-value"}
    assert specs["remote"].enabled is False
    assert specs["remote"].timeout == 12.0
    assert specs["aliased"].type == "sse"
    assert specs["zero-timeout"].timeout == 30.0


def test_write_chrome_devtools_config_defaults_to_the_user_scope(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    path = write_chrome_devtools_config()

    assert path == home / ".vela" / "mcp.json"
    entry = json.loads(path.read_text(encoding="utf-8"))["mcpServers"]["chrome-devtools"]
    assert entry["type"] == "stdio"
    assert entry["command"] == "npx"
    assert entry["args"] == ["-y", "chrome-devtools-mcp@latest", "--no-usage-statistics"]


def test_write_chrome_devtools_config_appends_selected_flags(tmp_path):
    path = write_chrome_devtools_config(
        scope_root=tmp_path,
        browser_url="http://127.0.0.1:9222",
        headless=True,
        slim=True,
        no_usage_statistics=False,
    )

    entry = json.loads(path.read_text(encoding="utf-8"))["mcpServers"]["chrome-devtools"]

    assert path == tmp_path / ".vela" / "mcp.json"
    assert entry["args"] == [
        "-y",
        "chrome-devtools-mcp@latest",
        "--slim",
        "--headless",
        "--browser-url=http://127.0.0.1:9222",
    ]


def test_write_chrome_devtools_config_preserves_other_servers(tmp_path):
    config_dir = tmp_path / ".vela"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({"mcpServers": {"other": {"command": "keep-me"}}}),
        encoding="utf-8",
    )

    path = write_chrome_devtools_config(scope_root=tmp_path)
    servers = json.loads(path.read_text(encoding="utf-8"))["mcpServers"]

    assert servers["other"] == {"command": "keep-me"}
    assert "chrome-devtools" in servers


def test_write_chrome_devtools_config_replaces_a_malformed_file(tmp_path):
    config_dir = tmp_path / ".vela"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text("{not json", encoding="utf-8")

    path = write_chrome_devtools_config(scope_root=tmp_path)

    assert "chrome-devtools" in json.loads(path.read_text(encoding="utf-8"))["mcpServers"]
