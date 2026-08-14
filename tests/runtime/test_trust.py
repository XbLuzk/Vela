from __future__ import annotations

import stat

from vela.trust import (
    ProjectTrustStore,
    has_trust_sensitive_resources,
    resolve_project_trust,
)


def test_trust_store_uses_resolved_exact_paths_and_private_permissions(tmp_path) -> None:
    path = tmp_path / "home" / ".vela" / "trust.json"
    project = tmp_path / "project"
    project.mkdir()
    store = ProjectTrustStore(path)

    store.set(project / ".", True)

    assert store.get(project) is True
    assert store.get(tmp_path / "other") is None
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_trust_resolution_is_fail_closed_noninteractive_and_persists_prompt(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = ProjectTrustStore(tmp_path / "trust.json")

    assert not resolve_project_trust(project, interactive=False, store=store)
    assert resolve_project_trust(
        project,
        interactive=True,
        store=store,
        prompt=lambda root: root == project,
    )
    assert resolve_project_trust(project, interactive=False, store=store)
    assert not resolve_project_trust(
        project,
        interactive=True,
        override=False,
        store=store,
        prompt=lambda _root: True,
    )


def test_sensitive_resource_detection_covers_config_env_mcp_instructions_and_skills(
    tmp_path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    assert not has_trust_sensitive_resources(project)

    (project / ".env").write_text("VELA_MODEL=test\n", encoding="utf-8")
    assert has_trust_sensitive_resources(project)
    (project / ".env").unlink()

    (project / "AGENTS.md").write_text("project rules", encoding="utf-8")
    assert has_trust_sensitive_resources(project)
    (project / "AGENTS.md").unlink()

    skill = project / ".vela" / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("demo", encoding="utf-8")
    assert has_trust_sensitive_resources(project)
