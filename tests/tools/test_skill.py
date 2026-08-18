from __future__ import annotations

import asyncio
import json
from pathlib import Path

from vela.config import load_config
from vela.skill import SkillContextBuffer, SkillMatcher, SkillRegistry
from vela.tools.base import ToolContext
from vela.tools.builtins import _load_skill


def test_skill_registry_layers_are_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    project = tmp_path / "project"
    _write_skill(builtin, "web-access", "builtin desc", "v0")
    _write_skill(user, "web-access", "user desc", "v1")
    _write_skill(project / ".vela" / "skills", "project-only", "project desc", "v2")

    registry = SkillRegistry(project, builtin_root=builtin, user_root=user)

    assert [skill.name for skill in registry.list()] == ["project-only", "web-access"]
    assert registry.load("web-access").description == "user desc"
    assert registry.load("web-access").source == "user"


def test_untrusted_skill_registry_ignores_project_skills(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    project = tmp_path / "project"
    _write_skill(user, "user-skill", "user desc", "v1")
    _write_skill(project / ".vela" / "skills", "project-skill", "project desc", "v1")

    registry = SkillRegistry(
        project,
        builtin_root=builtin,
        user_root=user,
        include_project=False,
    )

    assert [skill.name for skill in registry.list()] == ["user-skill"]


def test_skill_context_buffer_is_one_shot_and_capped():
    buffer = SkillContextBuffer(limit=3)
    buffer.push("a", "A")
    buffer.push("b", "B")
    buffer.push("c", "C")
    buffer.push("d", "D")

    drained = buffer.drain()

    assert "Loaded Skill: a" not in drained
    assert "Loaded Skill: b" in drained
    assert "Loaded Skill: c" in drained
    assert "Loaded Skill: d" in drained
    assert buffer.drain() == ""


def test_load_skill_pushes_body_into_context_buffer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".vela" / "skills", "demo", "demo desc", "v1", body="demo body")
    config = load_config(project_root=tmp_path)
    buffer = SkillContextBuffer()
    context = ToolContext(cwd=str(tmp_path), config=config, skill_context_buffer=buffer)

    result = asyncio.run(_load_skill({"name": "demo"}, context))

    assert not result.is_error
    drained = buffer.drain()
    assert "Loaded Skill: demo" in drained
    assert "demo body" in drained


def test_load_skill_is_idempotent_within_one_tool_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".vela" / "skills", "demo", "demo desc", "v1", body="demo body")
    config = load_config(project_root=tmp_path)
    buffer = SkillContextBuffer()
    context = ToolContext(cwd=str(tmp_path), config=config, skill_context_buffer=buffer)

    first = asyncio.run(_load_skill({"name": "demo"}, context))
    first_context = buffer.drain()
    second = asyncio.run(_load_skill({"name": "demo"}, context))

    assert not first.is_error
    assert "Loaded Skill: demo" in first_context
    assert not second.is_error
    assert second.content == 'Skill "demo" is already loaded for this run.'
    assert buffer.drain() == ""


def test_load_skill_can_be_loaded_again_in_a_new_tool_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".vela" / "skills", "demo", "demo desc", "v1", body="demo body")
    config = load_config(project_root=tmp_path)
    first_buffer = SkillContextBuffer()
    second_buffer = SkillContextBuffer()

    first = asyncio.run(
        _load_skill(
            {"name": "demo"},
            ToolContext(cwd=str(tmp_path), config=config, skill_context_buffer=first_buffer),
        )
    )
    second = asyncio.run(
        _load_skill(
            {"name": "demo"},
            ToolContext(cwd=str(tmp_path), config=config, skill_context_buffer=second_buffer),
        )
    )

    assert not first.is_error
    assert not second.is_error
    assert "Loaded Skill: demo" in first_buffer.drain()
    assert "Loaded Skill: demo" in second_buffer.drain()


def test_skill_matcher_ranks_explicit_names_and_chinese_english_terms(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    user = tmp_path / "user"
    builtin = tmp_path / "builtin"
    _write_skill(
        project / ".vela" / "skills",
        "pdf-ocr",
        "从扫描图片和 PDF 文档中提取文字",
        "v1",
        tags=["ocr", "文档识别"],
    )
    _write_skill(
        user,
        "web-access",
        "Live web research and webpage fetching",
        "v1",
        tags=["web", "research"],
    )
    registry = SkillRegistry(project, builtin_root=builtin, user_root=user)

    assert registry.match("请从扫描图片里提取文字", top_k=1)[0].name == "pdf-ocr"
    assert registry.match("research the latest webpage", top_k=1)[0].name == "web-access"
    explicit = registry.match("请用 pdf-ocr，再 research the latest webpage", top_k=2)
    assert explicit[0].name == "pdf-ocr"

    matcher = SkillMatcher(registry.list())
    assert len(matcher.match("web research", top_k=1)) == 1


def _write_skill(
    root: Path,
    name: str,
    desc: str,
    version: str,
    *,
    body: str | None = None,
    tags: list[str] | None = None,
) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dir.joinpath("SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\nversion: {version}\n"
        f"tags: {json.dumps(tags or [], ensure_ascii=False)}\n---\n"
        f"{body or f'body for {name}'}\n",
        encoding="utf-8",
    )
