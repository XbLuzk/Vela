"""Persistent trust decisions for project-local executable configuration."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from vela.storage import (
    ensure_private_dir,
    exclusive_lock,
    user_state_path,
    vela_dir,
    write_private_text,
)

_TRUST_SENSITIVE_FILES = (
    Path(".env"),
    Path(".vela/config.json"),
    Path(".vela/mcp.json"),
)
DEFAULT_PROJECT_INSTRUCTION_PATHS = (
    Path("AGENTS.md"),
    Path(".vela/AGENTS.md"),
    Path("PAI.md"),
    Path(".vela/PAI.md"),
    Path("PAI.local.md"),
    Path(".vela/PAI.local.md"),
)


class ProjectTrustStore:
    """Store exact, resolved project paths in a private user-level JSON file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or user_state_path("trust.json")).expanduser()

    def get(self, project_root: str | Path) -> bool | None:
        projects = self._read().get("projects", {})
        value = projects.get(_project_key(project_root)) if isinstance(projects, dict) else None
        return value if isinstance(value, bool) else None

    def set(self, project_root: str | Path, trusted: bool) -> None:
        ensure_private_dir(self.path.parent)
        with exclusive_lock(self.path, busy_message="Project trust store is busy"):
            data = self._read()
            projects = data.get("projects")
            if not isinstance(projects, dict):
                projects = {}
                data["projects"] = projects
            projects[_project_key(project_root)] = trusted
            self._write(data)

    def _read(self) -> dict[str, object]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"projects": {}}
        return value if isinstance(value, dict) else {"projects": {}}

    def _write(self, value: dict[str, object]) -> None:
        write_private_text(
            self.path,
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        )


def resolve_project_trust(
    project_root: str | Path,
    *,
    interactive: bool,
    override: bool | None = None,
    store: ProjectTrustStore | None = None,
    prompt: Callable[[Path], bool] | None = None,
) -> bool:
    """Resolve one-run override, saved decision, prompt, then fail-closed default."""
    if override is not None:
        return override
    trust_store = store or ProjectTrustStore()
    saved = trust_store.get(project_root)
    if saved is not None:
        return saved
    if not interactive or prompt is None:
        return False
    root = Path(project_root).resolve()
    trusted = bool(prompt(root))
    trust_store.set(root, trusted)
    return trusted


def has_trust_sensitive_resources(project_root: str | Path) -> bool:
    """Return whether this project can change Vela configuration or capabilities."""
    root = Path(project_root).expanduser().resolve()
    sensitive_files = _TRUST_SENSITIVE_FILES + DEFAULT_PROJECT_INSTRUCTION_PATHS
    if any((root / relative).is_file() for relative in sensitive_files):
        return True
    skills = vela_dir(root) / "skills"
    return skills.is_dir() and any(skills.glob("*/SKILL.md"))


def _project_key(project_root: str | Path) -> str:
    return str(Path(project_root).expanduser().resolve())
