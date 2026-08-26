from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from vela.storage import user_state_path, vela_dir


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    content: str
    source: str = "project"
    tags: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return _strip_frontmatter(self.content).strip()


class SkillMatcher:
    """Rank available skills against a user request using lightweight lexical signals."""

    def __init__(self, skills: Iterable[Skill]):
        self.skills = tuple(skills)

    def match(self, query: str, *, top_k: int = 5) -> list[Skill]:
        if top_k <= 0 or not query.strip():
            return []
        query_terms = _match_terms(query)
        ranked: list[tuple[int, str, Skill]] = []
        for skill in self.skills:
            score = self._score(query, query_terms, skill)
            if score > 0:
                ranked.append((score, skill.name, skill))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [skill for _score, _name, skill in ranked[:top_k]]

    @staticmethod
    def _score(query: str, query_terms: set[str], skill: Skill) -> int:
        score = 0
        if _explicit_skill_name(query, skill.name):
            score += 10_000 + len(_compact_match_text(skill.name))

        name_terms = _match_terms(skill.name)
        tag_terms = _match_terms(" ".join(skill.tags))
        description_terms = _match_terms(skill.description)
        score += 12 * len(query_terms & name_terms)
        score += 6 * len(query_terms & tag_terms)
        score += 2 * len(query_terms & description_terms)
        return score


class SkillContextBuffer:
    def __init__(self, limit: int = 3):
        self.limit = limit
        self._items: OrderedDict[str, str] = OrderedDict()

    def push(self, name: str | None, body: str | None) -> None:
        if not name or not body:
            return
        if name in self._items:
            del self._items[name]
        self._items[name] = body
        while len(self._items) > self.limit:
            self._items.popitem(last=False)

    def drain(self) -> str:
        if not self._items:
            return ""
        chunks = [
            f"## Loaded Skill: {name}\n{body.strip()}"
            for name, body in self._items.items()
            if body.strip()
        ]
        self._items.clear()
        return "\n\n".join(chunks)

    def clear(self) -> None:
        self._items.clear()

    def is_empty(self) -> bool:
        return not self._items


class SkillRegistry:
    """Load SKILL.md files from built-in, user, and project locations."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        builtin_root: str | Path | None = None,
        user_root: str | Path | None = None,
        include_project: bool = True,
    ):
        self.project_root = Path(project_root).resolve()
        package_root = Path(__file__).resolve().parents[1]
        self.builtin_root = Path(builtin_root or package_root / "builtin_skills")
        self.user_root = Path(user_root or user_state_path("skills"))
        self.project_skill_root = vela_dir(self.project_root) / "skills"
        self.include_project = include_project
        self._skills: dict[str, Skill] | None = None

    def list(self) -> list[Skill]:
        skills = self._load_all()
        return [skills[name] for name in sorted(skills)]

    def load(self, name: str) -> Skill | None:
        return self._load_all().get(name)

    def match(self, query: str, *, top_k: int = 5) -> list[Skill]:
        return SkillMatcher(self.list()).match(query, top_k=top_k)

    def _load_all(self) -> dict[str, Skill]:
        if self._skills is not None:
            return self._skills
        skills: dict[str, Skill] = {}
        roots = [
            ("builtin", self.builtin_root),
            ("user", self.user_root),
        ]
        if self.include_project:
            roots.append(("project", self.project_skill_root))
        for source, root in roots:
            if not root.exists():
                continue
            for skill_file in sorted(root.glob("*/SKILL.md")):
                skill = self._load_skill_file(skill_file, source)
                if skill:
                    skills[skill.name] = skill
        self._skills = skills
        return skills

    def _load_skill_file(self, path: Path, source: str) -> Skill | None:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None
        metadata = _parse_frontmatter(content)
        name = metadata.get("name") or path.parent.name
        description = metadata.get("description") or ""
        tags = _parse_tags(metadata.get("tags", ""))
        return Skill(
            name=name,
            description=description,
            tags=tags,
            source=source,
            content=content,
        )


def _parse_frontmatter(content: str) -> dict[str, str]:
    if not content.startswith("---"):
        return {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*", content, re.S)
    if not match:
        return {}
    lines = match.group(1).splitlines()
    metadata: dict[str, str] = {}
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        if ":" not in raw_line:
            index += 1
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "|":
            index += 1
            block: list[str] = []
            while index < len(lines) and (lines[index].startswith(" ") or not lines[index].strip()):
                block.append(lines[index].strip())
                index += 1
            metadata[key] = " ".join(part for part in block if part)
            continue
        metadata[key] = value.strip().strip('"').strip("'")
        index += 1
    return metadata


def _strip_frontmatter(content: str) -> str:
    if not content.startswith("---"):
        return content
    return re.sub(r"^---\s*\n.*?\n---\s*", "", content, count=1, flags=re.S)


def _parse_tags(raw: str) -> list[str]:
    value = raw.strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [item.strip().strip('"').strip("'") for item in value.split(",") if item.strip()]


_ASCII_TERM = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_MATCH_STOPWORDS = {"and", "for", "from", "the", "this", "use", "with"}


def _normalize_match_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", " ", normalized)
    return " ".join(normalized.split())


def _compact_match_text(value: str) -> str:
    return _normalize_match_text(value).replace(" ", "")


def _explicit_skill_name(query: str, name: str) -> bool:
    normalized_query = _normalize_match_text(query)
    normalized_name = _normalize_match_text(name)
    if not normalized_name:
        return False
    tokens = normalized_name.split()
    pattern = r"(?<![a-z0-9])" + r"\s+".join(re.escape(token) for token in tokens)
    pattern += r"(?![a-z0-9])"
    return re.search(pattern, normalized_query) is not None


def _match_terms(value: str) -> set[str]:
    normalized = _normalize_match_text(value)
    terms = {
        term
        for term in _ASCII_TERM.findall(normalized)
        if len(term) >= 2 and term not in _MATCH_STOPWORDS
    }
    for run in _CJK_RUN.findall(normalized):
        if len(run) <= 4:
            terms.add(run)
        for size in (2, 3):
            terms.update(run[index : index + size] for index in range(len(run) - size + 1))
    return terms
