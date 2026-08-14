"""Pure lexical ranking rules for Memory recall."""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import UTC, datetime

from vela.memory.models import MemoryEntry

_WORD_RE = re.compile(r"[a-z0-9_]+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def parse_timestamp(value: str | datetime) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def relevance_score(query: str, entry: MemoryEntry, now: datetime) -> float:
    normalized_query = normalize_text(query)
    normalized_content = normalize_text(entry.content)
    query_features = _lexical_features(normalized_query)
    content_features = _lexical_features(normalized_content)
    if not query_features:
        return 0.0
    overlap = query_features & content_features
    substring_match = normalized_query in normalized_content
    if not overlap and not substring_match:
        return 0.0
    coverage = len(overlap) / len(query_features)
    lexical = min(1.0, 0.8 * coverage + (0.2 if substring_match else 0.0))
    updated_at = parse_timestamp(entry.updated_at) or parse_timestamp(entry.created_at) or now
    age_days = max((now - updated_at).total_seconds(), 0.0) / 86_400
    recency = 1.0 / (1.0 + age_days / 30.0)
    access = min(math.log1p(max(entry.access_count, 0)) / math.log(11), 1.0)
    return (
        0.72 * lexical
        + 0.12 * entry.importance
        + 0.08 * entry.confidence
        + 0.06 * recency
        + 0.02 * access
    )


def _lexical_features(value: str) -> set[str]:
    normalized = normalize_text(value)
    features = set(_WORD_RE.findall(normalized))
    for sequence in _CJK_RE.findall(normalized):
        characters = list(sequence)
        features.update(characters)
        features.update(
            "".join(characters[index : index + 2]) for index in range(len(characters) - 1)
        )
        if len(sequence) > 1:
            features.add(sequence)
    return features
