"""Data objects stored and recalled by Vela Memory."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MemoryEntry:
    id: int
    scope: str
    content: str
    created_at: str
    kind: str = "fact"
    source: str = "manual"
    importance: float = 0.5
    confidence: float = 1.0
    updated_at: str = ""
    expires_at: str | None = None
    access_count: int = 0
    content_hash: str = ""
