"""Small value objects used by the code index."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class CodeChunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    symbol: str
    content: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    path: str
    start_line: int
    end_line: int
    symbol: str
    content: str
    score: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IndexStats:
    root: str
    database: str
    files: int
    chunks: int
    updated_files: int = 0
    removed_files: int = 0
    retrieval_mode: str = "lexical"

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)
