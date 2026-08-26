from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vela.memory.models import MemoryEntry
from vela.memory.ranking import normalize_text, parse_timestamp, relevance_score

DEFAULT_MAX_ENTRIES_PER_SCOPE = 1_000
DEFAULT_MAX_CONTENT_LENGTH = 8_000
_MAX_METADATA_LENGTH = 128
_ENTRY_COLUMNS = """
    id, scope, content, created_at, kind, source, importance, confidence,
    updated_at, expires_at, access_count, content_hash
"""
_REQUIRED_COLUMNS = {
    "id",
    "scope",
    "content",
    "created_at",
    "kind",
    "source",
    "importance",
    "confidence",
    "updated_at",
    "expires_at",
    "access_count",
    "content_hash",
}


class MemoryManager:
    def __init__(
        self,
        db_path: str | Path,
        scope: str,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES_PER_SCOPE,
        max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH,
    ):
        self.db_path = Path(db_path).expanduser()
        self.scope = str(scope)
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.max_content_length = _positive_int(max_content_length, "max_content_length")
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.db_path.parent, 0o700)
        self._ensure_schema()
        self._secure_storage()

    def save(
        self,
        content: str,
        *,
        kind: str = "fact",
        source: str = "manual",
        importance: float = 0.5,
        confidence: float = 1.0,
        expires_at: str | datetime | None = None,
    ) -> int:
        cleaned = _clean_content(content)
        if len(cleaned) > self.max_content_length:
            raise ValueError(f"memory content exceeds {self.max_content_length} characters")
        normalized_kind = _validate_metadata(kind, "kind")
        normalized_source = _validate_metadata(source, "source")
        normalized_importance = _validate_score(importance, "importance")
        normalized_confidence = _validate_score(confidence, "confidence")
        normalized_expiry = _normalize_optional_timestamp(expires_at)
        now = _now_iso()
        content_hash = _hash_content(cleaned)

        with self._connect() as conn:
            conn.execute("begin immediate")
            self._purge_expired(conn)
            row = conn.execute(
                f"""
                select {_ENTRY_COLUMNS}
                from memories
                where scope = ? and content_hash = ?
                """,
                (self.scope, content_hash),
            ).fetchone()
            if row:
                existing = _entry_from_row(row)
                memory_id = existing.id
                conn.execute(
                    """
                    update memories
                    set content = ?, kind = ?, source = ?, importance = ?, confidence = ?,
                        updated_at = ?, expires_at = ?
                    where id = ? and scope = ?
                    """,
                    (
                        cleaned,
                        normalized_kind,
                        normalized_source,
                        max(existing.importance, normalized_importance),
                        max(existing.confidence, normalized_confidence),
                        now,
                        normalized_expiry if normalized_expiry is not None else existing.expires_at,
                        memory_id,
                        self.scope,
                    ),
                )
            else:
                cursor = conn.execute(
                    """
                    insert into memories(
                        scope, content, created_at, kind, source, importance, confidence,
                        updated_at, expires_at, access_count, content_hash
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        self.scope,
                        cleaned,
                        now,
                        normalized_kind,
                        normalized_source,
                        normalized_importance,
                        normalized_confidence,
                        now,
                        normalized_expiry,
                        content_hash,
                    ),
                )
                memory_id = int(cursor.lastrowid)
            self._enforce_quota(conn, protected_id=memory_id)
            return memory_id

    def list(self, limit: int = 20, *, kind: str | None = None) -> list[MemoryEntry]:
        normalized_limit = _normalize_limit(limit)
        if normalized_limit == 0:
            return []
        params: list[Any] = [self.scope, _now_iso()]
        kind_clause = ""
        if kind is not None:
            kind_clause = " and kind = ?"
            params.append(_validate_metadata(kind, "kind"))
        params.append(normalized_limit)
        with self._connect() as conn:
            self._purge_expired(conn)
            rows = conn.execute(
                f"""
                select {_ENTRY_COLUMNS}
                from memories
                where scope = ?
                  and (expires_at is null or expires_at > ?)
                  {kind_clause}
                order by updated_at desc, id desc
                limit ?
                """,
                params,
            ).fetchall()
        return [_entry_from_row(row) for row in rows]

    def recall(
        self,
        query: str,
        limit: int = 10,
        *,
        kinds: Iterable[str] | None = None,
        min_score: float = 0.0,
        mark_access: bool = True,
    ) -> list[MemoryEntry]:
        normalized_limit = _normalize_limit(limit)
        if normalized_limit == 0:
            return []
        cleaned_query = " ".join(str(query or "").split())
        if not cleaned_query:
            return self.list(normalized_limit)

        normalized_kinds = None
        if kinds is not None:
            normalized_kinds = {_validate_metadata(kind, "kind") for kind in kinds}
            if not normalized_kinds:
                return []

        params: list[Any] = [self.scope, _now_iso()]
        kinds_clause = ""
        if normalized_kinds:
            placeholders = ", ".join("?" for _ in normalized_kinds)
            kinds_clause = f" and kind in ({placeholders})"
            params.extend(sorted(normalized_kinds))

        with self._connect() as conn:
            self._purge_expired(conn)
            rows = conn.execute(
                f"""
                select {_ENTRY_COLUMNS}
                from memories
                where scope = ?
                  and (expires_at is null or expires_at > ?)
                  {kinds_clause}
                """,
                params,
            ).fetchall()

        now = datetime.now(UTC)
        ranked: list[tuple[float, MemoryEntry]] = []
        for row in rows:
            entry = _entry_from_row(row)
            score = relevance_score(cleaned_query, entry, now)
            if score > 0 and score >= float(min_score):
                ranked.append((score, entry))
        ranked.sort(
            key=lambda item: (
                item[0],
                item[1].importance,
                item[1].confidence,
                item[1].updated_at,
                item[1].id,
            ),
            reverse=True,
        )
        selected = [entry for _score, entry in ranked[:normalized_limit]]
        if mark_access and selected:
            self.mark_access(entry.id for entry in selected)
            for entry in selected:
                entry.access_count += 1
        return selected

    def mark_access(self, memory_ids: int | Iterable[int]) -> int:
        ids = [memory_ids] if isinstance(memory_ids, int) else list(memory_ids)
        normalized_ids = list(dict.fromkeys(int(memory_id) for memory_id in ids))
        if not normalized_ids:
            return 0

        changed = 0
        with self._connect() as conn:
            for start in range(0, len(normalized_ids), 500):
                chunk = normalized_ids[start : start + 500]
                placeholders = ", ".join("?" for _ in chunk)
                cursor = conn.execute(
                    f"""
                    update memories
                    set access_count = access_count + 1
                    where scope = ? and id in ({placeholders})
                    """,
                    [self.scope, *chunk],
                )
                changed += int(cursor.rowcount)
        return changed

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            self._purge_expired(conn)
            row = conn.execute(
                """
                select count(*) as total,
                       coalesce(sum(access_count), 0) as total_access_count,
                       coalesce(avg(importance), 0) as average_importance,
                       coalesce(avg(confidence), 0) as average_confidence,
                       min(created_at) as oldest_created_at,
                       max(updated_at) as newest_updated_at
                from memories
                where scope = ?
                """,
                (self.scope,),
            ).fetchone()
            kind_rows = conn.execute(
                """
                select kind, count(*) as count
                from memories
                where scope = ?
                group by kind
                order by kind
                """,
                (self.scope,),
            ).fetchall()
        return {
            "scope": self.scope,
            "total": int(row["total"]),
            "by_kind": {str(item["kind"]): int(item["count"]) for item in kind_rows},
            "total_access_count": int(row["total_access_count"]),
            "average_importance": float(row["average_importance"]),
            "average_confidence": float(row["average_confidence"]),
            "oldest_created_at": row["oldest_created_at"],
            "newest_updated_at": row["newest_updated_at"],
            "max_entries": self.max_entries,
        }

    def delete(self, memory_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "delete from memories where scope = ? and id = ?",
                (self.scope, int(memory_id)),
            )
            return bool(cursor.rowcount)

    def clear(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("delete from memories where scope = ?", (self.scope,))
            return int(cursor.rowcount)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists memories (
                    id integer primary key autoincrement,
                    scope text not null,
                    content text not null,
                    created_at text not null,
                    kind text not null default 'fact',
                    source text not null default 'manual',
                    importance real not null default 0.5,
                    confidence real not null default 1.0,
                    updated_at text not null default '',
                    expires_at text,
                    access_count integer not null default 0,
                    content_hash text not null default ''
                )
                """
            )
            columns = {
                str(row["name"]) for row in conn.execute("pragma table_info(memories)").fetchall()
            }
            missing = sorted(_REQUIRED_COLUMNS - columns)
            if missing:
                names = ", ".join(missing)
                raise RuntimeError(
                    f"Unsupported memory database schema at {self.db_path}; "
                    f"missing columns: {names}. "
                    "Move or delete the old database before starting Vela."
                )

            conn.execute("create index if not exists idx_memories_scope on memories(scope, id)")
            conn.execute(
                """
                create unique index if not exists idx_memories_scope_hash
                on memories(scope, content_hash)
                """
            )
            conn.execute(
                """
                create index if not exists idx_memories_scope_updated
                on memories(scope, updated_at desc, id desc)
                """
            )
            conn.execute(
                """
                create index if not exists idx_memories_scope_kind
                on memories(scope, kind)
                """
            )
            self._enforce_quota(conn)

    def _purge_expired(self, conn: sqlite3.Connection) -> int:
        cursor = conn.execute(
            """
            delete from memories
            where scope = ? and expires_at is not null and expires_at <= ?
            """,
            (self.scope, _now_iso()),
        )
        return int(cursor.rowcount)

    def _enforce_quota(
        self,
        conn: sqlite3.Connection,
        *,
        protected_id: int | None = None,
    ) -> int:
        self._purge_expired(conn)
        total = int(
            conn.execute("select count(*) from memories where scope = ?", (self.scope,)).fetchone()[
                0
            ]
        )
        excess = total - self.max_entries
        if excess <= 0:
            return 0
        protected_clause = ""
        params: list[Any] = [self.scope]
        if protected_id is not None:
            protected_clause = " and id != ?"
            params.append(protected_id)
        params.append(excess)
        cursor = conn.execute(
            f"""
            delete from memories
            where id in (
                select id from memories
                where scope = ? {protected_clause}
                order by importance asc, confidence asc, access_count asc,
                         updated_at asc, id asc
                limit ?
            )
            """,
            params,
        )
        return int(cursor.rowcount)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 30000")
        self._secure_storage()
        return conn

    def _secure_storage(self) -> None:
        """Keep long-term memory and its SQLite sidecars readable only by the owner."""
        for path in (
            self.db_path,
            Path(f"{self.db_path}-journal"),
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        ):
            if path.exists():
                os.chmod(path, 0o600)


def _entry_from_row(row: sqlite3.Row) -> MemoryEntry:
    return MemoryEntry(
        id=int(row["id"]),
        scope=str(row["scope"]),
        content=str(row["content"]),
        created_at=str(row["created_at"]),
        kind=str(row["kind"]),
        source=str(row["source"]),
        importance=float(row["importance"]),
        confidence=float(row["confidence"]),
        updated_at=str(row["updated_at"]),
        expires_at=str(row["expires_at"]) if row["expires_at"] is not None else None,
        access_count=int(row["access_count"]),
        content_hash=str(row["content_hash"]),
    )


def _clean_content(content: str) -> str:
    if not isinstance(content, str):
        raise TypeError("memory content must be a string")
    cleaned = content.strip()
    if not cleaned:
        raise ValueError("memory content must not be empty")
    return cleaned


def _hash_content(content: str) -> str:
    normalized = normalize_text(content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validate_metadata(value: str, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"memory {field} must not be empty")
    if len(normalized) > _MAX_METADATA_LENGTH:
        raise ValueError(f"memory {field} exceeds {_MAX_METADATA_LENGTH} characters")
    return normalized


def _validate_score(value: float, field: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"memory {field} must be a number between 0 and 1") from exc
    if not 0 <= normalized <= 1:
        raise ValueError(f"memory {field} must be between 0 and 1")
    return normalized


def _positive_int(value: int, field: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if normalized <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return normalized


def _normalize_limit(value: int) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer") from exc


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_optional_timestamp(value: str | datetime | None) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    parsed = parse_timestamp(value)
    if parsed is None:
        raise ValueError("expires_at must be a valid ISO-8601 timestamp")
    return parsed.isoformat()
