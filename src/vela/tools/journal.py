from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from vela.storage import apply_sqlite_pragmas, ensure_private_file
from vela.tools.base import ToolResult

JournalStatus = Literal["running", "completed", "uncertain"]
ClaimAction = Literal["execute", "replay", "uncertain"]


@dataclass(frozen=True, slots=True)
class JournalRecord:
    execution_key: str
    status: JournalStatus
    tool_name: str
    result: ToolResult | None
    attempts: int


@dataclass(frozen=True, slots=True)
class JournalClaim:
    action: ClaimAction
    record: JournalRecord


class ToolExecutionJournal:
    """Durable tool-boundary journal for resumable, effectively-once execution."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._initialize()

    def get(self, execution_key: str) -> JournalRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT execution_key, status, tool_name, result_content,
                       result_is_error, result_summary, attempts
                  FROM tool_executions
                 WHERE execution_key = ?
                """,
                (execution_key,),
            ).fetchone()
        return _record_from_row(row)

    def claim(
        self,
        *,
        execution_key: str,
        scope: str,
        sequence: int,
        tool_name: str,
        input_hash: str,
        allow_uncertain_retry: bool,
    ) -> JournalClaim:
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT execution_key, status, tool_name, result_content,
                       result_is_error, result_summary, attempts
                  FROM tool_executions
                 WHERE execution_key = ?
                """,
                (execution_key,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO tool_executions (
                        execution_key, scope, sequence, tool_name, input_hash,
                        status, attempts, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'running', 1, ?, ?)
                    """,
                    (execution_key, scope, sequence, tool_name, input_hash, now, now),
                )
                connection.commit()
                return JournalClaim(
                    "execute",
                    JournalRecord(execution_key, "running", tool_name, None, 1),
                )

            record = _record_from_row(row)
            if record is None:
                raise RuntimeError("tool journal returned an invalid record")
            if record.status == "completed":
                connection.commit()
                return JournalClaim("replay", record)
            if not allow_uncertain_retry:
                if record.status == "running":
                    connection.execute(
                        """
                        UPDATE tool_executions
                           SET status = 'uncertain', updated_at = ?
                         WHERE execution_key = ?
                        """,
                        (now, execution_key),
                    )
                    connection.commit()
                    record = JournalRecord(
                        record.execution_key,
                        "uncertain",
                        record.tool_name,
                        record.result,
                        record.attempts,
                    )
                else:
                    connection.commit()
                return JournalClaim("uncertain", record)

            attempts = record.attempts + 1
            connection.execute(
                """
                UPDATE tool_executions
                   SET status = 'running', attempts = ?, updated_at = ?
                 WHERE execution_key = ?
                """,
                (attempts, now, execution_key),
            )
            connection.commit()
            return JournalClaim(
                "execute",
                JournalRecord(execution_key, "running", tool_name, None, attempts),
            )

    def complete(self, execution_key: str, result: ToolResult) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE tool_executions
                   SET status = 'completed', result_content = ?, result_is_error = ?,
                       result_summary = ?, updated_at = ?
                 WHERE execution_key = ?
                """,
                (
                    result.content,
                    int(result.is_error),
                    result.display_summary,
                    _now(),
                    execution_key,
                ),
            )

    def mark_uncertain(self, execution_key: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE tool_executions
                   SET status = 'uncertain', updated_at = ?
                 WHERE execution_key = ? AND status = 'running'
                """,
                (_now(), execution_key),
            )

    def delete_scope_prefix(self, prefix: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM tool_executions
                 WHERE substr(scope, 1, length(?)) = ?
                """,
                (prefix, prefix),
            )
        return int(cursor.rowcount)

    def _initialize(self) -> None:
        ensure_private_file(self.path)
        with self._connect() as connection:
            apply_sqlite_pragmas(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_executions (
                    execution_key TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'uncertain')),
                    result_content TEXT,
                    result_is_error INTEGER,
                    result_summary TEXT,
                    attempts INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tool_executions_scope
                    ON tool_executions(scope, sequence)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)


def execution_identity(
    scope: str,
    sequence: int,
    tool_name: str,
    payload: dict[str, Any],
) -> tuple[str, str]:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    input_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    raw = f"{scope}\0{sequence}\0{tool_name}\0{input_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), input_hash


def _record_from_row(row: tuple[Any, ...] | None) -> JournalRecord | None:
    if row is None:
        return None
    content = row[3]
    result = None
    if str(row[1]) == "completed":
        result = ToolResult(
            content=str(content or ""),
            is_error=bool(row[4]),
            display_summary=str(row[5]) if row[5] is not None else None,
        )
    return JournalRecord(
        execution_key=str(row[0]),
        status=str(row[1]),  # type: ignore[arg-type]
        tool_name=str(row[2]),
        result=result,
        attempts=int(row[6]),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
