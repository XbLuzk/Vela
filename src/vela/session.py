from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vela.session_history import (
    bounded_tool_transcript,
    finalize_interrupted_history,
)
from vela.storage import (
    PRIVATE_FILE_MODE,
    ensure_private_dir,
    set_private_mode,
    user_state_path,
)
from vela.types import Message

__all__ = [
    "ActiveSession",
    "SessionRecord",
    "SessionStore",
    "bounded_tool_transcript",
    "finalize_interrupted_history",
]


@dataclass(slots=True)
class SessionRecord:
    id: str
    cwd: str
    created_at: str
    updated_at: str
    title: str
    message_count: int
    messages: list[Message] = field(default_factory=list)


@dataclass(slots=True)
class ActiveSession:
    store: SessionStore | None
    current: SessionRecord
    resumed: bool = False
    warning: str | None = None

    @classmethod
    def open(
        cls,
        cwd: str | Path,
        *,
        resume: bool = False,
        store: SessionStore | None = None,
    ) -> ActiveSession:
        try:
            session_store = store or SessionStore()
            record = session_store.resolve(cwd) if resume else None
        except Exception as exc:  # noqa: BLE001 - persistence must not disable the app
            return cls(
                None,
                _ephemeral_record(cwd),
                warning=f"Session storage unavailable; continuing in memory: {exc}",
            )
        if record is not None:
            return cls(session_store, record, resumed=True)
        try:
            return cls(session_store, session_store.create(cwd), resumed=False)
        except Exception as exc:  # noqa: BLE001 - persistence must not disable the app
            return cls(
                None,
                _ephemeral_record(cwd),
                warning=f"Session storage unavailable; continuing in memory: {exc}",
            )

    def save(self, messages: list[Message], *, title: str | None = None) -> SessionRecord:
        first_title = title if self.current.message_count == 0 else None
        if self.store is None:
            self.current = _updated_record(self.current, messages, title=first_title)
            return self.current
        try:
            self.current = self.store.save(self.current.id, messages, title=first_title)
        except Exception as exc:  # noqa: BLE001 - retain the in-memory transcript on disk failure
            self.current = _updated_record(self.current, messages, title=first_title)
            self.store = None
            self.warning = f"Could not save session; continuing in memory: {exc}"
        return self.current

    def list(self, *, limit: int = 20) -> list[SessionRecord]:
        if self.store is None:
            return [self.current]
        try:
            return self.store.list(self.current.cwd, limit=limit)
        except Exception as exc:  # noqa: BLE001 - keep the Web app usable
            self.warning = f"Could not list sessions: {exc}"
            return [self.current]

    def new(self) -> SessionRecord:
        """Start a new persisted conversation in the current workspace."""
        if self.store is None:
            self.current = _ephemeral_record(self.current.cwd)
            self.resumed = False
            return self.current
        previous_id = self.current.id
        self.current = self.store.create(self.current.cwd)
        self.resumed = False
        try:
            self.store.delete_empty(previous_id)
        except Exception as exc:  # noqa: BLE001 - the new session is already usable
            self.warning = f"Started a new session, but could not remove empty residue: {exc}"
        return self.current

    def switch(self, reference: str | None = None) -> SessionRecord | None:
        if self.store is None:
            self.warning = "Session storage is unavailable; cannot resume another session."
            return None
        exclude_id = self.current.id if not (reference or "").strip() else None
        try:
            record = self.store.resolve(
                self.current.cwd,
                reference,
                exclude_id=exclude_id,
            )
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the Web app usable
            self.warning = f"Could not resume session: {exc}"
            return None
        if record is None:
            return None
        previous_id = self.current.id
        self.current = record
        self.resumed = True
        if previous_id != record.id:
            try:
                self.store.delete_empty(previous_id)
            except Exception as exc:  # noqa: BLE001 - switching already succeeded
                self.warning = f"Resumed session, but could not remove empty residue: {exc}"
        return record

    def delete(self, reference: str) -> tuple[SessionRecord, SessionRecord] | None:
        """Delete one conversation and keep a usable current session."""
        value = reference.strip()
        if self.store is None:
            if value not in {self.current.id, "1"} and not self.current.id.startswith(value):
                return None
            deleted = self.current
            self.current = _ephemeral_record(self.current.cwd)
            self.resumed = False
            return deleted, self.current

        try:
            record = self.store.resolve(self.current.cwd, value)
            if record is None or not self.store.delete(record.id, cwd=self.current.cwd):
                return None
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the current conversation usable
            self.warning = f"Could not delete session: {exc}"
            return None

        if record.id == self.current.id:
            try:
                replacement = self.store.resolve(self.current.cwd)
                self.current = replacement or self.store.create(self.current.cwd)
                self.resumed = replacement is not None
            except Exception as exc:  # noqa: BLE001 - deletion already succeeded
                self.current = _ephemeral_record(self.current.cwd)
                self.store = None
                self.resumed = False
                self.warning = f"Session deleted, but storage became unavailable: {exc}"
        return record, self.current

    def close(self) -> None:
        if self.store is None:
            return
        try:
            self.store.delete_empty(self.current.id)
        except Exception as exc:  # noqa: BLE001 - shutdown should remain graceful
            self.warning = f"Could not finalize session: {exc}"

    def take_warning(self) -> str | None:
        warning, self.warning = self.warning, None
        return warning


class SessionStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or user_state_path("sessions", "sessions.db"))
        self.db_path = self.db_path.expanduser()
        ensure_private_dir(self.db_path.parent, verify=True)
        self._ensure_schema()
        self._secure_storage()

    def create(self, cwd: str | Path) -> SessionRecord:
        now = datetime.now(UTC).isoformat()
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        session_id = f"session_{timestamp}_{secrets.token_hex(2)}"
        scope = _scope(cwd)
        with self._connect() as connection:
            connection.execute(
                """
                insert into sessions(
                    id, cwd, created_at, updated_at, title, message_count, messages_json
                ) values (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, scope, now, now, "New session", 0, "[]"),
            )
        record = self.get(session_id, cwd=scope)
        if record is None:  # pragma: no cover - SQLite insert/read invariant
            raise RuntimeError(f"Failed to create session {session_id}")
        return record

    def save(
        self,
        session_id: str,
        messages: list[Message],
        *,
        title: str | None = None,
    ) -> SessionRecord:
        existing = self._get_metadata(session_id)
        if existing is None:
            raise KeyError(f"Unknown session: {session_id}")
        now = datetime.now(UTC).isoformat()
        payload = json.dumps([asdict(message) for message in messages], ensure_ascii=False)
        if title:
            session_title = _clean_title(title)
        elif existing.title != "New session":
            session_title = existing.title
        else:
            session_title = _session_title(messages)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                update sessions
                set updated_at = ?, title = ?, message_count = ?, messages_json = ?
                where id = ?
                """,
                (now, session_title, len(messages), payload, session_id),
            )
        if cursor.rowcount == 0:  # pragma: no cover - guarded by the lookup above
            raise KeyError(f"Unknown session: {session_id}")
        return SessionRecord(
            id=existing.id,
            cwd=existing.cwd,
            created_at=existing.created_at,
            updated_at=now,
            title=session_title,
            message_count=len(messages),
            messages=list(messages),
        )

    def get(self, session_id: str, *, cwd: str | Path | None = None) -> SessionRecord | None:
        parameters: list[Any] = [session_id]
        scope_filter = ""
        if cwd is not None:
            scope_filter = " and cwd = ?"
            parameters.append(_scope(cwd))
        with self._connect() as connection:
            row = connection.execute(
                f"""
                select id, cwd, created_at, updated_at, title, message_count, messages_json
                from sessions
                where id = ?{scope_filter}
                """,
                parameters,
            ).fetchone()
        return _record(row) if row else None

    def list(self, cwd: str | Path, *, limit: int = 20) -> list[SessionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select id, cwd, created_at, updated_at, title, message_count
                from sessions
                where cwd = ?
                order by updated_at desc, created_at desc
                limit ?
                """,
                (_scope(cwd), max(1, limit)),
            ).fetchall()
        return [_metadata_record(row) for row in rows]

    def resolve(
        self,
        cwd: str | Path,
        reference: str | None = None,
        *,
        exclude_id: str | None = None,
    ) -> SessionRecord | None:
        sessions = [item for item in self.list(cwd, limit=20) if item.id != exclude_id]
        value = (reference or "").strip()
        if not value:
            selected = next((item for item in sessions if item.message_count > 0), None)
            return self.get(selected.id, cwd=cwd) if selected else None
        if value.isdigit():
            index = int(value) - 1
            selected = sessions[index] if 0 <= index < len(sessions) else None
            return self.get(selected.id, cwd=cwd) if selected else None
        exact = self.get(value, cwd=cwd)
        if exact:
            return exact
        matches = [item for item in sessions if item.id.startswith(value)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous session reference: {value}")
        return self.get(matches[0].id, cwd=cwd) if matches else None

    def delete_empty(self, session_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "delete from sessions where id = ? and message_count = 0",
                (session_id,),
            )
        return cursor.rowcount > 0

    def delete(self, session_id: str, *, cwd: str | Path) -> bool:
        """Delete a session only when it belongs to the requested workspace."""
        with self._connect() as connection:
            cursor = connection.execute(
                "delete from sessions where id = ? and cwd = ?",
                (session_id, _scope(cwd)),
            )
        return cursor.rowcount > 0

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                create table if not exists sessions (
                    id text primary key,
                    cwd text not null,
                    created_at text not null,
                    updated_at text not null,
                    title text not null,
                    message_count integer not null,
                    messages_json text not null
                )
                """
            )
            connection.execute(
                """
                create index if not exists idx_sessions_cwd_updated
                on sessions(cwd, updated_at desc)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=0.25)
        connection.execute("pragma busy_timeout = 250")
        connection.row_factory = sqlite3.Row
        self._secure_storage()
        return connection

    def _get_metadata(self, session_id: str) -> SessionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                select id, cwd, created_at, updated_at, title, message_count
                from sessions where id = ?
                """,
                (session_id,),
            ).fetchone()
        return _metadata_record(row) if row else None

    def _secure_storage(self) -> None:
        for path in (
            self.db_path,
            Path(f"{self.db_path}-journal"),
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        ):
            if path.exists():
                set_private_mode(path, PRIVATE_FILE_MODE, verify=True)


def _record(row: sqlite3.Row) -> SessionRecord:
    raw_messages = json.loads(row["messages_json"])
    record = _metadata_record(row)
    record.messages = [_message(item) for item in raw_messages]
    return record


def _metadata_record(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        id=str(row["id"]),
        cwd=str(row["cwd"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        title=str(row["title"]),
        message_count=int(row["message_count"]),
    )


def _message(value: dict[str, Any]) -> Message:
    return Message(
        role=value["role"],
        content=value.get("content", ""),
        name=value.get("name"),
        tool_call_id=value.get("tool_call_id"),
        tool_calls=list(value.get("tool_calls") or []),
    )


def _session_title(messages: list[Message]) -> str:
    first_user = next((message for message in messages if message.role == "user"), None)
    if first_user is None:
        return "New session"
    if isinstance(first_user.content, str):
        text = first_user.content
    else:
        text = next(
            (
                str(item.get("text") or "")
                for item in first_user.content
                if item.get("type") == "text"
            ),
            "Image message",
        )
    return _clean_title(text)


def _clean_title(value: str) -> str:
    compact = " ".join(value.split())
    return compact[:60] or "Untitled session"


def _scope(cwd: str | Path) -> str:
    return str(Path(cwd).expanduser().resolve())


def _ephemeral_record(cwd: str | Path) -> SessionRecord:
    now = datetime.now(UTC).isoformat()
    return SessionRecord(
        id=f"memory_{secrets.token_hex(4)}",
        cwd=_scope(cwd),
        created_at=now,
        updated_at=now,
        title="New session",
        message_count=0,
    )


def _updated_record(
    record: SessionRecord,
    messages: list[Message],
    *,
    title: str | None,
) -> SessionRecord:
    session_title = (
        _clean_title(title)
        if title
        else record.title
        if record.title != "New session"
        else _session_title(messages)
    )
    return SessionRecord(
        id=record.id,
        cwd=record.cwd,
        created_at=record.created_at,
        updated_at=datetime.now(UTC).isoformat(),
        title=session_title,
        message_count=len(messages),
        messages=list(messages),
    )
