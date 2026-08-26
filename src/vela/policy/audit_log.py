from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vela.run_trace.context import current_run_id
from vela.storage import PRIVATE_FILE_MODE, ensure_private_file

SENSITIVE_KEYS = ("token", "key", "password", "secret", "authorization", "bearer")
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\b(?:bearer|basic)\s+[\w\-.~+/=]{8,}"),
    re.compile(r"(?i)\b[\w.-]*(?:token|key|password|passwd|secret)[\w.-]*\s*[=:]\s*\S+"),
    re.compile(r"\b(?:sk|pk|rk|gh[pousr]|xox[abposr])[-_][A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{16,}"),
)


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.last_warning: str | None = None

    def record(
        self,
        *,
        tool_name: str,
        input_data: dict[str, Any],
        outcome: str,
        approver: str,
        cwd: str,
    ) -> None:
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": current_run_id(),
            "tool_name": tool_name,
            "input": self._redact(input_data),
            "outcome": outcome,
            "approver": approver,
            "cwd": cwd,
        }
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with self._open_private() as handle:
            handle.write(line)

    def tail(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the newest audit events, reporting entries that could not be read.

        Read failures and corrupt entries are recorded in :attr:`last_warning`
        instead of being dropped, so a truncated audit trail is visible rather
        than looking like an empty one.
        """
        self.last_warning = None
        if limit < 1:
            return []
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        except (OSError, UnicodeDecodeError) as exc:
            self.last_warning = f"Audit log could not be read: {exc}"
            return []
        events = []
        skipped = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
        if skipped:
            suffix = "entry" if skipped == 1 else "entries"
            self.last_warning = f"Skipped {skipped} corrupt audit log {suffix}"
        return events

    def _open_private(self):
        """Append to an owner-only log file inside an owner-only directory."""
        ensure_private_file(self.path)
        descriptor = os.open(
            self.path,
            os.O_CREAT | os.O_WRONLY | os.O_APPEND,
            PRIVATE_FILE_MODE,
        )
        return os.fdopen(descriptor, "a", encoding="utf-8", closefd=True)

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            redacted = {}
            for key, item in value.items():
                if any(marker in key.lower() for marker in SENSITIVE_KEYS):
                    redacted[key] = "***"
                else:
                    redacted[key] = self._redact(item)
            return redacted
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, str):
            return _redact_secret_values(value)
        return value


def _redact_secret_values(value: str) -> str:
    for pattern in _SECRET_VALUE_PATTERNS:
        value = pattern.sub(_mask, value)
    return value


def _mask(match: re.Match[str]) -> str:
    text = match.group(0)
    for separator in ("=", ":", " "):
        prefix, found, _ = text.partition(separator)
        if found:
            return f"{prefix}{separator}***"
    return "***"
