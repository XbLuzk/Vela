from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vela.run_trace.context import current_run_id

SENSITIVE_KEYS = ("token", "key", "password", "secret", "authorization", "bearer")


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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": current_run_id(),
            "tool_name": tool_name,
            "input": self._redact(input_data),
            "outcome": outcome,
            "approver": approver,
            "cwd": cwd,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def tail(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the newest audit events, reporting entries that could not be read.

        Read failures and corrupt entries are recorded in :attr:`last_warning`
        instead of being dropped, so a truncated audit trail is visible rather
        than looking like an empty one.
        """
        self.last_warning = None
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
        return value
