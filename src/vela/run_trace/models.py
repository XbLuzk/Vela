"""Serializable run and span models."""

from __future__ import annotations

from dataclasses import dataclass, field

from vela.events import (
    RunStatus,
    RunTracePayload,
    TraceSpanKind,
    TraceSpanPayload,
    TraceSpanStatus,
)
from vela.types import Usage


@dataclass(slots=True)
class TraceSpan:
    """One timed operation inside an Agent run."""

    span_id: str
    parent_span_id: str | None
    kind: TraceSpanKind
    name: str
    status: TraceSpanStatus
    started_at: str
    finished_at: str | None = None
    duration_ms: int = 0
    attributes: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def to_dict(self) -> TraceSpanPayload:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "attributes": dict(self.attributes),
        }


@dataclass(slots=True)
class RunTrace:
    """Durable summary and operation tree for one Agent request."""

    run_id: str
    status: RunStatus
    mode: str
    model: str
    provider: str
    cwd: str
    session_id: str | None
    started_at: str
    finished_at: str | None = None
    duration_ms: int = 0
    turns: int = 0
    usage: Usage = field(default_factory=Usage)
    tool_calls: int = 0
    tool_errors: int = 0
    replayed_tools: int = 0
    error: str | None = None
    spans: list[TraceSpan] = field(default_factory=list)

    def to_dict(self) -> RunTracePayload:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "mode": self.mode,
            "model": self.model,
            "provider": self.provider,
            "cwd": self.cwd,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "turns": self.turns,
            "usage": self.usage.to_dict(),
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "replayed_tools": self.replayed_tools,
            "error": self.error,
            "spans": [span.to_dict() for span in self.spans],
        }
