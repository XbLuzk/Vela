"""Run lifecycle tracking and privacy-safe JSONL persistence."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from filelock import FileLock, Timeout

from vela.events import AgentEvent, RunStatus, RunTracePayload
from vela.types import Usage


class _TraceScanLimitReached(Exception):
    """Internal signal that a bounded newest-first scan did not reach file start."""


@dataclass(slots=True)
class RunTrace:
    """Small, durable summary of one ReAct or Plan request."""

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
        }


class RunTraceStore:
    """Append-only local store for completed run summaries."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or Path.home() / ".vela" / "runs.jsonl").expanduser()
        self.last_warning: str | None = None

    def append(self, trace: RunTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = (json.dumps(trace.to_dict(), ensure_ascii=False) + "\n").encode("utf-8")
        try:
            with FileLock(f"{self.path}.lock", timeout=5):
                self._append_locked(payload)
        except Timeout as exc:
            raise OSError("Run trace store is busy") from exc

    def _append_locked(self, payload: bytes) -> None:
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        original_size = 0
        try:
            os.chmod(self.path, 0o600)
            original_size = os.lseek(descriptor, 0, os.SEEK_END)
            separator = b""
            if original_size:
                os.lseek(descriptor, original_size - 1, os.SEEK_SET)
                separator = b"" if os.read(descriptor, 1) == b"\n" else b"\n"
                os.lseek(descriptor, 0, os.SEEK_END)
            try:
                _write_all(descriptor, separator + payload)
            except BaseException:
                try:
                    os.ftruncate(descriptor, original_size)
                except OSError as rollback_error:
                    raise OSError(
                        f"Run trace append failed and rollback failed: {rollback_error}"
                    ) from rollback_error
                raise
        finally:
            os.close(descriptor)

    def list(self, *, limit: int = 20) -> list[RunTracePayload]:
        self.last_warning = None
        if limit < 1:
            return []
        traces: list[RunTracePayload] = []
        for trace in self._iter_traces(max_scan_bytes=2 * 1024 * 1024):
            traces.append(trace)
            if len(traces) >= limit:
                break
        return traces

    def _iter_traces(self, *, max_scan_bytes: int | None = None) -> Iterator[RunTracePayload]:
        if not self.path.exists():
            return
        try:
            lines = _read_lines_newest_first(self.path, max_scan_bytes=max_scan_bytes)
            for raw_line in lines:
                try:
                    line = raw_line.decode("utf-8")
                    value = json.loads(line)
                except (UnicodeError, json.JSONDecodeError):
                    continue
                trace = _normalize_trace(value)
                if trace is not None:
                    yield trace
        except _TraceScanLimitReached:
            self.last_warning = "Run trace scan limit reached; use a Run ID to search older records"
        except OSError as exc:
            self.last_warning = f"Run traces could not be read: {exc}"

    def find(self, reference: str) -> RunTracePayload | None:
        self.last_warning = None
        reference = reference.strip()
        if reference.isdigit() and len(reference) < 12:
            target_index = int(reference)
            if target_index < 1:
                return None
            for index, trace in enumerate(self._iter_traces(), start=1):
                if index == target_index:
                    return trace
            return None
        if reference and not reference.startswith("run_"):
            reference = f"run_{reference}"
        if not reference:
            return None

        full_run_id = len(reference) == len("run_") + 12
        match: RunTracePayload | None = None
        for trace in self._iter_traces():
            run_id = trace["run_id"]
            if full_run_id and run_id == reference:
                return trace
            if not full_run_id and run_id.startswith(reference):
                if match is not None:
                    raise ValueError(f"Ambiguous Run ID prefix: {reference}")
                match = trace
        return match


def _write_all(descriptor: int, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count < 1:
            raise OSError("Run trace append made no progress")
        written += count


def _read_lines_newest_first(
    path: Path,
    *,
    chunk_size: int = 64 * 1024,
    max_scan_bytes: int | None = None,
    max_line_bytes: int = 256 * 1024,
) -> Iterator[bytes]:
    """Yield binary lines from the file tail without loading the full JSONL store."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        scan_start = max(0, position - max_scan_bytes) if max_scan_bytes is not None else 0
        buffer = b""
        discarding_oversized_line = False
        while position > scan_start:
            read_size = min(chunk_size, position - scan_start)
            position -= read_size
            handle.seek(position)
            data = handle.read(read_size)
            if discarding_oversized_line:
                boundary = data.rfind(b"\n")
                if boundary < 0:
                    continue
                data = data[:boundary]
                discarding_oversized_line = False
            parts = (data + buffer).split(b"\n")
            buffer = parts[0]
            for line in reversed(parts[1:]):
                if line and len(line) <= max_line_bytes:
                    yield line
            if len(buffer) > max_line_bytes:
                buffer = b""
                discarding_oversized_line = True
        if scan_start == 0 and buffer and not discarding_oversized_line:
            yield buffer
        elif scan_start > 0:
            raise _TraceScanLimitReached


def _normalize_trace(value: object) -> RunTracePayload | None:
    """Reject schema-corrupt JSON records before they reach CLI renderers."""
    if not isinstance(value, dict):
        return None
    string_fields = ("run_id", "status", "mode", "model", "provider", "cwd", "started_at")
    if any(not isinstance(value.get(field), str) or not value[field] for field in string_fields):
        return None
    if value["status"] not in {"planning", "running", "cancelled", "completed", "failed"}:
        return None
    session_id = value.get("session_id")
    finished_at = value.get("finished_at")
    error = value.get("error")
    if session_id is not None and not isinstance(session_id, str):
        return None
    if finished_at is not None and not isinstance(finished_at, str):
        return None
    if error is not None and not isinstance(error, str):
        return None
    usage_value = value.get("usage")
    if not isinstance(usage_value, dict):
        return None
    number_fields = ("duration_ms", "turns", "tool_calls", "tool_errors", "replayed_tools")
    numbers: dict[str, int] = {}
    for name in number_fields:
        raw = value.get(name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            return None
        numbers[name] = raw
    usage = Usage.from_mapping(usage_value)
    return {
        "run_id": str(value["run_id"]),
        "status": cast(RunStatus, value["status"]),
        "mode": str(value["mode"]),
        "model": str(value["model"]),
        "provider": str(value["provider"]),
        "cwd": str(value["cwd"]),
        "session_id": session_id,
        "started_at": str(value["started_at"]),
        "finished_at": finished_at,
        "duration_ms": numbers["duration_ms"],
        "turns": numbers["turns"],
        "usage": usage.to_dict(),
        "tool_calls": numbers["tool_calls"],
        "tool_errors": numbers["tool_errors"],
        "replayed_tools": numbers["replayed_tools"],
        "error": error,
    }


class RunTracker:
    """Decorate an Agent event stream and settle its trace exactly once."""

    def __init__(
        self,
        *,
        mode: str,
        model: str,
        provider: str,
        cwd: str,
        session_id: str | None = None,
        store: RunTraceStore | None = None,
    ) -> None:
        initial_status: RunStatus = "planning" if mode == "plan" else "running"
        self.trace = RunTrace(
            run_id=f"run_{uuid.uuid4().hex[:12]}",
            status=initial_status,
            mode=mode,
            model=model,
            provider=provider,
            cwd=cwd,
            session_id=session_id,
            started_at=_now(),
        )
        self.store = store
        self.warning: str | None = None
        self._started = time.monotonic()
        self._finished = False

    async def stream(self, events: AsyncIterable[AgentEvent]) -> AsyncIterator[AgentEvent]:
        try:
            yield self._start_event()
            async for event in events:
                decorated = self._observe(event)
                if decorated["type"] in {"done", "error"}:
                    status = self._terminal_status(decorated)
                    error = decorated.get("error")
                    if status == "failed" and error is None:
                        error = RuntimeError("Plan finished with status failed")
                    await self._finish(status, error)
                yield decorated
        except asyncio.CancelledError as exc:
            if exc.__cause__ is not None:
                self._add_warning(f"cancellation cleanup failed: {type(exc.__cause__).__name__}")
            await self._finish("cancelled")
            raise
        except GeneratorExit:
            await self._finish(self.trace.status if self._finished else "cancelled")
            raise
        except BaseException as exc:
            await self._finish("failed", exc)
            raise
        else:
            if not self._finished:
                error = RuntimeError("Agent stream ended without done")
                await self._finish("failed", error)
                yield {
                    "type": "error",
                    "error": error,
                    "run_id": self.trace.run_id,
                }
            yield self._finish_event()
        finally:
            if not self._finished:
                await self._finish("cancelled")

    def _start_event(self) -> AgentEvent:
        return {
            "type": "run_started",
            "run_id": self.trace.run_id,
            "status": self.trace.status,
            "mode": self.trace.mode,
            "model": self.trace.model,
            "provider": self.trace.provider,
            "cwd": self.trace.cwd,
            "session_id": self.trace.session_id,
            "started_at": self.trace.started_at,
        }

    def _observe(self, event: AgentEvent) -> AgentEvent:
        decorated: AgentEvent = {**event, "run_id": self.trace.run_id}
        event_type = event["type"]
        if event_type == "plan_status":
            phase = str(event.get("phase") or "")
            if phase == "planning":
                self.trace.status = "planning"
            elif phase == "execution":
                self.trace.status = "running"
        elif event_type == "turn_complete":
            self.trace.turns = max(self.trace.turns, int(event.get("turn") or 0))
        elif event_type == "tool_call":
            self.trace.tool_calls += 1
        elif event_type == "tool_result":
            if event.get("is_error"):
                self.trace.tool_errors += 1
            if event.get("replayed") or event.get("recovery_status") == "replayed":
                self.trace.replayed_tools += 1
        elif event_type == "usage":
            self.trace.usage = self.trace.usage + Usage.from_mapping(event.get("usage") or {})
        elif event_type == "done":
            self.trace.turns = max(self.trace.turns, int(event.get("total_turns") or 0))
            done_usage = Usage.from_mapping(event.get("usage") or {})
            if done_usage.total_tokens:
                self.trace.usage = done_usage
        return decorated

    def _terminal_status(self, event: AgentEvent) -> RunStatus:
        if event["type"] == "done":
            graph = event.get("langgraph")
            graph_status = str(graph.get("status") or "") if isinstance(graph, dict) else ""
            if graph_status in {"failed", "cancelled"}:
                return cast(RunStatus, graph_status)
            return "completed"
        error = event.get("error")
        return "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"

    async def _finish(self, status: RunStatus, error: object | None = None) -> None:
        if self._finished:
            return
        self.trace.status = status
        self.trace.finished_at = _now()
        self.trace.duration_ms = max(0, round((time.monotonic() - self._started) * 1_000))
        if error is not None:
            self.trace.error = _safe_error(error)
        self._finished = True
        if self.store is not None:
            try:
                await asyncio.to_thread(self.store.append, self.trace)
            except OSError as exc:
                self._add_warning(f"Run trace was not saved: {exc}")

    def _add_warning(self, warning: str) -> None:
        self.warning = "; ".join(value for value in (self.warning, warning) if value)

    def _finish_event(self) -> AgentEvent:
        return trace_finished_event(self.trace, warning=self.warning)


def trace_finished_event(trace: RunTrace, *, warning: str | None = None) -> AgentEvent:
    """Build the public terminal event for an already-settled trace."""
    event: AgentEvent = {
        "type": "run_finished",
        "run_id": trace.run_id,
        "status": trace.status,
        "mode": trace.mode,
        "model": trace.model,
        "provider": trace.provider,
        "cwd": trace.cwd,
        "session_id": trace.session_id,
        "started_at": trace.started_at,
        "finished_at": trace.finished_at,
        "duration_ms": trace.duration_ms,
        "trace": trace.to_dict(),
    }
    if warning:
        event["warning"] = warning
    return event


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _safe_error(error: object) -> str:
    """Persist only the exception class; provider messages may contain credentials."""
    return type(error).__name__
