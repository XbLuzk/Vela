"""Privacy-safe append-only JSONL storage for run traces."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from filelock import FileLock, Timeout

from vela.events import (
    RunStatus,
    RunTracePayload,
    TraceSpanKind,
    TraceSpanPayload,
    TraceSpanStatus,
)
from vela.run_trace.models import RunTrace
from vela.types import Usage


class _TraceScanLimitReached(Exception):
    """Internal signal that a bounded tail scan did not reach file start."""


class RunTraceStore:
    """Append and query completed traces without loading the whole file."""

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
            for raw_line in _read_lines_newest_first(self.path, max_scan_bytes=max_scan_bytes):
                try:
                    value = json.loads(raw_line.decode("utf-8"))
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
            return next(
                (
                    trace
                    for index, trace in enumerate(self._iter_traces(), start=1)
                    if index == target_index
                ),
                None,
            )
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
    required_strings = ("run_id", "status", "mode", "model", "provider", "cwd", "started_at")
    if any(not isinstance(value.get(name), str) or not value[name] for name in required_strings):
        return None
    if value["status"] not in {"planning", "running", "cancelled", "completed", "failed"}:
        return None
    optional_strings = ("session_id", "finished_at", "error")
    if any(
        value.get(name) is not None and not isinstance(value[name], str)
        for name in optional_strings
    ):
        return None
    usage_value = value.get("usage")
    if not isinstance(usage_value, dict):
        return None
    number_names = ("duration_ms", "turns", "tool_calls", "tool_errors", "replayed_tools")
    numbers: dict[str, int] = {}
    for name in number_names:
        raw = value.get(name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            return None
        numbers[name] = raw
    raw_spans = value.get("spans", [])
    if not isinstance(raw_spans, list):
        return None
    spans = [span for item in raw_spans if (span := _normalize_span(item)) is not None]
    return {
        "run_id": str(value["run_id"]),
        "status": cast(RunStatus, value["status"]),
        "mode": str(value["mode"]),
        "model": str(value["model"]),
        "provider": str(value["provider"]),
        "cwd": str(value["cwd"]),
        "session_id": value.get("session_id"),
        "started_at": str(value["started_at"]),
        "finished_at": value.get("finished_at"),
        "duration_ms": numbers["duration_ms"],
        "turns": numbers["turns"],
        "usage": Usage.from_mapping(usage_value).to_dict(),
        "tool_calls": numbers["tool_calls"],
        "tool_errors": numbers["tool_errors"],
        "replayed_tools": numbers["replayed_tools"],
        "error": value.get("error"),
        "spans": spans,
    }


def _normalize_span(value: object) -> TraceSpanPayload | None:
    if not isinstance(value, dict):
        return None
    if value.get("kind") not in {"context", "model_turn", "plan_node", "tool_call"}:
        return None
    if value.get("status") not in {"running", "completed", "failed", "cancelled"}:
        return None
    for name in ("span_id", "name", "started_at"):
        if not isinstance(value.get(name), str) or not value[name]:
            return None
    if value.get("parent_span_id") is not None and not isinstance(value["parent_span_id"], str):
        return None
    if value.get("finished_at") is not None and not isinstance(value["finished_at"], str):
        return None
    duration = value.get("duration_ms")
    attributes = value.get("attributes")
    if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
        return None
    if not isinstance(attributes, dict):
        return None
    return {
        "span_id": str(value["span_id"]),
        "parent_span_id": value.get("parent_span_id"),
        "kind": cast(TraceSpanKind, value["kind"]),
        "name": str(value["name"]),
        "status": cast(TraceSpanStatus, value["status"]),
        "started_at": str(value["started_at"]),
        "finished_at": value.get("finished_at"),
        "duration_ms": duration,
        "attributes": cast(dict[str, str | int | float | bool | None], attributes),
    }
