"""Translate Agent events into a hierarchical execution trace."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from datetime import UTC, datetime
from typing import cast

from vela.events import AgentEvent, RunStatus, TraceSpanKind, TraceSpanStatus
from vela.run_trace.models import RunTrace, TraceSpan
from vela.run_trace.store import RunTraceStore
from vela.types import Usage


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
        self._span_started: dict[str, float] = {}
        self._plan_spans: dict[str, str] = {}
        self._turn_spans: dict[str, str] = {}
        self._tool_spans: dict[tuple[str, str], str] = {}

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
                yield {"type": "error", "error": error, "run_id": self.trace.run_id}
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
        task_id = str(event.get("task_id") or "")
        if event_type == "plan_status":
            self._observe_plan_status(event)
        elif event_type == "plan_task_started":
            self._start_plan_span(task_id, event)
        elif event_type == "plan_task_done":
            status = "failed" if event.get("task_status") == "failed" else "completed"
            self._close_known_span(self._plan_spans.pop(task_id, None), status)
        elif event_type == "context_compressed":
            self._record_context_span(task_id, event)
        elif event_type == "turn_started":
            self._start_turn_span(task_id, event)
        elif event_type == "model_response_complete":
            self._mark_model_response(task_id, event)
        elif event_type == "turn_complete":
            self.trace.turns = max(self.trace.turns, int(event.get("turn") or 0))
            self._close_known_span(self._turn_spans.pop(task_id, None), "completed")
        elif event_type == "tool_call":
            self.trace.tool_calls += 1
            self._start_tool_span(task_id, event)
        elif event_type == "tool_result":
            self._observe_tool_result(task_id, event)
        elif event_type == "usage":
            self.trace.usage = self.trace.usage + Usage.from_mapping(event.get("usage") or {})
        elif event_type == "done":
            self.trace.turns = max(self.trace.turns, int(event.get("total_turns") or 0))
            done_usage = Usage.from_mapping(event.get("usage") or {})
            if done_usage.total_tokens:
                self.trace.usage = done_usage
        return decorated

    def _observe_plan_status(self, event: AgentEvent) -> None:
        phase = str(event.get("phase") or "")
        if phase == "planning":
            self.trace.status = "planning"
        elif phase == "execution":
            self.trace.status = "running"

    def _start_plan_span(self, task_id: str, event: AgentEvent) -> None:
        if not task_id:
            return
        span = self._start_span(
            kind="plan_node",
            name=task_id,
            parent_span_id=None,
            attributes={"description": str(event.get("task_description") or "")},
        )
        self._plan_spans[task_id] = span.span_id

    def _record_context_span(self, task_id: str, event: AgentEvent) -> None:
        span = self._start_span(
            kind="context",
            name="prepare_context",
            parent_span_id=self._plan_spans.get(task_id),
            attributes={
                "before_tokens": int(event.get("before_tokens") or 0),
                "after_tokens": int(event.get("after_tokens") or 0),
                "summarized_messages": int(event.get("summarized_messages") or 0),
                "truncated_tool_results": int(event.get("truncated_tool_results") or 0),
            },
        )
        self._close_span(span, "completed")

    def _start_turn_span(self, task_id: str, event: AgentEvent) -> None:
        previous = self._turn_spans.pop(task_id, None)
        self._close_known_span(previous, "failed")
        turn = int(event.get("turn") or 0)
        span = self._start_span(
            kind="model_turn",
            name=f"turn_{turn}",
            parent_span_id=self._plan_spans.get(task_id),
            attributes={"turn": turn},
        )
        self._turn_spans[task_id] = span.span_id

    def _mark_model_response(self, task_id: str, event: AgentEvent) -> None:
        span = self._span(self._turn_spans.get(task_id))
        if span is None:
            return
        started = self._span_started.get(span.span_id, time.monotonic())
        span.attributes["model_response_ms"] = max(0, round((time.monotonic() - started) * 1_000))
        span.attributes["stop_reason"] = str(event.get("stop_reason") or "")

    def _start_tool_span(self, task_id: str, event: AgentEvent) -> None:
        call_id = str(event.get("tool_call_id") or "")
        span = self._start_span(
            kind="tool_call",
            name=str(event.get("name") or "unknown"),
            parent_span_id=self._turn_spans.get(task_id),
            attributes={"tool_call_id": call_id},
        )
        self._tool_spans[(task_id, call_id)] = span.span_id

    def _observe_tool_result(self, task_id: str, event: AgentEvent) -> None:
        if event.get("is_error"):
            self.trace.tool_errors += 1
        if event.get("replayed") or event.get("recovery_status") == "replayed":
            self.trace.replayed_tools += 1
        call_id = str(event.get("tool_call_id") or "")
        status: TraceSpanStatus = "failed" if event.get("is_error") else "completed"
        self._close_known_span(self._tool_spans.pop((task_id, call_id), None), status)

    def _start_span(
        self,
        *,
        kind: TraceSpanKind,
        name: str,
        parent_span_id: str | None,
        attributes: dict[str, str | int | float | bool | None],
    ) -> TraceSpan:
        span = TraceSpan(
            span_id=f"span_{uuid.uuid4().hex[:12]}",
            parent_span_id=parent_span_id,
            kind=kind,
            name=name,
            status="running",
            started_at=_now(),
            attributes=attributes,
        )
        self.trace.spans.append(span)
        self._span_started[span.span_id] = time.monotonic()
        return span

    def _span(self, span_id: str | None) -> TraceSpan | None:
        if span_id is None:
            return None
        return next((span for span in self.trace.spans if span.span_id == span_id), None)

    def _close_known_span(self, span_id: str | None, status: TraceSpanStatus) -> None:
        span = self._span(span_id)
        if span is not None:
            self._close_span(span, status)

    def _close_span(self, span: TraceSpan, status: TraceSpanStatus) -> None:
        if span.status != "running":
            return
        span.status = status
        span.finished_at = _now()
        started = self._span_started.pop(span.span_id, time.monotonic())
        span.duration_ms = max(0, round((time.monotonic() - started) * 1_000))

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
        span_status: TraceSpanStatus = (
            "cancelled"
            if status == "cancelled"
            else "failed"
            if status == "failed"
            else "completed"
        )
        for span in self.trace.spans:
            self._close_span(span, span_status)
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
