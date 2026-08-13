"""Shared CLI and REPL presentation for persisted Run traces."""

from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from vela.run_trace import RunTraceStore


def show_run_traces(
    console: Console,
    store: RunTraceStore,
    *,
    reference: str = "",
    limit: int = 20,
    json_output: bool = False,
) -> bool:
    """Render one trace or a newest-first trace list; return whether data was found."""
    if reference:
        try:
            trace = store.find(reference)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return False
        if trace is None:
            _print_store_warning(console, store)
            console.print(f"[red]Run trace not found:[/red] {reference}")
            return False
        _print_store_warning(console, store)
        console.print_json(json.dumps(trace, ensure_ascii=False))
        return True

    traces = store.list(limit=limit)
    _print_store_warning(console, store)
    if json_output:
        console.print_json(json.dumps(traces, ensure_ascii=False))
        return bool(traces)
    if not traces:
        console.print("(no run traces)")
        return False

    table = Table(title="Vela Runs")
    table.add_column("#", justify="right")
    table.add_column("Run")
    table.add_column("Status")
    table.add_column("Mode")
    table.add_column("Turns", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Tools", justify="right")
    table.add_column("Duration", justify="right")
    for index, trace in enumerate(traces, start=1):
        usage = trace.get("usage") or {}
        table.add_row(
            str(index),
            str(trace.get("run_id") or "").removeprefix("run_"),
            str(trace.get("status") or ""),
            str(trace.get("mode") or ""),
            str(trace.get("turns") or 0),
            str(usage.get("total_tokens") or 0),
            str(trace.get("tool_calls") or 0),
            _duration(int(trace.get("duration_ms") or 0)),
        )
    console.print(table)
    return True


def parse_trace_args(raw: str) -> tuple[str, bool]:
    parts = raw.split()
    json_output = "--json" in parts
    reference = next((part for part in parts if part != "--json"), "")
    return reference, json_output


def _duration(milliseconds: int) -> str:
    if milliseconds < 1_000:
        return f"{milliseconds}ms"
    return f"{milliseconds / 1_000:.2f}s"


def _print_store_warning(console: Console, store: RunTraceStore) -> None:
    if store.last_warning:
        Console(stderr=True).print(f"[yellow]{store.last_warning}[/yellow]")
