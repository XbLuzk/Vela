from __future__ import annotations

import json
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from vela.branding import PRODUCT_NAME
from vela.events import AgentEvent

RICH_STYLE_RULES = {
    "accent": "bold blue",
    "border": "bright_black",
    "thinking": "bold magenta",
    "tool": "bold yellow",
    "success": "bold green",
    "error": "bold red",
    "logo": "bold blue",
    "identity": "bold",
}


class RichRenderer:
    def __init__(
        self,
        console: Console | None = None,
        *,
        live_markdown: bool = False,
        context_window: int | None = None,
    ):
        self.console = console or Console()
        self._buffer: list[str] = []
        self._thinking_buffer: list[str] = []
        self._thinking_scope: str | None = None
        self._live_markdown = live_markdown
        self._live: Live | None = None
        self._context_window = context_window or 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._last_input_tokens = 0
        self._last_turns = 0
        self._last_total_tokens = 0
        self._last_context_ratio = 0.0
        self._last_has_usage = False
        self._current_run_id = ""

    def set_context_window(self, context_window: int | None) -> None:
        self._context_window = context_window or self._context_window

    def start_run(self) -> None:
        self._buffer.clear()
        self._thinking_buffer.clear()
        self._thinking_scope = None
        self._stop_live_markdown()
        self._input_tokens = 0
        self._output_tokens = 0
        self._last_input_tokens = 0
        self._current_run_id = ""

    def toolbar_status(self) -> dict[str, Any]:
        return {
            "turns": self._last_turns,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self._last_total_tokens,
            "context_ratio": self._last_context_ratio,
            "has_usage": self._last_has_usage,
        }

    def banner(
        self,
        *,
        version: str = "0.5.0",
        api_key_configured: bool = False,
    ) -> None:
        top = Table.grid(expand=True)
        top.add_column(ratio=1)
        top.add_column(ratio=2)
        top.add_row(
            self._identity_panel(version=version, api_key_configured=api_key_configured),
            self._release_panel(version=version),
        )
        self.console.print()
        self.console.print(top)
        self.console.print(Align.right(Text("? for shortcuts", style="dim")))
        self.console.rule(style="grey23")
        self.console.print()

    def handle(self, event: AgentEvent) -> None:
        event_type = event.get("type")
        if event_type in {"text_delta", "thinking_delta", "turn_complete"}:
            self._handle_response_event(event_type, event)
        elif event_type in {"plan_status", "plan_review", "plan_task_started"}:
            self._handle_plan_event(event_type, event)
        elif event_type in {"tool_call", "tool_result"}:
            self._handle_tool_event(event_type, event)
        elif event_type in {
            "run_started",
            "usage",
            "error",
            "done",
            "run_finished",
        }:
            self._handle_run_event(event_type, event)

    def _handle_response_event(self, event_type: str, event: AgentEvent) -> None:
        if event_type == "text_delta":
            self._flush_thinking()
            text = str(event.get("text") or "")
            self._buffer.append(text)
            self._update_live_markdown()
        elif event_type == "thinking_delta":
            scope = _thinking_scope(event)
            if self._thinking_buffer and scope != self._thinking_scope:
                self._flush_thinking()
            self._thinking_scope = scope
            thinking = str(event.get("thinking") or "")
            self._thinking_buffer.append(thinking)
        else:
            stop_reason = str(event.get("stop_reason") or "end_turn")
            title = "Assistant Output" if stop_reason == "tool_use" else "Final Output"
            self._flush_thinking()
            self._flush_markdown(title=title)

    def _handle_plan_event(self, event_type: str, event: AgentEvent) -> None:
        self._flush_thinking()
        self._flush_markdown(title="Plan")
        if event_type == "plan_review":
            self.console.print(
                "[bold blue]确认计划：[/bold blue]输入 [bold]execute[/bold] 执行、"
                "[bold]modify[/bold] 修改或 [bold]cancel[/bold] 取消。"
            )
        elif event_type == "plan_task_started":
            task_id = str(event.get("task_id") or "task")
            description = str(event.get("task_description") or "")
            line = Text.assemble(
                ("◆ ", RICH_STYLE_RULES["accent"]),
                (f"Running {task_id}", RICH_STYLE_RULES["accent"]),
                (" · ", "dim"),
                (_compact_text(description, 500), "dim"),
            )
            self._print_compact_line(line)

    def _handle_tool_event(self, event_type: str, event: AgentEvent) -> None:
        self._flush_thinking()
        self._flush_markdown(title="Assistant Output")
        if event_type == "tool_call":
            self._print_tool_call(event)
        else:
            self._print_tool_result(event)

    def _handle_run_event(self, event_type: str, event: AgentEvent) -> None:
        if event_type == "run_started":
            self._current_run_id = str(event.get("run_id") or "")
        elif event_type == "usage":
            self._record_usage(event.get("usage") or {})
        elif event_type == "error":
            self._flush_thinking()
            self._flush_markdown(title="Assistant Output")
            self.console.print(f"[red]Error:[/red] {event.get('error')}")
        elif event_type == "done":
            self._flush_thinking()
            self._flush_markdown(title="Final Output")
            self._record_run_summary(event)
            warning = str(event.get("warning") or "")
            if warning:
                self.console.print(Text(warning, style="yellow"))
        elif event_type == "run_finished":
            self._print_run_finished(event)

    def newline(self) -> None:
        self._flush_thinking()
        self._flush_markdown(title="Final Output")
        self.console.print()

    def _flush_markdown(self, *, title: str) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer.clear()
        self._stop_live_markdown()
        if text.strip():
            self.console.print(_message_block(text, title=title))

    def _update_live_markdown(self) -> None:
        if not self._live_markdown or not self.console.is_terminal:
            return
        text = "".join(self._buffer)
        if not text.strip():
            return
        renderable = _message_block(text, title="Assistant Output")
        if self._live is None:
            self._live = Live(
                renderable,
                console=self.console,
                refresh_per_second=12,
                transient=True,
                vertical_overflow="visible",
            )
            self._live.start(refresh=True)
            return
        self._live.update(renderable, refresh=True)

    def _stop_live_markdown(self) -> None:
        if self._live is None:
            return
        self._live.stop()
        self._live = None

    def _flush_thinking(self) -> None:
        if not self._thinking_buffer:
            return
        text = "".join(self._thinking_buffer)
        self._thinking_buffer.clear()
        scope = self._thinking_scope
        self._thinking_scope = None
        if text.strip():
            line = Text.assemble(
                ("✦ ", RICH_STYLE_RULES["thinking"]),
                (_thinking_title(scope), RICH_STYLE_RULES["thinking"]),
                (" · ", "dim"),
                (_compact_text(text, 500), "dim"),
            )
            self._print_compact_line(line)

    def _record_usage(self, usage: dict[str, Any]) -> None:
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        if input_tokens:
            self._last_input_tokens = input_tokens

    def _print_tool_call(self, event: AgentEvent) -> None:
        name = str(event.get("name") or "unknown")
        payload = event.get("input") or {}
        line = Text("● ", style=RICH_STYLE_RULES["tool"])
        line.append(_scoped_title(name, event), style=RICH_STYLE_RULES["tool"])
        if payload:
            line.append("  ")
            line.append(_compact_text(_format_payload(payload), 180), style="dim")
        self._print_compact_line(line)

    def _print_tool_result(self, event: AgentEvent) -> None:
        is_error = bool(event.get("is_error"))
        name = str(event.get("name") or "unknown")
        result = _compact_text(str(event.get("result") or "(empty result)"), 240)
        title_style = RICH_STYLE_RULES["error" if is_error else "success"]
        status = "error" if is_error else "ok"
        recovery_status = str(event.get("recovery_status") or "")
        if recovery_status in {"replayed", "reconciled", "uncertain"}:
            status = f"{status} · {recovery_status}"
        line = Text("  └ ", style="dim")
        line.append(_scoped_title(name, event), style=title_style)
        line.append(f" · {status} · ", style=title_style)
        line.append(result, style="dim" if not is_error else "red")
        self._print_compact_line(line)

    def _record_run_summary(self, event: AgentEvent) -> None:
        total_tokens = int(event.get("total_tokens") or self._input_tokens + self._output_tokens)
        turns = int(event.get("total_turns") or 0)
        has_usage = total_tokens > 0 or self._input_tokens > 0 or self._output_tokens > 0
        context_ratio = (
            self._last_input_tokens / self._context_window if self._context_window > 0 else 0
        )
        self._last_turns = turns
        self._last_total_tokens = total_tokens
        self._last_context_ratio = context_ratio
        self._last_has_usage = has_usage

    def _print_run_finished(self, event: AgentEvent) -> None:
        trace = event.get("trace") or {}
        run_id = str(trace.get("run_id") or self._current_run_id)
        status = str(trace.get("status") or "completed")
        duration_ms = int(trace.get("duration_ms") or 0)
        turns = int(trace.get("turns") or 0)
        usage = trace.get("usage") or {}
        tokens = int(usage.get("total_tokens") or 0)
        tools = int(trace.get("tool_calls") or 0)
        label = {
            "completed": "Completed",
            "cancelled": "Cancelled",
            "failed": "Failed",
        }.get(status, status.title())
        details = [
            f"{duration_ms / 1_000:.2f}s",
            _plural_count(turns, "turn"),
            _plural_count(tools, "tool"),
        ]
        if tokens:
            details.append(f"{_compact_number(tokens)} tokens")
        summary = f"✦ {label} in " + " · ".join(details)
        if label != "Completed" and run_id:
            summary += f" · {run_id.removeprefix('run_')}"
        self._print_compact_line(Text(summary, style="dim"))
        warning = str(event.get("warning") or "")
        if warning:
            self.console.print(Text(warning, style="yellow"))

    def _print_compact_line(self, line: Text) -> None:
        line.truncate(max(20, self.console.width - 1), overflow="ellipsis")
        self.console.print(line, no_wrap=True)

    def _identity_panel(self, *, version: str, api_key_configured: bool) -> Table:
        logo = Text("\n".join(_VELA_MARK), style=RICH_STYLE_RULES["logo"])
        identity = Text()
        identity.append(f"{PRODUCT_NAME} ", style=RICH_STYLE_RULES["identity"])
        identity.append(f"v{version}", style="dim")
        identity.append("\n\n")
        if api_key_configured:
            identity.append("Signed in ", style=RICH_STYLE_RULES["identity"])
            identity.append("API Key", style="dim")
        else:
            identity.append("Missing ", style="bold red")
            identity.append("API Key", style="dim")

        grid = Table.grid(padding=(0, 2))
        grid.add_column(no_wrap=True)
        grid.add_column()
        grid.add_row(logo, Align.center(identity, vertical="middle"))
        return grid

    def _release_panel(self, *, version: str) -> Panel:
        notes = Text()
        for line in [
            "A quieter terminal workspace for focused agent runs",
            "ReAct, LangGraph Plan, tools, skills, and MCP",
            "Use /help for commands and /config for settings",
        ]:
            notes.append("- ", style="dim")
            notes.append(line, style="dim")
            notes.append("\n")
        notes.append("/help", style=RICH_STYLE_RULES["accent"])
        notes.append(" for more", style="dim")
        return Panel(
            notes,
            title=Text(f"What's new (v{version})", style=RICH_STYLE_RULES["accent"]),
            border_style=RICH_STYLE_RULES["border"],
            box=box.ROUNDED,
            padding=(0, 2),
        )


_VELA_MARK = (
    "✦       ·",
    " ╲     ╱ ",
    "  ·   ✦  ",
    "   ╲ ╱   ",
    "    ✦    ",
)


def _format_payload(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except TypeError:
        return str(payload)


def _compact_text(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _compact_number(value: int) -> str:
    if value < 1_000:
        return str(value)
    if value < 10_000:
        return f"{value / 1_000:.1f}k"
    return f"{value // 1_000}k"


def _plural_count(value: int, noun: str) -> str:
    suffix = "" if value == 1 else "s"
    return f"{value} {noun}{suffix}"


def _thinking_scope(event: AgentEvent) -> str | None:
    task_id = str(event.get("task_id") or "").strip()
    if task_id:
        return task_id
    if event.get("phase") == "planning":
        return "planning"
    return None


def _thinking_title(scope: str | None) -> str:
    if scope == "planning":
        return "Thinking · planning"
    if scope:
        return f"Thinking · {scope}"
    return "Thinking"


def _scoped_title(title: str, event: AgentEvent) -> str:
    task_id = str(event.get("task_id") or "").strip()
    return f"{title} · {task_id}" if task_id else title


def _message_block(text: str, *, title: str) -> Table:
    body = Table.grid(padding=(0, 1), expand=True)
    body.add_column(no_wrap=True)
    body.add_column(ratio=1)
    prefix = Text("●", style=RICH_STYLE_RULES["accent"])
    if title == "Plan":
        prefix.append(" Plan", style=RICH_STYLE_RULES["accent"])
    body.add_row(prefix, Markdown(text))
    return body
