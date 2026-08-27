"""Prompt input, shortcuts, and status-line rendering for the interactive REPL."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Condition, is_done, to_filter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    FloatContainer,
    HSplit,
    VerticalAlign,
    VSplit,
    Window,
)
from rich.align import Align
from rich.console import Console
from rich.text import Text

from vela.config import VelaConfig
from vela.image import ClipboardImageResult, grab_clipboard_image
from vela.task_control import InteractiveTaskController, TaskState

REPL_STYLE_RULES = {
    "prompt": "ansibrightblack",
    "placeholder": "italic ansibrightblack",
    "input.rule": "ansibrightblack",
    "prompt.dim": "ansibrightblack",
    "toolbar.model": "bold",
    "toolbar.ctx.bar": "ansigreen",
    "toolbar.ctx.value": "",
    "toolbar.mode.ask": "bold ansigreen",
    "toolbar.mode.auto": "bold ansiyellow",
    "toolbar.task": "bold ansimagenta",
    "toolbar.gap": "",
    "bottom-toolbar": "noreverse",
    "bottom-toolbar.text": "noreverse",
}

ApprovalMode = Literal["ask", "auto"]


@dataclass
class ApprovalModeController:
    """Toggle tool approval prompts without changing safety guards."""

    config: VelaConfig
    mode: ApprovalMode = field(init=False)

    def __post_init__(self) -> None:
        configured = self.config.policy.approval_mode
        self.set(cast(ApprovalMode, configured if configured in {"ask", "auto"} else "ask"))

    def set(self, mode: ApprovalMode) -> ApprovalMode:
        self.mode = mode
        self.config.policy.approval_mode = mode
        return self.mode

    def toggle(self) -> ApprovalMode:
        return self.set("auto" if self.mode == "ask" else "ask")


class FixedComposerPromptSession(PromptSession):
    """Prompt session pinned above an ephemeral bottom status bar."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("reserve_space_for_menu", 0)
        kwargs.setdefault("erase_when_done", True)
        super().__init__(*args, **kwargs)

    def _create_layout(self):
        layout = super()._create_layout()
        root = layout.container
        if not isinstance(root, HSplit):
            raise RuntimeError("Unsupported Prompt Toolkit layout")
        status_section = root.children[-1]
        status_section.filter = Condition(lambda: self.bottom_toolbar is not None) & ~is_done
        main_section = root.children[0]
        if not isinstance(main_section, ConditionalContainer):
            raise RuntimeError("Unsupported Prompt Toolkit input section")
        main_input = main_section.alternative_content
        if not isinstance(main_input, FloatContainer):
            raise RuntimeError("Unsupported Prompt Toolkit input container")
        input_stack = main_input.content
        if not isinstance(input_stack, HSplit):
            raise RuntimeError("Unsupported Prompt Toolkit input stack")
        input_stack.align = VerticalAlign.BOTTOM
        for child in input_stack.children:
            if isinstance(child, ConditionalContainer) and isinstance(child.content, Window):
                child.content.dont_extend_height = to_filter(True)
        input_stack.children.insert(0, input_border_row())
        input_stack.children.append(input_border_row())
        return layout


def permission_key_bindings(
    approval_mode: ApprovalModeController,
    task_controller: InteractiveTaskController | None = None,
    *,
    console: Console | None = None,
    clipboard_grabber: Callable[[], ClipboardImageResult] = grab_clipboard_image,
) -> KeyBindings:
    """Build the Shift+Tab, Esc, Ctrl+V, and Enter shortcuts."""

    bindings = KeyBindings()
    clipboard_jobs: set[asyncio.Task[None]] = set()

    async def submit(event) -> None:
        if clipboard_jobs:
            await asyncio.gather(*tuple(clipboard_jobs), return_exceptions=True)
        if event.app.is_done:
            return
        buffer = event.app.current_buffer
        if (
            task_controller is not None
            and task_controller.active
            and not task_controller.awaiting_approval
            and not task_controller.awaiting_plan_review
            and buffer.text.strip() != "/cancel"
        ):
            return
        buffer.validate_and_handle()

    @bindings.add(Keys.BackTab)
    def toggle_approval_mode(event) -> None:
        approval_mode.toggle()
        event.app.invalidate()

    @bindings.add(Keys.Escape)
    def cancel_running_task(event) -> None:
        if task_controller is not None and task_controller.request_cancel():
            event.app.invalidate()

    @bindings.add(Keys.ControlV)
    def paste_clipboard_image(event):
        async def capture() -> None:
            buffer = event.app.current_buffer
            grabbed = await asyncio.to_thread(clipboard_grabber)
            if event.app.is_done:
                return
            if grabbed.ok and grabbed.path is not None:
                buffer.insert_text(f"@image:<{grabbed.path.resolve()}> ")
            elif console is not None:
                console.print(f"[yellow]Ctrl+V 抓图失败:[/yellow] {grabbed.error}")
            event.app.invalidate()

        job = asyncio.create_task(capture())
        clipboard_jobs.add(job)
        job.add_done_callback(clipboard_jobs.discard)
        return job

    @bindings.add(Keys.Enter)
    async def submit_after_clipboard(event) -> None:
        await submit(event)

    return bindings


def prompt_message() -> list[tuple[str, str]]:
    """Build the small editable prompt that can be safely redrawn."""

    return [("class:prompt", "❯ ")]


def user_history_message(message: str) -> Align:
    """Render submitted input without copying the interactive composer chrome."""

    lines = message.splitlines() or [""]
    rendered = Text("❯ ", style="bright_black")
    rendered.append(lines[0])
    for line in lines[1:]:
        rendered.append("\n  ")
        rendered.append(line)
    return Align.left(rendered, style="on grey93")


def prompt_placeholder(task_controller: InteractiveTaskController) -> list[tuple[str, str]]:
    if task_controller.awaiting_approval:
        text = "Approve tool: y / n / a / s"
    elif task_controller.awaiting_plan_review:
        text = "Plan review: execute / modify / cancel"
    elif task_controller.cancelling:
        text = "Cancelling current task..."
    elif task_controller.active:
        text = "Task running — draft the next message; Enter is disabled"
    else:
        text = "Type a message, @image:<path>, or Ctrl+V"
    return [("class:placeholder", text)]


def prompt_status(
    *,
    model: str,
    stats: dict[str, Any] | None = None,
    approval_mode: ApprovalMode = "ask",
    task_state: TaskState | None = None,
) -> list[tuple[str, str]]:
    """Build the single compact status line below the editable prompt."""

    return bottom_toolbar(
        model,
        stats,
        approval_mode=approval_mode,
        task_state=task_state,
    )


def bottom_toolbar(
    model: str,
    stats: dict[str, Any] | None = None,
    *,
    approval_mode: ApprovalMode = "ask",
    task_state: TaskState | None = None,
) -> list[tuple[str, str]]:
    stats = stats or {}
    has_usage = bool(stats.get("has_usage"))
    context_ratio = float(stats.get("context_ratio") or 0)
    context_text = _format_toolbar_percent(context_ratio) if has_usage else "0%"
    toolbar = [
        ("class:toolbar.model", model),
        ("class:toolbar.gap", "    "),
        ("class:toolbar.ctx.bar", _format_toolbar_bar(context_ratio if has_usage else 0)),
        ("class:toolbar.gap", " "),
        ("class:toolbar.ctx.value", context_text),
        ("class:toolbar.gap", "  "),
        (f"class:toolbar.mode.{approval_mode}", approval_mode_label(approval_mode)),
        ("class:toolbar.gap", "  Shift+Tab"),
    ]
    if task_state is not None:
        toolbar.extend(
            [
                ("class:toolbar.gap", "  Task "),
                ("class:toolbar.task", task_state.value),
            ]
        )
    return toolbar


def input_border_row() -> VSplit:
    """Keep the input rule inside Prompt Toolkit's measured layout."""
    return VSplit(
        [
            Window(width=2),
            Window(char="─", style="class:input.rule"),
            Window(width=1),
        ],
        height=1,
    )


def approval_mode_label(mode: ApprovalMode) -> str:
    return "Auto" if mode == "auto" else "Ask"


def _format_toolbar_bar(value: float, *, width: int = 12) -> str:
    bounded = max(0.0, min(value, 1.0))
    filled = round(bounded * width)
    if bounded > 0 and filled == 0:
        filled = 1
    return "█" * filled + "░" * (width - filled)


def _format_toolbar_percent(value: float) -> str:
    if value <= 0:
        return "0%"
    if value < 0.01:
        return "<1%"
    return f"{value:.0%}"
