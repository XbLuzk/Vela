"""Prompt input, shortcuts, and status-line rendering for the interactive REPL."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import Float, FloatContainer, Window
from rich.console import Console

from vela.config import VelaConfig
from vela.image import ClipboardImageResult, grab_clipboard_image
from vela.task_control import InteractiveTaskController, TaskState

REPL_STYLE_RULES = {
    "prompt": "bold ansiblue",
    "placeholder": "italic ansibrightblack",
    "input.rule": "ansibrightblack",
    "prompt.dim": "ansibrightblack",
    "prompt.count.agents": "bold ansiblue",
    "prompt.count.mcp": "bold ansiblue",
    "prompt.count.skills": "bold ansiblue",
    "prompt.tools": "bold ansiblue",
    "toolbar.model": "bold",
    "toolbar.ctx.bar": "ansigreen",
    "toolbar.ctx.value": "",
    "toolbar.cwd.value": "ansiblue",
    "toolbar.mode.default": "bold ansigreen",
    "toolbar.mode.auto": "bold ansiyellow",
    "toolbar.task": "bold ansimagenta",
    "toolbar.gap": "",
}

PermissionMode = Literal["default", "auto"]
MessageDelivery = Literal["steering", "follow_up"]


@dataclass(slots=True)
class MessageDeliveryController:
    """Remember which submit shortcut produced the next prompt value."""

    _next: MessageDelivery = "steering"

    def mark(self, delivery: MessageDelivery) -> None:
        self._next = delivery

    def consume(self) -> MessageDelivery:
        delivery = self._next
        self._next = "steering"
        return delivery


@dataclass
class PermissionModeController:
    """Switch the live config between guarded and full-access modes."""

    config: VelaConfig
    mode: PermissionMode = "default"

    def __post_init__(self) -> None:
        self._default_hitl_mode = self.config.policy.hitl_mode
        self._default_path_guard_enabled = self.config.policy.path_guard_enabled
        self._default_command_guard_enabled = self.config.policy.command_guard_enabled
        self.set(self.mode)

    def set(self, mode: PermissionMode) -> PermissionMode:
        self.mode = mode
        if mode == "auto":
            self.config.policy.hitl_mode = "never"
            self.config.policy.path_guard_enabled = False
            self.config.policy.command_guard_enabled = False
        else:
            self.config.policy.hitl_mode = self._default_hitl_mode
            self.config.policy.path_guard_enabled = self._default_path_guard_enabled
            self.config.policy.command_guard_enabled = self._default_command_guard_enabled
        return self.mode

    def toggle(self) -> PermissionMode:
        return self.set("auto" if self.mode == "default" else "default")


class BorderedPromptSession(PromptSession):
    """Prompt Toolkit session with a rule below the current input line."""

    def _create_layout(self):
        layout = super()._create_layout()
        layout.container = FloatContainer(
            content=layout.container,
            floats=[input_border_float()],
        )
        return layout


def permission_key_bindings(
    permission_mode: PermissionModeController,
    task_controller: InteractiveTaskController | None = None,
    *,
    message_delivery: MessageDeliveryController | None = None,
    console: Console | None = None,
    clipboard_grabber: Callable[[], ClipboardImageResult] = grab_clipboard_image,
) -> KeyBindings:
    """Build the Shift+Tab, Esc, Ctrl+V, and Enter shortcuts."""

    bindings = KeyBindings()
    clipboard_jobs: set[asyncio.Task[None]] = set()

    async def submit(event, delivery: MessageDelivery) -> None:
        if clipboard_jobs:
            await asyncio.gather(*tuple(clipboard_jobs), return_exceptions=True)
        if event.app.is_done:
            return
        if message_delivery is not None:
            message_delivery.mark(delivery)
        event.app.current_buffer.validate_and_handle()

    @bindings.add(Keys.BackTab)
    def toggle_permission_mode(event) -> None:
        permission_mode.toggle()
        event.app.invalidate()

    @bindings.add(Keys.Escape)
    def cancel_running_task(event) -> None:
        if task_controller is not None and task_controller.request_cancel():
            event.app.current_buffer.reset()
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
        await submit(event, "steering")

    @bindings.add(Keys.Escape, Keys.Enter)
    async def submit_follow_up(event) -> None:
        await submit(event, "follow_up")

    return bindings


def prompt_message(
    *,
    cwd: str,
    model: str,
    tools: int,
    agents_files: int,
    mcp_servers: int,
    skills: int,
    stats: dict[str, Any] | None = None,
    permission_mode: PermissionMode = "default",
    task_state: TaskState | None = None,
) -> list[tuple[str, str]]:
    """Build the text shown above and beside the input cursor."""

    return [
        ("class:prompt.count.agents", str(agents_files)),
        ("class:prompt.dim", f" {_plural_label(agents_files, 'AGENTS.md file')} · "),
        ("class:prompt.count.mcp", str(mcp_servers)),
        ("class:prompt.dim", f" {_plural_label(mcp_servers, 'MCP server')} · "),
        ("class:prompt.count.skills", str(skills)),
        ("class:prompt.dim", f" {_plural_label(skills, 'skill')} · Tools "),
        ("class:prompt.tools", str(tools)),
        ("class:prompt.dim", "\n"),
        *bottom_toolbar(
            cwd,
            model,
            stats,
            permission_mode=permission_mode,
            task_state=task_state,
        ),
        ("class:prompt.dim", "\n\n"),
        ("class:prompt", "* "),
    ]


def bottom_toolbar(
    cwd: str,
    model: str,
    stats: dict[str, Any] | None = None,
    *,
    permission_mode: PermissionMode = "default",
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
        ("class:toolbar.cwd.value", _shorten_home(cwd)),
        ("class:toolbar.gap", "  "),
        (f"class:toolbar.mode.{permission_mode}", permission_mode_label(permission_mode)),
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


def input_border_float() -> Float:
    return Float(
        content=Window(char="─", style="class:input.rule"),
        left=2,
        right=1,
        height=1,
        ycursor=True,
    )


def permission_mode_label(mode: PermissionMode) -> str:
    return "Auto (full access)" if mode == "auto" else "Default"


def _plural_label(count: int, singular: str) -> str:
    return singular if count == 1 else singular + "s"


def _shorten_home(path: str) -> str:
    home = str(Path.home())
    if path == home:
        return "~"
    prefix = home + os.sep
    if path.startswith(prefix):
        return "~/" + path[len(prefix) :]
    return path


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
