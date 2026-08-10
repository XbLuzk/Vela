from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from vela.llm.model_profiles import ModelProfile


@dataclass(slots=True)
class ModelSelectorState:
    profiles: list[ModelProfile]
    current_provider: str
    current_model: str
    index: int = 0

    def __post_init__(self) -> None:
        for index, profile in enumerate(self.profiles):
            if (
                profile.provider == self.current_provider.lower()
                and profile.model == self.current_model
            ):
                self.index = index
                break

    def move(self, delta: int) -> None:
        self.index = (self.index + delta) % max(len(self.profiles), 1)

    def selected_profile(self) -> ModelProfile:
        return self.profiles[self.index]

    def render(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = [
            ("class:command", "> /model\n\n"),
            ("class:title", f"Models ({len(self.profiles)})\n"),
            ("class:line", "─" * 78 + "\n"),
            ("class:heading", "Current\n"),
            ("class:muted", "  Provider : "),
            ("", self.current_provider + "\n"),
            ("class:muted", "  Model    : "),
            ("", self.current_model + "\n\n"),
        ]
        for index, profile in enumerate(self.profiles):
            selected = index == self.index
            current = (
                profile.provider == self.current_provider.lower()
                and profile.model == self.current_model
            )
            marker = "> " if selected else "  "
            style = "class:selected" if selected else "class:model"
            check = " ✓" if current else ""
            fragments.append((style, f"{marker}{profile.name}{check}\n"))
            fragments.append(
                (
                    "class:muted",
                    f"    {profile.provider} · modelID: {profile.model} · "
                    f"context: {profile.context_window:,}\n",
                )
            )
            if profile.description:
                fragments.append(("class:muted", f"    {profile.description}\n"))
        fragments.append(("class:footer", "\n↑↓ navigate · Enter select · Esc back"))
        return fragments


async def run_model_selector(state: ModelSelectorState) -> ModelProfile | None:
    bindings = KeyBindings()
    control = FormattedTextControl(text=state.render, focusable=True, show_cursor=False)

    def refresh(event) -> None:
        event.app.invalidate()

    @bindings.add("up")
    def _up(event) -> None:
        state.move(-1)
        refresh(event)

    @bindings.add("down")
    def _down(event) -> None:
        state.move(1)
        refresh(event)

    @bindings.add("enter")
    def _select(event) -> None:
        event.app.exit(result=state.selected_profile())

    @bindings.add("escape")
    @bindings.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(result=None)

    application: Application[ModelProfile | None] = Application(
        layout=Layout(Window(control, wrap_lines=False, always_hide_cursor=True)),
        key_bindings=bindings,
        style=Style.from_dict(
            {
                "command": "#c084fc",
                "title": "bold #ffffff",
                "heading": "bold #ffffff",
                "line": "#555555",
                "model": "#f3f4f6",
                "selected": "bold #22c55e",
                "muted": "#9a9a9a",
                "footer": "italic #9a9a9a",
            }
        ),
        full_screen=False,
        mouse_support=False,
    )
    return await application.run_async()
