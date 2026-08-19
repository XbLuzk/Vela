from __future__ import annotations

import asyncio
import time
from io import StringIO

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import set_app
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    FloatContainer,
    HSplit,
    VerticalAlign,
    VSplit,
    Window,
)
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.styles import Style, merge_styles
from prompt_toolkit.styles.defaults import default_ui_style
from rich.console import Console

from vela.config import load_config
from vela.entrypoints.repl_ui import (
    REPL_STYLE_RULES,
    BorderedPromptSession,
    PermissionModeController,
    bottom_toolbar,
    permission_key_bindings,
    prompt_message,
    prompt_status,
)
from vela.image import ClipboardImageResult
from vela.task_control import InteractiveTaskController, TaskState


def test_repl_palette_follows_terminal_theme():
    assert all("bg:" not in style for style in REPL_STYLE_RULES.values())
    assert all("#" not in style for style in REPL_STYLE_RULES.values())


def test_status_bar_is_ephemeral_and_above_the_input():
    session = BorderedPromptSession(
        output=DummyOutput(),
        bottom_toolbar=[("class:toolbar.model", "model")],
    )
    root = session.app.layout.container

    assert isinstance(root, HSplit)
    status_section = root.children[0]
    assert isinstance(status_section, ConditionalContainer)
    assert isinstance(status_section.content, Window)
    assert status_section.content.style == "class:bottom-toolbar"

    async def visibility() -> tuple[bool, bool]:
        future = asyncio.get_running_loop().create_future()
        session.app.future = future
        with set_app(session.app):
            visible_while_prompting = status_section.filter()
        future.set_result(None)
        with set_app(session.app):
            hidden_when_done = not status_section.filter()
        return visible_while_prompting, hidden_when_done

    assert asyncio.run(visibility()) == (True, True)


def test_input_border_is_inside_the_main_input_stack():
    session = BorderedPromptSession(output=DummyOutput())
    root = session.app.layout.container

    assert session.reserve_space_for_menu == 0
    assert isinstance(root, HSplit)
    main_section = root.children[1]
    assert isinstance(main_section, ConditionalContainer)
    main_input = main_section.alternative_content
    assert isinstance(main_input, FloatContainer)
    input_stack = main_input.content
    assert isinstance(input_stack, HSplit)
    assert input_stack.align == VerticalAlign.TOP

    input_windows = [
        child.content
        for child in input_stack.children
        if isinstance(child, ConditionalContainer) and isinstance(child.content, Window)
    ]
    assert input_windows
    assert all(window.dont_extend_height() for window in input_windows)

    border = input_stack.children[-1]
    assert isinstance(border, VSplit)
    assert len(border.children) == 3
    rule = border.children[1]
    assert isinstance(rule, Window)
    assert rule.char == "─"
    assert rule.style == "class:input.rule"
    effective_style = merge_styles([default_ui_style(), Style.from_dict(REPL_STYLE_RULES)])
    assert not effective_style.get_attrs_for_style_str("class:bottom-toolbar").reverse
    assert not effective_style.get_attrs_for_style_str("class:bottom-toolbar.text").reverse
    count_style = effective_style.get_attrs_for_style_str(
        "class:bottom-toolbar class:bottom-toolbar.text class:prompt.count.agents"
    )
    assert count_style.color == "ansiblue"
    assert count_style.bgcolor == ""
    assert not count_style.reverse


def test_status_is_separate_from_the_redrawn_input_prompt():
    status = prompt_status(
        cwd="/tmp/project",
        model="deepseek-v4-flash",
        tools=12,
        agents_files=2,
        mcp_servers=1,
        skills=3,
        stats={"total_tokens": 13187, "context_ratio": 0.013, "has_usage": True},
    )
    status_text = "".join(text for _style, text in status)
    prompt_text = "".join(text for _style, text in prompt_message())

    assert "2 AGENTS.md files" in status_text
    assert "1 MCP server" in status_text
    assert "3 skills · Tools 12" in status_text
    assert "Default  Shift+Tab" in status_text
    assert "deepseek-v4-flash" in status_text
    assert "█░░░░░░░░░░░ 1%" in status_text
    assert "/tmp/project" in status_text
    assert "* " not in status_text
    assert prompt_text == "* "


def test_permission_mode_toggle_applies_and_restores_full_access_policy(tmp_path):
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = "always"
    controller = PermissionModeController(config)

    assert controller.mode == "default"
    assert config.policy.hitl_mode == "always"
    assert config.policy.path_guard_enabled
    assert config.policy.command_guard_enabled

    assert controller.toggle() == "auto"
    assert config.policy.hitl_mode == "never"
    assert not config.policy.path_guard_enabled
    assert not config.policy.command_guard_enabled

    assert controller.toggle() == "default"
    assert config.policy.hitl_mode == "always"
    assert config.policy.path_guard_enabled
    assert config.policy.command_guard_enabled


def test_shift_tab_is_bound_to_permission_mode_toggle(tmp_path):
    controller = PermissionModeController(load_config(project_root=tmp_path))
    bindings = permission_key_bindings(controller)

    assert any(binding.keys == (Keys.BackTab,) for binding in bindings.bindings)


def test_escape_is_bound_to_running_task_cancel(tmp_path):
    permission = PermissionModeController(load_config(project_root=tmp_path))
    task_controller = InteractiveTaskController()
    bindings = permission_key_bindings(permission, task_controller)

    assert any(binding.keys == (Keys.Escape,) for binding in bindings.bindings)


def test_running_task_keeps_draft_but_blocks_submit_until_completion(tmp_path):
    permission = PermissionModeController(load_config(project_root=tmp_path))
    task_controller = InteractiveTaskController()

    async def run_prompt() -> tuple[bool, str, str]:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=permission_key_bindings(permission, task_controller),
            )

            release_task = asyncio.Event()

            async def keep_running() -> None:
                await release_task.wait()

            task_controller.start(
                keep_running(),
                initial_state=TaskState.RUNNING,
                label="active task",
            )
            prompt = asyncio.create_task(session.prompt_async())
            pipe_input.send_text("next task\r")
            await asyncio.sleep(0.05)
            submitted_while_running = prompt.done()
            buffered_draft = session.default_buffer.text
            release_task.set()
            await task_controller.wait()
            pipe_input.send_text("\r")
            result = await prompt
            return submitted_while_running, buffered_draft, result

    submitted_while_running, buffered_draft, result = asyncio.run(run_prompt())

    assert not submitted_while_running
    assert buffered_draft == "next task"
    assert result == "next task"


def test_running_task_blocks_enter_that_was_buffered_before_prompt_started(tmp_path):
    permission = PermissionModeController(load_config(project_root=tmp_path))
    task_controller = InteractiveTaskController()

    async def run_prompt() -> tuple[bool, str]:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=permission_key_bindings(permission, task_controller),
            )

            async def keep_running() -> None:
                await asyncio.Event().wait()

            task_controller.start(
                keep_running(),
                initial_state=TaskState.RUNNING,
                label="active task",
            )
            pipe_input.send_text("next task\r")
            prompt = asyncio.create_task(session.prompt_async())
            await asyncio.sleep(0.05)
            result = prompt.done(), session.default_buffer.text
            prompt.cancel()
            await asyncio.gather(prompt, return_exceptions=True)
            task_controller.request_cancel()
            await task_controller.wait()
            return result

    submitted, draft = asyncio.run(run_prompt())

    assert not submitted
    assert draft == "next task"


def test_running_task_still_allows_cancel_command(tmp_path):
    permission = PermissionModeController(load_config(project_root=tmp_path))
    task_controller = InteractiveTaskController()

    async def run_prompt() -> str:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=permission_key_bindings(permission, task_controller),
            )

            async def keep_running() -> None:
                await asyncio.Event().wait()

            task_controller.start(
                keep_running(),
                initial_state=TaskState.RUNNING,
                label="active task",
            )
            prompt = asyncio.create_task(session.prompt_async())
            pipe_input.send_text("/cancel\r")
            result = await prompt
            task_controller.request_cancel()
            await task_controller.wait()
            return result

    assert asyncio.run(run_prompt()) == "/cancel"


def test_running_task_allows_tool_approval_response(tmp_path):
    permission = PermissionModeController(load_config(project_root=tmp_path))
    task_controller = InteractiveTaskController()
    approval_result = ""

    async def run_prompt() -> str:
        nonlocal approval_result
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=permission_key_bindings(permission, task_controller),
            )

            async def await_approval() -> None:
                nonlocal approval_result
                approval_result = await task_controller.request_approval(
                    {"tool_name": "write_file", "danger_level": "write", "input": {}}
                )

            task_controller.start(
                await_approval(),
                initial_state=TaskState.RUNNING,
                label="approval task",
            )
            while not task_controller.awaiting_approval:
                await asyncio.sleep(0)
            prompt = asyncio.create_task(session.prompt_async())
            pipe_input.send_text("y\r")
            result = await prompt
            task_controller.submit_approval(result)
            await task_controller.wait()
            return result

    assert asyncio.run(run_prompt()) == "y"
    assert approval_result == "approve"


def test_running_task_allows_plan_review_response(tmp_path):
    permission = PermissionModeController(load_config(project_root=tmp_path))
    task_controller = InteractiveTaskController()

    async def run_prompt() -> tuple[str, str]:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=permission_key_bindings(permission, task_controller),
            )

            review = asyncio.create_task(task_controller.request_plan_review(object()))
            while not task_controller.awaiting_plan_review:
                await asyncio.sleep(0)
            prompt = asyncio.create_task(session.prompt_async())
            pipe_input.send_text("execute\r")
            result = await prompt
            message = task_controller.submit_plan_review(result)
            decision = await review
            return message, decision.action

    message, action = asyncio.run(run_prompt())

    assert "开始执行" in message
    assert action == "execute"


def test_ctrl_v_is_bound_to_clipboard_image(tmp_path):
    permission = PermissionModeController(load_config(project_root=tmp_path))
    bindings = permission_key_bindings(permission)

    assert any(binding.keys == (Keys.ControlV,) for binding in bindings.bindings)


def test_ctrl_v_injects_image_reference_without_submitting(tmp_path):
    permission = PermissionModeController(load_config(project_root=tmp_path))
    image_path = tmp_path / "screen shot.png"

    async def run_prompt() -> str:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=permission_key_bindings(
                    permission,
                    clipboard_grabber=lambda: ClipboardImageResult.success(image_path),
                ),
            )

            pipe_input.send_text("\x16\r")
            return await session.prompt_async()

    result = asyncio.run(run_prompt())

    assert result == f"@image:<{image_path}> "


def test_ctrl_v_failure_preserves_existing_input(tmp_path):
    permission = PermissionModeController(load_config(project_root=tmp_path))
    console = Console(file=StringIO(), color_system=None)

    async def run_prompt() -> str:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=permission_key_bindings(
                    permission,
                    console=console,
                    clipboard_grabber=lambda: ClipboardImageResult.failure("no image"),
                ),
            )

            async def feed_input() -> None:
                pipe_input.send_text("hello\x16")
                await asyncio.sleep(0.05)
                pipe_input.send_text("\r")

            feeder = asyncio.create_task(feed_input())
            result = await session.prompt_async()
            await feeder
            return result

    result = asyncio.run(run_prompt())

    assert result == "hello"
    assert "no image" in console.file.getvalue()


def test_ctrl_v_capture_does_not_block_prompt_event_loop(tmp_path):
    permission = PermissionModeController(load_config(project_root=tmp_path))
    image_path = tmp_path / "screen.png"

    def slow_grabber() -> ClipboardImageResult:
        time.sleep(0.1)
        return ClipboardImageResult.success(image_path)

    async def run_prompt() -> tuple[str, bool]:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=permission_key_bindings(
                    permission,
                    clipboard_grabber=slow_grabber,
                ),
            )
            event_loop_advanced = False

            async def feed_input() -> None:
                nonlocal event_loop_advanced
                pipe_input.send_text("\x16")
                await asyncio.sleep(0.02)
                event_loop_advanced = True
                await asyncio.sleep(0.12)
                pipe_input.send_text("\r")

            feeder = asyncio.create_task(feed_input())
            result = await session.prompt_async()
            await feeder
            return result, event_loop_advanced

    result, event_loop_advanced = asyncio.run(run_prompt())

    assert event_loop_advanced
    assert result == f"@image:<{image_path}> "


def test_shift_tab_input_toggles_live_permission_mode(tmp_path):
    controller = PermissionModeController(load_config(project_root=tmp_path))

    async def run_prompt() -> None:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=permission_key_bindings(controller),
            )
            pipe_input.send_text("\x1b[Z\r")
            await session.prompt_async()

    asyncio.run(run_prompt())

    assert controller.mode == "auto"
    assert controller.config.policy.hitl_mode == "never"


def test_bottom_toolbar_uses_runtime_summary_segments():
    toolbar = bottom_toolbar(
        "/Users/me/project",
        "deepseek-v4-flash",
        {"turns": 1, "total_tokens": 13187, "context_ratio": 0.013, "has_usage": True},
    )

    assert ("class:toolbar.model", "deepseek-v4-flash") in toolbar
    assert ("class:toolbar.ctx.bar", "█░░░░░░░░░░░") in toolbar
    assert ("class:toolbar.ctx.value", "1%") in toolbar
    assert ("class:toolbar.cwd.value", "/Users/me/project") in toolbar
    assert not any(text == " TURN " for _style, text in toolbar)
    assert not any("Token" in text for _style, text in toolbar)


def test_bottom_toolbar_exposes_unified_task_state():
    toolbar = bottom_toolbar(
        "/tmp/project",
        "fake-model",
        task_state=TaskState.CANCELLING,
    )

    assert ("class:toolbar.task", "cancelling") in toolbar
