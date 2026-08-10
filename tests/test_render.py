from __future__ import annotations

import asyncio
import time
from io import StringIO

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from vela.config import load_config
from vela.entrypoints.repl_ui import (
    REPL_STYLE_RULES,
    PermissionModeController,
    bottom_toolbar,
    input_border_float,
    permission_key_bindings,
    prompt_message,
)
from vela.image import ClipboardImageResult
from vela.render import RichRenderer
from vela.render.rich_renderer import RICH_STYLE_RULES
from vela.task_control import InteractiveTaskController, TaskState


def test_repl_palette_follows_terminal_theme():
    assert all("bg:" not in style for style in REPL_STYLE_RULES.values())
    assert all("#" not in style for style in REPL_STYLE_RULES.values())


def test_rich_palette_follows_terminal_theme():
    assert all("bg:" not in style for style in RICH_STYLE_RULES.values())
    assert all("#" not in style for style in RICH_STYLE_RULES.values())


def test_input_border_is_a_layout_float_below_the_cursor():
    border = input_border_float()

    assert border.ycursor
    assert border.left == 2
    assert border.right == 1
    assert border.height == 1
    assert border.content.char == "─"
    assert border.content.style == "class:input.rule"
    assert "bottom-toolbar" not in REPL_STYLE_RULES


def test_banner_renders_vela_constellation_layout():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=200)
    renderer = RichRenderer(console=console)

    renderer.banner(
        model="deepseek-v4-flash",
        provider="deepseek",
        cwd="/tmp/project",
        tools=12,
        version="0.1.0",
        api_key_configured=True,
        mcp_servers=1,
        skills=3,
        agents_files=2,
        hitl_mode="never",
    )

    output = stream.getvalue()
    assert "✦       ·" in output
    assert "·   ✦" in output
    assert "Vela v0.1.0" in output
    assert "Signed in API Key" in output
    assert "What's new (v0.1.0)" in output


def test_prompt_message_keeps_status_and_input_together():
    prompt = prompt_message(
        cwd="/tmp/project",
        model="deepseek-v4-flash",
        tools=12,
        agents_files=2,
        mcp_servers=1,
        skills=3,
        stats={"total_tokens": 13187, "context_ratio": 0.013, "has_usage": True},
    )
    plain = "".join(text for _style, text in prompt)

    assert "2 AGENTS.md files" in plain
    assert "1 MCP server" in plain
    assert "3 skills · Tools 12" in plain
    assert "Default  Shift+Tab" in plain
    assert "deepseek-v4-flash" in plain
    assert "█░░░░░░░░░░░ 1%" in plain
    assert "/tmp/project" in plain
    assert "\n\n* " in plain
    assert plain.endswith("\n* ")


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


def test_text_deltas_render_as_markdown_on_turn_complete():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console, context_window=1000)

    renderer.handle({"type": "thinking_delta", "thinking": "需要先确认项目结构"})
    renderer.handle({"type": "text_delta", "text": "你好，我是 **Ve"})
    renderer.handle({"type": "text_delta", "text": "la**\n\n- `read_file`\n- **网页搜索**"})
    renderer.handle({"type": "usage", "usage": {"input_tokens": 250, "output_tokens": 50}})
    renderer.handle({"type": "turn_complete"})
    renderer.handle({"type": "done", "total_turns": 1, "total_tokens": 300})

    output = stream.getvalue()
    assert "Thinking" in output
    assert "需要先确认项目结构" in output
    assert "Final Output" in output
    assert "Vela" in output
    assert "read_file" in output
    assert "网页搜索" in output
    assert "Run Summary" not in output
    assert "**Vela**" not in output
    assert "`read_file`" not in output

    stats = renderer.toolbar_status()
    assert stats["turns"] == 1
    assert stats["input_tokens"] == 250
    assert stats["output_tokens"] == 50
    assert stats["total_tokens"] == 300
    assert stats["context_ratio"] == 0.25
    assert stats["has_usage"] is True


def test_interleaved_thinking_does_not_repeat_assistant_output_panels():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "text_delta", "text": "第一段"})
    renderer.handle({"type": "thinking_delta", "thinking": "中途补充思考"})
    renderer.handle({"type": "text_delta", "text": "第二段"})
    renderer.handle({"type": "turn_complete"})

    output = stream.getvalue()
    assert output.count("Assistant Output") == 0
    assert output.count("Final Output") == 1
    assert output.count("Thinking") == 1
    assert "第一段第二段" in output


def test_plan_status_and_scoped_thinking_render_with_task_identity():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "text_delta", "text": "正在规划任务"})
    renderer.handle({"type": "plan_status", "phase": "planning"})
    renderer.handle(
        {
            "type": "thinking_delta",
            "thinking": "先拆分任务",
            "phase": "planning",
        }
    )
    renderer.handle(
        {
            "type": "plan_task_started",
            "task_id": "task_1",
            "task_description": "检查模型配置",
        }
    )
    renderer.handle(
        {
            "type": "thinking_delta",
            "thinking": "读取配置文件",
            "phase": "execution",
            "task_id": "task_1",
        }
    )
    renderer.handle(
        {
            "type": "tool_call",
            "name": "read_file",
            "input": {"path": "config.py"},
            "task_id": "task_1",
        }
    )

    output = stream.getvalue()
    assert "Plan" in output
    assert "正在规划任务" in output
    assert "Thinking · planning" in output
    assert "先拆分任务" in output
    assert "Running task_1" in output
    assert "检查模型配置" in output
    assert "Thinking · task_1" in output
    assert "读取配置文件" in output
    assert "Tool Use · task_1" in output


def test_plan_review_flushes_plan_and_prints_choices():
    stream = StringIO()
    renderer = RichRenderer(console=Console(file=stream, color_system=None, width=120))

    renderer.handle({"type": "text_delta", "text": "计划内容"})
    renderer.handle({"type": "plan_review"})

    output = stream.getvalue()
    assert "计划内容" in output
    assert "execute" in output
    assert "modify" in output
    assert "cancel" in output


def test_streaming_text_waits_for_turn_boundary_by_default():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120, force_terminal=True)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "text_delta", "text": "chunk 1"})
    renderer.handle({"type": "text_delta", "text": "chunk 2"})

    assert "Assistant Output" not in stream.getvalue()
    renderer.handle({"type": "turn_complete"})
    assert stream.getvalue().count("Final Output") == 1


def test_tool_use_and_result_render_as_structured_panels():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "tool_call", "name": "list_dir", "input": {"path": "."}})
    renderer.handle(
        {
            "type": "tool_result",
            "name": "list_dir",
            "result": "README.md\nsrc/",
            "is_error": False,
        }
    )

    output = stream.getvalue()
    assert "Tool Use" in output
    assert "list_dir" in output
    assert '"path": "."' in output
    assert "Tool Result · list_dir · ok" in output
    assert "README.md" in output


def test_replayed_tool_result_is_visible_to_user():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle(
        {
            "type": "tool_result",
            "name": "write_file",
            "result": "Wrote result.txt",
            "is_error": False,
            "replayed": True,
            "recovery_status": "replayed",
        }
    )

    assert "Tool Result · write_file · ok · replayed" in stream.getvalue()


def test_start_run_resets_token_usage():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console, context_window=1000)

    renderer.handle({"type": "usage", "usage": {"input_tokens": 900, "output_tokens": 10}})
    renderer.start_run()
    renderer.handle({"type": "usage", "usage": {"input_tokens": 100, "output_tokens": 20}})
    renderer.handle({"type": "done", "total_turns": 1, "total_tokens": 120})

    assert "900" not in stream.getvalue()
    stats = renderer.toolbar_status()
    assert stats["input_tokens"] == 100
    assert stats["output_tokens"] == 20
    assert stats["total_tokens"] == 120
    assert stats["context_ratio"] == 0.1


def test_missing_usage_keeps_toolbar_tokens_unavailable():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console, context_window=1000)

    renderer.handle({"type": "done", "total_turns": 1, "total_tokens": 0})

    assert "Run Summary" not in stream.getvalue()
    toolbar = bottom_toolbar("/tmp/project", "deepseek-v4-flash", renderer.toolbar_status())
    assert ("class:toolbar.model", "deepseek-v4-flash") in toolbar
    assert ("class:toolbar.ctx.bar", "░░░░░░░░░░░░") in toolbar
    assert ("class:toolbar.ctx.value", "0%") in toolbar
