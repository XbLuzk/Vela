from __future__ import annotations

from io import StringIO

from rich.console import Console

from vela.entrypoints.repl_ui import bottom_toolbar
from vela.render import RichRenderer
from vela.render.rich_renderer import RICH_STYLE_RULES


def test_rich_palette_follows_terminal_theme():
    assert all("bg:" not in style for style in RICH_STYLE_RULES.values())
    assert all("#" not in style for style in RICH_STYLE_RULES.values())


def test_banner_renders_vela_constellation_layout():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=200)
    renderer = RichRenderer(console=console)

    renderer.banner(
        version="0.1.0",
        api_key_configured=True,
    )

    output = stream.getvalue()
    assert "✦       ·" in output
    assert "·   ✦" in output
    assert "Vela v0.1.0" in output
    assert "Signed in API Key" in output
    assert "What's new (v0.1.0)" in output


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
    assert "Final Output" not in output
    assert "Vela" in output
    assert "read_file" in output
    assert "网页搜索" in output
    assert "╭" not in output
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


def test_interleaved_thinking_keeps_one_compact_assistant_message():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "text_delta", "text": "第一段"})
    renderer.handle({"type": "thinking_delta", "thinking": "中途补充思考"})
    renderer.handle({"type": "text_delta", "text": "第二段"})
    renderer.handle({"type": "turn_complete"})

    output = stream.getvalue()
    assert "Assistant Output" not in output
    assert "Final Output" not in output
    assert output.count("Thinking") == 1
    assert output.count("●") == 1
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
    assert "read_file · task_1" in output
    assert "Tool Use" not in output
    assert "╭" not in output


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
    assert "chunk 1chunk 2" in stream.getvalue()
    assert "Final Output" not in stream.getvalue()


def test_streaming_thinking_waits_for_output_boundary_by_default():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120, force_terminal=True)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "thinking_delta", "thinking": "chunk 1"})
    renderer.handle({"type": "thinking_delta", "thinking": "chunk 2"})

    assert stream.getvalue() == ""
    renderer.handle({"type": "text_delta", "text": "done"})
    assert stream.getvalue().count("Thinking") == 1


def test_tool_use_and_result_render_as_compact_lines():
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
    assert "Tool Use" not in output
    assert "Tool Result" not in output
    assert "list_dir" in output
    assert '"path": "."' in output
    assert "list_dir · ok · README.md src/" in output
    assert "╭" not in output


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

    assert "write_file · ok · replayed · Wrote result.txt" in stream.getvalue()


def test_long_tool_details_stay_on_single_lines():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=80)
    renderer = RichRenderer(console=console)

    renderer.handle(
        {"type": "tool_call", "name": "search_memory", "input": {"query": "用户信息" * 80}}
    )
    renderer.handle(
        {
            "type": "tool_result",
            "name": "search_memory",
            "result": "搜索结果" * 100,
            "is_error": False,
        }
    )

    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert all(line.endswith("…") for line in lines)


def test_long_thinking_is_collapsed_to_one_summary_line():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=200)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "thinking_delta", "thinking": "分析细节 " * 80})
    renderer.handle({"type": "text_delta", "text": "结论"})
    renderer.handle({"type": "turn_complete"})

    output = stream.getvalue()
    thinking_line = next(line for line in output.splitlines() if "Thinking" in line)
    assert thinking_line.endswith("…")
    assert len(thinking_line) < 190


def test_run_summary_is_compact_and_has_no_persisted_identifier():
    stream = StringIO()
    renderer = RichRenderer(console=Console(file=stream, color_system=None, width=120))

    renderer.print_run_summary(
        status="completed",
        duration_ms=4020,
        turns=2,
        tool_calls=1,
        total_tokens=6513,
    )
    renderer.print_run_summary(
        status="failed",
        duration_ms=500,
        turns=0,
        tool_calls=0,
        total_tokens=0,
    )

    output = stream.getvalue()
    assert "✦ Completed in 4.02s · 2 turns · 1 tool · 6.5k tokens" in output
    assert "✦ Failed in 0.50s · 0 turns · 0 tools" in output
    assert "run_" not in output


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
    toolbar = bottom_toolbar("deepseek-v4-flash", renderer.toolbar_status())
    assert ("class:toolbar.model", "deepseek-v4-flash") in toolbar
    assert ("class:toolbar.ctx.bar", "░░░░░░░░░░░░") in toolbar
    assert ("class:toolbar.ctx.value", "0%") in toolbar
