from __future__ import annotations

import asyncio

from vela.config import load_config
from vela.tools import get_builtin_tools
from vela.tools.base import Tool, ToolContext


def _context(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.policy.approval_mode = "auto"
    return ToolContext(cwd=str(tmp_path), config=config)


def _tool(name: str) -> Tool:
    return next(tool for tool in get_builtin_tools() if tool.name == name)


def _run(tool: Tool, payload: dict, context: ToolContext):
    return asyncio.run(tool.execute(payload, context))


def test_workspace_tools_are_declared_with_matching_write_semantics():
    tools = {tool.name: tool for tool in get_builtin_tools()}

    read_only_names = ("read_file", "list_dir", "glob", "grep", "directory_tree", "get_file_info")

    assert set(read_only_names) <= set(tools)
    for name in read_only_names:
        assert tools[name].is_read_only
        assert tools[name].danger_level == "safe"
    for name in ("write_file", "edit_file"):
        assert not tools[name].is_read_only
        assert not tools[name].is_concurrency_safe
        assert tools[name].danger_level == "medium"


def test_edit_file_tool_edits_and_previews(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)
    (tmp_path / "code.txt").write_text("alpha\n", encoding="utf-8")
    tool = _tool("edit_file")

    preview = _run(
        tool,
        {"path": "code.txt", "old_text": "alpha", "new_text": "beta", "dry_run": True},
        context,
    )
    applied = _run(tool, {"path": "code.txt", "old_text": "alpha", "new_text": "beta"}, context)

    assert "[DRY RUN]" in preview.content
    assert not applied.is_error
    assert (tmp_path / "code.txt").read_text(encoding="utf-8") == "beta\n"


def test_list_dir_glob_and_grep_tools_read_the_workspace(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("needle = 1\n", encoding="utf-8")

    listed = _run(_tool("list_dir"), {"path": "."}, context)
    globbed = _run(_tool("glob"), {"pattern": "**/*.py", "limit": 5}, context)
    grepped = _run(_tool("grep"), {"pattern": "needle", "path": "pkg"}, context)
    literal = _run(_tool("grep"), {"pattern": "needle = 1", "regex": False}, context)

    assert "pkg/" in listed.content
    assert globbed.content == "pkg/a.py"
    assert "pkg/a.py:1: needle = 1" in grepped.content
    assert "pkg/a.py:1: needle = 1" in literal.content


def test_directory_tree_and_file_info_tools_expose_metadata(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x", encoding="utf-8")

    tree = _run(_tool("directory_tree"), {"max_depth": 2}, context)
    pruned = _run(_tool("directory_tree"), {"exclude_patterns": ["pkg"]}, context)
    info = _run(_tool("get_file_info"), {"path": "pkg/a.py"}, context)

    assert "pkg/" in tree.content
    assert "a.py" in tree.content
    assert "pkg" not in pruned.content
    assert "Type: file" in info.content


def test_write_file_reconciles_only_identical_non_append_content(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)
    tool = _tool("write_file")
    assert tool.reconcile is not None
    (tmp_path / "done.txt").write_text("finished\n", encoding="utf-8")

    matched = asyncio.run(tool.reconcile({"path": "done.txt", "content": "finished\n"}, context))
    mismatched = asyncio.run(tool.reconcile({"path": "done.txt", "content": "other\n"}, context))
    appended = asyncio.run(
        tool.reconcile({"path": "done.txt", "content": "finished\n", "append": True}, context)
    )
    missing = asyncio.run(tool.reconcile({"path": "absent.txt", "content": "any"}, context))

    assert matched is not None
    assert matched.recovery_status == "reconciled"
    assert matched.content == "Wrote done.txt (reconciled existing content)"
    assert mismatched is None
    assert appended is None
    assert missing is None


def test_bash_tool_reports_timeouts_as_errors(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)

    result = _run(_tool("bash"), {"command": "sleep 5", "timeout": 0.1}, context)

    assert result.is_error
    assert "timed out after 0s" in result.content


def test_bash_tool_returns_combined_output_and_exit_status(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)

    ok = _run(_tool("bash"), {"command": "echo out; echo err 1>&2"}, context)
    failed = _run(_tool("bash"), {"command": "exit 3"}, context)

    assert not ok.is_error
    assert "out" in ok.content
    assert "err" in ok.content
    assert failed.is_error
    assert failed.content == "(exit 3, no output)"


def test_bash_tool_truncates_very_long_output(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)

    result = _run(
        _tool("bash"),
        {"command": "python -c \"print('x' * 30000)\""},
        context,
    )

    assert result.content.endswith("... [truncated]")
    assert len(result.content) < 21_000
