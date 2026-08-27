from __future__ import annotations

import asyncio
import os
import signal

import pytest

from vela.config import load_config
from vela.tools import ToolRegistry, get_builtin_tools
from vela.tools.base import ToolContext
from vela.tools.builtins import _save_memory, _search_memory
from vela.tools.process import stop_subprocess


def test_read_write_file_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.policy.approval_mode = "auto"
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    context = ToolContext(cwd=str(tmp_path), config=config)

    async def run():
        write = registry.get("write_file")
        read = registry.get("read_file")
        assert write and read
        write_result = await write.execute(
            {"path": "hello.txt", "content": "hello\nworld\n"},
            context,
        )
        read_result = await read.execute({"path": "hello.txt"}, context)
        return write_result, read_result

    write_result, read_result = asyncio.run(run())
    assert not write_result.is_error
    assert "1: hello" in read_result.content
    assert "2: world" in read_result.content


def test_builtin_tools_include_memory_recall_and_read_only_skills():
    names = {tool.name for tool in get_builtin_tools()}

    assert "search_memory" in names
    assert "load_skill" in names
    assert "save_skill" not in names
    assert "web_search" not in names
    assert "web_fetch" not in names


def test_bash_tool_cancellation_stops_process_group(tmp_path, monkeypatch):
    config = load_config(project_root=tmp_path)
    context = ToolContext(cwd=str(tmp_path), config=config)
    bash = next(tool for tool in get_builtin_tools() if tool.name == "bash")
    signals = []

    class FakeProcess:
        pid = 4242
        returncode = None

        async def communicate(self):
            await asyncio.Event().wait()

        async def wait(self):
            self.returncode = -15
            return self.returncode

    async def create_process(*args, **kwargs):  # noqa: ARG001
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_shell", create_process)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    async def run():
        task = asyncio.create_task(bash.execute({"command": "sleep 60"}, context))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    assert signals
    assert signals[0][0] == 4242


def test_subprocess_stop_escalates_to_sigkill(monkeypatch):
    signals = []

    class FakeProcess:
        pid = 4343
        returncode = None

        def __init__(self):
            self.waits = 0

        async def wait(self):
            self.waits += 1
            if self.waits == 1:
                raise TimeoutError
            self.returncode = -9
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    asyncio.run(stop_subprocess(process))

    assert signals == [(4343, signal.SIGTERM), (4343, signal.SIGKILL)]
    assert process.waits == 2


def test_subprocess_stop_bounds_wait_after_sigkill(monkeypatch):
    signals = []

    class FakeProcess:
        pid = 4344
        returncode = None

        def __init__(self):
            self.waits = 0

        async def wait(self):
            self.waits += 1
            raise TimeoutError

    process = FakeProcess()
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    asyncio.run(stop_subprocess(process))

    assert signals == [(4344, signal.SIGTERM), (4344, signal.SIGKILL)]
    assert process.waits == 2


def test_memory_tools_save_metadata_and_recall_relevant_items(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    context = ToolContext(cwd=str(tmp_path), config=config)

    saved = asyncio.run(
        _save_memory(
            {
                "content": "用户偏好用 uv 执行 Python 测试",
                "kind": "preference",
                "importance": 0.9,
            },
            context,
        )
    )
    recalled = asyncio.run(_search_memory({"query": "怎么执行测试"}, context))

    assert not saved.is_error
    assert not recalled.is_error
    assert "uv" in recalled.content
    assert "preference" in recalled.content


def test_memory_tools_share_the_feature_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.features.memory = False
    context = ToolContext(cwd=str(tmp_path), config=config)

    saved = asyncio.run(_save_memory({"content": "do not save"}, context))
    recalled = asyncio.run(_search_memory({"query": "anything"}, context))

    assert saved.is_error
    assert recalled.is_error
    assert saved.content == "Long-term memory is disabled."
    assert recalled.content == "Long-term memory is disabled."
