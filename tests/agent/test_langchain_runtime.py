from __future__ import annotations

import asyncio
from typing import Any

from vela.agent import Agent
from vela.agent.langchain_model import VelaChatModel
from vela.agent.langchain_runtime import run_langchain_agent
from vela.config import load_config
from vela.tools import ToolRegistry, get_builtin_tools
from vela.tools.base import Tool, ToolResult, object_schema
from vela.types import Message


class FakeClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "read_file", "arguments": '{"path":'},
                },
            }
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "function": {"arguments": '"note.txt"}'},
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
        else:
            tool_messages = [message for message in messages if message.role == "tool"]
            assert tool_messages
            assert "1: hello" in tool_messages[-1].content
            yield {"type": "text_delta", "text": "done"}
            yield {"type": "message_end", "stop_reason": "end_turn"}


class SkillLoadingClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            user_content = str(messages[-1].content)
            assert "Relevant skill candidates" in user_content
            assert "demo-skill" in user_content
            assert any(tool["function"]["name"] == "load_skill" for tool in tools)
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": "load_1",
                    "function": {"name": "load_skill", "arguments": '{"name":"demo-skill"}'},
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return

        tool_messages = [message for message in messages if message.role == "tool"]
        assert tool_messages
        assert "## Loaded Skill: demo-skill" in str(tool_messages[-1].content)
        assert "Apply the demo workflow now." in str(tool_messages[-1].content)
        yield {"type": "text_delta", "text": "skill applied"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class UsageAndCompressionClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 800

    def __init__(self):
        self.saw_summary = False

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.saw_summary = any(
            "conversation-summary" in str(message.content) for message in messages
        )
        yield {"type": "text_delta", "text": "compressed"}
        yield {"type": "message_end", "stop_reason": "end_turn"}
        yield {
            "type": "usage",
            "usage": {
                "input_tokens": 120,
                "output_tokens": 8,
                "prompt_cache_hit_tokens": 20,
                "prompt_cache_miss_tokens": 100,
            },
        }


class SingleTurnStreamingClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000
    api_key = "must-not-be-serialized"

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        yield {"type": "thinking_delta", "thinking": "checking"}
        yield {"type": "text_delta", "text": "answer"}
        yield {"type": "message_end", "stop_reason": "end_turn"}
        yield {
            "type": "usage",
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        }


class ParallelReadClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            for index, name in enumerate(("read_a", "write", "read_b")):
                yield {
                    "type": "tool_call_delta",
                    "tool_call": {
                        "index": index,
                        "id": f"call_{index}",
                        "function": {"name": name, "arguments": "{}"},
                    },
                }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return
        assert len([message for message in messages if message.role == "tool"]) == 3
        yield {"type": "text_delta", "text": "finished"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class DuplicateToolIdClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            for index, name in enumerate(("write_a", "write_b")):
                yield {
                    "type": "tool_call_delta",
                    "tool_call": {
                        "index": index,
                        "id": "duplicate",
                        "function": {"name": name, "arguments": "{}"},
                    },
                }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return
        yield {"type": "text_delta", "text": "finished"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_agent_executes_tool_and_replays_result(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    agent = Agent(
        llm_client=FakeClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def run() -> Any:
        return await agent.run_complete("read note")

    result = asyncio.run(run())
    assert result.text == "done"
    assert result.turns == 2


def test_react_loop_preserves_stream_event_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    agent = Agent(
        llm_client=FakeClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def run():
        return [event async for event in agent.run("read note")]

    events = asyncio.run(run())

    assert [event["type"] for event in events] == [
        "turn_complete",
        "tool_call",
        "tool_result",
        "text_delta",
        "turn_complete",
        "done",
    ]
    assert events[1]["name"] == "read_file"
    assert "1: hello" in events[2]["result"]
    assert events[-1]["total_turns"] == 2


def test_load_skill_is_injected_in_the_same_query_next_model_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    skill_dir = tmp_path / ".vela" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Use for the demo workflow\n"
        "tags: [demo]\n---\nApply the demo workflow now.\n",
        encoding="utf-8",
    )
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    client = SkillLoadingClient()
    agent = Agent(
        llm_client=client,
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    result = asyncio.run(agent.run_complete("请使用 demo-skill 完成这个任务"))

    assert result.text == "skill applied"
    assert result.turns == 2
    assert client.calls == 2


def test_runtime_integrates_context_compression_and_detailed_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.llm.max_tokens = 100
    config.memory.compression_reserve_tokens = 20
    config.memory.compression_threshold = 0.6
    client = UsageAndCompressionClient()
    agent = Agent(
        llm_client=client,
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
    )
    history = []
    for index in range(8):
        history.extend(
            [
                Message(role="user", content=f"old request {index} " + "x" * 220),
                Message(role="assistant", content=f"old answer {index} " + "y" * 220),
            ]
        )

    agent.history = history
    result = asyncio.run(agent.run_complete("new request"))

    assert result.text == "compressed"
    assert client.saw_summary
    assert result.total_tokens == 128
    assert result.usage.cache_hit_tokens == 20
    assert result.usage.cache_miss_tokens == 100


def test_langchain_runtime_keeps_one_model_call_and_vela_stream_events(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    client = SingleTurnStreamingClient()
    agent = Agent(
        llm_client=client,
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
    )

    async def run():
        return [event async for event in agent.run("hello")]

    events = asyncio.run(run())

    assert client.calls == 1
    assert [event["type"] for event in events] == [
        "thinking_delta",
        "text_delta",
        "usage",
        "turn_complete",
        "done",
    ]
    assert events[-1]["total_turns"] == 1
    assert events[-1]["total_tokens"] == 7


def test_langchain_model_serialization_excludes_provider_client_secret():
    model = VelaChatModel(client=SingleTurnStreamingClient())

    assert "client" not in model.model_dump()


def test_direct_langchain_runtime_returns_complete_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False

    async def run():
        events = []
        async for event in run_langchain_agent(
            llm_client=SingleTurnStreamingClient(),
            tool_registry=ToolRegistry(),
            system_prompt="test",
            user_message="hello",
            history=None,
            cwd=str(tmp_path),
            config=config,
        ):
            events.append(event)
        return events

    done = asyncio.run(run())[-1]

    assert [message.role for message in done["messages"]] == ["user", "assistant"]
    assert done["messages"][-1].content == "answer"


def test_langchain_runtime_preserves_parallel_reads_and_serial_write(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    config.policy.hitl_mode = "never"
    config.features.audit_log = False
    both_reads_started = asyncio.Event()
    release_reads = asyncio.Event()
    order: list[str] = []

    async def read(payload, context):  # noqa: ARG001
        name = str(payload["name"])
        order.append(f"{name}:start")
        if sum(item.endswith(":start") for item in order) == 2:
            both_reads_started.set()
        await release_reads.wait()
        order.append(f"{name}:end")
        return ToolResult(name)

    async def write(payload, context):  # noqa: ARG001
        order.append("write")
        return ToolResult("written")

    registry = ToolRegistry()
    for name in ("read_a", "read_b"):

        async def named_read(payload, context, *, tool_name=name):
            return await read({**payload, "name": tool_name}, context)

        registry.register(
            Tool(
                name=name,
                description=name,
                parameters=object_schema({}),
                handler=named_read,
            )
        )
    registry.register(
        Tool(
            name="write",
            description="write",
            parameters=object_schema({}),
            handler=write,
            is_read_only=False,
            is_concurrency_safe=False,
        )
    )
    agent = Agent(
        llm_client=ParallelReadClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def run():
        task = asyncio.create_task(agent.run_complete("run tools"))
        await asyncio.wait_for(both_reads_started.wait(), timeout=1)
        assert "write" not in order
        release_reads.set()
        return await task

    result = asyncio.run(run())

    assert result.text == "finished"
    assert order.index("read_a:end") < order.index("write")
    assert order.index("read_b:end") < order.index("write")


def test_duplicate_tool_call_ids_do_not_deadlock_serial_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    config.policy.hitl_mode = "never"
    order: list[str] = []

    registry = ToolRegistry()
    for name in ("write_a", "write_b"):

        async def write(payload, context, *, tool_name=name):  # noqa: ARG001
            order.append(tool_name)
            return ToolResult(tool_name)

        registry.register(
            Tool(
                name=name,
                description=name,
                parameters=object_schema({}),
                handler=write,
                is_read_only=False,
                is_concurrency_safe=False,
            )
        )
    agent = Agent(
        llm_client=DuplicateToolIdClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    result = asyncio.run(asyncio.wait_for(agent.run_complete("run tools"), timeout=1))

    assert result.text == "finished"
    assert order == ["write_a", "write_b"]
