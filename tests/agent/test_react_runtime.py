from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vela.agent import Agent
from vela.agent.react_runtime import run_react_agent
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
        assistant = next(message for message in reversed(messages) if message.role == "assistant")
        tool_messages = [message for message in messages if message.role == "tool"]
        assert [call["id"] for call in assistant.tool_calls] == ["duplicate", "duplicate_2"]
        assert [message.tool_call_id for message in tool_messages] == [
            "duplicate",
            "duplicate_2",
        ]
        yield {"type": "text_delta", "text": "finished"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class ToolResultClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self, name: str, arguments: str = "{}", expected_result: str = ""):
        self.name = name
        self.arguments = arguments
        self.expected_result = expected_result
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": self.name, "arguments": self.arguments},
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return
        tool_message = next(message for message in reversed(messages) if message.role == "tool")
        assert tool_message.tool_call_id == "call_1"
        assert self.expected_result in str(tool_message.content)
        yield {"type": "text_delta", "text": "recovered"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class IncompleteToolClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {
            "type": "tool_call_delta",
            "tool_call": {
                "index": 0,
                "id": "call_1",
                "function": {"arguments": "{}"},
            },
        }
        yield {"type": "message_end", "stop_reason": "tool_use"}


class AmbiguousFragmentClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        for index, name in enumerate(("write_a", "write_b")):
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": index,
                    "id": f"call_{index}",
                    "function": {"name": name, "arguments": '{"value":'},
                },
            }
        yield {
            "type": "tool_call_delta",
            "tool_call": {"function": {"arguments": "1}"}},
        }
        yield {"type": "message_end", "stop_reason": "tool_use"}


class DuplicateIdRepeatClient(AmbiguousFragmentClient):
    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        for index, name in enumerate(("write_a", "write_b")):
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": index,
                    "id": "duplicate",
                    "function": {"name": name, "arguments": "{}"},
                },
            }
        yield {
            "type": "tool_call_delta",
            "tool_call": {
                "id": "duplicate",
                "function": {"name": "write_a", "arguments": "{}"},
            },
        }
        yield {"type": "message_end", "stop_reason": "tool_use"}


class SplitToolNameClient(ToolResultClient):
    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "read_", "arguments": self.arguments},
                },
            }
            yield {
                "type": "tool_call_delta",
                "tool_call": {"index": 0, "function": {"name": "file"}},
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return
        tool_message = next(message for message in reversed(messages) if message.role == "tool")
        assert tool_message.tool_call_id == "call_1"
        assert self.expected_result in str(tool_message.content)
        yield {"type": "text_delta", "text": "recovered"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


async def _collect(events):
    return [event async for event in events]


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
        "run_started",
        "turn_complete",
        "tool_call",
        "tool_result",
        "text_delta",
        "turn_complete",
        "done",
        "run_finished",
    ]
    assert events[2]["name"] == "read_file"
    assert "1: hello" in events[3]["result"]
    done = next(event for event in events if event["type"] == "done")
    assert done["total_turns"] == 2
    assert events[0]["run_id"] == events[-1]["run_id"]


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

    events = asyncio.run(_collect(agent.run("请使用 demo-skill 完成这个任务")))
    skill_result = next(event for event in events if event.get("type") == "tool_result")
    done = next(event for event in events if event["type"] == "done")
    tool_message = next(message for message in done["messages"] if message.role == "tool")

    assert "## Loaded Skill: demo-skill" in skill_result["result"]
    assert skill_result["result"] == tool_message.content
    assert "".join(str(event.get("text") or "") for event in events) == "skill applied"
    assert done["total_turns"] == 2
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
    events = asyncio.run(_collect(agent.run("new request")))
    compressed = next(event for event in events if event["type"] == "context_compressed")
    done = next(event for event in events if event["type"] == "done")

    assert "".join(str(event.get("text") or "") for event in events) == "compressed"
    assert client.saw_summary
    assert compressed["before_tokens"] > compressed["after_tokens"]
    assert compressed["summarized_messages"] > 0
    assert done["total_tokens"] == 128
    assert done["usage"]["cache_hit_tokens"] == 20
    assert done["usage"]["cache_miss_tokens"] == 100


def test_react_runtime_keeps_one_model_call_and_vela_stream_events(tmp_path, monkeypatch):
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
        "run_started",
        "thinking_delta",
        "text_delta",
        "usage",
        "turn_complete",
        "done",
        "run_finished",
    ]
    done = next(event for event in events if event["type"] == "done")
    assert done["total_turns"] == 1
    assert done["total_tokens"] == 7


def test_direct_react_runtime_returns_complete_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False

    async def run():
        events = []
        async for event in run_react_agent(
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


def test_react_runtime_preserves_parallel_reads_and_serial_write(tmp_path, monkeypatch):
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


def test_closing_after_first_tool_result_cancels_pending_concurrent_tool(tmp_path) -> None:
    class TwoReadClient:
        model_name = "fake-model"
        provider_name = "fake-provider"
        max_context_window = 1_000

        async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
            for index, name in enumerate(("fast_read", "slow_read")):
                yield {
                    "type": "tool_call_delta",
                    "tool_call": {
                        "index": index,
                        "id": f"call_{index}",
                        "function": {"name": name, "arguments": "{}"},
                    },
                }
            yield {"type": "message_end", "stop_reason": "tool_use"}

    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    config.features.audit_log = False
    slow_started = asyncio.Event()
    slow_cancelled = asyncio.Event()

    async def fast_read(payload, context):  # noqa: ARG001
        await slow_started.wait()
        return ToolResult("fast")

    async def slow_read(payload, context):  # noqa: ARG001
        slow_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            slow_cancelled.set()

    registry = ToolRegistry()
    registry.register(Tool("fast_read", "fast", object_schema({}), fast_read))
    registry.register(Tool("slow_read", "slow", object_schema({}), slow_read))
    agent = Agent(
        llm_client=TwoReadClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def close_after_result() -> None:
        stream = agent.run("read twice")
        async for event in stream:
            if event["type"] == "tool_result":
                break
        await stream.aclose()
        assert slow_cancelled.is_set()

    asyncio.run(close_after_result())


@pytest.mark.parametrize(
    ("decision", "expected_executions", "is_error", "expected_result"),
    [
        ("approve", 1, False, "written"),
        ("deny", 0, True, "denied by approval policy"),
    ],
)
def test_react_runtime_forwards_hitl_approval(
    tmp_path,
    monkeypatch,
    decision,
    expected_executions,
    is_error,
    expected_result,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    config.policy.hitl_mode = "always"
    approvals = []
    executions = 0

    async def approve(request):
        approvals.append(request)
        return decision

    async def write(payload, context):  # noqa: ARG001
        nonlocal executions
        executions += 1
        return ToolResult("written")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="write",
            description="write",
            parameters=object_schema({}),
            handler=write,
            is_read_only=False,
            requires_approval=True,
        )
    )
    agent = Agent(
        llm_client=ToolResultClient("write", expected_result=expected_result),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
        approval_callback=approve,
    )

    events = asyncio.run(_collect(agent.run("write once")))
    result = next(event for event in events if event["type"] == "tool_result")

    assert result["is_error"] is is_error
    assert executions == expected_executions
    assert approvals == [
        {
            "tool_name": "write",
            "input": {},
            "danger_level": "safe",
            "description": "write",
        }
    ]


def test_unknown_tool_with_malformed_input_returns_error_and_model_recovers(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    agent = Agent(
        llm_client=ToolResultClient(
            "missing_tool",
            arguments="{bad json",
            expected_result='Tool "missing_tool" not found',
        ),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
    )

    events = asyncio.run(_collect(agent.run("recover from bad tool")))
    call = next(event for event in events if event["type"] == "tool_call")
    result = next(event for event in events if event["type"] == "tool_result")
    done = next(event for event in events if event["type"] == "done")

    assert call["input"] == {"raw": "{bad json"}
    assert result["is_error"] is True
    assert done["messages"][-2].tool_call_id == "call_1"
    assert "".join(str(event.get("text") or "") for event in events) == "recovered"


def test_split_tool_name_fragments_are_reassembled(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    client = SplitToolNameClient(
        "read_file",
        arguments='{"path":"note.txt"}',
        expected_result="1: hello",
    )
    agent = Agent(llm_client=client, tool_registry=registry, config=config, cwd=str(tmp_path))

    result = asyncio.run(agent.run_complete("read note"))

    assert result.text == "recovered"


def test_ambiguous_unindexed_tool_fragment_fails_before_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    config.policy.hitl_mode = "never"
    executions = []

    async def write(payload, context):  # noqa: ARG001
        executions.append(payload)
        return ToolResult("written")

    registry = ToolRegistry()
    for name in ("write_a", "write_b"):
        registry.register(
            Tool(
                name=name,
                description=name,
                parameters=object_schema({}),
                handler=write,
                is_read_only=False,
            )
        )
    agent = Agent(
        llm_client=AmbiguousFragmentClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="Ambiguous tool-call fragment"):
        asyncio.run(agent.run_complete("do not misroute writes"))

    assert executions == []


def test_duplicate_id_repeat_fails_before_extra_write(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    config.policy.hitl_mode = "never"
    executions = []

    async def write(payload, context):  # noqa: ARG001
        executions.append(payload)
        return ToolResult("written")

    registry = ToolRegistry()
    for name in ("write_a", "write_b"):
        registry.register(
            Tool(
                name=name,
                description=name,
                parameters=object_schema({}),
                handler=write,
                is_read_only=False,
            )
        )
    agent = Agent(
        llm_client=DuplicateIdRepeatClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="duplicate id"):
        asyncio.run(agent.run_complete("do not duplicate writes"))

    assert executions == []


def test_incomplete_tool_stream_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    agent = Agent(
        llm_client=IncompleteToolClient(),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="incomplete tool-call stream"):
        asyncio.run(agent.run_complete("must not report success"))


def test_completed_tool_result_is_persisted_before_event_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    config.policy.hitl_mode = "never"

    async def write(payload, context):  # noqa: ARG001
        return ToolResult("written")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="write",
            description="write",
            parameters=object_schema({}),
            handler=write,
            is_read_only=False,
        )
    )
    agent = Agent(
        llm_client=ToolResultClient("write", expected_result="written"),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def stop_after_result():
        stream = agent.run("write once")
        async for event in stream:
            if event["type"] == "tool_result":
                await stream.aclose()
                return

    asyncio.run(stop_after_result())

    tool_message = next(message for message in agent.history if message.role == "tool")
    assert tool_message.content == "written"
    assert tool_message.tool_call_id == "call_1"


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


def test_react_loop_fails_explicitly_at_the_model_turn_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    config.policy.hitl_mode = "never"
    calls: list[str] = []

    async def write(payload, context):  # noqa: ARG001
        calls.append("write")
        return ToolResult("written")

    registry = ToolRegistry()
    for name in ("write_a", "write_b"):
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
        max_turns=1,
    )

    with pytest.raises(RuntimeError, match="model turn limit"):
        asyncio.run(agent.run_complete("run tools forever"))

    assert calls == []
    assistant = next(message for message in agent.history if message.role == "assistant")
    tool_messages = [message for message in agent.history if message.role == "tool"]
    assert len(assistant.tool_calls) == 2
    assert [message.tool_call_id for message in tool_messages] == [
        "duplicate",
        "duplicate_2",
    ]
    assert all("was not executed" in str(message.content) for message in tool_messages)

    result = asyncio.run(agent.run_complete("continue after the limit"))

    assert result.text == "finished"


@pytest.mark.parametrize("max_turns", [0, -1])
def test_react_loop_rejects_non_positive_turn_limits(tmp_path, monkeypatch, max_turns):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.context_compression = False
    client = SingleTurnStreamingClient()
    with pytest.raises(ValueError, match="max_turns must be at least 1"):
        Agent(
            llm_client=client,
            tool_registry=ToolRegistry(),
            config=config,
            cwd=str(tmp_path),
            max_turns=max_turns,
        )

    assert client.calls == 0
