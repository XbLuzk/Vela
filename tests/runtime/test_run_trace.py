from __future__ import annotations

import asyncio
import json
import os

import pytest
from filelock import FileLock

import vela.agent.agent as agent_module
from vela.agent import Agent
from vela.config import load_config
from vela.events import AgentEvent
from vela.run_trace import RunTraceStore, RunTracker
from vela.tools import ToolRegistry


def test_tracker_emits_lifecycle_and_persists_completed_summary(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    tracker = _tracker(store)

    events = asyncio.run(
        _collect(
            tracker.stream(
                _events(
                    {"type": "tool_call", "name": "read_file", "input": {}},
                    {
                        "type": "tool_result",
                        "name": "read_file",
                        "result": "ok",
                        "replayed": True,
                    },
                    {
                        "type": "done",
                        "total_turns": 2,
                        "total_tokens": 30,
                        "usage": _usage(20, 10),
                    },
                )
            )
        )
    )

    assert [event["type"] for event in events] == [
        "run_started",
        "tool_call",
        "tool_result",
        "done",
        "run_finished",
    ]
    assert {event["run_id"] for event in events} == {tracker.trace.run_id}
    trace = store.list(limit=1)[0]
    assert trace["status"] == "completed"
    assert trace["turns"] == 2
    assert trace["usage"]["total_tokens"] == 30
    assert trace["tool_calls"] == 1
    assert trace["replayed_tools"] == 1
    assert trace["finished_at"]


def test_error_trace_is_settled_before_consumer_closes_stream(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    tracker = _tracker(store)

    async def consume_until_error() -> None:
        stream = tracker.stream(
            _events({"type": "error", "error": RuntimeError("provider failed")})
        )
        assert (await anext(stream))["type"] == "run_started"
        assert (await anext(stream))["type"] == "error"
        await stream.aclose()

    asyncio.run(consume_until_error())

    trace = store.list(limit=1)[0]
    assert trace["status"] == "failed"
    assert trace["error"] == "RuntimeError"


def test_failed_trace_keeps_usage_seen_before_error(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    tracker = _tracker(store)

    asyncio.run(
        _collect(
            tracker.stream(
                _events(
                    {"type": "usage", "usage": _usage(10, 5)},
                    {"type": "error", "error": RuntimeError("provider failed")},
                )
            )
        )
    )

    trace = store.list(limit=1)[0]
    assert trace["status"] == "failed"
    assert trace["usage"]["total_tokens"] == 15


def test_failed_plan_done_event_persists_failed_status(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    tracker = RunTracker(
        mode="plan",
        model="fake-model",
        provider="fake-provider",
        cwd="/tmp/project",
        store=store,
    )

    events = asyncio.run(
        _collect(
            tracker.stream(
                _events(
                    {
                        "type": "done",
                        "total_turns": 1,
                        "usage": _usage(5, 2),
                        "langgraph": {"thread_id": "thread_1", "status": "failed"},
                    }
                )
            )
        )
    )

    assert events[-1]["status"] == "failed"
    assert store.list(limit=1)[0]["status"] == "failed"
    assert store.list(limit=1)[0]["error"] == "RuntimeError"


def test_cancelled_stream_persists_cancelled_trace(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    tracker = _tracker(store)
    blocked = asyncio.Event()

    async def never_finishes():
        blocked.set()
        await asyncio.Event().wait()
        yield {"type": "done"}  # pragma: no cover

    async def cancel() -> None:
        task = asyncio.create_task(_collect(tracker.stream(never_finishes())))
        await blocked.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())

    assert store.list(limit=1)[0]["status"] == "cancelled"


def test_closing_immediately_after_run_started_persists_cancelled_trace(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    tracker = _tracker(store)

    async def close_after_start() -> None:
        stream = tracker.stream(_events({"type": "done"}))
        assert (await anext(stream))["type"] == "run_started"
        await stream.aclose()

    asyncio.run(close_after_start())

    assert store.list(limit=1)[0]["status"] == "cancelled"


def test_closing_public_agent_stream_settles_trace_before_aclose_returns(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.skill = False
    config.features.memory = False
    config.features.context_compression = False
    agent = Agent(
        llm_client=_DoneClient(),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
        trace_store=store,
    )

    async def close_after_start() -> None:
        stream = agent.run("hello")
        assert (await anext(stream))["type"] == "run_started"
        await stream.aclose()
        assert agent.last_run_trace is not None
        assert agent.last_run_trace.status == "cancelled"
        assert store.list(limit=1)[0]["status"] == "cancelled"

    asyncio.run(close_after_start())


def test_run_complete_exposes_trace_write_failure_before_returning(tmp_path) -> None:
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.skill = False
    config.features.memory = False
    config.features.context_compression = False
    agent = Agent(
        llm_client=_ErrorClient(),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
        trace_store=_BrokenStore(tmp_path / "runs.jsonl"),
    )

    async def run_and_check() -> None:
        with pytest.raises(RuntimeError, match="provider failed"):
            await agent.run_complete("hello")
        assert agent.last_run_trace is not None
        assert agent.last_run_trace.status == "failed"
        assert agent.last_run_trace_warning == "Run trace was not saved: disk full"

    asyncio.run(run_and_check())


def test_stream_without_terminal_event_fails_explicitly(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    tracker = _tracker(store)

    events = asyncio.run(_collect(tracker.stream(_events({"type": "text_delta", "text": "x"}))))

    assert [event["type"] for event in events] == [
        "run_started",
        "text_delta",
        "error",
        "run_finished",
    ]
    assert events[-1]["status"] == "failed"
    assert store.list(limit=1)[0]["error"] == "RuntimeError"


def test_trace_store_skips_bad_lines_and_resolves_index_or_prefix(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    first = _tracker(store)
    second = _tracker(store)
    asyncio.run(first._finish("completed"))  # noqa: SLF001 - focused persistence contract test
    asyncio.run(second._finish("failed", RuntimeError("boom")))  # noqa: SLF001
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")

    assert [trace["run_id"] for trace in store.list(limit=2)] == [
        second.trace.run_id,
        first.trace.run_id,
    ]
    assert store.find("1")["run_id"] == second.trace.run_id
    assert store.find(first.trace.run_id[:10])["run_id"] == first.trace.run_id


def test_trace_store_skips_invalid_utf8_and_schema_corrupt_records(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    first = _tracker(store)
    second = _tracker(store)
    asyncio.run(first._finish("completed"))  # noqa: SLF001
    with store.path.open("ab") as handle:
        handle.write(b"\xff\n")
        handle.write(b'{"run_id":"run_corrupt","usage":"bad","duration_ms":"bad"}\n')
    asyncio.run(second._finish("completed"))  # noqa: SLF001

    assert [trace["run_id"] for trace in store.list(limit=10)] == [
        second.trace.run_id,
        first.trace.run_id,
    ]


def test_trace_store_limit_reads_only_the_file_tail(tmp_path, monkeypatch) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    for _ in range(200):
        tracker = _tracker(store)
        asyncio.run(tracker._finish("completed"))  # noqa: SLF001
    file_size = store.path.stat().st_size
    original_open = type(store.path).open
    bytes_read = 0

    class CountingFile:
        def __init__(self, handle):
            self._handle = handle

        def read(self, size=-1):
            nonlocal bytes_read
            data = self._handle.read(size)
            bytes_read += len(data)
            return data

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

    def counting_open(path, *args, **kwargs):
        return CountingFile(original_open(path, *args, **kwargs))

    monkeypatch.setattr(type(store.path), "open", counting_open)

    assert len(store.list(limit=1)) == 1
    assert bytes_read < file_size


def test_trace_store_exposes_read_failure(tmp_path, monkeypatch) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    tracker = _tracker(store)
    asyncio.run(tracker._finish("completed"))  # noqa: SLF001

    def fail_open(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(type(store.path), "open", fail_open)

    assert store.list(limit=1) == []
    assert store.last_warning == "Run traces could not be read: disk unavailable"


def test_failed_partial_append_does_not_corrupt_next_trace(tmp_path, monkeypatch) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    original_write = os.write
    calls = 0

    def fail_after_partial_write(descriptor, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            partial = bytes(payload[: max(1, len(payload) // 2)])
            return original_write(descriptor, partial)
        raise OSError("disk full")

    monkeypatch.setattr(os, "write", fail_after_partial_write)
    failed = _tracker(store)
    with pytest.raises(OSError, match="disk full"):
        store.append(failed.trace)

    monkeypatch.setattr(os, "write", original_write)
    healthy = _tracker(store)
    asyncio.run(healthy._finish("completed"))  # noqa: SLF001

    assert [trace["run_id"] for trace in store.list(limit=10)] == [healthy.trace.run_id]


def test_unterminated_old_tail_does_not_swallow_next_trace(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    store.path.write_bytes(b'{"truncated":')
    healthy = _tracker(store)
    asyncio.run(healthy._finish("completed"))  # noqa: SLF001

    assert [trace["run_id"] for trace in store.list(limit=10)] == [healthy.trace.run_id]


def test_trace_store_lock_contention_does_not_block_event_loop(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    lock = FileLock(f"{store.path}.lock")
    tracker = _tracker(store)

    async def finish_after_contention() -> None:
        lock.acquire()
        task = asyncio.create_task(tracker._finish("completed"))  # noqa: SLF001
        await asyncio.sleep(0.01)
        assert not task.done()
        lock.release()
        await task

    asyncio.run(finish_after_contention())

    assert store.list(limit=1)[0]["run_id"] == tracker.trace.run_id


def test_oversized_corrupt_tail_has_bounded_scan(tmp_path, monkeypatch) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    store.path.write_bytes(b"x" * (4 * 1024 * 1024))
    original_open = type(store.path).open
    bytes_read = 0

    class CountingFile:
        def __init__(self, handle):
            self._handle = handle

        def read(self, size=-1):
            nonlocal bytes_read
            data = self._handle.read(size)
            bytes_read += len(data)
            return data

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

    monkeypatch.setattr(
        type(store.path),
        "open",
        lambda path, *args, **kwargs: CountingFile(original_open(path, *args, **kwargs)),
    )

    assert store.list(limit=1) == []
    assert bytes_read <= 2 * 1024 * 1024
    assert "scan limit reached" in str(store.last_warning)


def test_exact_run_id_searches_beyond_bounded_list_scan(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    old = _tracker(store)
    asyncio.run(old._finish("completed"))  # noqa: SLF001
    with store.path.open("ab") as handle:
        handle.write(b"x" * (3 * 1024 * 1024) + b"\n")

    assert store.list(limit=1) == []
    assert "scan limit reached" in str(store.last_warning)
    assert store.find(old.trace.run_id)["run_id"] == old.trace.run_id
    assert store.last_warning is None


def test_numeric_twelve_character_run_id_is_not_treated_as_list_index(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    tracker = _tracker(store)
    tracker.trace.run_id = "run_123456789012"
    asyncio.run(tracker._finish("completed"))  # noqa: SLF001

    assert store.find("123456789012")["run_id"] == "run_123456789012"


@pytest.mark.parametrize(
    "message",
    [
        "api_key=secret-value",
        "access_token=secret-value",
        "Cookie: session=secret-value",
        "credential=secret-value",
        "X-Amz-Signature=secret-value",
        "sig=secret-value",
        "eyJhbGciOiJIUzI1NiJ9.secret-value.signature",
        "-----BEGIN PRIVATE KEY----- secret-value",
        "https://example.test/?code=secret-value",
    ],
)
def test_sensitive_exception_text_is_not_persisted(tmp_path, message) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    tracker = _tracker(store)
    asyncio.run(tracker._finish("failed", RuntimeError(message)))  # noqa: SLF001

    serialized = store.path.read_text(encoding="utf-8")
    assert "secret-value" not in serialized
    assert json.loads(serialized)["error"] == "RuntimeError"


def _tracker(store: RunTraceStore) -> RunTracker:
    return RunTracker(
        mode="react",
        model="fake-model",
        provider="fake-provider",
        cwd="/tmp/project",
        session_id="session_1",
        store=store,
    )


async def _events(*events: AgentEvent):
    for event in events:
        yield event


async def _collect(events):
    return [event async for event in events]


def _usage(input_tokens: int, output_tokens: int):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": input_tokens + output_tokens,
    }


class _DoneClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1_000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_closing_public_agent_stream_closes_child_runtime(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.skill = False
    config.features.memory = False
    config.features.context_compression = False
    client = _ClosableClient()
    agent = Agent(
        llm_client=client,
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
        trace_store=store,
    )

    async def close_after_model_event() -> None:
        stream = agent.run("hello")
        assert (await anext(stream))["type"] == "run_started"
        assert (await anext(stream))["type"] == "text_delta"
        await stream.aclose()
        assert client.closed
        assert agent.last_run_trace is not None
        assert agent.last_run_trace.status == "cancelled"

    asyncio.run(close_after_model_event())


def test_cleanup_failure_keeps_trace_and_surfaces_warning(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.skill = False
    config.features.memory = False
    config.features.context_compression = False
    agent = Agent(
        llm_client=_CleanupErrorClient(),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
        trace_store=store,
    )

    async def close_after_model_event() -> None:
        stream = agent.run("hello")
        assert (await anext(stream))["type"] == "run_started"
        assert (await anext(stream))["type"] == "text_delta"
        await stream.aclose()
        assert agent.last_run_trace is not None
        assert agent.last_run_trace.status == "cancelled"
        assert "cleanup failed" in str(agent.last_run_trace_warning)

    asyncio.run(close_after_model_event())


def test_cleanup_failure_does_not_replace_task_cancellation(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.skill = False
    config.features.memory = False
    config.features.context_compression = False
    client = _CleanupErrorClient()
    agent = Agent(
        llm_client=client,
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
        trace_store=store,
    )

    async def cancel_during_model_stream() -> None:
        task = asyncio.create_task(agent.run_complete("hello"))
        await client.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert agent.last_run_trace is not None
        assert agent.last_run_trace.status == "cancelled"
        assert "cancellation cleanup failed: RuntimeError" in str(agent.last_run_trace_warning)

    asyncio.run(cancel_during_model_stream())


def test_closing_plan_stream_closes_plan_runtime(tmp_path, monkeypatch) -> None:
    store = RunTraceStore(tmp_path / "runs.jsonl")
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.features.skill = False
    config.features.memory = False
    config.features.context_compression = False
    plan_runtime = _ClosablePlanRuntime()
    monkeypatch.setattr(agent_module, "LangGraphPlanAgent", lambda **_kwargs: plan_runtime)
    agent = Agent(
        llm_client=_DoneClient(),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
        mode="plan",
        trace_store=store,
    )

    async def close_after_plan_event() -> None:
        stream = agent.run("hello")
        assert (await anext(stream))["type"] == "run_started"
        assert (await anext(stream))["type"] == "text_delta"
        await stream.aclose()
        assert plan_runtime.closed
        assert agent.last_run_trace is not None
        assert agent.last_run_trace.status == "cancelled"

    asyncio.run(close_after_plan_event())


class _ClosableClient(_DoneClient):
    def __init__(self) -> None:
        self.closed = False

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        try:
            yield {"type": "text_delta", "text": "partial"}
            await asyncio.Event().wait()
        finally:
            self.closed = True


class _CleanupErrorClient(_DoneClient):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        try:
            yield {"type": "text_delta", "text": "partial"}
            self.started.set()
            await asyncio.Event().wait()
        finally:
            raise RuntimeError("cleanup failed")


class _ErrorClient(_DoneClient):
    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {"type": "error", "error": RuntimeError("provider failed")}


class _BrokenStore(RunTraceStore):
    def append(self, trace):  # noqa: ARG002
        raise OSError("disk full")


class _ClosablePlanRuntime:
    def __init__(self) -> None:
        self.closed = False
        self.history = []

    async def run(self, message):  # noqa: ARG002
        try:
            yield {"type": "text_delta", "text": "partial"}
            await asyncio.Event().wait()
        finally:
            self.closed = True
