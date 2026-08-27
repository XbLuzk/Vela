from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from vela.plan.models import ExecutionPlan, Task, TaskType
from vela.web import runtime as runtime_module
from vela.web.runtime import EventHub, RuntimeManager, serialize_agent_event


def test_serialize_agent_event_converts_errors_and_plan_dataclasses():
    plan = ExecutionPlan(id="plan-1", goal="ship web", summary="Web only")
    plan.add_task(Task(id="T1", description="Build UI", type=TaskType.FILE_WRITE))

    payload = serialize_agent_event(
        {
            "type": "plan_created",
            "plan": plan,
            "error": RuntimeError("boom"),
        }
    )

    assert payload["type"] == "plan_created"
    assert payload["error"] == "boom"
    assert payload["plan"]["tasks"]["T1"]["type"] == "FILE_WRITE"


def test_event_hub_fans_out_to_connected_streams():
    async def scenario():
        hub = EventHub()
        stream = hub.stream()
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)

        await hub.publish({"type": "text_delta", "text": "hello"})

        assert await pending == {"type": "text_delta", "text": "hello"}
        await stream.aclose()

    asyncio.run(scenario())


def test_pending_project_trust_starts_with_builtin_capabilities(monkeypatch, tmp_path):
    manager = RuntimeManager(tmp_path)
    rebuild = AsyncMock()
    monkeypatch.setattr(runtime_module, "has_trust_sensitive_resources", lambda _cwd: True)
    monkeypatch.setattr(manager.trust_store, "get", lambda _cwd: None)
    monkeypatch.setattr(manager, "rebuild", rebuild)

    asyncio.run(manager.initialize())

    assert manager.project_extensions_pending is True
    assert manager.project_trusted is False
    rebuild.assert_awaited_once_with()
