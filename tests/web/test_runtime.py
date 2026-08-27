from __future__ import annotations

import asyncio

from vela.plan.models import ExecutionPlan, Task, TaskType
from vela.web.runtime import EventHub, serialize_agent_event


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
