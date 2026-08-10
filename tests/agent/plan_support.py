from __future__ import annotations

import asyncio
from typing import Any

from vela.tools import ToolRegistry
from vela.tools.base import Tool, ToolResult, object_schema


class FakeClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {"type": "text_delta", "text": "{}"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class TwoToolClient(FakeClient):
    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        for index, name in enumerate(("first_tool", "second_tool")):
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": index,
                    "id": f"call_{index + 1}",
                    "function": {"name": name, "arguments": "{}"},
                },
            }
        yield {"type": "message_end", "stop_reason": "tool_use"}


def _two_tool_registry(second_started: asyncio.Event) -> ToolRegistry:
    async def first_tool(payload, context):  # noqa: ARG001
        return ToolResult("first completed")

    async def second_tool(payload, context):  # noqa: ARG001
        second_started.set()
        await asyncio.Event().wait()
        return ToolResult("unreachable")

    registry = ToolRegistry()
    for name, handler in (("first_tool", first_tool), ("second_tool", second_tool)):
        registry.register(
            Tool(
                name=name,
                description=name,
                parameters=object_schema({}),
                handler=handler,
                is_read_only=False,
            )
        )
    return registry


async def _consume(events) -> None:
    async for _ in events:
        pass


async def _collect(events) -> list[dict[str, Any]]:
    return [event async for event in events]


class ParallelPlanClient(FakeClient):
    def __init__(self):
        self.current_concurrency = 0
        self.peak_concurrency = 0
        self.ready = asyncio.Event()
        self.task_system_prompts: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            yield {"type": "thinking_delta", "thinking": "先拆分可并行任务"}
            yield {
                "type": "text_delta",
                "text": (
                    '{"summary":"parallel","tasks":['
                    '{"id":"a","description":"任务 A","type":"ANALYSIS","dependencies":[]},'
                    '{"id":"b","description":"任务 B","type":"ANALYSIS","dependencies":[]}'
                    "]}"
                ),
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        if "任务 A" in body or "任务 B" in body:
            self.task_system_prompts.append(system_prompt)
            self.current_concurrency += 1
            self.peak_concurrency = max(self.peak_concurrency, self.current_concurrency)
            if self.current_concurrency == 2:
                self.ready.set()
            await asyncio.wait_for(self.ready.wait(), timeout=2)
            self.current_concurrency -= 1
            text = "A 的结果" if "任务 A" in body else "B 的结果"
            yield {"type": "thinking_delta", "thinking": f"分析{text}"}
            yield {"type": "text_delta", "text": text}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        yield {"type": "text_delta", "text": "fallback"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class ReviewPlanClient(ParallelPlanClient):
    def __init__(self):
        super().__init__()
        self.plan_requests = 0
        self.task_requests = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            self.plan_requests += 1
            yield {
                "type": "text_delta",
                "text": (
                    '{"summary":"review","tasks":['
                    '{"id":"a","description":"验证任务","type":"VERIFICATION",'
                    '"dependencies":[]}]}'
                ),
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        self.task_requests += 1
        yield {"type": "text_delta", "text": "验证完成"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class ResumePlanClient(FakeClient):
    def __init__(
        self,
        *,
        second_started: asyncio.Event | None = None,
        block_second: bool = False,
    ) -> None:
        self.second_started = second_started
        self.block_second = block_second
        self.plan_requests = 0
        self.task_requests: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            self.plan_requests += 1
            yield {
                "type": "text_delta",
                "text": (
                    '{"summary":"resume","tasks":['
                    '{"id":"first","description":"任务一","type":"ANALYSIS",'
                    '"dependencies":[]},'
                    '{"id":"second","description":"任务二","type":"VERIFICATION",'
                    '"dependencies":["first"]}]}'
                ),
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        task_name = "任务一" if "当前任务 [task_1]" in body else "任务二"
        self.task_requests.append(task_name)
        if task_name == "任务二" and self.second_started is not None:
            self.second_started.set()
        if task_name == "任务二" and self.block_second:
            await asyncio.Event().wait()
        yield {"type": "text_delta", "text": f"{task_name}完成"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class ParallelCheckpointClient(FakeClient):
    def __init__(
        self,
        *,
        block_second: bool,
        second_started: asyncio.Event | None = None,
    ) -> None:
        self.block_second = block_second
        self.second_started = second_started
        self.plan_requests = 0
        self.task_requests: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            self.plan_requests += 1
            yield {
                "type": "text_delta",
                "text": (
                    '{"summary":"parallel checkpoint","tasks":['
                    '{"id":"first","description":"任务一","type":"ANALYSIS",'
                    '"dependencies":[]},'
                    '{"id":"second","description":"任务二","type":"ANALYSIS",'
                    '"dependencies":[]}]}'
                ),
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        task_name = "任务一" if "当前任务 [task_1]" in body else "任务二"
        self.task_requests.append(task_name)
        if task_name == "任务二" and self.second_started is not None:
            self.second_started.set()
        if task_name == "任务二" and self.block_second:
            await asyncio.Event().wait()
        yield {"type": "text_delta", "text": f"{task_name}完成"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class SingleTaskClient(FakeClient):
    def __init__(self, description: str) -> None:
        self.description = description
        self.plan_requests = 0
        self.task_requests: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            self.plan_requests += 1
            yield {
                "type": "text_delta",
                "text": (
                    '{"summary":"fresh","tasks":['
                    f'{{"id":"only","description":"{self.description}",'
                    '"type":"ANALYSIS","dependencies":[]}]} '
                ),
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        self.task_requests.append(self.description)
        yield {"type": "text_delta", "text": f"{self.description}完成"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class FailingTaskClient(SingleTaskClient):
    def __init__(self) -> None:
        super().__init__("失败任务")

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            async for event in super().chat(messages, tools, system_prompt=system_prompt):
                yield event
            return
        self.task_requests.append(self.description)
        raise RuntimeError("simulated task failure")


class JournalResumeClient(FakeClient):
    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        if messages[-1].role == "tool":
            yield {"type": "text_delta", "text": "两个工具均已完成"}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        for index, name in enumerate(("first_tool", "second_tool")):
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": index,
                    "id": f"call_{index + 1}",
                    "function": {"name": name, "arguments": "{}"},
                },
            }
        yield {"type": "message_end", "stop_reason": "tool_use"}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)
