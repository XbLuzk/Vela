from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from vela.llm.base import LlmClient
from vela.plan.models import ExecutionPlan, Task, TaskType
from vela.types import Message, Usage

PLANNER_PROMPT = """你是 Vela 的任务规划器。
请为用户任务创建一个简洁、可执行的 DAG，并仅返回以下结构的 JSON：
{
  "summary": "short summary",
  "tasks": [
    {
      "id": "stable_source_id",
      "description": "concrete executable step",
      "type": "FILE_READ|FILE_WRITE|COMMAND|ANALYSIS|VERIFICATION",
      "dependencies": ["stable_source_id"]
    }
  ]
}
可以并行的独立任务应放在同一执行批次中。
summary 和 description 必须使用与用户目标相同的语言；用户目标包含中文时，必须使用中文。
JSON 字段名、任务 id 和 type 枚举值保持上述英文格式。
"""


class Planner:
    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client
        self.last_usage = Usage()

    async def stream_plan(self, goal: str) -> AsyncIterator[dict[str, Any]]:
        """Create a plan while preserving provider reasoning and usage events."""
        self.last_usage = Usage()
        text = ""
        messages = [Message(role="user", content=f"请为以下目标创建执行计划：\n{goal}")]
        async for event in self.llm_client.chat(messages, [], system_prompt=PLANNER_PROMPT):
            event_type = event.get("type")
            if event_type == "text_delta":
                # Planner text is machine-readable JSON. Keep it out of the user-facing
                # stream and expose the parsed plan below instead.
                text += str(event.get("text") or "")
            elif event_type == "thinking_delta":
                yield {
                    "type": "thinking_delta",
                    "thinking": str(event.get("thinking") or ""),
                    "phase": "planning",
                }
            elif event_type == "usage":
                usage = Usage.from_mapping(event.get("usage") or {})
                self.last_usage = self.last_usage + usage
                yield {"type": "usage", "usage": usage.to_dict(), "phase": "planning"}
            elif event_type == "error":
                raise event["error"]

        yield {"type": "plan_created", "plan": self.parse_plan(goal, text)}

    def parse_plan(self, goal: str, plan_json: str) -> ExecutionPlan:
        data = _parse_json_object(plan_json)
        task_nodes = data.get("tasks") or data.get("steps") or []
        if not isinstance(task_nodes, list) or not task_nodes:
            raise ValueError("planner output did not contain a non-empty tasks/steps array")

        plan = ExecutionPlan(id=f"plan_{int(time.time() * 1000)}", goal=goal)
        plan.summary = str(data.get("summary") or "")
        id_mapping: dict[str, str] = {}

        for index, node in enumerate(task_nodes, start=1):
            if not isinstance(node, dict):
                continue
            original_id = str(node.get("id") or f"task_{index}")
            new_id = f"task_{index}"
            id_mapping[original_id] = new_id
            plan.add_task(
                Task(
                    id=new_id,
                    description=str(node.get("description") or original_id),
                    type=_parse_task_type(str(node.get("type") or "ANALYSIS")),
                )
            )

        for index, node in enumerate(task_nodes, start=1):
            if not isinstance(node, dict):
                continue
            task = plan.get_task(f"task_{index}")
            if not task:
                continue
            dependencies = node.get("dependencies") or []
            if not isinstance(dependencies, list):
                continue
            for raw_dep in dependencies:
                dep_id = id_mapping.get(str(raw_dep), str(raw_dep))
                if dep_id in plan.tasks:
                    task.add_dependency(dep_id)

        if not plan.compute_execution_order():
            raise ValueError("plan contains a cyclic dependency")
        return plan


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").replace("```", "").strip()
    if not cleaned:
        raise ValueError("empty planner output")
    return json.loads(cleaned)


def _parse_task_type(value: str) -> TaskType:
    normalized = value.upper()
    try:
        return TaskType(normalized)
    except ValueError:
        return TaskType.ANALYSIS
