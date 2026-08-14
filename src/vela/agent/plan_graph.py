from __future__ import annotations

import asyncio
import operator
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import aiosqlite
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Overwrite, Send, interrupt

from vela.agent.react_runtime import run_react_agent
from vela.config import VelaConfig
from vela.events import AgentEvent
from vela.llm.base import LlmClient
from vela.plan import ExecutionPlan, Planner, Task, TaskStatus, TaskType
from vela.prompt import PromptAssembler
from vela.session_history import bounded_tool_transcript
from vela.skill import SkillContextBuffer
from vela.task_control import (
    PlanReviewAction,
    PlanReviewDecision,
    TaskCancelledError,
    resolve_plan_review,
)
from vela.tools.journal import ToolExecutionJournal
from vela.tools.registry import ToolRegistry
from vela.types import Message, Usage, UsagePayload


class PlanGraphState(TypedDict, total=False):
    goal: str
    planning_goal: str
    plan: dict[str, Any]
    task_results: Annotated[list[dict[str, Any]], operator.add]
    usage_events: Annotated[list[UsagePayload], operator.add]
    review_required: bool
    execution_started: bool
    status: str
    final_text: str


class PlanTaskState(TypedDict):
    goal: str
    plan: dict[str, Any]
    task: dict[str, Any]
    task_results: list[dict[str, Any]]


PlanRoute = Literal["plan", "dispatch", "cancel"]
PlanEventWriter = Callable[[Any], None]


@dataclass(slots=True)
class _TaskRun:
    text_parts: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    error: str = ""

    @property
    def result_text(self) -> str:
        return "".join(self.text_parts).strip() or "\n".join(self.tool_results).strip()

    @property
    def status(self) -> str:
        return "failed" if self.error else "completed"


class LangGraphPlanAgent:
    """The sole Plan-and-Execute engine, backed by a persistent LangGraph."""

    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: VelaConfig,
        cwd: str,
        approval_callback=None,
        planner: Planner | None = None,
        max_task_turns: int = 8,
        plan_review_callback: (
            Callable[[ExecutionPlan], PlanReviewDecision | Awaitable[PlanReviewDecision]] | None
        ) = None,
        thread_id: str | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        if max_task_turns < 1:
            raise ValueError("max_task_turns must be at least 1.")
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.planner = planner or Planner(llm_client)
        self.max_task_turns = max_task_turns
        self.plan_review_callback = plan_review_callback
        self._persistent = thread_id is not None or checkpoint_path is not None
        self.thread_id = thread_id or str(uuid.uuid4())
        self.checkpoint_path = Path(
            checkpoint_path or Path.home() / ".vela" / "langgraph" / "checkpoints.sqlite"
        ).expanduser()
        self.resume = resume
        self.history: list[Message] = []
        self._allow_uncertain_tool_retry = False

    async def run(self, message: str) -> AsyncIterator[AgentEvent]:
        self.history = [Message(role="user", content=message or "继续之前的计划")]
        try:
            async with _open_checkpointer(
                self.checkpoint_path if self._persistent else None
            ) as checkpointer:
                graph = self._build_graph(checkpointer)
                graph_config = {
                    "configurable": {"thread_id": self.thread_id},
                    "recursion_limit": 100,
                }
                graph_input, pending_plan, pending_count = await self._prepare_graph_input(
                    graph,
                    graph_config,
                    message,
                )
                if pending_plan is not None:
                    yield {
                        "type": "text_delta",
                        "text": (
                            "检测到未完成的执行节点。该节点中的工具可能已经产生部分副作用，"
                            "继续时会重放已完成的工具，并仅重试无法对账的不确定调用；"
                            "请确认后再恢复。\n\n"
                        ),
                    }
                    yield {
                        "type": "plan_resume_warning",
                        "pending_tasks": pending_count,
                    }
                    await self._confirm_resume(pending_plan)
                stream = self._stream_graph(graph, graph_input, graph_config)
                try:
                    async for event in stream:
                        yield event
                finally:
                    await stream.aclose()
                saved = await graph.aget_state(graph_config)
                yield self._finish_graph(dict(saved.values))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve the shared streaming error contract
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise asyncio.CancelledError from exc
            yield {"type": "error", "error": exc}

    async def _prepare_graph_input(
        self,
        graph: Any,
        graph_config: dict[str, Any],
        message: str,
    ) -> tuple[dict[str, Any] | Command | None, ExecutionPlan | None, int]:
        if not self.resume:
            return self._fresh_graph_input(message), None, 0

        saved = await graph.aget_state(graph_config)
        if not saved.next:
            raise ValueError("当前 Session 没有可恢复的 LangGraph Plan")

        pending_execution = [task for task in saved.tasks if task.name == "execute_task"]
        if not pending_execution:
            return None, None, 0

        plan = _plan_from_payload(dict(saved.values).get("plan") or {})
        return None, plan, len(pending_execution)

    async def _confirm_resume(self, plan: ExecutionPlan) -> None:
        if self.plan_review_callback is None:
            raise ValueError("恢复未完成的执行节点需要显式确认回调")
        decision = await resolve_plan_review(self.plan_review_callback, plan)
        if decision.action == PlanReviewAction.CANCEL:
            raise TaskCancelledError("计划恢复已取消")
        if decision.action == PlanReviewAction.MODIFY:
            raise ValueError("执行中的计划不能直接修改；请取消后创建新计划")
        self._allow_uncertain_tool_retry = True

    def _fresh_graph_input(self, message: str) -> dict[str, Any]:
        return {
            "goal": message,
            "planning_goal": message,
            "plan": {},
            "task_results": Overwrite([]),
            "usage_events": Overwrite([]),
            "review_required": self.plan_review_callback is not None,
            "execution_started": False,
            "status": "planning",
            "final_text": "",
        }

    async def _stream_graph(
        self,
        graph: Any,
        graph_input: dict[str, Any] | Command | None,
        graph_config: dict[str, Any],
    ) -> AsyncIterator[AgentEvent]:
        while True:
            interruption: dict[str, Any] | None = None
            events = graph.astream(
                graph_input,
                graph_config,
                stream_mode=["custom", "updates"],
            )
            try:
                async for mode, chunk in events:
                    if mode == "custom":
                        yield chunk
                        continue
                    interrupts = chunk.get("__interrupt__") if isinstance(chunk, dict) else None
                    if interrupts:
                        interruption = dict(interrupts[0].value)
            finally:
                await events.aclose()
            if interruption is None:
                return

            yield {"type": "plan_review", "interrupt": interruption}
            plan = _plan_from_payload(interruption["plan"])
            if self.plan_review_callback is None:
                raise ValueError("恢复待确认的计划需要显式确认回调")
            decision = await resolve_plan_review(self.plan_review_callback, plan)
            graph_input = Command(resume=_decision_payload(decision))

    def _finish_graph(self, values: dict[str, Any]) -> AgentEvent:
        final_text = str(values.get("final_text") or "")
        if final_text:
            self.history.append(Message(role="assistant", content=final_text))
        self._delete_terminal_journal(values)
        if values.get("status") == "cancelled":
            raise TaskCancelledError(final_text or "计划已取消")

        usage = _sum_usage(values.get("usage_events") or [])
        turns = sum(int(result.get("turns") or 0) for result in values.get("task_results") or [])
        return {
            "type": "done",
            "total_turns": turns,
            "total_tokens": usage.total_tokens,
            "usage": usage.to_dict(),
            "messages": self.history,
            "langgraph": {
                "thread_id": self.thread_id,
                "status": values.get("status"),
            },
        }

    def _delete_terminal_journal(self, values: dict[str, Any]) -> None:
        terminal = values.get("status") in {"cancelled", "completed", "failed"}
        if not self._persistent or not terminal:
            return
        plan_id = str(dict(values.get("plan") or {}).get("id") or "")
        journal_path = Path(self.config.tools.execution_journal_path).expanduser()
        if plan_id and journal_path.exists():
            with suppress(Exception):
                ToolExecutionJournal(journal_path).delete_scope_prefix(
                    f"{self.thread_id}:{plan_id}:"
                )

    def _build_graph(self, checkpointer: AsyncSqliteSaver | InMemorySaver):
        builder = StateGraph(PlanGraphState)
        builder.add_node("plan", self._plan_node)
        builder.add_node(
            "review",
            self._review_node,
            destinations=("plan", "dispatch", "cancel"),
        )
        builder.add_node("dispatch", self._dispatch_node)
        builder.add_node("execute_task", self._execute_task_node)
        builder.add_node("finalize", self._finalize_node)
        builder.add_node("cancel", self._cancel_node)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "review")
        builder.add_conditional_edges("dispatch", self._route_ready_tasks)
        builder.add_edge("execute_task", "dispatch")
        builder.add_edge("finalize", END)
        builder.add_edge("cancel", END)
        return builder.compile(checkpointer=checkpointer)

    async def _plan_node(self, state: PlanGraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        planning_goal = state.get("planning_goal") or state["goal"]
        writer({"type": "text_delta", "text": f"正在规划任务：{planning_goal}\n\n"})
        writer({"type": "plan_status", "phase": "planning"})
        plan: ExecutionPlan | None = None
        usage_events: list[UsagePayload] = []
        async for event in self.planner.stream_plan(planning_goal):
            if event.get("type") == "plan_created":
                plan = event["plan"]
                continue
            if event.get("type") == "usage":
                usage_events.append(dict(event.get("usage") or {}))
            writer(event)
        if plan is None:
            raise ValueError("planner did not produce an execution plan")
        writer({"type": "text_delta", "text": plan.summarize() + "\n\n"})
        return {
            "plan": _plan_to_payload(plan),
            "task_results": Overwrite([]),
            "usage_events": usage_events,
            "execution_started": False,
            "status": "planning",
            "final_text": "",
        }

    def _review_node(self, state: PlanGraphState) -> Command[PlanRoute]:
        if not state.get("review_required"):
            return Command(goto="dispatch")
        decision = interrupt(
            {
                "kind": "plan_review",
                "thread_id": self.thread_id,
                "plan": state["plan"],
            }
        )
        action = PlanReviewAction(str(decision.get("action") or ""))
        feedback = str(decision.get("feedback") or "").strip()
        if action == PlanReviewAction.MODIFY:
            if not feedback:
                raise ValueError("plan modification requires feedback")
            planning_goal = f"{state['goal']}\n\n用户对执行计划的修改要求：\n{feedback}"
            get_stream_writer()(
                {
                    "type": "text_delta",
                    "text": f"正在按要求修改计划：{feedback}\n\n",
                }
            )
            return Command(update={"planning_goal": planning_goal}, goto="plan")
        if action == PlanReviewAction.CANCEL:
            return Command(update={"status": "cancelled"}, goto="cancel")
        return Command(goto="dispatch")

    def _dispatch_node(self, state: PlanGraphState) -> dict[str, Any]:
        if not state.get("execution_started"):
            writer = get_stream_writer()
            writer({"type": "text_delta", "text": "开始执行计划……\n\n"})
            writer({"type": "plan_status", "phase": "execution"})
        return {"execution_started": True, "status": "running"}

    def _route_ready_tasks(self, state: PlanGraphState) -> list[Send] | Literal["finalize"]:
        tasks = list(state["plan"].get("tasks") or [])
        results = _latest_results(state.get("task_results") or [])
        pending = [task for task in tasks if task["id"] not in results]
        ready = [
            task
            for task in pending
            if all(
                results.get(dependency, {}).get("status") == "completed"
                for dependency in task.get("dependencies") or []
            )
        ]
        if not ready:
            return "finalize"
        if len(ready) > 1:
            get_stream_writer()(
                {
                    "type": "text_delta",
                    "text": f"正在并行执行：{', '.join(task['id'] for task in ready)}\n\n",
                }
            )
        return [
            Send(
                "execute_task",
                {
                    "goal": state["goal"],
                    "plan": state["plan"],
                    "task": task,
                    "task_results": [
                        result
                        for result in state.get("task_results") or []
                        if result.get("task_id") in (task.get("dependencies") or [])
                    ],
                },
            )
            for task in ready
        ]

    async def _execute_task_node(self, state: PlanTaskState) -> dict[str, Any]:
        writer = get_stream_writer()
        plan = _plan_from_payload(state["plan"], state.get("task_results") or [])
        task = _task_from_payload(state["task"])
        task.mark_started()
        writer(
            {
                "type": "plan_task_started",
                "task_id": task.id,
                "task_description": task.description,
            }
        )
        run = await self._run_task(plan, task, writer)
        self._write_task_result(writer, task, run)
        return {
            "task_results": [
                {
                    "task_id": task.id,
                    "status": run.status,
                    "text": run.result_text,
                    "error": run.error,
                    "usage": run.usage.to_dict(),
                    "turns": run.turns,
                }
            ],
            "usage_events": [run.usage.to_dict()],
        }

    async def _run_task(
        self,
        plan: ExecutionPlan,
        task: Task,
        writer: PlanEventWriter,
    ) -> _TaskRun:
        run = _TaskRun()
        transcript: list[Message] = []
        try:
            async for event in run_react_agent(
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                system_prompt=self._task_system_prompt(plan, task),
                user_message=_task_context(plan, task),
                history=transcript,
                cwd=self.cwd,
                config=self.config,
                approval_callback=self.approval_callback,
                skill_context_buffer=SkillContextBuffer(),
                tool_execution_scope=(
                    f"{self.thread_id}:{plan.id}:{task.id}" if self._persistent else None
                ),
                allow_uncertain_tool_retry=self._allow_uncertain_tool_retry,
                max_turns=self.max_task_turns,
            ):
                self._collect_task_event(run, task, event, writer)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - task failures are graph state, not graph crashes
            run.error = str(exc)
        finally:
            self.history.extend(bounded_tool_transcript(transcript))
        return run

    def _collect_task_event(
        self,
        run: _TaskRun,
        task: Task,
        event: AgentEvent,
        writer: PlanEventWriter,
    ) -> None:
        event_type = event.get("type")
        if event_type == "text_delta":
            run.text_parts.append(str(event.get("text") or ""))
        elif event_type == "tool_result":
            content = str(event.get("result") or "")
            if content:
                run.tool_results.append(content)
            writer(_with_task_context(event, task))
        elif event_type in {
            "thinking_delta",
            "tool_call",
            "context_compressed",
            "turn_started",
            "model_response_complete",
            "turn_complete",
        }:
            writer(_with_task_context(event, task))
        elif event_type == "done":
            run.turns += int(event.get("total_turns") or 0)
            run.usage = run.usage + Usage.from_mapping(event.get("usage") or {})
        elif event_type == "error":
            raise event["error"]

    def _write_task_result(
        self,
        writer: PlanEventWriter,
        task: Task,
        run: _TaskRun,
    ) -> None:
        if run.error:
            writer({"type": "text_delta", "text": f"失败 [{task.id}]：{run.error}\n\n"})
        else:
            writer(
                {
                    "type": "text_delta",
                    "text": f"已完成 [{task.id}]：{_preview(run.result_text)}\n\n",
                }
            )
        writer({"type": "usage", "usage": run.usage.to_dict()})
        writer(
            {
                "type": "plan_task_done",
                "task_id": task.id,
                "task_description": task.description,
                "task_status": run.status,
                "turns": run.turns,
                "tokens": run.usage.total_tokens,
            }
        )

    def _finalize_node(self, state: PlanGraphState) -> dict[str, Any]:
        tasks = list(state["plan"].get("tasks") or [])
        results = _latest_results(state.get("task_results") or [])
        all_completed = tasks and all(
            results.get(task["id"], {}).get("status") == "completed" for task in tasks
        )
        if all_completed:
            status = "completed"
            final_text = _build_plan_result(state["plan"], results)
        elif any(result.get("status") == "failed" for result in results.values()):
            status = "failed"
            final_text = "计划部分完成，但有任务执行失败。\n"
        else:
            status = "failed"
            final_text = "计划已停滞，因为任务依赖未满足。\n"
        get_stream_writer()({"type": "text_delta", "text": final_text})
        return {"status": status, "final_text": final_text}

    def _cancel_node(self, _state: PlanGraphState) -> dict[str, Any]:
        final_text = "计划已取消，未执行任何任务。"
        get_stream_writer()({"type": "text_delta", "text": final_text + "\n"})
        return {"status": "cancelled", "final_text": final_text}

    def _task_system_prompt(self, plan: ExecutionPlan, task: Task) -> str:
        base = PromptAssembler(
            config=self.config,
            cwd=self.cwd,
            tool_names=self.tool_registry.list_names(),
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
        ).build_static()
        return (
            base
            + "\n\n你正在执行 LangGraph Plan-and-Execute DAG 中的一个任务。\n"
            + f"任务 id：{task.id}\n任务类型：{task.type.value}\n"
            + "请具体完成任务，并在需要时使用工具。"
            + _task_language_instruction(plan.goal)
        )


@asynccontextmanager
async def _open_checkpointer(
    path: Path | None,
) -> AsyncIterator[AsyncSqliteSaver | InMemorySaver]:
    if path is None:
        yield InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=()))
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(descriptor)
    os.chmod(path, 0o600)
    async with aiosqlite.connect(path, timeout=5) as connection:
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA busy_timeout=5000")
        serializer = JsonPlusSerializer(allowed_msgpack_modules=())
        yield AsyncSqliteSaver(connection, serde=serializer)


def _decision_payload(decision: PlanReviewDecision) -> dict[str, str]:
    return {"action": decision.action.value, "feedback": decision.feedback}


def _plan_to_payload(plan: ExecutionPlan) -> dict[str, Any]:
    return {
        "id": plan.id,
        "goal": plan.goal,
        "summary": plan.summary,
        "tasks": [
            {
                "id": task.id,
                "description": task.description,
                "type": task.type.value,
                "dependencies": list(task.dependencies),
            }
            for task in plan.all_tasks()
        ],
    }


def _plan_from_payload(
    payload: dict[str, Any],
    results: list[dict[str, Any]] | None = None,
) -> ExecutionPlan:
    plan = ExecutionPlan(
        id=str(payload.get("id") or "plan"),
        goal=str(payload.get("goal") or ""),
    )
    plan.summary = str(payload.get("summary") or "")
    for task_payload in payload.get("tasks") or []:
        plan.add_task(_task_from_payload(task_payload))
    plan.compute_execution_order()
    for task_id, result in _latest_results(results or []).items():
        task = plan.get_task(task_id)
        if task is None:
            continue
        if result.get("status") == "completed":
            task.mark_completed(str(result.get("text") or ""))
        elif result.get("status") == "failed":
            task.mark_failed(str(result.get("error") or ""))
    return plan


def _task_from_payload(payload: dict[str, Any]) -> Task:
    try:
        task_type = TaskType(str(payload.get("type") or "ANALYSIS"))
    except ValueError:
        task_type = TaskType.ANALYSIS
    return Task(
        id=str(payload.get("id") or "task"),
        description=str(payload.get("description") or ""),
        type=task_type,
        dependencies=[str(item) for item in payload.get("dependencies") or []],
    )


def _latest_results(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(result.get("task_id")): result for result in results}


def _with_task_context(event: AgentEvent, task: Task) -> AgentEvent:
    return {
        **event,
        "phase": "execution",
        "task_id": task.id,
        "task_description": task.description,
    }


def _task_context(plan: ExecutionPlan, task: Task) -> str:
    lines = [
        f"目标：{plan.goal}",
        f"当前任务 [{task.id}]：{task.description}",
        "",
        "已完成的依赖任务结果：",
    ]
    for dependency in task.dependencies:
        completed = plan.get_task(dependency)
        if completed and completed.status == TaskStatus.COMPLETED:
            lines.append(
                f"- [{completed.id}] {completed.description}: {_preview(completed.result, 800)}"
            )
    return "\n".join(lines)


def _build_plan_result(
    plan_payload: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> str:
    lines = ["计划执行完成。", "", "任务摘要："]
    for task in plan_payload.get("tasks") or []:
        result = results.get(str(task.get("id"))) or {}
        label = "已完成" if result.get("status") == "completed" else "失败"
        lines.append(f"- [{task['id']}] {label}：{task['description']}")
        if result.get("text"):
            lines.append(f"  结果：{_preview(str(result['text']))}")
    return "\n".join(lines) + "\n"


def _task_language_instruction(goal: str) -> str:
    if any("\u4e00" <= char <= "\u9fff" for char in goal):
        return "\n用户目标包含中文；所有进度说明、分析和最终结果都必须使用中文。"
    return "\nUse the same language as the user's goal for all progress and results."


def _preview(text: str, max_len: int = 160) -> str:
    value = (text or "").replace("\r\n", "\n").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _sum_usage(events: list[UsagePayload]) -> Usage:
    total = Usage()
    for event in events:
        total = total + Usage.from_mapping(event)
    return total
