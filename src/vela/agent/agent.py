"""The stateful Agent used by both the CLI and interactive REPL."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import Any, Literal

from vela.agent.plan_graph import LangGraphPlanAgent
from vela.agent.react_runtime import run_react_agent
from vela.config import VelaConfig
from vela.events import AgentEvent
from vela.llm.base import LlmClient
from vela.prompt import PromptAssembler
from vela.run_trace import RunTrace, RunTraceStore, RunTracker
from vela.skill import SkillContextBuffer
from vela.task_control import PlanReviewDecision
from vela.tools.registry import ToolRegistry
from vela.types import Message, QueryResult, Usage

AgentMode = Literal["react", "plan"]


class Agent:
    """A terminal AI agent that connects an LLM to tools for task execution.

    The Agent owns the conversation history, manages context compression, and
    delegates the actual LLM-tool interaction loop to mode-specific runners.
    All ``run()`` methods yield the same streaming event protocol so that UI
    layers (REPL, CLI, or programmatic) can render progress uniformly.

    Typical usage::

        agent = Agent(
            llm_client=my_client,
            tool_registry=my_registry,
            config=my_config,
            cwd="/workspace",
        )
        async for event in agent.run("list all Python files"):
            if event["type"] == "text_delta":
                print(event["text"], end="")
    """

    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: VelaConfig,
        cwd: str,
        approval_callback: Callable | None = None,
        mode: AgentMode = "react",
        system_prompt: str | None = None,
        max_turns: int = 20,
        plan_review_callback: (
            Callable[[Any], PlanReviewDecision | Awaitable[PlanReviewDecision]] | None
        ) = None,
        steering_callback: Callable[[], str | None] | None = None,
        trace_store: RunTraceStore | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.mode = mode
        self.max_turns = max_turns
        self.plan_review_callback = plan_review_callback
        self.steering_callback = steering_callback
        self.graph_thread_id: str | None = None
        self.trace_store = trace_store
        self.last_run_trace: RunTrace | None = None
        self.last_run_trace_warning: str | None = None

        self.system_prompt = (
            system_prompt
            or PromptAssembler(
                config=config,
                cwd=cwd,
                tool_names=tool_registry.list_names(),
                model=llm_client.model_name,
                provider=llm_client.provider_name,
            ).build_static()
        )

        self.history: list[Message] = []
        self.skill_context_buffer = SkillContextBuffer()
        self.last_usage = Usage()
        self._validate_config()

    # ------------------------------------------------------------------
    # Public API — run the agent
    # ------------------------------------------------------------------

    async def run(self, message: str) -> AsyncIterator[AgentEvent]:
        """Run one request in the selected mode and yield progress events."""
        runner = self._run_plan if self.mode == "plan" else self._run_react
        stream = self.track_events(runner(message), mode=self.mode)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    async def track_events(
        self,
        events: AsyncIterable[AgentEvent],
        *,
        mode: AgentMode,
    ) -> AsyncIterator[AgentEvent]:
        """Attach one Run ID and durable summary to an Agent event stream."""
        tracker = RunTracker(
            mode=mode,
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
            cwd=self.cwd,
            session_id=self.graph_thread_id,
            store=self.trace_store,
        )
        stream = tracker.stream(events)
        try:
            async for event in stream:
                yield event
        finally:
            cleanup_warnings: list[str] = []
            try:
                await stream.aclose()
            except Exception as exc:  # noqa: BLE001 - cleanup must not hide the run outcome
                cleanup_warnings.append(f"tracker cleanup failed: {type(exc).__name__}")
            try:
                close = getattr(events, "aclose", None)
                if close is not None:
                    await close()
            except Exception as exc:  # noqa: BLE001 - retain trace even when a provider misbehaves
                cleanup_warnings.append(f"child cleanup failed: {type(exc).__name__}")
            finally:
                self.last_run_trace = tracker.trace
                warnings = [warning for warning in [tracker.warning, *cleanup_warnings] if warning]
                self.last_run_trace_warning = "; ".join(warnings) or None

    async def run_complete(self, message: str) -> QueryResult:
        """Run the agent synchronously (collect all events) and return a result."""
        text = ""
        tokens = 0
        turns = 0
        usage = Usage()
        stream = self.run(message)
        try:
            async for event in stream:
                event_type = event.get("type")
                if event_type == "text_delta":
                    text += str(event.get("text") or "")
                elif event_type == "error":
                    raise event["error"]  # type: ignore[arg-type]
                elif event_type == "done":
                    tokens = int(event.get("total_tokens") or 0)
                    turns = int(event.get("total_turns") or 0)
                    usage = Usage.from_mapping(event.get("usage") or {})
        finally:
            await stream.aclose()
        return QueryResult(text=text, total_tokens=tokens, turns=turns, usage=usage)

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        """Reset conversation history and skill context buffer."""
        self.history = []
        self.skill_context_buffer.clear()
        self.last_usage = Usage()

    # ------------------------------------------------------------------
    # Mode runners
    # ------------------------------------------------------------------

    async def _run_react(self, message: str) -> AsyncIterator[AgentEvent]:
        """Standard ReAct loop (the default and most common mode)."""
        stream = run_react_agent(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            system_prompt=self.system_prompt,
            user_message=message,
            history=self.history,
            cwd=self.cwd,
            config=self.config,
            approval_callback=self.approval_callback,
            skill_context_buffer=self.skill_context_buffer,
            steering_callback=self.steering_callback,
            max_turns=self.max_turns,
        )
        try:
            async for event in stream:
                if event.get("type") == "done":
                    self.history = list(event.get("messages") or [])
                    self.last_usage = Usage.from_mapping(event.get("usage") or {})
                yield event
        finally:
            await stream.aclose()

    async def _run_plan(self, message: str) -> AsyncIterator[AgentEvent]:
        """Plan-then-execute mode: the planner creates a DAG, then workers run it."""
        agent = LangGraphPlanAgent(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            max_task_turns=self.max_turns,
            plan_review_callback=self.plan_review_callback,
            thread_id=self.graph_thread_id,
        )
        previous_history = list(self.history)
        agent.history = list(previous_history)
        self.history = [*previous_history, Message(role="user", content=message)]

        stream = agent.run(message)
        try:
            async for event in stream:
                if event.get("type") == "done":
                    self.history = [*previous_history, *list(event.get("messages") or [])]
                    self.last_usage = Usage.from_mapping(event.get("usage") or {})
                yield event
        finally:
            await stream.aclose()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Raise early if the configuration is obviously wrong."""
        if self.mode not in {"react", "plan"}:
            raise ValueError("Agent mode must be react or plan.")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        if not self.config.llm.api_key:
            raise ValueError(
                "LLM API key is not configured. "
                "Set VELA_API_KEY (or DEEPSEEK_API_KEY / GLM_API_KEY / etc.) "
                "in the environment or in ~/.vela/config.json."
            )
        if not self.llm_client.max_context_window:
            raise ValueError(
                "LLM context window is not configured. "
                "Set a VELA_CONTEXT_WINDOW environment variable or configure "
                "it in ~/.vela/config.json."
            )
