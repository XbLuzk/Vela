"""agent.py — Core terminal Agent inspired by Claude Code.

Integrates LLM API, parses natural-language instructions, schedules command
and file operations via the tool system, and generates streaming responses.

Supports three execution modes:
    - react  : Standard ReAct loop (thought → tool → observation → answer)
    - plan   : Plan-then-execute with a DAG of sub-tasks
    - team   : Multi-agent orchestrator (planner → workers → reviewer)

All modes share the same streaming event protocol so callers can render
progress incrementally.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal

from vela.agent.orchestrator import AgentOrchestrator
from vela.agent.plan_graph import LangGraphPlanAgent
from vela.agent.query import query
from vela.config import VelaConfig
from vela.llm.base import LlmClient
from vela.prompt import PromptAssembler
from vela.skill import SkillContextBuffer
from vela.task_control import PlanReviewDecision
from vela.tools.registry import ToolRegistry
from vela.types import Message, QueryResult, Usage

AgentMode = Literal["react", "plan", "team"]


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
        max_plan_depth: int = 1,
        plan_review_callback: (
            Callable[[Any], PlanReviewDecision | Awaitable[PlanReviewDecision]] | None
        ) = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.mode = mode
        self.max_turns = max_turns
        self.max_plan_depth = max_plan_depth
        self.plan_review_callback = plan_review_callback
        self.graph_thread_id: str | None = None

        # Build the base system prompt from personality profile and config.
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

        # Conversation state.
        self.history: list[Message] = []
        self.skill_context_buffer = SkillContextBuffer()

        # Accumulated usage across all turns of the session.
        self.last_usage = Usage()

        # Validate configuration.
        self._validate_config()

    # ------------------------------------------------------------------
    # Public API — run the agent
    # ------------------------------------------------------------------

    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Execute *message* and yield streaming events.

        Events follow this protocol (``type`` field determines the shape):

        ``text_delta``
            {"type": "text_delta", "text": "..."}
        ``thinking_delta``
            {"type": "thinking_delta", "thinking": "..."}
        ``tool_call``
            {"type": "tool_call", "name": "...", "input": {...}}
        ``tool_result``
            {"type": "tool_result", "name": "...", "result": "...", "is_error": bool}
        ``usage``
            {"type": "usage", "usage": {...}}
        ``turn_complete``
            {"type": "turn_complete", "turn": int, "stop_reason": str}
        ``context_compressed``
            {"type": "context_compressed", "before_tokens": ..., "after_tokens": ...,
             "summarized_messages": int}
        ``plan_status``
            {"type": "plan_status", "phase": "planning" | "execution"}
        ``plan_review``
            {"type": "plan_review"}; interactive callers should collect an
            execute/modify/cancel decision through the configured review callback.
        ``plan_task_started``
            {"type": "plan_task_started", "task_id": str, "task_description": str}
        ``plan_task_done``
            {"type": "plan_task_done", "turns": int, "tokens": int}
        ``error``
            {"type": "error", "error": Exception}
        ``done``
            {"type": "done", "total_turns": int, "total_tokens": int,
             "usage": {...}, "messages": [Message, ...]}

        Cooperative cancellation raises ``asyncio.CancelledError`` instead of
        emitting ``done``. Callers should persist ``history`` in a ``finally`` block.
        """
        if self.mode == "plan":
            runner = self._run_plan
        elif self.mode == "team":
            runner = self._run_team
        else:
            runner = self._run_react

        async for event in runner(message):
            yield event

    async def run_complete(self, message: str) -> QueryResult:
        """Run the agent synchronously (collect all events) and return a result."""
        text = ""
        tokens = 0
        turns = 0
        usage = Usage()
        async for event in self.run(message):
            event_type = event.get("type")
            if event_type == "text_delta":
                text += str(event.get("text") or "")
            elif event_type == "error":
                raise event["error"]  # type: ignore[arg-type]
            elif event_type == "done":
                tokens = int(event.get("total_tokens") or 0)
                turns = int(event.get("total_turns") or 0)
                usage = Usage.from_mapping(event.get("usage") or {})
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

    async def _run_react(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Standard ReAct loop (the default and most common mode)."""
        async for event in query(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            system_prompt=self.system_prompt,
            user_message=message,
            history=self.history,
            history_sink=self.history,
            cwd=self.cwd,
            config=self.config,
            approval_callback=self.approval_callback,
            skill_context_buffer=self.skill_context_buffer,
            max_turns=self.max_turns,
        ):
            if event.get("type") == "done":
                self.history = list(event.get("messages") or [])
                self.last_usage = Usage.from_mapping(event.get("usage") or {})
            yield event

    async def _run_plan(self, message: str) -> AsyncIterator[dict[str, Any]]:
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

        async for event in agent.run(message):
            if event.get("type") == "done":
                self.history = [*previous_history, *list(event.get("messages") or [])]
                self.last_usage = Usage.from_mapping(event.get("usage") or {})
            yield event

    async def _run_team(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Multi-agent team mode: planner → parallel workers → reviewer."""
        previous_history = list(self.history)
        orchestrator = AgentOrchestrator(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            default_worker_mode="react",
            plan_review_callback=self.plan_review_callback,
        )
        self.history = [*previous_history, Message(role="user", content=message)]
        async for event in orchestrator.run(message):
            if event.get("type") == "done":
                self.history = [*previous_history, *list(event.get("messages") or [])]
                self.last_usage = Usage.from_mapping(event.get("usage") or {})
            yield event

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Raise early if the configuration is obviously wrong."""
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
