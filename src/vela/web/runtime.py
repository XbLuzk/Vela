from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from vela import __version__
from vela.agent import Agent
from vela.bootstrap import build_tool_registry
from vela.config import VelaConfig, config_to_public_dict, load_config, update_user_config
from vela.events import AgentEvent
from vela.llm import create_llm_client
from vela.llm.model_profiles import DEFAULT_MODEL_PROFILES
from vela.mcp import McpClientManager
from vela.session import ActiveSession, SessionRecord
from vela.session_history import finalize_interrupted_history
from vela.task_control import TaskController, TaskState
from vela.trust import ProjectTrustStore, has_trust_sensitive_resources


class EventHub:
    """Fan out JSON-safe runtime events to connected browser streams."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    async def publish(self, event: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(event)

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)


class WebRuntime:
    """One workspace-scoped Agent, session, and foreground task."""

    def __init__(
        self,
        *,
        cwd: Path,
        config: VelaConfig,
        agent: Agent,
        mcp_manager: McpClientManager,
        active_session: ActiveSession,
        events: EventHub,
    ) -> None:
        self.cwd = cwd
        self.config = config
        self.agent = agent
        self.mcp_manager = mcp_manager
        self.active_session = active_session
        self.events = events
        self.current_run_id: str | None = None
        self.controller = TaskController(on_change=self._schedule_state_event)
        self.agent.approval_callback = self._request_approval
        self.agent.plan_review_callback = self.controller.request_plan_review

    @classmethod
    async def open(
        cls,
        cwd: Path,
        config: VelaConfig,
        events: EventHub,
        warnings: list[str],
    ) -> WebRuntime:
        if not config.llm.api_key:
            raise ValueError("LLM API key is not configured. Open Settings to configure a model.")
        registry, mcp_manager = await build_tool_registry(config=config, cwd=str(cwd))
        warnings.extend(mcp_manager.config_warnings)
        warnings.extend(
            f"MCP server {name} failed to load: {error}"
            for name, error in mcp_manager.last_errors.items()
        )
        active_session = ActiveSession.open(cwd, resume=True)
        client = create_llm_client(config.llm)
        agent = Agent(
            llm_client=client,
            tool_registry=registry,
            config=config,
            cwd=str(cwd),
            mode=config.prompt.agent_mode,
        )
        agent.graph_thread_id = active_session.current.id
        agent.history = list(active_session.current.messages)
        return cls(
            cwd=cwd,
            config=config,
            agent=agent,
            mcp_manager=mcp_manager,
            active_session=active_session,
            events=events,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "task": self.task_snapshot(),
            "session": _session_payload(self.active_session.current, include_messages=True),
            "sessions": self.list_sessions(),
            "tool_count": len(self.agent.tool_registry.list_names()),
        }

    def task_snapshot(self) -> dict[str, Any]:
        approval = self.controller.approval_request
        return {
            "active": self.controller.active,
            "state": self.controller.state.value if self.controller.state else "idle",
            "run_id": self.current_run_id,
            "approval": _json_value(approval) if approval else None,
            "awaiting_plan_review": self.controller.awaiting_plan_review,
            "review_feedback_pending": self.controller.review_feedback_pending,
            "error": str(self.controller.error) if self.controller.error else None,
        }

    async def send(self, message: str, mode: str) -> str:
        text = message.strip()
        if not text:
            raise ValueError("Message cannot be empty")
        if self.controller.active:
            raise RuntimeError("A task is already running")
        if mode not in {"react", "plan"}:
            raise ValueError("Mode must be react or plan")
        self.agent.mode = mode
        run_id = uuid.uuid4().hex
        self.current_run_id = run_id
        await self.events.publish(
            {
                "type": "user_message",
                "run_id": run_id,
                "message": {"role": "user", "content": text},
            }
        )
        self.controller.start(
            self._run(text, run_id),
            initial_state=TaskState.PLANNING if mode == "plan" else TaskState.RUNNING,
            label=text,
        )
        return run_id

    async def _run(self, message: str, run_id: str) -> None:
        pending_error: Exception | None = None
        try:
            stream = self.agent.run(message)
            try:
                async for event in stream:
                    if event.get("type") == "plan_status":
                        self.controller.set_phase(str(event.get("phase") or ""))
                    await self.events.publish({**serialize_agent_event(event), "run_id": run_id})
                    if event.get("type") == "error":
                        error = event.get("error")
                        pending_error = (
                            error if isinstance(error, Exception) else RuntimeError(str(error))
                        )
            finally:
                await stream.aclose()
            if pending_error is not None:
                raise pending_error
        except asyncio.CancelledError:
            self.agent.history = finalize_interrupted_history(
                self.agent.history, status="cancelled"
            )
            await self.events.publish({"type": "run_cancelled", "run_id": run_id})
            raise
        except Exception as exc:
            self.agent.history = finalize_interrupted_history(
                self.agent.history,
                status="failed",
                detail=str(exc),
            )
            await self.events.publish({"type": "run_failed", "run_id": run_id, "error": str(exc)})
            raise
        finally:
            self.active_session.save(self.agent.history, title=message)
            await self.events.publish(
                {
                    "type": "session_updated",
                    "run_id": run_id,
                    "session": _session_payload(self.active_session.current, include_messages=True),
                }
            )

    def cancel(self) -> bool:
        return self.controller.request_cancel()

    def submit_approval(self, value: str) -> str:
        return self.controller.submit_approval(value)

    def submit_plan_review(self, value: str) -> str:
        return self.controller.submit_plan_review(value)

    def list_sessions(self) -> list[dict[str, Any]]:
        return [
            _session_payload(record, include_messages=False)
            for record in self.active_session.list(limit=30)
        ]

    async def new_session(self) -> dict[str, Any]:
        self._require_idle()
        record = self.active_session.new()
        self._activate_session(record)
        await self.events.publish(
            {
                "type": "session_changed",
                "session": _session_payload(record, include_messages=True),
            }
        )
        return _session_payload(record, include_messages=True)

    async def switch_session(self, reference: str) -> dict[str, Any]:
        self._require_idle()
        record = self.active_session.switch(reference)
        if record is None:
            raise KeyError(f"Session not found: {reference}")
        self._activate_session(record)
        await self.events.publish(
            {
                "type": "session_changed",
                "session": _session_payload(record, include_messages=True),
            }
        )
        return _session_payload(record, include_messages=True)

    async def close(self) -> None:
        if self.controller.active:
            self.controller.request_cancel()
            await self.controller.wait()
        self.active_session.close()

    async def _request_approval(self, request: dict[str, Any]) -> str:
        decision = await self.controller.request_approval(request)
        if decision == "auto":
            self.config.policy.approval_mode = "auto"
            return "approve"
        return decision

    def _activate_session(self, record: SessionRecord) -> None:
        self.agent.clear_history()
        self.agent.history = list(record.messages)
        self.agent.graph_thread_id = record.id

    def _require_idle(self) -> None:
        if self.controller.active:
            raise RuntimeError("Cancel the running task before changing sessions")

    def _schedule_state_event(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.events.publish({"type": "task_state", "task": self.task_snapshot()}))


class RuntimeManager:
    """Initialize and rebuild the Web runtime around trust and model settings."""

    def __init__(self, cwd: str | Path) -> None:
        self.cwd = Path(cwd).expanduser().resolve()
        self.events = EventHub()
        self.runtime: WebRuntime | None = None
        self.config: VelaConfig | None = None
        self.config_warnings: list[str] = []
        self.error: str | None = None
        self.trust_store = ProjectTrustStore()
        self.project_extensions_pending = False
        self.project_trusted = False

    async def initialize(self) -> None:
        sensitive = has_trust_sensitive_resources(self.cwd)
        saved = self.trust_store.get(self.cwd)
        self.project_extensions_pending = sensitive and saved is None
        self.project_trusted = True if not sensitive else bool(saved)
        await self.rebuild()

    async def rebuild(self) -> None:
        if self.runtime is not None:
            await self.runtime.close()
            self.runtime = None
        self.config_warnings = []
        self.config = load_config(
            project_trusted=self.project_trusted,
            warnings=self.config_warnings,
        )
        try:
            self.runtime = await WebRuntime.open(
                self.cwd,
                self.config,
                self.events,
                self.config_warnings,
            )
        except Exception as exc:  # noqa: BLE001 - UI must remain available for repair
            self.error = str(exc)
        else:
            self.error = None
        await self.events.publish({"type": "bootstrap", "bootstrap": self.snapshot()})

    def snapshot(self) -> dict[str, Any]:
        config = self.config or load_config(project_trusted=self.project_trusted)
        warnings = list(self.config_warnings)
        if self.runtime is not None:
            warnings.extend(_session_warnings(self.runtime.active_session))
        base = {
            "version": __version__,
            "ready": self.runtime is not None,
            "error": self.error,
            "cwd": str(self.cwd),
            "project_extensions_pending": self.project_extensions_pending,
            "project_trusted": self.project_trusted,
            "config": config_to_public_dict(config),
            "model_profiles": [asdict(profile) for profile in DEFAULT_MODEL_PROFILES],
            "warnings": warnings,
        }
        if self.runtime is not None:
            base.update(self.runtime.snapshot())
        return base

    async def set_trust(self, trusted: bool) -> None:
        if self.runtime is not None and self.runtime.controller.active:
            raise RuntimeError("Cancel the running task before changing project trust")
        self.trust_store.set(self.cwd, trusted)
        self.project_trusted = trusted
        self.project_extensions_pending = False
        await self.rebuild()

    async def update_settings(self, values: dict[str, Any]) -> None:
        if self.runtime is not None and self.runtime.controller.active:
            raise RuntimeError("Cancel the running task before changing settings")
        update_user_config(values)
        await self.rebuild()

    async def close(self) -> None:
        if self.runtime is not None:
            await self.runtime.close()


def serialize_agent_event(event: AgentEvent) -> dict[str, Any]:
    return {str(key): _json_value(value) for key, value in event.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _session_payload(record: SessionRecord, include_messages: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": record.id,
        "title": record.title,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "message_count": record.message_count,
    }
    if include_messages:
        payload["messages"] = [_json_value(message) for message in record.messages]
    return payload


def _session_warnings(active_session: ActiveSession) -> list[str]:
    warning = active_session.take_warning()
    return [warning] if warning else []
