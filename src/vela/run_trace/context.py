"""Task-local access to the Run ID currently executing Agent work."""

from __future__ import annotations

from contextvars import ContextVar, Token

_CURRENT_RUN_ID: ContextVar[str | None] = ContextVar("vela_run_id", default=None)


def current_run_id() -> str | None:
    """Return the Run ID bound to the current async task, if any."""
    return _CURRENT_RUN_ID.get()


def bind_run_id(run_id: str) -> Token[str | None]:
    """Bind a Run ID while one unit of Agent work is being pulled."""
    return _CURRENT_RUN_ID.set(run_id)


def reset_run_id(token: Token[str | None]) -> None:
    """Restore the previous task-local Run ID."""
    _CURRENT_RUN_ID.reset(token)
