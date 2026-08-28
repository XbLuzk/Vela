from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vela import __version__
from vela.web.directory_picker import pick_directory
from vela.web.runtime import RuntimeManager

SSE_POLL_INTERVAL_SECONDS = 0.5


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=100_000)
    mode: str = "react"


class DecisionRequest(BaseModel):
    value: str = Field(min_length=1, max_length=10_000)


class TrustRequest(BaseModel):
    trusted: bool


class SessionMetadataRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    pinned: bool | None = None


class ProjectRequest(BaseModel):
    path: str = Field(min_length=1, max_length=10_000)


class SettingsRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    context_window: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    agent_mode: str | None = None
    approval_mode: str | None = None


def create_app(
    cwd: str | Path,
    *,
    manager: RuntimeManager | None = None,
    initialize: bool = True,
    directory_picker: Callable[[str | Path], str | None] = pick_directory,
) -> FastAPI:
    runtime_manager = manager or RuntimeManager(cwd)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if initialize:
            await runtime_manager.initialize()
        yield
        await runtime_manager.close()

    app = FastAPI(title="Vela", version=__version__, lifespan=lifespan)
    app.state.runtime_manager = runtime_manager

    @app.middleware("http")
    async def reject_cross_origin_writes(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"} and request.url.path.startswith(
            "/api/"
        ):
            origin = request.headers.get("origin")
            if origin and not _is_local_origin(origin):
                return JSONResponse(
                    {"detail": "Cross-origin Web actions are not allowed"},
                    status_code=403,
                )
        return await call_next(request)

    @app.get("/api/bootstrap")
    async def bootstrap() -> dict[str, Any]:
        return runtime_manager.snapshot()

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        return StreamingResponse(
            _event_stream(request, runtime_manager),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/messages", status_code=202)
    async def send_message(request: MessageRequest) -> dict[str, str]:
        runtime = _runtime(runtime_manager)
        try:
            run_id = await runtime.send(request.message, request.mode)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"run_id": run_id}

    @app.post("/api/cancel")
    async def cancel() -> dict[str, bool]:
        return {"cancelled": _runtime(runtime_manager).cancel()}

    @app.post("/api/approval")
    async def approve(request: DecisionRequest) -> dict[str, str]:
        return {"message": _runtime(runtime_manager).submit_approval(request.value)}

    @app.post("/api/plan-review")
    async def review_plan(request: DecisionRequest) -> dict[str, str]:
        return {"message": _runtime(runtime_manager).submit_plan_review(request.value)}

    @app.get("/api/sessions")
    async def list_sessions() -> list[dict[str, Any]]:
        return runtime_manager.list_sessions()

    @app.post("/api/sessions")
    async def new_session() -> dict[str, Any]:
        try:
            return await runtime_manager.new_session()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/sessions/{reference}")
    async def switch_session(reference: str) -> dict[str, Any]:
        try:
            return await runtime_manager.switch_session(reference)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.delete("/api/sessions/{reference}")
    async def delete_session(reference: str) -> dict[str, Any]:
        try:
            return await runtime_manager.delete_session(reference)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.patch("/api/sessions/{reference}")
    async def update_session(reference: str, request: SessionMetadataRequest) -> dict[str, Any]:
        if request.title is None and request.pinned is None:
            raise HTTPException(status_code=422, detail="Provide a session title or pin state")
        try:
            if request.title is not None:
                return await runtime_manager.rename_session(reference, request.title)
            return await runtime_manager.pin_session(reference, bool(request.pinned))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/project/pick")
    async def pick_project() -> dict[str, Any]:
        try:
            selected = await asyncio.to_thread(directory_picker, runtime_manager.cwd)
            if selected is None:
                return {"selected": False}
            snapshot = await runtime_manager.select_project(selected)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"selected": True, "bootstrap": snapshot}

    @app.post("/api/project/select")
    async def select_project(request: ProjectRequest) -> dict[str, Any]:
        try:
            return await runtime_manager.select_project(request.path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/trust")
    async def set_trust(request: TrustRequest) -> dict[str, Any]:
        try:
            await runtime_manager.set_trust(request.trusted)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return runtime_manager.snapshot()

    @app.put("/api/settings")
    async def update_settings(request: SettingsRequest) -> dict[str, Any]:
        values = request.model_dump(exclude_none=True)
        try:
            await runtime_manager.update_settings(values)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return runtime_manager.snapshot()

    static_dir = Path(__file__).with_name("static")
    assets_dir = static_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/favicon.svg", include_in_schema=False)
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(static_dir / "favicon.svg", media_type="image/svg+xml")

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str) -> FileResponse:
        _ = path
        index = static_dir / "index.html"
        if not index.is_file():
            raise HTTPException(
                status_code=503,
                detail="Web assets are missing. Run npm install && npm run build in web/.",
            )
        return FileResponse(index)

    return app


def _runtime(manager: RuntimeManager):
    if manager.runtime is None:
        raise HTTPException(status_code=503, detail=manager.error or "Vela is not ready")
    return manager.runtime


def _sse(event: dict[str, Any]) -> str:
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"data: {payload}\n\n"


async def _event_stream(
    request: Request,
    runtime_manager: RuntimeManager,
    *,
    poll_interval: float = SSE_POLL_INTERVAL_SECONDS,
) -> AsyncIterator[str]:
    yield _sse({"type": "connected"})
    # EventSource reconnects automatically. Re-send authoritative state so
    # a brief disconnect cannot leave the browser showing a stale task.
    yield _sse({"type": "bootstrap", "bootstrap": runtime_manager.snapshot()})
    event_stream = runtime_manager.events.stream()
    pending_event: asyncio.Task[dict[str, Any]] | None = asyncio.create_task(anext(event_stream))
    try:
        while True:
            if await request.is_disconnected():
                return
            done, _ = await asyncio.wait({pending_event}, timeout=poll_interval)
            if not done:
                yield ": keep-alive\n\n"
                continue
            try:
                event = pending_event.result()
            except StopAsyncIteration:
                return
            yield _sse(event)
            pending_event = asyncio.create_task(anext(event_stream))
    except asyncio.CancelledError:
        return
    finally:
        if pending_event is not None and not pending_event.done():
            pending_event.cancel()
            with suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending_event
        await event_stream.aclose()


def _is_local_origin(origin: str) -> bool:
    try:
        return urlsplit(origin).hostname in {"127.0.0.1", "localhost", "::1"}
    except ValueError:
        return False
