from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from vela.web.app import _event_stream, create_app
from vela.web.runtime import EventHub


class FakeRuntime:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.cancelled = False
        self.approvals: list[str] = []
        self.reviews: list[str] = []

    async def send(self, message: str, mode: str) -> str:
        self.sent.append((message, mode))
        return "run-1"

    def cancel(self) -> bool:
        self.cancelled = True
        return True

    def submit_approval(self, value: str) -> str:
        self.approvals.append(value)
        return "ok"

    def submit_plan_review(self, value: str) -> str:
        self.reviews.append(value)
        return "ok"

    def list_sessions(self):
        return [{"id": "session-1", "title": "Demo"}]

    async def new_session(self):
        return {"id": "session-2", "title": "New session", "messages": []}

    async def switch_session(self, reference: str):
        return {"id": reference, "title": "Demo", "messages": []}


class FakeManager:
    def __init__(self) -> None:
        self.cwd = Path("/workspace")
        self.events = EventHub()
        self.runtime = FakeRuntime()
        self.error = None
        self.trust_values: list[bool] = []
        self.settings: list[dict] = []

    def snapshot(self):
        return {"ready": True, "cwd": str(self.cwd), "version": "test"}

    async def set_trust(self, trusted: bool) -> None:
        self.trust_values.append(trusted)

    async def update_settings(self, values: dict) -> None:
        self.settings.append(values)

    def list_sessions(self):
        return self.runtime.list_sessions()

    async def new_session(self):
        return await self.runtime.new_session()

    async def switch_session(self, reference: str):
        return await self.runtime.switch_session(reference)

    async def delete_session(self, reference: str):
        return {**self.snapshot(), "session": {"id": "session-2"}, "sessions": []}

    async def rename_session(self, reference: str, title: str):
        return {"id": reference, "title": title, "pinned": False}

    async def pin_session(self, reference: str, pinned: bool):
        return {"id": reference, "title": "Demo", "pinned": pinned}

    async def select_project(self, path: str):
        self.cwd = Path(path)
        return self.snapshot()

    async def close(self) -> None:
        return None


class DisconnectingRequest:
    def __init__(self) -> None:
        self.checks = 0

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self.checks > 1


def test_web_api_routes_messages_and_interactions():
    manager = FakeManager()
    app = create_app("/workspace", manager=manager, initialize=False)

    with TestClient(app) as client:
        assert client.get("/api/bootstrap").json()["ready"] is True
        assert client.post(
            "/api/messages",
            json={"message": "hello", "mode": "plan"},
        ).json() == {"run_id": "run-1"}
        assert client.post("/api/cancel").json() == {"cancelled": True}
        assert client.post("/api/approval", json={"value": "approve"}).json() == {"message": "ok"}
        assert client.post("/api/plan-review", json={"value": "execute"}).json() == {
            "message": "ok"
        }

    assert manager.runtime.sent == [("hello", "plan")]
    assert manager.runtime.cancelled is True
    assert manager.runtime.approvals == ["approve"]
    assert manager.runtime.reviews == ["execute"]


def test_event_stream_stops_after_browser_disconnect():
    async def scenario():
        stream = _event_stream(
            DisconnectingRequest(),  # type: ignore[arg-type]
            FakeManager(),  # type: ignore[arg-type]
            poll_interval=0.01,
        )

        assert '"type":"connected"' in await anext(stream)
        assert '"type":"bootstrap"' in await anext(stream)
        assert await anext(stream) == ": keep-alive\n\n"
        try:
            await anext(stream)
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("disconnected browsers must end their event stream")

    asyncio.run(scenario())


def test_web_api_updates_trust_settings_and_sessions():
    manager = FakeManager()
    app = create_app("/workspace", manager=manager, initialize=False)

    with TestClient(app) as client:
        assert client.get("/api/sessions").json()[0]["id"] == "session-1"
        assert client.post("/api/sessions").json()["id"] == "session-2"
        assert client.post("/api/sessions/session-1").json()["id"] == "session-1"
        assert client.delete("/api/sessions/session-1").json()["session"]["id"] == "session-2"
        assert (
            client.patch("/api/sessions/session-1", json={"title": "Renamed"}).json()["title"]
            == "Renamed"
        )
        assert (
            client.patch("/api/sessions/session-1", json={"pinned": True}).json()["pinned"] is True
        )
        assert client.post("/api/trust", json={"trusted": True}).status_code == 200
        response = client.put(
            "/api/settings",
            json={
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_key": "secret",
                "agent_mode": "react",
                "approval_mode": "ask",
            },
        )
        assert response.status_code == 200

    assert manager.trust_values == [True]
    assert manager.settings == [
        {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": "secret",
            "agent_mode": "react",
            "approval_mode": "ask",
        }
    ]


def test_web_api_uses_native_picker_to_switch_project():
    manager = FakeManager()
    selected = "/workspace/another-project"
    app = create_app(
        "/workspace",
        manager=manager,
        initialize=False,
        directory_picker=lambda current: selected if str(current) == "/workspace" else None,
    )

    with TestClient(app) as client:
        response = client.post("/api/project/pick")

    assert response.status_code == 200
    assert response.json()["selected"] is True
    assert response.json()["bootstrap"]["cwd"] == selected


def test_web_api_treats_cancelled_project_picker_as_no_change():
    manager = FakeManager()
    app = create_app(
        "/workspace",
        manager=manager,
        initialize=False,
        directory_picker=lambda _current: None,
    )

    with TestClient(app) as client:
        response = client.post("/api/project/pick")

    assert response.json() == {"selected": False}
    assert manager.cwd == Path("/workspace")


def test_web_api_selects_a_recent_project():
    manager = FakeManager()
    app = create_app("/workspace", manager=manager, initialize=False)

    with TestClient(app) as client:
        response = client.post("/api/project/select", json={"path": "/workspace/recent"})

    assert response.status_code == 200
    assert response.json()["cwd"] == "/workspace/recent"


def test_web_api_rejects_cross_origin_mutations():
    manager = FakeManager()
    app = create_app("/workspace", manager=manager, initialize=False)

    with TestClient(app) as client:
        response = client.post(
            "/api/cancel",
            headers={"Origin": "https://attacker.example"},
        )

    assert response.status_code == 403
    assert manager.runtime.cancelled is False


def test_web_serves_vela_favicon_instead_of_spa_fallback():
    app = create_app("/workspace", manager=FakeManager(), initialize=False)

    with TestClient(app) as client:
        svg_response = client.get("/favicon.svg")
        ico_response = client.get("/favicon.ico")

    assert svg_response.status_code == 200
    assert svg_response.headers["content-type"].startswith("image/svg+xml")
    assert "Vela" in svg_response.text
    assert ico_response.status_code == 200
    assert ico_response.headers["content-type"].startswith("image/svg+xml")
