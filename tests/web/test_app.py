from __future__ import annotations

from fastapi.testclient import TestClient

from vela.web.app import create_app
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
        self.events = EventHub()
        self.runtime = FakeRuntime()
        self.error = None
        self.trust_values: list[bool] = []
        self.settings: list[dict] = []

    def snapshot(self):
        return {"ready": True, "cwd": "/workspace", "version": "test"}

    async def set_trust(self, trusted: bool) -> None:
        self.trust_values.append(trusted)

    async def update_settings(self, values: dict) -> None:
        self.settings.append(values)

    async def close(self) -> None:
        return None


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


def test_web_api_updates_trust_settings_and_sessions():
    manager = FakeManager()
    app = create_app("/workspace", manager=manager, initialize=False)

    with TestClient(app) as client:
        assert client.get("/api/sessions").json()[0]["id"] == "session-1"
        assert client.post("/api/sessions").json()["id"] == "session-2"
        assert client.post("/api/sessions/session-1").json()["id"] == "session-1"
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
