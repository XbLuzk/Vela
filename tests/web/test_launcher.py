from pathlib import Path

import pytest

from vela.entrypoints import web


def test_launcher_starts_local_web_without_opening_browser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    fake_app = object()

    def create_app(cwd: Path) -> object:
        calls["cwd"] = cwd
        return fake_app

    monkeypatch.setattr(web, "create_app", create_app)
    monkeypatch.setattr(
        web.uvicorn,
        "run",
        lambda app, **kwargs: calls.update(app=app, kwargs=kwargs),
    )
    monkeypatch.setattr(
        web,
        "_open_browser_soon",
        lambda _url: pytest.fail("browser should not open"),
    )

    web.app(["--cwd", str(tmp_path), "--port", "4312", "--no-open"])

    assert calls["cwd"] == tmp_path
    assert calls["app"] is fake_app
    assert calls["kwargs"] == {
        "host": "127.0.0.1",
        "port": 4312,
        "log_level": "warning",
    }


def test_launcher_opens_local_browser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    opened: list[str] = []
    monkeypatch.setattr(web, "create_app", lambda _cwd: object())
    monkeypatch.setattr(web.uvicorn, "run", lambda _app, **_kwargs: None)
    monkeypatch.setattr(web, "_open_browser_soon", opened.append)

    web.app(["--cwd", str(tmp_path)])

    assert opened == ["http://127.0.0.1:3080"]


def test_launcher_refuses_non_loopback_host(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        web.app(["--cwd", str(tmp_path), "--host", "0.0.0.0"])


@pytest.mark.parametrize("value", ["0", "65536"])
def test_launcher_rejects_invalid_port(value: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        web.app(["--cwd", str(tmp_path), "--port", value])
