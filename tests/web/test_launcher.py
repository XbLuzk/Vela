import signal
import socket
import subprocess
import sys
import time
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
        web,
        "_run_server",
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
    }


def test_launcher_opens_local_browser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    opened: list[str] = []
    monkeypatch.setattr(web, "create_app", lambda _cwd: object())
    monkeypatch.setattr(web, "_run_server", lambda _app, **_kwargs: None)
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


def test_launcher_exits_cleanly_with_connected_sse(tmp_path: Path) -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    process = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "vela.entrypoints.web",
            "--cwd",
            str(tmp_path),
            "--port",
            str(port),
            "--no-open",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    connection: socket.socket | None = None
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                connection = socket.create_connection(("127.0.0.1", port), timeout=0.2)
                break
            except OSError:
                time.sleep(0.05)
        assert connection is not None, "Vela did not start in time"
        connection.sendall(
            b"GET /api/events HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Accept: text/event-stream\r\n"
            b"Connection: keep-alive\r\n\r\n"
        )
        connection.settimeout(3)
        assert b"200 OK" in connection.recv(4096)

        process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=8)

        assert process.returncode == 0
        assert "timeout graceful shutdown exceeded" not in output
        assert "CancelledError" not in output
    finally:
        if connection is not None:
            connection.close()
        if process.poll() is None:
            process.kill()
            process.wait()
