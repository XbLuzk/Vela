"""Start Vela's local Web application."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from socket import socket

import uvicorn
from fastapi import FastAPI

from vela import __version__
from vela.web import create_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3080
GRACEFUL_SHUTDOWN_TIMEOUT = 2
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class _ShutdownAwareServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, on_shutdown: Callable[[], None]) -> None:
        super().__init__(config)
        self._on_shutdown = on_shutdown

    async def shutdown(self, sockets: list[socket] | None = None) -> None:
        self._on_shutdown()
        await super().shutdown(sockets)


def app(argv: Sequence[str] | None = None) -> None:
    """Launch the local server and open Vela in the default browser."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.host not in LOCAL_HOSTS:
        parser.error("--host must be a loopback address (127.0.0.1, localhost, or ::1)")

    cwd = args.cwd.expanduser().resolve()
    if not cwd.is_dir():
        parser.error(f"workspace does not exist or is not a directory: {cwd}")

    url = _browser_url(args.host, args.port)
    if not args.no_open:
        _open_browser_soon(url)

    print(f"Vela {__version__} is available at {url}")
    print("Press Ctrl+C to stop the local server.")
    _run_server(
        create_app(cwd),
        host=args.host,
        port=args.port,
    )


def _run_server(application: FastAPI, *, host: str, port: int) -> None:
    config = uvicorn.Config(
        application,
        host=host,
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT,
    )
    server = _ShutdownAwareServer(
        config,
        application.state.runtime_manager.events.close,
    )
    with suppress(KeyboardInterrupt):
        server.run()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vela",
        description="Open Vela's local Web workspace.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="workspace directory (default: current directory)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=_port, default=DEFAULT_PORT, help="local server port")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    parser.add_argument("--version", action="version", version=f"Vela {__version__}")
    return parser


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _browser_url(host: str, port: int) -> str:
    display_host = f"[{host}]" if ":" in host else host
    return f"http://{display_host}:{port}"


def _open_browser_soon(url: str) -> None:
    timer = threading.Timer(0.6, webbrowser.open, args=(url,))
    timer.daemon = True
    timer.start()


if __name__ == "__main__":
    app()
