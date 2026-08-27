from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from vela.config import load_config
from vela.task_control import TaskState
from vela.web.runtime import RuntimeManager

pytestmark = pytest.mark.live


def test_real_model_react_web_runtime(tmp_path: Path) -> None:
    result = _run_web(
        tmp_path,
        "只回复 VELA_LIVE_OK，不要添加其他内容，也不要调用工具。",
    )

    assert result["mode"] == "react"
    assert "VELA_LIVE_OK" in result["text"]


def test_real_model_calls_stdio_mcp(tmp_path: Path) -> None:
    request_text = f"request-{secrets.token_hex(8)}"
    response_text = f"response-{secrets.token_hex(16)}"
    audit_path = tmp_path.parent / f"{tmp_path.name}-mcp-call.json"
    server = tmp_path / "live_mcp_server.py"
    server.write_text(
        """
import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("vela-live")

@mcp.tool()
def echo(text: str) -> str:
    response = os.environ["VELA_MCP_RESPONSE"]
    Path(os.environ["VELA_MCP_AUDIT"]).write_text(
        json.dumps({"text": text, "response": response}),
        encoding="utf-8",
    )
    return response

if __name__ == "__main__":
    mcp.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    _write_mcp_config(
        tmp_path,
        {
            "live": {
                "type": "stdio",
                "command": sys.executable,
                "args": [str(server)],
                "env": {
                    "VELA_MCP_RESPONSE": "${VELA_MCP_RESPONSE}",
                    "VELA_MCP_AUDIT": "${VELA_MCP_AUDIT}",
                },
            }
        },
    )

    result = _run_web(
        tmp_path,
        f"必须调用 mcp__live__echo，参数 text 必须是 {request_text}；然后只回复工具返回值。",
        extra_env={
            "VELA_MCP_RESPONSE": response_text,
            "VELA_MCP_AUDIT": str(audit_path),
        },
    )

    assert response_text in result["text"]
    assert json.loads(audit_path.read_text(encoding="utf-8")) == {
        "text": request_text,
        "response": response_text,
    }


def test_real_model_uses_browser_mcp(tmp_path: Path) -> None:
    if os.getenv("VELA_LIVE_BROWSER", "").lower() != "true":
        pytest.skip("set VELA_LIVE_BROWSER=true to run Chrome acceptance")
    _write_mcp_config(
        tmp_path,
        {
            "chrome-devtools": {
                "type": "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "chrome-devtools-mcp@1.6.0",
                    "--no-usage-statistics",
                    "--headless",
                ],
            }
        },
    )

    marker = f"browser-{secrets.token_hex(16)}"
    with _local_marker_page(marker) as (url, requested):
        result = _run_web(
            tmp_path,
            f"必须使用 Chrome DevTools MCP 打开 {url}；读取页面中的随机标记，然后只回复该标记。",
            timeout=360,
        )

    assert requested.is_set(), "the local page was not requested"
    assert marker in result["text"]


def test_real_model_executes_plan(tmp_path: Path) -> None:
    if os.getenv("VELA_LIVE_PLAN", "").lower() != "true":
        pytest.skip("set VELA_LIVE_PLAN=true to run real Plan acceptance")
    marker = f"plan-{secrets.token_hex(16)}"
    (tmp_path / "LIVE_PLAN_MARKER.txt").write_text(marker + "\n", encoding="utf-8")

    result = _run_web(
        tmp_path,
        "制定并执行一个计划：读取 LIVE_PLAN_MARKER.txt，最终答案必须包含文件中的标记。",
        mode="plan",
        timeout=360,
    )

    assert result["mode"] == "plan"
    assert marker in result["text"]


def _run_web(
    workspace: Path,
    prompt: str,
    *,
    mode: str = "react",
    timeout: int = 240,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    config = load_config()
    if not config.llm.api_key:
        pytest.fail("live acceptance requires VELA_API_KEY or a provider-specific API key")

    home = workspace / "home"
    home.mkdir(exist_ok=True)
    env = {
        "HOME": str(home),
        "VELA_API_KEY": config.llm.api_key,
        "VELA_PROVIDER": config.llm.provider,
        "VELA_MODEL": config.llm.model,
        "VELA_APPROVAL_MODE": "auto",
    }
    if config.llm.base_url:
        env["VELA_BASE_URL"] = config.llm.base_url
    if config.llm.context_window:
        env["VELA_CONTEXT_WINDOW"] = str(config.llm.context_window)
    env.update(extra_env or {})

    async def scenario() -> dict[str, Any]:
        manager = RuntimeManager(workspace)
        manager.project_trusted = True
        try:
            await manager.rebuild()
            runtime = manager.runtime
            assert runtime is not None, manager.error
            await runtime.send(prompt, mode)
            state = await asyncio.wait_for(_complete_run(runtime), timeout=timeout)
            assert state == TaskState.COMPLETED, runtime.controller.error
            assistant_text = "\n".join(
                str(message.content)
                for message in runtime.agent.history
                if message.role == "assistant" and isinstance(message.content, str)
            )
            return {"text": assistant_text, "mode": runtime.agent.mode}
        finally:
            await manager.close()

    with patch.dict(os.environ, env, clear=False):
        return asyncio.run(scenario())


async def _complete_run(runtime) -> TaskState | None:
    while runtime.controller.active:
        if runtime.controller.awaiting_plan_review:
            runtime.submit_plan_review("execute")
        await asyncio.sleep(0.05)
    return await runtime.controller.wait()


def _write_mcp_config(workspace: Path, servers: dict[str, Any]) -> None:
    config_dir = workspace / ".vela"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "mcp.json").write_text(
        json.dumps({"mcpServers": servers}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@contextmanager
def _local_marker_page(marker: str) -> Iterator[tuple[str, threading.Event]]:
    requested = threading.Event()

    class MarkerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            requested.set()
            body = f"<html><body><main>{marker}</main></body></html>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), MarkerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", requested
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
