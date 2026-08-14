"""cli.py — Vela terminal AI agent CLI entry point.

Inspired by Claude Code: a terminal-native AI agent that understands your
workspace, executes commands, edits files, and answers questions — all from
the command line.

Usage modes::

    # Interactive REPL (default)
    vela
    vela --api-key sk-xxx
    vela --model deepseek-chat --provider deepseek

    # Single-shot prompt
    vela -p "list all Python files"
    vela -p "refactor this module" --mode plan

    # Tooling
    vela doctor              # system health check
    vela mcp init-chrome     # Configure Chrome DevTools MCP
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from vela import __version__
from vela.agent import Agent
from vela.bootstrap import build_tool_registry
from vela.branding import CLI_NAME, PRODUCT_NAME
from vela.config import get_config_paths, load_config
from vela.entrypoints.eval_command import (
    compare_eval_files,
    format_eval_summary,
    run_eval_suite,
)
from vela.entrypoints.repl import start_repl
from vela.entrypoints.trace_command import show_run_traces
from vela.llm import create_llm_client
from vela.mcp import load_mcp_server_specs, write_chrome_devtools_config
from vela.run_trace import RunTraceStore
from vela.trust import (
    ProjectTrustStore,
    has_trust_sensitive_resources,
    resolve_project_trust,
)

app = typer.Typer(
    name=CLI_NAME,
    help=f"{PRODUCT_NAME} — terminal AI agent for focused project work",
    invoke_without_command=True,
    no_args_is_help=False,
)
mcp_app = typer.Typer(help="External MCP server management")
eval_app = typer.Typer(help="Repeatable Agent task evaluation")
app.add_typer(mcp_app, name="mcp")
app.add_typer(eval_app, name="eval")
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{CLI_NAME} {__version__}")
        raise typer.Exit()


def _resolve_cli_project_trust(
    root: Path,
    *,
    interactive: bool,
    override: bool | None,
) -> bool:
    if override is not None:
        return override
    trust_store = ProjectTrustStore()
    saved = trust_store.get(root)
    if saved is not None:
        return saved
    if not has_trust_sensitive_resources(root):
        return True
    return resolve_project_trust(
        root,
        interactive=interactive,
        override=None,
        store=trust_store,
        prompt=lambda path: typer.confirm(
            f"Trust project {path}? This enables its .env, Vela config, MCP servers, and Skills",
            default=False,
        ),
    )


@app.callback()
def main(
    ctx: typer.Context,
    # --- Interaction mode ---
    prompt: Annotated[
        str | None,
        typer.Option("-p", "--prompt", help="Single prompt (non-interactive mode)"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume the most recent interactive session"),
    ] = False,
    # --- LLM configuration ---
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="LLM API key (overrides env/config)"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("-m", "--model", help="Override LLM model name"),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Override LLM provider (e.g. deepseek, glm)"),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="Override LLM API base URL"),
    ] = None,
    # --- Agent mode ---
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="Agent mode: react or plan"),
    ] = None,
    # --- Output ---
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit result and usage as JSON (single-prompt only)"),
    ] = False,
    # --- Workspace ---
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory (default: current dir)"),
    ] = None,
    trust_project: Annotated[
        bool | None,
        typer.Option(
            "--trust-project/--no-trust-project",
            help="Allow or ignore project-local config, MCP servers, and Skills for this run",
        ),
    ] = None,
    # --- Version ---
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version"),
    ] = False,
) -> None:
    """Vela — a terminal AI agent that works in your workspace.

    Start an interactive session (default), run a single prompt with -p,
    or use the subcommands for diagnostics and MCP management.
    """
    _ = version
    if ctx.invoked_subcommand is not None:
        return

    root = (cwd or Path.cwd()).resolve()
    if prompt is not None and resume:
        raise typer.BadParameter(
            "--resume is available only for interactive sessions",
            param_hint="--resume",
        )
    project_trusted = _resolve_cli_project_trust(
        root,
        interactive=prompt is None,
        override=trust_project,
    )

    # Build overrides dict from all explicit CLI flags.
    overrides: dict = {}
    llm_overrides: dict = {}
    if api_key is not None:
        llm_overrides["api_key"] = api_key
    if provider is not None:
        llm_overrides["provider"] = provider
    if model is not None:
        llm_overrides["model"] = model
    if base_url is not None:
        llm_overrides["base_url"] = base_url
    if llm_overrides:
        overrides["llm"] = llm_overrides
    config = load_config(
        project_root=root,
        overrides=overrides,
        include_project=project_trusted,
    )

    if prompt is not None:
        selected_mode = (mode or config.prompt.agent_mode or "react").lower()
        if selected_mode not in {"react", "plan"}:
            raise typer.BadParameter("mode must be react or plan", param_hint="--mode")
        asyncio.run(
            _run_prompt(
                prompt,
                str(root),
                config,
                mode=selected_mode,
                json_output=json_output,
            )
        )
    else:
        asyncio.run(start_repl(str(root), config, resume=resume))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command("doctor")
def doctor(
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    """Inspect the system: Python, uv, Node, API key, and config."""
    root = (cwd or Path.cwd()).resolve()
    project_trusted = _resolve_cli_project_trust(root, interactive=False, override=None)
    config = load_config(project_root=root, include_project=project_trusted)
    checks = {
        "python": sys.version.split()[0],
        "uv": shutil.which("uv") or "missing",
        "node": _version_of("node"),
        "npx": shutil.which("npx") or "missing",
        "rg": shutil.which("rg") or "missing",
        "api_key": "configured" if config.llm.api_key else "missing",
        "provider": config.llm.provider,
        "model": config.llm.model,
        "cwd": str(root),
        "project_trusted": project_trusted,
        "config_paths": [
            str(path) for path in get_config_paths(root, include_project=project_trusted)
        ],
    }
    console.print_json(json.dumps(checks, ensure_ascii=False))


@app.command("trace")
def trace_command(
    reference: Annotated[
        str | None,
        typer.Argument(help="Run ID, unique prefix, or list number"),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Number of recent runs")] = 20,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
) -> None:
    """List recent Agent runs or inspect one persisted trace."""
    store = RunTraceStore()
    found = show_run_traces(
        console,
        store,
        reference=reference or "",
        limit=max(1, limit),
        json_output=json_output,
    )
    if store.last_warning or (reference and not found):
        raise typer.Exit(1)


@eval_app.command("run")
def eval_run(
    suite: Annotated[
        Path | None,
        typer.Argument(help="Optional custom JSON suite; defaults to Vela's coding smoke suite"),
    ] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Project root")] = None,
    output: Annotated[Path | None, typer.Option("--output", help="Result JSON path")] = None,
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", help="Directory for isolated case workspaces"),
    ] = None,
    allow_code_execution: Annotated[
        bool,
        typer.Option(
            "--allow-code-execution",
            help="Trust a custom suite that can instruct the Agent and run assertion commands",
        ),
    ] = False,
) -> None:
    """Run fixed tasks and record success, latency, tokens, and tool calls."""
    root = (cwd or Path.cwd()).resolve()
    project_trusted = _resolve_cli_project_trust(root, interactive=False, override=None)
    config = load_config(project_root=root, include_project=project_trusted)
    if not config.llm.api_key:
        raise typer.BadParameter("LLM API key is required to run an evaluation")
    if suite is not None and not allow_code_execution:
        raise typer.BadParameter(
            "Custom eval suites can execute Agent tools and assertion commands; "
            "review the file and pass --allow-code-execution",
            param_hint="suite",
        )
    target, result = asyncio.run(
        run_eval_suite(
            suite.resolve() if suite else None,
            project_root=root,
            config=config,
            output=output.resolve() if output else None,
            workspace_root=workspace.resolve() if workspace else None,
        )
    )
    typer.echo(format_eval_summary(result, target))
    if float(result["success_rate"]) < 1:
        raise typer.Exit(1)


@eval_app.command("compare")
def eval_compare(
    baseline: Annotated[Path, typer.Argument(help="Baseline result JSON")],
    current: Annotated[Path, typer.Argument(help="Current result JSON")],
) -> None:
    """Compare two evaluation runs and fail when a case regresses."""
    try:
        comparison = compare_eval_files(baseline.resolve(), current.resolve())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison["regressions"]:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# MCP subcommands
# ---------------------------------------------------------------------------


@mcp_app.command("init-chrome")
def mcp_init_chrome(
    scope: Annotated[
        str,
        typer.Option("--scope", help="Config scope: user or project"),
    ] = "project",
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
    browser_url: Annotated[
        str | None,
        typer.Option("--browser-url", help="Connect to an existing Chrome remote debugging URL"),
    ] = None,
    headless: Annotated[bool, typer.Option("--headless", help="Start Chrome headless")] = False,
    slim: Annotated[bool, typer.Option("--slim", help="Use Chrome DevTools slim mode")] = False,
) -> None:
    """Write Chrome DevTools MCP config to the project or user scope."""
    if scope not in {"user", "project"}:
        raise typer.BadParameter("scope must be user or project")
    root = None if scope == "user" else (cwd or Path.cwd()).resolve()
    path = write_chrome_devtools_config(
        scope_root=root,
        browser_url=browser_url,
        headless=headless,
        slim=slim,
    )
    typer.echo(f"Wrote Chrome DevTools MCP config to {path}")


@mcp_app.command("list")
def mcp_list(
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    """List configured MCP servers."""
    root = (cwd or Path.cwd()).resolve()
    specs = load_mcp_server_specs(root)
    if not specs:
        typer.echo("No MCP servers configured.")
        return
    for spec in specs.values():
        target = spec.url or f"{spec.command} {' '.join(spec.args)}".strip()
        typer.echo(f"{spec.name}\t{spec.type}\t{target}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _run_prompt(
    prompt: str,
    cwd: str,
    config,
    *,
    mode: str = "react",
    json_output: bool = False,
) -> None:
    """Execute a single prompt and print the result."""
    if not config.llm.api_key:
        typer.echo(
            "Fatal error: LLM API key is not configured. Set it with --api-key, "
            "via the VELA_API_KEY environment variable, or in "
            "~/.vela/config.json or .vela/config.json.",
            err=True,
        )
        raise typer.Exit(1)
    registry, manager = await build_tool_registry(config=config, cwd=cwd)
    if manager and manager.last_errors:
        for name, error in manager.last_errors.items():
            typer.echo(f"MCP server {name} failed to load: {error}", err=True)
    agent = Agent(
        llm_client=create_llm_client(config.llm),
        tool_registry=registry,
        config=config,
        cwd=cwd,
        mode=mode,
        trace_store=RunTraceStore(),
    )
    try:
        result = await agent.run_complete(prompt)
    except Exception as exc:  # noqa: BLE001 - CLI should report model/config errors cleanly
        if agent.last_run_trace_warning:
            typer.echo(agent.last_run_trace_warning, err=True)
        typer.echo(f"Fatal error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if agent.last_run_trace_warning:
        typer.echo(agent.last_run_trace_warning, err=True)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "text": result.text,
                    "run_id": agent.last_run_trace.run_id if agent.last_run_trace else None,
                    "status": agent.last_run_trace.status if agent.last_run_trace else None,
                    "mode": mode,
                    "turns": result.turns,
                    "total_tokens": result.total_tokens,
                    "usage": result.usage.to_dict(),
                },
                ensure_ascii=False,
            )
        )
    else:
        typer.echo(result.text)


def _version_of(command: str) -> str:
    if not shutil.which(command):
        return "missing"
    try:
        result = subprocess.run(
            [command, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001
        return "unknown"
    return (result.stdout or result.stderr).strip() or "unknown"
