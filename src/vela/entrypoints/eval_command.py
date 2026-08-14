"""CLI wiring for repeatable Agent evaluations."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from vela.agent import Agent
from vela.config import VelaConfig
from vela.eval import (
    EvalCase,
    EvalRunner,
    compare_results,
    load_builtin_suite,
    load_result,
    load_suite,
    write_result,
)
from vela.llm import create_llm_client
from vela.tools import ToolRegistry, get_builtin_tools


async def run_eval_suite(
    suite_path: Path | None,
    *,
    project_root: Path,
    config: VelaConfig,
    output: Path | None = None,
    workspace_root: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Run a suite with production Agent code and isolated built-in tools."""
    suite = load_suite(suite_path) if suite_path else load_builtin_suite()
    run_key = f"{datetime.now(UTC):%Y%m%d-%H%M%S-%f}-{uuid.uuid4().hex[:8]}"
    workspaces = workspace_root or project_root / ".vela" / "eval-work" / run_key

    async def create_agent(workspace: Path, case: EvalCase) -> Agent:
        case_config = deepcopy(config)
        case_config.policy.hitl_mode = "never"
        case_config.features.mcp = False
        case_config.features.memory = False
        case_config.features.skill = False
        registry = ToolRegistry()
        registry.register_all(get_builtin_tools())
        return Agent(
            llm_client=create_llm_client(case_config.llm),
            tool_registry=registry,
            config=case_config,
            cwd=str(workspace),
            mode=case.mode,
        )

    result = await EvalRunner(create_agent, workspaces).run(suite)
    suite_filename = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in suite.name
    )
    target = output or project_root / "eval-results" / f"{suite_filename}-{run_key}.json"
    write_result(result, target)
    return target, result.to_dict()


def compare_eval_files(baseline: Path, current: Path) -> dict[str, object]:
    return compare_results(load_result(baseline), load_result(current))


def format_eval_summary(result: dict[str, object], output: Path) -> str:
    summary = {
        "suite": result["suite"],
        "model": result["model"],
        "success_rate": result["success_rate"],
        "total_tokens": result["total_tokens"],
        "duration_p50_ms": result["duration_p50_ms"],
        "duration_p95_ms": result["duration_p95_ms"],
        "output": str(output),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)
