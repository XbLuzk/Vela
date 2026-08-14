"""Execute fixed tasks through the real Agent boundary and score outcomes."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import subprocess
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vela.agent import Agent
from vela.eval.models import (
    AssertionResult,
    CaseResult,
    EvalAssertion,
    EvalCase,
    EvalSuite,
    SuiteResult,
)

AgentFactory = Callable[[Path, EvalCase], Awaitable[Agent]]


class EvalRunner:
    """Run every case in an isolated workspace."""

    def __init__(self, agent_factory: AgentFactory, workspace_root: str | Path) -> None:
        self.agent_factory = agent_factory
        self.workspace_root = Path(workspace_root).resolve()
        self._model = ""
        self._provider = ""

    async def run(self, suite: EvalSuite) -> SuiteResult:
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        results = [await self._run_case(case) for case in suite.cases]
        durations = [result.duration_ms for result in results]
        return SuiteResult(
            suite=suite.name,
            model=self._model,
            provider=self._provider,
            started_at=started_at,
            success_rate=round(sum(result.passed for result in results) / len(results), 4),
            total_tokens=sum(result.total_tokens for result in results),
            duration_p50_ms=_percentile(durations, 50),
            duration_p95_ms=_percentile(durations, 95),
            cases=tuple(results),
        )

    async def _run_case(self, case: EvalCase) -> CaseResult:
        workspace = _case_workspace(self.workspace_root, case.id)
        agent: Agent | None = None
        result = None
        response = ""
        error: str | None = None
        started = time.monotonic()
        try:
            if workspace.exists():
                shutil.rmtree(workspace)
            workspace.mkdir(parents=True)
            _write_fixtures(workspace, case.files)
            agent = await self.agent_factory(workspace, case)
            self._model = agent.llm_client.model_name
            self._provider = agent.llm_client.provider_name
            result = await agent.run_complete(case.prompt)
            response = result.text
        except Exception as exc:  # noqa: BLE001 - failed cases are evaluation data
            error = type(exc).__name__
        duration_ms = max(0, round((time.monotonic() - started) * 1_000))
        assertion_results = tuple(
            await asyncio.gather(
                *(
                    _score_assertion_safely(assertion, workspace, response)
                    for assertion in case.assertions
                )
            )
        )
        passed_count = sum(item.passed for item in assertion_results)
        return CaseResult(
            id=case.id,
            passed=passed_count == len(assertion_results) and error is None,
            score=round(passed_count / len(assertion_results), 4),
            duration_ms=duration_ms,
            total_tokens=result.total_tokens if result else 0,
            turns=result.turns if result else 0,
            tool_calls=(agent.last_run_trace.tool_calls if agent and agent.last_run_trace else 0),
            response=response,
            error=error,
            assertions=assertion_results,
        )


def write_result(result: SuiteResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def load_result(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_result(value)
    return value


def _validate_result(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("Invalid eval result")
    _require_non_empty_string(value, "suite")
    _require_rate(value, "success_rate")
    for field in ("total_tokens", "duration_p50_ms", "duration_p95_ms"):
        _require_nonnegative_int(value, field)
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Invalid eval result cases")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Invalid eval result case")
        _require_non_empty_string(case, "id")
        if not isinstance(case.get("passed"), bool):
            raise ValueError("Invalid eval result case passed")


def _require_non_empty_string(value: dict[str, Any], field: str) -> None:
    if not isinstance(value.get(field), str) or not value[field].strip():
        raise ValueError(f"Invalid eval result {field}")


def _require_rate(value: dict[str, Any], field: str) -> None:
    item = value.get(field)
    if (
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        or not 0 <= item <= 1
    ):
        raise ValueError(f"Invalid eval result {field}")


def _require_nonnegative_int(value: dict[str, Any], field: str) -> None:
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise ValueError(f"Invalid eval result {field}")


def compare_results(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Return concise quality, cost, and latency deltas between two runs."""
    if baseline.get("suite") != current.get("suite"):
        raise ValueError(
            f"Eval suites differ: {baseline.get('suite')!r} != {current.get('suite')!r}"
        )
    baseline_cases = {str(case["id"]): case for case in baseline["cases"]}
    current_cases = {str(case["id"]): case for case in current["cases"]}
    if len(baseline_cases) != len(baseline["cases"]) or len(current_cases) != len(current["cases"]):
        raise ValueError("Eval result case IDs must be unique")
    baseline_ids = set(baseline_cases)
    current_ids = set(current_cases)
    if baseline_ids != current_ids:
        removed = sorted(baseline_ids - current_ids)
        added = sorted(current_ids - baseline_ids)
        raise ValueError(f"Eval case sets differ: removed={removed}, added={added}")
    shared = sorted(baseline_ids)
    regressions = [
        case_id
        for case_id in shared
        if bool(baseline_cases[case_id].get("passed"))
        and not bool(current_cases[case_id].get("passed"))
    ]
    improvements = [
        case_id
        for case_id in shared
        if not bool(baseline_cases[case_id].get("passed"))
        and bool(current_cases[case_id].get("passed"))
    ]
    return {
        "suite": current.get("suite"),
        "success_rate_delta": round(
            float(current.get("success_rate") or 0) - float(baseline.get("success_rate") or 0),
            4,
        ),
        "total_tokens_delta": int(current.get("total_tokens") or 0)
        - int(baseline.get("total_tokens") or 0),
        "duration_p50_ms_delta": int(current.get("duration_p50_ms") or 0)
        - int(baseline.get("duration_p50_ms") or 0),
        "duration_p95_ms_delta": int(current.get("duration_p95_ms") or 0)
        - int(baseline.get("duration_p95_ms") or 0),
        "regressions": regressions,
        "improvements": improvements,
    }


async def _score_assertion(
    assertion: EvalAssertion,
    workspace: Path,
    response: str,
) -> AssertionResult:
    if assertion.type == "response_contains":
        passed = assertion.value.lower() in response.lower()
        return AssertionResult(assertion.type, passed, f"response contains {assertion.value!r}")
    if assertion.type == "command_succeeds":
        return await asyncio.to_thread(_run_command_assertion, assertion, workspace)

    path = _safe_path(workspace, assertion.path)
    if assertion.type == "file_exists":
        return AssertionResult(assertion.type, path.is_file(), f"file exists: {assertion.path}")
    if assertion.type == "file_not_exists":
        return AssertionResult(assertion.type, not path.exists(), f"file absent: {assertion.path}")
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        content = ""
    passed = assertion.value in content
    return AssertionResult(
        assertion.type,
        passed,
        f"{assertion.path} contains {assertion.value!r}",
    )


async def _score_assertion_safely(
    assertion: EvalAssertion,
    workspace: Path,
    response: str,
) -> AssertionResult:
    try:
        return await _score_assertion(assertion, workspace, response)
    except (OSError, RuntimeError, ValueError) as exc:
        return AssertionResult(
            assertion.type,
            False,
            f"assertion failed: {type(exc).__name__}",
        )


def _run_command_assertion(assertion: EvalAssertion, workspace: Path) -> AssertionResult:
    try:
        completed = subprocess.run(
            assertion.command,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        detail = f"{' '.join(assertion.command)} exited {completed.returncode}"
        return AssertionResult(assertion.type, completed.returncode == 0, detail)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return AssertionResult(assertion.type, False, f"command failed: {type(exc).__name__}")


def _write_fixtures(workspace: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = _safe_path(workspace, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _safe_path(workspace: Path, relative: str) -> Path:
    path = (workspace / relative).resolve()
    if not path.is_relative_to(workspace):
        raise ValueError(f"Eval path escapes workspace: {relative}")
    return path


def _case_workspace(root: Path, case_id: str) -> Path:
    safe_id = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in case_id
    )
    return root / safe_id


def _percentile(values: list[int], percentile: int) -> int:
    ordered = sorted(values)
    rank = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return ordered[rank]
