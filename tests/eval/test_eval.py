from __future__ import annotations

import asyncio
import json

import pytest
from typer.testing import CliRunner

from vela.agent import Agent
from vela.config import load_config
from vela.entrypoints import cli
from vela.eval import EvalRunner, compare_results, load_suite, write_result
from vela.eval.models import CaseResult, EvalCase, SuiteResult
from vela.tools import ToolRegistry, get_builtin_tools


class WritingClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 4_000

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": "write_1",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps(
                            {"path": "answer.txt", "content": "Vela evaluation passed\n"}
                        ),
                    },
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return
        yield {"type": "text_delta", "text": "finished"}
        yield {"type": "message_end", "stop_reason": "end_turn"}
        yield {
            "type": "usage",
            "usage": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
        }


def test_eval_runner_uses_real_agent_tools_and_records_metrics(tmp_path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "smoke",
                "cases": [
                    {
                        "id": "write-answer",
                        "prompt": "write the answer",
                        "files": {},
                        "assertions": [
                            {"type": "file_contains", "path": "answer.txt", "value": "Vela"},
                            {"type": "response_contains", "value": "finished"},
                            {
                                "type": "command_succeeds",
                                "command": [
                                    "python",
                                    "-c",
                                    "from pathlib import Path; assert Path('answer.txt').exists()",
                                ],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    async def agent_factory(workspace, case: EvalCase) -> Agent:
        config = load_config(project_root=workspace)
        config.llm.api_key = "test"
        config.policy.hitl_mode = "never"
        config.features.context_compression = False
        registry = ToolRegistry()
        registry.register_all(get_builtin_tools())
        return Agent(
            llm_client=WritingClient(),
            tool_registry=registry,
            config=config,
            cwd=str(workspace),
            mode=case.mode,
        )

    result = asyncio.run(EvalRunner(agent_factory, tmp_path / "work").run(load_suite(suite_path)))

    assert result.success_rate == 1
    assert result.model == "fake-model"
    assert result.total_tokens == 11
    assert result.cases[0].tool_calls == 1
    assert result.cases[0].score == 1


def test_load_suite_rejects_fixture_paths_that_escape_workspace(tmp_path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "unsafe",
                "cases": [
                    {
                        "id": "escape",
                        "prompt": "write",
                        "files": {"../outside.txt": "bad"},
                        "assertions": [{"type": "file_exists", "path": "outside.txt"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    async def unused_factory(workspace, case):  # pragma: no cover
        raise AssertionError((workspace, case))

    with pytest.raises(ValueError, match="escapes workspace"):
        asyncio.run(EvalRunner(unused_factory, tmp_path / "work").run(load_suite(suite_path)))


def test_compare_results_identifies_regressions_and_improvements() -> None:
    baseline = _result(passed={"case-a": True, "case-b": False}, tokens=10).to_dict()
    current = _result(passed={"case-a": False, "case-b": True}, tokens=14).to_dict()

    comparison = compare_results(baseline, current)

    assert comparison["regressions"] == ["case-a"]
    assert comparison["improvements"] == ["case-b"]
    assert comparison["total_tokens_delta"] == 4


def test_eval_compare_command_fails_when_a_case_regresses(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    write_result(_result(passed={"case-a": True}, tokens=10), baseline)
    write_result(_result(passed={"case-a": False}, tokens=8), current)

    result = CliRunner().invoke(cli.app, ["eval", "compare", str(baseline), str(current)])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["regressions"] == ["case-a"]


def _result(*, passed: dict[str, bool], tokens: int) -> SuiteResult:
    cases = tuple(
        CaseResult(
            id=case_id,
            passed=value,
            score=float(value),
            duration_ms=100,
            total_tokens=tokens,
            turns=1,
            tool_calls=0,
            response="",
            error=None,
            assertions=(),
        )
        for case_id, value in passed.items()
    )
    return SuiteResult(
        suite="smoke",
        model="fake",
        provider="fake",
        started_at="2026-08-14T00:00:00+00:00",
        success_rate=sum(passed.values()) / len(passed),
        total_tokens=tokens,
        duration_p50_ms=100,
        duration_p95_ms=100,
        cases=cases,
    )
