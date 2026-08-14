from __future__ import annotations

import asyncio
import json

import pytest
from typer.testing import CliRunner

from vela.agent import Agent
from vela.config import load_config
from vela.entrypoints import cli
from vela.eval import (
    EvalRunner,
    compare_results,
    load_builtin_suite,
    load_result,
    load_suite,
    write_result,
)
from vela.eval.models import CaseResult, EvalAssertion, EvalCase, EvalSuite, SuiteResult
from vela.tools import ToolRegistry, get_builtin_tools


class WritingClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 20_000

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


def test_builtin_suite_is_packaged_and_loadable() -> None:
    suite = load_builtin_suite()

    assert suite.name == "coding-smoke"
    assert len(suite.cases) == 3


@pytest.mark.parametrize(
    "assertion",
    [
        {"type": "response_contains", "value": ""},
        {"type": "file_contains", "path": "answer.txt", "value": ""},
        {"type": "file_exists", "path": ""},
        {"type": "command_succeeds", "command": [""]},
    ],
)
def test_load_suite_rejects_empty_assertion_inputs(tmp_path, assertion) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "invalid",
                "cases": [{"id": "case", "prompt": "work", "assertions": [assertion]}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_suite(suite_path)


def test_compare_results_rejects_different_suite_or_case_sets() -> None:
    baseline = _result(passed={"case-a": True, "case-b": True}, tokens=10).to_dict()
    different_suite = {**baseline, "suite": "other"}
    removed_case = _result(passed={"case-a": True}, tokens=10).to_dict()

    with pytest.raises(ValueError, match="suites differ"):
        compare_results(baseline, different_suite)
    with pytest.raises(ValueError, match="case sets differ"):
        compare_results(baseline, removed_case)


def test_eval_runner_isolates_factory_and_assertion_failures(tmp_path) -> None:
    suite = EvalSuite(
        name="isolation",
        cases=(
            EvalCase(
                id="factory-fails",
                prompt="work",
                assertions=(EvalAssertion(type="file_exists", path="missing.txt"),),
            ),
            EvalCase(
                id="assertion-fails",
                prompt="work",
                assertions=(EvalAssertion(type="file_exists", path="../outside.txt"),),
            ),
        ),
    )

    async def factory(workspace, case):
        if case.id == "factory-fails":
            raise RuntimeError("provider unavailable")
        config = load_config(project_root=workspace)
        config.llm.api_key = "test"
        config.features.context_compression = False
        return Agent(
            llm_client=WritingClient(),
            tool_registry=ToolRegistry(),
            config=config,
            cwd=str(workspace),
        )

    result = asyncio.run(EvalRunner(factory, tmp_path / "work").run(suite))

    assert [case.id for case in result.cases] == ["factory-fails", "assertion-fails"]
    assert result.cases[0].error == "RuntimeError"
    assert not result.cases[1].assertions[0].passed
    assert result.success_rate == 0


def test_eval_runner_contains_fixture_setup_failure_and_continues(tmp_path) -> None:
    suite = EvalSuite(
        name="fixture-isolation",
        cases=(
            EvalCase(
                id="conflicting-fixtures",
                prompt="work",
                files={"a": "file", "a/nested.txt": "cannot create"},
                assertions=(EvalAssertion(type="file_exists", path="a"),),
            ),
            EvalCase(
                id="later-case",
                prompt="work",
                assertions=(EvalAssertion(type="response_contains", value="finished"),),
            ),
        ),
    )

    async def factory(workspace, case):  # noqa: ARG001
        config = load_config(project_root=workspace)
        config.llm.api_key = "test"
        config.features.context_compression = False
        return Agent(
            llm_client=WritingClient(),
            tool_registry=ToolRegistry(),
            config=config,
            cwd=str(workspace),
        )

    result = asyncio.run(EvalRunner(factory, tmp_path / "work").run(suite))

    assert [case.id for case in result.cases] == ["conflicting-fixtures", "later-case"]
    assert result.cases[0].error == "FileExistsError"
    assert result.cases[1].error is None
    assert result.cases[1].passed


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"suite": "smoke", "cases": []},
        {
            "suite": "smoke",
            "success_rate": 1.0,
            "total_tokens": 1,
            "duration_p50_ms": 1,
            "duration_p95_ms": 1,
            "cases": [],
        },
        {
            "suite": "smoke",
            "success_rate": "one",
            "total_tokens": 1,
            "duration_p50_ms": 1,
            "duration_p95_ms": 1,
            "cases": [{"id": "case", "passed": True}],
        },
        {
            "suite": "smoke",
            "success_rate": 1.0,
            "total_tokens": 1,
            "duration_p50_ms": 1,
            "duration_p95_ms": 1,
            "cases": [{}],
        },
        {
            "suite": "smoke",
            "success_rate": 1.0,
            "total_tokens": 1,
            "duration_p50_ms": 1,
            "duration_p95_ms": 1,
            "cases": [{"id": "case", "passed": "yes"}],
        },
    ],
)
def test_load_result_normalizes_malformed_result_shapes_to_value_error(tmp_path, payload) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_result(path)


def test_eval_compare_command_fails_when_a_case_regresses(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    write_result(_result(passed={"case-a": True}, tokens=10), baseline)
    write_result(_result(passed={"case-a": False}, tokens=8), current)

    result = CliRunner().invoke(cli.app, ["eval", "compare", str(baseline), str(current)])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["regressions"] == ["case-a"]


def test_eval_run_requires_explicit_trust_for_custom_suite(tmp_path, monkeypatch) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("VELA_API_KEY", "test")

    result = CliRunner().invoke(
        cli.app,
        ["eval", "run", str(suite), "--cwd", str(tmp_path)],
        terminal_width=160,
    )

    assert result.exit_code == 2
    assert "--allow-code-execution" in result.output


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
