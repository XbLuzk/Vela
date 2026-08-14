"""Data contracts for evaluation suites and their results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Literal

AssertionType = Literal[
    "command_succeeds",
    "file_contains",
    "file_exists",
    "file_not_exists",
    "response_contains",
]


@dataclass(frozen=True, slots=True)
class EvalAssertion:
    type: AssertionType
    path: str = ""
    value: str = ""
    command: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    prompt: str
    mode: Literal["react", "plan"] = "react"
    files: dict[str, str] = field(default_factory=dict)
    assertions: tuple[EvalAssertion, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalSuite:
    name: str
    cases: tuple[EvalCase, ...]


@dataclass(frozen=True, slots=True)
class AssertionResult:
    type: AssertionType
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    id: str
    passed: bool
    score: float
    duration_ms: int
    total_tokens: int
    turns: int
    tool_calls: int
    response: str
    error: str | None
    assertions: tuple[AssertionResult, ...]


@dataclass(frozen=True, slots=True)
class SuiteResult:
    suite: str
    model: str
    provider: str
    started_at: str
    success_rate: float
    total_tokens: int
    duration_p50_ms: int
    duration_p95_ms: int
    cases: tuple[CaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_suite(path: str | Path) -> EvalSuite:
    """Load and validate a compact JSON task suite."""
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    return _parse_suite(value, fallback_name=source.stem)


def load_builtin_suite() -> EvalSuite:
    """Load the trusted evaluation suite shipped inside the Vela wheel."""
    resource = files("vela.eval").joinpath("coding-smoke.json")
    value = json.loads(resource.read_text(encoding="utf-8"))
    return _parse_suite(value, fallback_name="coding-smoke")


def _parse_suite(value: object, *, fallback_name: str) -> EvalSuite:
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError("Eval suite must contain a cases array")
    name = str(value.get("name") or fallback_name).strip()
    if not name:
        raise ValueError("Eval suite name must not be empty")
    cases = tuple(_parse_case(item) for item in value["cases"])
    if not cases:
        raise ValueError("Eval suite must contain at least one case")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Eval case IDs must be unique")
    return EvalSuite(name=name, cases=cases)


def _parse_case(value: object) -> EvalCase:
    if not isinstance(value, dict):
        raise ValueError("Each eval case must be an object")
    case_id = str(value.get("id") or "").strip()
    prompt = str(value.get("prompt") or "").strip()
    mode = str(value.get("mode") or "react")
    files = value.get("files") or {}
    assertions = value.get("assertions") or []
    if not case_id or not prompt:
        raise ValueError("Each eval case needs a non-empty id and prompt")
    if mode not in {"react", "plan"}:
        raise ValueError(f"Unsupported eval mode: {mode}")
    if not isinstance(files, dict) or not all(
        isinstance(key, str) and isinstance(content, str) for key, content in files.items()
    ):
        raise ValueError(f"Eval case {case_id} files must map paths to text")
    for path in files:
        _validate_relative_path(path, label=f"Eval case {case_id} fixture")
    if not isinstance(assertions, list) or not assertions:
        raise ValueError(f"Eval case {case_id} needs at least one assertion")
    return EvalCase(
        id=case_id,
        prompt=prompt,
        mode=mode,  # type: ignore[arg-type]
        files=dict(files),
        assertions=tuple(_parse_assertion(item) for item in assertions),
    )


def _parse_assertion(value: object) -> EvalAssertion:
    if not isinstance(value, dict):
        raise ValueError("Eval assertions must be objects")
    assertion_type = str(value.get("type") or "")
    supported = {
        "command_succeeds",
        "file_contains",
        "file_exists",
        "file_not_exists",
        "response_contains",
    }
    if assertion_type not in supported:
        raise ValueError(f"Unsupported eval assertion: {assertion_type}")
    command = value.get("command") or []
    if assertion_type == "command_succeeds" and (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item.strip() for item in command)
    ):
        raise ValueError("command_succeeds needs a non-empty command array")
    path = str(value.get("path") or "").strip()
    expected = str(value.get("value") or "")
    if assertion_type in {"file_contains", "file_exists", "file_not_exists"}:
        _validate_relative_path(path, label=assertion_type)
    if assertion_type in {"file_contains", "response_contains"} and not expected.strip():
        raise ValueError(f"{assertion_type} needs a non-empty value")
    return EvalAssertion(
        type=assertion_type,  # type: ignore[arg-type]
        path=path,
        value=expected,
        command=tuple(command),
    )


def _validate_relative_path(path: str, *, label: str) -> None:
    normalized = path.strip().replace("\\", "/")
    parsed = PurePosixPath(normalized)
    if not normalized or parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"{label} path escapes workspace: {path}")
