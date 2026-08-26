from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from vela.storage import user_state_path, vela_dir


@dataclass(slots=True)
class LlmConfig:
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    api_key: str = ""
    base_url: str | None = None
    context_window: int | None = None
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: float = 120.0


@dataclass(slots=True)
class ToolsConfig:
    timeout: float = 60.0
    max_concurrent_read: int = 4
    execution_journal_path: str = "~/.vela/tool-executions.sqlite"


@dataclass(slots=True)
class MemoryConfig:
    max_conversation_history: int = 100
    long_term_db_path: str = "~/.vela/memory.db"
    max_long_term_entries: int = 1_000
    max_memory_chars: int = 8_000
    recall_limit: int = 6
    recall_min_score: float = 0.05
    compression_threshold: float = 0.8
    compression_target: float = 0.55
    compression_reserve_tokens: int = 1_024
    min_recent_messages: int = 6
    summary_max_chars: int = 6_000


@dataclass(slots=True)
class PolicyConfig:
    hitl_mode: str = "auto"
    path_guard_enabled: bool = True
    command_guard_enabled: bool = True
    command_blacklist: list[str] = field(
        default_factory=lambda: [
            "sudo",
            "rm -rf /",
            "rm -rf ~",
            "mkfs",
            "dd if=/dev/zero",
            ":(){:|:&};:",
            "chmod -R 777 /",
            "curl | sh",
            "curl|sh",
            "shutdown",
            "reboot",
        ]
    )
    audit_log_path: str = "~/.vela/audit.jsonl"


@dataclass(slots=True)
class PromptConfig:
    personality: str = "default"
    agent_mode: str = "react"
    custom_prompt_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FeatureConfig:
    mcp: bool = True
    skill: bool = True
    memory: bool = True
    audit_log: bool = True
    context_compression: bool = True


@dataclass(slots=True)
class VelaConfig:
    llm: LlmConfig = field(default_factory=LlmConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    project_trusted: bool = True


def load_config(
    project_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str | None] | None = None,
    *,
    include_project: bool = True,
    warnings: list[str] | None = None,
) -> VelaConfig:
    """Build the effective configuration from defaults, files, env, and overrides.

    Unreadable or malformed configuration sources are skipped so a broken file
    cannot make Vela unusable, but every skipped source and every rejected value
    is appended to *warnings* so the caller can report it instead of leaving the
    user with silently ignored settings.
    """
    sink = warnings if warnings is not None else []
    env_map = env if env is not None else os.environ
    data = _config_to_dict(VelaConfig())

    user_config = _read_json(user_state_path("config.json"), sink)
    if user_config:
        data = _deep_merge(data, user_config)

    root = Path(project_root).resolve() if project_root else None
    if root and include_project:
        project_config = _read_json(vela_dir(root) / "config.json", sink)
        if project_config:
            data = _deep_merge(data, project_config)
        project_env = _read_env(root / ".env", sink)
        if project_env:
            data = _apply_env(data, project_env, sink, source=str(root / ".env"))

    if overrides:
        data = _deep_merge(data, overrides)

    data = _apply_env(data, env_map, sink, source="environment")
    config = _dict_to_config(data, sink)
    config.project_trusted = include_project
    config.memory.long_term_db_path = _expand_home(config.memory.long_term_db_path)
    config.policy.audit_log_path = _expand_home(config.policy.audit_log_path)
    config.tools.execution_journal_path = _expand_home(config.tools.execution_journal_path)
    return config


def get_config_paths(
    project_root: str | Path | None = None,
    *,
    include_project: bool = True,
) -> list[Path]:
    paths = [user_state_path("config.json")]
    if project_root and include_project:
        paths.append(vela_dir(Path(project_root).resolve()) / "config.json")
    return paths


def config_to_public_dict(config: VelaConfig) -> dict[str, Any]:
    data = _config_to_dict(config)
    if data.get("llm", {}).get("api_key"):
        data["llm"]["api_key"] = "***"
    return data


def _read_json(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        warnings.append(f"Ignored config file {path}: {exc}")
        return None
    except json.JSONDecodeError as exc:
        warnings.append(f"Ignored config file {path}: invalid JSON at line {exc.lineno}: {exc.msg}")
        return None
    if not isinstance(raw, dict):
        warnings.append(f"Ignored config file {path}: expected a JSON object")
        return None
    return raw


def _read_env(path: Path, warnings: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        warnings.append(f"Ignored env file {path}: {exc}")
        return result
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        result[key] = value
    return result


def _apply_env(
    data: dict[str, Any],
    env: dict[str, str | None],
    warnings: list[str],
    *,
    source: str,
) -> dict[str, Any]:
    result = deepcopy(data)
    llm = result.setdefault("llm", {})
    features = result.setdefault("features", {})
    policy = result.setdefault("policy", {})

    mappings: list[tuple[str, str, Any]] = [
        ("VELA_API_KEY", "api_key", str),
        ("VELA_PROVIDER", "provider", str),
        ("VELA_MODEL", "model", str),
        ("VELA_BASE_URL", "base_url", str),
        ("VELA_CONTEXT_WINDOW", "context_window", int),
        ("VELA_MAX_TOKENS", "max_tokens", int),
        ("VELA_TEMPERATURE", "temperature", float),
    ]
    for env_key, config_key, caster in mappings:
        raw = env.get(env_key)
        if raw in (None, ""):
            continue
        try:
            llm[config_key] = caster(raw)
        except (TypeError, ValueError):
            warnings.append(f"Ignored {env_key}={raw!r} from {source}: expected {caster.__name__}")

    provider = str(llm.get("provider") or "").lower()
    if not llm.get("api_key"):
        provider_key_map = {
            "deepseek": ("DEEPSEEK_API_KEY",),
            "glm": ("ZAI_API_KEY", "GLM_API_KEY"),
            "zhipu": ("ZAI_API_KEY", "GLM_API_KEY"),
            "step": ("STEP_API_KEY",),
            "kimi": ("KIMI_API_KEY",),
            "moonshot": ("KIMI_API_KEY",),
            "freellmapi": ("FREELLMAPI_API_KEY",),
            "xfyun": ("XFYUN_API_KEY",),
            "agnes": ("AGNES_API_KEY",),
        }
        for provider_key in provider_key_map.get(provider, ()):
            if env.get(provider_key):
                llm["api_key"] = env[provider_key]
                break

    provider_model_key = f"{provider.upper()}_MODEL" if provider else ""
    provider_base_url_key = f"{provider.upper()}_BASE_URL" if provider else ""
    if provider_model_key and env.get(provider_model_key):
        llm["model"] = env[provider_model_key]
    if provider_base_url_key and env.get(provider_base_url_key):
        llm["base_url"] = env[provider_base_url_key]

    for env_key, feature_key in [
        ("VELA_MCP", "mcp"),
        ("VELA_SKILL", "skill"),
        ("VELA_MEMORY", "memory"),
    ]:
        raw = env.get(env_key)
        if raw == "false":
            features[feature_key] = False
        elif raw == "true":
            features[feature_key] = True
        elif raw not in (None, ""):
            warnings.append(f"Ignored {env_key}={raw!r} from {source}: expected true or false")

    hitl = env.get("VELA_HITL")
    if hitl in {"always", "auto", "never"}:
        policy["hitl_mode"] = hitl
    elif hitl not in (None, ""):
        warnings.append(
            f"Ignored VELA_HITL={hitl!r} from {source}: expected always, auto, or never"
        )

    return result


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(target)
    for key, value in source.items():
        if value is None:
            continue
        old = result.get(key)
        if isinstance(old, dict) and isinstance(value, dict):
            result[key] = _deep_merge(old, value)
        else:
            result[key] = deepcopy(value)
    return result


def _config_to_dict(config: VelaConfig) -> dict[str, Any]:
    data = asdict(config)
    data.pop("project_trusted", None)
    return data


def _dict_to_config(data: dict[str, Any], warnings: list[str]) -> VelaConfig:
    return VelaConfig(
        llm=_section(LlmConfig, data, "llm", warnings),
        tools=_section(ToolsConfig, data, "tools", warnings),
        memory=_section(MemoryConfig, data, "memory", warnings),
        policy=_section(PolicyConfig, data, "policy", warnings),
        prompt=_section(PromptConfig, data, "prompt", warnings),
        features=_section(FeatureConfig, data, "features", warnings),
    )


def _section(factory: Any, data: dict[str, Any], name: str, warnings: list[str]) -> Any:
    raw = data.get(name, {})
    if not isinstance(raw, dict):
        warnings.append(f"Ignored config section {name!r}: expected an object")
        return factory()
    known = {f.name for f in fields(factory)}
    unknown = sorted(set(raw) - known)
    if unknown:
        warnings.append(f"Ignored unknown {name} config keys: {', '.join(unknown)}")
    return factory(**{key: value for key, value in raw.items() if key in known})


def _expand_home(path: str) -> str:
    return str(Path(path).expanduser())
