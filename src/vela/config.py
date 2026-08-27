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


@dataclass(slots=True)
class PolicyConfig:
    approval_mode: str = "ask"
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


LLM_ENV_FIELDS: tuple[tuple[str, str, Any], ...] = (
    ("VELA_API_KEY", "api_key", str),
    ("VELA_PROVIDER", "provider", str),
    ("VELA_MODEL", "model", str),
    ("VELA_BASE_URL", "base_url", str),
    ("VELA_CONTEXT_WINDOW", "context_window", int),
    ("VELA_MAX_TOKENS", "max_tokens", int),
    ("VELA_TEMPERATURE", "temperature", float),
)

PROVIDER_API_KEYS: dict[str, tuple[str, ...]] = {
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

FEATURE_ENV_FIELDS: tuple[tuple[str, str], ...] = (
    ("VELA_MCP", "mcp"),
    ("VELA_SKILL", "skill"),
    ("VELA_MEMORY", "memory"),
)


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
    warning_sink = warnings if warnings is not None else []
    root = Path(project_root).resolve() if project_root else None
    data = _load_config_files(root, include_project=include_project, warnings=warning_sink)
    data = _load_runtime_overrides(
        data,
        root=root,
        include_project=include_project,
        overrides=overrides,
        env=env if env is not None else os.environ,
        warnings=warning_sink,
    )
    return _build_config(data, include_project=include_project, warnings=warning_sink)


def get_config_paths(
    project_root: str | Path | None = None,
    *,
    include_project: bool = True,
) -> list[Path]:
    paths = [user_state_path("config.json")]
    if project_root and include_project:
        paths.append(vela_dir(Path(project_root).resolve()) / "config.json")
    return paths


def _load_config_files(
    root: Path | None,
    *,
    include_project: bool,
    warnings: list[str],
) -> dict[str, Any]:
    """Merge default, user, and optional project JSON configuration."""
    data = _config_to_dict(VelaConfig())
    for path in get_config_paths(root, include_project=include_project):
        loaded = _read_json(path, warnings)
        if loaded:
            data = _deep_merge(data, loaded)
    return data


def _load_runtime_overrides(
    data: dict[str, Any],
    *,
    root: Path | None,
    include_project: bool,
    overrides: dict[str, Any] | None,
    env: dict[str, str | None],
    warnings: list[str],
) -> dict[str, Any]:
    """Apply project dotenv, CLI values, migration, then process environment."""
    result = data
    if root is not None and include_project:
        dotenv = _read_env(root / ".env", warnings)
        if dotenv:
            result = _apply_env(result, dotenv, warnings, source=str(root / ".env"))
    if overrides:
        result = _deep_merge(result, overrides)
    result = _migrate_legacy_policy(result, warnings)
    return _apply_env(result, env, warnings, source="environment")


def _build_config(
    data: dict[str, Any],
    *,
    include_project: bool,
    warnings: list[str],
) -> VelaConfig:
    """Validate merged values and normalize paths used by persistent stores."""
    config = _dict_to_config(data, warnings)
    config.project_trusted = include_project
    config.memory.long_term_db_path = _expand_home(config.memory.long_term_db_path)
    config.tools.execution_journal_path = _expand_home(config.tools.execution_journal_path)
    return config


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
    _apply_typed_env(llm, env, LLM_ENV_FIELDS, warnings, source=source)
    _apply_provider_env(llm, env)
    _apply_feature_env(features, env, warnings, source=source)
    _apply_approval_env(policy, env, warnings, source=source)
    return result


def _apply_typed_env(
    target: dict[str, Any],
    env: dict[str, str | None],
    fields_to_apply: tuple[tuple[str, str, Any], ...],
    warnings: list[str],
    *,
    source: str,
) -> None:
    for env_key, config_key, caster in fields_to_apply:
        raw = env.get(env_key)
        if raw in (None, ""):
            continue
        try:
            target[config_key] = caster(raw)
        except (TypeError, ValueError):
            warnings.append(f"Ignored {env_key}={raw!r} from {source}: expected {caster.__name__}")


def _apply_provider_env(llm: dict[str, Any], env: dict[str, str | None]) -> None:
    provider = str(llm.get("provider") or "").lower()
    if not llm.get("api_key"):
        for provider_key in PROVIDER_API_KEYS.get(provider, ()):
            if env.get(provider_key):
                llm["api_key"] = env[provider_key]
                break

    provider_model_key = f"{provider.upper()}_MODEL" if provider else ""
    provider_base_url_key = f"{provider.upper()}_BASE_URL" if provider else ""
    if provider_model_key and env.get(provider_model_key):
        llm["model"] = env[provider_model_key]
    if provider_base_url_key and env.get(provider_base_url_key):
        llm["base_url"] = env[provider_base_url_key]


def _apply_feature_env(
    features: dict[str, Any],
    env: dict[str, str | None],
    warnings: list[str],
    *,
    source: str,
) -> None:
    for env_key, feature_key in FEATURE_ENV_FIELDS:
        raw = env.get(env_key)
        if raw == "false":
            features[feature_key] = False
        elif raw == "true":
            features[feature_key] = True
        elif raw not in (None, ""):
            warnings.append(f"Ignored {env_key}={raw!r} from {source}: expected true or false")


def _apply_approval_env(
    policy: dict[str, Any],
    env: dict[str, str | None],
    warnings: list[str],
    *,
    source: str,
) -> None:
    legacy_hitl = env.get("VELA_HITL")
    if legacy_hitl in {"always", "auto", "never"}:
        policy["approval_mode"] = _legacy_approval_mode(legacy_hitl)
        warnings.append(
            f"Migrated legacy VELA_HITL={legacy_hitl!r} from {source}; "
            "use VELA_APPROVAL_MODE=ask|auto"
        )
    elif legacy_hitl not in (None, ""):
        warnings.append(
            f"Ignored VELA_HITL={legacy_hitl!r} from {source}: expected always, auto, or never"
        )

    approval_mode = env.get("VELA_APPROVAL_MODE")
    if approval_mode in {"ask", "auto"}:
        policy["approval_mode"] = approval_mode
    elif approval_mode not in (None, ""):
        warnings.append(
            f"Ignored VELA_APPROVAL_MODE={approval_mode!r} from {source}: expected ask or auto"
        )


def _migrate_legacy_policy(data: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Translate persisted three-state HITL config without broadening permissions."""
    result = deepcopy(data)
    policy = result.get("policy")
    if not isinstance(policy, dict) or "hitl_mode" not in policy:
        return result
    legacy = policy.pop("hitl_mode")
    if legacy in {"always", "auto", "never"}:
        policy["approval_mode"] = _legacy_approval_mode(str(legacy))
        warnings.append(
            "Migrated policy.hitl_mode to policy.approval_mode; "
            "use ask or auto in future config files"
        )
    return result


def _legacy_approval_mode(value: str) -> str:
    return "auto" if value == "never" else "ask"


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
