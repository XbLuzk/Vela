from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from vela.providers import provider_api_key_envs
from vela.storage import user_state_path, write_private_text


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
    agent_mode: str = "react"
    custom_prompt_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VelaConfig:
    llm: LlmConfig = field(default_factory=LlmConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
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
USER_LLM_FIELDS = tuple(dict.fromkeys(config_key for _, config_key, _ in LLM_ENV_FIELDS))


def load_config(
    env: dict[str, str | None] | None = None,
    *,
    project_trusted: bool = True,
    warnings: list[str] | None = None,
) -> VelaConfig:
    """Build config in one order: defaults, user file, then environment.

    Unreadable or malformed configuration sources are skipped so a broken file
    cannot make Vela unusable, but every skipped source and every rejected value
    is appended to *warnings* so the caller can report it instead of leaving the
    user with silently ignored settings.
    """
    warning_sink = warnings if warnings is not None else []
    data = _config_to_dict(VelaConfig())
    user_config = _read_json(user_state_path("config.json"), warning_sink)
    if user_config:
        data = _deep_merge(data, user_config)

    environment = env if env is not None else os.environ
    data = _apply_env(data, environment, warning_sink)
    protected = {
        config_key
        for env_key, config_key, _caster in LLM_ENV_FIELDS
        if environment.get(env_key) not in (None, "")
    }
    _apply_provider_env(
        data.setdefault("llm", {}),
        environment,
        protected=protected,
    )
    return _build_config(data, project_trusted=project_trusted, warnings=warning_sink)


def _build_config(
    data: dict[str, Any],
    *,
    project_trusted: bool,
    warnings: list[str],
) -> VelaConfig:
    """Validate merged values and normalize paths used by persistent stores."""
    config = _dict_to_config(data, warnings)
    config.project_trusted = project_trusted
    config.memory.long_term_db_path = _expand_home(config.memory.long_term_db_path)
    config.tools.execution_journal_path = _expand_home(config.tools.execution_journal_path)
    return config


def config_to_public_dict(config: VelaConfig) -> dict[str, Any]:
    data = _config_to_dict(config)
    if data.get("llm", {}).get("api_key"):
        data["llm"]["api_key"] = "***"
    return data


def update_user_config(values: dict[str, Any]) -> None:
    """Merge settings from the Web form into the user-level config file."""
    path = user_state_path("config.json")
    current = _read_json(path, []) or {}

    llm_values = {
        key: values[key]
        for key in USER_LLM_FIELDS
        if key in values and values[key] is not None and (key != "api_key" or values[key])
    }
    _update_section(current, "llm", llm_values)

    agent_mode = values.get("agent_mode")
    if agent_mode in {"react", "plan"}:
        _update_section(current, "prompt", {"agent_mode": agent_mode})

    approval_mode = values.get("approval_mode")
    if approval_mode in {"ask", "auto"}:
        _update_section(current, "policy", {"approval_mode": approval_mode})

    write_private_text(path, json.dumps(current, ensure_ascii=False, indent=2) + "\n")


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


def _apply_env(
    data: dict[str, Any],
    env: dict[str, str | None],
    warnings: list[str],
) -> dict[str, Any]:
    result = deepcopy(data)
    llm = result.setdefault("llm", {})
    policy = result.setdefault("policy", {})
    _apply_typed_env(llm, env, LLM_ENV_FIELDS, warnings)
    _apply_approval_env(policy, env, warnings)
    return result


def _apply_typed_env(
    target: dict[str, Any],
    env: dict[str, str | None],
    fields_to_apply: tuple[tuple[str, str, Any], ...],
    warnings: list[str],
) -> None:
    for env_key, config_key, caster in fields_to_apply:
        raw = env.get(env_key)
        if raw in (None, ""):
            continue
        try:
            target[config_key] = caster(raw)
        except (TypeError, ValueError):
            warnings.append(f"Ignored {env_key}={raw!r}: expected {caster.__name__}")


def _apply_provider_env(
    llm: dict[str, Any],
    env: dict[str, str | None],
    *,
    protected: set[str],
) -> None:
    provider = str(llm.get("provider") or "").lower()
    if not llm.get("api_key"):
        for provider_key in provider_api_key_envs(provider):
            if env.get(provider_key):
                llm["api_key"] = env[provider_key]
                break

    provider_model_key = f"{provider.upper()}_MODEL" if provider else ""
    provider_base_url_key = f"{provider.upper()}_BASE_URL" if provider else ""
    if "model" not in protected and provider_model_key and env.get(provider_model_key):
        llm["model"] = env[provider_model_key]
    if "base_url" not in protected and provider_base_url_key and env.get(provider_base_url_key):
        llm["base_url"] = env[provider_base_url_key]


def _apply_approval_env(
    policy: dict[str, Any],
    env: dict[str, str | None],
    warnings: list[str],
) -> None:
    approval_mode = env.get("VELA_APPROVAL_MODE")
    if approval_mode in {"ask", "auto"}:
        policy["approval_mode"] = approval_mode
    elif approval_mode not in (None, ""):
        warnings.append(f"Ignored VELA_APPROVAL_MODE={approval_mode!r}: expected ask or auto")


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


def _update_section(config: dict[str, Any], name: str, values: dict[str, Any]) -> None:
    if not values:
        return
    section = config.get(name)
    if not isinstance(section, dict):
        section = {}
        config[name] = section
    section.update(values)


def _config_to_dict(config: VelaConfig) -> dict[str, Any]:
    data = asdict(config)
    data.pop("project_trusted", None)
    return data


def _dict_to_config(data: dict[str, Any], warnings: list[str]) -> VelaConfig:
    known_sections = {"llm", "tools", "memory", "policy", "prompt"}
    unknown_sections = sorted(set(data) - known_sections)
    if unknown_sections:
        warnings.append(f"Ignored unknown config sections: {', '.join(unknown_sections)}")
    return VelaConfig(
        llm=_section(LlmConfig, data, "llm", warnings),
        tools=_section(ToolsConfig, data, "tools", warnings),
        memory=_section(MemoryConfig, data, "memory", warnings),
        policy=_section(PolicyConfig, data, "policy", warnings),
        prompt=_section(PromptConfig, data, "prompt", warnings),
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
