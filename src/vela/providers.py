from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    base_url: str
    context_window: int
    api_key_envs: tuple[str, ...] = ()


PROVIDERS: dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec(
        "https://api.deepseek.com/v1",
        64_000,
        ("DEEPSEEK_API_KEY",),
    ),
    "openai": ProviderSpec("https://api.openai.com/v1", 128_000, ("OPENAI_API_KEY",)),
    "openai-compatible": ProviderSpec(
        "https://api.openai.com/v1",
        128_000,
        ("OPENAI_API_KEY",),
    ),
    "glm": ProviderSpec(
        "https://open.bigmodel.cn/api/paas/v4",
        128_000,
        ("ZAI_API_KEY", "GLM_API_KEY"),
    ),
    "kimi": ProviderSpec("https://api.moonshot.cn/v1", 128_000, ("KIMI_API_KEY",)),
    "step": ProviderSpec("https://api.stepfun.com/v1", 128_000, ("STEP_API_KEY",)),
}

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
    "deepseek-coder": 128_000,
    "glm-5.2": 200_000,
    "glm-5.1": 200_000,
    "glm-4.7": 200_000,
}


def provider_spec(name: str) -> ProviderSpec | None:
    return PROVIDERS.get(name.lower())


def provider_api_key_envs(name: str) -> tuple[str, ...]:
    spec = provider_spec(name)
    return spec.api_key_envs if spec else ()


def model_context_window(model: str, fallback: int) -> int:
    return MODEL_CONTEXT_WINDOWS.get(model.lower(), fallback)
