from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    provider: str
    model: str
    base_url: str
    context_window: int
    description: str = ""

    def resolve_api_key(
        self,
        *,
        current_provider: str = "",
        current_api_key: str = "",
        env: Mapping[str, str] | None = None,
    ) -> str:
        env_map = env if env is not None else os.environ
        candidates = (*PROVIDER_API_KEY_ENVS.get(self.provider, ()), "VELA_API_KEY")
        for key in candidates:
            if key and env_map.get(key):
                return str(env_map[key])
        if self.provider == current_provider.lower():
            return current_api_key
        return ""


DEFAULT_MODEL_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        name="DeepSeek V4 Flash",
        provider="deepseek",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        context_window=1_000_000,
        description="Fast, cost-efficient Agent model with thinking support",
    ),
    ModelProfile(
        name="DeepSeek V4 Pro",
        provider="deepseek",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        context_window=1_000_000,
        description="Higher-quality DeepSeek model for difficult coding tasks",
    ),
    ModelProfile(
        name="GLM-5.2",
        provider="glm",
        model="glm-5.2",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        context_window=200_000,
        description="Zhipu flagship model for long-running Agent tasks",
    ),
    ModelProfile(
        name="GLM-5.1",
        provider="glm",
        model="glm-5.1",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        context_window=200_000,
        description="Zhipu general-purpose coding and reasoning model",
    ),
    ModelProfile(
        name="GLM-4.7",
        provider="glm",
        model="glm-4.7",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        context_window=200_000,
        description="Agentic coding model with tool calling",
    ),
)


PROVIDER_API_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "deepseek": ("DEEPSEEK_API_KEY",),
    "glm": ("ZAI_API_KEY", "GLM_API_KEY"),
    "zhipu": ("ZAI_API_KEY", "GLM_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "openai-compatible": ("OPENAI_API_KEY",),
}
