from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from vela.providers import model_context_window, provider_api_key_envs, provider_spec


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
        candidates = (*provider_api_key_envs(self.provider), "VELA_API_KEY")
        for key in candidates:
            if key and env_map.get(key):
                return str(env_map[key])
        if self.provider == current_provider.lower():
            return current_api_key
        return ""


def _profile(
    name: str,
    provider: str,
    model: str,
    description: str,
) -> ModelProfile:
    spec = provider_spec(provider)
    if spec is None:
        raise ValueError(f"Missing provider metadata for {provider}")
    return ModelProfile(
        name=name,
        provider=provider,
        model=model,
        base_url=spec.base_url,
        context_window=model_context_window(model, spec.context_window),
        description=description,
    )


DEFAULT_MODEL_PROFILES: tuple[ModelProfile, ...] = (
    _profile(
        name="DeepSeek V4 Flash",
        provider="deepseek",
        model="deepseek-v4-flash",
        description="Fast, cost-efficient Agent model with thinking support",
    ),
    _profile(
        name="DeepSeek V4 Pro",
        provider="deepseek",
        model="deepseek-v4-pro",
        description="Higher-quality DeepSeek model for difficult coding tasks",
    ),
    _profile(
        name="GLM-5.2",
        provider="glm",
        model="glm-5.2",
        description="Zhipu flagship model for long-running Agent tasks",
    ),
    _profile(
        name="GLM-5.1",
        provider="glm",
        model="glm-5.1",
        description="Zhipu general-purpose coding and reasoning model",
    ),
    _profile(
        name="GLM-4.7",
        provider="glm",
        model="glm-4.7",
        description="Agentic coding model with tool calling",
    ),
)
