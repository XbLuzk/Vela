from __future__ import annotations

from vela.config import LlmConfig
from vela.llm.openai_compatible import OpenAICompatibleClient
from vela.providers import model_context_window, provider_spec


def create_llm_client(config: LlmConfig) -> OpenAICompatibleClient:
    provider = config.provider.lower()
    spec = provider_spec(provider)
    base_url = config.base_url or (spec.base_url if spec else None)
    if not base_url:
        raise ValueError(f"Provider {provider!r} requires an explicit base_url")
    default_context_window = spec.context_window if spec else 128_000
    context_window = config.context_window or model_context_window(
        config.model, default_context_window
    )
    return OpenAICompatibleClient(
        provider_name=provider,
        model=config.model,
        api_key=config.api_key,
        base_url=base_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
        max_context_window=context_window,
    )
