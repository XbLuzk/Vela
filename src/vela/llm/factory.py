from __future__ import annotations

from vela.config import LlmConfig
from vela.llm.openai_compatible import OpenAICompatibleClient

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
PROVIDER_BASE_URLS = {
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "kimi": "https://api.moonshot.cn/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "step": "https://api.stepfun.com/v1",
}

MODEL_CONTEXT_WINDOWS = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
    "deepseek-coder": 128_000,
    "glm-5.2": 200_000,
    "glm-5.1": 200_000,
    "glm-4.7": 200_000,
}


def create_llm_client(config: LlmConfig) -> OpenAICompatibleClient:
    provider = config.provider.lower()
    if provider == "deepseek":
        default_base_url = DEEPSEEK_BASE_URL
        default_context_window = 64_000
    elif provider in {"openai", "openai-compatible", "compatible"}:
        default_base_url = OPENAI_BASE_URL
        default_context_window = 128_000
    elif provider in PROVIDER_BASE_URLS:
        default_base_url = PROVIDER_BASE_URLS[provider]
        default_context_window = 128_000
    else:
        default_base_url = DEEPSEEK_BASE_URL
        default_context_window = 64_000

    context_window = config.context_window or MODEL_CONTEXT_WINDOWS.get(
        config.model.lower(),
        default_context_window,
    )
    return OpenAICompatibleClient(
        provider_name=provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url or default_base_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
        max_context_window=context_window,
    )
