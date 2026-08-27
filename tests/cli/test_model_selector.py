from __future__ import annotations

from types import SimpleNamespace

import pytest

from vela.config import load_config
from vela.entrypoints.model_command import _profile_from_argument, activate_model
from vela.entrypoints.model_selector import ModelSelectorState
from vela.llm import create_llm_client
from vela.llm.model_profiles import DEFAULT_MODEL_PROFILES


def test_selector_navigates_builtin_models():
    state = ModelSelectorState(
        profiles=list(DEFAULT_MODEL_PROFILES),
        current_provider="deepseek",
        current_model="deepseek-v4-flash",
    )

    assert state.selected_profile().model == "deepseek-v4-flash"
    state.move(1)
    assert state.selected_profile().model == "deepseek-v4-pro"
    state.move(-1)
    assert state.selected_profile().model == "deepseek-v4-flash"


def test_selector_starts_on_active_model():
    state = ModelSelectorState(
        profiles=list(DEFAULT_MODEL_PROFILES),
        current_provider="glm",
        current_model="glm-5.2",
    )

    assert state.selected_profile().model == "glm-5.2"
    plain = "".join(text for _style, text in state.render())
    assert "Models (5)" in plain
    assert "GLM-5.2 ✓" in plain


def test_direct_model_argument_keeps_runtime_provider_switching(tmp_path):
    config = load_config(env={})

    profile = _profile_from_argument("GLM glm-custom", config)

    assert profile.provider == "glm"
    assert profile.model == "glm-custom"
    assert profile.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert (
        profile.resolve_api_key(env={"VELA_API_KEY": "generic-key", "ZAI_API_KEY": "glm-key"})
        == "glm-key"
    )


def test_direct_model_argument_keeps_custom_provider_base_url():
    config = load_config(
        overrides={
            "llm": {
                "provider": "custom",
                "model": "private-model",
                "base_url": "https://models.example.com/v1",
            }
        },
        env={},
    )

    profile = _profile_from_argument("custom private-model-v2", config)

    assert profile.provider == "custom"
    assert profile.base_url == "https://models.example.com/v1"


def test_activate_model_rebuilds_live_client_without_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ZAI_API_KEY", "glm-secret")
    config = load_config()
    old_client = create_llm_client(config.llm)
    registry = SimpleNamespace(list_names=lambda: ["read_file"])
    agent = SimpleNamespace(
        llm_client=old_client,
        system_prompt="old",
        config=config,
        cwd=str(tmp_path),
        tool_registry=registry,
    )
    renderer = SimpleNamespace(context_window=None)
    renderer.set_context_window = lambda value: setattr(renderer, "context_window", value)
    profile = next(item for item in DEFAULT_MODEL_PROFILES if item.model == "glm-5.2")

    activate_model(profile, agent, renderer)

    assert agent.llm_client is not old_client
    assert agent.llm_client.provider_name == "glm"
    assert agent.llm_client.model_name == "glm-5.2"
    assert agent.llm_client.api_key == "glm-secret"
    assert config.llm.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert renderer.context_window == 200_000
    assert "You are Vela" in agent.system_prompt


def test_glm_startup_uses_official_zai_api_key_and_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(
        env={
            "VELA_PROVIDER": "glm",
            "VELA_MODEL": "glm-5.2",
            "ZAI_API_KEY": "official-key",
        },
    )

    client = create_llm_client(config.llm)

    assert config.llm.api_key == "official-key"
    assert client.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert client.max_context_window == 200_000


@pytest.mark.parametrize(
    ("provider", "model", "expected_base_url", "expected_context_window"),
    [
        ("deepseek", "deepseek-chat", "https://api.deepseek.com/v1", 1_000_000),
        ("openai", "gpt-4o", "https://api.openai.com/v1", 128_000),
        ("kimi", "moonshot-v1", "https://api.moonshot.cn/v1", 128_000),
    ],
)
def test_llm_factory_preserves_provider_defaults(
    tmp_path,
    provider,
    model,
    expected_base_url,
    expected_context_window,
):
    config = load_config(env={})
    config.llm.provider = provider
    config.llm.model = model
    config.llm.base_url = None
    config.llm.context_window = None

    client = create_llm_client(config.llm)

    assert client.provider_name == provider
    assert client.base_url == expected_base_url
    assert client.max_context_window == expected_context_window


def test_custom_provider_requires_explicit_base_url():
    config = load_config(env={})
    config.llm.provider = "custom"
    config.llm.model = "private-model"

    with pytest.raises(ValueError, match="requires an explicit base_url"):
        create_llm_client(config.llm)
