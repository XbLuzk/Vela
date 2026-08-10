from __future__ import annotations

import stat
from types import SimpleNamespace

import pytest

from vela.config import load_config
from vela.entrypoints.model_selector import ModelSelectorState
from vela.entrypoints.repl import _activate_model
from vela.llm import create_llm_client
from vela.llm.model_profiles import (
    DEFAULT_MODEL_PROFILES,
    CustomModelStore,
    ModelProfile,
)


def test_selector_tabs_between_default_and_custom_models():
    custom = ModelProfile.custom_profile(
        name="My GLM",
        provider="glm",
        model="glm-custom",
        base_url="https://example.com/v1",
        context_window=128_000,
    )
    state = ModelSelectorState(
        defaults=list(DEFAULT_MODEL_PROFILES),
        custom=[custom],
        current_provider="deepseek",
        current_model="deepseek-v4-flash",
    )

    assert state.tab == "default"
    assert state.selected_action().profile.model == "deepseek-v4-flash"

    state.switch_tab()
    assert state.tab == "custom"
    assert state.selected_action().profile == custom

    state.move(1)
    assert state.selected_action().kind == "add"
    state.move(1)
    assert state.selected_action().profile == custom
    assert state.delete_action().profile == custom


def test_selector_starts_on_active_custom_model():
    custom = ModelProfile.custom_profile(
        name="Private gateway",
        provider="openai-compatible",
        model="company-model",
        base_url="https://llm.example.com/v1",
        context_window=64_000,
    )
    state = ModelSelectorState(
        defaults=list(DEFAULT_MODEL_PROFILES),
        custom=[custom],
        current_provider=custom.provider,
        current_model=custom.model,
    )

    assert state.tab == "custom"
    plain = "".join(text for _style, text in state.render())
    assert "Custom (1)" in plain
    assert "Private gateway ✓" in plain
    assert "[+] Add custom model" in plain


def test_custom_model_store_round_trips_and_uses_private_permissions(tmp_path):
    path = tmp_path / "models.json"
    store = CustomModelStore(path)
    profile = ModelProfile.custom_profile(
        name="DeepSeek proxy",
        provider="deepseek",
        model="deepseek-v4-flash",
        base_url="https://proxy.example.com/v1",
        context_window=1_000_000,
        api_key="secret",
        api_key_env="DEEPSEEK_API_KEY",
    )

    store.add(profile)

    assert store.list() == [profile]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert store.delete(profile.id)
    assert store.list() == []
    assert not store.delete(profile.id)


def test_custom_profile_validates_endpoint_and_context_window():
    with pytest.raises(ValueError, match="http"):
        ModelProfile.custom_profile(
            name="bad",
            provider="glm",
            model="glm-5.2",
            base_url="open.bigmodel.cn",
            context_window=200_000,
        )
    with pytest.raises(ValueError, match="greater than zero"):
        ModelProfile.custom_profile(
            name="bad",
            provider="glm",
            model="glm-5.2",
            base_url="https://open.bigmodel.cn",
            context_window=0,
        )


def test_activate_model_rebuilds_live_client_without_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ZAI_API_KEY", "glm-secret")
    config = load_config(project_root=tmp_path)
    old_client = create_llm_client(config.llm)
    agent = SimpleNamespace(llm_client=old_client, system_prompt="old")
    registry = SimpleNamespace(list_names=lambda: ["read_file"])
    renderer = SimpleNamespace(context_window=None)
    renderer.set_context_window = lambda value: setattr(renderer, "context_window", value)
    profile = next(item for item in DEFAULT_MODEL_PROFILES if item.model == "glm-5.2")

    _activate_model(profile, config, agent, registry, renderer, str(tmp_path))

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
        project_root=tmp_path,
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
        ("custom", "private-model", "https://api.deepseek.com/v1", 64_000),
    ],
)
def test_llm_factory_preserves_provider_defaults(
    tmp_path,
    provider,
    model,
    expected_base_url,
    expected_context_window,
):
    config = load_config(project_root=tmp_path, env={})
    config.llm.provider = provider
    config.llm.model = model
    config.llm.base_url = None
    config.llm.context_window = None

    client = create_llm_client(config.llm)

    assert client.provider_name == provider
    assert client.base_url == expected_base_url
    assert client.max_context_window == expected_context_window
