from __future__ import annotations

from rich.console import Console

from vela.agent import Agent
from vela.config import VelaConfig
from vela.entrypoints.model_selector import ModelSelectorState, run_model_selector
from vela.llm import create_llm_client
from vela.llm.model_profiles import DEFAULT_MODEL_PROFILES, ModelProfile
from vela.prompt import PromptAssembler
from vela.providers import provider_spec
from vela.render import RichRenderer


async def handle_model_command(
    arg: str,
    console: Console,
    agent: Agent,
    renderer: RichRenderer,
) -> None:
    config = agent.config
    profile = (
        _profile_from_argument(arg, config)
        if arg
        else await run_model_selector(
            ModelSelectorState(
                profiles=list(DEFAULT_MODEL_PROFILES),
                current_provider=agent.llm_client.provider_name,
                current_model=agent.llm_client.model_name,
            )
        )
    )
    if profile is None:
        return

    activate_model(profile, agent, renderer)
    console.print(
        f"[green]Switched model:[/green] {profile.name} "
        f"[dim]({profile.provider}/{profile.model})[/dim]"
    )


def activate_model(
    profile: ModelProfile,
    agent: Agent,
    renderer: RichRenderer,
) -> None:
    config = agent.config
    old_provider = config.llm.provider.lower()
    old_api_key = config.llm.api_key
    config.llm.provider = profile.provider
    config.llm.model = profile.model
    config.llm.base_url = profile.base_url
    config.llm.context_window = profile.context_window
    config.llm.api_key = profile.resolve_api_key(
        current_provider=old_provider,
        current_api_key=old_api_key,
    )

    client = create_llm_client(config.llm)
    agent.llm_client = client
    agent.system_prompt = PromptAssembler(
        config=config,
        cwd=agent.cwd,
        tool_names=agent.tool_registry.list_names(),
        model=client.model_name,
        provider=client.provider_name,
    ).build_static()
    renderer.set_context_window(client.max_context_window)


def _profile_from_argument(arg: str, config: VelaConfig) -> ModelProfile:
    parts = arg.split(maxsplit=1)
    provider = (config.llm.provider if len(parts) == 1 else parts[0]).lower()
    model = parts[-1]
    known = next(
        (
            profile
            for profile in DEFAULT_MODEL_PROFILES
            if profile.provider == provider.lower() and profile.model == model
        ),
        None,
    )
    if known is not None:
        return known

    base_url = (
        config.llm.base_url
        if provider == config.llm.provider.lower() and config.llm.base_url
        else _default_base_url(provider)
    )
    return ModelProfile(
        name=model,
        provider=provider,
        model=model,
        base_url=base_url,
        context_window=config.llm.context_window or 128_000,
        description="Selected from /model arguments",
    )


def _default_base_url(provider: str) -> str:
    spec = provider_spec(provider)
    if spec is None:
        raise ValueError(f"Unknown provider {provider!r}; configure its base_url first")
    return spec.base_url
