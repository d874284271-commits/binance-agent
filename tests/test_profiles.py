from __future__ import annotations

import pytest

from binance_agent.infra.paths import resolve_data_path
from binance_agent.infra.profiles import apply_profile, resolve_profile
from binance_agent.infra.settings import Settings


def test_cli_profile_overrides_environment() -> None:
    assert (
        resolve_profile(["--headless", "--profile", "demo"], {"BINANCE_AGENT_PROFILE": "demo-live"})
        == "demo"
    )
    assert resolve_profile(["--profile=demo-live"], {}) == "demo-live"


def test_invalid_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        resolve_profile([], {"BINANCE_AGENT_PROFILE": "unsafe"})


def test_demo_profile_is_offline_and_uses_safe_defaults() -> None:
    settings = apply_profile(Settings(), "demo")

    assert settings.profile == "demo"
    assert settings.runtime.strategy_planner_mode == "off"
    assert settings.runtime.checkpoint_backend == "memory"
    assert settings.runtime.enable_outcome_tracking is True
    assert settings.runtime.preflight_refetch_enabled is True
    assert settings.llm.active_endpoint == "demo"
    assert settings.llm.endpoints[0].base_url == "offline://scripted"
    assert settings.llm.agent_models["strategy_synth"] == ("demo", "demo-strategy_synth")


def test_demo_live_enables_shadow_checkpoint_and_outcomes() -> None:
    settings = apply_profile(Settings(), "demo-live")

    assert settings.profile == "demo-live"
    assert settings.runtime.strategy_planner_mode == "llm_shadow"
    assert settings.runtime.strategy_planner_enabled is True
    assert settings.runtime.strategy_planner_llm_actions is True
    assert settings.runtime.checkpoint_backend == "sqlite"
    assert settings.runtime.enable_outcome_tracking is True


def test_demo_profile_uses_isolated_data_directory() -> None:
    default_path = resolve_data_path(".runs/runs.jsonl")
    demo_path = resolve_data_path(".runs/runs.jsonl", profile="demo")

    assert demo_path != default_path
    assert demo_path == default_path.parent.parent / ".demo" / ".runs" / "runs.jsonl"
