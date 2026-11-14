"""运行 profile 解析与安全默认配置。"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence

from .settings import LlmConfig, LlmEndpoint, ProfileName, Settings, planner_flags_for_mode

PROFILE_ENV = "BINANCE_AGENT_PROFILE"
_PROFILES: set[str] = {"default", "demo", "demo-live"}
_DEMO_AGENTS: tuple[str, ...] = (
    "research_supervisor",
    "news_researcher",
    "evidence_verifier",
    "technical",
    "funding",
    "news_sentiment",
    "bull_debater",
    "bear_debater",
    "research_manager",
    "strategy_planner",
    "strategy_synth",
)


def resolve_profile(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProfileName:
    """解析 ``--profile``,命令行优先于 ``BINANCE_AGENT_PROFILE``。"""

    args = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if environ is None else environ
    raw = str(env.get(PROFILE_ENV, "default") or "default").strip().lower()
    for index, arg in enumerate(args):
        if arg.startswith("--profile="):
            raw = arg.partition("=")[2].strip().lower()
            break
        if arg == "--profile":
            if index + 1 >= len(args):
                raise ValueError("--profile requires one of: default, demo, demo-live")
            raw = args[index + 1].strip().lower()
            break
    if raw not in _PROFILES:
        raise ValueError(f"unknown profile {raw!r}; expected default, demo, or demo-live")
    return raw  # type: ignore[return-value]


def apply_profile(settings: Settings, profile: ProfileName) -> Settings:
    """就地应用 profile;default 不改动用户的正常配置。"""

    settings.profile = profile
    if profile == "default":
        return settings

    settings.runtime.preflight_refetch_enabled = True
    if profile == "demo":
        settings.network.proxy = None
        settings.runtime.strategy_planner_mode = "off"
        settings.runtime.strategy_planner_enabled = False
        settings.runtime.strategy_planner_llm_actions = False
        settings.runtime.checkpoint_backend = (
            "sqlite" if settings.collaboration.enabled else "memory"
        )
        settings.runtime.enable_outcome_tracking = True
        settings.llm = _demo_llm_config()
        return settings

    settings.runtime.strategy_planner_mode = "llm_shadow"
    enabled, llm_actions, _ = planner_flags_for_mode("llm_shadow")
    settings.runtime.strategy_planner_enabled = enabled
    settings.runtime.strategy_planner_llm_actions = llm_actions
    settings.runtime.checkpoint_backend = "sqlite"
    settings.runtime.enable_outcome_tracking = True
    return settings


def _demo_llm_config() -> LlmConfig:
    models = [f"demo-{agent}" for agent in _DEMO_AGENTS]
    endpoint = LlmEndpoint(
        name="demo",
        base_url="offline://scripted",
        api_key_ref="BINANCE_AGENT_DEMO_OFFLINE",
        model="demo-technical",
        models=models,
        temperature=0,
        timeout=5,
        max_retries=0,
        structured_output=False,
    )
    return LlmConfig(
        active_endpoint="demo",
        endpoints=[endpoint],
        agent_models={agent: ("demo", f"demo-{agent}") for agent in _DEMO_AGENTS},
    )
