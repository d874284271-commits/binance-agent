"""runtime_config 离线测试:save/load 往返 + apply 合并 + 不落 key 明文。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from binance_agent.app.server import _current_overrides
from binance_agent.infra.runtime_config import (
    apply_overrides,
    clear_overrides,
    load_overrides,
    overrides_present,
    save_overrides,
)
from binance_agent.infra.settings import LlmEndpoint, Settings, effective_strategy_planner_mode
from binance_agent.llm.registry import LlmRegistry


def test_save_load_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "rc.json"
    data = {
        "llm": {"active": "x", "agent_models": {"technical": ["x", "m"]}},
        "intel_sources": {"reddit": False},
        "params": {"time_window_hours": 12},
    }
    save_overrides(data, target)
    assert load_overrides(target) == data


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    assert load_overrides(tmp_path / "nope.json") == {}


def test_overrides_present_and_clear(tmp_path: Path) -> None:
    target = tmp_path / "rc.json"
    assert overrides_present(target) is False  # 文件不存在
    save_overrides({"params": {"time_window_hours": 12}}, target)
    assert overrides_present(target) is True
    clear_overrides(target)
    assert not target.exists()
    assert overrides_present(target) is False


def test_overrides_present_false_for_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "rc.json"
    save_overrides({}, target)  # 空 overrides
    assert target.exists()
    assert overrides_present(target) is False


def test_clear_overrides_missing_is_noop(tmp_path: Path) -> None:
    clear_overrides(tmp_path / "nope.json")  # 不存在不报错


def test_apply_overrides_merges() -> None:
    settings = Settings()
    overrides = {
        "llm": {
            "active": "blaze",
            "endpoints": [
                {
                    "name": "blaze",
                    "base_url": "https://x/api",
                    "api_key_ref": "llm_blaze_key",
                    "model": "m",
                    "models": ["m", "m2"],
                }
            ],
            "agent_models": {"technical": ["blaze", "m"]},
        },
        "intel_sources": {"reddit": False},
        "params": {"time_window_hours": 12, "max_debate_rounds": 5},
    }
    apply_overrides(settings, overrides)
    assert settings.llm.active_endpoint == "blaze"
    assert [e.name for e in settings.llm.endpoints] == ["blaze"]
    assert settings.llm.endpoints[0].models == ["m", "m2"]
    assert settings.llm.agent_models["technical"] == ("blaze", "m")
    assert settings.intel.sources["reddit"] is False
    assert settings.intel.time_window_hours == 12
    assert settings.debate.max_rounds == 5


def test_apply_overrides_restores_strategy_planner_runtime_flags() -> None:
    settings = Settings()
    overrides = {
        "params": {
            "strategy_planner_enabled": True,
            "strategy_planner_max_steps": 5,
            "strategy_planner_llm_actions": True,
        }
    }

    apply_overrides(settings, overrides)

    assert settings.runtime.strategy_planner_enabled is True
    assert settings.runtime.strategy_planner_max_steps == 5
    assert settings.runtime.strategy_planner_llm_actions is True


def test_apply_overrides_restores_strategy_planner_mode() -> None:
    settings = Settings()
    overrides = {"params": {"strategy_planner_mode": "llm_shadow", "strategy_planner_max_steps": 4}}

    apply_overrides(settings, overrides)

    assert settings.runtime.strategy_planner_mode == "llm_shadow"
    assert settings.runtime.strategy_planner_enabled is True
    assert settings.runtime.strategy_planner_llm_actions is True
    assert settings.runtime.strategy_planner_max_steps == 4
    assert effective_strategy_planner_mode(settings.runtime) == "llm_shadow"


def test_apply_overrides_uses_new_preflight_refetch_flag() -> None:
    settings = Settings()

    apply_overrides(settings, {"params": {"preflight_refetch_enabled": False}})

    assert settings.runtime.preflight_refetch_enabled is False


def test_apply_overrides_migrates_legacy_planner_refetch_flag() -> None:
    settings = Settings()

    with pytest.warns(DeprecationWarning, match="preflight_refetch_enabled"):
        apply_overrides(settings, {"params": {"strategy_planner_refetch_enabled": False}})

    assert settings.runtime.preflight_refetch_enabled is False


def test_apply_overrides_restores_confidence_policy() -> None:
    settings = Settings()

    apply_overrides(
        settings,
        {
            "confidence": {
                "policy_version": "confidence-policy/v1",
                "bucket_caps": {"[0.6,0.8)": 0.65},
            }
        },
    )

    assert settings.confidence.bucket_caps == {"[0.6,0.8)": 0.65}


def test_apply_overrides_proxy_error_does_not_expose_credentials() -> None:
    settings = Settings()
    proxy = "http://alice:secret@127.0.0.1:7897"

    with pytest.raises(ValidationError) as captured:
        apply_overrides(settings, {"network": {"proxy": proxy}})

    error = str(captured.value)
    assert "credentials" in error
    assert proxy not in error
    assert "alice" not in error
    assert "secret" not in error


def test_current_overrides_persists_strategy_planner_runtime_flags() -> None:
    settings = Settings()
    settings.runtime.strategy_planner_mode = "llm_enforce"
    settings.runtime.strategy_planner_enabled = True
    settings.runtime.strategy_planner_max_steps = 4
    settings.runtime.strategy_planner_llm_actions = True
    registry = LlmRegistry(settings.llm)

    overrides = _current_overrides(settings, registry)

    assert overrides["params"]["strategy_planner_enabled"] is True
    assert overrides["params"]["strategy_planner_max_steps"] == 4
    assert overrides["params"]["strategy_planner_llm_actions"] is True
    assert overrides["params"]["strategy_planner_mode"] == "llm_enforce"


def test_apply_overrides_endpoint_upsert_keeps_others() -> None:
    settings = Settings()
    settings.llm.endpoints = [
        LlmEndpoint(name="a", base_url="u", api_key_ref="r", model="m"),
    ]

    apply_overrides(
        settings,
        {
            "llm": {
                "endpoints": [
                    {"name": "a", "base_url": "u", "api_key_ref": "r", "model": "m"},
                    {"name": "b", "base_url": "u2", "api_key_ref": "r2", "model": "m2"},
                ]
            }
        },
    )

    names = {e.name for e in settings.llm.endpoints}
    assert names == {"a", "b"}


def test_apply_overrides_endpoint_list_replaces_base_config_for_deletes() -> None:
    settings = Settings()
    settings.llm.active_endpoint = "blaze"
    settings.llm.endpoints = [
        LlmEndpoint(name="default", base_url="https://openai/v1", api_key_ref="OPENAI", model="m"),
        LlmEndpoint(name="blaze", base_url="https://blaze/api", api_key_ref="BLAZE", model="b"),
    ]

    apply_overrides(
        settings,
        {
            "llm": {
                "active": "custom",
                "endpoints": [
                    {
                        "name": "custom",
                        "base_url": "https://custom/v1",
                        "api_key_ref": "CUSTOM",
                        "model": "c",
                    }
                ],
            }
        },
    )

    assert [e.name for e in settings.llm.endpoints] == ["custom"]
    assert settings.llm.active_endpoint == "custom"
