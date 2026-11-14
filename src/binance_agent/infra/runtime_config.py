"""运行时配置 overrides 的 JSON 持久化(非敏感部分)。

UI 运行时增改的供应商(name/base_url/model/models/api_key_ref)、active 端点、per-agent 映射、
数据源开关、分析参数存于 ~/.binance_agent/runtime_config.json。**API key 不在此**
(经 infra.secrets 加密落盘,这里只存其引用名 api_key_ref)。
启动时以 config.toml 为基,叠加本文件覆盖,实现运行时配置的重启保留。
"""

from __future__ import annotations

import json
import os
import warnings
from math import isfinite
from pathlib import Path
from typing import Any, cast

from ..agents.context import ContextPolicy
from .file_io import atomic_write, file_lock
from .settings import (
    ConfidenceConfig,
    DesktopConfig,
    HotkeyConfig,
    LlmEndpoint,
    NetworkConfig,
    Settings,
    StrategyPlannerMode,
    is_strategy_planner_mode,
    planner_flags_for_mode,
    planner_mode_from_flags,
)


class RuntimeConfigError(ValueError):
    """已存配置不可安全读取或校验,调用者必须保留原文件并报告错误。"""


_ENV_PATH = "BINANCE_AGENT_RUNTIME_CONFIG"


def _default_path() -> Path:
    env = os.environ.get(_ENV_PATH)
    return Path(env) if env else Path.home() / ".binance_agent" / "runtime_config.json"


def load_overrides(path: Path | None = None) -> dict[str, Any]:
    """读取非敏感覆盖;缺失文件返回空字典,损坏文件抛错,不静默重置配置。"""
    target = path or _default_path()
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("root must be a JSON object")
        return data
    except (ValueError, OSError) as exc:
        raise RuntimeConfigError(
            f"运行时配置读取失败: {target}; 保留原文件后修复其 JSON/字段"
        ) from exc


def save_overrides(data: dict[str, Any], path: Path | None = None) -> None:
    """拒绝非有限 JSON 数值,持文件锁原子保存;失败时不激活任何运行时配置。

    data 只能包含配置和密钥引用,不能包含明文凭证。
    """
    target = path or _default_path()
    encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    with file_lock(target):
        atomic_write(target, encoded)


def overrides_present(path: Path | None = None) -> bool:
    """是否存在(非空的)运行时覆盖。供 UI 显示「生效来源:运行时覆盖 / 默认」。"""
    target = path or _default_path()
    if not target.exists():
        return False
    return bool(load_overrides(target))


def clear_overrides(path: Path | None = None) -> None:
    """删除运行时覆盖文件(重置为 config.toml 基底);文件不存在则忽略。"""
    target = path or _default_path()
    target.unlink(missing_ok=True)


def build_candidate(settings: Settings, overrides: dict[str, Any]) -> Settings:
    """深拷贝后应用覆盖并完整校验,返回独立候选;不修改原对象、不写磁盘。"""
    if not isinstance(overrides, dict):
        raise RuntimeConfigError("运行时配置根必须是 object")
    for section in (
        "llm",
        "intel_sources",
        "intel_keys",
        "confidence",
        "network",
        "params",
        "desktop",
        "hotkey",
        "research",
    ):
        if section in overrides and not isinstance(overrides[section], dict):
            raise RuntimeConfigError(f"运行时配置 {section} 必须是 object")
    candidate = settings.model_copy(deep=True)
    _apply_unchecked(candidate, overrides)
    return Settings.model_validate(candidate.model_dump())


def activate_candidate(settings: Settings, candidate: Settings) -> None:
    """保留 Settings 对象身份并激活已校验候选;持久化及并发互斥由调用者先完成。"""
    settings.__dict__.update(candidate.model_copy(deep=True).__dict__)
    settings.__pydantic_fields_set__.update(candidate.__pydantic_fields_set__)


def apply_overrides(settings: Settings, overrides: dict[str, Any]) -> None:
    """全部校验成功后才修改传入配置;用于内存装配,不负责持久化。"""
    activate_candidate(settings, build_candidate(settings, overrides))


def _apply_unchecked(settings: Settings, overrides: dict[str, Any]) -> None:
    if "desktop" in overrides:
        settings.desktop = DesktopConfig.model_validate(overrides["desktop"])
    if "hotkey" in overrides:
        settings.hotkey = HotkeyConfig.model_validate(overrides["hotkey"])
    if "research" in overrides:
        settings.research = type(settings.research).model_validate(
            {**settings.research.model_dump(), **overrides["research"]}
        )
    llm = overrides.get("llm") or {}
    raw_endpoints = llm.get("endpoints")
    if "endpoints" in llm and not isinstance(raw_endpoints, list):
        raise RuntimeConfigError("运行时配置 llm.endpoints 必须是 array")
    if "agent_models" in llm and not isinstance(llm["agent_models"], dict):
        raise RuntimeConfigError("运行时配置 llm.agent_models 必须是 object")
    if isinstance(raw_endpoints, list):
        # runtime_config 由 _current_overrides() 写入完整 endpoint 列表。
        # 因此这里必须替换基础 TOML 列表,否则用户删除 config.toml 内置端点后重启会复活。
        by_name: dict[str, LlmEndpoint] = {}
        for raw in raw_endpoints:
            ep = LlmEndpoint.model_validate(raw)
            by_name[ep.name] = ep
        settings.llm.endpoints = list(by_name.values())
    if "active" in llm:
        if not isinstance(llm["active"], str) or not llm["active"]:
            raise RuntimeConfigError("运行时配置 llm.active 必须是非空字符串")
        settings.llm.active_endpoint = llm["active"]
    endpoint_names = {e.name for e in settings.llm.endpoints}
    if endpoint_names and settings.llm.active_endpoint not in endpoint_names:
        if "active" in llm:
            raise RuntimeConfigError("运行时配置 llm.active 未引用有效 endpoint")
        settings.llm.active_endpoint = settings.llm.endpoints[0].name
    for agent, pair in (llm.get("agent_models") or {}).items():
        if not (
            isinstance(pair, (list, tuple))
            and len(pair) == 2
            and all(isinstance(item, str) for item in pair)
        ):
            raise RuntimeConfigError(f"运行时配置 llm.agent_models.{agent} 必须是两个字符串")
        settings.llm.agent_models[agent] = (pair[0], pair[1])

    for source, enabled in (overrides.get("intel_sources") or {}).items():
        if not isinstance(enabled, bool):
            raise RuntimeConfigError(f"运行时配置 intel_sources.{source} 必须是 boolean")
        settings.intel.sources[source] = enabled

    for source, ref in (overrides.get("intel_keys") or {}).items():
        if not isinstance(ref, str) or not ref.strip():
            raise RuntimeConfigError(f"运行时配置 intel_keys.{source} 必须是非空字符串引用")
        settings.intel.keys[source] = ref

    confidence = overrides.get("confidence")
    if isinstance(confidence, dict):
        settings.confidence = ConfidenceConfig.model_validate(confidence)

    network = overrides.get("network")
    if isinstance(network, dict) and "proxy" in network:
        raw_proxy = network.get("proxy")
        if raw_proxy is not None and not isinstance(raw_proxy, str):
            raise RuntimeConfigError("运行时配置 network.proxy 必须是字符串或 null")
        proxy = str(raw_proxy).strip() or None if raw_proxy else None
        settings.network.proxy = NetworkConfig(proxy=proxy).proxy

    params = overrides.get("params") or {}
    for name in (
        "time_window_hours",
        "per_source_timeout",
        "max_news_items",
        "max_debate_rounds",
        "strategy_planner_max_steps",
    ):
        if name not in params:
            continue
        value = params[name]
        try:
            finite = (
                not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value)
            )
        except OverflowError:
            finite = False
        allow_zero = name in {"max_news_items", "max_debate_rounds"}
        if (
            not finite
            or value < 0
            or (value == 0 and not allow_zero)
            or (name != "per_source_timeout" and int(value) != value)
        ):
            bound = "非负数" if allow_zero else "正数"
            raise RuntimeConfigError(f"运行时配置 params.{name} 必须是有限{bound}(计数必须为整数)")
    for name in (
        "strategy_planner_enabled",
        "preflight_refetch_enabled",
        "strategy_planner_refetch_enabled",
        "strategy_planner_llm_actions",
    ):
        if name in params and not isinstance(params[name], bool):
            raise RuntimeConfigError(f"运行时配置 params.{name} 必须是 boolean")
    if "context" in params and not isinstance(params["context"], dict):
        raise RuntimeConfigError("运行时配置 params.context 必须是 object")
    if "strategy_planner_mode" in params and not is_strategy_planner_mode(
        params["strategy_planner_mode"]
    ):
        raise RuntimeConfigError("运行时配置 params.strategy_planner_mode 无效")
    if "context" in params:
        settings.runtime.context = ContextPolicy.model_validate(
            {**settings.runtime.context.model_dump(), **params["context"]}
        )
    if "time_window_hours" in params:
        settings.intel.time_window_hours = int(params["time_window_hours"])
    if "per_source_timeout" in params:
        settings.intel.per_source_timeout = float(params["per_source_timeout"])
    if "max_news_items" in params:
        settings.intel.max_news_items = int(params["max_news_items"])
    if "max_debate_rounds" in params:
        settings.debate.max_rounds = int(params["max_debate_rounds"])
    legacy_planner_changed = False
    mode = params.get("strategy_planner_mode")
    if is_strategy_planner_mode(mode):
        mode = cast(StrategyPlannerMode, mode)
        settings.runtime.strategy_planner_mode = mode
        enabled, llm_actions, _ = planner_flags_for_mode(mode)
        settings.runtime.strategy_planner_enabled = enabled
        settings.runtime.strategy_planner_llm_actions = llm_actions
    if "strategy_planner_enabled" in params:
        settings.runtime.strategy_planner_enabled = bool(params["strategy_planner_enabled"])
        legacy_planner_changed = True
    if "strategy_planner_max_steps" in params:
        settings.runtime.strategy_planner_max_steps = int(params["strategy_planner_max_steps"])
    if "preflight_refetch_enabled" in params:
        settings.runtime.preflight_refetch_enabled = bool(params["preflight_refetch_enabled"])
    if "strategy_planner_refetch_enabled" in params:
        warnings.warn(
            "strategy_planner_refetch_enabled is deprecated; use preflight_refetch_enabled",
            DeprecationWarning,
            stacklevel=2,
        )
        legacy_refetch = bool(params["strategy_planner_refetch_enabled"])
        settings.runtime.preflight_refetch_enabled = legacy_refetch
        settings.runtime.strategy_planner_refetch_enabled = legacy_refetch
    if "strategy_planner_llm_actions" in params:
        settings.runtime.strategy_planner_llm_actions = bool(params["strategy_planner_llm_actions"])
        legacy_planner_changed = True
    if not is_strategy_planner_mode(mode) and legacy_planner_changed:
        settings.runtime.strategy_planner_mode = planner_mode_from_flags(
            settings.runtime.strategy_planner_enabled,
            settings.runtime.strategy_planner_llm_actions,
        )
