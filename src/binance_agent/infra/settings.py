"""应用配置:TOML(config/config.toml) + .env 环境变量。

密钥不入配置文件,仅以 ``*_ref`` 引用变量名,运行时经 infra.secrets 解析。
配置结构用 pydantic 校验,缺失文件时回退默认值以保证服务可启动。
"""

from __future__ import annotations

import os
import sys
import warnings
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..agents.context import ContextPolicy

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib

_ENV_CONFIG = "BINANCE_AGENT_CONFIG"
StrategyPlannerMode = Literal["off", "deterministic", "llm_shadow", "llm_enforce"]
ProfileName = Literal["default", "demo", "demo-live"]
OutcomeKlineInterval = Literal[
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
]
_STRATEGY_PLANNER_MODES = {"off", "deterministic", "llm_shadow", "llm_enforce"}
_CONFIDENCE_BUCKET_LABELS = {"[0.0,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"}


class AppConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"
    max_concurrent_runs: int = Field(default=2, ge=1, le=8)

    @field_validator("host")
    @classmethod
    def _loopback_only(cls, value: str) -> str:
        host = value.strip()
        if host.lower() == "localhost":
            return host
        try:
            parsed = ip_address(host.strip("[]"))
        except ValueError as exc:
            raise ValueError("app.host must be a loopback address") from exc
        if not parsed.is_loopback:
            raise ValueError("app.host must be a loopback address")
        return host


class MarketConfig(BaseModel):
    default_market: Literal["fapi", "api"] = "fapi"
    testnet: bool = False


class HotkeyConfig(BaseModel):
    enabled: bool = False
    combo: str = "<ctrl>+<alt>+b"


class DesktopConfig(BaseModel):
    on_top: bool = False


class DebateConfig(BaseModel):
    max_rounds: int = 2


class LlmEndpoint(BaseModel):
    name: str
    base_url: str
    api_key_ref: str
    model: str
    models: list[str] = Field(default_factory=list)  # 可选模型列表(UI 下拉;空则仅 model)
    temperature: float = 0.3
    timeout: int = 60
    max_retries: int = 2
    # 仅当端点确认支持 OpenAI response_format=json_object 时开启;默认关,
    # 避免不支持的第三方兼容端点收到该参数返回 400 触发降级。
    structured_output: bool = False
    fallback_verified: bool = False
    max_concurrency: int = Field(default=2, ge=1, le=16)


class LlmConfig(BaseModel):
    active_endpoint: str = "default"
    endpoints: list[LlmEndpoint] = Field(default_factory=list)
    # agent 名 -> (endpoint 名, model);TOML 中以二元数组表示
    agent_models: dict[str, tuple[str, str]] = Field(default_factory=dict)


class RssChannelConfig(BaseModel):
    """Named RSS group.  ``feeds`` remains the legacy aggregate input."""

    id: str = Field(min_length=1, max_length=80)
    label: str = ""
    feeds: list[str] = Field(default_factory=list)
    source_kind: Literal["official", "media", "community", "discovery", "unknown"] = "media"
    per_feed: int = Field(default=15, ge=1, le=100)


class RssConfig(BaseModel):
    feeds: list[str] = Field(default_factory=list)
    channels: list[RssChannelConfig] = Field(default_factory=list)


class GithubConfig(BaseModel):
    repositories: list[str] = Field(default_factory=list)
    per_repo: int = Field(default=5, ge=1, le=20)
    max_age_hours: int = Field(default=168, ge=1, le=8760)


class IntelConfig(BaseModel):
    time_window_hours: int = 24
    per_source_timeout: float = 8.0
    max_news_items: int = 40
    sources: dict[str, bool] = Field(default_factory=dict)
    keys: dict[str, str] = Field(default_factory=dict)
    rss: RssConfig = Field(default_factory=RssConfig)
    github: GithubConfig = Field(default_factory=GithubConfig)


class ResearchConfig(BaseModel):
    """Bounded evidence research; the repository config enables it by default."""

    enabled: bool = False
    max_workers: int = Field(default=3, ge=1, le=3)
    max_tasks: int = Field(default=6, ge=1, le=6)
    max_rounds: int = Field(default=2, ge=1, le=2)
    max_documents: int = Field(default=12, ge=1, le=12)
    # Explicit operator values: do not guess provider latency/cost before probing.
    max_requests: int | None = Field(default=None, ge=1, le=500)
    deadline_s: float | None = Field(default=None, gt=0, le=3600)
    input_mode: Literal["title", "summary", "evidence"] = "evidence"
    workflow: Literal["single", "multi_no_debate", "multi_debate"] = "multi_debate"


class ConfidenceConfig(BaseModel):
    policy_version: str = "confidence-policy/v2"
    bucket_caps: dict[str, float] = Field(default_factory=dict)

    @field_validator("bucket_caps")
    @classmethod
    def _validate_bucket_caps(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(value) - _CONFIDENCE_BUCKET_LABELS)
        if unknown:
            raise ValueError(f"unknown confidence bucket(s): {', '.join(unknown)}")
        for label, cap in value.items():
            if not 0.1 <= float(cap) <= 0.95:
                raise ValueError(f"confidence bucket cap out of range for {label}")
        return {label: float(cap) for label, cap in value.items()}


class RuntimeConfig(BaseModel):
    context: ContextPolicy = Field(default_factory=ContextPolicy)
    checkpoint_backend: Literal["memory", "sqlite"] = "memory"
    checkpoint_path: str = ".runs/checkpoints.sqlite"
    checkpoint_retention: int | None = 100
    strategy_planner_mode: StrategyPlannerMode = "off"
    strategy_planner_enabled: bool = False
    strategy_planner_max_steps: int = 5  # 1-8;默认 5 覆盖确定性补抓最坏路径(质检+补2次+重查+指标)
    strategy_planner_llm_actions: bool = False
    # 分析前数据预检:数据质量不足时,行情/资讯各最多补抓一次再决定是否启动 LLM 图。
    preflight_refetch_enabled: bool = True
    # v0.2 兼容旧配置名;运行时覆盖会迁移到 preflight_refetch_enabled。
    strategy_planner_refetch_enabled: bool | None = None
    # 效果闭环(观测版):建议落库 → T+N 小时 CLI 打标 → 质量统计。默认关闭(仅门控只读端点,
    # 不影响 /analyze 主链路);CLI(python -m binance_agent.eval)为显式触发,不受此开关限制。
    enable_outcome_tracking: bool = False
    outcome_window_hours: int = Field(default=24, ge=1, le=720)
    outcome_kline_interval: OutcomeKlineInterval = "5m"

    @model_validator(mode="after")
    def _migrate_legacy_refetch_flag(self) -> RuntimeConfig:
        if self.strategy_planner_refetch_enabled is not None:
            warnings.warn(
                "strategy_planner_refetch_enabled is deprecated; use preflight_refetch_enabled",
                DeprecationWarning,
                stacklevel=2,
            )
            self.preflight_refetch_enabled = self.strategy_planner_refetch_enabled
        return self


class NetworkConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    # 显式代理(http://host:port 或 socks5://host:port)。留空则回退环境变量
    # (HTTPS_PROXY/HTTP_PROXY/ALL_PROXY,httpx trust_env 自动读取)。
    # 用于 Binance REST/WS 与资讯源出站;LLM 端点也经此路由(若已设)。
    proxy: str | None = None

    @field_validator("proxy")
    @classmethod
    def _proxy_without_credentials(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        proxy = value.strip()
        parsed = urlsplit(proxy)
        if parsed.scheme not in {"http", "https", "socks5", "socks5h"} or not parsed.hostname:
            raise ValueError("proxy must be an http(s) or socks5 URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("proxy credentials must be supplied through environment variables")
        return proxy


class CollaborationConfig(BaseModel):
    enabled: bool = False
    idle_unload_s: float = Field(default=60, gt=0)
    max_resident_agents: int = Field(default=32, ge=1)


class Settings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    profile: ProfileName = "default"
    app: AppConfig = Field(default_factory=AppConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    hotkey: HotkeyConfig = Field(default_factory=HotkeyConfig)
    desktop: DesktopConfig = Field(default_factory=DesktopConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    intel: IntelConfig = Field(default_factory=IntelConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    collaboration: CollaborationConfig = Field(default_factory=CollaborationConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)


def is_strategy_planner_mode(value: object) -> bool:
    return isinstance(value, str) and value in _STRATEGY_PLANNER_MODES


def planner_mode_from_flags(enabled: bool, llm_actions: bool) -> StrategyPlannerMode:
    if not enabled:
        return "off"
    return "llm_enforce" if llm_actions else "deterministic"


def planner_flags_for_mode(mode: StrategyPlannerMode) -> tuple[bool, bool, bool]:
    """Return (planner_enabled, llm_actions_enabled, enforce_decision)."""
    if mode == "off":
        return False, False, False
    if mode == "deterministic":
        return True, False, True
    if mode == "llm_shadow":
        return True, True, False
    return True, True, True


def effective_strategy_planner_mode(runtime: RuntimeConfig) -> StrategyPlannerMode:
    if runtime.strategy_planner_mode != "off":
        return runtime.strategy_planner_mode
    return planner_mode_from_flags(
        runtime.strategy_planner_enabled, runtime.strategy_planner_llm_actions
    )


def default_config_path() -> Path:
    """定位 config.toml:环境变量覆盖 > 当前工作目录 > 冻结包 > 包内项目根。"""
    env = os.environ.get(_ENV_CONFIG)
    if env:
        return Path(env)
    cwd_path = Path.cwd() / "config" / "config.toml"
    if cwd_path.exists():
        return cwd_path
    if getattr(sys, "frozen", False):
        exe_config = Path(sys.executable).resolve().parent / "config" / "config.toml"
        if exe_config.exists():
            return exe_config
        bundle_dir = getattr(sys, "_MEIPASS", None)
        if bundle_dir:
            bundled_config = Path(bundle_dir) / "config" / "config.toml"
            if bundled_config.exists():
                return bundled_config
    return Path(__file__).resolve().parents[3] / "config" / "config.toml"


def load_settings(path: Path | str | None = None) -> Settings:
    """加载并校验配置。先注入 .env,再读 TOML;无配置文件则用默认值。"""
    from dotenv import load_dotenv

    load_dotenv()
    config_path = Path(path) if path else default_config_path()
    if not config_path.exists():
        return Settings()
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    return Settings.model_validate(data)
