"""多智能体编排状态与结构化产物(数据结构基石)。

并行写同一 key 必须配 reducer:analyst_reports / debate_turns 用 operator.add 累加。
"""

from __future__ import annotations

import json
import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator
from typing_extensions import NotRequired

DISCLAIMER = "本工具仅供信息分析,不构成任何投资建议,加密交易风险极高,决策与盈亏由用户自负。"


def merge_context_compactions(
    previous: dict[str, Any], additions: dict[str, Any]
) -> dict[str, Any]:
    """合并并行节点按输入版本 hash 键控的压缩缓存,返回新字典;同键以 additions 为准。"""
    return {**previous, **additions}


class AnalystReport(BaseModel):
    analyst: str  # technical / funding / news_sentiment
    stance: Literal["bullish", "bearish", "neutral"] = "neutral"
    confidence: float = 0.5  # 0..1
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    # 由节点代码按实际输入填写(不来自 LLM);置信度校准据此把同源分析师合并为一票。
    data_sources: list[str] = Field(default_factory=list)


class DebateClaim(BaseModel):
    statement: str
    evidence: str
    source_path: str | None = None
    source_value: str | int | float | bool | None = None
    # v0.1 历史记录读取兼容;新 prompt / 新持久化产物只使用 source_path/source_value。
    source_ref: str | None = Field(default=None, exclude=True, repr=False)
    confidence: float = 0.5
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def _read_legacy_source_ref(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("source_path"):
            return value
        legacy = str(value.get("source_ref") or "").strip()
        if not legacy:
            return value
        path, separator, raw_value = legacy.partition("=")
        migrated = dict(value)
        migrated["source_path"] = path.strip() or None
        if separator:
            migrated["source_value"] = _legacy_scalar(raw_value.strip())
        return migrated


class DebateTurn(BaseModel):
    side: Literal["bull", "bear"]
    round: int
    argument: str
    structured_claims: list[DebateClaim] = Field(default_factory=list)
    snapshot_ref_at: datetime | None = None


class StrategyAdvice(BaseModel):
    symbol: str
    direction: Literal["long", "short", "neutral"] = "neutral"
    decision_type: Literal["trade", "no_trade", "watchlist"] = "trade"
    entry_zone: str | None = None
    stop_loss: str | None = None
    targets: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    model_confidence: float | None = None
    evidence_confidence: float | None = None
    outcome_confidence: float | None = None
    pre_policy_confidence: float | None = None
    confidence_cap: float | None = None
    confidence_bucket: str | None = None
    confidence_policy_version: str | None = None
    confidence_reasons: list[str] = Field(default_factory=list)
    rationale: str = ""
    risk_warnings: list[str] = Field(default_factory=list)
    no_trade_reason: str | None = None
    reentry_conditions: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    evidence_status: str = "legacy_no_original_evidence"
    disclaimer: str = DISCLAIMER


class AgentState(TypedDict):
    mode: NotRequired[str]
    target: NotRequired[dict[str, Any]]
    title: NotRequired[str]
    final_report: NotRequired[dict[str, Any] | None]
    discovery: NotRequired[dict[str, Any]]
    input_revision: NotRequired[int]
    context_compactions: Annotated[dict[str, Any], merge_context_compactions]
    candidate_advice: NotRequired[StrategyAdvice | None]
    research_demo: NotRequired[bool]
    research_legacy: NotRequired[bool]
    research_run_id: NotRequired[str]
    research_question: NotRequired[str]
    analysis_at: NotRequired[str]
    research_replay: NotRequired[bool]
    news_research: NotRequired[dict[str, Any]]
    research_budget: NotRequired[dict[str, Any]]
    evidence_documents: NotRequired[list[dict[str, Any]]]
    symbol: str
    market: Literal["fapi", "api"]
    market_snapshot: dict[str, Any]  # 行情
    preflight_refetch_enabled: NotRequired[bool]
    snapshot_freshness_enabled: NotRequired[bool]
    snapshot_freshness_now: NotRequired[datetime | None]
    latest_price: NotRequired[float | None]
    intel_snapshot: dict[str, Any]  # 资讯/情绪/宏观
    analyst_reports: Annotated[list[AnalystReport], operator.add]  # 并行累加
    debate_turns: Annotated[list[DebateTurn], operator.add]
    runtime_events: Annotated[list[dict[str, Any]], operator.add]
    tool_events: Annotated[list[dict[str, Any]], operator.add]
    planner_steps: Annotated[list[dict[str, Any]], operator.add]
    planner_decision: dict[str, Any] | None
    debate_round: int
    max_debate_rounds: int  # 硬上限,防不收敛
    debate_stop_reason: NotRequired[str | None]  # max_rounds / no_new_evidence
    research_summary: str | None
    preflight_report: NotRequired[dict[str, Any] | None]
    debate_grounding: NotRequired[dict[str, Any] | None]  # referee 报告(无 reducer,收敛后写一次)
    risk_assessment: dict[str, Any] | None
    risk_gate_result: dict[str, Any] | None
    final_advice: StrategyAdvice | None


def _legacy_scalar(raw: str) -> str | int | float | bool | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    return value if isinstance(value, (str, int, float, bool)) or value is None else raw
