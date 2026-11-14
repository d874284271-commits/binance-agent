"""确定性规则置信评分:裁剪模型候选值,保留依据与人工策略版本。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .state import AnalystReport, StrategyAdvice
from .validation import DataQualityReport, RiskGateResult


class CalibratedConfidence(BaseModel):
    calibrated_confidence: float
    # Kept optional for compatibility; evidence support is calculated from
    # citation validation in the research graph, not from this rule score.
    evidence_confidence: float | None = None
    pre_policy_confidence: float
    confidence_cap: float
    policy_bucket: str
    policy_version: str
    confidence_reasons: list[str] = Field(default_factory=list)


def calibrate_confidence(
    *,
    advice: StrategyAdvice,
    data_quality: DataQualityReport,
    risk_gate: RiskGateResult,
    analyst_reports: list[AnalystReport],
    runtime_events: list[dict[str, Any]],
    risk_assessment: dict[str, Any] | None,
    debate_grounding: dict[str, Any] | None = None,
    bucket_caps: dict[str, float] | None = None,
    policy_version: str = "confidence-policy/v2",
) -> CalibratedConfidence:
    """计算经过质量、风控、来源共识及 runtime 可靠性裁剪的规则置信分。

    先得到 pre_policy_confidence,再按其所属桶应用人工上限;返回解释原因,
    不修改 advice。该规则分不代表盈利概率,证据引用比例由研究图单独计算。
    """

    reasons: list[str] = []
    cap = 0.95
    model_confidence = _bounded(advice.model_confidence, default=advice.confidence)
    groups = _independent_groups(analyst_reports)
    analyst_avg = _average_confidence(groups)
    consensus = _consensus(advice.direction, analyst_reports, groups)

    if data_quality.trade_blocked:
        cap = _cap(cap, 0.25)
        reasons.append("数据质量阻断交易,置信度上限 25%")
    elif data_quality.level == "critical":
        cap = _cap(cap, 0.35)
        reasons.append("数据质量 critical,置信度上限 35%")
    elif data_quality.level == "degraded":
        cap = _cap(cap, 0.55)
        reasons.append("数据质量 degraded,置信度上限 55%")

    if data_quality.score < 0.75:
        cap = _cap(cap, max(0.25, data_quality.score))
        reasons.append(f"数据质量分 {data_quality.score:.0%},限制最终置信度")
    if data_quality.source_success_rate < 0.5:
        cap = _cap(cap, 0.45)
        reasons.append(f"资讯源成功率 {data_quality.source_success_rate:.0%},限制最终置信度")

    if risk_gate.veto:
        cap = _cap(cap, 0.25)
        reasons.append(f"RiskGate veto: {risk_gate.reason or '风险闸门否决'}")
    if risk_gate.confidence_cap is not None:
        cap = _cap(cap, risk_gate.confidence_cap)
        reasons.append(f"RiskGate 置信度上限 {risk_gate.confidence_cap:.0%}")
    if any("风险收益比" in warning for warning in risk_gate.warnings):
        cap = _cap(cap, 0.45)
        reasons.append("风险收益比不足,限制最终置信度")

    if (risk_assessment or {}).get("level") == "high":
        cap = _cap(cap, 0.25)
        reasons.append("宏观/事件风险 high,置信度上限 25%")

    strategy_runtime = _strategy_runtime(runtime_events)
    if strategy_runtime is not None:
        if not bool(strategy_runtime.get("ok")):
            cap = _cap(cap, 0.25)
            reasons.append("strategy_synth fallback,置信度上限 25%")
        elif bool(strategy_runtime.get("repaired")):
            cap = _cap(cap, 0.65)
            reasons.append("strategy_synth JSON repair,置信度上限 65%")
        elif strategy_runtime.get("errors"):
            cap = _cap(cap, 0.75)
            reasons.append("strategy_synth runtime 出现错误后恢复,置信度上限 75%")

    failed_runtime = [
        event
        for event in runtime_events
        if not bool(event.get("ok", True)) and event.get("agent") != "strategy_synth"
    ]
    if failed_runtime:
        cap = _cap(cap, 0.6)
        reasons.append("存在 analyst/debater runtime fallback,置信度上限 60%")

    for group in groups:
        if len(group) > 1:
            names = "/".join(sorted(report.analyst for report in group))
            reasons.append(f"分析师 {names} 共用数据源,方向共识按 1 票计")
    if consensus["contested"]:
        cap = _cap(cap, 0.55)
        reasons.append("分析师分歧大,不允许高置信单边策略")
    if consensus["weak"]:
        cap = _cap(cap, 0.5)
        reasons.append("分析师平均置信度低,限制最终置信度")
    if consensus["aligned"] and analyst_avg >= 0.75 and data_quality.level == "ok":
        reasons.append("分析师方向一致且数据质量良好,允许较高置信度")
    elif advice.decision_type == "trade" and not consensus["aligned"]:
        cap = _cap(cap, 0.6)
        reasons.append("交易方向缺少分析师一致支持,限制最终置信度")

    # 证据完整性只衡量结构化引用路径和值;0 claims 明确按 0% 处理。
    # 有方向时看支持方去重后的独立数据点,并比较双方可核验证据的多寡。
    grounding = debate_grounding or {}
    by_side = grounding.get("by_side")
    support_side = {"long": "bull", "short": "bear"}.get(advice.direction)
    if support_side and isinstance(by_side, dict) and isinstance(by_side.get(support_side), dict):
        support = by_side[support_side]
        oppose = by_side.get("bear" if support_side == "bull" else "bull") or {}
        label = "多" if support_side == "bull" else "空"
        source_ratio = _bounded(support.get("source_verified_ratio"), default=0.0)
        if source_ratio < 0.5:
            cap = _cap(cap, 0.6)
            reasons.append(f"辩论支持方({label})独立证据完整性 {source_ratio:.0%},限制最终置信度")
        support_count = int(support.get("verified_sources") or 0)
        oppose_count = int(oppose.get("verified_sources") or 0)
        if oppose_count > support_count:
            cap = _cap(cap, 0.55)
            reasons.append(
                f"辩论反方可核验独立证据多于支持方({oppose_count}>{support_count}),限制最终置信度"
            )
    else:
        verified_ratio = grounding.get("verified_ratio")
        if verified_ratio is None:  # v0.1 历史报告读取兼容
            verified_ratio = grounding.get("grounded_ratio")
        if isinstance(verified_ratio, (int, float)) and verified_ratio < 0.5:
            cap = _cap(cap, 0.6)
            reasons.append(f"辩论证据完整性 {verified_ratio:.0%},限制最终置信度")

    rule_signal = _evidence_confidence(model_confidence, analyst_avg, data_quality.score)
    pre_policy = min(advice.confidence, rule_signal, cap)
    policy_bucket = confidence_bucket(pre_policy)
    policy_cap = (bucket_caps or {}).get(policy_bucket)
    calibrated = apply_bucket_caps(pre_policy, bucket_caps)
    if isinstance(policy_cap, (int, float)):
        policy_cap = _bounded(policy_cap)
        cap = _cap(cap, policy_cap)
        if calibrated < pre_policy:
            reasons.append(
                f"人工校准策略 {policy_version} 将 {policy_bucket} 桶置信度限制为 {policy_cap:.0%}"
            )
    return CalibratedConfidence(
        calibrated_confidence=round(_bounded(calibrated), 4),
        evidence_confidence=None,
        pre_policy_confidence=round(_bounded(pre_policy), 4),
        confidence_cap=round(_bounded(cap), 4),
        policy_bucket=policy_bucket,
        policy_version=policy_version,
        confidence_reasons=_dedup(reasons),
    )


def apply_bucket_caps(value: float, bucket_caps: dict[str, float] | None) -> float:
    """按应用策略前的分桶施加一次上限;缺失或布尔 cap 忽略,结果只能降低。

    不会因降低后落入另一桶而迭代应用其他桶,runtime 与校准验证共用此口径。
    """
    confidence = _bounded(value)
    cap = (bucket_caps or {}).get(confidence_bucket(confidence))
    if not isinstance(cap, bool) and isinstance(cap, (int, float)):
        return min(confidence, _bounded(cap))
    return confidence


def confidence_bucket(value: Any) -> str:
    """把值约束到 [0, 1] 后映射到固定置信桶;缺失或无法转换的值按 0 处理。"""
    bounded = _bounded(value, default=0.0)
    if bounded < 0.4:
        return "[0.0,0.4)"
    if bounded < 0.6:
        return "[0.4,0.6)"
    if bounded < 0.8:
        return "[0.6,0.8)"
    return "[0.8,1.0]"


def _strategy_runtime(runtime_events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(runtime_events):
        if event.get("agent") == "strategy_synth":
            return event
    return None


def _independent_groups(reports: list[AnalystReport]) -> list[list[AnalystReport]]:
    """data_sources 有交集的分析师合并为一组;未声明来源的报告(历史记录)视为独立。

    已有组两两不相交,新报告只需与每组比较,即可得到传递闭包。
    """
    groups: list[tuple[set[str], list[AnalystReport]]] = []
    for report in reports:
        sources = set(report.data_sources)
        merged_sources, merged_reports = set(sources), []
        remaining: list[tuple[set[str], list[AnalystReport]]] = []
        for group_sources, members in groups:
            if sources & group_sources:
                merged_sources |= group_sources
                merged_reports.extend(members)
            else:
                remaining.append((group_sources, members))
        groups = [*remaining, (merged_sources, [*merged_reports, report])]
    return [members for _, members in groups]


def _consensus(
    direction: str, reports: list[AnalystReport], groups: list[list[AnalystReport]]
) -> dict[str, bool]:
    """分歧按原始 stance 判断(保守);方向一致按独立来源组计票,同源分析师只算一票。"""
    if not reports:
        return {"aligned": False, "contested": False, "weak": True}
    stances = [report.stance for report in reports]
    contested = "bullish" in stances and "bearish" in stances
    group_stances = [{report.stance for report in group} for group in groups]
    bull_votes = sum(1 for s in group_stances if "bullish" in s and "bearish" not in s)
    bear_votes = sum(1 for s in group_stances if "bearish" in s and "bullish" not in s)
    if direction == "long":
        aligned = bull_votes >= 2 and "bearish" not in stances
    elif direction == "short":
        aligned = bear_votes >= 2 and "bullish" not in stances
    else:
        aligned = sum(1 for s in group_stances if s == {"neutral"}) >= 2
    return {
        "aligned": aligned,
        "contested": contested,
        "weak": _average_confidence(groups) < 0.5,
    }


def _average_confidence(groups: list[list[AnalystReport]]) -> float:
    """先组内平均再跨组平均,同源分析师不被重复加权。"""
    if not groups:
        return 0.0
    means = [sum(_bounded(report.confidence) for report in g) / len(g) for g in groups]
    return sum(means) / len(means)


def _evidence_confidence(model_confidence: float, analyst_avg: float, data_score: float) -> float:
    return _bounded(model_confidence * 0.45 + analyst_avg * 0.4 + _bounded(data_score) * 0.15)


def _bounded(value: Any, *, default: float = 0.5) -> float:
    try:
        out = float(default if value is None else value)
    except (TypeError, ValueError):
        out = default
    return max(0.0, min(1.0, out))


def _cap(current: float, cap: float) -> float:
    return min(current, _bounded(cap))


def _dedup(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out
