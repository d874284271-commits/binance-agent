"""策略可信度闭环:数据质量门控 + 可 veto 风险闸门。

纯函数,无 LLM/网络。LLM 只负责候选策略,本模块用确定性规则决定是否允许交易、
是否降级为 watchlist/no_trade,并追加可解释风险原因。
"""

from __future__ import annotations

import re
from math import isfinite
from typing import Any

from pydantic import BaseModel, Field

from ..indicators import ta
from .state import AnalystReport, StrategyAdvice

_PRICE_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:万|千|[kK])?")
# 剥离「数字+倍」记号(如 1.5倍ATR / 2倍),避免把 ATR 倍数误当价位。
_MULTIPLIER_RE = re.compile(r"-?\d+(?:\.\d+)?\s*倍")
_CORE_INTERVALS = ("1h", "4h")
_MIN_KLINES = 30
# 策略价位相对当前价(lastPrice)允许的偏差范围;超出则视为与行情不一致。
PLAUSIBLE_PRICE_RANGE = 0.20


class DataQualityReport(BaseModel):
    level: str = "ok"
    score: float = 1.0
    trade_blocked: bool = False
    reasons: list[str] = Field(default_factory=list)
    kline_completeness: dict[str, float] = Field(default_factory=dict)
    missing_market_fields: list[str] = Field(default_factory=list)
    market_error_count: int = 0
    source_success_rate: float = 1.0
    failed_sources: list[str] = Field(default_factory=list)
    unavailable_sources: list[str] = Field(default_factory=list)


class RiskGateResult(BaseModel):
    veto: bool = False
    warnings: list[str] = Field(default_factory=list)
    confidence_cap: float | None = None
    decision_type: str | None = None
    reason: str | None = None
    reentry_conditions: list[str] = Field(default_factory=list)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if isfinite(out) else None


def _extract_prices(value: Any) -> list[float]:
    if value in (None, "", "null"):
        return []
    prices: list[float] = []
    # 先剥离「数字+倍」记号(如 1.5倍ATR),避免把 ATR 倍数误当价位;
    # 用代码剥离而非改正则,防止把 1.5倍 误拆成 1 与 5。
    cleaned = _MULTIPLIER_RE.sub(" ", str(value))
    for match in _PRICE_RE.finditer(cleaned):
        token = match.group(0).strip()
        multiplier = 1.0
        if token.endswith("万"):
            token = token[:-1]
            multiplier = 10000.0
        elif token.endswith(("千", "k", "K")):
            token = token[:-1]
            multiplier = 1000.0
        number = _as_float(token.replace(",", "").strip())
        if number is not None:
            prices.append(abs(number * multiplier))
    return prices


def _entry_price(advice: StrategyAdvice) -> float | None:
    prices = _extract_prices(advice.entry_zone)
    if not prices:
        return None
    return sum(prices) / len(prices)


def _target_prices(advice: StrategyAdvice) -> list[float]:
    prices: list[float] = []
    for target in advice.targets:
        extracted = _extract_prices(target)
        if extracted:
            prices.append(sum(extracted) / len(extracted))
    return prices


def _price_consistency_issues(advice: StrategyAdvice, market_snapshot: dict[str, Any]) -> list[str]:
    """检测策略价位是否与当前行情明显脱节(凭空编造)。

    锚点取 ``ticker_24h.lastPrice``(与 LLM 收到的 ``last_price`` 同源)。
    无锚点时跳过(不阻断)。对入场参考价、止损、每个目标,超出锚点
    ``±PLAUSIBLE_PRICE_RANGE`` 视为不一致,返回可读问题;口径与风险闸门取价一致。
    """
    ticker = market_snapshot.get("ticker_24h") or {}
    anchor = _as_float(ticker.get("lastPrice"))
    if anchor is None:
        return []

    issues: list[str] = []
    entry = _entry_price(advice)
    if entry is not None and _out_of_range(entry, anchor):
        issues.append(f"[风控] 入场价 {entry:g} 相对当前价 {anchor:g} 偏差过大,疑为凭空编造")
    stop = _extract_prices(advice.stop_loss)
    if stop and _out_of_range(stop[0], anchor):
        issues.append(f"[风控] 止损价 {stop[0]:g} 相对当前价 {anchor:g} 偏差过大,疑为凭空编造")
    for target in _target_prices(advice):
        if _out_of_range(target, anchor):
            issues.append(f"[风控] 目标价 {target:g} 相对当前价 {anchor:g} 偏差过大,疑为凭空编造")
    return issues


def _out_of_range(value: float, anchor: float) -> bool:
    if anchor == 0:
        return False
    return abs(value - anchor) / abs(anchor) > PLAUSIBLE_PRICE_RANGE


# 公共别名:供 eval 效果闭环打标模块复用价格解析,避免跨模块 import 私有(下划线)符号。
extract_prices = _extract_prices
entry_price = _entry_price
target_prices = _target_prices


def _has_key_levels(market_snapshot: dict[str, Any]) -> bool:
    rows = (market_snapshot.get("klines") or {}).get("1h") or []
    if len(rows) < _MIN_KLINES:
        return False
    try:
        highs = [float(r["high"]) for r in rows]
        lows = [float(r["low"]) for r in rows]
        closes = [float(r["close"]) for r in rows]
    except (KeyError, TypeError, ValueError):
        return False
    summary = ta.summarize(highs, lows, closes)
    return (
        _as_float(summary.get("atr14")) is not None and _as_float(summary.get("close")) is not None
    )


def build_data_quality_report(
    market_snapshot: dict[str, Any],
    intel_snapshot: dict[str, Any],
) -> DataQualityReport:
    """根据行情近端 K 线完整度、ticker 与资讯源状态生成确定性质量报告。

    只检查传入快照,不补抓数据;报告的分数与交易阻断状态用于预检和下游闸门,
    不能表示新闻语义真实或预测准确率。
    """
    reasons: list[str] = []
    missing: list[str] = []
    score = 1.0

    klines = market_snapshot.get("klines") or {}
    completeness: dict[str, float] = {}
    for interval in ("5m", "1h", "4h", "1d"):
        rows = klines.get(interval) or []
        valid = 0
        # 币安 K 线按时间升序返回,取尾部(最新)_MIN_KLINES 根校验——指标计算依赖的正是近端数据
        for row in rows[-_MIN_KLINES:]:
            try:
                float(row["high"])
                float(row["low"])
                float(row["close"])
            except (KeyError, TypeError, ValueError):
                continue
            valid += 1
        completeness[interval] = min(1.0, valid / _MIN_KLINES)

    for interval in _CORE_INTERVALS:
        if completeness.get(interval, 0.0) < 1.0:
            missing.append(f"klines[{interval}]")

    ticker = market_snapshot.get("ticker_24h") or {}
    for field in ("lastPrice", "highPrice", "lowPrice", "volume", "quoteVolume"):
        if _as_float(ticker.get(field)) is None:
            missing.append(f"ticker_24h.{field}")

    errors = [str(err) for err in (market_snapshot.get("errors") or [])]
    if errors:
        reasons.append(f"行情源返回 {len(errors)} 个错误")
        score -= min(0.3, 0.08 * len(errors))

    if missing:
        reasons.append("关键行情字段缺失:" + ", ".join(missing[:6]))
        score -= 0.2 + min(0.3, 0.04 * len(missing))

    source_status = intel_snapshot.get("source_status") or {}
    ok_sources: list[str] = []
    failed_sources: list[str] = []
    unavailable_sources: list[str] = []
    for name, status in source_status.items():
        category = _source_status_category(status)
        if category == "ok":
            ok_sources.append(name)
        elif category == "unavailable":
            unavailable_sources.append(name)
        else:
            failed_sources.append(name)
    available_sources = len(ok_sources) + len(failed_sources)
    if source_status:
        source_success_rate = len(ok_sources) / available_sources if available_sources else 1.0
    else:
        source_success_rate = 0.0
        reasons.append("无资讯源状态")
        score -= 0.15
    if available_sources and source_success_rate < 0.5:
        reasons.append(f"资讯源成功率过低:{source_success_rate:.0%}")
        score -= 0.25
    elif available_sources and source_success_rate < 0.8:
        reasons.append(f"资讯源部分失败:{source_success_rate:.0%}")
        score -= 0.1

    if not (intel_snapshot.get("news_items") or intel_snapshot.get("sentiment_features")):
        reasons.append("资讯/情绪数据为空")
        score -= 0.1

    score = max(0.0, min(1.0, score))
    critical_market_missing = any(item.startswith("klines[") for item in missing) or any(
        item == "ticker_24h.lastPrice" for item in missing
    )
    trade_blocked = critical_market_missing or score < 0.45
    if trade_blocked:
        level = "critical"
    elif score < 0.75 or (available_sources and source_success_rate < 0.8):
        level = "degraded"
    else:
        level = "ok"

    return DataQualityReport(
        level=level,
        score=score,
        trade_blocked=trade_blocked,
        reasons=reasons,
        kline_completeness=completeness,
        missing_market_fields=missing,
        market_error_count=len(errors),
        source_success_rate=round(source_success_rate, 4),
        failed_sources=failed_sources,
        unavailable_sources=unavailable_sources,
    )


def _source_status_category(status: Any) -> str:
    raw = str(status or "").strip()
    normalized = raw.lower()
    if normalized == "ok":
        return "ok"
    if normalized.startswith("disabled") or raw.startswith("缺少 key") or "未安装" in raw:
        return "unavailable"
    if (
        normalized.startswith("error")
        or "timeout" in normalized
        or "http " in normalized
        or "rate_limited" in normalized
        or "blocked" in normalized
    ):
        return "failed"
    return "failed"


def _rsi_1h(market_snapshot: dict[str, Any]) -> float | None:
    klines = market_snapshot.get("klines") or {}
    rows = klines.get("1h") or klines.get("4h") or []
    if len(rows) < 30:
        return None
    try:
        highs = [float(r["high"]) for r in rows]
        lows = [float(r["low"]) for r in rows]
        closes = [float(r["close"]) for r in rows]
    except (KeyError, TypeError, ValueError):
        return None
    return ta.summarize(highs, lows, closes).get("rsi14")


def validate_advice(
    advice: StrategyAdvice,
    market_snapshot: dict[str, Any],
    analyst_reports: list[AnalystReport],
) -> list[str]:
    """检查建议方向与 RSI、置信度及分析师分歧,返回附加警告,不修改 advice。"""
    warnings: list[str] = []
    rsi = _rsi_1h(market_snapshot)

    if advice.direction == "long" and rsi is not None and rsi > 75:
        warnings.append(f"[校验] 看多但 RSI={rsi:.0f} 超买(>75),警惕追高回调")
    if advice.direction == "short" and rsi is not None and rsi < 25:
        warnings.append(f"[校验] 看空但 RSI={rsi:.0f} 超卖(<25),警惕超跌反弹")

    if advice.direction != "neutral" and advice.confidence < 0.4:
        warnings.append(
            f"[校验] 方向 {advice.direction} 但置信度仅 {advice.confidence:.0%},信号不明确"
        )

    stances = [r.stance for r in analyst_reports]
    bulls = stances.count("bullish")
    bears = stances.count("bearish")
    if advice.direction == "long" and bears > bulls:
        warnings.append(f"[校验] 建议看多,但多数分析师看空({bears}空/{bulls}多),分歧较大")
    if advice.direction == "short" and bulls > bears:
        warnings.append(f"[校验] 建议看空,但多数分析师看多({bulls}多/{bears}空),分歧较大")

    return warnings


def evaluate_risk_gate(
    advice: StrategyAdvice,
    market_snapshot: dict[str, Any],
    risk_assessment: dict[str, Any] | None,
    analyst_reports: list[AnalystReport],
) -> RiskGateResult:
    """以价位方向、风险收益、行情关键值和宏观风险计算确定性闸门结果。

    返回 veto、置信上限或 watchlist 条件,不直接修改建议;数据缺失可导致
    降级,错误的止损/目标方向或高宏观风险可否决交易。
    """
    result = RiskGateResult()

    if advice.direction == "neutral" or advice.decision_type == "no_trade":
        return result

    entry = _entry_price(advice)
    stops = _extract_prices(advice.stop_loss)
    stop = stops[0] if stops else None
    targets = _target_prices(advice)

    if entry is None or stop is None or not targets:
        result.warnings.append("[风控] 入场/止损/目标不完整,无法评估风险收益")
        result.confidence_cap = _cap(result.confidence_cap, 0.35)
        result.decision_type = "watchlist"
        result.reentry_conditions.append("补齐入场、止损和目标价后重新评估")
    else:
        if advice.direction == "long":
            if stop >= entry:
                return RiskGateResult(
                    veto=True,
                    warnings=["[风控] long 止损必须低于入场区间"],
                    reason="long 止损方向错误",
                    reentry_conditions=["止损回到入场价下方并重新计算风险收益比"],
                )
            bad_targets = [target for target in targets if target <= entry]
            reward = max(targets) - entry
            risk_distance = entry - stop
        else:
            if stop <= entry:
                return RiskGateResult(
                    veto=True,
                    warnings=["[风控] short 止损必须高于入场区间"],
                    reason="short 止损方向错误",
                    reentry_conditions=["止损回到入场价上方并重新计算风险收益比"],
                )
            bad_targets = [target for target in targets if target >= entry]
            reward = entry - min(targets)
            risk_distance = stop - entry

        if bad_targets:
            return RiskGateResult(
                veto=True,
                warnings=["[风控] 目标价方向与交易方向相反"],
                reason="目标价方向错误",
                reentry_conditions=["目标价回到顺势方向后重新评估"],
            )

        if risk_distance <= 0:
            return RiskGateResult(
                veto=True,
                warnings=["[风控] 风险距离无效"],
                reason="风险距离无效",
                reentry_conditions=["重新给出有效入场和止损"],
            )

        reward_risk = reward / risk_distance
        if reward_risk < 1.0:
            return RiskGateResult(
                veto=True,
                warnings=[f"[风控] 风险收益比 {reward_risk:.2f} < 1.0"],
                reason="风险收益比不足",
                reentry_conditions=["等待更优入场或更明确目标,使风险收益比至少达到 1.5"],
            )
        if reward_risk < 1.5:
            result.warnings.append(f"[风控] 风险收益比 {reward_risk:.2f} < 1.5")
            result.confidence_cap = _cap(result.confidence_cap, 0.45)
            result.decision_type = "watchlist"
            result.reentry_conditions.append("等待风险收益比提升至 1.5 以上")

        # 价位一致性:LLM 给价是否明显偏离当前行情(凭空编造)。软封顶,不硬否决。
        for issue in _price_consistency_issues(advice, market_snapshot):
            result.warnings.append(issue)
            result.confidence_cap = _cap(result.confidence_cap, 0.35)
            result.decision_type = result.decision_type or "watchlist"
            result.reentry_conditions.append("等待策略价位与当前行情一致后重新评估")

    if not _has_key_levels(market_snapshot):
        result.warnings.append("[风控] ATR/关键价位缺失,策略价位可信度降低")
        result.confidence_cap = _cap(result.confidence_cap, 0.45)
        result.decision_type = result.decision_type or "watchlist"
        result.reentry_conditions.append("等待 1h K 线足够计算 ATR 与关键价位")

    risk = risk_assessment or {}
    if risk.get("level") == "high":
        return RiskGateResult(
            veto=True,
            warnings=["[风控] 高重要性宏观事件窗口内不允许开新单"],
            reason="宏观事件风险过高",
            reentry_conditions=["等待宏观事件落地且波动收敛后重新评估"],
        )

    stances = [report.stance for report in analyst_reports]
    bulls = stances.count("bullish")
    bears = stances.count("bearish")
    avg_confidence = (
        sum(report.confidence for report in analyst_reports) / len(analyst_reports)
        if analyst_reports
        else 0.0
    )
    contested = bulls > 0 and bears > 0
    weak_consensus = avg_confidence < 0.5
    if contested or weak_consensus:
        reason = "分析师分歧大" if contested else "分析师低置信"
        result.warnings.append(f"[风控] {reason},不允许高置信单边策略")
        result.confidence_cap = _cap(result.confidence_cap, 0.55)
        if advice.confidence >= 0.7:
            result.decision_type = result.decision_type or "watchlist"
            result.reentry_conditions.append("等待分析师共识增强或置信度修复")

    return result


def apply_quality_and_risk_gates(
    advice: StrategyAdvice,
    data_quality: DataQualityReport,
    risk_assessment: dict[str, Any] | None,
    analyst_reports: list[AnalystReport],
    market_snapshot: dict[str, Any] | None = None,
    pre_computed_risk_gate: RiskGateResult | None = None,
) -> StrategyAdvice:
    """将数据质量与风险闸门应用到建议,返回降级后的同一 advice 对象。

    可能原地调整置信度、方向、价位、警告及重评条件;调用方需要保留候选原文时
    应先复制。可复用已计算的 risk gate,数据质量阻断优先转为 no_trade。
    """
    warnings = [*advice.risk_warnings]
    warnings.extend(f"[数据质量] {reason}" for reason in data_quality.reasons)

    if data_quality.trade_blocked:
        return _to_no_trade(
            advice,
            reason="数据质量不足:" + "; ".join(data_quality.reasons or ["关键行情缺失"]),
            warnings=warnings,
            reentry_conditions=[
                "等待关键 K 线、ticker 与行情错误恢复",
                "资讯源成功率恢复后重新评估",
            ],
            confidence=min(advice.confidence, 0.25),
        )

    if advice.decision_type == "no_trade":
        return _to_no_trade(
            advice,
            reason=advice.no_trade_reason or "策略明确选择观望",
            warnings=warnings,
            reentry_conditions=advice.reentry_conditions or ["等待触发重新评估条件"],
            confidence=advice.confidence,
        )

    if data_quality.level == "degraded":
        warnings.append("[数据质量] 资讯源或行情源部分失败,下调策略置信度")
        advice.confidence = min(advice.confidence, 0.55)

    # 复用调用方已算好的 risk_gate(strategy_synth 已为持久化/置信度校准算过一次),避免重复计算
    gate = (
        pre_computed_risk_gate
        if pre_computed_risk_gate is not None
        else evaluate_risk_gate(advice, market_snapshot or {}, risk_assessment, analyst_reports)
    )
    warnings.extend(gate.warnings)
    if gate.veto:
        return _to_no_trade(
            advice,
            reason=gate.reason or "风险闸门否决",
            warnings=warnings,
            reentry_conditions=gate.reentry_conditions,
            confidence=min(advice.confidence, 0.25),
        )

    if gate.confidence_cap is not None:
        advice.confidence = min(advice.confidence, gate.confidence_cap)
    if gate.decision_type == "watchlist":
        advice.decision_type = "watchlist"
        advice.reentry_conditions = _dedup([*advice.reentry_conditions, *gate.reentry_conditions])
    elif advice.direction == "neutral":
        advice.decision_type = "no_trade"
        advice.entry_zone = None
        advice.stop_loss = None
        advice.targets = []
        if not advice.no_trade_reason:
            advice.no_trade_reason = "方向中性,没有足够业务证据支持交易"
        if not advice.reentry_conditions:
            advice.reentry_conditions = ["等待方向共识增强后重新评估"]
    else:
        advice.decision_type = "trade"

    advice.risk_warnings = _dedup(warnings)
    return advice


def _cap(current: float | None, cap: float) -> float:
    return cap if current is None else min(current, cap)


def _dedup(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _to_no_trade(
    advice: StrategyAdvice,
    *,
    reason: str,
    warnings: list[str],
    reentry_conditions: list[str],
    confidence: float,
) -> StrategyAdvice:
    advice.direction = "neutral"
    advice.decision_type = "no_trade"
    advice.entry_zone = None
    advice.stop_loss = None
    advice.targets = []
    advice.confidence = max(0.0, min(1.0, confidence))
    advice.no_trade_reason = reason
    advice.reentry_conditions = _dedup([*advice.reentry_conditions, *reentry_conditions])
    advice.risk_warnings = _dedup(warnings)
    return advice
