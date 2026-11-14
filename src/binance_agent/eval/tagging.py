"""效果闭环打标核心(纯函数,无 IO / 无时钟依赖,便于确定性单测)。

输入:一条 run 记录(含 ``final_advice`` / ``symbol`` / ``market``)+ 建议生成后回拉的
K 线序列;输出:一条 :class:`OutcomeRecord`。判定边界见下方各步注释。

价格解析复用 :mod:`binance_agent.agents.validation` 的公共别名(``extract_prices``),
与策略校验层同源,避免重复实现区间/万千单位解析。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import ceil, floor
from typing import Any

from ..agents.validation import extract_prices

# outcome 枚举
TARGET_HIT = "target_hit"
STOP_HIT = "stop_hit"
NOT_ENTERED = "not_entered"
OPEN = "open"
NOT_APPLICABLE = "not_applicable"
UNEVALUABLE = "unevaluable"
FETCH_FAILED = "fetch_failed"
OUTCOME_POLICY_VERSION = "ohlc-conservative-v1"


@dataclass
class OutcomeRecord:
    """打标结果。字段与 :meth:`FileOutcomeStore.append` 的入参一一对应。"""

    run_id: str
    symbol: str
    market: str | None
    direction: str | None
    decision_type: str | None
    confidence: float | None
    pre_policy_confidence: float | None
    model_confidence: float | None
    outcome: str
    entry: float | None = None
    stop: float | None = None
    targets: list[float] = field(default_factory=list)
    r_realized: float | None = None
    entered_at: str | None = None
    evaluation_started_at: str | None = None
    evaluation_ended_at: str | None = None
    entry_fill: float | None = None
    target_evaluated: float | None = None
    policy_version: str | None = None
    kline_interval: str | None = None
    window_hours: int = 0
    reason: str = ""

    def append_kwargs(self) -> dict[str, Any]:
        """转成 ``FileOutcomeStore.append(**kwargs)`` 的入参(resolved_at 由 store 补当前时刻)。"""
        return {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "market": self.market,
            "outcome": self.outcome,
            "direction": self.direction,
            "decision_type": self.decision_type,
            "confidence": self.confidence,
            "pre_policy_confidence": self.pre_policy_confidence,
            "model_confidence": self.model_confidence,
            "entry": self.entry,
            "stop": self.stop,
            "targets": self.targets,
            "r_realized": self.r_realized,
            "entered_at": self.entered_at,
            "evaluation_started_at": self.evaluation_started_at,
            "evaluation_ended_at": self.evaluation_ended_at,
            "entry_fill": self.entry_fill,
            "target_evaluated": self.target_evaluated,
            "policy_version": self.policy_version,
            "kline_interval": self.kline_interval,
            "window_hours": self.window_hours,
            "reason": self.reason,
        }


def normalize_klines(raw: Any) -> list[dict[str, float | int]]:
    """把币安 ``/klines`` 原始数组 [open_time, open, high, low, close, volume, close_time, ...]
    归一化为 ``{open_time, high, low, close, close_time}`` 字典列表;异常行跳过。
    """
    out: list[dict[str, float | int]] = []
    for row in raw or []:
        try:
            out.append(
                {
                    "open_time": int(row[0]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "close_time": int(row[6]),
                }
            )
        except (TypeError, ValueError, IndexError):
            continue
    return out


def evaluation_window_ms(completed: datetime, window_hours: int) -> tuple[int, int]:
    """返回严格位于建议完成之后的毫秒窗口。"""
    end = completed + timedelta(hours=window_hours)
    return ceil(completed.timestamp() * 1000), floor(end.timestamp() * 1000)


def tag_run(
    run: dict[str, Any],
    klines: list[dict[str, Any]],
    *,
    window_hours: int,
    kline_interval: str | None = None,
) -> OutcomeRecord:
    """对一条 run 建议按回拉 K 线回溯裁决,产出 OutcomeRecord。

    判定顺序:not_applicable → unevaluable(价位缺失/无效)→ not_entered →
    出场裁决(同根 K 线止损与止盈同时触及时保守判 stop_hit)→ open。
    """
    final_advice = run.get("final_advice") or {}
    if not isinstance(final_advice, dict):
        final_advice = {}
    direction = final_advice.get("direction")
    decision_type = final_advice.get("decision_type")

    completed = _as_datetime(run.get("completed_at"))
    evaluation_end = completed + timedelta(hours=window_hours) if completed else None
    base = OutcomeRecord(
        run_id=str(run.get("run_id") or ""),
        symbol=str(run.get("symbol") or final_advice.get("symbol") or ""),
        market=run.get("market"),
        direction=direction,
        decision_type=decision_type,
        confidence=_as_float(final_advice.get("confidence")),
        pre_policy_confidence=_as_float(final_advice.get("pre_policy_confidence")),
        model_confidence=_as_float(final_advice.get("model_confidence")),
        outcome=NOT_APPLICABLE,
        kline_interval=kline_interval,
        window_hours=window_hours,
        evaluation_started_at=completed.isoformat() if completed else None,
        evaluation_ended_at=evaluation_end.isoformat() if evaluation_end else None,
        policy_version=OUTCOME_POLICY_VERSION,
    )

    # 1) 不适用:非 trade 决策或方向中性 → 无价位可评估
    if decision_type != "trade" or direction not in ("long", "short"):
        base.reason = f"决策类型 {decision_type!r}/方向 {direction!r} 无可评估交易价位"
        return base

    # 2) 解析价位;任一必需价位缺失/无效 → unevaluable
    entry_prices = extract_prices(final_advice.get("entry_zone"))
    stop_prices = extract_prices(final_advice.get("stop_loss"))
    target_nums = _target_numbers(final_advice.get("targets"))
    if not entry_prices:
        return _unevaluable(base, "缺入场区间或无法解析")
    if not stop_prices:
        return _unevaluable(base, "缺止损价或无法解析")
    if not target_nums:
        return _unevaluable(base, "缺目标价或无法解析")

    entry_low, entry_high = min(entry_prices), max(entry_prices)
    entry_ref = (entry_low + entry_high) / 2.0
    stop = stop_prices[0]

    # 顺势第一目标:long 取高于入场的最近目标,short 取低于入场的最近目标
    if direction == "long":
        valid = [t for t in target_nums if t > entry_ref]
        t1 = min(valid) if valid else None
    else:
        valid = [t for t in target_nums if t < entry_ref]
        t1 = max(valid) if valid else None
    if t1 is None:
        return _unevaluable(base, "无顺势方向的有效目标价")

    base.entry = entry_ref
    base.stop = stop
    base.targets = target_nums
    base.target_evaluated = t1

    start_ms, end_ms = evaluation_window_ms(completed, window_hours) if completed else (None, None)
    ordered = sorted(
        (
            k
            for k in klines
            if _as_float(k.get("high")) is not None
            and _is_within_evaluation_window(k, start_ms=start_ms, end_ms=end_ms)
        ),
        key=lambda k: int(k.get("open_time") or 0),
    )
    if not ordered:
        return _unevaluable(base, "回拉窗口内无有效 K 线数据")

    # 3) 入场判定:某根 K 线区间 [low, high] 与入场区间 [entry_low, entry_high] 相交
    entered_idx: int | None = None
    for idx, k in enumerate(ordered):
        high, low = float(k["high"]), float(k["low"])
        if low <= entry_high and high >= entry_low:
            entered_idx = idx
            break
    if entered_idx is None:
        base.outcome = NOT_ENTERED
        base.reason = "回拉窗口内价格未进入入场区间"
        return base

    entry_candle = ordered[entered_idx]
    entry_high_seen = float(entry_candle["high"])
    entry_low_seen = float(entry_candle["low"])
    intersection_low = max(entry_low_seen, entry_low)
    intersection_high = min(entry_high_seen, entry_high)
    entry_fill = intersection_high if direction == "long" else intersection_low
    base.entry_fill = entry_fill
    base.entered_at = _ms_to_iso(entry_candle.get("open_time"))

    if direction == "long":
        valid = [target for target in target_nums if target > entry_fill]
        t1 = min(valid) if valid else None
        risk = entry_fill - stop
    else:
        valid = [target for target in target_nums if target < entry_fill]
        t1 = max(valid) if valid else None
        risk = stop - entry_fill
    if t1 is None:
        return _unevaluable(base, "实际模拟入场价之后无顺势方向的有效目标价")
    if risk <= 0:
        return _unevaluable(base, "风险距离无效(实际模拟入场价与止损方向错误)")
    base.target_evaluated = t1

    # 4) 出场判定:自入场那根起,先触发者胜;同根同时触及 → 保守判 stop_hit
    for k in ordered[entered_idx:]:
        high, low = float(k["high"]), float(k["low"])
        if direction == "long":
            stop_trig, tgt_trig = low <= stop, high >= t1
        else:
            stop_trig, tgt_trig = high >= stop, low <= t1
        if stop_trig:  # 悲观假设:同根 K 线止损优先,不粉饰策略表现
            base.outcome = STOP_HIT
            base.r_realized = -1.0
            base.reason = "触发止损(同根同时触及止盈时保守判止损)" if tgt_trig else "触发止损"
            return base
        if tgt_trig:
            base.outcome = TARGET_HIT
            base.r_realized = round(abs(t1 - entry_fill) / risk, 4)
            base.reason = "触发第一目标止盈"
            return base

    # 5) 窗口内未触发任一:浮动盈亏(以末根 close 计)
    last_close = float(ordered[-1]["close"])
    signed = (last_close - entry_fill) if direction == "long" else (entry_fill - last_close)
    base.outcome = OPEN
    base.r_realized = round(signed / risk, 4)
    base.reason = "回拉窗口结束仍未触发止损或止盈,记浮动盈亏"
    return base


def _target_numbers(targets: Any) -> list[float]:
    """把 targets(list[str])逐条解析为数值(单条为区间时取均值),保留列表顺序。"""
    nums: list[float] = []
    for target in targets or []:
        prices = extract_prices(target)
        if prices:
            nums.append(sum(prices) / len(prices))
    return nums


def _unevaluable(base: OutcomeRecord, reason: str) -> OutcomeRecord:
    base.outcome = UNEVALUABLE
    base.reason = reason
    return base


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _ms_to_iso(ms: Any) -> str | None:
    value = _as_float(ms)
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).isoformat()


def _as_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _is_within_evaluation_window(
    kline: dict[str, Any], *, start_ms: int | None, end_ms: int | None
) -> bool:
    if start_ms is None or end_ms is None:
        return True
    open_time = _as_float(kline.get("open_time"))
    close_time = _as_float(kline.get("close_time"))
    return (
        open_time is not None
        and close_time is not None
        and open_time >= start_ms
        and close_time <= end_ms
    )
