"""效果闭环打标核心 tag_run 的确定性单测(纯函数,无 IO / 无时钟依赖)。

覆盖判定顺序的每个分支:not_applicable → unevaluable → not_entered →
出场裁决(target_hit / stop_hit,含同根保守判止损)→ open,以及 r_realized 计算。
"""

from __future__ import annotations

from typing import Any

from binance_agent.eval.tagging import (
    NOT_APPLICABLE,
    NOT_ENTERED,
    OPEN,
    STOP_HIT,
    TARGET_HIT,
    UNEVALUABLE,
    normalize_klines,
    tag_run,
)


def _run(**advice: Any) -> dict[str, Any]:
    """构造一条最小 run:trade + long 默认价位,可用 kwargs 覆盖 final_advice 字段。"""
    final_advice = {
        "symbol": "BTCUSDT",
        "direction": "long",
        "decision_type": "trade",
        "confidence": 0.7,
        "model_confidence": 0.8,
        "entry_zone": "100-102",
        "stop_loss": "95",
        "targets": ["110"],
    }
    final_advice.update(advice)
    return {
        "run_id": "run-x",
        "symbol": "BTCUSDT",
        "market": "fapi",
        "final_advice": final_advice,
    }


def _kline(
    open_time: int,
    high: float,
    low: float,
    close: float,
    *,
    close_time: int | None = None,
) -> dict[str, Any]:
    row = {"open_time": open_time, "high": high, "low": low, "close": close}
    if close_time is not None:
        row["close_time"] = close_time
    return row


def test_no_trade_is_not_applicable() -> None:
    record = tag_run(_run(decision_type="no_trade", direction="neutral"), [], window_hours=24)
    assert record.outcome == NOT_APPLICABLE
    assert record.entry is None


def test_trade_but_neutral_direction_is_not_applicable() -> None:
    record = tag_run(_run(direction="neutral"), [], window_hours=24)
    assert record.outcome == NOT_APPLICABLE


def test_missing_entry_is_unevaluable() -> None:
    record = tag_run(_run(entry_zone=None), [_kline(0, 103, 99, 101)], window_hours=24)
    assert record.outcome == UNEVALUABLE
    assert "入场" in record.reason


def test_missing_stop_is_unevaluable() -> None:
    record = tag_run(_run(stop_loss=""), [_kline(0, 103, 99, 101)], window_hours=24)
    assert record.outcome == UNEVALUABLE
    assert "止损" in record.reason


def test_missing_targets_is_unevaluable() -> None:
    record = tag_run(_run(targets=[]), [_kline(0, 103, 99, 101)], window_hours=24)
    assert record.outcome == UNEVALUABLE
    assert "目标" in record.reason


def test_long_target_without_valid_direction_is_unevaluable() -> None:
    # long 但唯一目标价低于入场参考价 → 无顺势有效目标
    record = tag_run(_run(targets=["90"]), [_kline(0, 103, 99, 101)], window_hours=24)
    assert record.outcome == UNEVALUABLE
    assert "顺势" in record.reason


def test_price_never_enters_zone_is_not_entered() -> None:
    # 全程价格远低于入场区间 [100,102]:high 均 < 100 → 从不相交
    klines = [_kline(0, 98, 90, 95), _kline(1, 97, 88, 92)]
    record = tag_run(_run(), klines, window_hours=24)
    assert record.outcome == NOT_ENTERED
    assert record.entry == 101.0  # 入场参考价仍记录(区间均值)


def test_long_target_hit() -> None:
    # 入场(k0 与 [100,102] 相交)→ k1 触及第一目标 110
    klines = [_kline(0, 103, 99, 101), _kline(1, 112, 105, 108)]
    record = tag_run(_run(), klines, window_hours=24)
    assert record.outcome == TARGET_HIT
    # long 按入场 K 线与区间交集的较高价 102 模拟成交
    assert record.r_realized == round((110 - 102) / (102 - 95), 4)
    assert record.entered_at is not None


def test_outcome_preserves_pre_policy_confidence_for_calibration() -> None:
    record = tag_run(
        _run(confidence=0.6, pre_policy_confidence=0.9),
        [_kline(0, 103, 99, 101), _kline(1, 112, 105, 108)],
        window_hours=24,
    )

    assert record.confidence == 0.6
    assert record.pre_policy_confidence == 0.9
    assert record.append_kwargs()["pre_policy_confidence"] == 0.9


def test_long_uses_higher_intersection_price_as_conservative_fill() -> None:
    klines = [_kline(0, 102.5, 101, 102), _kline(1, 112, 105, 108)]

    record = tag_run(_run(), klines, window_hours=24)

    assert record.outcome == TARGET_HIT
    assert record.entry == 101.0  # 保留原入场区间参考价以兼容历史数据
    assert record.entry_fill == 102.0
    assert record.r_realized == round((110 - 102) / (102 - 95), 4)


def test_long_stop_hit() -> None:
    klines = [_kline(0, 103, 99, 101), _kline(1, 104, 94, 96)]
    record = tag_run(_run(), klines, window_hours=24)
    assert record.outcome == STOP_HIT
    assert record.r_realized == -1.0


def test_same_candle_hits_both_conservatively_stop() -> None:
    # 入场后同一根 K 线同时触及止盈(high≥110)与止损(low≤95)→ 保守判止损
    klines = [_kline(0, 103, 99, 101), _kline(1, 112, 94, 100)]
    record = tag_run(_run(), klines, window_hours=24)
    assert record.outcome == STOP_HIT
    assert "同根" in record.reason


def test_evaluation_excludes_candle_that_started_before_advice_completed() -> None:
    run = _run()
    run["completed_at"] = "1970-01-01T00:00:01.500000+00:00"
    klines = [
        _kline(1000, 112, 94, 100, close_time=1999),
        _kline(2000, 112, 99, 108, close_time=2999),
    ]

    record = tag_run(run, klines, window_hours=1)

    assert record.outcome == TARGET_HIT
    assert record.evaluation_started_at == run["completed_at"]
    assert record.evaluation_ended_at == "1970-01-01T01:00:01.500000+00:00"


def test_evaluation_start_rounds_up_to_avoid_sub_millisecond_lookahead() -> None:
    run = _run()
    run["completed_at"] = "1970-01-01T00:00:01.500500+00:00"
    klines = [
        _kline(1500, 112, 94, 100, close_time=1999),
        _kline(1501, 112, 99, 108, close_time=2500),
    ]

    record = tag_run(run, klines, window_hours=1)

    assert record.outcome == TARGET_HIT


def test_evaluation_excludes_candle_not_closed_before_window_end() -> None:
    run = _run()
    run["completed_at"] = "1970-01-01T00:00:01.500000+00:00"
    klines = [
        _kline(2000, 103, 99, 101, close_time=2999),
        _kline(3_601_000, 112, 100, 108, close_time=3_601_999),
    ]

    record = tag_run(run, klines, window_hours=1)

    assert record.outcome == OPEN
    assert record.entry_fill == 102.0


def test_long_open_floating_pnl() -> None:
    # 入场后既不触止损也不触止盈,窗口结束按末根 close 记浮动盈亏
    klines = [_kline(0, 103, 99, 101), _kline(1, 108, 100, 105)]
    record = tag_run(_run(), klines, window_hours=24)
    assert record.outcome == OPEN
    # signed = 105-102 = 3; risk = 7 → R = 0.4286
    assert record.r_realized == 0.4286


def test_short_target_hit() -> None:
    # short:止损高于入场,目标低于入场
    run = _run(direction="short", stop_loss="108", targets=["90"])
    klines = [_kline(0, 103, 99, 101), _kline(1, 100, 88, 92)]
    record = tag_run(run, klines, window_hours=24)
    assert record.outcome == TARGET_HIT
    # short 按交集较低价 100 成交:risk=8,reward=10
    assert record.entry_fill == 100.0
    assert record.r_realized == 1.25


def test_short_stop_hit() -> None:
    run = _run(direction="short", stop_loss="108", targets=["90"])
    klines = [_kline(0, 103, 99, 101), _kline(1, 109, 104, 108)]
    record = tag_run(run, klines, window_hours=24)
    assert record.outcome == STOP_HIT
    assert record.r_realized == -1.0


def test_no_klines_is_unevaluable() -> None:
    record = tag_run(_run(), [], window_hours=24)
    assert record.outcome == UNEVALUABLE
    assert "K 线" in record.reason


def test_normalize_klines_skips_malformed_rows() -> None:
    raw = [
        [1000, "100", "103", "99", "101", "12.3", 1999],  # 正常
        [2000, "x"],  # 过短/坏值 → 跳过
        "not-a-row",  # 类型错误 → 跳过
        [3000, "105", "112", "104", "108", "5.0", 3999],  # 正常
    ]
    rows = normalize_klines(raw)
    assert len(rows) == 2
    assert rows[0] == {
        "open_time": 1000,
        "high": 103.0,
        "low": 99.0,
        "close": 101.0,
        "close_time": 1999,
    }
    assert rows[1]["open_time"] == 3000
