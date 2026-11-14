"""效果闭环统计聚合 stats 的纯函数单测:命中率、平均 R、confidence 分桶。"""

from __future__ import annotations

from typing import Any

from binance_agent.eval.stats import aggregate, format_summary


def _rec(
    outcome: str, *, confidence: float, r: float | None, decision_type: str = "trade"
) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "confidence": confidence,
        "r_realized": r,
        "decision_type": decision_type,
    }


def test_hit_rate_counts_only_decisive_outcomes() -> None:
    records = [
        _rec("target_hit", confidence=0.9, r=1.5),
        _rec("target_hit", confidence=0.85, r=2.0),
        _rec("stop_hit", confidence=0.7, r=-1.0),
        _rec("not_entered", confidence=0.6, r=None),
        _rec("open", confidence=0.5, r=0.3),
    ]
    stats = aggregate(records)
    overall = stats["overall"]
    assert overall["count"] == 5
    assert overall["target_hit"] == 2
    assert overall["stop_hit"] == 1
    # 命中率仅在决胜样本(止盈+止损=3)上计:2/3
    assert overall["hit_rate"] == round(2 / 3, 4)
    # 平均 R 仅统计有数值的(1.5, 2.0, -1.0, 0.3),忽略 not_entered 的 None
    assert overall["avg_r"] == round((1.5 + 2.0 - 1.0 + 0.3) / 4, 4)


def test_confidence_buckets_partition_records() -> None:
    records = [
        _rec("target_hit", confidence=0.9, r=1.5),  # [0.8,1.0]
        _rec("stop_hit", confidence=0.82, r=-1.0),  # [0.8,1.0]
        _rec("target_hit", confidence=0.5, r=1.2),  # [0.4,0.6)
    ]
    buckets = aggregate(records)["by_confidence_bucket"]
    assert set(buckets) == {"[0.8,1.0]", "[0.4,0.6)"}
    assert buckets["[0.8,1.0]"]["count"] == 2
    assert buckets["[0.8,1.0]"]["hit_rate"] == 0.5  # 1 止盈 / (1+1)
    assert buckets["[0.4,0.6)"]["count"] == 1
    assert buckets["[0.4,0.6)"]["hit_rate"] == 1.0


def test_by_decision_type_split() -> None:
    records = [
        _rec("target_hit", confidence=0.9, r=1.5, decision_type="trade"),
        _rec("not_applicable", confidence=0.2, r=None, decision_type="no_trade"),
    ]
    by_dt = aggregate(records)["by_decision_type"]
    assert set(by_dt) == {"trade", "no_trade"}
    assert by_dt["trade"]["target_hit"] == 1
    assert by_dt["no_trade"]["not_applicable"] == 1


def test_empty_records_have_null_rates() -> None:
    stats = aggregate([])
    assert stats["total"] == 0
    assert stats["overall"]["hit_rate"] is None
    assert stats["overall"]["avg_r"] is None
    assert stats["by_confidence_bucket"] == {}


def test_no_decisive_outcomes_hit_rate_none() -> None:
    records = [
        _rec("not_entered", confidence=0.5, r=None),
        _rec("open", confidence=0.6, r=0.2),
    ]
    assert aggregate(records)["overall"]["hit_rate"] is None


def test_format_summary_renders_without_error() -> None:
    records = [
        _rec("target_hit", confidence=0.9, r=1.5),
        _rec("stop_hit", confidence=0.85, r=-1.0),
    ]
    text = format_summary(aggregate(records))
    assert "命中率" in text
    assert "分桶" in text  # 有决胜样本时输出分桶行


def test_format_summary_handles_empty() -> None:
    text = format_summary(aggregate([]))
    assert "总计 0 条" in text
    assert "N/A" in text  # 无命中率时占位
