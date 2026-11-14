"""效果闭环统计聚合(纯函数)。

从 outcome 台账聚合出命中率、平均 R 实现值,并**按 confidence 分桶**给出各桶命中率——
这一分桶正面回应「高 confidence 是否真的更准 / 置信度魔数有无依据」的核心观测问题。
"""

from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Any

# confidence 分桶:左闭右开,最后一桶右闭到 1.0
CONFIDENCE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("[0.0,0.4)", 0.0, 0.4),
    ("[0.4,0.6)", 0.4, 0.6),
    ("[0.6,0.8)", 0.6, 0.8),
    ("[0.8,1.0]", 0.8, 1.0001),
)


def _subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(r.get("outcome") or "") for r in records)
    target_hit = counts.get("target_hit", 0)
    stop_hit = counts.get("stop_hit", 0)
    decisive = target_hit + stop_hit
    r_values = [float(r["r_realized"]) for r in records if _finite_number(r.get("r_realized"))]
    return {
        "count": len(records),
        "target_hit": target_hit,
        "stop_hit": stop_hit,
        "not_entered": counts.get("not_entered", 0),
        "open": counts.get("open", 0),
        "not_applicable": counts.get("not_applicable", 0),
        "unevaluable": counts.get("unevaluable", 0),
        "hit_rate": round(target_hit / decisive, 4) if decisive else None,
        "avg_r": round(sum(r_values) / len(r_values), 4) if r_values else None,
    }


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value)


def _bucket_label(confidence: Any) -> str | None:
    if not _finite_number(confidence):
        return None
    value = float(confidence)
    for label, lo, hi in CONFIDENCE_BUCKETS:
        if lo <= value < hi:
            return label
    return None


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合 outcome 记录。返回可直接 JSON 序列化的统计视图。"""
    overall = _subset(records)

    decision_types = sorted(
        {str(r.get("decision_type") or "") for r in records if r.get("decision_type")}
    )
    by_decision_type = {
        dt: _subset([r for r in records if r.get("decision_type") == dt]) for dt in decision_types
    }

    by_confidence_bucket: dict[str, Any] = {}
    for label, _lo, _hi in CONFIDENCE_BUCKETS:
        bucket_records = [r for r in records if _bucket_label(r.get("confidence")) == label]
        if bucket_records:
            by_confidence_bucket[label] = _subset(bucket_records)

    return {
        "total": len(records),
        "overall": overall,
        "by_decision_type": by_decision_type,
        "by_confidence_bucket": by_confidence_bucket,
    }


def format_summary(stats: dict[str, Any]) -> str:
    """人类可读的一段式摘要,供 CLI 打印。"""
    overall = stats.get("overall") or {}
    lines = [
        f"总计 {stats.get('total', 0)} 条 outcome",
        (
            f"  命中率 {_pct(overall.get('hit_rate'))} "
            f"(止盈 {overall.get('target_hit', 0)} / 止损 {overall.get('stop_hit', 0)}), "
            f"平均 R {_num(overall.get('avg_r'))}, "
            f"未入场 {overall.get('not_entered', 0)}, 持有中 {overall.get('open', 0)}, "
            f"不适用 {overall.get('not_applicable', 0)}, 无法评估 {overall.get('unevaluable', 0)}"
        ),
    ]
    buckets = stats.get("by_confidence_bucket") or {}
    if buckets:
        lines.append("  按 confidence 分桶命中率(观测高置信是否更准):")
        for label, sub in buckets.items():
            lines.append(
                f"    {label}: 命中率 {_pct(sub.get('hit_rate'))} "
                f"(样本 {sub.get('count', 0)}, 决胜 {sub.get('target_hit', 0) + sub.get('stop_hit', 0)}, "
                f"平均 R {_num(sub.get('avg_r'))})"
            )
    return "\n".join(lines)


def _pct(value: Any) -> str:
    return f"{float(value):.0%}" if _finite_number(value) else "N/A"


def _num(value: Any) -> str:
    return f"{float(value):+.2f}" if _finite_number(value) else "N/A"
