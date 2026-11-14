"""置信度校准分析(纯函数):让 by_confidence_bucket 反哺,量化"高置信是否真更准"。

confidence.py 的权重与 cap 阈值原是先验魔数;本模块用 outcome 台账把它变成可量化的校准信号:
- reliability:每个 confidence 桶的命中率、Wilson 95% 区间、gap 与单调性,
  以及只计决胜样本的 decisive Brier 和删失结果比例;
- suggest_caps:仅当充足样本的 Wilson 上界仍低于置信桶下界时给出 cap 建议;
- build_proposal/build_export_payload:返回可审计的纯数据,供 API/CLI 人工确认。

**只出建议,不自动改** confidence.py 阈值(人工审阅采纳,守住"确定性规则可审计"原则)。
决胜样本口径与 stats 一致:hit = target_hit / (target_hit + stop_hit)。

**已知限制**:当前反哺只落在 ``bucket_caps``(分桶上限)。confidence.py 里组合权重
(0.45/0.4/0.15)与各 cap 阈值(0.25~0.95)仍是先验魔数,本模块尚未对其产出调整建议;
这是后续可扩展方向,不在此自动改。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from math import isfinite
from typing import Any

from ..agents.confidence import apply_bucket_caps
from .stats import CONFIDENCE_BUCKETS, _bucket_label, _finite_number

_MIN_SAMPLES = 20  # 桶内决胜样本低于此值,统计噪声大,标 insufficient 不给建议


def time_split_calibration(
    records: list[dict[str, Any]],
    *,
    current_policy: dict[str, Any] | None = None,
    train_fraction: float = 0.7,
    min_samples: int = _MIN_SAMPLES,
) -> dict[str, Any]:
    """Fit only on earlier outcomes; compare policies on the same later records."""
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    timed = [(timestamp, record) for record in records if (timestamp := _record_time(record))]
    timed.sort(key=lambda pair: pair[0])
    target = int(len(timed) * train_fraction)
    boundaries = [i for i in range(1, len(timed)) if timed[i - 1][0] < timed[i][0]]
    # Never split equal maturity timestamps across fitting and validation.
    cut = min(boundaries, key=lambda i: (abs(i - target), -i)) if boundaries else len(timed)
    train = [record for _, record in timed[:cut]]
    validation = [record for _, record in timed[cut:]]
    policy = current_policy or {"policy_version": "confidence-policy/v1", "bucket_caps": {}}
    train_report = reliability(train, min_samples=min_samples)
    validation_report = reliability(validation, min_samples=min_samples)
    candidate = build_proposal(
        train,
        current_policy=policy,
        generated_at=timed[cut - 1][0].isoformat() if cut else "no-matured-outcomes",
        min_samples=min_samples,
    )
    proposed_caps = candidate["suggested_bucket_caps"]
    current_caps = dict(policy.get("bucket_caps") or {})
    candidate_caps = {**current_caps, **proposed_caps}
    comparison = _compare_policies(validation, current_caps, candidate_caps, proposed_caps)
    train_sufficient = any(not bucket["insufficient"] for bucket in train_report["buckets"])
    validation_sufficient = any(
        not bucket["insufficient"] for bucket in validation_report["buckets"]
    )
    status, reason = "inconclusive", "验证样本不足或没有严格早于验证段的训练记录"
    if not proposed_caps:
        reason = "训练段没有可下调的有效 cap"
    elif (
        train_sufficient
        and comparison["count"] >= min_samples
        and all(count >= min_samples for count in comparison["affected_bucket_counts"].values())
    ):
        delta = comparison["brier_delta"]
        if delta is not None and delta < -1e-8:
            status, reason = "validated", "候选在独立时间后段的 Brier 严格改善"
        elif delta is not None and delta > 1e-8:
            status, reason = "rejected", "候选在独立时间后段的 Brier 恶化"
        else:
            reason = "候选在验证段没有改善"
    split = {
        "rule_version": "maturity-policy-brier/v2",
        "train_fraction": train_fraction,
        "min_samples": min_samples,
        "train_end": timed[cut - 1][0].isoformat() if cut else None,
        "validation_start": timed[cut][0].isoformat() if cut < len(timed) else None,
    }
    proposal = None
    if status == "validated":
        identity = {
            "dataset_fingerprint": dataset_fingerprint(records),
            "current_policy": policy,
            "suggested_bucket_caps": proposed_caps,
            "split": split,
        }
        proposal = {
            **candidate,
            "proposal_id": hashlib.sha256(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode()
            ).hexdigest()[:24],
            "training_dataset_fingerprint": candidate["dataset_fingerprint"],
            "dataset_fingerprint": identity["dataset_fingerprint"],
            "validation_status": "validated",
            "applicable": True,
            "split": split,
            "validation": comparison,
        }
    return {
        "status": status,
        "reason": reason,
        "train_fraction": train_fraction,
        "min_samples": min_samples,
        "train_count": len(train),
        "validation_count": len(validation),
        "invalid_time_count": len(records) - len(timed),
        "validation_decisive": comparison["count"],
        "train_has_sufficient_bucket": train_sufficient,
        "validation_has_sufficient_bucket": validation_sufficient,
        "train": train_report,
        "validation": validation_report,
        "comparison": comparison,
        "split": split,
        "proposal": proposal,
        "diagnostic_proposal": candidate,
        "limitations": [
            "按成熟时间切分;同一成熟时刻不跨段;缺失或无效成熟时间不参与验证",
            "仅决胜样本;Brier 改善不证明统计显著、无条件盈利概率或交易收益",
        ],
    }


def _compare_policies(
    records: list[dict[str, Any]],
    current_caps: dict[str, float],
    candidate_caps: dict[str, float],
    proposed_caps: dict[str, float],
) -> dict[str, Any]:
    baseline, candidate = [], []
    affected = dict.fromkeys(proposed_caps, 0)
    sources = {"pre_policy": 0, "legacy_final": 0}
    excluded_legacy = 0
    for record in records:
        confidence = _calibration_confidence(record)
        bucket = _bucket_label(confidence)
        outcome = record.get("outcome")
        if bucket is None or confidence is None or outcome not in {"target_hit", "stop_hit"}:
            continue
        has_pre_policy = _bucket_label(record.get("pre_policy_confidence")) is not None
        if current_caps and not has_pre_policy:
            # A final legacy score cannot reconstruct its original pre-cap bucket.
            excluded_legacy += 1
            continue
        sources["pre_policy" if has_pre_policy else "legacy_final"] += 1
        hit = float(outcome == "target_hit")
        baseline.append((apply_bucket_caps(confidence, current_caps) - hit) ** 2)
        candidate.append((apply_bucket_caps(confidence, candidate_caps) - hit) ** 2)
        if bucket in affected:
            affected[bucket] += 1
    baseline_brier = sum(baseline) / len(baseline) if baseline else None
    candidate_brier = sum(candidate) / len(candidate) if candidate else None
    delta = (
        candidate_brier - baseline_brier
        if baseline_brier is not None and candidate_brier is not None
        else None
    )
    return {
        "count": len(baseline),
        "baseline_brier": round(baseline_brier, 8) if baseline_brier is not None else None,
        "candidate_brier": round(candidate_brier, 8) if candidate_brier is not None else None,
        "brier_delta": round(delta, 8) if delta is not None else None,
        "affected_bucket_counts": affected,
        "confidence_source_counts": sources,
        "excluded_legacy_with_existing_caps": excluded_legacy,
    }


def _record_time(record: dict[str, Any]) -> datetime | None:
    # Analysis completion/creation is not proof that a future outcome had matured.
    for key in ("resolved_at", "evaluation_ended_at"):
        raw = record.get(key)
        if raw is None:
            continue
        if not isinstance(raw, str) or not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    return None


def reliability(
    records: list[dict[str, Any]], *, min_samples: int = _MIN_SAMPLES
) -> dict[str, Any]:
    """按 confidence 桶产出可靠性诊断。

    每桶:样本数、决胜样本数、预测置信度均值、实际命中率、gap(=均值-命中率)。
    整体:Brier 分数(仅决胜样本)、单调性(命中率是否随置信度桶递增)。
    """
    buckets: list[dict[str, Any]] = []
    for label, lo, _hi in CONFIDENCE_BUCKETS:
        in_bucket = [r for r in records if _bucket_label(_calibration_confidence(r)) == label]
        decisive = [r for r in in_bucket if str(r.get("outcome")) in ("target_hit", "stop_hit")]
        n_decisive = len(decisive)
        hits = sum(1 for r in decisive if str(r.get("outcome")) == "target_hit")
        confidences = [
            confidence for r in decisive if (confidence := _calibration_confidence(r)) is not None
        ]
        wilson_95_ci = _wilson_interval(hits, n_decisive)
        if n_decisive < min_samples or n_decisive == 0:
            buckets.append(
                {
                    "bucket": label,
                    "lower": lo,
                    "count": len(in_bucket),
                    "decisive": n_decisive,
                    "hit_rate": round(hits / n_decisive, 4) if n_decisive else None,
                    "mean_confidence": (
                        round(sum(confidences) / len(confidences), 4) if confidences else None
                    ),
                    "gap": None,
                    "wilson_95_ci": wilson_95_ci,
                    "insufficient": True,
                }
            )
            continue
        hit_rate = hits / n_decisive
        mean_conf = sum(confidences) / len(confidences) if confidences else hit_rate
        buckets.append(
            {
                "bucket": label,
                "lower": lo,
                "count": len(in_bucket),
                "decisive": n_decisive,
                "hit_rate": round(hit_rate, 4),
                "mean_confidence": round(mean_conf, 4),
                "gap": round(mean_conf - hit_rate, 4),  # 正 = 过度自信(预测 > 实际)
                "wilson_95_ci": wilson_95_ci,
                "insufficient": False,
            }
        )

    decisive_brier = _brier(records)
    return {
        "buckets": buckets,
        "decisive_brier": decisive_brier,
        "brier": decisive_brier,  # 一个版本的读取兼容别名
        "censoring": _censoring(records),
        "monotonic": _monotonic([b for b in buckets if not b["insufficient"]]),
        "min_samples": min_samples,
    }


def suggest_caps(report: dict[str, Any]) -> dict[str, float]:
    """仅当 95% Wilson 上界仍低于桶下界时, 才给出保守 cap 建议。"""
    suggestions: dict[str, float] = {}
    for bucket in report.get("buckets") or []:
        if bucket.get("insufficient"):
            continue
        lower = bucket.get("lower")
        interval = bucket.get("wilson_95_ci") or {}
        upper = interval.get("high") if isinstance(interval, dict) else None
        if not _finite_number(upper) or not _finite_number(lower):
            continue
        assert isinstance(upper, (int, float)) and isinstance(lower, (int, float))
        if upper < lower:
            # 用统计上界而非点估计, 向上取整到 0.05; 仍须确实低于当前桶下界。
            cap = max(0.1, round(_ceil_step(upper, 0.05), 4))
            if cap >= lower:
                cap = round(max(0.1, lower - 0.05), 4)
            suggestions[str(bucket["bucket"])] = cap
    return suggestions


def dataset_fingerprint(records: list[dict[str, Any]]) -> str:
    """返回与顺序无关、只覆盖校准相关字段的数据集指纹。"""
    fields = (
        "outcome_id",
        "run_id",
        "outcome",
        "confidence",
        "pre_policy_confidence",
        "decision_type",
        "policy_version",
        "resolved_at",
        "evaluation_ended_at",
    )
    canonical_records = [{field: record.get(field) for field in fields} for record in records]
    canonical_records.sort(
        key=lambda record: json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )
    )
    encoded = json.dumps(
        canonical_records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_proposal(
    records: list[dict[str, Any]],
    *,
    current_policy: dict[str, Any],
    generated_at: str | datetime,
    min_samples: int = _MIN_SAMPLES,
) -> dict[str, Any]:
    """全样本诊断建议;未经时间后段验证,不得直接采纳。"""
    report = reliability(records, min_samples=min_samples)
    return _build_proposal(
        records,
        report=report,
        current_policy=current_policy,
        generated_at=generated_at,
    )


def build_export_payload(
    records: list[dict[str, Any]],
    *,
    current_policy: dict[str, Any],
    generated_at: str | datetime,
    min_samples: int = _MIN_SAMPLES,
) -> dict[str, Any]:
    """返回 CLI/API 可直接 JSON 序列化的校准报告与 proposal。"""
    report = reliability(records, min_samples=min_samples)
    time_split = time_split_calibration(
        records,
        current_policy=current_policy,
        min_samples=min_samples,
    )
    return {
        "schema_version": 2,
        "report": report,
        "time_split": time_split,
        "proposal": time_split["proposal"],
        "diagnostic_proposal": _build_proposal(
            records,
            report=report,
            current_policy=current_policy,
            generated_at=generated_at,
        ),
    }


def _build_proposal(
    records: list[dict[str, Any]],
    *,
    report: dict[str, Any],
    current_policy: dict[str, Any],
    generated_at: str | datetime,
) -> dict[str, Any]:
    generated = generated_at.isoformat() if isinstance(generated_at, datetime) else generated_at
    if not isinstance(generated, str) or not generated:
        raise ValueError("generated_at 必须是非空 ISO 时间字符串或 datetime")
    policy = json.loads(json.dumps(current_policy, ensure_ascii=False, default=str))
    caps = suggest_caps(report)
    current_caps = policy.get("bucket_caps") if isinstance(policy, dict) else None
    if isinstance(current_caps, dict):
        caps = {
            label: cap
            for label, cap in caps.items()
            if not _finite_number(current_caps.get(label)) or cap < float(current_caps[label])
        }
    fingerprint = dataset_fingerprint(records)
    counts = Counter(str(record.get("outcome") or "") for record in records)
    sample_counts = {
        "total": len(records),
        "decisive": counts.get("target_hit", 0) + counts.get("stop_hit", 0),
        "target_hit": counts.get("target_hit", 0),
        "stop_hit": counts.get("stop_hit", 0),
        "not_entered": counts.get("not_entered", 0),
        "open": counts.get("open", 0),
        "unevaluable": counts.get("unevaluable", 0),
        "not_applicable": counts.get("not_applicable", 0),
    }
    identity = json.dumps(
        {
            "dataset_fingerprint": fingerprint,
            "current_policy": policy,
            "suggested_bucket_caps": caps,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {
        "proposal_id": hashlib.sha256(identity).hexdigest()[:24],
        "generated_at": generated,
        "dataset_fingerprint": fingerprint,
        "sample_counts": sample_counts,
        "current_policy": policy,
        "suggested_bucket_caps": caps,
        "statistics": report,
        "validation_status": "unvalidated",
        "applicable": False,
    }


def format_reliability(report: dict[str, Any], caps: dict[str, float]) -> str:
    """人类可读的一段式摘要,供 CLI 打印。"""
    lines = [
        f"可靠性诊断(decisive Brier {_num(report.get('decisive_brier'))}, "
        f"单调性 {'是' if report.get('monotonic') else '否'}):"
    ]
    censoring = report.get("censoring") or {}
    lines.append(
        f"  删失:未入场 {_pct(censoring.get('not_entered_ratio'))}, "
        f"持有中 {_pct(censoring.get('open_ratio'))}, "
        f"无法评估 {_pct(censoring.get('unevaluable_ratio'))}"
    )
    for bucket in report.get("buckets") or []:
        if bucket.get("insufficient"):
            lines.append(
                f"  {bucket['bucket']}: 样本不足(决胜 {bucket['decisive']} < "
                f"{report.get('min_samples')}),暂不校准"
            )
        else:
            lines.append(
                f"  {bucket['bucket']}: 预测均值 {_pct(bucket['mean_confidence'])} vs "
                f"实际命中 {_pct(bucket['hit_rate'])}(gap {_signed(bucket['gap'])})"
            )
    if caps:
        pairs = ", ".join(f"{k}→cap {v:.2f}" for k, v in caps.items())
        lines.append(f"  建议下调 cap:{pairs}(仅建议,需人工审阅采纳)")
    else:
        lines.append("  无 cap 下调建议")
    return "\n".join(lines)


def _brier(records: list[dict[str, Any]]) -> float | None:
    """Brier = mean((confidence - hit)^2);hit: target_hit=1 / stop_hit=0,仅决胜样本。"""
    total = 0.0
    n = 0
    for record in records:
        outcome = str(record.get("outcome"))
        if outcome not in ("target_hit", "stop_hit"):
            continue
        conf = _calibration_confidence(record)
        if conf is None:
            continue
        hit = 1.0 if outcome == "target_hit" else 0.0
        total += (conf - hit) ** 2
        n += 1
    return round(total / n, 4) if n else None


def _censoring(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record.get("outcome") or "") for record in records)
    total = len(records)
    decisive = counts.get("target_hit", 0) + counts.get("stop_hit", 0)
    not_entered = counts.get("not_entered", 0)
    open_count = counts.get("open", 0)
    unevaluable = counts.get("unevaluable", 0)
    censored = not_entered + open_count + unevaluable
    return {
        "total": total,
        "decisive": decisive,
        "decisive_ratio": round(decisive / total, 4) if total else None,
        "not_entered": not_entered,
        "not_entered_ratio": round(not_entered / total, 4) if total else None,
        "open": open_count,
        "open_ratio": round(open_count / total, 4) if total else None,
        "unevaluable": unevaluable,
        "unevaluable_ratio": round(unevaluable / total, 4) if total else None,
        "not_applicable": counts.get("not_applicable", 0),
        "censored": censored,
        "censored_ratio": round(censored / total, 4) if total else None,
    }


def _wilson_interval(hits: int, total: int) -> dict[str, float | None]:
    if total <= 0:
        return {"low": None, "high": None}
    z = 1.959963984540054
    rate = hits / total
    denominator = 1.0 + (z * z) / total
    centre = rate + (z * z) / (2.0 * total)
    margin = z * ((rate * (1.0 - rate) / total + (z * z) / (4.0 * total * total)) ** 0.5)
    return {
        "low": round((centre - margin) / denominator, 4),
        "high": round((centre + margin) / denominator, 4),
    }


def _calibration_confidence(record: dict[str, Any]) -> float | None:
    """优先使用策略应用前置信度;历史记录缺失时回退到最终 confidence。"""
    for key in ("pre_policy_confidence", "confidence"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        confidence = float(value)
        if isfinite(confidence):
            return confidence
    return None


def _monotonic(buckets: list[dict[str, Any]]) -> bool | None:
    """命中率是否随置信度桶(按下界升序)非递减。少于 2 个有效桶 → None(无法判定)。"""
    rates = [b["hit_rate"] for b in sorted(buckets, key=lambda b: b["lower"])]
    if len(rates) < 2:
        return None
    return all(a <= b + 1e-9 for a, b in pairwise(rates))


def _ceil_step(value: float, step: float) -> float:
    import math

    return math.ceil(value / step) * step


def _pct(value: Any) -> str:
    return f"{float(value):.0%}" if _finite_number(value) else "N/A"


def _signed(value: Any) -> str:
    return f"{float(value):+.0%}" if _finite_number(value) else "N/A"


def _num(value: Any) -> str:
    return f"{float(value):.4f}" if _finite_number(value) else "N/A"
