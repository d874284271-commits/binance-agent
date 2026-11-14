"""置信度校准分析 reliability/suggest_caps 的纯函数单测。

用合成 outcome 样本验证:完美校准(高桶高命中)、反校准(高桶低命中)、Brier 手算对拍、
样本不足桶标 insufficient 不给建议。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from binance_agent.eval.calibration import (
    build_export_payload,
    build_proposal,
    dataset_fingerprint,
    format_reliability,
    reliability,
    suggest_caps,
    time_split_calibration,
)


def _recs(confidence: float, *, target: int, stop: int) -> list[dict[str, Any]]:
    """构造某置信度下 target_hit x target + stop_hit x stop 条决胜样本。"""
    return [{"outcome": "target_hit", "confidence": confidence} for _ in range(target)] + [
        {"outcome": "stop_hit", "confidence": confidence} for _ in range(stop)
    ]


def test_well_calibrated_is_monotonic_no_cap_suggestions() -> None:
    # 低桶低命中、高桶高命中 → 单调递增、gap 小、无 cap 建议
    records = (
        _recs(0.5, target=11, stop=11)  # [0.4,0.6): 命中 50%
        + _recs(0.7, target=15, stop=6)  # [0.6,0.8): 命中 ~71%
        + _recs(0.9, target=19, stop=2)  # [0.8,1.0]: 命中 ~90%
    )
    report = reliability(records, min_samples=20)

    assert report["monotonic"] is True
    active = {b["bucket"]: b for b in report["buckets"] if not b["insufficient"]}
    assert "[0.8,1.0]" in active
    assert active["[0.8,1.0]"]["hit_rate"] >= 0.85
    assert suggest_caps(report) == {}


def test_anti_calibrated_high_bucket_low_hit_suggests_cap() -> None:
    # 高置信桶命中率极低(过度自信)→ 非单调 + 对该桶给出下调建议
    records = (
        _recs(0.5, target=15, stop=6)  # [0.4,0.6): 命中 ~71%
        + _recs(0.9, target=4, stop=17)  # [0.8,1.0]: 命中 ~19% ≪ 下界 0.8
    )
    report = reliability(records, min_samples=20)

    assert report["monotonic"] is False
    caps = suggest_caps(report)
    assert "[0.8,1.0]" in caps
    assert caps["[0.8,1.0]"] <= 0.6  # cap 下调到命中率附近


def test_cap_requires_wilson_upper_bound_to_confirm_overconfidence() -> None:
    # 13/20 的点估计仅 65%,但 95% Wilson 上界仍高于高置信桶下界 80%。
    report = reliability(_recs(0.9, target=13, stop=7), min_samples=20)

    assert suggest_caps(report) == {}


def test_brier_matches_hand_computation() -> None:
    # 2 条 target_hit(conf 0.8)+ 2 条 stop_hit(conf 0.8)
    # Brier = mean[(0.8-1)^2, (0.8-1)^2, (0.8-0)^2, (0.8-0)^2]
    #       = mean[0.04, 0.04, 0.64, 0.64] = 0.34
    records = _recs(0.8, target=2, stop=2)
    report = reliability(records, min_samples=1)
    assert report["decisive_brier"] == 0.34


def test_calibration_uses_pre_policy_confidence_before_applied_caps() -> None:
    report = reliability(
        [
            {
                "outcome": "target_hit",
                "confidence": 0.2,
                "pre_policy_confidence": 0.9,
            }
        ],
        min_samples=1,
    )

    high = next(bucket for bucket in report["buckets"] if bucket["bucket"] == "[0.8,1.0]")
    assert high["decisive"] == 1
    assert high["mean_confidence"] == 0.9
    assert report["decisive_brier"] == 0.01


def test_historical_outcomes_without_pre_policy_confidence_fall_back_to_confidence() -> None:
    report = reliability(
        [{"outcome": "target_hit", "confidence": 0.9}],
        min_samples=1,
    )

    high = next(bucket for bucket in report["buckets"] if bucket["bucket"] == "[0.8,1.0]")
    assert high["decisive"] == 1
    assert report["decisive_brier"] == 0.01


def test_dataset_fingerprint_changes_when_pre_policy_confidence_changes() -> None:
    first = [
        {
            "outcome_id": "outcome-1",
            "run_id": "run-1",
            "outcome": "target_hit",
            "confidence": 0.6,
            "pre_policy_confidence": 0.8,
        }
    ]
    second = [{**first[0], "pre_policy_confidence": 0.9}]

    assert dataset_fingerprint(first) != dataset_fingerprint(second)


def test_time_split_calibration_keeps_late_outcomes_for_validation() -> None:
    records = [
        {
            "outcome": "target_hit" if index % 2 == 0 else "stop_hit",
            "confidence": 0.7,
            "resolved_at": f"2026-01-{index + 1:02d}T00:00:00+00:00",
        }
        for index in range(10)
    ]
    report = time_split_calibration(
        records,
        current_policy={"policy_version": "test", "bucket_caps": {}},
        train_fraction=0.7,
        min_samples=20,
    )
    assert report["train_count"] == 7
    assert report["validation_count"] == 3
    assert report["status"] == "inconclusive"
    assert report["proposal"] is None


def test_report_separates_censored_outcome_counts_and_ratios() -> None:
    records = [
        *_recs(0.8, target=1, stop=1),
        {"outcome": "not_entered", "confidence": 0.8},
        {"outcome": "open", "confidence": 0.8},
        {"outcome": "unevaluable", "confidence": 0.8},
    ]

    censoring = reliability(records, min_samples=1)["censoring"]

    assert censoring["total"] == 5
    assert censoring["decisive"] == 2
    assert censoring["not_entered"] == 1
    assert censoring["open"] == 1
    assert censoring["unevaluable"] == 1
    assert censoring["censored"] == 3
    assert censoring["censored_ratio"] == pytest.approx(0.6)


def test_each_non_empty_bucket_reports_wilson_95_interval() -> None:
    report = reliability(_recs(0.9, target=10, stop=10), min_samples=20)
    high = next(bucket for bucket in report["buckets"] if bucket["bucket"] == "[0.8,1.0]")

    assert high["wilson_95_ci"]["low"] == pytest.approx(0.2993, abs=1e-4)
    assert high["wilson_95_ci"]["high"] == pytest.approx(0.7007, abs=1e-4)


def test_insufficient_bucket_flagged_and_no_suggestion() -> None:
    # 决胜样本少于 min_samples → insufficient,不参与单调性、不给建议
    records = _recs(0.9, target=1, stop=5)  # 6 条 < 20
    report = reliability(records, min_samples=20)

    high = next(b for b in report["buckets"] if b["bucket"] == "[0.8,1.0]")
    assert high["insufficient"] is True
    assert high["gap"] is None
    assert suggest_caps(report) == {}


def test_non_decisive_outcomes_excluded_from_brier() -> None:
    # not_entered/open 不是决胜样本,不进 Brier;全非决胜 → Brier None
    records = [
        {"outcome": "not_entered", "confidence": 0.9},
        {"outcome": "open", "confidence": 0.7},
    ]
    report = reliability(records)
    assert report["brier"] is None


def test_format_reliability_renders() -> None:
    records = _recs(0.9, target=4, stop=17) + _recs(0.5, target=15, stop=6)
    report = reliability(records, min_samples=20)
    text = format_reliability(report, suggest_caps(report))
    assert "Brier" in text
    assert "cap" in text
    assert "删失" in text


def test_proposal_and_export_payload_are_deterministic_auditable_data() -> None:
    records = _recs(0.9, target=4, stop=17)
    for index, record in enumerate(records):
        record.update({"run_id": f"run-{index}", "outcome_id": f"outcome-{index}"})
    current_policy = {"version": "confidence-v1", "bucket_caps": {}}
    generated_at = "2026-01-02T00:00:00+00:00"

    proposal = build_proposal(
        records,
        current_policy=current_policy,
        generated_at=generated_at,
    )
    regenerated = build_proposal(
        records,
        current_policy=current_policy,
        generated_at="2026-01-03T00:00:00+00:00",
    )
    payload = build_export_payload(
        records,
        current_policy=current_policy,
        generated_at=generated_at,
    )

    assert proposal["proposal_id"]
    assert regenerated["proposal_id"] == proposal["proposal_id"]
    assert proposal["generated_at"] == generated_at
    assert proposal["dataset_fingerprint"] == dataset_fingerprint(records)
    assert proposal["dataset_fingerprint"] == dataset_fingerprint(list(reversed(records)))
    assert proposal["sample_counts"] == {
        "total": 21,
        "decisive": 21,
        "target_hit": 4,
        "stop_hit": 17,
        "not_entered": 0,
        "open": 0,
        "unevaluable": 0,
        "not_applicable": 0,
    }
    assert proposal["current_policy"] == current_policy
    assert "[0.8,1.0]" in proposal["suggested_bucket_caps"]
    assert payload["schema_version"] == 2
    assert payload["proposal"] is None
    assert payload["diagnostic_proposal"] == proposal
    assert proposal["applicable"] is False
    assert payload["report"]["decisive_brier"] is not None
    json.dumps(payload)  # 可直接交给 CLI/API 导出, 无需自定义编码器


def test_proposal_never_suggests_raising_an_existing_bucket_cap() -> None:
    records = _recs(0.9, target=4, stop=17)

    proposal = build_proposal(
        records,
        current_policy={"bucket_caps": {"[0.8,1.0]": 0.3}},
        generated_at="2026-01-02T00:00:00+00:00",
    )

    assert proposal["suggested_bucket_caps"] == {}


def _timed_records(*, train_hits: int = 0, validation_hits: int = 0) -> list[dict[str, Any]]:
    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        {
            "outcome_id": f"cal-{i}",
            "outcome": "target_hit"
            if (i < train_hits or 70 <= i < 70 + validation_hits)
            else "stop_hit",
            "confidence": 0.9,
            "resolved_at": (start + timedelta(hours=i)).isoformat(),
        }
        for i in range(100)
    ]


def test_time_split_rejects_candidate_that_worsens_validation_brier() -> None:
    result = time_split_calibration(_timed_records(validation_hits=30))
    assert result["status"] == "rejected"
    assert result["proposal"] is None
    assert result["comparison"]["baseline_brier"] == 0.01
    assert result["comparison"]["candidate_brier"] == 0.81


def test_time_split_validates_actual_improvement_and_binds_full_dataset() -> None:
    records = _timed_records(train_hits=10, validation_hits=3)
    result = time_split_calibration(records)
    assert result["status"] == "validated"
    assert result["comparison"]["candidate_brier"] < result["comparison"]["baseline_brier"]
    assert result["proposal"]["dataset_fingerprint"] == dataset_fingerprint(records)
    assert result["proposal"]["validation_status"] == "validated"
    assert result["proposal"]["applicable"] is True


def test_time_split_keeps_equal_maturity_times_in_one_partition() -> None:
    records = _timed_records()
    for record in records:
        record["resolved_at"] = records[0]["resolved_at"]
    result = time_split_calibration(records)
    assert result["status"] == "inconclusive"
    assert result["validation_count"] == 0
    assert result["proposal"] is None


def test_time_split_missing_maturity_time_cannot_use_creation_time() -> None:
    records = _timed_records()
    for record in records:
        record["created_at"] = record.pop("resolved_at")
    result = time_split_calibration(records)
    assert result["status"] == "inconclusive"
    assert result["invalid_time_count"] == 100


def test_calibration_fingerprint_changes_when_maturity_time_changes() -> None:
    records = _timed_records()
    changed = [dict(record) for record in records]
    changed[-1]["resolved_at"] = "2026-01-01T00:00:00+00:00"
    assert dataset_fingerprint(records) != dataset_fingerprint(changed)


def test_validation_compares_existing_policy_instead_of_raw_confidence() -> None:
    records = _timed_records(validation_hits=3)
    for record in records:
        record["pre_policy_confidence"] = 0.9
        record["confidence"] = 0.3
    result = time_split_calibration(
        records,
        current_policy={"policy_version": "existing", "bucket_caps": {"[0.8,1.0]": 0.3}},
    )
    assert result["status"] == "validated"
    assert result["comparison"]["baseline_brier"] == 0.13
    assert result["comparison"]["candidate_brier"] == 0.09


def test_affected_bucket_requires_its_own_late_samples() -> None:
    records = _timed_records()
    for record in records[70:]:
        record["confidence"] = 0.5
    result = time_split_calibration(records)
    assert result["status"] == "inconclusive"
    assert result["comparison"]["affected_bucket_counts"]["[0.8,1.0]"] == 0


def test_validation_equal_score_is_not_evidence_of_improvement() -> None:
    result = time_split_calibration(_timed_records(validation_hits=15))
    assert result["status"] == "inconclusive"
    assert result["comparison"]["brier_delta"] == 0.0
    assert result["proposal"] is None


def test_legacy_final_score_is_not_capped_again_under_an_existing_policy() -> None:
    result = time_split_calibration(
        _timed_records(),
        current_policy={"bucket_caps": {"[0.8,1.0]": 0.3}},
    )
    assert result["status"] == "inconclusive"
    assert result["comparison"]["count"] == 0
    assert result["comparison"]["excluded_legacy_with_existing_caps"] == 30
