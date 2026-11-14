"""ConfidenceCalibrator:最终置信度由确定性规则校准。"""

from __future__ import annotations

from typing import Any, Literal

from binance_agent.agents.confidence import calibrate_confidence
from binance_agent.agents.state import AnalystReport, StrategyAdvice
from binance_agent.agents.validation import DataQualityReport, RiskGateResult


def _advice(**kwargs: Any) -> StrategyAdvice:
    data: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "direction": "long",
        "decision_type": "trade",
        "entry_zone": "100",
        "stop_loss": "95",
        "targets": ["115"],
        "confidence": 0.9,
    }
    data.update(kwargs)
    return StrategyAdvice(**data)


def _analysts(
    *stances: Literal["bullish", "bearish", "neutral"], confidence: float = 0.8
) -> list[AnalystReport]:
    return [
        AnalystReport(analyst=f"a{i}", stance=stance, confidence=confidence)
        for i, stance in enumerate(stances, start=1)
    ]


def _runtime_event(**kwargs: object) -> dict[str, object]:
    data: dict[str, object] = {
        "agent": "strategy_synth",
        "ok": True,
        "repaired": False,
        "attempts": 1,
        "errors": [],
    }
    data.update(kwargs)
    return data


def test_data_quality_trade_block_caps_confidence() -> None:
    result = calibrate_confidence(
        advice=_advice(confidence=0.95),
        data_quality=DataQualityReport(
            level="critical",
            score=0.2,
            trade_blocked=True,
            reasons=["关键行情字段缺失:ticker_24h.lastPrice"],
        ),
        risk_gate=RiskGateResult(),
        analyst_reports=_analysts("bullish", "bullish", "bullish", confidence=0.9),
        runtime_events=[_runtime_event()],
        risk_assessment={"level": "low"},
    )

    assert result.calibrated_confidence <= 0.25
    assert result.confidence_cap <= 0.25
    assert any("数据质量" in reason for reason in result.confidence_reasons)


def test_low_verified_ratio_caps_confidence() -> None:
    result = calibrate_confidence(
        advice=_advice(confidence=0.9),
        data_quality=DataQualityReport(level="ok", score=1.0),
        risk_gate=RiskGateResult(),
        analyst_reports=_analysts("bullish", "bullish", "bullish", confidence=0.85),
        runtime_events=[_runtime_event()],
        risk_assessment={"level": "low"},
        debate_grounding={"verified_ratio": 0.2, "invalid_count": 4, "total_claims": 5},
    )

    assert result.confidence_cap <= 0.6
    assert result.calibrated_confidence <= 0.6
    assert any("证据完整性" in reason for reason in result.confidence_reasons)


def test_zero_claims_is_zero_verified_ratio_and_caps_confidence() -> None:
    result = calibrate_confidence(
        advice=_advice(confidence=0.9),
        data_quality=DataQualityReport(level="ok", score=1.0),
        risk_gate=RiskGateResult(),
        analyst_reports=_analysts("bullish", "bullish", "bullish", confidence=0.85),
        runtime_events=[_runtime_event()],
        risk_assessment={"level": "low"},
        debate_grounding={"verified_ratio": 0.0, "total_claims": 0},
    )

    assert result.confidence_cap <= 0.6
    assert any("证据完整性 0%" in reason for reason in result.confidence_reasons)


def test_high_grounded_ratio_no_grounding_penalty() -> None:
    result = calibrate_confidence(
        advice=_advice(confidence=0.9),
        data_quality=DataQualityReport(level="ok", score=1.0),
        risk_gate=RiskGateResult(),
        analyst_reports=_analysts("bullish", "bullish", "bullish", confidence=0.85),
        runtime_events=[_runtime_event()],
        risk_assessment={"level": "low"},
        debate_grounding={"grounded_ratio": 0.8, "ungrounded_count": 1, "total_claims": 5},
    )

    assert not any("缺快照支撑" in reason for reason in result.confidence_reasons)


def test_analyst_disagreement_prevents_high_confidence_trade() -> None:
    result = calibrate_confidence(
        advice=_advice(confidence=0.95),
        data_quality=DataQualityReport(level="ok", score=1.0),
        risk_gate=RiskGateResult(),
        analyst_reports=_analysts("bullish", "bearish", "neutral", confidence=0.75),
        runtime_events=[_runtime_event()],
        risk_assessment={"level": "low"},
    )

    assert result.calibrated_confidence <= 0.55
    assert result.confidence_cap <= 0.55
    assert any("分析师分歧" in reason for reason in result.confidence_reasons)


def test_strategy_json_repair_lowers_confidence() -> None:
    repaired = calibrate_confidence(
        advice=_advice(confidence=0.9),
        data_quality=DataQualityReport(level="ok", score=1.0),
        risk_gate=RiskGateResult(),
        analyst_reports=_analysts("bullish", "bullish", "bullish", confidence=0.85),
        runtime_events=[_runtime_event(repaired=True)],
        risk_assessment={"level": "low"},
    )

    assert repaired.calibrated_confidence <= 0.65
    assert any("JSON repair" in reason for reason in repaired.confidence_reasons)


def test_strategy_fallback_caps_confidence_to_low_value() -> None:
    fallback = calibrate_confidence(
        advice=_advice(direction="neutral", decision_type="no_trade", confidence=0.9),
        data_quality=DataQualityReport(level="ok", score=1.0),
        risk_gate=RiskGateResult(),
        analyst_reports=_analysts("bullish", "bullish", "bullish", confidence=0.85),
        runtime_events=[_runtime_event(ok=False, error_category="timeout")],
        risk_assessment={"level": "low"},
    )

    assert fallback.calibrated_confidence <= 0.25
    assert any("fallback" in reason.lower() for reason in fallback.confidence_reasons)


def test_risk_gate_veto_no_trade_cannot_keep_high_confidence() -> None:
    result = calibrate_confidence(
        advice=_advice(direction="neutral", decision_type="no_trade", confidence=0.95),
        data_quality=DataQualityReport(level="ok", score=1.0),
        risk_gate=RiskGateResult(veto=True, reason="风险收益比不足"),
        analyst_reports=_analysts("bullish", "bullish", "bullish", confidence=0.9),
        runtime_events=[_runtime_event()],
        risk_assessment={"level": "low"},
    )

    assert result.calibrated_confidence <= 0.25
    assert result.confidence_cap <= 0.25
    assert any("RiskGate veto" in reason for reason in result.confidence_reasons)


def test_high_quality_consensus_allows_high_confidence() -> None:
    result = calibrate_confidence(
        advice=_advice(confidence=0.92),
        data_quality=DataQualityReport(level="ok", score=1.0),
        risk_gate=RiskGateResult(),
        analyst_reports=_analysts("bullish", "bullish", "bullish", confidence=0.9),
        runtime_events=[_runtime_event()],
        risk_assessment={"level": "low"},
    )

    assert result.calibrated_confidence >= 0.75
    assert result.confidence_cap >= 0.85
    assert any("分析师方向一致" in reason for reason in result.confidence_reasons)
    assert result.evidence_confidence is None


def test_manual_bucket_cap_applies_after_existing_rules() -> None:
    result = calibrate_confidence(
        advice=_advice(confidence=0.9, model_confidence=0.9),
        data_quality=DataQualityReport(level="ok", score=1.0),
        risk_gate=RiskGateResult(),
        analyst_reports=_analysts("bullish", "bullish", "bullish", confidence=0.9),
        runtime_events=[_runtime_event()],
        risk_assessment={"level": "low"},
        bucket_caps={"[0.8,1.0]": 0.72},
        policy_version="confidence-policy/test",
    )

    assert result.pre_policy_confidence == 0.9
    assert result.policy_bucket == "[0.8,1.0]"
    assert result.calibrated_confidence == 0.72
    assert result.confidence_cap == 0.72
    assert result.policy_version == "confidence-policy/test"
    assert any("人工校准策略" in reason for reason in result.confidence_reasons)


def _sourced(
    analyst: str, stance: Literal["bullish", "bearish", "neutral"], *sources: str
) -> AnalystReport:
    return AnalystReport(
        analyst=analyst, stance=stance, confidence=0.85, data_sources=list(sources)
    )


def _calibrate(**kwargs: Any) -> Any:
    data: dict[str, Any] = {
        "advice": _advice(confidence=0.9),
        "data_quality": DataQualityReport(level="ok", score=1.0),
        "risk_gate": RiskGateResult(),
        "analyst_reports": _analysts("bullish", "bullish", "bullish", confidence=0.85),
        "runtime_events": [_runtime_event()],
        "risk_assessment": {"level": "low"},
    }
    data.update(kwargs)
    return calibrate_confidence(**data)


def test_shared_source_analysts_count_as_one_vote() -> None:
    # funding 与 news 都读同一份资金费率:两个 bullish 只算 1 票,不构成方向一致
    result = _calibrate(
        analyst_reports=[
            _sourced("funding", "bullish", "market.funding", "market.open_interest"),
            _sourced("news_sentiment", "bullish", "market.funding", "intel.fear_greed"),
            _sourced("technical", "neutral", "market.klines"),
        ]
    )

    assert result.calibrated_confidence <= 0.6
    assert any("缺少分析师一致支持" in reason for reason in result.confidence_reasons)
    assert "分析师 funding/news_sentiment 共用数据源,方向共识按 1 票计" in (
        result.confidence_reasons
    )


def test_shared_sources_merge_transitively() -> None:
    # a 与 c 不直接相交,但都与 b 相交 → 三者同组
    result = _calibrate(
        analyst_reports=[
            _sourced("a", "bullish", "x"),
            _sourced("c", "bullish", "y"),
            _sourced("b", "bullish", "x", "y"),
        ]
    )

    assert "分析师 a/b/c 共用数据源,方向共识按 1 票计" in result.confidence_reasons
    assert result.confidence_cap <= 0.6


def test_disjoint_sources_keep_independent_votes() -> None:
    result = _calibrate(
        analyst_reports=[
            _sourced("technical", "bullish", "market.klines"),
            _sourced("funding", "bullish", "market.funding"),
            _sourced("news_sentiment", "bullish", "intel.fear_greed"),
        ]
    )

    assert not any("共用数据源" in reason for reason in result.confidence_reasons)
    assert any("方向一致" in reason for reason in result.confidence_reasons)
    assert result.calibrated_confidence > 0.6


def _by_side(bull: tuple[int, int], bear: tuple[int, int]) -> dict[str, Any]:
    def side(sources: int, verified: int) -> dict[str, Any]:
        return {
            "sources": sources,
            "verified_sources": verified,
            "source_verified_ratio": round(verified / sources, 4) if sources else 0.0,
        }

    return {"verified_ratio": 0.9, "by_side": {"bull": side(*bull), "bear": side(*bear)}}


def test_supporting_side_low_independent_evidence_caps_confidence() -> None:
    # 整体 verified_ratio 0.9 不再能掩盖支持方(多)只有 1/4 独立数据点可核验
    result = _calibrate(debate_grounding=_by_side(bull=(4, 1), bear=(1, 1)))

    assert result.confidence_cap <= 0.6
    assert "辩论支持方(多)独立证据完整性 25%,限制最终置信度" in result.confidence_reasons


def test_opposing_side_with_more_verified_evidence_caps_confidence() -> None:
    result = _calibrate(
        advice=_advice(direction="short", stop_loss="105", targets=["85"]),
        analyst_reports=_analysts("bearish", "bearish", "bearish", confidence=0.85),
        debate_grounding=_by_side(bull=(4, 4), bear=(2, 2)),
    )

    assert result.confidence_cap <= 0.55
    assert "辩论反方可核验独立证据多于支持方(4>2),限制最终置信度" in result.confidence_reasons
    assert not any("支持方(空)独立证据完整性" in r for r in result.confidence_reasons)


def test_strong_supporting_evidence_has_no_debate_penalty() -> None:
    result = _calibrate(debate_grounding=_by_side(bull=(3, 3), bear=(2, 1)))

    assert not any("辩论" in reason for reason in result.confidence_reasons)


def test_neutral_direction_uses_overall_ratio() -> None:
    grounding = _by_side(bull=(4, 0), bear=(4, 0))
    grounding["verified_ratio"] = 0.2
    result = _calibrate(
        advice=_advice(direction="neutral", decision_type="no_trade"),
        debate_grounding=grounding,
    )

    assert "辩论证据完整性 20%,限制最终置信度" in result.confidence_reasons
    assert not any("支持方" in reason for reason in result.confidence_reasons)
