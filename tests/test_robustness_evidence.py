"""Generated analyst prose and metadata cannot manufacture independent evidence."""

from __future__ import annotations

from typing import Any

from binance_agent.agents.confidence import calibrate_confidence
from binance_agent.agents.referee import ground_debate, new_verified_sources_in_round
from binance_agent.agents.state import AnalystReport, DebateClaim, DebateTurn, StrategyAdvice
from binance_agent.agents.validation import DataQualityReport, RiskGateResult
from binance_agent.infra.settings import ConfidenceConfig


def _report(*paths: str) -> dict[str, Any]:
    reports = {
        f"a{i}": {
            "stance": "bullish",
            "confidence": 0.9,
            "summary": f"claim{i}",
            "key_points": [f"point{i}"],
        }
        for i in range(10)
    }
    snapshot = {
        "fetched_at": "2026-10-18",
        "enabled": True,
        "errors": ["warning"],
        "ticker_24h": {"lastPrice": "100"},
        "funding": {"last_funding_rate": 0.001},
        "open_interest": 7,
        "open_interest_hist_tail": [{"sumOpenInterest": "7"}],
        "long_short_ratio_tail": [{"long_short_ratio": 1.2}],
        "technicals": {"1h": {"rsi14": 65}},
    }
    namespace = {"analyst_reports": reports, "snapshot_data": snapshot}
    claims = []
    for path in paths:
        current = namespace
        for key in path.split("."):
            current = current[int(key)] if isinstance(current, list) else current[key]
        claims.append(
            DebateClaim(
                statement="matched", evidence="literal", source_path=path, source_value=current
            )
        )
    turns = [DebateTurn(side="bull", round=0, argument="argument", structured_claims=claims)]
    result = ground_debate(turns, namespace)
    assert (
        new_verified_sources_in_round(turns, namespace, 0)
        == result["rounds"][0]["new_verified_sources"]
    )
    return result


def _confidence(report: dict[str, Any]):
    return calibrate_confidence(
        advice=StrategyAdvice(
            symbol="BTCUSDT", direction="long", decision_type="trade", confidence=0.9
        ),
        data_quality=DataQualityReport(),
        risk_gate=RiskGateResult(),
        analyst_reports=[
            AnalystReport(analyst=f"a{i}", stance="bullish", confidence=0.9) for i in range(3)
        ],
        runtime_events=[{"agent": "strategy_synth", "ok": True}],
        risk_assessment={"level": "low"},
        debate_grounding=report,
    )


def test_narrative_padding_is_invalid_and_cannot_lift_confidence_cap() -> None:
    report = _report(
        *(
            f"analyst_reports.a{i}.{field}"
            for i in range(10)
            for field in ("summary", "key_points.0")
        )
    )
    assert report["invalid_count"] == 20
    assert report["by_side"]["bull"]["verified_sources"] == 0
    assert report["rounds"][0]["new_verified_sources"] == 0
    assert _confidence(report).calibrated_confidence <= 0.6


def test_analyst_literal_matches_and_metadata_do_not_count_as_independent_support() -> None:
    report = _report(
        "analyst_reports.a0.stance",
        "analyst_reports.a0.confidence",
        "snapshot_data.fetched_at",
        "snapshot_data.enabled",
        "snapshot_data.errors.0",
    )
    assert report["verified_count"] == 5
    assert report["by_side"]["bull"]["sources"] == 0
    assert report["rounds"][0]["new_verified_sources"] == 0
    assert _confidence(report).calibrated_confidence <= 0.6


def test_real_market_groups_supply_independent_support() -> None:
    report = _report(
        "snapshot_data.ticker_24h.lastPrice",
        "snapshot_data.funding.last_funding_rate",
        "snapshot_data.open_interest",
        "snapshot_data.open_interest_hist_tail.0.sumOpenInterest",
        "snapshot_data.long_short_ratio_tail.0.long_short_ratio",
        "snapshot_data.technicals.1h.rsi14",
    )
    assert report["by_side"]["bull"]["verified_sources"] == 6
    assert report["rounds"][0]["new_verified_sources"] == 6
    result = _confidence(report)
    assert result.calibrated_confidence > 0.6
    assert result.policy_version == "confidence-policy/v2"
    assert ConfidenceConfig().policy_version == "confidence-policy/v2"
