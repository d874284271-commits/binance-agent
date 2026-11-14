"""Typed deterministic tools for bounded agent runtime."""

from __future__ import annotations

from typing import Any

from binance_agent.agents.state import AnalystReport, StrategyAdvice
from binance_agent.agents.tools import build_default_tool_registry
from binance_agent.agents.validation import build_data_quality_report


async def test_default_tool_registry_exposes_typed_schemas() -> None:
    registry = build_default_tool_registry()

    assert set(registry.names()) == {
        "market_snapshot",
        "intel_snapshot",
        "technical_indicators",
        "data_quality_check",
        "risk_gate_check",
        "confidence_calibration",
    }
    market_tool = registry.get("market_snapshot")
    assert market_tool.input_model.model_fields["symbol"].is_required()
    assert market_tool.output_model.model_fields["market_snapshot"].is_required()
    assert market_tool.timeout_s is not None
    assert market_tool.timeout_s > 0
    assert registry.get("intel_snapshot").timeout_s is None


async def test_intel_snapshot_tool_keeps_partial_provider_successes() -> None:
    class _PartialAggregator:
        async def gather(self, symbol: str) -> Any:
            return {
                "symbol": symbol,
                "source_status": {
                    "rss": "ok",
                    "fear_greed": "ok",
                    "reddit": "error: blocked_public_endpoint: HTTP 403",
                    "gdelt": "rate_limited: HTTP 429",
                },
                "sentiment_features": {"fear_greed": {"value": 11}},
                "news_items": [{"title": "headline", "source": "rss"}],
                "macro_events": [],
            }

    result = await build_default_tool_registry().invoke(
        "intel_snapshot",
        {"symbol": "BTCUSDT"},
        context={"aggregator": _PartialAggregator()},
    )

    assert result.ok is True
    assert result.output is not None
    snapshot = result.output["intel_snapshot"]
    assert snapshot["source_status"]["rss"] == "ok"
    assert snapshot["source_status"]["reddit"].startswith("error:")
    assert snapshot["news_items"] == [{"title": "headline", "source": "rss"}]
    assert snapshot["sentiment_features"]["fear_greed"]["value"] == 11


async def test_technical_indicators_tool_returns_multi_interval_summary() -> None:
    registry = build_default_tool_registry()

    result = await registry.invoke(
        "technical_indicators",
        {"market_snapshot": _complete_market_snapshot()},
    )

    assert result.ok is True
    assert result.tool == "technical_indicators"
    assert result.output is not None
    assert set(result.output["indicators"]) == {"1h", "4h"}
    assert result.output["indicators"]["1h"]["atr14"] is not None
    assert result.errors == []


async def test_data_quality_and_risk_gate_tools_return_typed_outputs() -> None:
    registry = build_default_tool_registry()

    quality = await registry.invoke(
        "data_quality_check",
        {
            "market_snapshot": {"klines": {}, "ticker_24h": {}, "errors": ["ticker24h: boom"]},
            "intel_snapshot": _intel_snapshot(),
        },
    )
    assert quality.ok is True
    assert quality.output is not None
    assert quality.output["data_quality_report"]["trade_blocked"] is True

    risk = await registry.invoke(
        "risk_gate_check",
        {
            "advice": StrategyAdvice(
                symbol="BTCUSDT",
                direction="long",
                entry_zone="100",
                stop_loss="103",
                targets=["110"],
            ).model_dump(mode="json"),
            "market_snapshot": _complete_market_snapshot(),
            "risk_assessment": {"warnings": []},
            "analyst_reports": [
                AnalystReport(analyst="technical", stance="bullish").model_dump(mode="json")
            ],
        },
    )

    assert risk.ok is True
    assert risk.output is not None
    assert risk.output["risk_gate_result"]["veto"] is True
    assert "止损" in risk.output["risk_gate_result"]["reason"]


async def test_confidence_calibration_tool_caps_poor_data_quality() -> None:
    registry = build_default_tool_registry()
    data_quality = build_data_quality_report(
        {"klines": {}, "ticker_24h": {}, "errors": ["klines[1h]: boom"]},
        _intel_snapshot(),
    )

    result = await registry.invoke(
        "confidence_calibration",
        {
            "advice": StrategyAdvice(
                symbol="BTCUSDT",
                direction="long",
                decision_type="trade",
                confidence=0.9,
                model_confidence=0.9,
            ).model_dump(mode="json"),
            "data_quality_report": data_quality.model_dump(mode="json"),
            "risk_gate_result": {"veto": False, "warnings": []},
            "analyst_reports": [
                AnalystReport(analyst="technical", stance="bullish", confidence=0.9).model_dump(
                    mode="json"
                )
            ],
            "runtime_events": [],
            "risk_assessment": {"level": "low"},
        },
    )

    assert result.ok is True
    assert result.output is not None
    assert result.output["calibration"]["confidence_cap"] <= 0.25
    assert result.output["calibration"]["calibrated_confidence"] <= 0.25


async def test_tool_error_returns_safe_fallback_without_raising() -> None:
    registry = build_default_tool_registry()

    result = await registry.invoke(
        "market_snapshot",
        {"symbol": "BTCUSDT", "market": "fapi"},
        context={},
    )

    assert result.ok is False
    assert result.output is not None
    assert result.output["market_snapshot"]["symbol"] == "BTCUSDT"
    assert result.output["market_snapshot"]["errors"]
    assert result.errors[0]["category"] == "dependency"


def _complete_market_snapshot() -> dict[str, Any]:
    rows = [
        {
            "open": i,
            "high": i + 2,
            "low": i - 2,
            "close": i + 0.5,
            "volume": 10,
            "close_time": 1700000000000 + i * 60000,
        }
        for i in range(1, 50)
    ]
    return {
        "market": "fapi",
        "klines": {"1h": rows, "4h": rows},
        "ticker_24h": {
            "lastPrice": "100",
            "highPrice": "105",
            "lowPrice": "95",
            "volume": "1000",
            "quoteVolume": "100000",
        },
        "errors": [],
    }


def _intel_snapshot() -> dict[str, Any]:
    return {
        "source_status": {"rss": "ok", "reddit": "ok"},
        "sentiment_features": {},
        "news_items": [{"title": "headline"}],
        "macro_events": [],
    }
