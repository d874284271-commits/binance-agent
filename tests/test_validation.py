"""M8 轻量信号校验测试:方向与指标/共识背离时产出告警。"""

from __future__ import annotations

from typing import Any

from binance_agent.agents.state import AnalystReport, StrategyAdvice
from binance_agent.agents.validation import (
    _extract_prices,
    apply_quality_and_risk_gates,
    build_data_quality_report,
    validate_advice,
)


def _klines_high_rsi() -> dict[str, Any]:
    # 严格递增 → RSI ≈ 100(超买)
    rows = [{"high": i + 1, "low": i - 1, "close": i + 0.5} for i in range(1, 40)]
    return {"klines": {"1h": rows}}


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
        "klines": {"5m": rows, "1h": rows, "4h": rows, "1d": rows},
        "ticker_24h": {
            "lastPrice": "100",
            "priceChangePercent": "1.2",
            "highPrice": "105",
            "lowPrice": "95",
            "volume": "1000",
            "quoteVolume": "100000",
        },
        "funding": {"last_funding_rate": 0.0001, "mark_price": 100},
        "open_interest": 1000,
        "errors": [],
    }


def _intel_snapshot(**source_status: str) -> dict[str, Any]:
    return {
        "source_status": source_status or {"fear_greed": "ok", "reddit": "ok", "rss": "ok"},
        "sentiment_features": {},
        "news_items": [{"title": "BTC headline"}],
        "macro_events": [],
    }


def test_validate_overbought_long() -> None:
    advice = StrategyAdvice(symbol="BTCUSDT", direction="long", confidence=0.7)
    reports = [AnalystReport(analyst="technical", stance="bullish")]
    warns = validate_advice(advice, _klines_high_rsi(), reports)
    assert any("超买" in w for w in warns)


def test_validate_consensus_mismatch() -> None:
    advice = StrategyAdvice(symbol="BTCUSDT", direction="long", confidence=0.7)
    reports = [
        AnalystReport(analyst="a", stance="bearish"),
        AnalystReport(analyst="b", stance="bearish"),
    ]
    warns = validate_advice(advice, {"klines": {}}, reports)
    assert any("分歧" in w for w in warns)


def test_validate_low_confidence() -> None:
    advice = StrategyAdvice(symbol="BTCUSDT", direction="short", confidence=0.3)
    warns = validate_advice(advice, {"klines": {}}, [])
    assert any("置信度" in w for w in warns)


def test_validate_clean_neutral() -> None:
    advice = StrategyAdvice(symbol="BTCUSDT", direction="neutral", confidence=0.6)
    warns = validate_advice(advice, {"klines": {}}, [])
    assert warns == []


def test_data_quality_blocks_trade_when_critical_market_data_missing() -> None:
    report = build_data_quality_report(
        {"market": "fapi", "klines": {}, "ticker_24h": {}, "errors": ["klines[1h]: boom"]},
        _intel_snapshot(),
    )
    assert report.trade_blocked
    assert report.level == "critical"

    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        entry_zone="100-101",
        stop_loss="95",
        targets=["110"],
        confidence=0.8,
    )
    gated = apply_quality_and_risk_gates(advice, report, {}, [])
    assert gated.decision_type == "no_trade"
    assert gated.direction == "neutral"
    assert gated.entry_zone is None
    assert gated.stop_loss is None
    assert gated.targets == []
    assert gated.no_trade_reason
    assert gated.reentry_conditions


def test_data_quality_degrades_confidence_when_most_intel_sources_fail() -> None:
    report = build_data_quality_report(
        _complete_market_snapshot(),
        _intel_snapshot(
            fear_greed="error: timeout",
            reddit="error: 403",
            rss="ok",
            gdelt="error: timeout",
        ),
    )
    assert report.level == "degraded"
    assert report.source_success_rate == 0.25

    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        entry_zone="100",
        stop_loss="95",
        targets=["115"],
        confidence=0.75,
    )
    gated = apply_quality_and_risk_gates(advice, report, {}, [], _complete_market_snapshot())
    assert gated.decision_type == "trade"
    assert gated.confidence < 0.6
    assert any("资讯源" in w for w in gated.risk_warnings)


def test_data_quality_excludes_unavailable_optional_sources_from_failure_rate() -> None:
    report = build_data_quality_report(
        _complete_market_snapshot(),
        _intel_snapshot(
            fear_greed="ok",
            rss="ok",
            cryptopanic="缺少 key(CRYPTOPANIC_TOKEN)",
            finnhub="缺少 key(FINNHUB_KEY)",
            akshare_cn="akshare 未安装(uv sync --extra intel-cn)",
        ),
    )

    assert report.level == "ok"
    assert report.source_success_rate == 1.0
    assert report.failed_sources == []
    assert set(report.unavailable_sources) == {"cryptopanic", "finnhub", "akshare_cn"}


def test_data_quality_counts_blocked_and_rate_limited_sources_as_failures() -> None:
    report = build_data_quality_report(
        _complete_market_snapshot(),
        _intel_snapshot(
            fear_greed="ok",
            rss="ok",
            reddit="error: blocked_public_endpoint: HTTP 403",
            gdelt="rate_limited: HTTP 429",
            cryptopanic="缺少 key(CRYPTOPANIC_TOKEN)",
        ),
    )

    assert report.level == "degraded"
    assert report.source_success_rate == 0.5
    assert set(report.failed_sources) == {"reddit", "gdelt"}
    assert report.unavailable_sources == ["cryptopanic"]


def test_risk_gate_vetoes_long_stop_loss_above_entry() -> None:
    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        entry_zone="100-101",
        stop_loss="103",
        targets=["110"],
        confidence=0.7,
    )
    gated = apply_quality_and_risk_gates(
        advice,
        build_data_quality_report(_complete_market_snapshot(), _intel_snapshot()),
        {"warnings": []},
        [],
    )
    assert gated.decision_type == "no_trade"
    assert gated.direction == "neutral"
    assert "止损" in (gated.no_trade_reason or "")


def test_risk_gate_penalizes_insufficient_reward_to_risk() -> None:
    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        entry_zone="100",
        stop_loss="98",
        targets=["102"],
        confidence=0.75,
    )
    gated = apply_quality_and_risk_gates(
        advice,
        build_data_quality_report(_complete_market_snapshot(), _intel_snapshot()),
        {"warnings": []},
        [],
    )
    assert gated.decision_type in {"watchlist", "no_trade"}
    assert gated.confidence <= 0.45
    assert any("风险收益比" in w for w in gated.risk_warnings)


def test_risk_gate_parses_thousands_separators_in_price_strings() -> None:
    # lastPrice 校准到与价位同量级(100000),使新的一致性闸门不误伤,专测千分分隔解析。
    snapshot = _complete_market_snapshot()
    snapshot["ticker_24h"]["lastPrice"] = "100000"
    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        entry_zone="100,000-101,000",
        stop_loss="95,000",
        targets=["110,000"],
        confidence=0.75,
    )
    gated = apply_quality_and_risk_gates(
        advice,
        build_data_quality_report(snapshot, _intel_snapshot()),
        {"warnings": []},
        [
            AnalystReport(analyst="technical", stance="bullish", confidence=0.8),
            AnalystReport(analyst="funding", stance="bullish", confidence=0.7),
        ],
        snapshot,
    )
    assert gated.decision_type == "trade"
    assert gated.direction == "long"


def test_risk_gate_handles_unparseable_price_strings_without_crashing() -> None:
    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        entry_zone="现价附近",
        stop_loss="待确认",
        targets=["突破后看高"],
        confidence=0.7,
    )
    gated = apply_quality_and_risk_gates(
        advice,
        build_data_quality_report(_complete_market_snapshot(), _intel_snapshot()),
        {"warnings": []},
        [],
        _complete_market_snapshot(),
    )
    assert gated.decision_type == "watchlist"
    assert gated.confidence <= 0.35
    assert any("无法评估风险收益" in w for w in gated.risk_warnings)


def test_declared_no_trade_clears_prices_even_if_direction_is_trade() -> None:
    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        decision_type="no_trade",
        entry_zone="100",
        stop_loss="95",
        targets=["115"],
        confidence=0.6,
        no_trade_reason="等待数据确认",
        reentry_conditions=["站回关键均线"],
    )
    gated = apply_quality_and_risk_gates(
        advice,
        build_data_quality_report(_complete_market_snapshot(), _intel_snapshot()),
        {"warnings": []},
        [],
        _complete_market_snapshot(),
    )
    assert gated.decision_type == "no_trade"
    assert gated.direction == "neutral"
    assert gated.entry_zone is None
    assert gated.stop_loss is None
    assert gated.targets == []
    assert gated.no_trade_reason
    assert gated.reentry_conditions


def test_data_quality_block_takes_priority_over_declared_no_trade() -> None:
    report = build_data_quality_report(
        {"market": "fapi", "klines": {}, "ticker_24h": {}, "errors": ["ticker24h: boom"]},
        _intel_snapshot(),
    )
    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="neutral",
        decision_type="no_trade",
        confidence=0.9,
        no_trade_reason="模型选择观望",
        reentry_conditions=["等待突破"],
    )
    gated = apply_quality_and_risk_gates(advice, report, {}, [])
    assert gated.decision_type == "no_trade"
    assert gated.confidence <= 0.25
    assert "数据质量不足" in (gated.no_trade_reason or "")


def test_risk_gate_caps_high_confidence_when_analysts_disagree() -> None:
    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        entry_zone="100",
        stop_loss="95",
        targets=["115"],
        confidence=0.9,
    )
    reports = [
        AnalystReport(analyst="technical", stance="bullish", confidence=0.45),
        AnalystReport(analyst="funding", stance="bearish", confidence=0.45),
        AnalystReport(analyst="news_sentiment", stance="neutral", confidence=0.35),
    ]
    gated = apply_quality_and_risk_gates(
        advice,
        build_data_quality_report(_complete_market_snapshot(), _intel_snapshot()),
        {"warnings": []},
        reports,
    )
    assert gated.confidence <= 0.55
    assert any("分析师分歧" in w or "低置信" in w for w in gated.risk_warnings)


def test_kline_completeness_validates_recent_tail_not_oldest_head() -> None:
    corrupt = [{"high": None, "low": None, "close": None} for _ in range(30)]
    valid_rows: list[dict[str, Any]] = [
        {"high": i + 2, "low": i - 2, "close": i + 0.5} for i in range(1, 41)
    ]
    intervals = ("5m", "1h", "4h", "1d")

    # 头部(最旧)30 根损坏 + 尾部 40 根有效:指标只依赖近端数据,应判满分
    snapshot = _complete_market_snapshot()
    snapshot["klines"] = dict.fromkeys(intervals, corrupt + valid_rows)
    report = build_data_quality_report(snapshot, _intel_snapshot())
    assert all(v == 1.0 for v in report.kline_completeness.values())

    # 反向:尾部(最新)30 根损坏 → 完整度应归零,而非被旧数据掩盖
    snapshot["klines"] = dict.fromkeys(intervals, valid_rows + corrupt)
    report = build_data_quality_report(snapshot, _intel_snapshot())
    assert all(v == 0.0 for v in report.kline_completeness.values())


# ───────────────────────── 价格解析器 ─────────────────────────


def test_extract_prices_parses_clean_range_and_thousands() -> None:
    assert _extract_prices("100-102") == [100.0, 102.0]
    assert _extract_prices("100,000-101,000") == [100000.0, 101000.0]


def test_extract_prices_strips_atr_multiplier_markers() -> None:
    # ATR 倍数/文字不能被当成价位(这是价格解析器被污染的根因)。
    assert _extract_prices("止损=入场价减去1.5倍ATR,约54500") == [54500.0]
    assert _extract_prices("距现价2倍ATR处 54500") == [54500.0]


# ───────────────────────── 价位一致性闸门 ─────────────────────────


def test_gate_allows_prices_within_plausible_range() -> None:
    # lastPrice=100,entry 100/stop 95/target 115 均在 ±20% 内 → 不触发一致性警告,仍 trade。
    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        entry_zone="100",
        stop_loss="95",
        targets=["115"],
        confidence=0.75,
    )
    gated = apply_quality_and_risk_gates(
        advice,
        build_data_quality_report(_complete_market_snapshot(), _intel_snapshot()),
        {"warnings": []},
        [
            AnalystReport(analyst="technical", stance="bullish", confidence=0.8),
            AnalystReport(analyst="funding", stance="bullish", confidence=0.7),
        ],
        _complete_market_snapshot(),
    )
    assert gated.decision_type == "trade"
    assert not any("[风控]" in w and "偏差过大" in w for w in gated.risk_warnings)


def test_gate_soft_caps_when_prices_far_from_current_price() -> None:
    # lastPrice=100,但 stop=10 / target=500 明显偏离 → 软封顶 watchlist + cap。
    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        entry_zone="100",
        stop_loss="10",
        targets=["500"],
        confidence=0.8,
    )
    gated = apply_quality_and_risk_gates(
        advice,
        build_data_quality_report(_complete_market_snapshot(), _intel_snapshot()),
        {"warnings": []},
        [],
        _complete_market_snapshot(),
    )
    assert gated.decision_type == "watchlist"
    assert gated.confidence <= 0.35
    assert any("偏差过大" in w for w in gated.risk_warnings)


def test_gate_skips_when_no_price_anchor() -> None:
    # 无 lastPrice 锚点时一致性闸门不介入(不产生「偏差过大」警告)。
    advice = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        entry_zone="100",
        stop_loss="95",
        targets=["115"],
        confidence=0.75,
    )
    gated = apply_quality_and_risk_gates(
        advice,
        build_data_quality_report(_complete_market_snapshot(), _intel_snapshot()),
        {"warnings": []},
        [],
        {"klines": {"1h": []}, "ticker_24h": {}},
    )
    assert not any("偏差过大" in w for w in gated.risk_warnings)
