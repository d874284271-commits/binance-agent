"""证据完整性 referee:只验证 canonical 标量引用与值,不宣称验证论点语义。"""

from __future__ import annotations

from typing import Any

from binance_agent.agents.referee import ground_debate, new_verified_sources_in_round
from binance_agent.agents.state import DebateClaim, DebateTurn


def _namespace() -> dict[str, Any]:
    return {
        "snapshot_data": {
            "technicals": {"1h": {"rsi14": 65.0}, "4h": {"rsi14": 72.0}},
            "funding": {"last_funding_rate": 0.0006},
            "ticker_24h": {"lastPrice": "100"},
            "open_interest_hist_tail": [{"sumOpenInterest": "1200"}],
        },
        "analyst_reports": {"technical": {"confidence": 0.7, "stance": "bullish"}},
    }


def _claim(
    statement: str,
    path: str | None,
    value: str | int | float | bool | None,
) -> DebateClaim:
    return DebateClaim(
        statement=statement,
        evidence="evidence",
        source_path=path,
        source_value=value,
        confidence=0.6,
    )


def _report(*claims: DebateClaim) -> dict[str, Any]:
    turn = DebateTurn(side="bull", round=0, argument="arg", structured_claims=list(claims))
    return ground_debate([turn], _namespace())


def test_ground_debate_verifies_structured_scalar_evidence() -> None:
    report = _report(
        _claim("1h RSI 偏强", "snapshot_data.technicals.1h.rsi14", 65),
        _claim("技术分析偏多", "analyst_reports.technical.stance", "bullish"),
        _claim("OI 为 1200", "snapshot_data.open_interest_hist_tail.0.sumOpenInterest", 1200),
    )

    assert report["verified_ratio"] == 1.0
    assert report["grounded_ratio"] == 1.0  # v0.1 读取别名
    assert report["verified_count"] == 3
    assert [claim["verdict"] for claim in report["claims"]] == ["verified"] * 3


def test_ground_debate_reports_missing_mismatch_and_invalid_separately() -> None:
    report = _report(
        _claim("不存在", "snapshot_data.technicals.1h.macd", 1),
        _claim("值错误", "snapshot_data.technicals.1h.rsi14", 30),
        _claim("容器引用", "snapshot_data.technicals.1h", 65),
        _claim("白名单外", "intel_snapshot.macro_events.0", "CPI"),
        _claim("空值", "snapshot_data.ticker_24h.lastPrice", None),
    )

    assert report["missing_count"] == 1
    assert report["mismatch_count"] == 1
    assert report["invalid_count"] == 3
    assert report["verified_count"] == 0
    assert report["verified_ratio"] == 0.0
    assert [claim["verdict"] for claim in report["claims"]] == [
        "missing",
        "mismatch",
        "invalid",
        "invalid",
        "invalid",
    ]


def test_numeric_comparison_rejects_even_small_rounding_difference() -> None:
    report = _report(
        _claim("费率", "snapshot_data.funding.last_funding_rate", 0.00061),
    )

    assert report["verified_count"] == 0
    assert report["mismatch_count"] == 1


def test_bool_is_not_equal_to_number_or_string() -> None:
    namespace = _namespace()
    namespace["snapshot_data"]["enabled"] = True
    turn = DebateTurn(
        side="bull",
        round=0,
        argument="arg",
        structured_claims=[
            _claim("数字不是布尔", "snapshot_data.enabled", 1),
            _claim("字符串不是布尔", "snapshot_data.enabled", "true"),
            _claim("布尔精确匹配", "snapshot_data.enabled", True),
        ],
    )

    report = ground_debate([turn], namespace)

    assert [claim["verdict"] for claim in report["claims"]] == [
        "mismatch",
        "mismatch",
        "verified",
    ]


def test_legacy_source_ref_is_read_into_structured_contract() -> None:
    claim = DebateClaim.model_validate(
        {
            "statement": "历史证据",
            "evidence": "RSI 65",
            "source_ref": "snapshot_data.technicals.1h.rsi14=65",
            "confidence": 0.6,
        }
    )

    report = _report(claim)

    assert claim.source_path == "snapshot_data.technicals.1h.rsi14"
    assert claim.source_value == 65
    assert "source_ref" not in claim.model_dump(mode="json")
    assert report["verified_count"] == 1


def test_ground_debate_empty_is_zero_verified_ratio() -> None:
    report = ground_debate([], _namespace())

    assert report["total_claims"] == 0
    assert report["verified_ratio"] == 0.0
    assert report["grounded_ratio"] == 0.0


def test_repeating_one_path_does_not_inflate_independent_evidence() -> None:
    # 同一路径刷 5 条:按 claim 计比例很高,按独立数据点计只算 1 个
    padded = [_claim(f"RSI 偏强 {i}", "snapshot_data.technicals.1h.rsi14", 65) for i in range(5)]
    report = _report(*padded, _claim("值错误", "snapshot_data.funding.last_funding_rate", 1))

    assert report["verified_ratio"] == round(5 / 6, 4)
    assert report["duplicate_claims"] == 4
    bull = report["by_side"]["bull"]
    assert (bull["sources"], bull["verified_sources"]) == (2, 1)
    assert bull["source_verified_ratio"] == 0.5


def test_same_side_contradictory_values_do_not_count_as_verified_source() -> None:
    report = _report(
        _claim("RSI 65", "snapshot_data.technicals.1h.rsi14", 65),
        _claim("RSI 30", "snapshot_data.technicals.1h.rsi14", 30),
    )

    bull = report["by_side"]["bull"]
    assert (bull["sources"], bull["verified_sources"]) == (1, 0)
    assert bull["source_verified_ratio"] == 0.0


def test_by_side_separates_evidence_and_reports_shared_sources() -> None:
    turns = [
        DebateTurn(
            side="bull",
            round=0,
            argument="bull",
            structured_claims=[
                _claim("RSI 偏强", "snapshot_data.technicals.1h.rsi14", 65),
                _claim("4h 超买但趋势强", "snapshot_data.technicals.4h.rsi14", 72),
            ],
        ),
        DebateTurn(
            side="bear",
            round=0,
            argument="bear",
            structured_claims=[
                _claim("4h 超买", "snapshot_data.technicals.4h.rsi14", 72),
                _claim("编造", "snapshot_data.nope", 1),
            ],
        ),
    ]

    report = ground_debate(turns, _namespace())

    assert report["by_side"]["bull"]["verified_sources"] == 2
    assert report["by_side"]["bear"]["verified_sources"] == 1
    # Unknown top-level snapshot paths are not independent market data units.
    assert report["by_side"]["bear"]["source_verified_ratio"] == 1.0
    # 双方都核验引用了 4h RSI:同一数据的不同解读,是核心分歧点
    assert report["shared_sources"] == ["snapshot_data.technicals.4h.rsi14"]


def test_empty_side_reports_zero_evidence() -> None:
    report = ground_debate([], _namespace())

    assert report["by_side"]["bull"] == {
        "total_claims": 0,
        "verified_count": 0,
        "sources": 0,
        "verified_sources": 0,
        "source_verified_ratio": 0.0,
    }
    assert report["rounds"] == []


def test_rounds_count_only_first_verified_citation_as_new_evidence() -> None:
    turns = [
        DebateTurn(
            side="bull",
            round=0,
            argument="r0",
            structured_claims=[_claim("RSI", "snapshot_data.technicals.1h.rsi14", 65)],
        ),
        DebateTurn(
            side="bear",
            round=1,
            argument="r1",
            structured_claims=[
                _claim("反驳 RSI", "snapshot_data.technicals.1h.rsi14", 65),
                _claim("编造新路径", "snapshot_data.nope", 1),
            ],
        ),
        DebateTurn(
            side="bull",
            round=2,
            argument="r2",
            structured_claims=[_claim("费率", "snapshot_data.funding.last_funding_rate", 0.0006)],
        ),
        DebateTurn(side="bear", round=3, argument="r3 无 claims"),
    ]

    report = ground_debate(turns, _namespace())

    assert [(r["round"], r["new_verified_sources"]) for r in report["rounds"]] == [
        (0, 1),
        (1, 0),
        (2, 1),
        (3, 0),
    ]
    assert new_verified_sources_in_round(turns, _namespace(), 1) == 0
    assert new_verified_sources_in_round(turns, _namespace(), 2) == 1
    assert new_verified_sources_in_round(turns, _namespace(), 9) == 0
