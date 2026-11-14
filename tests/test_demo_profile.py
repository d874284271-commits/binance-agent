from __future__ import annotations

import pytest

from binance_agent.agents.validation import build_data_quality_report
from binance_agent.demo import (
    DemoIntelAggregator,
    DemoMarketData,
    DemoResearchContent,
    DemoRestClient,
)


async def test_demo_btc_is_complete_without_refetch() -> None:
    data = DemoMarketData()
    market = await data.build_snapshot(DemoRestClient("fapi"), "BTCUSDT")
    intel = await DemoIntelAggregator().gather("BTCUSDT")

    report = build_data_quality_report(
        market.model_dump(mode="json"), intel.model_dump(mode="json")
    )
    assert report.level == "ok"
    assert report.trade_blocked is False
    assert data.calls_for("BTCUSDT") == 1


async def test_demo_research_content_is_local_body_evidence() -> None:
    document = await DemoResearchContent().fetch("demo://headline/BTCUSDT")
    assert document.status == "available"
    assert document.level == "body"
    assert "no material supply change" in document.text


async def test_demo_eth_recovers_on_one_market_refetch() -> None:
    data = DemoMarketData()
    client = DemoRestClient("fapi")
    initial = await data.build_snapshot(client, "ETHUSDT")
    recovered = await data.build_snapshot(client, "ETHUSDT")

    assert initial.klines == {}
    assert len(recovered.klines["1h"]) == 36
    assert data.calls_for("ETHUSDT") == 2


async def test_demo_sol_refetch_fails_deterministically() -> None:
    data = DemoMarketData()
    client = DemoRestClient("fapi")
    initial = await data.build_snapshot(client, "SOLUSDT")

    assert initial.klines == {}
    with pytest.raises(RuntimeError, match="scripted market refetch failure"):
        await data.build_snapshot(client, "SOLUSDT")
    assert data.calls_for("SOLUSDT") == 2
