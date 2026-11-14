"""GDELT provider 离线测试:429/非200 显式降级(Bug3 回归)+ 正常 200 解析(respx mock)。"""

from __future__ import annotations

import httpx
import respx

from binance_agent.intel.base import FetchContext
from binance_agent.intel.providers.gdelt import GdeltProvider

_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"


def _ctx() -> FetchContext:
    return FetchContext(symbol="BTCUSDT", base_asset="BTC", window_hours=24)


@respx.mock
async def test_gdelt_rate_limited_reports_failure() -> None:
    # 免费档「每5秒1次」超限返回 429:此前被静默吞成 ok=True/空,污染 source_success_rate
    respx.get(_DOC).mock(return_value=httpx.Response(429, text="rate limited"))
    async with httpx.AsyncClient() as client:
        result = await GdeltProvider(client).fetch(_ctx())

    assert result.ok is False
    assert result.error == "rate_limited: HTTP 429"
    assert result.sentiment_features == {}
    assert result.news_items == []


@respx.mock
async def test_gdelt_rate_limit_cooldown_skips_second_http_call() -> None:
    route = respx.get(_DOC).mock(return_value=httpx.Response(429, text="rate limited"))
    async with httpx.AsyncClient() as client:
        provider = GdeltProvider(client)
        first = await provider.fetch(_ctx())
        second = await provider.fetch(_ctx())

    assert first.error == "rate_limited: HTTP 429"
    assert second.error == "rate_limited_cooldown"
    assert route.calls.call_count == 1


@respx.mock
async def test_gdelt_ok_parses_tone_and_articles() -> None:
    payload = {
        "tonechart": [
            {
                "bin": -3,
                "count": 2,
                "toparts": [{"url": "http://n/1", "title": "Crypto crash fears"}],
            },
            {"bin": 4, "count": 6, "toparts": [{"url": "http://n/2", "title": "Bitcoin surges"}]},
        ]
    }
    respx.get(_DOC).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient() as client:
        result = await GdeltProvider(client).fetch(_ctx())

    assert result.ok is True
    assert result.sentiment_features["gdelt_tone"] == 2.25  # (-3*2 + 4*6) / 8
    titles = {n.title for n in result.news_items}
    assert titles == {"Crypto crash fears", "Bitcoin surges"}
