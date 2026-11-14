"""Finnhub provider 离线测试:新闻解析 + HTTP 错误不泄漏 key + 畸形/非 list JSON。"""

from __future__ import annotations

import httpx
import respx

from binance_agent.intel.base import FetchContext
from binance_agent.intel.providers.finnhub import FinnhubProvider

_URL = "https://finnhub.io/api/v1/news"
_KEY = "finnhubsecretkey456"


def _ctx() -> FetchContext:
    return FetchContext(symbol="BTCUSDT", base_asset="BTC", window_hours=24)


@respx.mock
async def test_finnhub_parses_news() -> None:
    payload = [
        {
            "headline": "Bitcoin ETF inflows surge",
            "summary": "Institutional demand grows.",
            "source": "Reuters",
            "url": "https://example.com/etf",
            "datetime": 1700000000,
        },
        {"headline": "", "summary": "no title skipped"},
    ]
    respx.get(_URL).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient() as client:
        result = await FinnhubProvider(client, _KEY).fetch(_ctx())

    assert result.ok is True
    assert len(result.news_items) == 1
    item = result.news_items[0]
    assert item.title == "Bitcoin ETF inflows surge"
    assert item.published_at.year == 2023  # 1700000000 epoch → 2023-11-14 UTC


@respx.mock
async def test_finnhub_http_error_does_not_leak_key() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(429, text="rate limit"))
    async with httpx.AsyncClient() as client:
        result = await FinnhubProvider(client, _KEY).fetch(_ctx())

    assert result.ok is False
    assert result.error == "HTTP 429"
    assert _KEY not in (result.error or "")


@respx.mock
async def test_finnhub_non_list_json_reports_failure() -> None:
    # 顶层应为 list;dict(如错误对象)或非 JSON 都按解析失败处理
    respx.get(_URL).mock(return_value=httpx.Response(200, json={"error": "wrong shape"}))
    async with httpx.AsyncClient() as client:
        result = await FinnhubProvider(client, _KEY).fetch(_ctx())

    assert result.ok is False
    assert result.error == "json_parse_error"


@respx.mock
async def test_finnhub_malformed_json_reports_failure() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(200, text="not json"))
    async with httpx.AsyncClient() as client:
        result = await FinnhubProvider(client, _KEY).fetch(_ctx())

    assert result.ok is False
    assert result.error == "json_parse_error"
