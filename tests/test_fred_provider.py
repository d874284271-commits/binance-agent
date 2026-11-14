"""FRED provider 离线测试:发布日历过滤 + HTTP 错误不泄漏 key + 畸形 JSON。"""

from __future__ import annotations

import httpx
import respx

from binance_agent.intel.base import FetchContext
from binance_agent.intel.providers.fred import FredProvider

_URL = "https://api.stlouisfed.org/fred/releases/dates"
_KEY = "fredsecretkey789"


def _ctx() -> FetchContext:
    return FetchContext(symbol="BTCUSDT", base_asset="BTC", window_hours=24)


@respx.mock
async def test_fred_keeps_only_important_releases() -> None:
    payload = {
        "release_dates": [
            {"release_name": "Consumer Price Index", "date": "2026-06-11"},
            {"release_name": "Consumer Price Index", "date": "2026-06-11"},  # 重复去重
            {"release_name": "Obscure Regional Survey", "date": "2026-06-12"},  # 低重要性过滤
            {"release_name": "FOMC Press Conference", "date": "2026-06-13"},
        ]
    }
    respx.get(_URL).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient() as client:
        result = await FredProvider(client, _KEY).fetch(_ctx())

    assert result.ok is True
    names = [e.name for e in result.macro_events]
    assert names == ["Consumer Price Index", "FOMC Press Conference"]
    assert all(e.importance >= 2 for e in result.macro_events)


@respx.mock
async def test_fred_http_error_does_not_leak_key() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(403, text="forbidden"))
    async with httpx.AsyncClient() as client:
        result = await FredProvider(client, _KEY).fetch(_ctx())

    assert result.ok is False
    assert result.error == "HTTP 403"
    assert _KEY not in (result.error or "")


@respx.mock
async def test_fred_malformed_json_reports_failure() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(200, text="not json"))
    async with httpx.AsyncClient() as client:
        result = await FredProvider(client, _KEY).fetch(_ctx())

    assert result.ok is False
    assert result.error == "json_parse_error"
