"""CryptoPanic provider 离线测试:投票情绪解析 + HTTP 错误不泄漏 token + 畸形 JSON。"""

from __future__ import annotations

import httpx
import respx

from binance_agent.intel.base import FetchContext
from binance_agent.intel.providers.cryptopanic import CryptoPanicProvider

_URL = "https://cryptopanic.com/api/v1/posts/"
_TOKEN = "supersecrettoken123"


def _ctx() -> FetchContext:
    return FetchContext(symbol="BTCUSDT", base_asset="BTC", window_hours=24)


@respx.mock
async def test_cryptopanic_parses_posts_and_votes() -> None:
    payload = {
        "results": [
            {
                "title": "BTC breaks resistance",
                "source": {"title": "NewsDesk"},
                "url": "https://example.com/1",
                "published_at": "2026-06-09T12:00:00Z",
                "votes": {"positive": 10, "negative": 2},
            },
            {
                "title": "Exchange hacked",
                "url": "https://example.com/2",
                "published_at": "2026-06-09T13:00:00Z",
                "votes": {"positive": 1, "negative": 8},
            },
        ]
    }
    respx.get(_URL).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient() as client:
        result = await CryptoPanicProvider(client, _TOKEN).fetch(_ctx())

    assert result.ok is True
    assert [n.sentiment for n in result.news_items] == ["bull", "bear"]
    votes = result.sentiment_features["cryptopanic_votes"]
    assert votes == {"positive": 11, "negative": 10, "posts": 2}


@respx.mock
async def test_cryptopanic_http_error_does_not_leak_token() -> None:
    # 401 时 error 必须是规范短串,绝不能内嵌带 auth_token 的请求 URL
    respx.get(_URL).mock(return_value=httpx.Response(401, text="unauthorized"))
    async with httpx.AsyncClient() as client:
        result = await CryptoPanicProvider(client, _TOKEN).fetch(_ctx())

    assert result.ok is False
    assert result.error == "HTTP 401"
    assert _TOKEN not in (result.error or "")


@respx.mock
async def test_cryptopanic_malformed_json_reports_failure() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(200, text="not json"))
    async with httpx.AsyncClient() as client:
        result = await CryptoPanicProvider(client, _TOKEN).fetch(_ctx())

    assert result.ok is False
    assert result.error == "json_parse_error"
