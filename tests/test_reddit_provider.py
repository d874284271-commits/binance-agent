"""M8+ Reddit provider 离线测试:帖子 → news_items + reddit_buzz(respx mock,无需联网)。"""

from __future__ import annotations

import httpx
import respx

from binance_agent.intel.base import FetchContext
from binance_agent.intel.providers.reddit import RedditProvider

_SEARCH_URL = "https://www.reddit.com/r/CryptoCurrency/search.json"


def _ctx() -> FetchContext:
    return FetchContext(symbol="BTCUSDT", base_asset="BTC", window_hours=24)


@respx.mock
async def test_reddit_parses_posts() -> None:
    payload = {
        "data": {
            "children": [
                {
                    "data": {
                        "title": "BTC to the moon",
                        "permalink": "/r/CryptoCurrency/comments/1/",
                        "subreddit": "CryptoCurrency",
                        "score": 120,
                        "num_comments": 30,
                        "created_utc": 1700000000,
                    }
                },
                {
                    "data": {
                        "title": "Bear market incoming?",
                        "permalink": "/r/CryptoCurrency/comments/2/",
                        "subreddit": "CryptoCurrency",
                        "score": 45,
                        "num_comments": 12,
                        "created_utc": 1700000100,
                    }
                },
            ]
        }
    }
    respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json=payload))

    async with httpx.AsyncClient() as client:
        result = await RedditProvider(client).fetch(_ctx())

    assert result.ok
    assert result.source == "reddit"
    assert len(result.news_items) == 2
    item = result.news_items[0]
    assert item.title == "BTC to the moon"
    assert item.source == "r/CryptoCurrency"
    assert item.url == "https://www.reddit.com/r/CryptoCurrency/comments/1/"
    assert item.published_at.year == 2023  # 1700000000 epoch → 2023-11-14 UTC
    buzz = result.sentiment_features["reddit_buzz"]
    assert buzz["posts"] == 2
    assert buzz["total_score"] == 165
    assert buzz["total_comments"] == 42


@respx.mock
async def test_reddit_empty_ok() -> None:
    respx.get(_SEARCH_URL).mock(return_value=httpx.Response(200, json={"data": {"children": []}}))
    async with httpx.AsyncClient() as client:
        result = await RedditProvider(client).fetch(_ctx())

    assert result.ok
    assert result.news_items == []
    assert result.sentiment_features["reddit_buzz"]["posts"] == 0


@respx.mock
async def test_reddit_blocked_public_endpoint_reports_clear_error() -> None:
    respx.get(_SEARCH_URL).mock(return_value=httpx.Response(403, text="blocked"))
    async with httpx.AsyncClient() as client:
        result = await RedditProvider(client).fetch(_ctx())

    assert result.ok is False
    assert result.error == "blocked_public_endpoint: HTTP 403"
    assert result.news_items == []
    assert result.sentiment_features == {}
