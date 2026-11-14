"""Fear & Greed provider 离线测试:正常解析 + HTTP 错误 + 畸形 JSON(respx mock)。"""

from __future__ import annotations

import httpx
import respx

from binance_agent.intel.base import FetchContext
from binance_agent.intel.providers.fear_greed import FearGreedProvider

_URL = "https://api.alternative.me/fng/"


def _ctx() -> FetchContext:
    return FetchContext(symbol="BTCUSDT", base_asset="BTC", window_hours=24)


@respx.mock
async def test_fear_greed_parses_value() -> None:
    payload = {"data": [{"value": "72", "value_classification": "Greed"}]}
    respx.get(_URL).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient() as client:
        result = await FearGreedProvider(client).fetch(_ctx())

    assert result.ok is True
    assert result.sentiment_features["fear_greed"] == {"value": 72, "classification": "Greed"}


@respx.mock
async def test_fear_greed_http_error_reports_failure() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(500, text="oops"))
    async with httpx.AsyncClient() as client:
        result = await FearGreedProvider(client).fetch(_ctx())

    assert result.ok is False
    assert result.error == "HTTP 500"


@respx.mock
async def test_fear_greed_malformed_json_reports_failure() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(200, text="<html>not json</html>"))
    async with httpx.AsyncClient() as client:
        result = await FearGreedProvider(client).fetch(_ctx())

    assert result.ok is False
    assert result.error == "json_parse_error"


@respx.mock
async def test_fear_greed_empty_data_reports_failure() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    async with httpx.AsyncClient() as client:
        result = await FearGreedProvider(client).fetch(_ctx())

    assert result.ok is False
    assert result.error == "empty response"
