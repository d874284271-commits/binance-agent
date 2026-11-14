"""M7/M8 WS 测试:组合流解析 + price_stream 双市场字段映射(mock ws_connect,无需联网)。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import binance_agent.market.ws_manager as wm


class _FakeWs:
    def __init__(self, messages: list[str]) -> None:
        self._messages = list(messages)

    async def __aenter__(self) -> _FakeWs:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def __aiter__(self) -> Any:
        async def _gen() -> Any:
            for message in self._messages:
                yield message

        return _gen()


async def test_ws_manager_parses_combined_stream(monkeypatch: Any) -> None:
    messages = ['{"stream":"btcusdt@markPrice","data":{"s":"BTCUSDT","p":"70000.0","r":"0.0001"}}']
    monkeypatch.setattr(wm, "ws_connect", lambda *a, **k: _FakeWs(messages))

    manager = wm.BinanceWsManager()
    stop = asyncio.Event()
    received = []
    async for data in manager.stream(["btcusdt@markPrice"], stop=stop):
        received.append(data)
        stop.set()  # 收一条即停,避免重连循环

    assert received
    assert received[0]["s"] == "BTCUSDT"
    assert received[0]["p"] == "70000.0"


def _capture_connect(captured: dict[str, Any], messages: list[str]) -> Any:
    def _connect(url: str, *args: Any, **kwargs: Any) -> _FakeWs:
        captured["url"] = url
        return _FakeWs(messages)

    return _connect


async def test_price_stream_fapi(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    messages = ['{"data":{"s":"BTCUSDT","p":"70000.5","r":"0.0001","T":1234567890}}']
    monkeypatch.setattr(wm, "ws_connect", _capture_connect(captured, messages))

    manager = wm.BinanceWsManager("fapi")
    stop = asyncio.Event()
    out = []
    async for evt in manager.price_stream("BTCUSDT", stop=stop):
        out.append(evt)
        stop.set()

    assert "fstream.binance.com" in captured["url"]
    assert "btcusdt@markprice" in captured["url"].lower()
    assert out[0]["market"] == "fapi"
    assert out[0]["price"] == "70000.5"
    assert out[0]["mark_price"] == "70000.5"
    assert out[0]["funding_rate"] == "0.0001"
    assert out[0]["next_funding_time"] == 1234567890


async def test_price_stream_spot(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    messages = ['{"data":{"s":"BTCUSDT","c":"71000.0"}}']
    monkeypatch.setattr(wm, "ws_connect", _capture_connect(captured, messages))

    manager = wm.BinanceWsManager("api")
    stop = asyncio.Event()
    out = []
    async for evt in manager.price_stream("BTCUSDT", stop=stop):
        out.append(evt)
        stop.set()

    assert "stream.binance.com:9443" in captured["url"]
    assert "btcusdt@ticker" in captured["url"]
    assert out[0]["market"] == "api"
    assert out[0]["price"] == "71000.0"
    assert out[0]["mark_price"] is None
    assert out[0]["funding_rate"] is None
    assert out[0]["next_funding_time"] is None


def test_ws_manager_rejects_unknown_market() -> None:
    with pytest.raises(ValueError):
        wm.BinanceWsManager("spot")
