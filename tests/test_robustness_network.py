"""Offline regressions for input isolation, bounded HTTP and websocket lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import zlib
from typing import Any

import httpx
import pytest

import binance_agent.market.ws_manager as wm
from binance_agent.infra.bounded_http import (
    MAX_RESPONSE_BYTES,
    BoundedAsyncClient,
    ResponseTooLarge,
)
from binance_agent.intel.aggregator import IntelAggregator
from binance_agent.intel.base import FetchContext
from binance_agent.intel.content import ContentFetcher
from binance_agent.intel.providers.cryptopanic import CryptoPanicProvider
from binance_agent.intel.providers.fear_greed import FearGreedProvider
from binance_agent.intel.providers.finnhub import FinnhubProvider
from binance_agent.intel.providers.gdelt import GdeltProvider
from binance_agent.intel.providers.http_news import GithubReleasesProvider, GuardianProvider
from binance_agent.intel.providers.reddit import RedditProvider
from binance_agent.market.snapshot import build_market_snapshot

CTX = FetchContext("BTCUSDT", "BTC", 24)


@pytest.mark.parametrize(
    "provider,payload",
    [
        (
            lambda c: FinnhubProvider(c, "fake"),
            [None, {"headline": "Bitcoin update", "datetime": 1700000000}],
        ),
        (
            lambda c: GuardianProvider(c, "fake"),
            {"response": {"results": [None, {"webTitle": "Bitcoin update"}]}},
        ),
        (FearGreedProvider, {"data": [None, {"value": "52"}]}),
        (
            RedditProvider,
            {
                "data": {
                    "children": [
                        None,
                        {"data": {"title": "Bitcoin update", "created_utc": 1700000000}},
                    ]
                }
            },
        ),
        (CryptoPanicProvider, {"results": [None, {"title": "Bitcoin update"}]}),
        (
            GdeltProvider,
            {
                "tonechart": [
                    None,
                    {"bin": 2, "count": 1, "toparts": [None, {"title": "Bitcoin update"}]},
                ]
            },
        ),
    ],
)
async def test_provider_retains_valid_rows_and_surfaces_warnings(
    provider: Any, payload: Any
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as client:
        # CryptoPanic has a required fake token.
        source = provider(client, "fake") if provider is CryptoPanicProvider else provider(client)
        result = await source.fetch(CTX)
    assert result.ok
    assert result.warnings and result.warnings[0].startswith("skipped_invalid_rows:")
    assert result.news_items or result.sentiment_features


@pytest.mark.parametrize(
    "payload", [[None], [{"headline": "x", "datetime": "NaN"}], [{"headline": [], "datetime": 1}]]
)
async def test_all_invalid_rows_mark_source_failed(payload: Any) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as client:
        result = await FinnhubProvider(client, "fake").fetch(CTX)
    assert not result.ok and result.error == "invalid_rows"
    assert not result.news_items


async def test_aggregator_preserves_partial_source_warnings() -> None:
    payload = [None, {"headline": "Bitcoin update", "datetime": 1700000000}]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as client:
        snapshot = await IntelAggregator([FinnhubProvider(client, "fake")]).gather("BTCUSDT")
    assert snapshot.source_status["finnhub"] == "ok"
    assert snapshot.source_warnings == {"finnhub": ["skipped_invalid_rows: 1"]}


async def test_github_repo_failure_does_not_discard_other_repositories() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/broken/" in request.url.path:
            raise httpx.ConnectError("secret URL must not be copied", request=request)
        return httpx.Response(200, json=[None, {"name": "release"}])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await GithubReleasesProvider(client, ["broken/repo", "good/repo"]).fetch(CTX)
    assert result.ok and len(result.news_items) == 1
    assert "repository_unavailable" in result.warnings
    assert "secret" not in str(result)


class WireStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.closed = False
        self.consumed = 0

    async def __aiter__(self):
        for offset in range(0, len(self.content), 4096):
            self.consumed += 1
            yield self.content[offset : offset + 4096]

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "encoding,compress",
    [("gzip", gzip.compress), ("deflate", zlib.compress), ("identity", lambda b: b)],
)
async def test_bounded_http_accepts_valid_encoding(encoding: str, compress: Any) -> None:
    stream = WireStream(compress(b'{"value": 7}'))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, headers={"content-encoding": encoding}, stream=stream)

    async with BoundedAsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await client.get("https://source.invalid/news")
    assert response.json() == {"value": 7}
    assert stream.closed


@pytest.mark.parametrize(
    "encoding,compress",
    [("gzip", gzip.compress), ("deflate", zlib.compress), ("identity", lambda b: b)],
)
async def test_bounded_http_rejects_oversized_wire_or_decoded_body(
    encoding: str, compress: Any
) -> None:
    stream = WireStream(compress(b"a" * (MAX_RESPONSE_BYTES + 100_000)))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-encoding": encoding}, stream=stream)

    async with BoundedAsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ResponseTooLarge, match="response_too_large"):
            await client.get("https://source.invalid/news")
    assert stream.closed


async def test_bounded_http_real_stream_closes_before_reading_oversized_payload() -> None:
    payload = gzip.compress(b"a" * (64 * 1024 * 1024))
    finished = asyncio.Event()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nConnection: close\r\n\r\n")
            for offset in range(0, len(payload), 1024):
                writer.write(payload[offset : offset + 1024])
                await writer.drain()
                await asyncio.sleep(0)
        except (OSError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
            finished.set()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        async with BoundedAsyncClient(trust_env=False) as client:
            with pytest.raises(ResponseTooLarge):
                await client.get(f"http://127.0.0.1:{port}/")
        await asyncio.wait_for(finished.wait(), 2)


async def test_content_extractor_failure_becomes_unavailable() -> None:
    class Transport:
        async def fetch(self, *args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                200, content=b"<p>body</p>", headers={"content-type": "text/html"}
            )

    async def resolver(host: str) -> tuple[str, ...]:
        return ("93.184.216.34",)

    def extractor(html: str, url: str) -> str:
        raise RuntimeError("private extractor state")

    document = await ContentFetcher(
        transport=Transport(), resolver=resolver, extractor=extractor
    ).fetch("https://example.com/x")
    assert document.status == "unavailable"
    assert document.fetch_error == "extractor_failed"


class FakeWs:
    def __init__(self, messages: list[str], *, idle: bool = False) -> None:
        self.messages = messages
        self.idle = idle
        self.closed = False
        self.cancelled = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: Any):
        self.closed = True

    def __aiter__(self):
        async def iterate():
            for message in self.messages:
                yield message
            if self.idle:
                try:
                    await asyncio.Future()
                finally:
                    self.cancelled = True

        return iterate()


async def test_ws_non_object_frames_are_dropped_without_reconnect(monkeypatch: Any) -> None:
    ws = FakeWs(["[]", "null", '{"data": []}', '{"data": {"s": "BTCUSDT"}}'])
    calls = 0

    def connect(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return ws

    monkeypatch.setattr(wm, "ws_connect", connect)
    stop = asyncio.Event()
    async for data in wm.BinanceWsManager().stream(["x"], stop=stop):
        assert data == {"s": "BTCUSDT"}
        stop.set()
    assert calls == 1 and ws.closed


async def test_ws_stop_while_idle_collects_receive_task(monkeypatch: Any) -> None:
    ws = FakeWs([], idle=True)
    monkeypatch.setattr(wm, "ws_connect", lambda *a, **k: ws)
    stop = asyncio.Event()

    async def consume():
        async for _ in wm.BinanceWsManager().stream(["x"], stop=stop):
            pass

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(consumer, 0.5)
    assert ws.closed and ws.cancelled


async def test_ws_normal_close_uses_exponential_backoff(monkeypatch: Any) -> None:
    delays = []

    def connect(*args: Any, **kwargs: Any):
        return FakeWs([])

    async def sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 7:
            raise asyncio.CancelledError

    monkeypatch.setattr(wm, "ws_connect", connect)
    monkeypatch.setattr(wm.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        async for _ in wm.BinanceWsManager().stream(["x"]):
            pytest.fail("empty connection must not yield")
    assert delays == [1, 2, 4, 8, 16, 30, 30]


async def test_snapshot_bad_rows_are_not_fabricated_as_zero() -> None:
    class Client:
        market = "fapi"
        has_derivatives = True

        async def klines(self, symbol, interval):
            return [None, [1, "NaN", 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]]

        async def premium_index(self, symbol):
            return {"markPrice": "NaN", "lastFundingRate": True}

        async def funding_rate_history(self, symbol, **kwargs):
            return [None, {"fundingRate": "0.1"}, None]

        async def open_interest(self, symbol):
            return {"openInterest": "Infinity"}

        async def open_interest_hist(self, symbol):
            return [None, {"sumOpenInterest": "7", "timestamp": 1}]

        async def global_long_short_ratio(self, symbol):
            return [
                None,
                {"timestamp": "bad", "longShortRatio": "NaN"},
                {"timestamp": 1, "longShortRatio": "2"},
            ]

        async def ticker_24h(self, symbol):
            return {"lastPrice": "NaN", "volume": True}

    snapshot = await build_market_snapshot(Client(), "BTCUSDT", ["1h"])
    assert len(snapshot.klines["1h"]) == 1
    assert len(snapshot.long_short_ratio) == 1
    assert snapshot.long_short_ratio[0].timestamp == 1
    assert snapshot.funding.last_funding_rate == 0.1
    assert snapshot.open_interest is None
    assert len(snapshot.open_interest_hist) == 1
    assert snapshot.ticker_24h == {}
    assert snapshot.errors


@pytest.mark.parametrize("streaming", [True, False])
async def test_bounded_http_accepts_exact_limit_and_mock_materialized_response(
    streaming: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if streaming:
            return httpx.Response(200, stream=WireStream(b"a" * MAX_RESPONSE_BYTES))
        return httpx.Response(200, content=b"a" * MAX_RESPONSE_BYTES)

    async with BoundedAsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await client.get("https://source.invalid/news")
    assert len(response.content) == MAX_RESPONSE_BYTES


async def test_bounded_http_declared_oversize_closes_without_consuming() -> None:
    stream = WireStream(b"unused")
    async with BoundedAsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, headers={"content-length": str(MAX_RESPONSE_BYTES + 1)}, stream=stream
            )
        )
    ) as client:
        with pytest.raises(ResponseTooLarge):
            await client.get("https://source.invalid/news")
    assert stream.closed and stream.consumed == 0


async def test_content_extractor_cancellation_propagates() -> None:
    class Transport:
        async def fetch(self, *args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(
                200, content=b"<p>body</p>", headers={"content-type": "text/html"}
            )

    async def resolver(host: str) -> tuple[str, ...]:
        return ("93.184.216.34",)

    def extractor(html: str, url: str) -> str:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await ContentFetcher(transport=Transport(), resolver=resolver, extractor=extractor).fetch(
            "https://example.com/x"
        )


async def test_redirect_body_is_closed_unread_and_cross_origin_auth_is_stripped() -> None:
    stream = WireStream(gzip.compress(b"x" * (64 * 1024 * 1024)))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "first.invalid":
            return httpx.Response(
                302,
                headers={"location": "https://other.invalid/final", "content-encoding": "gzip"},
                stream=stream,
            )
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"ok": True})

    async with BoundedAsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        headers={"Authorization": "Bearer fake"},
    ) as client:
        response = await client.get("https://first.invalid/start")
    assert response.json() == {"ok": True}
    assert len(response.history) == 1
    assert stream.closed and stream.consumed == 0
