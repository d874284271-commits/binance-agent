"""M1 行情数据层离线测试:缓存 / 限速器 / 快照组装(respx mock,无需联网)。"""

from __future__ import annotations

import httpx
import pytest
import respx

from binance_agent.market.cache import TTLCache
from binance_agent.market.rate_limiter import RateLimiter
from binance_agent.market.rest_client import FAPI_BASE, SPOT_BASE, BinanceRestClient
from binance_agent.market.snapshot import build_market_snapshot

_KLINE_ROW = [
    1700000000000,
    "70000.0",
    "70500.0",
    "69500.0",
    "70200.0",
    "1234.5",
    1700003599999,
    "0",
    0,
    "0",
    "0",
    "0",
]


def test_ttl_cache_hit_and_expiry() -> None:
    cache = TTLCache()
    cache.set("k", 123, ttl=60)
    assert cache.get("k") == 123
    cache.set("expired", 1, ttl=-1)  # 立即过期
    assert cache.get("expired") is None
    assert cache.get("missing") is None


def test_rate_limiter_observe_parses_weight() -> None:
    limiter = RateLimiter(soft_limit=2000)
    limiter.observe({"X-MBX-USED-WEIGHT-1M": "123"})
    assert limiter.used_weight == 123
    limiter.observe({"x-mbx-used-weight-1m": "not-a-number"})  # 容错:保持旧值
    assert limiter.used_weight == 123


def test_rate_limiter_reset_clears_weight() -> None:
    limiter = RateLimiter(soft_limit=2000)
    limiter.observe({"X-MBX-USED-WEIGHT-1M": "1999"})
    limiter.reset()  # 429/418 退避后调用,避免基于过期权重二次节流
    assert limiter.used_weight == 0


@respx.mock
async def test_shared_cache_isolates_markets() -> None:
    # 显式共享 TTLCache 时,fapi/api 同路径同参数不得互相污染(缓存键含市场前缀)
    respx.get(f"{FAPI_BASE}/fapi/v1/ticker/24hr").mock(
        return_value=httpx.Response(200, json={"lastPrice": "1.0"})
    )
    respx.get(f"{SPOT_BASE}/api/v3/ticker/24hr").mock(
        return_value=httpx.Response(200, json={"lastPrice": "2.0"})
    )
    shared = TTLCache()
    fapi = BinanceRestClient("fapi", cache=shared)
    spot = BinanceRestClient("api", cache=shared)
    try:
        fapi_data = await fapi.ticker_24h("BTCUSDT")
        spot_data = await spot.ticker_24h("BTCUSDT")
    finally:
        await fapi.aclose()
        await spot.aclose()

    assert fapi_data["lastPrice"] == "1.0"
    assert spot_data["lastPrice"] == "2.0"


@respx.mock
async def test_klines_range_paginates_and_deduplicates_overlapping_pages() -> None:
    calls: list[int] = []

    def page(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["startTime"])
        calls.append(start)
        rows_by_start = {
            1000: [
                [1000, "1", "2", "0.5", "1.5", "1", 1999],
                [2000, "1", "2", "0.5", "1.5", "1", 2999],
            ],
            2001: [
                [2000, "1", "2", "0.5", "1.5", "1", 2999],
                [3000, "1", "2", "0.5", "1.5", "1", 3999],
            ],
            3001: [[4000, "1", "2", "0.5", "1.5", "1", 4999]],
        }
        return httpx.Response(200, json=rows_by_start[start])

    respx.get(f"{FAPI_BASE}/fapi/v1/klines").mock(side_effect=page)
    client = BinanceRestClient()
    try:
        rows = await client.klines_range("BTCUSDT", "1m", start_time=1000, end_time=5000, limit=2)
    finally:
        await client.aclose()

    assert [row[0] for row in rows] == [1000, 2000, 3000, 4000]
    assert calls == [1000, 2001, 3001]


@respx.mock
async def test_klines_range_fetches_more_than_binance_single_page_limit() -> None:
    first_page = [
        [index * 60_000, "1", "2", "0.5", "1.5", "1", index * 60_000 + 59_999]
        for index in range(1000)
    ]
    second_page = [
        [60_000_000, "1", "2", "0.5", "1.5", "1", 60_059_999],
        [60_060_000, "1", "2", "0.5", "1.5", "1", 60_119_999],
    ]

    def page(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["startTime"])
        return httpx.Response(200, json=first_page if start == 0 else second_page)

    route = respx.get(f"{FAPI_BASE}/fapi/v1/klines").mock(side_effect=page)
    client = BinanceRestClient()
    try:
        rows = await client.klines_range("BTCUSDT", "1m", start_time=0, end_time=60_120_000)
    finally:
        await client.aclose()

    assert len(rows) == 1002
    assert route.call_count == 2


@respx.mock
async def test_klines_range_stops_when_upstream_page_makes_no_progress() -> None:
    repeated = [
        [1000, "1", "2", "0.5", "1.5", "1", 1999],
        [2000, "1", "2", "0.5", "1.5", "1", 2999],
    ]
    route = respx.get(f"{FAPI_BASE}/fapi/v1/klines").mock(
        return_value=httpx.Response(200, json=repeated)
    )
    client = BinanceRestClient()
    try:
        with pytest.raises(RuntimeError, match="无进展"):
            await client.klines_range("BTCUSDT", "1m", start_time=1000, end_time=5000, limit=2)
    finally:
        await client.aclose()

    assert route.call_count == 2


async def test_klines_range_rejects_unsupported_interval_before_network() -> None:
    client = BinanceRestClient()
    try:
        with pytest.raises(ValueError, match="interval"):
            await client.klines_range("BTCUSDT", "7m", start_time=1000, end_time=5000)
    finally:
        await client.aclose()


def _mock_all_endpoints() -> None:
    respx.get(f"{FAPI_BASE}/fapi/v1/klines").mock(
        return_value=httpx.Response(200, json=[_KLINE_ROW])
    )
    respx.get(f"{FAPI_BASE}/fapi/v1/premiumIndex").mock(
        return_value=httpx.Response(
            200,
            json={
                "symbol": "BTCUSDT",
                "markPrice": "70250.0",
                "lastFundingRate": "0.0001",
                "nextFundingTime": 1700000000000,
            },
        )
    )
    respx.get(f"{FAPI_BASE}/fapi/v1/fundingRate").mock(
        return_value=httpx.Response(200, json=[{"symbol": "BTCUSDT", "fundingRate": "0.0001"}])
    )
    respx.get(f"{FAPI_BASE}/fapi/v1/openInterest").mock(
        return_value=httpx.Response(200, json={"symbol": "BTCUSDT", "openInterest": "100000.0"})
    )
    respx.get(f"{FAPI_BASE}/futures/data/openInterestHist").mock(
        return_value=httpx.Response(
            200,
            json=[{"sumOpenInterest": "100000", "timestamp": 1700000000000}],
        )
    )
    respx.get(f"{FAPI_BASE}/futures/data/globalLongShortAccountRatio").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "longShortRatio": "1.5",
                    "longAccount": "0.6",
                    "shortAccount": "0.4",
                    "timestamp": 1700000000000,
                }
            ],
        )
    )
    respx.get(f"{FAPI_BASE}/fapi/v1/ticker/24hr").mock(
        return_value=httpx.Response(
            200,
            json={
                "lastPrice": "70200.0",
                "priceChangePercent": "1.5",
                "highPrice": "71000",
                "lowPrice": "69000",
                "volume": "50000",
                "quoteVolume": "3.5e9",
            },
        )
    )


@respx.mock
async def test_build_market_snapshot_full() -> None:
    _mock_all_endpoints()
    client = BinanceRestClient()
    try:
        snap = await build_market_snapshot(client, "btcusdt")
    finally:
        await client.aclose()

    assert snap.symbol == "BTCUSDT"
    assert set(snap.klines) == {"5m", "1h", "4h", "1d"}
    assert all(len(bars) == 1 for bars in snap.klines.values())
    assert snap.klines["1h"][0].close == 70200.0
    assert snap.funding.mark_price == 70250.0
    assert snap.funding.last_funding_rate == 0.0001
    assert snap.open_interest == 100000.0
    assert len(snap.open_interest_hist) == 1
    assert len(snap.long_short_ratio) == 1
    assert snap.long_short_ratio[0].long_short_ratio == 1.5
    assert snap.ticker_24h["lastPrice"] == "70200.0"
    assert snap.errors == []


@respx.mock
async def test_build_market_snapshot_partial_failure() -> None:
    _mock_all_endpoints()
    # 单源故障:持仓量端点 500,应被记录到 errors 而不影响其余字段
    respx.get(f"{FAPI_BASE}/fapi/v1/openInterest").mock(
        return_value=httpx.Response(500, json={"code": -1, "msg": "boom"})
    )
    client = BinanceRestClient()
    try:
        snap = await build_market_snapshot(client, "BTCUSDT")
    finally:
        await client.aclose()

    assert snap.open_interest is None
    assert any("openInterest" in err for err in snap.errors)
    # 其余字段仍正常解析
    assert snap.funding.mark_price == 70250.0
    assert len(snap.klines) == 4


@respx.mock
async def test_build_market_snapshot_spot() -> None:
    # 现货(api)只调 /api/v3 的 K线+ticker;不应调用任何 /fapi 衍生品端点
    respx.get(f"{SPOT_BASE}/api/v3/klines").mock(
        return_value=httpx.Response(200, json=[_KLINE_ROW])
    )
    respx.get(f"{SPOT_BASE}/api/v3/ticker/24hr").mock(
        return_value=httpx.Response(
            200,
            json={
                "lastPrice": "70200.0",
                "priceChangePercent": "1.5",
                "highPrice": "71000",
                "lowPrice": "69000",
                "volume": "50000",
                "quoteVolume": "3.5e9",
            },
        )
    )
    client = BinanceRestClient("api")
    try:
        snap = await build_market_snapshot(client, "BTCUSDT")
    finally:
        await client.aclose()

    assert snap.market == "api"
    assert set(snap.klines) == {"5m", "1h", "4h", "1d"}
    assert snap.ticker_24h["lastPrice"] == "70200.0"
    # 现货无衍生品数据
    assert snap.funding.mark_price is None
    assert snap.open_interest is None
    assert snap.long_short_ratio == []
    assert snap.errors == []
