"""M2 资讯聚合离线测试:合并 / 去重 / 单源失败隔离 / 超时 / 时间窗 / 禁用上报。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from binance_agent.intel.aggregator import IntelAggregator
from binance_agent.intel.base import (
    FetchContext,
    Provider,
    ProviderResult,
    split_base_asset,
)
from binance_agent.intel.models import NewsItem


def _news(title: str, url: str = "", hours_ago: float = 1.0) -> NewsItem:
    return NewsItem(
        title=title,
        url=url,
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
    )


class FakeProvider(Provider):
    def __init__(
        self,
        name: str,
        *,
        result: ProviderResult | None = None,
        exc: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.name = name
        self._result = result
        self._exc = exc
        self._delay = delay

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._exc is not None:
            raise self._exc
        return self._result if self._result is not None else ProviderResult(source=self.name)


def test_split_base_asset() -> None:
    assert split_base_asset("BTCUSDT") == "BTC"
    assert split_base_asset("ETHUSDC") == "ETH"
    assert split_base_asset("ETHFDUSD") == "ETH"
    assert split_base_asset("WEIRD") == "WEIRD"


async def test_aggregator_merges_and_dedups() -> None:
    p1 = FakeProvider(
        "a",
        result=ProviderResult(
            source="a",
            sentiment_features={"fear_greed": {"value": 50}},
            news_items=[_news("Bitcoin rallies hard", "http://x/1")],
        ),
    )
    p2 = FakeProvider(
        "b",
        result=ProviderResult(
            source="b",
            sentiment_features={"gdelt_tone": -2.0},
            news_items=[
                _news("Bitcoin rallies hard", "http://x/1"),  # 同 URL → 去重
                _news("ETH upgrade ships", "http://x/2"),
            ],
        ),
    )
    agg = IntelAggregator([p1, p2], window_hours=24, max_news_items=40)
    snap = await agg.gather("BTCUSDT", research_question="protocol event")

    assert snap.symbol == "BTCUSDT"
    assert snap.sentiment_features == {"fear_greed": {"value": 50}, "gdelt_tone": -2.0}
    assert len(snap.news_items) == 2
    assert snap.source_status["a"] == "ok"
    assert snap.source_status["b"] == "ok"


async def test_aggregator_single_source_failure_isolated() -> None:
    good = FakeProvider("good", result=ProviderResult(source="good", sentiment_features={"x": 1}))
    bad = FakeProvider("bad", exc=RuntimeError("boom"))
    agg = IntelAggregator([good, bad])
    snap = await agg.gather("BTCUSDT")

    assert snap.sentiment_features == {"x": 1}
    assert snap.source_status["good"] == "ok"
    assert "error" in snap.source_status["bad"]
    assert "boom" in snap.source_status["bad"]


async def test_aggregator_timeout() -> None:
    slow = FakeProvider("slow", result=ProviderResult(source="slow"), delay=0.5)
    agg = IntelAggregator([slow], per_source_timeout=0.05)
    snap = await agg.gather("BTCUSDT")
    assert snap.source_status["slow"] == "error: timeout"


async def test_aggregator_window_filter() -> None:
    provider = FakeProvider(
        "p",
        result=ProviderResult(
            source="p",
            news_items=[
                _news("recent", "http://x/r", hours_ago=1),
                _news("stale", "http://x/o", hours_ago=48),
            ],
        ),
    )
    agg = IntelAggregator([provider], window_hours=24)
    snap = await agg.gather("BTCUSDT")
    titles = {n.title for n in snap.news_items}
    assert "recent" in titles
    assert "stale" not in titles


async def test_aggregator_reports_disabled() -> None:
    agg = IntelAggregator([], disabled={"finnhub": "缺少 key(FINNHUB_KEY)"})
    snap = await agg.gather("BTCUSDT")
    assert snap.source_status["finnhub"] == "缺少 key(FINNHUB_KEY)"


async def test_aggregator_short_titles_dedup_by_url_only() -> None:
    provider = FakeProvider(
        "a",
        result=ProviderResult(
            source="a",
            news_items=[
                _news("BTC", "http://x/1"),  # 规范化后过短("btc"),不应按标题去重
                _news("BTC!", "http://x/2"),  # 同样规范化为 "btc",URL 不同 → 保留
                # 显式错开时间戳使「最新优先」排序确定:x/3 较新 → 存活,x/4 被同形标题去重。
                # (若都用 now(),Windows ~15ms 时钟粒度与 Linux 微秒级会选出不同幸存者,平台 flaky)
                _news("Bitcoin breaks all time high record", "http://x/3", hours_ago=1.0),
                _news(
                    "Bitcoin breaks all-time high record!", "http://x/4", hours_ago=2.0
                ),  # 长标题同形 → 去重
            ],
        ),
    )
    agg = IntelAggregator([provider], window_hours=24, max_news_items=40)
    snap = await agg.gather("BTCUSDT")

    urls = {item.url for item in snap.news_items}
    assert urls == {"http://x/1", "http://x/2", "http://x/3"}


async def test_relevance_is_ranked_before_news_cap() -> None:
    provider = FakeProvider(
        "a",
        result=ProviderResult(
            source="a",
            news_items=[
                _news("Market commentary 1", "http://x/1", hours_ago=1),
                _news("Market commentary 2", "http://x/2", hours_ago=2),
                _news("Bitcoin protocol event", "http://x/3", hours_ago=12),
            ],
        ),
    )
    # The directly relevant item is older than the unrelated headlines but is
    # selected before the cap.
    agg = IntelAggregator([provider], window_hours=24, max_news_items=1)
    snap = await agg.gather("BTCUSDT")
    assert [item.title for item in snap.news_items] == ["Bitcoin protocol event"]
