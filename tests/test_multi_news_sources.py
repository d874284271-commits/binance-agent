from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from binance_agent.agents.research import _sanitize_research
from binance_agent.infra.settings import Settings, load_settings
from binance_agent.intel import aggregator as aggregator_mod
from binance_agent.intel.aggregator import IntelAggregator, build_providers
from binance_agent.intel.base import FetchContext, Provider, ProviderResult
from binance_agent.intel.content import document_from_news_item
from binance_agent.intel.models import NewsItem
from binance_agent.intel.providers.http_news import (
    GithubReleasesProvider,
    GNewsProvider,
    GuardianProvider,
    MediastackProvider,
    NewsApiProvider,
)
from binance_agent.intel.providers.rss import RssProvider


class FakeClient:
    def __init__(self, responses: dict[str, httpx.Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append((url, kwargs))
        response = self.responses.get(url)
        if response is None:
            return httpx.Response(404, request=httpx.Request("GET", url))
        return response


def _ctx() -> FetchContext:
    return FetchContext(symbol="BTCUSDT", base_asset="BTC", window_hours=24)


def _response(
    url: str, status: int, payload: Any = None, content: bytes | None = None
) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload if content is None else None,
        content=content,
        request=httpx.Request("GET", url),
    )


async def test_named_rss_uses_http_client_and_source_metadata() -> None:
    url = "https://official.example/feed.xml"
    xml = b"""<?xml version="1.0"?><rss><channel><title>Official</title>
    <item><title>Bitcoin policy update</title><description>Policy details</description>
    <link>https://official.example/item</link><pubDate>Tue, 09 Jun 2026 12:00:00 GMT</pubDate></item>
    </channel></rss>"""
    client = FakeClient({url: _response(url, 200, content=xml)})

    result = await RssProvider(
        [url], name="sec_rss", label="SEC", source_kind="official", client=client
    ).fetch(_ctx())

    assert result.ok is True
    assert client.calls[0][0] == url
    assert result.news_items[0].source_id == "sec_rss"
    assert result.news_items[0].source_kind == "official"
    assert result.news_items[0].summary == "Policy details"


@pytest.mark.parametrize(
    ("provider_cls", "url", "payload", "title"),
    [
        (
            GuardianProvider,
            "https://content.guardianapis.com/search",
            {"response": {"results": [{"webTitle": "Bitcoin news", "webUrl": "https://g/1"}]}},
            "Bitcoin news",
        ),
        (
            GNewsProvider,
            "https://gnews.io/api/v4/search",
            {"articles": [{"title": "Ethereum news", "url": "https://g/2"}]},
            "Ethereum news",
        ),
        (
            NewsApiProvider,
            "https://newsapi.org/v2/everything",
            {"articles": [{"title": "Market news", "url": "https://g/3"}]},
            "Market news",
        ),
        (
            MediastackProvider,
            "https://api.mediastack.com/v1/news",
            {"data": [{"title": "Crypto news", "url": "https://g/4"}]},
            "Crypto news",
        ),
    ],
)
async def test_http_news_providers_return_discovery_items(
    provider_cls: type[Provider], url: str, payload: dict[str, Any], title: str
) -> None:
    client = FakeClient({url: _response(url, 200, payload=payload)})
    result = await provider_cls(client, "secret").fetch(_ctx())  # type: ignore[call-arg]

    assert result.ok is True
    assert result.news_items[0].title == title
    assert result.news_items[0].source_kind == "discovery"
    assert result.news_items[0].source_id == result.source
    assert "secret" not in str(client.calls[0][0])


async def test_http_provider_surfaces_rate_limit_without_retry() -> None:
    url = "https://gnews.io/api/v4/search"
    client = FakeClient({url: _response(url, 429)})
    result = await GNewsProvider(client, "secret").fetch(_ctx())

    assert result.ok is False
    assert result.error == "rate_limited: HTTP 429"
    assert len(client.calls) == 1


async def test_github_releases_filters_old_rows() -> None:
    url = "https://api.github.com/repos/bitcoin/bitcoin/releases"
    recent = datetime.now(timezone.utc) - timedelta(hours=2)
    old = datetime.now(timezone.utc) - timedelta(days=5)
    client = FakeClient(
        {
            url: _response(
                url,
                200,
                payload=[
                    {
                        "tag_name": "v1",
                        "name": "Recent",
                        "body": "release body",
                        "html_url": "https://github.com/bitcoin/bitcoin/releases/v1",
                        "published_at": recent.isoformat(),
                    },
                    {
                        "tag_name": "old",
                        "name": "Old",
                        "published_at": old.isoformat(),
                        "html_url": "https://github.com/bitcoin/bitcoin/releases/old",
                    },
                ],
            )
        }
    )
    result = await GithubReleasesProvider(client, ["bitcoin/bitcoin"]).fetch(_ctx())

    assert result.ok is True
    assert [item.title for item in result.news_items] == ["bitcoin/bitcoin Recent"]
    assert result.news_items[0].source_kind == "official"


class _NewsProvider(Provider):
    name = "duplicate_source"
    source_kind = "discovery"

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        return ProviderResult(
            source=self.name,
            news_items=[
                NewsItem(
                    title="Bitcoin ETF update",
                    url="https://same.example/article",
                    source="Duplicate",
                )
            ],
        )


async def test_dedup_retains_source_refs_and_counts_distinct_sources() -> None:
    first = _NewsProvider()
    second = _NewsProvider()
    second.name = "second_source"
    snap = await IntelAggregator([first, second]).gather("BTCUSDT")

    assert len(snap.news_items) == 1
    assert {ref.source_id for ref in snap.news_items[0].source_refs} == {
        "duplicate_source",
        "second_source",
    }


def test_discovery_item_does_not_upgrade_document_kind() -> None:
    item = NewsItem(
        title="Candidate headline",
        summary="Candidate summary",
        source_id="gnews_api",
        source_kind="discovery",
    )
    assert document_from_news_item(item).source_kind == "discovery"


def test_discovery_evidence_cannot_be_confirmed_in_source() -> None:
    document = document_from_news_item(
        NewsItem(
            title="Candidate headline",
            summary="Candidate summary",
            source_id="gnews_api",
            source_kind="discovery",
        )
    )
    research = _sanitize_research(
        {
            "tasks": [
                {
                    "task_id": "discovery-task",
                    "critical": False,
                    "facts": [
                        {
                            "fact_id": "fact-1",
                            "statement": "The candidate event was reported.",
                            "evidence_id": document.evidence_id,
                            "quote": document.text,
                            "confirmation_status": "confirmed_in_source",
                        }
                    ],
                    "document_coverage": [],
                }
            ]
        },
        [document],
    )
    fact = research["facts"][0]
    assert fact["source_kind"] == "discovery"
    assert fact["confirmation_status"] == "reported"


def test_new_config_and_legacy_rss_config_load_together(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[intel.sources]
sec_rss = true
github_releases = true
[intel.rss]
feeds = ["https://legacy.example/feed.xml"]
[[intel.rss.channels]]
id = "sec_rss"
label = "SEC"
source_kind = "official"
feeds = ["https://sec.example/feed.xml"]
[intel.github]
repositories = ["bitcoin/bitcoin"]
""",
        encoding="utf-8",
    )
    settings = load_settings(config)

    assert settings.intel.rss.feeds == ["https://legacy.example/feed.xml"]
    assert settings.intel.rss.channels[0].id == "sec_rss"
    assert settings.intel.github.repositories == ["bitcoin/bitcoin"]


def test_build_providers_disables_keyed_sources_without_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aggregator_mod, "resolve_secret", lambda _ref: None)
    settings = Settings(
        intel={
            "sources": {"guardian_api": True, "github_releases": False},
            "keys": {"guardian_api": "GUARDIAN_API_KEY"},
        }
    )
    providers, disabled = build_providers(settings, FakeClient({}), object())  # type: ignore[arg-type]

    assert "guardian_api" in disabled
    assert disabled["guardian_api"] == "缺少 key(GUARDIAN_API_KEY)"
    assert disabled["github_releases"] == "disabled by config"
    assert all(provider.name != "guardian_api" for provider in providers)
