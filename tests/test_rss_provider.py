"""RSS provider 离线测试:XML 解析 + HTML 清洗 + 单源异常隔离(feedparser 可直接吃字符串)。"""

from __future__ import annotations

import httpx
import pytest

from binance_agent.intel.base import FetchContext
from binance_agent.intel.providers import rss as rss_mod
from binance_agent.intel.providers.rss import RssProvider

_FEED_XML = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Crypto Feed</title>
    <item>
      <title>Bitcoin rallies past resistance</title>
      <description>&lt;p&gt;Strong &lt;b&gt;momentum&lt;/b&gt; today.&lt;/p&gt;</description>
      <link>https://example.com/rally</link>
      <pubDate>Tue, 09 Jun 2026 12:00:00 GMT</pubDate>
    </item>
    <item>
      <title></title>
      <description>untitled entry skipped</description>
    </item>
  </channel>
</rss>
"""


def _ctx() -> FetchContext:
    return FetchContext(symbol="BTCUSDT", base_asset="BTC", window_hours=24)


class _HttpClient:
    def __init__(self, responses: dict[str, httpx.Response]) -> None:
        self.responses = responses

    async def get(self, url: str, **kwargs: object) -> httpx.Response:
        return self.responses[url]


async def test_rss_parses_entries_and_strips_html() -> None:
    # feedparser.parse 接受原始 XML 字符串,无需联网
    result = await RssProvider([_FEED_XML]).fetch(_ctx())

    assert result.ok is True
    assert len(result.news_items) == 1
    item = result.news_items[0]
    assert item.title == "Bitcoin rallies past resistance"
    assert item.source == "Test Crypto Feed"
    assert item.summary == "Strong momentum today."  # HTML 标签已清洗
    assert item.published_at.year == 2026


async def test_rss_malformed_feed_yields_no_items() -> None:
    result = await RssProvider(["%%% not xml at all %%%"]).fetch(_ctx())

    assert result.ok is True  # RSS 无 key/无状态码,坏源等价于空源
    assert result.news_items == []


async def test_rss_single_feed_exception_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    real_parse = rss_mod.feedparser.parse

    def flaky_parse(url: str):  # type: ignore[no-untyped-def]
        if url == "BOOM":
            raise RuntimeError("network down")
        return real_parse(url)

    monkeypatch.setattr(rss_mod.feedparser, "parse", flaky_parse)
    result = await RssProvider(["BOOM", _FEED_XML]).fetch(_ctx())

    assert result.ok is True
    assert [n.title for n in result.news_items] == ["Bitcoin rallies past resistance"]


async def test_http_verification_page_is_reported_as_feed_failure() -> None:
    url = "https://example.com/feed.xml"
    client = _HttpClient(
        {
            url: httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                content=b"<html><body>checking your browser</body></html>",
                request=httpx.Request("GET", url),
            )
        }
    )

    result = await RssProvider([url], client=client).fetch(_ctx())

    assert result.ok is False
    assert result.error == "invalid_content_type"
    assert result.news_items == []


async def test_http_broken_xml_isolated_from_valid_feed() -> None:
    broken_url = "https://broken.example/feed.xml"
    valid_url = "https://valid.example/feed.xml"
    client = _HttpClient(
        {
            broken_url: httpx.Response(
                200,
                headers={"content-type": "application/rss+xml"},
                content=b"<?xml version='1.0'?><rss><channel><title>broken",
                request=httpx.Request("GET", broken_url),
            ),
            valid_url: httpx.Response(
                200,
                headers={"content-type": "application/rss+xml"},
                content=_FEED_XML.encode(),
                request=httpx.Request("GET", valid_url),
            ),
        }
    )

    result = await RssProvider([broken_url, valid_url], client=client).fetch(_ctx())

    assert result.ok is True
    assert [item.title for item in result.news_items] == ["Bitcoin rallies past resistance"]


async def test_upgraded_parser_preserves_atom_unicode_and_sanitizes_xhtml() -> None:
    """Exercise the replacement SGML parser through feedparser's HTML sanitizer."""
    atom = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"><title>公告</title>
    <entry><id>urn:release:1</id><title>市场 &amp; 政策</title>
    <link href="https://example.com/policy"/>
    <updated>2026-11-14T10:00:00+08:00</updated>
    <summary type="html"><![CDATA[<p>中文 &amp; policy <b>update</b>
    <script>unsafe_script()</script></p>]]></summary></entry></feed>"""
    result = await RssProvider([atom]).fetch(_ctx())
    assert result.ok
    assert len(result.news_items) == 1
    item = result.news_items[0]
    assert item.title == "市场 & 政策"
    assert "中文" in item.summary and "policy" in item.summary and "update" in item.summary
    assert "unsafe_script" not in item.summary
    assert "<" not in item.summary
    assert item.published_at.hour == 2


async def test_upgraded_parser_recovers_malformed_html_without_losing_valid_feed() -> None:
    xml = _FEED_XML.replace(
        "&lt;p&gt;Strong &lt;b&gt;momentum&lt;/b&gt; today.&lt;/p&gt;",
        "<![CDATA[<p>Strong <b>momentum</p><img src='x' onerror='unsafe()'> today.]]>",
    )
    result = await RssProvider([xml]).fetch(_ctx())
    assert result.ok
    assert len(result.news_items) == 1
    assert "Strong" in result.news_items[0].summary
    assert "momentum" in result.news_items[0].summary
    assert "unsafe" not in result.news_items[0].summary
