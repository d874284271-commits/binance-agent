"""Named RSS providers with legacy URL-list compatibility."""

from __future__ import annotations

import asyncio
import calendar
import logging
import re
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx

from ..base import FetchContext, Provider, ProviderResult, row_result
from ..models import NewsItem, SourceKind

logger = logging.getLogger(__name__)
_TAG_RE = re.compile(r"<[^>]+>")
_XML_CONTENT_TYPES = {
    "application/atom+xml",
    "application/rdf+xml",
    "application/rss+xml",
    "application/xml",
    "text/xml",
}


def _clean(text: str) -> str:
    return _TAG_RE.sub("", text or "").strip()[:500]


def _published(entry: Any) -> datetime | None:
    struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if struct:
        return datetime.fromtimestamp(calendar.timegm(struct), tz=timezone.utc)
    return None


class RssProvider(Provider):
    supports_independent = True
    """Fetch one named channel; old tests can still pass raw XML as a feed."""

    name = "rss"
    source_kind: SourceKind = "media"

    def __init__(
        self,
        feeds: list[str],
        per_feed: int = 15,
        *,
        name: str = "rss",
        label: str = "RSS",
        source_kind: SourceKind = "media",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.label = label
        self.source_kind = source_kind
        self._feeds = feeds
        self._per_feed = per_feed
        self._client = client

    async def _parse_feed(self, feed: str) -> Any:
        # Raw XML is retained as a deterministic, network-free fixture path.
        if feed.lstrip().startswith(("<", "<?xml")) or "://" not in feed:
            return await asyncio.to_thread(feedparser.parse, feed)
        if self._client is None:
            parsed = await asyncio.to_thread(feedparser.parse, feed)
            self._validate_http_feed(parsed, "")
            return parsed
        response = await self._client.get(
            feed,
            headers={"User-Agent": "binance-agent/0.2 (+local analysis)"},
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"HTTP {response.status_code}")
        parsed = await asyncio.to_thread(feedparser.parse, response.content)
        self._validate_http_feed(parsed, response.headers.get("content-type", ""))
        return parsed

    @staticmethod
    def _validate_http_feed(parsed: Any, content_type: str) -> None:
        """Reject verification pages and malformed HTTP feeds deterministically.

        Raw XML fixtures intentionally bypass this check for backwards
        compatibility.  HTTP responses must parse as a feed; a valid but
        empty feed remains usable, while HTML pages and broken XML become a
        per-feed failure.
        """
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type.startswith(("text/html", "application/xhtml+xml")):
            raise RuntimeError("invalid_content_type")
        if getattr(parsed, "bozo", False):
            raise RuntimeError("invalid_feed")

        version = str(getattr(parsed, "version", "") or "").strip()
        entries = list(getattr(parsed, "entries", []) or [])
        valid_entries = any(str(entry.get("title", "")).strip() for entry in entries)
        if not version or (entries and not valid_entries):
            raise RuntimeError("invalid_feed")

        # A few feeds are mislabelled with another MIME type.  Keep those only
        # when feedparser proved that the payload has a real entry; a page with
        # no valid entries is treated as a verification or error response.
        if media_type and media_type not in _XML_CONTENT_TYPES and not valid_entries:
            raise RuntimeError("invalid_content_type")

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        """并发读取本源配置的 RSS,保留有效新闻及各 feed 的失败或坏行警告。

        所有 feed 均失败时整源失败,部分成功时继续返回内容;时间窗与跨源去重由聚合器处理。
        """
        parsed_list = await asyncio.gather(
            *(self._parse_feed(feed) for feed in self._feeds),
            return_exceptions=True,
        )
        items: list[NewsItem] = []
        successes = 0
        skipped = 0
        valid = 0
        failures: list[str] = []
        for feed, parsed in zip(self._feeds, parsed_list, strict=False):
            if isinstance(parsed, BaseException):
                failures.append(str(parsed)[:120])
                logger.debug("RSS 源失败 %s: %s", feed, parsed)
                continue
            successes += 1
            feed_title = self.label
            if getattr(parsed, "feed", None):
                feed_title = parsed.feed.get("title", self.label)
            for entry in parsed.entries[: self._per_feed]:
                try:
                    title = entry.get("title", "")
                    if not title:
                        continue
                    items.append(
                        NewsItem(
                            title=title,
                            summary=_clean(entry.get("summary", entry.get("description", ""))),
                            source=feed_title,
                            url=entry.get("link", ""),
                            source_id=self.name,
                            source_kind=self.source_kind,
                            published_at=_published(entry),
                        )
                    )
                    valid += 1
                except (AttributeError, TypeError, ValueError, OverflowError):
                    skipped += 1
        if self._feeds and successes == 0:
            return ProviderResult(
                source=self.name,
                ok=False,
                error="; ".join(failures) or "feed_unavailable",
            )
        result = row_result(self.name, skipped, valid, news_items=items)
        result.warnings.extend(failures)
        return result
