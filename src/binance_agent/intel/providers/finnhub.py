"""Finnhub —— 免费 key(60 req/min),category=crypto 全球加密新闻。"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from ..base import (
    FetchContext,
    Provider,
    ProviderResult,
    finite_number,
    parse_json_response,
    row_result,
    text_field,
)
from ..models import NewsItem

_URL = "https://finnhub.io/api/v1/news"


class FinnhubProvider(Provider):
    name = "finnhub"
    source_kind = "discovery"

    def __init__(self, client: httpx.AsyncClient, key: str, limit: int = 25) -> None:
        self._client = client
        self._key = key
        self._limit = limit

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        """采集 crypto 新闻列表,按本源条数上限截取并将 Unix 秒时间转为 UTC。

        坏行跳过并记录警告;标题摘要只是线索,正文由后续证据抓取流程处理。
        """
        resp = await self._client.get(_URL, params={"category": "crypto", "token": self._key})
        if resp.status_code != 200:
            return ProviderResult(source=self.name, ok=False, error=f"HTTP {resp.status_code}")
        rows = parse_json_response(resp)
        if not isinstance(rows, list):
            return ProviderResult(source=self.name, ok=False, error="json_parse_error")

        items: list[NewsItem] = []
        skipped = 0
        valid = 0
        for row in rows[: self._limit]:
            try:
                headline = text_field(row.get("headline", ""))
                if not headline.strip():
                    raise ValueError("missing_title")
                ts = finite_number(row["datetime"])
                published = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
                items.append(
                    NewsItem(
                        title=headline,
                        summary=text_field(row.get("summary", "") or "")[:500],
                        source=row.get("source", "Finnhub"),
                        url=row.get("url", ""),
                        published_at=published,
                    )
                )
                valid += 1
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError, OSError):
                skipped += 1
        return row_result(self.name, skipped, valid, news_items=items)
