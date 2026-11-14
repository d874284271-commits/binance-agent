"""GDELT DOC 2.0 —— 无 key,全球新闻语气(Tone, -100~100)+ 代表文章。

GDELT 免费档限「每 5 秒 1 次请求」,故仅发 1 次 ToneChart 调用:
既得平均语气(数值特征),又从各 tone bin 的 toparts 取代表文章(按 bin 语气赋倾向)。
"""

from __future__ import annotations

import logging
import time
from typing import Any

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

logger = logging.getLogger(__name__)
_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"


def _sentiment_for(tone: float) -> str:
    if tone > 1:
        return "bull"
    if tone < -1:
        return "bear"
    return "neutral"


class GdeltProvider(Provider):
    supports_independent = True
    name = "gdelt"
    source_kind = "discovery"

    def __init__(self, client: httpx.AsyncClient, max_records: int = 20) -> None:
        self._client = client
        self._max = max_records
        self._rate_limited_until = 0.0

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        """用一次 ToneChart 请求取得数量加权语气及代表文章,不额外抓取正文。

        429 后本实例冷却 5 秒并显式失败;坏分箱/文章隔离,未知发布时间保留 None。
        """
        now_mono = time.monotonic()
        if now_mono < self._rate_limited_until:
            return ProviderResult(source=self.name, ok=False, error="rate_limited_cooldown")

        query = (
            ctx.question or f'("{ctx.search_name}" OR {ctx.base_asset}) (cryptocurrency OR crypto)'
        )
        time_params = {"timespan": f"{max(1, ctx.window_hours)}h"}
        if ctx.start_date is not None or ctx.end_date is not None:
            time_params = {}
            if ctx.start_date is not None:
                time_params["startdatetime"] = ctx.start_date.strftime("%Y%m%d") + "000000"
            if ctx.end_date is not None:
                time_params["enddatetime"] = ctx.end_date.strftime("%Y%m%d") + "235959"

        resp = await self._client.get(
            _DOC,
            params={
                "query": query,
                "mode": "ToneChart",
                "format": "json",
                **time_params,
            },
        )
        # 免费档限「每 5 秒 1 次」,429/非200 显式降级为失败,避免静默吞错虚高 source_success_rate
        if resp.status_code == 429:
            self._rate_limited_until = time.monotonic() + 5.0
            return ProviderResult(source=self.name, ok=False, error="rate_limited: HTTP 429")
        if resp.status_code != 200:
            return ProviderResult(source=self.name, ok=False, error=f"HTTP {resp.status_code}")
        payload = parse_json_response(resp)
        if not isinstance(payload, dict):
            return ProviderResult(source=self.name, ok=False, error="json_parse_error")
        bins = payload.get("tonechart") or []

        if not isinstance(bins, list):
            return ProviderResult(source=self.name, ok=False, error="invalid_payload")
        features: dict[str, Any] = {}
        valid_bins: list[dict[str, Any]] = []
        weighted = total = 0.0
        skipped = 0
        for tone_bin in bins:
            try:
                tone = finite_number(tone_bin["bin"])
                count = finite_number(tone_bin["count"])
                articles = tone_bin.get("toparts") or []
                if count < 0 or not isinstance(articles, list):
                    raise ValueError("invalid_bin")
                valid_bins.append({"tone": tone, "articles": articles})
                weighted += tone * count
                total += count
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                skipped += 1
        if total:
            features["gdelt_tone"] = round(weighted / total, 2)
        items: list[NewsItem] = []
        seen: set[str] = set()
        valid_articles = 0
        for tone_bin in valid_bins:
            sentiment = _sentiment_for(tone_bin["tone"])
            for article in tone_bin["articles"][:3]:
                try:
                    url = text_field(article.get("url", ""))
                    title = text_field(article.get("title", ""))
                    if not title or (url and url in seen):
                        continue
                    item = NewsItem(
                        title=title, source="GDELT", url=url, published_at=None, sentiment=sentiment
                    )
                    items.append(item)
                    valid_articles += 1
                    if url:
                        seen.add(url)
                except (AttributeError, TypeError, ValueError):
                    skipped += 1
                if len(items) >= self._max:
                    break
            if len(items) >= self._max:
                break
        return row_result(
            self.name,
            skipped,
            len(valid_bins) + valid_articles,
            sentiment_features=features,
            news_items=items,
        )
