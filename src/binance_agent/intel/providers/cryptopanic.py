"""CryptoPanic —— 免费 token,自带社区投票(bull/bear),filter=hot。"""

from __future__ import annotations

from datetime import datetime
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

_URL = "https://cryptopanic.com/api/v1/posts/"


def _parse_iso(value: str | None) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return datetime.fromisoformat(text_field(value).replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("invalid_datetime") from None


class CryptoPanicProvider(Provider):
    name = "cryptopanic"
    source_kind = "discovery"

    def __init__(self, client: httpx.AsyncClient, token: str) -> None:
        self._client = client
        self._token = token

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        """按基础资产获取热门帖子及社区投票,汇总有效帖子中的多空票数。

        非 200 或响应结构错误显式失败;坏行隔离,投票只作情绪线索,不表示独立事实核验。
        """
        resp = await self._client.get(
            _URL,
            params={
                "auth_token": self._token,
                "currencies": ctx.base_asset,
                "filter": "hot",
                "public": "true",
            },
        )
        if resp.status_code != 200:
            return ProviderResult(source=self.name, ok=False, error=f"HTTP {resp.status_code}")
        payload = parse_json_response(resp)
        if not isinstance(payload, dict):
            return ProviderResult(source=self.name, ok=False, error="json_parse_error")
        posts = payload.get("results") or []
        if not isinstance(posts, list):
            return ProviderResult(source=self.name, ok=False, error="invalid_payload")

        items: list[NewsItem] = []
        positive = negative = 0
        skipped = 0
        valid = 0
        for post in posts:
            try:
                title = text_field(post.get("title", ""))
                if not title.strip():
                    raise ValueError("missing_title")
                votes: dict[str, Any] = post.get("votes") or {}
                pos = int(finite_number(votes.get("positive", 0)))
                neg = int(finite_number(votes.get("negative", 0)))
                sentiment = None
                if pos > neg:
                    sentiment = "bull"
                elif neg > pos:
                    sentiment = "bear"
                items.append(
                    NewsItem(
                        title=title,
                        source=(post.get("source") or {}).get("title", "CryptoPanic"),
                        url=post.get("url", ""),
                        published_at=_parse_iso(post.get("published_at")),
                        sentiment=sentiment,
                    )
                )
                positive += pos
                negative += neg
                valid += 1
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError, OSError):
                skipped += 1
        return row_result(
            self.name,
            skipped,
            valid,
            news_items=items,
            sentiment_features={
                "cryptopanic_votes": {
                    "positive": positive,
                    "negative": negative,
                    "posts": len(items),
                }
            },
        )
