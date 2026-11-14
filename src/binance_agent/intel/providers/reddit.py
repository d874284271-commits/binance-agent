"""Reddit 社媒情绪 —— 无需 key 的公开 .json 端点(子版搜索)。

按资产名在指定 subreddit(默认 r/CryptoCurrency)搜索近一日热帖,产出新闻条目交 LLM 情绪分析,
并汇总热度特征(帖数/总点赞/总评论)。Reddit 公开 .json 需合规 User-Agent(由注入的 client 提供)。
"""

from __future__ import annotations

from datetime import datetime, timezone
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

_BASE = "https://www.reddit.com"


class RedditProvider(Provider):
    name = "reddit"
    source_kind = "community"

    def __init__(self, client: httpx.AsyncClient, subreddit: str = "CryptoCurrency") -> None:
        self._client = client
        self._subreddit = subreddit

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        """搜索指定子版近一日的资产相关帖子,汇总有效帖的分数与评论数。

        403 明确记为公开端点受阻;部分坏行保留警告,社区热度不当作价格或事实证据。
        """
        resp = await self._client.get(
            f"{_BASE}/r/{self._subreddit}/search.json",
            params={
                "q": ctx.search_name,
                "restrict_sr": 1,
                "sort": "new",
                "t": "day",
                "limit": 25,
                "raw_json": 1,
            },
        )
        if resp.status_code == 403:
            return ProviderResult(
                source=self.name, ok=False, error="blocked_public_endpoint: HTTP 403"
            )
        if resp.status_code != 200:
            return ProviderResult(source=self.name, ok=False, error=f"HTTP {resp.status_code}")
        payload = parse_json_response(resp)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            return ProviderResult(source=self.name, ok=False, error="invalid_payload")
        children = payload["data"].get("children") or []
        if not isinstance(children, list):
            return ProviderResult(source=self.name, ok=False, error="invalid_payload")

        items: list[NewsItem] = []
        total_score = total_comments = 0
        skipped = 0
        valid = 0
        for child in children:
            try:
                post: dict[str, Any] = child.get("data") or {}
                title = text_field(post.get("title", ""))
                if not title.strip():
                    raise ValueError("missing_title")
                score = int(finite_number(post.get("score", 0) or 0))
                comments = int(finite_number(post.get("num_comments", 0) or 0))
                permalink = post.get("permalink") or ""
                items.append(
                    NewsItem(
                        title=title,
                        source=f"r/{post.get('subreddit', self._subreddit)}",
                        url=f"{_BASE}{permalink}" if permalink else post.get("url", ""),
                        published_at=datetime.fromtimestamp(
                            finite_number(post["created_utc"]), tz=timezone.utc
                        ),
                    )
                )
                total_score += score
                total_comments += comments
                valid += 1
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError, OSError):
                skipped += 1
        return row_result(
            self.name,
            skipped,
            valid,
            news_items=items,
            sentiment_features={
                "reddit_buzz": {
                    "posts": len(items),
                    "total_score": total_score,
                    "total_comments": total_comments,
                }
            },
        )
