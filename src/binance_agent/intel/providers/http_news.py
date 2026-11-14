"""HTTP 新闻与公开 GitHub Release 来源,统一返回短篇发现线索。

需要正文时继续使用现有 ContentFetcher;API 返回标题或摘要不代表已读取原文。
"""

from __future__ import annotations

import asyncio
import re
from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from ..base import (
    FetchContext,
    Provider,
    ProviderResult,
    parse_json_response,
    row_result,
    text_field,
)
from ..models import NewsItem, SourceKind

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(value: Any, limit: int = 500) -> str:
    return _TAG_RE.sub("", text_field(value or "")).strip()[:limit]


def _parse_datetime(value: Any) -> datetime | None:
    """解析 ISO 或邮件日期为带时区时间;无时区时按 UTC,缺失返回 None,非法值抛错。"""
    if value is None or value == "":
        return None
    raw = text_field(value).strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("invalid_datetime") from None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _error_for_response(response: httpx.Response) -> str:
    if response.status_code in {401, 403}:
        return f"unauthorized: HTTP {response.status_code}"
    if response.status_code == 429:
        return "rate_limited: HTTP 429"
    return f"HTTP {response.status_code}"


class HttpNewsProvider(Provider):
    supports_independent = True
    source_kind: SourceKind = "discovery"

    def __init__(self, client: httpx.AsyncClient, key: str, limit: int = 20) -> None:
        self._client = client
        self._key = key
        self._limit = limit

    @property
    @abstractmethod
    def endpoint(self) -> str: ...

    @abstractmethod
    def params(self, ctx: FetchContext) -> dict[str, Any]: ...

    @abstractmethod
    def rows(self, payload: dict[str, Any]) -> list[dict[str, Any]]: ...

    @abstractmethod
    def item_from_row(self, row: dict[str, Any]) -> NewsItem | None: ...

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        """请求子类定义的发现 API,校验响应结构并把有效行映射为新闻线索。

        单行错误不丢弃其他结果;鉴权、限速及响应错误以稳定类别返回,
        共享 client 的生命周期和整个来源的超时由应用及聚合器管理。
        """
        response = await self._client.get(self.endpoint, params=self.params(ctx))
        if response.status_code != 200:
            return ProviderResult(source=self.name, ok=False, error=_error_for_response(response))
        payload = parse_json_response(response)
        if not isinstance(payload, dict):
            return ProviderResult(source=self.name, ok=False, error="json_parse_error")
        try:
            rows = self.rows(payload)
        except (KeyError, TypeError, ValueError):
            return ProviderResult(source=self.name, ok=False, error="invalid_payload")
        if not isinstance(rows, list):
            return ProviderResult(source=self.name, ok=False, error="invalid_payload")
        items: list[NewsItem] = []
        skipped = 0
        valid = 0
        for row in rows[: self._limit]:
            try:
                item = self.item_from_row(row)
                if item is not None:
                    if not item.title.strip():
                        raise ValueError("missing_title")
                    items.append(item)
                valid += 1
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                skipped += 1
        return row_result(self.name, skipped, valid, news_items=items)


def _query(ctx: FetchContext) -> str:
    return ctx.question or f'"{ctx.search_name}" OR {ctx.base_asset} cryptocurrency'


class GuardianProvider(HttpNewsProvider):
    name = "guardian_api"
    endpoint = "https://content.guardianapis.com/search"

    def params(self, ctx: FetchContext) -> dict[str, Any]:
        return {
            "api-key": self._key,
            "q": _query(ctx),
            "order-by": "newest",
            "page-size": self._limit,
            "show-fields": "trailText",
            **({"from-date": ctx.start_date.isoformat()} if ctx.start_date else {}),
            **({"to-date": ctx.end_date.isoformat()} if ctx.end_date else {}),
        }

    def rows(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        response = payload.get("response") or {}
        return response.get("results") or []

    def item_from_row(self, row: dict[str, Any]) -> NewsItem | None:
        fields = row.get("fields") or {}
        return NewsItem(
            title=_clean(row.get("webTitle")),
            summary=_clean(fields.get("trailText")),
            source="The Guardian",
            url=text_field(row.get("webUrl") or ""),
            source_id=self.name,
            source_kind=self.source_kind,
            published_at=_parse_datetime(row.get("webPublicationDate")),
        )


class GNewsProvider(HttpNewsProvider):
    name = "gnews_api"
    endpoint = "https://gnews.io/api/v4/search"

    def params(self, ctx: FetchContext) -> dict[str, Any]:
        return {
            "apikey": self._key,
            "q": _query(ctx),
            "lang": "en",
            "max": self._limit,
            "sortby": "publishedAt",
            **({"from": ctx.start_date.isoformat() + "T00:00:00Z"} if ctx.start_date else {}),
            **({"to": ctx.end_date.isoformat() + "T23:59:59Z"} if ctx.end_date else {}),
        }

    def rows(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return payload.get("articles") or []

    def item_from_row(self, row: dict[str, Any]) -> NewsItem | None:
        return NewsItem(
            title=_clean(row.get("title")),
            summary=_clean(row.get("description")),
            source=str((row.get("source") or {}).get("name") or "GNews"),
            url=text_field(row.get("url") or ""),
            source_id=self.name,
            source_kind=self.source_kind,
            published_at=_parse_datetime(row.get("publishedAt")),
        )


class NewsApiProvider(HttpNewsProvider):
    name = "newsapi"
    endpoint = "https://newsapi.org/v2/everything"

    def params(self, ctx: FetchContext) -> dict[str, Any]:
        return {
            "apiKey": self._key,
            "q": _query(ctx),
            "sortBy": "publishedAt",
            **({"from": ctx.start_date.isoformat() + "T00:00:00Z"} if ctx.start_date else {}),
            **({"to": ctx.end_date.isoformat() + "T23:59:59Z"} if ctx.end_date else {}),
            "pageSize": self._limit,
            "language": "en",
        }

    def rows(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return payload.get("articles") or []

    def item_from_row(self, row: dict[str, Any]) -> NewsItem | None:
        title = _clean(row.get("title"))
        if not title or title == "[Removed]":
            return None
        return NewsItem(
            title=title,
            summary=_clean(row.get("description")),
            source=str((row.get("source") or {}).get("name") or "NewsAPI"),
            url=text_field(row.get("url") or ""),
            source_id=self.name,
            source_kind=self.source_kind,
            published_at=_parse_datetime(row.get("publishedAt")),
        )


class MediastackProvider(HttpNewsProvider):
    name = "mediastack"
    endpoint = "https://api.mediastack.com/v1/news"

    def params(self, ctx: FetchContext) -> dict[str, Any]:
        return {
            "access_key": self._key,
            "keywords": ctx.question or f"{ctx.search_name},{ctx.base_asset},cryptocurrency",
            "languages": "en",
            "limit": self._limit,
            "sort": "published_desc",
        }

    def rows(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return payload.get("data") or []

    def item_from_row(self, row: dict[str, Any]) -> NewsItem | None:
        return NewsItem(
            title=_clean(row.get("title")),
            summary=_clean(row.get("description")),
            source=str(row.get("source") or "Mediastack"),
            url=text_field(row.get("url") or ""),
            source_id=self.name,
            source_kind=self.source_kind,
            published_at=_parse_datetime(row.get("published_at")),
        )


class GithubReleasesProvider(Provider):
    name = "github_releases"
    source_kind: SourceKind = "official"

    def __init__(
        self,
        client: httpx.AsyncClient,
        repositories: list[str],
        *,
        per_repo: int = 5,
        max_age_hours: int = 168,
    ) -> None:
        self._client = client
        self._repositories = repositories
        self._per_repo = per_repo
        self._max_age_hours = max_age_hours

    async def _fetch_repo(
        self, repository: str
    ) -> tuple[str, list[dict[str, Any]] | None, str | None]:
        response = await self._client.get(
            f"https://api.github.com/repos/{repository}/releases",
            params={"per_page": self._per_repo},
            headers={"Accept": "application/vnd.github+json", "User-Agent": "binance-agent/0.2"},
        )
        if response.status_code != 200:
            return repository, None, _error_for_response(response)
        payload = parse_json_response(response)
        if not isinstance(payload, list):
            return repository, None, "json_parse_error"
        return repository, payload, None

    async def fetch(self, ctx: FetchContext) -> ProviderResult:
        """并发获取配置仓库的公开 Release,按来源时间窗和每仓库条数生成线索。

        部分仓库失败仍保留其他结果及警告;全部失败则整源失败,取消继续向上传播。
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=min(ctx.window_hours, self._max_age_hours)
        )
        results = await asyncio.gather(
            *(self._fetch_repo(repo) for repo in self._repositories), return_exceptions=True
        )
        items: list[NewsItem] = []
        failures: list[str] = []
        successes = 0
        skipped = 0
        valid = 0
        for result in results:
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                failures.append("repository_unavailable")
                continue
            repository, rows, error = result
            if error:
                failures.append(f"{repository}: {error}")
                continue
            successes += 1
            for row in (rows or [])[: self._per_repo]:
                try:
                    published = _parse_datetime(row.get("published_at") or row.get("created_at"))
                    if published is not None and published < cutoff:
                        valid += 1
                        continue
                    tag = _clean(row.get("tag_name"))
                    name = _clean(row.get("name")) or tag or "GitHub release"
                    title = f"{repository} {name}".strip()
                    items.append(
                        NewsItem(
                            title=title,
                            summary=_clean(row.get("body")),
                            source=f"GitHub:{repository}",
                            url=text_field(row.get("html_url") or ""),
                            source_id=self.name,
                            source_kind=self.source_kind,
                            published_at=published,
                        )
                    )
                    valid += 1
                except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                    skipped += 1
        if self._repositories and successes == 0:
            return ProviderResult(
                source=self.name,
                ok=False,
                error="; ".join(failures) or "github_unavailable",
            )
        outcome = row_result(self.name, skipped, valid, news_items=items)
        outcome.warnings.extend(failures)
        return outcome
