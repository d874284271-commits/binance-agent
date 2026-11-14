"""资讯多源并发聚合 —— 容错 + 去重 + 时间窗 → IntelSnapshot。

工厂 ``build_providers`` 据配置/密钥/库可用性构建启用的 provider,并记录禁用原因;
``IntelAggregator`` 并发拉取、单源超时/异常隔离、按 URL/标题去重、按时间窗过滤新闻。
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone

import httpx

from ..infra.logging import redact_text
from ..infra.secrets import resolve_secret
from ..infra.settings import Settings
from ..market.rest_client import BinanceRestClient
from .base import FetchContext, Provider, ProviderResult, split_base_asset
from .evidence import select_news_candidates
from .models import IntelSnapshot, NewsItem, SourceRef
from .providers import (
    AkshareCnProvider,
    BinanceDerivProvider,
    CryptoPanicProvider,
    FearGreedProvider,
    FinnhubProvider,
    FredProvider,
    GdeltProvider,
    GithubReleasesProvider,
    GNewsProvider,
    GuardianProvider,
    MediastackProvider,
    NewsApiProvider,
    RedditProvider,
    RssProvider,
)

logger = logging.getLogger(__name__)

_NORM_RE = re.compile(r"[^a-z0-9一-鿿]+")
# 规范化标题短于该长度时不参与标题去重(中文信息密度高,8 字已具区分度)
_MIN_TITLE_DEDUP_LEN = 8


def _norm_title(title: str) -> str:
    return _NORM_RE.sub("", title.lower())[:120]


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def build_providers(
    settings: Settings,
    http_client: httpx.AsyncClient,
    rest_client: BinanceRestClient,
) -> tuple[list[Provider], dict[str, str]]:
    """据配置构建启用的 provider 列表 + 禁用原因(用于 source_status 展示)。"""
    sources = settings.intel.sources
    keys = settings.intel.keys
    providers: list[Provider] = []
    disabled: dict[str, str] = {}

    def enabled(name: str) -> bool:
        return sources.get(name, True)

    if enabled("fear_greed"):
        providers.append(FearGreedProvider(http_client))
    else:
        disabled["fear_greed"] = "disabled by config"

    if enabled("reddit"):
        providers.append(RedditProvider(http_client))
    else:
        disabled["reddit"] = "disabled by config"

    if enabled("binance_deriv"):
        providers.append(BinanceDerivProvider(rest_client))
    else:
        disabled["binance_deriv"] = "disabled by config"

    if enabled("rss"):
        feeds = settings.intel.rss.feeds
        if feeds:
            providers.append(RssProvider(feeds, client=http_client))
        else:
            disabled["rss"] = "未配置 RSS feeds"

    for channel in settings.intel.rss.channels:
        if not enabled(channel.id):
            disabled[channel.id] = "disabled by config"
        elif not channel.feeds:
            disabled[channel.id] = "未配置 RSS feeds"
        else:
            providers.append(
                RssProvider(
                    channel.feeds,
                    per_feed=channel.per_feed,
                    name=channel.id,
                    label=channel.label or channel.id,
                    source_kind=channel.source_kind,
                    client=http_client,
                )
            )

    if enabled("gdelt"):
        providers.append(GdeltProvider(http_client))
    else:
        disabled["gdelt"] = "disabled by config"

    if enabled("cryptopanic"):
        token = resolve_secret(keys.get("cryptopanic"))
        if token:
            providers.append(CryptoPanicProvider(http_client, token))
        else:
            disabled["cryptopanic"] = "缺少 key(CRYPTOPANIC_TOKEN)"

    if enabled("finnhub"):
        key = resolve_secret(keys.get("finnhub"))
        if key:
            providers.append(FinnhubProvider(http_client, key))
        else:
            disabled["finnhub"] = "缺少 key(FINNHUB_KEY)"

    if enabled("fred"):
        key = resolve_secret(keys.get("fred"))
        if key:
            providers.append(FredProvider(http_client, key))
        else:
            disabled["fred"] = "缺少 key(FRED_KEY)"

    if enabled("akshare_cn"):
        if importlib.util.find_spec("akshare") is not None:
            providers.append(AkshareCnProvider())
        else:
            disabled["akshare_cn"] = "akshare 未安装(uv sync --extra intel-cn)"
    else:
        disabled["akshare_cn"] = "disabled by config"

    if enabled("github_releases"):
        github = settings.intel.github
        if github.repositories:
            providers.append(
                GithubReleasesProvider(
                    http_client,
                    github.repositories,
                    per_repo=github.per_repo,
                    max_age_hours=github.max_age_hours,
                )
            )
        else:
            disabled["github_releases"] = "未配置 GitHub repositories"
    else:
        disabled["github_releases"] = "disabled by config"

    api_providers: dict[str, Callable[[httpx.AsyncClient, str], Provider]] = {
        "guardian_api": GuardianProvider,
        "gnews_api": GNewsProvider,
        "newsapi": NewsApiProvider,
        "mediastack": MediastackProvider,
    }
    for name, provider_cls in api_providers.items():
        if not enabled(name):
            disabled[name] = "disabled by config"
            continue
        key = resolve_secret(keys.get(name))
        if not key:
            ref = keys.get(name, name.upper())
            disabled[name] = f"缺少 key({ref})"
            continue
        providers.append(provider_cls(http_client, key))

    return providers, disabled


class IntelAggregator:
    def __init__(
        self,
        providers: list[Provider],
        *,
        per_source_timeout: float = 8.0,
        window_hours: int = 24,
        max_news_items: int = 40,
        retain_syndication: bool = False,
        disabled: dict[str, str] | None = None,
    ) -> None:
        self._providers = providers
        self._timeout = per_source_timeout
        self._window_hours = window_hours
        self._max_news = max_news_items
        self._retain_syndication = retain_syndication
        self._disabled = disabled or {}

    async def gather(
        self,
        symbol: str | None,
        *,
        research_question: str = "",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> IntelSnapshot:
        """并发收集启用来源并返回统一资讯快照,单源失败保留状态而不丢弃其他结果。

        来源元数据与坏行警告随结果保留,新闻经时间窗、去重和问题相关性排序截取,
        宏观事件按时间排序;此处不把新闻线索升级为已抓取正文或核验事实。
        """
        ctx = FetchContext(
            symbol=symbol.upper() if symbol else None,
            base_asset=split_base_asset(symbol) if symbol else None,
            question=research_question,
            window_hours=self._window_hours,
            start_date=start_date,
            end_date=end_date,
        )
        snapshot = IntelSnapshot(symbol=ctx.symbol)
        snapshot.source_status.update(self._disabled)

        selected = [p for p in self._providers if symbol is not None or p.supports_independent]
        for provider in self._providers:
            if provider not in selected:
                snapshot.source_status[provider.name] = "not_applicable: requires asset"
        results = await asyncio.gather(*(self._run(provider, ctx) for provider in selected))
        providers_by_name = {provider.name: provider for provider in self._providers}
        for result in results:
            snapshot.source_status[result.source] = "ok" if result.ok else f"error: {result.error}"
            if result.warnings:
                snapshot.source_warnings[result.source] = result.warnings
            if not result.ok:
                continue
            self._stamp_source_metadata(result, providers_by_name.get(result.source))
            snapshot.sentiment_features.update(result.sentiment_features)
            snapshot.news_items.extend(result.news_items)
            snapshot.macro_events.extend(result.macro_events)

        snapshot.news_items = self._dedup_and_window(
            snapshot.news_items,
            asset=ctx.symbol or "",
            question=research_question,
            start_date=start_date,
            end_date=end_date,
        )
        snapshot.macro_events.sort(key=lambda event: _aware(event.time))
        return snapshot

    @staticmethod
    def _stamp_source_metadata(result: ProviderResult, provider: Provider | None) -> None:
        source_kind = provider.source_kind if provider is not None else "unknown"
        for item in result.news_items:
            if not item.source_id:
                item.source_id = result.source
            if item.source_kind == "unknown":
                item.source_kind = source_kind
            if not item.source_refs:
                item.source_refs = [
                    SourceRef(source_id=item.source_id, source=item.source, url=item.url)
                ]

    async def _run(self, provider: Provider, ctx: FetchContext) -> ProviderResult:
        """在每源秒级时限内执行 fetch;普通异常转为失败结果,任务取消继续传播。"""
        try:
            return await asyncio.wait_for(provider.fetch(ctx), timeout=self._timeout)
        except (TimeoutError, asyncio.TimeoutError):
            return ProviderResult(source=provider.name, ok=False, error="timeout")
        except Exception as exc:  # 单源异常隔离,不阻断整体
            logger.warning("provider %s 失败: %s", provider.name, exc)
            # 异常消息可能内嵌带 key 的请求 URL(如 httpx.HTTPStatusError),先脱敏再截断
            return ProviderResult(source=provider.name, ok=False, error=redact_text(str(exc))[:200])

    def _dedup_and_window(
        self,
        items: list[NewsItem],
        *,
        asset: str = "",
        question: str = "",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[NewsItem]:
        """剔除已知过期新闻,合并重复来源引用后按资产和问题相关性截取候选。

        未知发布时间保留;同 URL 去重始终启用,标题去重由 retain_syndication 控制。
        会修改保留新闻的 source_refs 与 relevance,不以“更新”标题挤掉更相关旧线索。
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._window_hours)
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        canonical_by_url: dict[str, NewsItem] = {}
        canonical_by_title: dict[str, NewsItem] = {}
        out: list[NewsItem] = []
        for item in sorted(
            items,
            key=lambda n: (
                _aware(n.published_at)
                if n.published_at
                else datetime.min.replace(tzinfo=timezone.utc)
            ),
            reverse=True,
        ):
            if item.published_at is not None:
                published = _aware(item.published_at).astimezone(timezone.utc)
                if start_date is not None or end_date is not None:
                    if start_date is not None and published.date() < start_date:
                        continue
                    if end_date is not None and published.date() > end_date:
                        continue
                elif published < cutoff:
                    continue
            url = item.url.strip()
            norm = _norm_title(item.title)
            # 过短的规范化标题(如 "btc"、"比特币")区分度不足,只按 URL 去重,避免误伤不同文章
            if len(norm) < _MIN_TITLE_DEDUP_LEN:
                norm = ""
            duplicate = canonical_by_url.get(url) if url else None
            if duplicate is None and not self._retain_syndication and norm:
                duplicate = canonical_by_title.get(norm)
            if duplicate is not None:
                self._merge_source_refs(duplicate, item)
                continue
            if url:
                seen_urls.add(url)
                canonical_by_url[url] = item
            if norm:
                seen_titles.add(norm)
                canonical_by_title[norm] = item
            if not item.source_refs:
                item.source_refs = [
                    SourceRef(source_id=item.source_id, source=item.source, url=item.url)
                ]
            out.append(item)
        # Apply the candidate cap only after URL/time de-duplication and asset
        # relevance ranking.  A recent batch of unrelated headlines must not
        # evict an older but directly relevant target-asset event.
        return (
            select_news_candidates(out, asset=asset, question=question, limit=self._max_news)
            if asset or question
            else out[: self._max_news]
        )

    @staticmethod
    def _merge_source_refs(canonical: NewsItem, duplicate: NewsItem) -> None:
        refs = duplicate.source_refs or [
            SourceRef(source_id=duplicate.source_id, source=duplicate.source, url=duplicate.url)
        ]
        known = {(ref.source_id, ref.source, ref.url) for ref in canonical.source_refs}
        for ref in refs:
            key = (ref.source_id, ref.source, ref.url)
            if key not in known:
                canonical.source_refs.append(ref)
                known.add(key)
