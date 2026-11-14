"""Stable source metadata shared by providers, config APIs and the UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import SourceKind

ProviderKind = Literal["builtin", "rss", "github", "api"]


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    label: str
    source_kind: SourceKind
    provider_kind: ProviderKind
    key_ref: str | None = None
    apply_url: str | None = None


SOURCE_SPECS: dict[str, SourceSpec] = {
    "fear_greed": SourceSpec("fear_greed", "Fear & Greed", "community", "builtin"),
    "reddit": SourceSpec("reddit", "Reddit", "community", "builtin"),
    "binance_deriv": SourceSpec("binance_deriv", "Binance Derivatives", "official", "builtin"),
    "rss": SourceSpec("rss", "RSS (legacy)", "media", "rss"),
    "gdelt": SourceSpec("gdelt", "GDELT", "discovery", "builtin"),
    "cryptopanic": SourceSpec(
        "cryptopanic",
        "CryptoPanic",
        "discovery",
        "builtin",
        "CRYPTOPANIC_TOKEN",
        "https://cryptopanic.com/developers/api",
    ),
    "finnhub": SourceSpec(
        "finnhub", "Finnhub", "discovery", "builtin", "FINNHUB_KEY", "https://finnhub.io/dashboard"
    ),
    "fred": SourceSpec(
        "fred",
        "FRED",
        "official",
        "builtin",
        "FRED_KEY",
        "https://fred.stlouisfed.org/docs/api/api_key.html",
    ),
    "akshare_cn": SourceSpec("akshare_cn", "AkShare CN", "media", "builtin"),
    "sec_rss": SourceSpec("sec_rss", "SEC", "official", "rss"),
    "cftc_rss": SourceSpec("cftc_rss", "CFTC", "official", "rss"),
    "federal_reserve_rss": SourceSpec("federal_reserve_rss", "Federal Reserve", "official", "rss"),
    "ecb_rss": SourceSpec("ecb_rss", "ECB", "official", "rss"),
    "ethereum_foundation_rss": SourceSpec(
        "ethereum_foundation_rss", "Ethereum Foundation", "official", "rss"
    ),
    "github_releases": SourceSpec("github_releases", "GitHub Releases", "official", "github"),
    "guardian_api": SourceSpec(
        "guardian_api",
        "The Guardian",
        "discovery",
        "api",
        "GUARDIAN_API_KEY",
        "https://open-platform.theguardian.com/access/",
    ),
    "gnews_api": SourceSpec(
        "gnews_api", "GNews", "discovery", "api", "GNEWS_API_KEY", "https://gnews.io/"
    ),
    "newsapi": SourceSpec(
        "newsapi", "NewsAPI", "discovery", "api", "NEWSAPI_KEY", "https://newsapi.org/pricing"
    ),
    "mediastack": SourceSpec(
        "mediastack",
        "Mediastack",
        "discovery",
        "api",
        "MEDIASTACK_ACCESS_KEY",
        "https://mediastack.com/pricing",
    ),
}


def source_spec(source_id: str) -> SourceSpec | None:
    return SOURCE_SPECS.get(source_id)


def source_catalog(sources: dict[str, bool], keys: dict[str, str]) -> list[dict[str, object]]:
    """为 /config 与桌面设置生成来源目录,只返回密钥引用名,不读取或返回密钥值。"""
    rows: list[dict[str, object]] = []
    for spec in SOURCE_SPECS.values():
        rows.append(
            {
                "source": spec.source_id,
                "label": spec.label,
                "source_kind": spec.source_kind,
                "provider_kind": spec.provider_kind,
                "enabled": sources.get(spec.source_id, True),
                "requires_key": spec.key_ref is not None,
                "key_ref": keys.get(spec.source_id, spec.key_ref),
                "apply_url": spec.apply_url,
            }
        )
    return rows
