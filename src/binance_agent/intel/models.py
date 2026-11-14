"""资讯情报层数据模型:新闻条目 / 宏观事件 / 聚合快照。

两类输出 → 不同消费方式:
- 自带分数 → sentiment_features 数值特征(无需 LLM)
- 原始文本 → news_items 交 LLM 情绪分析师
- 宏观事件 → macro_events 喂风险 agent(临近高重要性事件触发波动告警)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

SourceKind = Literal["official", "media", "community", "discovery", "unknown"]


class SourceRef(BaseModel):
    """Source attribution retained when duplicate items are merged."""

    source_id: str = ""
    source: str = ""
    url: str = ""


class Citation(BaseModel):
    """A claim-level reference to one saved content version."""

    citation_id: str
    evidence_id: str
    content_sha256: str = ""
    quote: str
    start: int | None = None
    end: int | None = None
    validation: Literal["unchecked", "valid", "invalid"] = "unchecked"


class Claim(BaseModel):
    """A downstream statement with an explicit type and evidence dependency."""

    claim_id: str
    statement: str
    claim_type: Literal["fact", "financial_inference", "community_sentiment"] = "fact"
    event_id: str | None = None
    fact_ids: list[str] = Field(default_factory=list)
    market_paths: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    support: Literal["supported", "partial", "unresolved"] = "unresolved"


class NewsItem(BaseModel):
    title: str
    summary: str = ""
    source: str = ""
    url: str = ""
    source_id: str = ""
    source_kind: SourceKind = "unknown"
    source_refs: list[SourceRef] = Field(default_factory=list)
    # Some feeds do not expose a publication time.  It must stay unknown instead
    # of being silently replaced with the collection time.
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sentiment: str | None = None  # 源自带情绪(bull/bear/neutral)则填,否则交 LLM
    relevance: Literal["direct", "market", "weak"] | None = None
    event_id: str | None = None
    content_status: Literal["not_attempted", "available", "unavailable", "failed"] = "not_attempted"
    body_evidence_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceDocument(BaseModel):
    """An immutable, run-local snapshot of one externally supplied text.

    ``evidence_id`` is derived from the stable source identity and text version,
    so replay refers to the original captured material rather than today's URL.
    """

    evidence_id: str
    url: str = ""
    source: str = ""
    title: str = ""
    text: str = ""
    links: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: Literal["title", "summary", "body"] = "title"
    status: Literal["available", "unavailable", "failed"] = "available"
    fetch_error: str | None = None
    source_kind: SourceKind = "unknown"
    content_sha256: str = ""
    # Dates found in captured text are kept separate from provider metadata;
    # extraction never silently replaces ``published_at``.
    extracted_dates: list[str] = Field(default_factory=list)


class FactCard(BaseModel):
    """A claim-sized record whose quote can be checked without an LLM."""

    fact_id: str
    event_id: str | None = None
    task_id: str | None = None
    statement: str
    subject: str = ""
    action: str = ""
    event_time: str | None = None
    scope: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    confirmation_status: Literal[
        "reported", "confirmed_in_source", "unconfirmed", "disputed", "unknown"
    ] = "unknown"
    evidence_id: str
    quote: str
    support: Literal["supported", "model_supported", "partial", "unresolved"] = "unresolved"
    uncertainty: list[str] = Field(default_factory=list)
    source_kind: SourceKind = "unknown"
    claim_type: Literal["fact", "financial_inference", "community_sentiment"] = "fact"
    citations: list[Citation] = Field(default_factory=list)


class MacroEvent(BaseModel):
    name: str
    time: datetime
    importance: int = 1  # 1-3 星
    forecast: str | None = None
    previous: str | None = None
    actual: str | None = None
    source: str = ""


class IntelSnapshot(BaseModel):
    symbol: str | None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sentiment_features: dict[str, Any] = Field(default_factory=dict)
    news_items: list[NewsItem] = Field(default_factory=list)
    macro_events: list[MacroEvent] = Field(default_factory=list)
    # provider 名 -> "ok" 或错误摘要,用于可观测性与降级展示
    source_status: dict[str, str] = Field(default_factory=dict)
    source_warnings: dict[str, list[str]] = Field(default_factory=dict)
