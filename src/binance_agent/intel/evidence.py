"""单次 run 的确定性证据工具。

引文字面匹配只说明引用指向已保存文本,不证明文本在语义上支持周围的金融结论。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .models import Citation, EvidenceDocument, FactCard, NewsItem

_WORD_RE = re.compile(r"[a-z0-9]{2,}|[\u4e00-\u9fff]{2,}", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?\b"
    r"|\b20\d{2}年\d{1,2}月\d{1,2}日\b"
)
_ALLOWED_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def make_evidence_id(*, url: str, source: str, title: str, text: str, level: str) -> str:
    """将 URL、来源、标题、层级和文本版本绑定为稳定 ID;正文变化会改变 ID。"""
    version = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = "\x1f".join((url.strip(), source.strip(), title.strip(), level, version))
    return "ev_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def evidence_document(
    *,
    url: str = "",
    source: str = "",
    title: str = "",
    text: str = "",
    links: list[str] | None = None,
    published_at: datetime | None = None,
    fetched_at: datetime | None = None,
    level: str = "title",
    status: str = "available",
    source_kind: str = "unknown",
) -> EvidenceDocument:
    """清理文本后建立证据对象,绑定内容 hash、稳定 ID 和文本中出现的日期。

    未提供 fetched_at 时使用当前 UTC;不抓取 URL,也不推断缺失的发布时间。
    """
    cleaned = _clean_text(text)
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return EvidenceDocument(
        evidence_id=make_evidence_id(
            url=url, source=source, title=title, text=cleaned, level=level
        ),
        url=url,
        source=source,
        title=title,
        text=cleaned,
        links=list(links or []),
        published_at=published_at,
        fetched_at=fetched_at or datetime.now(timezone.utc),
        level=level,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        source_kind=source_kind,  # type: ignore[arg-type]
        content_sha256=digest,
        extracted_dates=_extract_dates(cleaned),
    )


def select_news_candidates(
    items: list[NewsItem], *, asset: str, question: str = "", limit: int = 20
) -> list[NewsItem]:
    """按资产相关度、问题词重合数及已知发布时间排序,返回至多 limit 条候选。

    原地写入各 NewsItem 的 direct/market/weak 相关度;未知发布时间排在已知时间
    之后,不用抓取时间替代。排序只筛选线索,不确认新闻真伪。
    """
    asset_tokens = _asset_tokens(asset)
    question_tokens = set(_tokens(question))
    ranked: list[tuple[int, float, str, NewsItem]] = []
    for item in items:
        haystack = f"{item.title} {item.summary}".lower()
        words = set(_tokens(haystack))
        direct = bool(words & asset_tokens) if asset else bool(words & question_tokens)
        market = bool(words & _MARKET_TOKENS) or any(
            phrase in haystack for phrase in _MARKET_PHRASES
        )
        item.relevance = "direct" if direct else "market" if market else "weak"
        relevance_score = {"direct": 2, "market": 1, "weak": 0}[item.relevance]
        question_score = len(words & question_tokens)
        timestamp = item.published_at.timestamp() if item.published_at else float("-inf")
        ranked.append((relevance_score * 100 + question_score, timestamp, item.title, item))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return [row[3] for row in ranked[: max(0, limit)]]


def cluster_news_events(items: list[NewsItem]) -> dict[str, list[NewsItem]]:
    """将标题词项高度重合的新闻归为疑似转载事件,并原地写入 event_id。

    分组只表示可能重复报道,不能当作多来源独立确认。
    """
    groups: list[list[NewsItem]] = []
    fingerprints: list[set[str]] = []
    for item in items:
        words = set(_tokens(item.title))
        destination: int | None = None
        for index, known in enumerate(fingerprints):
            # Conservative: require a strong overlap and at least two shared terms.
            overlap = len(words & known)
            union = len(words | known) or 1
            if overlap >= 2 and overlap / union >= 0.72:
                destination = index
                break
        if destination is None:
            groups.append([item])
            fingerprints.append(words)
        else:
            groups[destination].append(item)

    result: dict[str, list[NewsItem]] = {}
    for group in groups:
        canonical = min((item.url or item.title) for item in group)
        event_id = "evt_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        for item in group:
            item.event_id = event_id
        result[event_id] = group
    return result


def validate_fact_citation(
    card: FactCard, documents: dict[str, EvidenceDocument]
) -> dict[str, object]:
    """检查证据是否存在、可用且包含规范化后的引文,不核验语义支持。

    semantic_support 固定为 not_checked,防止把字面匹配当作结论已证实。
    """
    document = documents.get(card.evidence_id)
    exists = document is not None
    quote_matches = bool(
        document and card.quote and _normalize(card.quote) in _normalize(document.text)
    )
    return {
        "evidence_id": card.evidence_id,
        "exists": exists,
        "quote_matches": quote_matches,
        "citation_valid": exists
        and quote_matches
        and document is not None
        and document.status == "available",
        "semantic_support": "not_checked",
    }


def citation_for_fact(fact: FactCard, document: EvidenceDocument) -> Citation:
    """建立绑定正文版本的引用;可用正文包含规范化引文时标记 valid。

    start/end 为原文字符半开偏移,仅能直接找到原始引文时填写;仅规范化后匹配
    时偏移保持 None,不虚构位置。
    """
    normalized_text = _normalize(document.text)
    normalized_quote = _normalize(fact.quote)
    start: int | None = None
    end: int | None = None
    validation: str = "invalid"
    if normalized_quote and normalized_quote in normalized_text and document.status == "available":
        raw_start = document.text.find(fact.quote)
        if raw_start >= 0:
            start = raw_start
            end = raw_start + len(fact.quote)
        validation = "valid"
    return Citation(
        citation_id="cit_"
        + hashlib.sha256(
            (fact.fact_id + document.evidence_id + fact.quote).encode("utf-8")
        ).hexdigest()[:24],
        evidence_id=document.evidence_id,
        content_sha256=document.content_sha256,
        quote=fact.quote,
        start=start,
        end=end,
        validation=validation,  # type: ignore[arg-type]
    )


def save_evidence_documents(
    run_root: Path | str, run_id: str, documents: list[EvidenceDocument]
) -> Path:
    """把证据快照单独写入 run 目录,经临时文件替换目标文件并返回路径。

    run_id 必须符合路径安全约束;该存储与 JSONL 事件分离,文件错误直接上抛。
    """
    target = _evidence_path(run_root, run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "documents": [doc.model_dump(mode="json") for doc in documents]}
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def load_evidence_documents(run_root: Path | str, run_id: str) -> list[EvidenceDocument]:
    """读取已保存证据,校验内容 hash 和证据 ID,全程不访问网络。

    文件不存在返回空列表;内容不一致的文档降为 failed,损坏 JSON 或模型数据
    校验失败继续抛出,避免把损坏历史悄悄当作空证据。
    """
    target = _evidence_path(run_root, run_id)
    if not target.exists():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    rows = payload.get("documents", []) if isinstance(payload, dict) else []
    documents: list[EvidenceDocument] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        document = EvidenceDocument.model_validate(row)
        digest = hashlib.sha256(_clean_text(document.text).encode("utf-8")).hexdigest()
        expected_id = make_evidence_id(
            url=document.url,
            source=document.source,
            title=document.title,
            text=document.text,
            level=document.level,
        )
        if document.content_sha256 != digest or document.evidence_id != expected_id:
            document = document.model_copy(
                update={
                    "status": "failed",
                    "fetch_error": "evidence_integrity_mismatch",
                }
            )
        documents.append(document)
    return documents


def _evidence_path(run_root: Path | str, run_id: str) -> Path:
    if not _ALLOWED_RUN_ID.fullmatch(run_id):
        raise ValueError("invalid run_id")
    root = Path(run_root).resolve()
    target = (root / "evidence" / f"{run_id}.json").resolve()
    if root not in target.parents:
        raise ValueError("evidence path escapes run root")
    return target


def _clean_text(value: str) -> str:
    # Keep paragraph boundaries for bounded long-document reading.  Collapse
    # incidental horizontal whitespace per line, but do not turn a multi-
    # paragraph article into one undifferentiated string.
    lines = [" ".join(line.split()) for line in value.splitlines()]
    return "\n".join(lines).strip()


def _normalize(value: str) -> str:
    return _clean_text(value).casefold()


def _extract_dates(value: str) -> list[str]:
    """Extract date-shaped strings for comparison, without assigning meaning."""
    return list(dict.fromkeys(_DATE_RE.findall(value)))[:20]


def _tokens(value: str) -> list[str]:
    return _WORD_RE.findall(value.casefold())


def _asset_tokens(asset: str) -> set[str]:
    raw = asset.upper()
    base = raw.removesuffix("USDT").removesuffix("USDC").removesuffix("USD")
    aliases = {"BTC": {"bitcoin"}, "ETH": {"ethereum"}, "SOL": {"solana"}, "DOGE": {"dogecoin"}}
    # Providers frequently write the full trading pair (``BTCUSDT``) while
    # others use the base asset or its English name.  Keep all forms so a
    # pair-specific headline is not incorrectly downgraded to weak relevance.
    return {raw.casefold(), base.casefold(), *aliases.get(base, set())}


_MARKET_TOKENS = {
    "cpi",
    "pce",
    "fomc",
    "inflation",
    "unemployment",
    "payroll",
    "gdp",
    "通胀",
    "非农",
    "crypto",
    "cryptocurrency",
    "bitcoin",
    "ethereum",
    "etf",
    "fed",
    "interest",
    "rate",
    "regulation",
    "sec",
    "exchange",
    "stablecoin",
    "加密",
    "比特币",
    "以太坊",
    "监管",
    "利率",
    "美联储",
}

_MARKET_PHRASES = (
    "consumer price index",
    "personal consumption expenditures",
    "federal reserve",
    "central bank",
    "消费者价格指数",
)
