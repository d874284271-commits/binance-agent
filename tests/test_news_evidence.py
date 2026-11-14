from __future__ import annotations

from datetime import datetime, timezone

import pytest

from binance_agent.intel.evidence import (
    cluster_news_events,
    evidence_document,
    load_evidence_documents,
    save_evidence_documents,
    select_news_candidates,
    validate_fact_citation,
)
from binance_agent.intel.models import FactCard, NewsItem


def _news(title: str, summary: str = "", url: str = "") -> NewsItem:
    return NewsItem(title=title, summary=summary, url=url, published_at=datetime.now(timezone.utc))


def test_candidate_selection_prioritizes_asset_then_market() -> None:
    weak = _news("Celebrity launches a new product")
    market = _news("Fed rate decision moves crypto market")
    direct = _news("Bitcoin ETF sees new inflows")
    selected = select_news_candidates([weak, market, direct], asset="BTCUSDT")
    assert [item.title for item in selected] == [direct.title, market.title, weak.title]
    assert direct.relevance == "direct"
    assert market.relevance == "market"
    assert weak.relevance == "weak"


def test_candidate_selection_accepts_full_trading_pair_headline() -> None:
    item = _news("BTCUSDT liquidity remains stable")
    selected = select_news_candidates([item], asset="BTCUSDT")
    assert selected == [item]
    assert item.relevance == "direct"


def test_event_clustering_marks_syndication_not_independent_confirmation() -> None:
    first = _news("Bitcoin ETF sees record daily inflows", url="https://a.example/1")
    second = _news("Bitcoin ETF sees record daily inflows today", url="https://b.example/2")
    third = _news("Ethereum upgrade date announced", url="https://c.example/3")
    groups = cluster_news_events([first, second, third])
    assert len(groups) == 2
    assert first.event_id == second.event_id
    assert third.event_id != first.event_id


def test_citation_check_is_literal_only() -> None:
    document = evidence_document(
        url="https://example.com/a",
        source="Example",
        text="The application was submitted, not approved.",
        level="body",
    )
    card = FactCard(
        fact_id="fact-1",
        statement="The application was approved.",
        evidence_id=document.evidence_id,
        quote="The application was submitted, not approved.",
    )
    result = validate_fact_citation(card, {document.evidence_id: document})
    assert result == {
        "evidence_id": document.evidence_id,
        "exists": True,
        "quote_matches": True,
        "citation_valid": True,
        "semantic_support": "not_checked",
    }


def test_extracted_dates_do_not_replace_provider_publication_time() -> None:
    published = datetime(2026, 1, 2, tzinfo=timezone.utc)
    document = evidence_document(
        text="The event occurred on 2024-01-03, while this article was published later.",
        published_at=published,
        level="body",
    )
    assert document.published_at == published
    assert document.extracted_dates == ["2024-01-03"]


def test_evidence_round_trip_and_traversal_rejection(tmp_path) -> None:
    document = evidence_document(text="Captured text", level="summary")
    target = save_evidence_documents(tmp_path, "run_01", [document])
    assert target == tmp_path / "evidence" / "run_01.json"
    assert load_evidence_documents(tmp_path, "run_01") == [document]
    with pytest.raises(ValueError, match="invalid run_id"):
        save_evidence_documents(tmp_path, "../outside", [document])


def test_tampered_saved_evidence_fails_closed(tmp_path) -> None:
    document = evidence_document(text="Captured text", level="body")
    save_evidence_documents(tmp_path, "run_02", [document])
    path = tmp_path / "evidence" / "run_02.json"
    payload = path.read_text(encoding="utf-8").replace("Captured text", "Tampered text")
    path.write_text(payload, encoding="utf-8")
    loaded = load_evidence_documents(tmp_path, "run_02")
    assert loaded[0].status == "failed"
    assert loaded[0].fetch_error == "evidence_integrity_mismatch"
