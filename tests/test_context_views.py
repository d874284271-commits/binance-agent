from __future__ import annotations

import copy
import json

import pytest

from binance_agent.agents.context_views import narrative_envelope, news_evidence_view
from binance_agent.agents.state import merge_context_compactions
from binance_agent.intel.evidence import citation_for_fact, evidence_document
from binance_agent.intel.models import FactCard


@pytest.mark.parametrize(
    "difference", [None, "validation", "quote", "start", "content_sha256", "support"]
)
def test_real_citation_identity_deduplicates_only_equivalent_facts(difference: str | None) -> None:
    doc = evidence_document(
        url="https://example.invalid/article", text="Approval not granted.", level="body"
    )
    rows = []
    for index in range(2):
        fact = FactCard(
            fact_id=f"fact-{index}",
            task_id=f"task-{index}",
            statement=doc.text,
            quote=doc.text,
            evidence_id=doc.evidence_id,
            support="model_supported",
        )
        fact.citations = [citation_for_fact(fact, doc)]
        rows.append(fact.model_dump(mode="json"))
    if difference == "support":
        rows[1]["support"] = "unresolved"
    elif difference is not None:
        rows[1]["citations"][0][difference] = {
            "validation": "invalid",
            "quote": "not granted",
            "start": 1,
            "content_sha256": "different",
        }[difference]
    research = {
        "facts": rows,
        "tasks": [{"task_id": f"task-{i}", "facts": [row]} for i, row in enumerate(rows)],
        "claims": [{"claim_id": "claim2", "fact_ids": ["fact-1"], "support": "supported"}],
    }
    original = copy.deepcopy(research)
    view = news_evidence_view(research)
    assert len(view["facts"]) == (1 if difference is None else 2)
    assert view["tasks"][1]["fact_ids"] == ["fact-0" if difference is None else "fact-1"]
    assert view["claims"][0]["fact_ids"] == view["tasks"][1]["fact_ids"]
    assert view["facts"][0]["citations"] == rows[0]["citations"]
    assert research == original


def test_evidence_projection_keeps_one_fact_table_without_erasing_conflicts() -> None:
    fact = {
        "fact_id": "f1",
        "statement": "Not approved",
        "quote": "Not approved",
        "support": "model_supported",
        "scope": ["France only"],
        "numbers": ["10 USD"],
    }
    duplicate = {**fact, "fact_id": "f2", "task_id": "task2"}
    opposite = {**fact, "fact_id": "f3", "statement": "Approved", "support": "unresolved"}
    research = {
        "facts": [fact, duplicate, opposite],
        "tasks": [
            {"task_id": "task1", "facts": [fact], "context_compactions": {"secret": "cache"}},
            {"task_id": "task2", "facts": [duplicate, opposite]},
        ],
        "claims": [
            {
                "claim_id": "c1",
                "statement": "Not approved",
                "fact_ids": ["f2"],
                "support": "supported",
                "citations": [{"quote": "Not approved"}],
            }
        ],
        "documents": [{"text": "ARTICLE_SENTINEL"}],
        "conflicts": ["approval disputed"],
        "reading_complete": False,
        "document_coverage": [{"complete": False}],
    }
    original = copy.deepcopy(research)
    projected = news_evidence_view(research)
    assert research == original
    assert len(projected["facts"]) == 2
    assert projected["tasks"][1]["fact_ids"] == ["f1", "f3"]
    assert projected["claims"][0]["fact_ids"] == ["f1"]
    assert "statement" not in projected["claims"][0]
    assert projected["facts"][0]["scope"] == ["France only"]
    assert projected["conflicts"] == ["approval disputed"]
    assert not projected["reading_complete"]
    assert "ARTICLE_SENTINEL" not in json.dumps(projected)
    assert "context_compactions" not in json.dumps(projected)


def test_only_explicit_narratives_are_compressible() -> None:
    payload = {
        "research_summary": "Interpretation, not observation",
        "debate_turns": [
            {"side": "bull", "round": 1, "argument": "Opinion", "claim_ids": ["bull:1:0"]}
        ],
        "claim_verdicts": [
            {
                "claim_id": "bull:1:0",
                "verdict": "mismatch",
                "source_value": "999",
                "observed": "10 USD",
            }
        ],
        "risk_assessment": {"warnings": ["未确认"]},
    }
    before = copy.deepcopy(payload)
    envelope = narrative_envelope(payload)
    assert payload == before
    rendered = envelope.render(
        [{"source_ids": [b.source_id for b in envelope.compressible_blocks], "text": "short"}]
    )
    assert rendered["claim_verdicts"] == payload["claim_verdicts"]
    assert rendered["risk_assessment"] == payload["risk_assessment"]
    assert {b.source_id for b in envelope.compressible_blocks} == {
        "research_summary",
        "debate:bull:1:argument",
    }


def test_parallel_compaction_updates_preserve_both_nodes() -> None:
    before = {"one": {"response": "first"}}
    after = merge_context_compactions(before, {"two": {"response": "second"}})
    assert set(after) == {"one", "two"}
    assert set(before) == {"one"}


def test_projection_reduces_repeated_facts_without_dropping_source_metadata() -> None:
    fact = {
        "fact_id": "one",
        "statement": "A qualified statement. " * 100,
        "quote": "Original quote. " * 100,
        "evidence_id": "document-one",
    }
    research = {
        "facts": [fact],
        "tasks": [{"task_id": str(i), "facts": [fact]} for i in range(6)],
        "documents": [
            {
                "evidence_id": "document-one",
                "url": "https://example.com/article",
                "content_sha256": "hash",
                "level": "body",
            }
        ],
    }
    view = news_evidence_view(research)
    assert len(json.dumps(view).encode()) < len(json.dumps(research).encode())
    assert view["sources"][0]["url"] == "https://example.com/article"
    assert all(task["fact_ids"] == ["one"] for task in view["tasks"])
