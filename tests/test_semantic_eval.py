from __future__ import annotations

import copy
import json

import pytest

from binance_agent.eval.semantic_eval import build_review_packet, paired_comparison, score_reviews
from test_research_real_eval import real_case


def packet_reviews():
    case = real_case()
    from binance_agent.intel.evidence import evidence_document

    source = case["sources"][0]
    doc = evidence_document(
        url=source["url"],
        title=source["title"],
        source=source["source"],
        text=source["body"],
        level="body",
    )
    quote = "Approval has not been granted."
    fact = {
        "statement": "No approval",
        "evidence_id": doc.evidence_id,
        "quote": quote,
        "citations": [
            {"evidence_id": doc.evidence_id, "quote": quote, "content_sha256": doc.content_sha256}
        ],
    }
    results = [
        {
            "case_id": case["case_id"],
            "input_mode": "evidence",
            "workflow": "single",
            "status": "completed",
            "output": "No approval",
            "runtime_events": [{"workflow": "secret"}],
            "news_research": {"facts": [fact]},
            "evidence_documents": [doc.model_dump(mode="json")],
        }
    ]
    packet, mapping = build_review_packet([case], results)
    reviews = []
    for item in packet["items"]:
        evidence = [{"source_id": "s1", "quote": "Approval has not been granted."}]
        reviews.append(
            {
                "review_id": item["review_id"],
                "required_facts": [
                    {
                        "fact_index": 0,
                        "supported": item["status"] == "completed",
                        "reason": "Matches original denial"
                        if item["status"] == "completed"
                        else "No output",
                        "evidence": evidence,
                    }
                ],
                "model_facts": [
                    {
                        "fact_index": i,
                        "supported": True,
                        "reason": "Explicit denial",
                        "evidence": evidence,
                    }
                    for i in range(len(item["model_facts"]))
                ],
                "final_output_reviewed": True,
                "unresolved_acknowledged": True,
                "serious_failure": False,
                "additional_assertions": [],
                "forbidden_assertions": [
                    {"assertion_index": 0, "asserted": False, "reason": "No approval claim"}
                ],
                "reason": "Reviewed original source and output",
            }
        )
    return packet, mapping, reviews


def test_anonymous_packet_and_failure_denominator() -> None:
    packet, mapping, reviews = packet_reviews()
    serialized = json.dumps(packet)
    assert "multi_debate" not in serialized
    assert "single" not in serialized
    assert "runtime_events" not in serialized
    assert "secret" not in serialized
    report = score_reviews(packet, mapping, reviews)
    assert len(report["rows"]) == 5
    assert report["cells"]["evidence:single"]["case_pass_rate"] == 1
    assert report["cells"]["evidence:multi_debate"]["case_denominator"] == 1
    assert report["cells"]["evidence:multi_debate"]["case_pass_rate"] == 0
    assert report["cells"]["evidence:multi_debate"]["failed_or_missing"] == 1


def test_case_summary_cannot_impersonate_official_source_or_other_input_modes() -> None:
    from binance_agent.eval.semantic_eval import citation_integrity
    from binance_agent.intel.evidence import evidence_document

    case = real_case()
    source = case["sources"][0]
    case["article"]["summary"] = "A derived summary combining several announcements."

    def item_for(doc, mode):
        fact = {
            "statement": doc.text,
            "evidence_id": doc.evidence_id,
            "quote": doc.text,
            "citations": [
                {
                    "evidence_id": doc.evidence_id,
                    "quote": doc.text,
                    "content_sha256": doc.content_sha256,
                }
            ],
        }
        packet, _ = build_review_packet(
            [case],
            [
                {
                    "case_id": case["case_id"],
                    "input_mode": mode,
                    "workflow": "multi_no_debate",
                    "status": "completed",
                    "output": doc.text,
                    "news_research": {"facts": [fact]},
                    "evidence_documents": [doc.model_dump(mode="json")],
                }
            ],
        )
        return next(i for i in packet["items"] if i["model_facts"])

    wrong = evidence_document(
        url=source["url"],
        source=source["source"],
        title=source["title"],
        text=case["article"]["summary"],
        level="summary",
    )
    assert citation_integrity(item_for(wrong, "summary"))

    from binance_agent.eval.research_eval import derived_summary_source

    derived = derived_summary_source(case["case_id"], case["article"])
    assert derived is not None
    correct = evidence_document(
        url=derived["url"],
        source=derived["source"],
        title=derived["title"],
        text=derived["summary"],
        level="summary",
    )
    summary_item = item_for(correct, "summary")
    assert not citation_integrity(summary_item)
    assert len(summary_item["sources"]) == len(case["sources"])
    for mode in ("title", "evidence"):
        assert citation_integrity(item_for(correct, mode))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "fact", "reason", "quote", "packet"])
def test_reviews_require_complete_grounded_judgements(fault: str) -> None:
    packet, mapping, reviews = packet_reviews()
    if fault == "missing":
        reviews.pop()
    elif fault == "duplicate":
        reviews.append(copy.deepcopy(reviews[0]))
    elif fault == "fact":
        reviews[0]["required_facts"] = []
    elif fault == "reason":
        reviews[0]["reason"] = ""
    elif fault == "quote":
        reviews[0]["required_facts"][0]["evidence"][0]["quote"] = "imaginary"
    else:
        packet["items"][0]["question"] = "altered"
    with pytest.raises(ValueError):
        score_reviews(packet, mapping, reviews)


def test_paired_bootstrap_demands_positive_lower_bound_and_no_serious_failure() -> None:
    rows = [
        {
            "case_id": str(i),
            "input_mode": "evidence",
            "workflow": workflow,
            "case_pass": workflow == "multi_debate",
            "serious_failure": False,
        }
        for i in range(16)
        for workflow in ("single", "multi_debate")
    ]
    result = paired_comparison(rows, candidate="multi_debate", baseline="single", samples=500)
    assert result["ci95"] == [1, 1]
    assert result["conclusion"] == "accuracy_advantage"
    rows[1]["serious_failure"] = True
    assert (
        paired_comparison(rows, candidate="multi_debate", baseline="single", samples=500)[
            "conclusion"
        ]
        == "inconclusive"
    )
    for row in rows:
        row.update(case_pass=False, serious_failure=False)
    assert paired_comparison(rows, candidate="multi_debate", baseline="single", samples=500)[
        "ci95"
    ] == [0, 0]


def test_shared_events_form_transitive_clusters_and_cannot_fake_independent_ci() -> None:
    from binance_agent.eval.semantic_eval import event_clusters

    cases = [
        {"case_id": "a", "event_id": "one", "event_ids": ["one"]},
        {"case_id": "b", "event_id": "two", "event_ids": ["two", "one"]},
        {"case_id": "c", "event_id": "three", "event_ids": ["three", "two"]},
        {"case_id": "d", "event_id": "four", "event_ids": ["four"]},
    ]
    clusters = event_clusters(cases)
    assert clusters["a"] == clusters["b"] == clusters["c"]
    assert clusters["d"] != clusters["a"]
    rows = [
        {
            "case_id": case_id,
            "event_cluster": clusters[case_id],
            "input_mode": "evidence",
            "workflow": workflow,
            "case_pass": workflow == "multi_debate",
            "serious_failure": False,
        }
        for case_id in ("a", "b", "c")
        for workflow in ("single", "multi_debate")
    ]
    report = paired_comparison(rows, candidate="multi_debate", baseline="single", samples=500)
    assert report["paired_cases"] == 3
    assert report["independent_event_clusters"] == 1
    assert report["ci95"] is None
    assert report["conclusion"] == "inconclusive"
    rows += [
        {
            "case_id": "d",
            "event_cluster": clusters["d"],
            "input_mode": "evidence",
            "workflow": workflow,
            "case_pass": False,
            "serious_failure": False,
        }
        for workflow in ("single", "multi_debate")
    ]
    report = paired_comparison(rows, candidate="multi_debate", baseline="single", samples=500)
    assert report["independent_event_clusters"] == 2
    assert report["ci95"] == [0, 1]


def test_fabricated_citation_is_serious_despite_reviewer_acceptance() -> None:
    from binance_agent.eval.semantic_eval import _digest

    packet, mapping, reviews = packet_reviews()
    item = next(item for item in packet["items"] if item["status"] == "completed")
    item["model_facts"][0]["citations"] = [
        {"evidence_id": "NONEXISTENT_DOCUMENT", "quote": "Approval has not been granted."}
    ]
    mapping["packet_sha256"] = _digest(packet)
    report = score_reviews(packet, mapping, reviews)
    row = next(row for row in report["rows"] if row["workflow"] == "single")
    assert row["case_pass"] is False
    assert row["serious_failure"] is True


@pytest.mark.parametrize("fault", ["hash", "quote", "source", "missing", "document_text"])
def test_citation_binding_cannot_be_overridden_by_semantic_verdict(fault: str) -> None:
    from binance_agent.eval.semantic_eval import _digest

    packet, mapping, reviews = packet_reviews()
    item = next(item for item in packet["items"] if item["status"] == "completed")
    fact = item["model_facts"][0]
    if fault == "hash":
        fact["citations"][0]["content_sha256"] = "forged"
    elif fault == "quote":
        fact["citations"][0]["quote"] = "Approval was granted"
    elif fault == "source":
        item["evidence_documents"][0]["url"] = "https://wrong-source.invalid"
    elif fault == "document_text":
        item["evidence_documents"][0]["text"] = "tampered"
    else:
        fact["citations"] = []
    mapping["packet_sha256"] = _digest(packet)
    report = score_reviews(packet, mapping, reviews)
    row = next(row for row in report["rows"] if row["workflow"] == "single")
    assert row["serious_failure"] is True
    assert row["case_pass"] is False
    assert row["citation_errors"]


def test_actual_citations_and_document_source_binding_survive_blinding() -> None:
    packet, _, _ = packet_reviews()
    item = next(item for item in packet["items"] if item["status"] == "completed")
    citation = item["model_facts"][0]["citations"][0]
    version = next(
        version
        for version in item["source_versions"]
        if version["evidence_id"] == citation["evidence_id"]
    )
    assert version["source_id"] == "s1"
    assert version["url"] == real_case()["sources"][0]["url"]
    assert version["frozen_source_sha256"] == real_case()["sources"][0]["content_sha256"]
    assert citation["content_sha256"] == version["content_sha256"]
