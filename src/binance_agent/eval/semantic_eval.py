"""Blinded, source-grounded review packets and paired semantic evaluation.

Packets contain no workflow, trace, candidate fingerprint or billing data. Keep the
mapping private from the reviewer. Semantic judgements are supplied by an independent
reviewer; this module validates completeness and computes statistics, never substitutes
keyword matching for a judgement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..intel.evidence import evidence_document
from .research_eval import derived_summary_source, load_cases


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def event_clusters(cases: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Connected components prevent single/compound cases counting shared events twice."""
    parents: dict[str, str] = {}

    def root(event: str) -> str:
        parents.setdefault(event, event)
        if parents[event] != event:
            parents[event] = root(parents[event])
        return parents[event]

    for case in cases:
        events = [str(value) for value in case.get("event_ids", [case["event_id"]])]
        for event in events[1:]:
            parents[root(event)] = root(events[0])
        root(events[0])
    components = sorted({root(event) for event in parents})
    anonymous = {component: f"C{i:03d}" for i, component in enumerate(components, 1)}
    return {
        str(case["case_id"]): anonymous[root(str(case.get("event_ids", [case["event_id"]])[0]))]
        for case in cases
    }


def _source_versions(case: Mapping[str, Any], *, input_mode: str) -> list[dict[str, Any]]:
    versions = []
    for source in case["sources"]:
        texts = {"body": source["body"], "title": source["title"]}
        for level, text in texts.items():
            doc = evidence_document(
                url=source["url"],
                source=source["source"],
                title=source["title"],
                text=text,
                level=level,
            )
            versions.append(
                {
                    "evidence_id": doc.evidence_id,
                    "source_id": source["source_id"],
                    "url": source["url"],
                    "content_sha256": doc.content_sha256,
                    "frozen_source_sha256": source["content_sha256"],
                    "level": level,
                }
            )
    if input_mode == "summary":
        source = derived_summary_source(str(case["case_id"]), case.get("article") or {})
        if source is not None:
            doc = evidence_document(
                url=source["url"],
                source=source["source"],
                title=source["title"],
                text=source["summary"],
                level="summary",
                source_kind="unknown",
            )
            versions.append(
                {
                    "evidence_id": doc.evidence_id,
                    "source_id": source["source_id"],
                    "url": source["url"],
                    "content_sha256": doc.content_sha256,
                    "frozen_source_sha256": doc.content_sha256,
                    "level": "summary",
                    "provenance": "derived_case_summary/v2",
                }
            )
    return versions


def citation_integrity(item: Mapping[str, Any]) -> list[str]:
    """Source truth cannot substitute for a model's actual document/quote attribution."""
    versions = {version["evidence_id"]: version for version in item.get("source_versions", [])}
    documents: dict[str, Mapping[str, Any]] = {}
    errors = []
    invalid_documents = set()
    for doc in item.get("evidence_documents", []):
        identifier = doc.get("evidence_id")
        version = versions.get(identifier)
        if (
            doc.get("status") != "available"
            or identifier in documents
            or not version
            or any(doc.get(key) != version.get(key) for key in ("url", "content_sha256", "level"))
            or hashlib.sha256(str(doc.get("text", "")).encode()).hexdigest()
            != doc.get("content_sha256")
        ):
            invalid_documents.add(identifier)
        documents[identifier] = doc
    for index, fact in enumerate(item["model_facts"]):
        fact_document = fact.get("evidence_id")
        if fact_document and (fact_document not in documents or fact_document in invalid_documents):
            errors.append(f"fact[{index}]:unknown_or_invalid_fact_document")
        citations = fact.get("citations") or []
        if not citations:
            # Title/summary arms can lack body support; do not invent a citation for them.
            if item.get("body_evidence_expected"):
                errors.append(f"fact[{index}]:missing_citation")
            continue
        if not isinstance(citations, list):
            errors.append(f"fact[{index}]:invalid_citations")
            continue
        for citation in citations:
            if not isinstance(citation, dict):
                errors.append(f"fact[{index}]:invalid_citation")
                continue
            identifier = str(citation.get("evidence_id") or "")
            doc = documents.get(identifier)
            if not doc or identifier in invalid_documents:
                errors.append(f"fact[{index}]:unknown_or_invalid_document")
                continue
            if fact.get("evidence_id") and fact["evidence_id"] != identifier:
                errors.append(f"fact[{index}]:wrong_document")
            if fact.get("url") and fact["url"] != doc.get("url"):
                errors.append(f"fact[{index}]:wrong_source_url")
            quote = citation.get("quote")
            if citation.get("content_sha256") != doc.get("content_sha256"):
                errors.append(f"fact[{index}]:citation_hash_mismatch")
            if not isinstance(quote, str) or not quote or quote not in str(doc.get("text", "")):
                errors.append(f"fact[{index}]:citation_quote_mismatch")
            if fact.get("quote") and fact["quote"] != quote:
                errors.append(f"fact[{index}]:fact_quote_mismatch")
    return sorted(set(errors))


def build_review_packet(
    cases: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]], *, seed: int = 94721
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create randomized anonymous items plus a separate private cell mapping.

    Include all five cells for each case, including missing/failed attempts. Duplicate
    attempts are rejected: an operator must explicitly select a candidate before review.
    """
    cells = [
        ("title", "multi_no_debate"),
        ("summary", "multi_no_debate"),
        ("evidence", "single"),
        ("evidence", "multi_no_debate"),
        ("evidence", "multi_debate"),
    ]
    indexed = {}
    case_ids = {str(case["case_id"]) for case in cases}
    for row in results:
        key = (str(row["case_id"]), str(row["input_mode"]), str(row["workflow"]))
        if key in indexed or key[0] not in case_ids or key[1:] not in cells:
            raise ValueError("duplicate attempt or unexpected result cell")
        indexed[key] = row
    items = []
    clusters = event_clusters(cases)
    identities = []
    for case in cases:
        for mode, workflow in cells:
            key = (str(case["case_id"]), mode, workflow)
            row = indexed.get(key, {})
            research = row.get("news_research") or {}
            facts = research.get("facts", []) if isinstance(research, dict) else []
            item = {
                "event_cluster": clusters[str(case["case_id"])],
                "body_evidence_expected": mode == "evidence",
                "source_versions": _source_versions(case, input_mode=mode),
                "evidence_documents": [
                    {
                        key: doc[key]
                        for key in (
                            "evidence_id",
                            "url",
                            "source",
                            "title",
                            "text",
                            "content_sha256",
                            "level",
                            "status",
                        )
                        if key in doc
                    }
                    for doc in row.get("evidence_documents", [])
                ],
                "question": case["question"],
                "analysis_at": case["analysis_at"],
                "sources": [
                    {
                        k: source[k]
                        for k in (
                            "source_id",
                            "url",
                            "title",
                            "body",
                            "published_at",
                            "content_sha256",
                            "extraction_kind",
                        )
                        if k in source
                    }
                    for source in case["sources"]
                ],
                "required_facts": [
                    {
                        "fact_index": i,
                        **{k: f[k] for k in ("statement", "source_id", "quote", "critical")},
                    }
                    for i, f in enumerate(case["expected_facts"])
                ],
                "forbidden_assertions": list(case.get("forbidden_assertions", [])),
                "critical_unresolved": bool(case.get("critical_unresolved")),
                "model_facts": [
                    {
                        "fact_index": i,
                        **{
                            k: f[k]
                            for k in (
                                "statement",
                                "quote",
                                "url",
                                "source",
                                "evidence_id",
                                "citations",
                            )
                            if k in f
                        },
                    }
                    for i, f in enumerate(facts)
                ],
                "output": row.get("output", ""),
                "status": row.get("status", "missing"),
            }
            items.append(item)
            identities.append(
                {
                    "case_id": key[0],
                    "event_cluster": clusters[key[0]],
                    "input_mode": mode,
                    "workflow": workflow,
                }
            )
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    packet_items = []
    mapping = {}
    for n, original in enumerate(order, 1):
        review_id = f"R{n:04d}"
        packet_items.append({"review_id": review_id, **items[original]})
        mapping[review_id] = identities[original]
    packet = {
        "schema_version": 1,
        "items": packet_items,
        "instructions": "For required facts, supported means explicitly present in the output AND supported by frozen sources. For model facts, supported means supported by frozen sources. Review the entire final output for additional unsupported assertions, forbidden claims and serious failures. Record exact source quotes and reasons; do not infer financial performance.",
    }
    private = {"schema_version": 1, "packet_sha256": _digest(packet), "mapping": mapping}
    return packet, private


def review_template(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Unfilled review form; null decisions intentionally fail validation."""

    def verdict(index: int) -> dict[str, Any]:
        return {"fact_index": index, "supported": None, "evidence": [], "reason": ""}

    return [
        {
            "review_id": item["review_id"],
            "required_facts": [verdict(i) for i in range(len(item["required_facts"]))],
            "model_facts": [verdict(i) for i in range(len(item["model_facts"]))],
            "additional_assertions": [],
            "forbidden_assertions": [
                {"assertion_index": i, "asserted": None, "reason": ""}
                for i in range(len(item["forbidden_assertions"]))
            ],
            "final_output_reviewed": False,
            "unresolved_acknowledged": None,
            "serious_failure": None,
            "reason": "",
        }
        for item in packet["items"]
    ]


def _verdicts(value: Any, count: int, sources: dict[str, str]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("missing or extra per-fact verdicts")
    if {v.get("fact_index") for v in value} != set(range(count)):
        raise ValueError("duplicate or missing fact_index")
    for verdict in value:
        _validate_verdict(verdict, sources)
    return value


def _validate_verdict(verdict: Mapping[str, Any], sources: dict[str, str]) -> None:
    if type(verdict.get("supported")) is not bool or not str(verdict.get("reason", "")).strip():
        raise ValueError("verdict requires supported boolean and reason")
    evidence = verdict.get("evidence")
    if not isinstance(evidence, list) or (verdict["supported"] and not evidence):
        raise ValueError("supported verdict requires source evidence")
    for entry in evidence:
        quote = entry.get("quote")
        if (
            not isinstance(quote, str)
            or not quote
            or quote not in sources.get(entry.get("source_id"), "")
        ):
            raise ValueError("review evidence quote not in frozen source")


def score_reviews(
    packet: dict[str, Any], private: dict[str, Any], reviews: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if private.get("packet_sha256") != _digest(packet):
        raise ValueError("packet changed after blinding")
    review_map = {review.get("review_id"): review for review in reviews}
    expected_ids = {item["review_id"] for item in packet["items"]}
    if (
        len(review_map) != len(reviews)
        or set(review_map) != expected_ids
        or set(private["mapping"]) != expected_ids
    ):
        raise ValueError("duplicate, missing or unknown review_id")
    rows = []
    for item in packet["items"]:
        review = review_map[item["review_id"]]
        sources = {source["source_id"]: source["body"] for source in item["sources"]}
        required = _verdicts(review.get("required_facts"), len(item["required_facts"]), sources)
        claims = _verdicts(review.get("model_facts"), len(item["model_facts"]), sources)
        if review.get("final_output_reviewed") is not True:
            raise ValueError("entire final output must be reviewed")
        if (
            type(review.get("unresolved_acknowledged")) is not bool
            or type(review.get("serious_failure")) is not bool
        ):
            raise ValueError("missing unresolved/serious failure judgement")
        if not str(review.get("reason", "")).strip():
            raise ValueError("review requires reason")
        additional = review.get("additional_assertions")
        if not isinstance(additional, list):
            raise ValueError("additional_assertions must be an explicit list")
        for assertion in additional:
            if not assertion.get("statement"):
                raise ValueError("assertion requires statement")
            _validate_verdict(assertion, sources)
        forbidden = review.get("forbidden_assertions")
        if not isinstance(forbidden, list) or len(forbidden) != len(item["forbidden_assertions"]):
            raise ValueError("every forbidden assertion requires a judgement")
        if {entry.get("assertion_index") for entry in forbidden} != set(range(len(forbidden))):
            raise ValueError("duplicate or missing assertion_index")
        for entry in forbidden:
            if type(entry.get("asserted")) is not bool or not str(entry.get("reason", "")).strip():
                raise ValueError("forbidden assertion requires asserted boolean and reason")
        forbidden_count = sum(entry["asserted"] for entry in forbidden)
        unsupported = forbidden_count + sum(not claim["supported"] for claim in claims + additional)
        covered = (
            sum(fact["supported"] for fact in required) if item["status"] == "completed" else 0
        )
        citation_errors = citation_integrity(item)
        serious_failure = review["serious_failure"] or bool(citation_errors)
        passed = (
            item["status"] == "completed"
            and covered == len(required)
            and unsupported == 0
            and not serious_failure
            and (not item["critical_unresolved"] or review["unresolved_acknowledged"])
        )
        rows.append(
            {
                **private["mapping"][item["review_id"]],
                "status": item["status"],
                "required_total": len(required),
                "required_supported": covered,
                "unsupported_assertions": unsupported,
                "serious_failure": serious_failure,
                "citation_errors": citation_errors,
                "case_pass": passed,
            }
        )
    cells: dict[str, Any] = {}
    for row in rows:
        key = row["input_mode"] + ":" + row["workflow"]
        cells.setdefault(key, []).append(row)
    summary = {}
    for key, cell in cells.items():
        total = sum(row["required_total"] for row in cell)
        summary[key] = {
            "case_denominator": len(cell),
            "case_pass_rate": sum(row["case_pass"] for row in cell) / len(cell),
            "fact_coverage": sum(row["required_supported"] for row in cell) / total
            if total
            else None,
            "unsupported_assertions": sum(row["unsupported_assertions"] for row in cell),
            "serious_failures": sum(row["serious_failure"] for row in cell),
            "failed_or_missing": sum(row["status"] != "completed" for row in cell),
        }
    comparisons = {}
    for candidate in ("multi_no_debate", "multi_debate"):
        comparisons[candidate + "_vs_single"] = paired_comparison(
            rows, candidate=candidate, baseline="single"
        )
    content_comparisons = {
        mode + "_vs_title": paired_comparison(
            rows,
            candidate="multi_no_debate",
            baseline="multi_no_debate",
            candidate_mode=mode,
            baseline_mode="title",
        )
        for mode in ("summary", "evidence")
    }
    return {
        "schema_version": 1,
        "metric_kind": "independent_semantic_review",
        "content_comparisons": content_comparisons,
        "rows": rows,
        "cells": summary,
        "workflow_comparisons": comparisons,
        "adoption": "inconclusive",
        "default_changed": False,
    }


def paired_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    baseline: str,
    candidate_mode: str = "evidence",
    baseline_mode: str = "evidence",
    samples: int = 20000,
    seed: int = 519,
) -> dict[str, Any]:
    left = {
        r["case_id"]: r
        for r in rows
        if r["input_mode"] == candidate_mode and r["workflow"] == candidate
    }
    right = {
        r["case_id"]: r
        for r in rows
        if r["input_mode"] == baseline_mode and r["workflow"] == baseline
    }
    if set(left) != set(right) or not left:
        return {"conclusion": "inconclusive", "reason": "missing paired cases"}
    ids = sorted(left)
    differences = [int(left[i]["case_pass"]) - int(right[i]["case_pass"]) for i in ids]
    grouped: dict[str, list[int]] = {}
    for case_id, difference in zip(ids, differences, strict=True):
        cluster = str(left[case_id].get("event_cluster", case_id))
        if cluster != str(right[case_id].get("event_cluster", case_id)):
            raise ValueError("paired cases have inconsistent event clusters")
        grouped.setdefault(cluster, []).append(difference)
    if len(grouped) < 2:
        return {
            "paired_cases": len(ids),
            "independent_event_clusters": len(grouped),
            "mean_difference": sum(differences) / len(ids),
            "ci95": None,
            "conclusion": "inconclusive",
            "reason": "fewer than two independent event clusters",
        }
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        drawn = rng.choices(list(grouped.values()), k=len(grouped))
        means.append(sum(sum(values) for values in drawn) / sum(len(values) for values in drawn))
    means.sort()
    lower, upper = means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]
    new_serious = sum(
        bool(left[i]["serious_failure"]) and not right[i]["serious_failure"] for i in ids
    )
    # A one-case corpus cannot establish a population accuracy advantage.
    advantage = (
        len(ids) >= 2
        and lower > 0
        and not new_serious
        and not any(left[i]["serious_failure"] for i in ids)
    )
    return {
        "paired_cases": len(ids),
        "mean_difference": sum(differences) / len(ids),
        "ci95": [lower, upper],
        "method": "paired connected-event-cluster percentile bootstrap",
        "independent_event_clusters": len(grouped),
        "seed": seed,
        "samples": samples,
        "new_serious_failures": new_serious,
        "conclusion": "accuracy_advantage" if advantage else "inconclusive",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    packet_parser = sub.add_parser("packet")
    packet_parser.add_argument("--cases", required=True)
    packet_parser.add_argument("--results", required=True)
    packet_parser.add_argument("--split", choices=["dev", "holdout"], default="holdout")
    packet_parser.add_argument("--packet", required=True)
    packet_parser.add_argument("--private-mapping", required=True)
    packet_parser.add_argument("--template", help="write an unfilled review JSON form")
    score_parser = sub.add_parser("score")
    for name in ("packet", "private-mapping", "reviews", "output"):
        score_parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "packet":
        cases = [c for c in load_cases(args.cases) if c["split"] == args.split]
        results = [
            json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()
        ]
        packet, private = build_review_packet(cases, results)
        Path(args.packet).write_text(json.dumps(packet, ensure_ascii=False, indent=2))
        Path(args.private_mapping).write_text(json.dumps(private, ensure_ascii=False, indent=2))
        Path(args.private_mapping).chmod(0o600)
        if args.template:
            Path(args.template).write_text(
                json.dumps(review_template(packet), ensure_ascii=False, indent=2)
            )
    else:

        def read(path: str) -> Any:
            return json.loads(Path(path).read_text())

        report = score_reviews(read(args.packet), read(args.private_mapping), read(args.reviews))
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
