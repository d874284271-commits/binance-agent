from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from binance_agent.eval.research_eval import build_manifest, load_cases, score_jsonl
from binance_agent.eval.research_live_eval import frozen_fetcher, frozen_sources


def real_case() -> dict:
    body = "The filing was submitted. Approval has not been granted."
    return {
        "case_id": "real-1",
        "event_id": "event-1",
        "event_ids": ["event-1", "event-2"],
        "split": "dev",
        "symbol": "BTCUSDT",
        "analysis_at": "2024-01-02T00:00:00Z",
        "question": "Was approval granted?",
        "article": {"title": "A filing", "summary": "An application"},
        "sources": [
            {
                "source_id": "s1",
                "url": "https://official.example/1",
                "title": "A filing",
                "source": "Authority",
                "source_kind": "official",
                "published_at": "2024-01-01T00:00:00Z",
                "fetched_at": "2026-01-01T00:00:00Z",
                "body": body,
                "content_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "status": "available",
            }
        ],
        "expected_facts": [
            {
                "statement": "No approval",
                "source_id": "s1",
                "quote": "Approval has not been granted.",
                "critical": True,
            }
        ],
        "forbidden_assertions": ["Approval granted"],
        "critical_unresolved": True,
        "required_terms": ["not granted"],
        "future_label": "GOLD SECRET",
    }


def corpus(tmp_path: Path, cases: list[dict]) -> Path:
    target = tmp_path / "cases.json"
    target.write_text(
        json.dumps({"schema_version": 2, "corpus_kind": "historical_sources", "cases": cases})
    )
    return target


def test_historical_whitelist_preserves_sources_but_no_labels(tmp_path: Path) -> None:
    cases = load_cases(corpus(tmp_path, [real_case()]))
    manifest = build_manifest(cases, input_mode="evidence", workflow="single")
    frozen = manifest["cases"][0]
    assert "future_label" not in frozen
    assert "expected_facts" not in frozen
    assert "No approval" not in json.dumps(frozen)
    assert frozen["research_input"]["sources"][0]["body"] == real_case()["sources"][0]["body"]
    for mode in ("title", "summary"):
        value = build_manifest(cases, input_mode=mode, workflow="single")["cases"][0]
        assert "body" not in value["research_input"]["sources"][0]


@pytest.mark.parametrize("mutation", ["hash", "future", "quote", "split", "source_split"])
def test_reject_corrupted_or_leaking_cases(tmp_path: Path, mutation: str) -> None:
    first = real_case()
    cases = [first]
    if mutation == "hash":
        first["sources"][0]["content_sha256"] = "bad"
    elif mutation == "future":
        first["sources"][0]["published_at"] = "2025-01-01T00:00:00Z"
    elif mutation == "quote":
        first["expected_facts"][0]["quote"] = "nonexistent"
    else:
        second = copy.deepcopy(first)
        second.update(case_id="real-2", split="holdout")
        if mutation == "source_split":
            second.update(event_id="new", event_ids=["new"])
        cases.append(second)
    with pytest.raises(ValueError):
        load_cases(corpus(tmp_path, cases))


async def test_multi_source_frozen_fetch_has_no_network(tmp_path: Path) -> None:
    case = real_case()
    extra = copy.deepcopy(case["sources"][0])
    extra.update(source_id="s2", url="https://official.example/2", title="Another event")
    case["sources"].append(extra)
    loaded = load_cases(corpus(tmp_path, [case]))
    frozen = build_manifest(loaded, input_mode="evidence", workflow="single")["cases"][0]
    fetch = frozen_fetcher(frozen)
    for source in case["sources"]:
        document = await fetch(source["url"])
        assert document.text == source["body"]
        assert document.source == "Authority"
        assert document.source_kind == "official"
        assert document.published_at.year == 2024
    assert (await fetch("https://not-in-corpus.example")).status == "unavailable"


def test_historical_summary_is_derived_and_never_attached_to_official_source(
    tmp_path: Path,
) -> None:
    cases = load_cases(corpus(tmp_path, [real_case()]))
    official_urls = {s["url"] for s in cases[0]["sources"]}
    for mode in ("title", "summary", "evidence"):
        case = build_manifest(cases, input_mode=mode, workflow="single")["cases"][0]
        sources = frozen_sources(case)
        official = [s for s in sources if s["url"] in official_urls]
        assert all(not s.get("summary") for s in official)
        derived = [s for s in sources if s["url"] not in official_urls]
        assert len(derived) == (1 if mode == "summary" else 0)
        if derived:
            assert derived[0]["summary"] == real_case()["article"]["summary"]
            assert derived[0]["source_kind"] == "unknown"
            assert derived[0].get("published_at") is None


async def test_summary_graph_keeps_derived_provenance_within_task_budget_without_fetching(
    tmp_path: Path,
) -> None:
    from binance_agent.agents.research import build_news_research_graph
    from binance_agent.infra.settings import ResearchConfig
    from test_news_research_graph import ResearchModel, model_for

    case = real_case()
    case["question"] = "Compare the provided Bitcoin announcements."
    case["sources"][0]["title"] = "Bitcoin ETF filing"
    case["article"] = {
        "title": "Bitcoin ETF filing / Bitcoin market data summary",
        "summary": "Bitcoin ETF application is pending, not approved.",
    }
    loaded = load_cases(corpus(tmp_path, [case]))
    frozen = build_manifest(loaded, input_mode="summary", workflow="multi_no_debate")["cases"][0]

    async def forbidden(url):
        raise AssertionError("summary arm must not fetch a body")

    graph = build_news_research_graph(
        model_for(ResearchModel(responses=["unused"])),
        config=ResearchConfig(enabled=True, input_mode="summary", max_tasks=2, max_documents=2),
        fetch=forbidden,
    )
    result = await graph.ainvoke(
        {
            "symbol": case["symbol"],
            "research_question": case["question"],
            "analysis_at": case["analysis_at"],
            "intel_snapshot": {"news_items": frozen_sources(frozen)},
        }
    )
    assert len(result["news_research"]["tasks"]) == 2
    summaries = [d for d in result["evidence_documents"] if d["level"] == "summary"]
    assert len(summaries) == 1
    assert summaries[0]["source_kind"] == "unknown"
    assert "fixture.invalid/corpus-summary/" in summaries[0]["url"]
    assert any(
        f["evidence_id"] == summaries[0]["evidence_id"] for f in result["news_research"]["facts"]
    )


def test_duplicate_attempt_is_not_an_independent_sample() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        score_jsonl([real_case()], [{"case_id": "real-1"}, {"case_id": "real-1"}])


async def test_historical_runner_policy_and_failed_resume_denominator(
    tmp_path: Path, monkeypatch
) -> None:
    import argparse

    from binance_agent.agents.state import StrategyAdvice
    from binance_agent.eval.research_live_eval import evaluate

    case = real_case()
    cases_path = corpus(tmp_path, [case])
    captured = []
    executed_cells = []

    class Graph:
        async def ainvoke(self, state):
            captured.append(state)
            return {
                "final_advice": StrategyAdvice(symbol="BTCUSDT", decision_type="no_trade"),
                "news_research": {},
                "runtime_events": [],
            }

    def graph(*args, **kwargs):
        config = kwargs["research_config"]
        assert config.max_rounds == 2
        assert kwargs["analysis_clock"]().isoformat() == case["analysis_at"].replace("Z", "+00:00")
        executed_cells.append((config.input_mode, config.workflow))
        return Graph()

    monkeypatch.setattr(
        "binance_agent.eval.research_live_eval.DEFAULT_EXPERIMENT_LEDGER",
        tmp_path / "experiment-ledger.json",
    )
    monkeypatch.setattr("binance_agent.eval.research_live_eval.build_graph", graph)
    monkeypatch.setattr(
        "binance_agent.eval.research_live_eval.candidate_fingerprint", lambda: "frozen"
    )
    args = argparse.Namespace(
        output=str(tmp_path / "output"),
        split="dev",
        limit=1,
        cases=str(cases_path),
        base_url="https://example.invalid/v1",
        model="fixture",
        ledger=None,
        case_timeout=600,
        input_per_million=0.3,
        output_per_million=1.2,
    )
    report = await evaluate(args, "not-real")
    assert captured[0]["analysis_at"] == case["analysis_at"]
    assert captured[0]["research_question"] == case["question"]
    assert captured[0]["research_demo"] is False
    assert captured[0]["intel_snapshot"]["news_items"][0]["source"] == "Authority"
    assert report["budget"]["max_requests"] == 100
    report = await evaluate(args, "not-real")
    # A blocked attempt is retained, never retried into a new independent sample.
    assert len(captured) == 2  # Next previously unattempted comparison cell only.
    assert report["executed_runs"] == 2
    frozen = json.loads((tmp_path / "output" / "dev-freeze.json").read_text())
    config = frozen["workflow_comparison"][0]["model_config"]
    assert config["max_output_tokens"] == 4096
    assert config["research_rounds"] == 2
    assert config["source_data_sha256"]
    assert report["execution_schedule"] == frozen["execution_schedule"]
    assert executed_cells == [
        (entry["input_mode"], entry["workflow"])
        for entry in frozen["execution_schedule"]["order"][:2]
    ]
    assert frozen["execution_schedule"]["seed"] == 94721
    assert len(frozen["execution_schedule"]["order"]) == 5
    args.execution_seed = 94722
    with pytest.raises(ValueError, match="frozen manifest differs"):
        await evaluate(args, "not-real")
    args.execution_seed = 94721
    args.model = "different-candidate"
    with pytest.raises(ValueError, match="frozen manifest differs"):
        await evaluate(args, "not-real")


def test_real_experiment_ledger_cannot_reset_with_output_directory(
    tmp_path: Path, monkeypatch
) -> None:
    import argparse

    from binance_agent.eval import research_live_eval as live

    canonical = tmp_path / "approved" / "experiment-ledger.json"
    monkeypatch.setattr(live, "DEFAULT_EXPERIMENT_LEDGER", canonical)
    args = argparse.Namespace(output=str(tmp_path / "first"), experiment_ledger=None)
    assert live.experiment_ledger_path(args, historical=True) == canonical.resolve()
    args.output = str(tmp_path / "different-parent" / "second")
    assert live.experiment_ledger_path(args, historical=True) == canonical.resolve()
    args.experiment_ledger = str(tmp_path / "bypass.json")
    with pytest.raises(ValueError, match="canonical"):
        live.experiment_ledger_path(args, historical=True)
