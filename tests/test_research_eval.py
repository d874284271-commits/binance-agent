from __future__ import annotations

import asyncio
from pathlib import Path

from binance_agent.eval.research_eval import (
    build_comparison_manifests,
    load_cases,
    run_fixture_manifest,
    score_jsonl,
)

CASES = Path(__file__).parents[1] / "evals" / "news_research_v1" / "cases.json"


def test_fixed_synthetic_cases_have_expected_split_and_no_event_leakage() -> None:
    cases = load_cases(CASES)
    assert len(cases) == 40
    assert sum(case["split"] == "dev" for case in cases) == 24
    assert sum(case["split"] == "holdout" for case in cases) == 16
    assert all(case["expected_facts"] for case in cases)


def test_comparison_manifests_hold_one_variable_fixed() -> None:
    cases = load_cases(CASES)
    groups = build_comparison_manifests(cases, model_config={"model": "fixture"})
    content = groups["input_mode_comparison"]
    workflow = groups["workflow_comparison"]
    assert [item["input_mode"] for item in content] == ["title", "summary", "evidence"]
    assert {item["workflow"] for item in content} == {"multi_no_debate"}
    assert {item["input_mode"] for item in workflow} == {"evidence"}
    assert [item["workflow"] for item in workflow] == ["single", "multi_no_debate", "multi_debate"]


def test_score_keeps_missing_and_unsupported_in_denominator() -> None:
    cases = load_cases(CASES)[:2]
    report = score_jsonl(
        cases,
        [{"case_id": "dev-01", "status": "unsupported", "output": "pilot not ETF"}],
    )
    assert report["case_denominator"] == 2
    assert report["unsupported_count"] == 1
    assert report["missing_result_count"] == 1
    assert report["metrics"]["cost_usd"] == {
        "known_count": 0,
        "unknown_count": 2,
        "total": None,
    }
    assert "不构成" in report["interpretation"]


def test_injected_async_runner_is_only_wiring() -> None:
    manifest = build_comparison_manifests(load_cases(CASES))["workflow_comparison"][0]

    async def runner(case: dict[str, object]) -> dict[str, object]:
        return {"status": "completed", "output": case["case_id"]}

    results = asyncio.run(
        run_fixture_manifest({**manifest, "cases": manifest["cases"][:2]}, runner)
    )
    assert [result["case_id"] for result in results] == ["dev-01", "dev-02"]
    assert all(result["input_mode"] == "evidence" for result in results)
