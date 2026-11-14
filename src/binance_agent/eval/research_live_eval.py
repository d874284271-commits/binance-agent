"""Budgeted real-model evaluation of the frozen *full research graph*.

No live news is downloaded: all arms use the same frozen source corpus.
Lexical scoring is diagnostic only; it is not independent semantic acceptance.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from ..agents.budget import EndpointPool, RunBudget, run_budget_scope
from ..agents.graph import build_graph
from ..agents.runtime import RuntimeModelCandidate
from ..demo import _complete_snapshot
from ..infra.settings import ResearchConfig
from ..intel.evidence import evidence_document
from .experiment_budget import DEFAULT_EXPERIMENT_LEDGER, record_phase_snapshot, reserve_phase
from .research_eval import (
    build_comparison_manifests,
    derived_summary_source,
    load_cases,
    score_jsonl,
)
from .research_live import save_json


def _symbol(title: str) -> str:
    lowered = title.lower()
    for asset, words in [
        ("ETH", ("eth", "ethereum")),
        ("SOL", ("sol", "solana")),
        ("XRP", ("xrp", "ripple")),
        ("DOGE", ("doge", "dogecoin")),
    ]:
        if any(word in lowered for word in words):
            return asset + "USDT"
    return "BTCUSDT"


def file_fingerprint(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
        digest.update(str(path.relative_to(Path(__file__).resolve().parents[1])).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def frozen_sources(case: dict[str, Any]) -> list[dict[str, Any]]:
    source = case["research_input"]
    if case.get("corpus_kind") == "historical_sources":
        items = [dict(item) for item in source["sources"]]
        if "body" not in source and source.get("summary"):
            derived = derived_summary_source(case["case_id"], source)
            if derived is not None:
                items.append(derived)
        return items
    return [
        {
            "url": f"https://fixture.invalid/{case['case_id']}",
            "title": source.get("title", ""),
            "summary": source.get("summary", ""),
            "body": source.get("body", ""),
            "source": "Synthetic eval",
        }
    ]


def frozen_fetcher(case: dict[str, Any]) -> Any:
    sources = {source["url"]: source for source in frozen_sources(case)}

    async def fetch(url: str) -> Any:
        source = sources.get(url, {})
        body = str(source.get("body") or "")
        unsupported = body.startswith("CONTENT_UNSUPPORTED_") or body == "CONTENT_FETCH_FAILED"
        return evidence_document(
            url=url,
            title=str(source.get("title", "")),
            source=str(source.get("source", "")),
            text="" if unsupported else body,
            level="body",
            status="available" if body and not unsupported else "unavailable",
            source_kind=str(source.get("source_kind", "unknown")),
            published_at=datetime.fromisoformat(source["published_at"].replace("Z", "+00:00"))
            if source.get("published_at")
            else None,
            fetched_at=datetime.fromisoformat(source["fetched_at"].replace("Z", "+00:00"))
            if source.get("fetched_at")
            else None,
        )

    return fetch


def experiment_ledger_path(args: argparse.Namespace, *, historical: bool) -> Path:
    canonical = DEFAULT_EXPERIMENT_LEDGER.resolve()
    supplied = getattr(args, "experiment_ledger", None)
    selected = Path(supplied).resolve() if supplied else canonical
    if selected != canonical:
        raise ValueError("experiments require the canonical cumulative experiment ledger")
    return selected


def paired_execution_schedule(
    cells: dict[tuple[str, str], dict[str, Any]], *, seed: int
) -> list[tuple[str, str, dict[str, Any]]]:
    """Keep all workflow arms adjacent per case, with seeded case/arm order."""
    rng = random.Random(seed)
    case_ids = sorted(
        {case["case_id"] for manifest in cells.values() for case in manifest["cases"]}
    )
    rng.shuffle(case_ids)
    by_cell = {
        cell: {case["case_id"]: case for case in manifest["cases"]}
        for cell, manifest in cells.items()
    }
    schedule: list[tuple[str, str, dict[str, Any]]] = []
    for case_id in case_ids:
        arms = sorted(cell for cell, cases in by_cell.items() if case_id in cases)
        rng.shuffle(arms)
        schedule.extend(
            (mode, workflow, by_cell[(mode, workflow)][case_id]) for mode, workflow in arms
        )
    return schedule


async def evaluate(args: argparse.Namespace, key: str) -> dict[str, Any]:
    root = Path(args.output)
    cases = [c for c in load_cases(args.cases) if c["split"] == args.split][: args.limit]
    if args.limit <= 0 or args.case_timeout <= 0 or not math.isfinite(args.case_timeout):
        raise ValueError("limit and case-timeout must be finite positive values")
    if not cases:
        raise ValueError("no selected cases")
    historical = cases[0].get("corpus_kind") == "historical_sources"
    input_rate = getattr(args, "input_per_million", None)
    output_rate = getattr(args, "output_per_million", None)
    if historical and (input_rate is None or output_rate is None):
        raise ValueError("historical evaluation requires explicit input/output pricing")
    if any(
        value is not None and (not math.isfinite(value) or value < 0)
        for value in (input_rate, output_rate)
    ):
        raise ValueError("pricing must be finite and nonnegative")
    json_mode = bool(getattr(args, "json_mode", False))
    streaming = bool(getattr(args, "streaming", False))
    reasoning_effort = getattr(args, "reasoning_effort", None)
    thinking = getattr(args, "thinking", None)
    if thinking not in {None, "enabled", "disabled"}:
        raise ValueError("thinking must be enabled or disabled")
    if reasoning_effort not in {None, "none", "minimal", "low", "medium", "high"}:
        raise ValueError("reasoning-effort must be none, minimal, low, medium or high")
    model_timeout = float(getattr(args, "model_timeout", 90.0))
    if not math.isfinite(model_timeout) or model_timeout <= 0:
        raise ValueError("model-timeout must be finite and positive")
    execution_seed = int(getattr(args, "execution_seed", 94721))
    manifests: dict[str, Any] = build_comparison_manifests(
        cases,
        model_config={
            "model": args.model,
            "endpoint": args.base_url,
            "max_retries": int(getattr(args, "max_retries", 0) or 0),
            "max_output_tokens": 4096,
            "json_mode": json_mode,
            "streaming": streaming,
            "stream_usage": streaming,
            "provider_stop_policy": "rate_timeout_5xx/v2",
            "reasoning_effort": reasoning_effort,
            "thinking": thinking,
            "model_timeout_s": model_timeout,
            "research_rounds": 2,
            "case_requests": 100,
            "case_timeout": args.case_timeout,
            "input_per_million": input_rate,
            "output_per_million": output_rate,
            "code_fingerprint": candidate_fingerprint(),
            "source_data_sha256": file_fingerprint(args.cases),
            "temperature": 0,
            "execution_seed": execution_seed,
            "execution_strategy": "case_paired_seeded/v1",
            "request_interval_s": max(0.0, float(getattr(args, "request_interval_s", 0.0) or 0.0)),
            "endpoint_concurrency": 1,
            "scope": "full_graph",
            "source_attribution": "derived_case_summary/v2",
            "analysis_clock": "frozen_case_analysis_at/v1",
            "synthetic": not historical,
        },
    )
    cells = {(m["input_mode"], m["workflow"]): m for group in manifests.values() for m in group}
    schedule = paired_execution_schedule(cells, seed=execution_seed)
    manifests["execution_schedule"] = {
        "strategy": "case_paired_seeded/v1",
        "seed": execution_seed,
        "order": [
            {"case_id": case["case_id"], "input_mode": mode, "workflow": workflow}
            for mode, workflow, case in schedule
        ],
    }
    frozen_path = root / f"{args.split}-freeze.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != manifests:
        raise ValueError("frozen manifest differs; do not overwrite or mix the experiment")
    save_json(frozen_path, manifests)
    ledger_path = Path(args.ledger) if args.ledger else root / f"{args.split}-ledger.json"
    experiment_ledger = experiment_ledger_path(args, historical=historical)
    phase_ceiling = reserve_phase(experiment_ledger, args.split)

    pool = EndpointPool(endpoint_concurrency=1)
    request_interval_s = max(0.0, float(getattr(args, "request_interval_s", 0.0) or 0.0))
    throttled_attempt_ids: set[str] = set()

    def on_change(snapshot: dict[str, Any]) -> None:
        if request_interval_s > 0:
            settled = {
                str(attempt["id"])
                for attempt in snapshot.get("attempts") or []
                if isinstance(attempt, dict)
                and attempt.get("id")
                and attempt.get("endpoint") == "authorized"
                and attempt.get("kind") != "tool"
                and attempt.get("status") in {"ok", "unknown", "error", "cancelled"}
            }
            if settled - throttled_attempt_ids:
                throttled_attempt_ids.update(settled)
                pool.cooldown_until["authorized"] = max(
                    pool.cooldown_until.get("authorized", 0.0),
                    time.monotonic() + request_interval_s,
                )
                snapshot = dict(snapshot)
                cooldowns = dict(snapshot.get("endpoint_cooldown_remaining_s") or {})
                cooldowns["authorized"] = max(
                    0.0, pool.cooldown_until["authorized"] - time.monotonic()
                )
                snapshot["endpoint_cooldown_remaining_s"] = cooldowns
        save_json(ledger_path, snapshot)
        record_phase_snapshot(experiment_ledger, args.split, snapshot)

    candidate_path = ledger_path.with_suffix(".candidate.json")
    candidate_config = manifests["workflow_comparison"][0]["model_config"]
    if ledger_path.exists() and not candidate_path.exists():
        raise ValueError(
            "legacy ledger has no frozen candidate; use a new ledger under shared experiment ceiling"
        )
    if candidate_path.exists() and json.loads(candidate_path.read_text()) != candidate_config:
        raise ValueError("ledger belongs to a different frozen candidate")
    save_json(candidate_path, candidate_config)
    if ledger_path.exists():
        budget = RunBudget.from_snapshot(
            json.loads(ledger_path.read_text()),
            endpoint_pool=pool,
            on_change=on_change,
        )
    else:
        budget = RunBudget(
            max_requests=100,
            deadline_s=args.case_timeout,
            max_cost_usd=phase_ceiling,
            endpoint_concurrency=1,
            endpoint_pool=pool,
            endpoint_prices={
                "authorized": {"input_per_million": input_rate, "output_per_million": output_rate}
            }
            if input_rate is not None and output_rate is not None
            else {},
            on_change=on_change,
        )
    if budget.max_cost_usd is not None:
        budget.max_cost_usd = min(budget.max_cost_usd, budget.cost_committed_usd + phase_ceiling)
    save_json(ledger_path, budget.snapshot())
    record_phase_snapshot(experiment_ledger, args.split, budget.snapshot())
    kwargs: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": key,
        "max_retries": 0,
        "temperature": 0,
        "timeout": model_timeout,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if streaming:
        kwargs.update(streaming=True, stream_usage=True)
    if thinking is not None:
        kwargs["extra_body"] = {"thinking": {"type": thinking}}
    model = ChatOpenAI(**kwargs)
    candidate = RuntimeModelCandidate(
        model=model,
        endpoint="authorized",
        timeout_s=model_timeout,
        max_retries=int(getattr(args, "max_retries", 0) or 0),
        max_output_tokens=4096,
        structured_output=json_mode,
    )
    results_path = root / f"{args.split}-results.jsonl"
    previous = (
        [json.loads(line) for line in results_path.read_text().splitlines()]
        if results_path.exists()
        else []
    )
    attempted = {(r["input_mode"], r["workflow"], r["case_id"]) for r in previous}
    stop_reason = None
    with run_budget_scope(budget):
        for mode, workflow, case in schedule:
            if (mode, workflow, case["case_id"]) in attempted:
                continue
            # New case = new research run. Keep cumulative attempt/fee limits,
            # but do not inherit the expired wall deadline of another case.
            budget.deadline_at = time.time() + args.case_timeout
            budget.max_requests = budget.requests_reserved + 100
            before = len(budget.attempts)
            started = time.perf_counter()
            source = case["research_input"]
            title = str(source.get("title", ""))
            fetch = frozen_fetcher(case)
            sources = frozen_sources(case)
            analysis_at = case.get("analysis_at", "2026-09-18T00:00:00+00:00")

            graph = build_graph(
                lambda _: model,
                research_config=ResearchConfig.model_validate(
                    {"enabled": True, "input_mode": mode, "workflow": workflow, "max_rounds": 2}
                ),
                runtime_config_for=lambda _: {"candidates": [candidate]},
                research_fetch=fetch,
                preflight_refetch_enabled=False,
                analysis_clock=partial(datetime.fromisoformat, analysis_at.replace("Z", "+00:00")),
            )
            state = {
                "symbol": case.get("symbol", _symbol(title)),
                "research_demo": not historical,
                "research_question": case.get("question")
                or "核查以下报道的事实、条件和不确定性: " + title,
                "analysis_at": analysis_at,
                "market": "fapi",
                "market_snapshot": _complete_snapshot(
                    case.get("symbol", _symbol(title)), "fapi"
                ).model_dump(mode="json"),
                "snapshot_freshness_enabled": False,
                "analyst_reports": [],
                "debate_turns": [],
                "runtime_events": [],
                "tool_events": [],
                "planner_steps": [],
                "planner_decision": None,
                "debate_round": 0,
                "max_debate_rounds": 2,
                "debate_stop_reason": None,
                "research_summary": None,
                "risk_assessment": None,
                "risk_gate_result": None,
                "final_advice": None,
                "intel_snapshot": {
                    "source_status": {
                        "historical_sources" if historical else "synthetic_fixture": "ok"
                    },
                    "news_items": [
                        {
                            key: item[key]
                            for key in (
                                "title",
                                "summary",
                                "url",
                                "source",
                                "source_kind",
                                "published_at",
                            )
                            if key in item
                        }
                        for item in sources
                    ],
                },
            }
            try:
                result = await asyncio.wait_for(graph.ainvoke(state), timeout=args.case_timeout)
                research = result["news_research"]
                row = {
                    "case_id": case["case_id"],
                    "input_mode": mode,
                    "workflow": workflow,
                    "status": "completed",
                    "output": result["final_advice"].model_dump(mode="json"),
                    "news_research": research,
                    "evidence_documents": result.get("evidence_documents", []),
                    "runtime_events": [
                        {k: v for k, v in e.items() if k != "budget"}
                        for e in result.get("runtime_events", [])
                    ],
                }
            except Exception as exc:
                row = {
                    "case_id": case["case_id"],
                    "input_mode": mode,
                    "workflow": workflow,
                    "status": "failed",
                    "error": type(exc).__name__,
                    "output": "",
                }
            attempts = budget.attempts[before:]
            models_sent = [
                a for a in attempts if a.get("kind") != "tool" and a.get("status") != "not_sent"
            ]
            runtime_failures = [e for e in row.get("runtime_events", []) if not e.get("ok")]
            if not models_sent:
                row["status"] = "blocked"
                row["error"] = "no_model_request"
            elif runtime_failures:
                row["status"] = "failed"
                row["error"] = "runtime_fallback"

            row.update(
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                token=sum((a.get("usage") or {}).get("total_tokens", 0) for a in models_sent)
                if models_sent and all(a.get("usage") for a in models_sent)
                else None,
                cost_usd=sum(a["cost_usd"] for a in attempts)
                if attempts and all(a.get("cost_usd") is not None for a in attempts)
                else None,
                requests=len(attempts),
                model_requests=len(models_sent),
                tool_requests=sum(a.get("kind") == "tool" for a in attempts),
                attempt_ids=[a["id"] for a in attempts],
                candidate_fingerprint=manifests["workflow_comparison"][0]["model_config"][
                    "code_fingerprint"
                ],
            )
            with results_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            previous.append(row)
            print(
                json.dumps(
                    {
                        "cell": [mode, workflow],
                        "case_id": case["case_id"],
                        "status": row["status"],
                        "requests": len(attempts),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if row["status"] == "blocked":
                stop_reason = "no_model_request"
                break
            if any(
                a.get("error_category") in {"rate_limit", "timeout", "http_5xx"} for a in attempts
            ):
                stop_reason = "provider_error_requires_new_execution_strategy"
                break
            if budget.max_cost_usd is not None and budget.cost_committed_usd >= budget.max_cost_usd:
                stop_reason = "budget_exhausted"
                break
            await asyncio.sleep(2)
    reports = {}
    for group in ("input_mode_comparison", "workflow_comparison"):
        group_manifests = manifests[group]
        reports[group] = [
            {
                "input_mode": m["input_mode"],
                "workflow": m["workflow"],
                "report": score_jsonl(
                    cases,
                    [
                        r
                        for r in previous
                        if r["input_mode"] == m["input_mode"] and r["workflow"] == m["workflow"]
                    ],
                ),
            }
            for m in group_manifests
        ]
    report = {
        "split": args.split,
        "case_count": len(cases),
        "unique_cells": len(cells),
        "planned_runs": len(cases) * len(cells),
        "executed_runs": len({(r["case_id"], r["input_mode"], r["workflow"]) for r in previous}),
        "stop_reason": stop_reason,
        "adoption": "inconclusive",
        "groups": reports,
        "budget": budget.snapshot(),
        "request_interval_s": request_interval_s,
        "execution_schedule": manifests["execution_schedule"],
        "limitations": [
            "Frozen historical sources"
            if historical
            else "Synthetic fixtures; not factual performance evidence.",
            "Market snapshot is a shared synthetic control input; this experiment does not evaluate returns.",
            "Single workflow uses one research worker with shared deterministic risk, synthesis and final verification; it is not a single model-call baseline.",
            "Lexical scoring cannot judge Chinese paraphrases or semantic truth; independent review is pending.",
            "No claim of improved factual accuracy or financial performance is permitted.",
            "A positive request interval is an operator throttle for providers that omit Retry-After; it is not a provider SLO.",
        ],
    }
    save_json(root / f"{args.split}-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", help="shared phase ledger across pilot and subsequent batches")
    parser.add_argument("--case-timeout", type=float, default=600)
    parser.add_argument("--max-retries", type=int, choices=[0, 1], default=1)
    parser.add_argument("--input-per-million", type=float, required=True)
    parser.add_argument("--output-per-million", type=float, required=True)
    parser.add_argument(
        "--detach", action="store_true", help="keep authorized experiment alive across UI turns"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cases", default="evals/news_research_v1/cases.json")
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--execution-seed", type=int, default=94721)
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="request json_object responses for structured workflow calls",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="aggregate SSE responses and request final usage; avoid non-streaming proxy idle timeouts",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default=None,
        help="optional provider reasoning effort",
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default=None,
        help="optional provider thinking control, sent as thinking.type in the request body",
    )
    parser.add_argument(
        "--model-timeout",
        type=float,
        default=90.0,
        help="positive finite per-model HTTP/runtime timeout in seconds",
    )
    parser.add_argument("--output", default="output/research-v1/eval-pilot")
    parser.add_argument("--experiment-ledger", help="cross-phase 50 USD budget ledger")
    parser.add_argument(
        "--request-interval",
        dest="request_interval_s",
        type=float,
        default=13.0,
        help="minimum seconds between authorized-endpoint requests",
    )
    args = parser.parse_args()
    key = os.environ.get(args.key_env) or getpass.getpass(
        "Authorized endpoint API key (not saved): "
    )
    if args.detach:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env[args.key_env] = key
        with (output / "execution.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "binance_agent.eval.research_live_eval",
                    *[a for a in sys.argv[1:] if a != "--detach"],
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=env,
                start_new_session=True,
            )
        save_json(
            output / "process.json",
            {"pid": child.pid, "split": args.split, "credential_saved": False},
        )
        print(json.dumps({"pid": child.pid, "output": str(output)}))
        return
    report = asyncio.run(evaluate(args, key))
    print(
        json.dumps(
            {
                "executed": report["executed_runs"],
                "planned": report["planned_runs"],
                "stop_reason": report["stop_reason"],
            }
        )
    )


if __name__ == "__main__":
    main()
