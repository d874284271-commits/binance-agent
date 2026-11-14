from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from binance_agent.agents.state import StrategyAdvice
from binance_agent.eval.research_live_eval import evaluate, paired_execution_schedule


async def test_zero_model_calls_are_blocked_not_completed(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "binance_agent.eval.research_live_eval.DEFAULT_EXPERIMENT_LEDGER",
        tmp_path / "experiment-ledger.json",
    )

    class Graph:
        async def ainvoke(self, state: Any) -> dict[str, Any]:
            return {
                "final_advice": StrategyAdvice(symbol="BTCUSDT", decision_type="no_trade"),
                "news_research": {},
                "runtime_events": [],
            }

    monkeypatch.setattr(
        "binance_agent.eval.research_live_eval.build_graph", lambda *a, **k: Graph()
    )
    args = argparse.Namespace(
        output=str(tmp_path),
        split="dev",
        limit=1,
        cases=str(Path(__file__).parents[1] / "evals/news_research_v1/cases.json"),
        base_url="https://example.invalid/v1",
        model="fixture",
        ledger=None,
        case_timeout=10,
    )
    report = await evaluate(args, "not-a-real-key")
    rows = [
        json.loads(line)
        for line in (tmp_path / "dev-results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["status"] == "blocked"
    assert rows[0]["model_requests"] == 0
    assert report["stop_reason"] == "no_model_request"


def test_execution_schedule_pairs_all_arms_per_case_and_is_seeded() -> None:
    arms = [
        ("title", "multi_no_debate"),
        ("summary", "multi_no_debate"),
        ("evidence", "single"),
        ("evidence", "multi_no_debate"),
        ("evidence", "multi_debate"),
    ]
    cells = {
        arm: {"cases": [{"case_id": case_id} for case_id in ("a", "b", "c", "d")]} for arm in arms
    }

    def identities(schedule: list) -> list:
        return [(case["case_id"], mode, workflow) for mode, workflow, case in schedule]

    schedule = paired_execution_schedule(cells, seed=94721)
    assert len(schedule) == 20
    assert len(set(identities(schedule))) == 20
    for start in range(0, len(schedule), len(arms)):
        block = schedule[start : start + len(arms)]
        assert len({case["case_id"] for _, _, case in block}) == 1
        assert {(mode, workflow) for mode, workflow, _ in block} == set(arms)
    assert identities(schedule) == identities(
        paired_execution_schedule(dict(reversed(list(cells.items()))), seed=94721)
    )
    assert identities(schedule) != identities(paired_execution_schedule(cells, seed=94722))


@pytest.mark.parametrize("configured", [False, True])
async def test_provider_strategy_reaches_http_and_runtime_and_is_frozen(
    tmp_path: Path, monkeypatch: Any, configured: bool
) -> None:
    from binance_agent.eval import research_live_eval as live

    monkeypatch.setattr(live, "DEFAULT_EXPERIMENT_LEDGER", tmp_path / "experiment-ledger.json")
    constructor_kwargs = []
    runtime_candidates = []

    def model(**kwargs: Any) -> object:
        constructor_kwargs.append(kwargs)
        return object()

    class Graph:
        async def ainvoke(self, state: Any) -> dict[str, Any]:
            return {
                "final_advice": StrategyAdvice(symbol="BTCUSDT", decision_type="no_trade"),
                "news_research": {},
                "runtime_events": [],
            }

    def graph(*args: Any, **kwargs: Any) -> Graph:
        runtime_candidates.extend(kwargs["runtime_config_for"]("test")["candidates"])
        return Graph()

    monkeypatch.setattr(live, "ChatOpenAI", model)
    monkeypatch.setattr(live, "build_graph", graph)
    args = argparse.Namespace(
        output=str(tmp_path / "output"),
        split="dev",
        limit=1,
        cases=str(Path(__file__).parents[1] / "evals/news_research_v1/cases.json"),
        base_url="https://example.invalid/v1",
        model="fixture",
        ledger=None,
        case_timeout=600,
    )
    if configured:
        args.json_mode = True
        args.reasoning_effort = "low"
        args.model_timeout = 180
        args.streaming = True
        args.thinking = "disabled"
    report = await evaluate(args, "unused")
    assert report["executed_runs"] == 1
    expected_timeout = 180 if configured else 90
    assert constructor_kwargs[0]["timeout"] == expected_timeout
    assert runtime_candidates[0].timeout_s == expected_timeout
    assert "model_kwargs" not in constructor_kwargs[0]
    assert constructor_kwargs[0].get("streaming", False) is configured
    if configured:
        assert constructor_kwargs[0]["stream_usage"] is True
        assert constructor_kwargs[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    else:
        assert "extra_body" not in constructor_kwargs[0]
    assert runtime_candidates[0].structured_output is configured
    if configured:
        assert constructor_kwargs[0]["reasoning_effort"] == "low"
    else:
        assert "reasoning_effort" not in constructor_kwargs[0]
    frozen = json.loads((Path(args.output) / "dev-freeze.json").read_text(encoding="utf-8"))
    for group in ("workflow_comparison", "input_mode_comparison"):
        for cell in frozen[group]:
            config = cell["model_config"]
            assert config["json_mode"] is configured
            assert config["reasoning_effort"] == ("low" if configured else None)
            assert config["model_timeout_s"] == expected_timeout
            assert config["streaming"] is configured
            assert config["thinking"] == ("disabled" if configured else None)
            assert config["case_timeout"] == 600
            assert config["max_output_tokens"] == 4096
    for name, changed in [
        ("json_mode", not configured),
        ("reasoning_effort", "medium"),
        ("model_timeout", 181),
        ("streaming", not configured),
        ("thinking", "enabled"),
    ]:
        original = getattr(args, name, None)
        existed = hasattr(args, name)
        setattr(args, name, changed)
        with pytest.raises(ValueError, match="frozen manifest differs"):
            await evaluate(args, "unused")
        if existed:
            setattr(args, name, original)
        else:
            delattr(args, name)
    assert len(constructor_kwargs) == 1  # Freeze rejection precedes any model construction.


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
async def test_invalid_model_timeout_is_rejected_before_freeze_or_budget(
    tmp_path: Path, monkeypatch: Any, timeout: float
) -> None:
    from binance_agent.eval import research_live_eval as live

    canonical = tmp_path / "experiment-ledger.json"
    monkeypatch.setattr(live, "DEFAULT_EXPERIMENT_LEDGER", canonical)
    args = argparse.Namespace(
        output=str(tmp_path / "output"),
        split="dev",
        limit=1,
        cases=str(Path(__file__).parents[1] / "evals/news_research_v1/cases.json"),
        base_url="https://example.invalid/v1",
        model="fixture",
        ledger=None,
        case_timeout=600,
        model_timeout=timeout,
    )
    with pytest.raises(ValueError, match="model-timeout"):
        await evaluate(args, "unused")
    assert not canonical.exists()
    output_path = tmp_path / "output"
    assert not output_path.exists()


async def test_json_mode_respects_structured_repair_text_and_compaction_sdk_contracts() -> None:
    import httpx
    from langchain_openai import ChatOpenAI

    from binance_agent.agents.context import ContextEnvelope, ContextUsage
    from binance_agent.agents.runtime import AgentRuntime, RuntimeModelCandidate

    payloads = []
    summary = json.dumps(
        {
            "status": "ok",
            "sections": [{"source_ids": ["analysis"], "text": "该解释仍待核验。"}],
            "unrepresented_source_ids": [],
        },
        ensure_ascii=False,
    )
    contents = iter(["not-json", '{"ok":true}', "结论保持普通中文散文。", summary])

    def transport(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-offline",
                "object": "chat.completion",
                "created": 0,
                "model": "grok-4.7",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": next(contents)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ChatOpenAI(
            model="grok-4.7",
            api_key="unused-not-a-real-key",
            base_url="https://example.invalid/v1",
            http_async_client=client,
            temperature=0,
            max_retries=0,
            timeout=180,
            reasoning_effort="low",
            extra_body={"thinking": {"type": "disabled"}},
        )
        candidate = RuntimeModelCandidate(
            model=model,
            endpoint="offline",
            structured_output=True,
            timeout_s=180,
            max_retries=0,
            max_output_tokens=4096,
        )
        runtime = AgentRuntime(candidates=[candidate])
        structured = await runtime.invoke_structured(
            "researcher", "Return JSON only.", {"input": "source"}, fallback_data={}
        )
        assert structured.ok and structured.repaired and structured.data == {"ok": True}
        text = await runtime.invoke_text(
            "research_manager",
            "只写中文散文,不要JSON。",
            {"input": "source"},
            fallback_text="失败",
        )
        assert text.ok and text.text == "结论保持普通中文散文。"
        envelope = ContextEnvelope(
            protected_data={"risk_gate": "watchlist"},
            compressible_blocks=[{"source_id": "analysis", "text": "待核验解释" * 200}],
        )
        compact_usage = ContextUsage()
        compact = await runtime._compact_context(
            candidate,
            "research_manager",
            "system",
            envelope,
            compact_usage,
            {},
            {},
        )
        assert compact is not None and compact["risk_gate"] == "watchlist"
        assert compact["derived_context"][0]["text"] == "该解释仍待核验。"
        assert compact_usage.compression_status == "compacted"
    assert len(payloads) == 4
    assert [p.get("response_format") for p in payloads] == [
        {"type": "json_object"},
        {"type": "json_object"},
        None,
        {"type": "json_object"},
    ]
    assert all(p["reasoning_effort"] == "low" for p in payloads)
    assert all(p["thinking"] == {"type": "disabled"} for p in payloads)
    assert all(p.get("max_completion_tokens", p.get("max_tokens", 0)) > 0 for p in payloads)


async def test_streamed_json_aggregates_content_and_usage_through_runtime() -> None:
    import httpx
    from langchain_openai import ChatOpenAI

    from binance_agent.agents.runtime import AgentRuntime, RuntimeModelCandidate

    payloads = []

    def transport(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": '{"ok":'},
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {"content": "true}"}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
            },
        ]
        body = (
            "".join(
                "data: "
                + json.dumps(
                    {
                        "id": "offline",
                        "object": "chat.completion.chunk",
                        "model": "grok-4.7",
                        "created": 0,
                        **chunk,
                    }
                )
                + "\n\n"
                for chunk in chunks
            )
            + "data: [DONE]\n\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ChatOpenAI(
            model="grok-4.7",
            api_key="unused",
            base_url="https://example.invalid/v1",
            http_async_client=client,
            streaming=True,
            stream_usage=True,
            max_retries=0,
        )
        runtime = AgentRuntime(
            candidates=[
                RuntimeModelCandidate(
                    model=model,
                    endpoint="offline",
                    structured_output=True,
                    max_retries=0,
                )
            ]
        )
        result = await runtime.invoke_structured("researcher", "Return JSON.", {}, fallback_data={})
    assert result.ok and result.data == {"ok": True}
    assert result.token_usage == {"input_tokens": 40, "output_tokens": 12, "total_tokens": 52}
    assert payloads[0]["stream"] is True
    assert payloads[0]["stream_options"] == {"include_usage": True}
    assert payloads[0]["response_format"] == {"type": "json_object"}


async def test_provider_5xx_stops_batch_before_starting_another_workflow(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from binance_agent.agents.budget import current_run_budget
    from binance_agent.eval import research_live_eval as live

    monkeypatch.setattr(live, "DEFAULT_EXPERIMENT_LEDGER", tmp_path / "experiment.json")
    monkeypatch.setattr(live, "ChatOpenAI", lambda **kwargs: object())
    started = []

    class Graph:
        async def ainvoke(self, state: Any) -> dict[str, Any]:
            started.append(state)
            budget = current_run_budget()
            assert budget is not None
            reservation = await budget.reserve(
                "authorized", "fixture", input_tokens_upper=10, output_tokens_limit=10
            )
            await budget.settle(
                reservation, outcome="error", possibly_billed=True, error_category="http_5xx"
            )
            return {
                "final_advice": StrategyAdvice(symbol="BTCUSDT", decision_type="no_trade"),
                "news_research": {},
                "runtime_events": [{"ok": False, "error_category": "http_5xx"}],
            }

    monkeypatch.setattr(live, "build_graph", lambda *args, **kwargs: Graph())
    args = argparse.Namespace(
        output=str(tmp_path / "output"),
        split="dev",
        limit=1,
        cases=str(Path(__file__).parents[1] / "evals/news_research_v1/cases.json"),
        base_url="https://example.invalid/v1",
        model="fixture",
        ledger=None,
        case_timeout=600,
        input_per_million=2.0,
        output_per_million=6.0,
    )
    report = await evaluate(args, "unused")
    assert len(started) == 1
    assert report["stop_reason"] == "provider_error_requires_new_execution_strategy"
    assert report["executed_runs"] == 1
    assert report["budget"]["unknown_cost_requests"] == 1


async def test_pacing_tracks_queued_completions_and_preserves_retry_after(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import asyncio
    import time

    from binance_agent.agents.budget import current_run_budget
    from binance_agent.eval import research_live_eval as live

    monkeypatch.setattr(live, "DEFAULT_EXPERIMENT_LEDGER", tmp_path / "experiment.json")
    monkeypatch.setattr(live, "ChatOpenAI", lambda **kwargs: object())
    cooldown_after_first = []
    sent = []
    finished = []

    class Graph:
        async def ainvoke(self, state: Any) -> dict[str, Any]:
            budget = current_run_budget()
            assert budget is not None
            gate = asyncio.Event()
            all_started = asyncio.Event()

            async def request(index):
                if index == 2:
                    all_started.set()
                reservation = await budget.reserve(
                    "authorized", str(index), input_tokens_upper=10, output_tokens_limit=10
                )
                sent.append(time.monotonic())
                if index == 0:
                    await gate.wait()
                await budget.settle(
                    reservation,
                    usage={"input_tokens": 10, "output_tokens": 2},
                    outcome="error" if index == 2 else "ok",
                    possibly_billed=index == 2,
                    error_category="rate_limit" if index == 2 else None,
                    retry_after_s=1.0 if index == 2 else None,
                )
                finished.append(time.monotonic())
                if index == 0:
                    cooldown_after_first.append(
                        budget.endpoint_pool.cooldown_until.get("authorized", 0) - time.monotonic()
                    )

            tasks = [asyncio.create_task(request(i)) for i in range(3)]
            await all_started.wait()
            assert len(budget.attempts) == 3
            gate.set()
            await asyncio.gather(*tasks)
            return {
                "final_advice": StrategyAdvice(symbol="BTCUSDT", decision_type="no_trade"),
                "news_research": {},
                "runtime_events": [{"ok": False}],
            }

    monkeypatch.setattr(live, "build_graph", lambda *args, **kwargs: Graph())
    args = argparse.Namespace(
        output=str(tmp_path / "output"),
        split="dev",
        limit=1,
        cases=str(Path(__file__).parents[1] / "evals/news_research_v1/cases.json"),
        base_url="https://example.invalid/v1",
        model="fixture",
        ledger=None,
        case_timeout=10,
        input_per_million=1.0,
        output_per_million=1.0,
        request_interval_s=0.04,
    )
    report = await evaluate(args, "unused")
    assert report["executed_runs"] == 1
    assert cooldown_after_first[0] > 0.025
    assert sent[1] - finished[0] >= 0.025
    snapshot = json.loads((Path(args.output) / "dev-ledger.json").read_text(encoding="utf-8"))
    assert snapshot["endpoint_cooldown_remaining_s"]["authorized"] > 0.5
