from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

from binance_agent.agents.budget import RunBudget, run_budget_scope
from binance_agent.agents.research import build_news_research_graph, finalize_evidence_advice
from binance_agent.agents.state import StrategyAdvice
from binance_agent.infra.settings import ResearchConfig
from binance_agent.intel.evidence import evidence_document


class ResearchModel(FakeListChatModel):
    seen_payloads: list[str] = Field(default_factory=list)
    invalid_quote: bool = False
    reject_support: bool = False

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        prompt = str(messages[0].content)
        self.seen_payloads.append(str(messages[-1].content))
        payload = json.loads(messages[-1].content)
        if "任务主管" in prompt:
            data = {
                "tasks": [
                    {
                        "candidate_ids": [k],
                        "question": "核实公告状态",
                        "critical": True,
                        "completion": "保留申请与批准区别",
                    }
                    for k in payload["candidates"]
                ]
            }
        elif "金融事件研究者" in prompt:
            docs = payload["documents"]
            bodies = [d for d in docs if d["level"] == "body"]
            doc = (bodies or docs)[0]
            data = {
                "facts": [
                    {
                        "statement": "Bitcoin ETF application is pending, not approved.",
                        "evidence_id": doc["evidence_id"],
                        "quote": "fabricated quote" if self.invalid_quote else doc["text"],
                        "confirmation_status": "confirmed_in_source",
                    }
                ],
                "stance": "neutral",
                "gaps": [],
                "conflicts": [],
            }
        elif "verdicts" in prompt:
            data = {
                "verdicts": [
                    {"index": i, "supported": not self.reject_support}
                    for i in range(len(payload["facts"]))
                ]
            }
        else:
            data = {"supported": True}
        return AIMessage(
            content=json.dumps(data),
            usage_metadata={"input_tokens": 30, "output_tokens": 20, "total_tokens": 50},
        )


def initial() -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "analysis_at": "2026-09-18T00:00:00Z",
        "research_question": "ETF是否获批?",
        "intel_snapshot": {
            "news_items": [
                {
                    "title": "Bitcoin ETF approved!",
                    "summary": "Application submitted.",
                    "url": "https://news.example.com/etf",
                    "source": "fixture",
                    "published_at": None,
                }
            ]
        },
    }


def model_for(model: ResearchModel) -> Any:
    return lambda _: model


async def test_body_corrects_misleading_title_and_keeps_citation() -> None:
    calls = []

    async def fetch(url: str) -> Any:
        calls.append(url)
        return evidence_document(
            url=url, text="Bitcoin ETF application is pending, not approved.", level="body"
        )

    model = ResearchModel(responses=["unused"])
    graph = build_news_research_graph(
        model_for(model), config=ResearchConfig(enabled=True), fetch=fetch
    )
    with run_budget_scope(RunBudget(max_requests=12)):
        result = await graph.ainvoke(initial())
    research = result["news_research"]
    assert calls == ["https://news.example.com/etf"]
    assert not research["critical_unresolved"]
    assert research["facts"][0]["statement"].endswith("not approved.")
    assert research["facts"][0]["support"] == "model_supported"
    assert "text" not in research["documents"][0]
    assert len(result["analyst_reports"]) == 1


async def test_macro_announcement_without_crypto_name_is_researched() -> None:
    calls: list[str] = []

    async def fetch(url: str) -> Any:
        calls.append(url)
        return evidence_document(url=url, text="Inflation rose 0.3 percent.", level="body")

    state = initial()
    state["research_question"] = "核对所给官方材料的事实、单位和条件"
    state["intel_snapshot"]["news_items"] = [
        {"title": "Consumer Price Index — March", "url": "https://macro.example/cpi"}
    ]
    graph = build_news_research_graph(
        model_for(ResearchModel(responses=["unused"])),
        config=ResearchConfig(enabled=True, workflow="single"),
        fetch=fetch,
    )
    result = await graph.ainvoke(state)
    assert calls == ["https://macro.example/cpi"]
    assert any(d["level"] == "body" for d in result["evidence_documents"])


@pytest.mark.parametrize("failure", ["body_unavailable", "quote_invalid", "not_supported"])
async def test_critical_gap_cannot_bypass_final_gate(failure: str) -> None:
    model = ResearchModel(
        responses=["unused"],
        invalid_quote=failure == "quote_invalid",
        reject_support=failure == "not_supported",
    )

    async def fetch(url: str) -> Any:
        if failure == "body_unavailable":
            return evidence_document(url=url, status="unavailable", level="body")
        return evidence_document(
            url=url, text="Bitcoin ETF application is pending, not approved.", level="body"
        )

    graph = build_news_research_graph(
        model_for(model), config=ResearchConfig(enabled=True), fetch=fetch
    )
    result = await graph.ainvoke(initial())
    assert result["news_research"]["critical_unresolved"]
    if failure != "body_unavailable":
        assert result["news_research"]["facts"] == []
    result["final_advice"] = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        decision_type="trade",
        entry_zone="100",
        stop_loss="90",
        targets=["120"],
    )
    final = await finalize_evidence_advice(result, llm=model)
    assert final["final_advice"].decision_type == "watchlist"
    assert final["final_advice"].targets == []


async def test_replay_reuses_captured_news_without_fetch_or_model() -> None:
    async def forbidden(url: str) -> Any:
        raise AssertionError("replay must not fetch")

    graph = build_news_research_graph(
        model_for(ResearchModel(responses=["unused"])),
        config=ResearchConfig(enabled=True),
        fetch=forbidden,
    )
    state = initial()
    state.update(
        research_replay=True, news_research={"facts": [], "tasks": [], "gaps": ["saved gap"]}
    )
    with run_budget_scope(RunBudget(max_requests=0)):
        result = await graph.ainvoke(state)
    assert result["news_research"]["gaps"] == ["saved gap"]


async def test_pending_body_placeholder_is_not_reused_as_successful_evidence() -> None:
    url = "https://news.example.com/etf"
    pending = evidence_document(url=url, level="body", status="unavailable")
    calls: list[str] = []

    async def fetch(target: str) -> Any:
        calls.append(target)
        return evidence_document(
            url=target, text="Bitcoin ETF application is pending, not approved.", level="body"
        )

    graph = build_news_research_graph(
        model_for(ResearchModel(responses=["unused"])),
        config=ResearchConfig(enabled=True),
        fetch=fetch,
        load=lambda _run_id: [pending],
    )
    state = initial()
    state["research_run_id"] = "run-pending"
    result = await graph.ainvoke(state)
    assert calls == [url]
    assert result["news_research"]["facts"]


async def test_legacy_replay_cannot_upgrade_to_original_evidence() -> None:
    async def forbidden(url: str) -> Any:
        raise AssertionError("legacy replay must not fetch")

    graph = build_news_research_graph(
        model_for(ResearchModel(responses=["unused"])),
        config=ResearchConfig(enabled=True),
        fetch=forbidden,
    )
    state = initial()
    state.update(research_replay=True, research_legacy=True, news_research={})
    with run_budget_scope(RunBudget(max_requests=0)):
        result = await graph.ainvoke(state)
    assert result["news_research"]["legacy_no_original_evidence"] is True
    assert result["news_research"]["facts"] == []


async def test_critical_unconfirmed_fact_forces_watchlist() -> None:
    class UnconfirmedModel(ResearchModel):
        async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
            response = await super().ainvoke(messages, *args, **kwargs)
            payload = json.loads(response.content)
            for fact in payload.get("facts", []):
                fact["confirmation_status"] = "unconfirmed"
            response.content = json.dumps(payload)
            return response

    model = UnconfirmedModel(responses=["unused"])

    async def fetch(url: str) -> Any:
        return evidence_document(
            url=url,
            text="An anonymous source alleges a Bitcoin exchange hack; it is unconfirmed.",
            level="body",
        )

    graph = build_news_research_graph(
        model_for(model), config=ResearchConfig(enabled=True), fetch=fetch
    )
    result = await graph.ainvoke(initial())
    assert result["news_research"]["critical_unresolved"]
    result["final_advice"] = StrategyAdvice(
        symbol="BTCUSDT",
        direction="long",
        decision_type="trade",
        entry_zone="100",
        stop_loss="90",
        targets=["120"],
    )
    final = await finalize_evidence_advice(result, llm=model)
    assert final["final_advice"].decision_type == "watchlist"


async def test_parallel_tasks_join_once_and_preserve_all_failures() -> None:
    state = initial()
    state["intel_snapshot"]["news_items"].append(
        {"title": "Bitcoin exchange hack", "url": "https://news.example.com/hack"}
    )
    graph = build_news_research_graph(
        model_for(ResearchModel(responses=["unused"])), config=ResearchConfig(enabled=True)
    )
    result = await graph.ainvoke(state)
    assert len(result["news_research"]["tasks"]) == 2
    assert len(result["analyst_reports"]) == 1
    assert result["news_research"]["critical_unresolved"]


async def test_full_graph_joins_analysts_and_gates_news(tmp_path: Any) -> None:
    from langgraph.checkpoint.memory import MemorySaver

    from binance_agent.agents.graph import build_graph
    from binance_agent.app.run_store import FileRunStore
    from test_graph_offline import _fakes, _initial_state

    fakes = _fakes()
    model = ResearchModel(responses=["unused"])
    fakes.update(
        dict.fromkeys(["research_supervisor", "news_researcher", "evidence_verifier"], model)
    )
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="BTCUSDT", market="fapi")
    state = _initial_state(with_event=False)
    state["intel_snapshot"]["news_items"] = initial()["intel_snapshot"]["news_items"]
    state["research_run_id"] = run_id
    state["research_question"] = "核实是否批准"
    graph = build_graph(
        lambda name: fakes[name],
        research_config=ResearchConfig(enabled=True),
        checkpointer=MemorySaver(),
        research_persist=store.save_evidence,
        research_load=store.load_evidence,
    )
    with run_budget_scope(RunBudget(max_requests=40)):
        result = await graph.ainvoke(state, {"configurable": {"thread_id": run_id}})
    assert len(result["analyst_reports"]) == 3
    assert result["news_research"]["critical_unresolved"]
    assert result["final_advice"].decision_type in {"watchlist", "no_trade"}
    assert not result["debate_turns"]  # no argument can resolve missing facts
    assert store.load_evidence(run_id)


async def test_single_workflow_does_not_run_parallel_analyst_roles() -> None:
    from binance_agent.agents.graph import build_graph
    from test_graph_offline import _fakes, _initial_state

    models = _fakes()
    model = ResearchModel(responses=["unused"])
    models.update(
        dict.fromkeys(["research_supervisor", "news_researcher", "evidence_verifier"], model)
    )
    state = _initial_state(with_event=False)
    state["intel_snapshot"]["news_items"] = initial()["intel_snapshot"]["news_items"]
    graph = build_graph(
        lambda name: models[name], research_config=ResearchConfig(enabled=True, workflow="single")
    )
    result = await graph.ainvoke(state)
    agents = {e["agent"] for e in result["runtime_events"]}
    assert (
        not {
            "technical",
            "funding",
            "research_supervisor",
            "research_manager",
            "bull_debater",
            "bear_debater",
        }
        & agents
    )
    assert len(result["analyst_reports"]) == 1
    assert result["final_advice"].decision_type in {"watchlist", "no_trade"}


async def test_title_mode_never_sends_summary_to_any_model() -> None:
    model = ResearchModel(responses=["unused"])
    state = initial()
    state["intel_snapshot"]["news_items"][0]["summary"] = "SUMMARY_MUST_NOT_LEAK"
    graph = build_news_research_graph(
        model_for(model), config=ResearchConfig(enabled=True, input_mode="title")
    )
    await graph.ainvoke(state)
    assert model.seen_payloads
    assert all("SUMMARY_MUST_NOT_LEAK" not in text for text in model.seen_payloads)


async def test_zero_document_allocations_cannot_fetch() -> None:
    model = ResearchModel(responses=["unused"])
    state = initial()
    state["intel_snapshot"]["news_items"] = [
        {"title": f"Bitcoin {word} announcement", "url": f"https://news.example.com/{word}"}
        for word in ["funding", "custody", "exchange", "mining", "wallet", "protocol"]
    ]
    calls = []

    async def fetch(url: str) -> Any:
        calls.append(url)
        return evidence_document(
            url=url, text="Bitcoin ETF application is pending, not approved.", level="body"
        )

    graph = build_news_research_graph(
        model_for(model), config=ResearchConfig(enabled=True, max_documents=1), fetch=fetch
    )
    with run_budget_scope(RunBudget(max_requests=60)):
        result = await graph.ainvoke(state)
    assert len(result["news_research"]["tasks"]) == 6
    assert len(calls) == 1


async def test_body_tools_share_the_run_request_budget() -> None:
    model = ResearchModel(responses=["unused"])
    calls = []

    async def fetch(url: str) -> Any:
        calls.append(url)
        return evidence_document(url=url, text="body", level="body")

    graph = build_news_research_graph(
        model_for(model), config=ResearchConfig(enabled=True), fetch=fetch
    )
    budget = RunBudget(max_requests=1)
    with run_budget_scope(budget):
        result = await graph.ainvoke(initial())
    assert not calls
    assert budget.requests_reserved == 1
    assert result["news_research"]["critical_unresolved"]
