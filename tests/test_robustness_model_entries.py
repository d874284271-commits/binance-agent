"""The same invalid model values must be rejected at every structured entry."""

from __future__ import annotations

import json
from typing import Any

import pytest

from binance_agent.agents.nodes import bear_debater, bull_debater
from binance_agent.agents.output_validation import (
    analyst_output,
    debate_output,
    evidence_verifier_output,
    final_verifier_output,
    planner_output,
    researcher_output,
    strategy_output,
    supervisor_output,
    supervisor_review_output,
)
from binance_agent.agents.runtime import AgentRuntime
from test_agents_nodes import _state
from test_robustness_core import CountingModel


@pytest.mark.parametrize("debater", [bull_debater, bear_debater])
@pytest.mark.parametrize("confidence", [10**400, True, float("nan"), float("inf"), [], "0.5"])
async def test_debate_invalid_confidence_is_schema_failure(debater: Any, confidence: Any) -> None:
    data = {
        "argument": "argument",
        "claims": [{"statement": "claim", "evidence": "price=1", "confidence": confidence}],
    }
    model = CountingModel(json.dumps(data))
    result = await debater(_state(), model)
    event = result["runtime_events"][0]
    assert event["ok"] is False
    assert event["error_category"] == "schema_invalid"
    assert result["debate_turns"][0].structured_claims == []
    assert model.calls == 1


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "validator,data",
    [
        (analyst_output, {"stance": "bullish", "summary": "x", "confidence": 10**400}),
        (
            strategy_output,
            {"direction": "neutral", "decision_type": "no_trade", "confidence": True},
        ),
        (debate_output, {"argument": {}, "claims": []}),
        (debate_output, {"argument": "x", "claims": [{"statement": "x", "evidence": []}]}),
        (planner_output, {"action": "continue_to_synth", "confidence_cap": 10**400}),
        (supervisor_output, {"tasks": [{"candidate_ids": {}, "question": "x"}]}),
        (supervisor_review_output, {"follow_up": "true", "query": {}}),
        (researcher_output, {"facts": [], "gaps": [], "conflicts": [], "confidence": 10**400}),
        (evidence_verifier_output, {"verdicts": [{"index": True, "supported": True}]}),
        (final_verifier_output, {"supported": "true"}),
    ],
)
async def test_structured_contracts_reject_invalid_first_and_repaired_output(
    validator: Any, data: Any, repair: bool
) -> None:
    content = json.dumps(data)
    responses = ["not json", content] if repair else [content]

    class Model(CountingModel):
        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            self.content = responses[min(self.calls, len(responses) - 1)]
            return await super().ainvoke(*args, **kwargs)

    model = Model()
    result = await AgentRuntime(model, max_retries=2).invoke_structured(
        "entry", "sys", {}, fallback_data={"fallback": True}, output_validator=validator
    )
    assert not result.ok and result.error_category == "schema_invalid"
    assert result.data == {"fallback": True}
    assert model.calls == len(responses)


async def test_planner_numeric_contract_applies_before_coercion() -> None:
    from binance_agent.agents.nodes import _run_strategy_action_planner
    from binance_agent.agents.tools import build_default_tool_registry

    model = CountingModel(json.dumps({"action": "continue_to_synth", "confidence_cap": 10**400}))
    result = await _run_strategy_action_planner(
        _state(),
        llm=model,
        registry=build_default_tool_registry(),
        max_steps=1,
        steps=[],
        tool_events=[],
    )
    assert result["runtime_events"][0]["error_category"] == "schema_invalid"
    assert result["planner_decision"]["decision_type"] == "watchlist"
    assert model.calls == 1


def test_output_schema_changes_resume_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    from binance_agent.app import server
    from binance_agent.infra.settings import Settings

    before = server._research_config_fingerprint(Settings())
    monkeypatch.setattr(server, "OUTPUT_SCHEMA_VERSION", "incompatible-output")
    assert server._research_config_fingerprint(Settings()) != before


@pytest.mark.parametrize(
    "target", ["research_supervisor", "news_researcher", "evidence_verifier", "supervisor_review"]
)
async def test_research_entry_contracts_are_connected_to_graph(target: str) -> None:
    from langchain_core.messages import AIMessage

    from binance_agent.agents.research import build_news_research_graph
    from binance_agent.infra.settings import ResearchConfig
    from binance_agent.intel.evidence import evidence_document
    from test_news_research_graph import ResearchModel, initial, model_for

    class Model(ResearchModel):
        async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
            prompt = str(messages[0].content)
            entry = (
                "supervisor_review"
                if "正在复核" in prompt
                else "research_supervisor"
                if "任务主管" in prompt
                else "news_researcher"
                if "金融事件研究者" in prompt
                else "evidence_verifier"
            )
            if entry == target:
                return AIMessage(content='{"unexpected":true}')
            return await super().ainvoke(messages, *args, **kwargs)

    async def fetch(url: str) -> Any:
        return evidence_document(
            url=url, text="Bitcoin ETF application is pending, not approved.", level="body"
        )

    model = Model(responses=["unused"], reject_support=target == "supervisor_review")
    graph = build_news_research_graph(
        model_for(model), config=ResearchConfig(enabled=True), fetch=fetch
    )
    result = await graph.ainvoke(initial())
    role = "research_supervisor" if target == "supervisor_review" else target
    events = [e for e in result["runtime_events"] if e["agent"] == role]
    assert any(not e["ok"] and e["error_category"] == "schema_invalid" for e in events)
    if target != "research_supervisor":
        assert result["news_research"]["critical_unresolved"]
    else:
        # Invalid allocation may use the existing deterministic candidate plan;
        # it cannot bypass the subsequent source verification.
        assert all(f["support"] == "model_supported" for f in result["news_research"]["facts"])


async def test_final_verifier_wrong_boolean_is_unknown_and_conservative() -> None:
    from binance_agent.agents.research import finalize_evidence_advice
    from binance_agent.agents.state import StrategyAdvice

    model = CountingModel('{"supported":"true"}')
    result = await finalize_evidence_advice(
        {"candidate_advice": StrategyAdvice(symbol="BTCUSDT", confidence=0.9)}, llm=model
    )
    event = result["runtime_events"][0]
    assert not event["ok"] and event["error_category"] == "schema_invalid"
    assert result["final_advice"].direction == "neutral"
    assert result["final_advice"].confidence <= 0.35
    assert model.calls == 1
