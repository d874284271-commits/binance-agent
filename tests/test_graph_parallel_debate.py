from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import Field

from binance_agent.agents.graph import build_graph
from binance_agent.agents.nodes import _sorted_debate_turns
from test_graph_offline import _initial_state


class _SlowDebateModel(FakeListChatModel):
    start_times: list[float] = Field(default_factory=list)
    human_payloads: list[str] = Field(default_factory=list)

    async def ainvoke(  # type: ignore[override]
        self, messages: Any, *args: Any, **kwargs: Any
    ) -> Any:
        self.start_times.append(time.perf_counter())
        self.human_payloads.append(str(messages[-1].content))
        await asyncio.sleep(0.05)
        return await super().ainvoke(messages, *args, **kwargs)


def _fakes() -> dict[str, FakeListChatModel]:
    return {
        "technical": FakeListChatModel(
            responses=['{"stance":"bullish","confidence":0.6,"summary":"technical"}']
        ),
        "funding": FakeListChatModel(
            responses=['{"stance":"bearish","confidence":0.5,"summary":"funding"}']
        ),
        "news_sentiment": FakeListChatModel(
            responses=['{"stance":"neutral","confidence":0.5,"summary":"news"}']
        ),
        "bull_debater": _SlowDebateModel(
            responses=[
                '{"argument":"bull r0","claims":[{"statement":"bull","evidence":"x=1","confidence":0.5}]}',
                '{"argument":"bull r1","claims":[{"statement":"bull","evidence":"x=2","confidence":0.5}]}',
            ]
        ),
        "bear_debater": _SlowDebateModel(
            responses=[
                '{"argument":"bear r0","claims":[{"statement":"bear","evidence":"y=1","confidence":0.5}]}',
                '{"argument":"bear r1","claims":[{"statement":"bear","evidence":"y=2","confidence":0.5}]}',
            ]
        ),
        "research_manager": FakeListChatModel(responses=["summary"]),
        "strategy_synth": FakeListChatModel(
            responses=[
                '{"direction":"neutral","decision_type":"no_trade","entry_zone":null,'
                '"stop_loss":null,"targets":[],"confidence":0.3,"rationale":"neutral"}'
            ]
        ),
    }


async def test_parallel_debate_rounds_produce_two_turns_per_round() -> None:
    fakes = _fakes()
    graph = build_graph(lambda agent: fakes[agent])
    state = _initial_state(with_event=False)
    state["max_debate_rounds"] = 2

    result = await graph.ainvoke(state)

    ordered = _sorted_debate_turns(result["debate_turns"])
    assert result["debate_round"] == 2
    assert [(turn.side, turn.round) for turn in ordered] == [
        ("bull", 0),
        ("bear", 0),
        ("bull", 1),
        ("bear", 1),
    ]

    bull_model = fakes["bull_debater"]
    bear_model = fakes["bear_debater"]
    assert isinstance(bull_model, _SlowDebateModel)
    assert isinstance(bear_model, _SlowDebateModel)
    assert len(bull_model.start_times) == 2
    assert len(bear_model.start_times) == 2
    assert abs(bull_model.start_times[0] - bear_model.start_times[0]) < 0.03
    assert abs(bull_model.start_times[1] - bear_model.start_times[1]) < 0.03


async def test_second_round_receives_prior_round_turns() -> None:
    fakes = _fakes()
    graph = build_graph(lambda agent: fakes[agent])
    state = _initial_state(with_event=False)
    state["max_debate_rounds"] = 2

    await graph.ainvoke(state)

    bull_model = fakes["bull_debater"]
    bear_model = fakes["bear_debater"]
    assert isinstance(bull_model, _SlowDebateModel)
    assert isinstance(bear_model, _SlowDebateModel)

    for model in (bull_model, bear_model):
        first = json.loads(model.human_payloads[0])
        second = json.loads(model.human_payloads[1])
        # 首轮:无前轮论点,独立分析
        assert first["prior_round_turns"] == []
        # 次轮:能看到第 0 轮双方全文(argument + claims),可针对性交锋
        sides = [(t["side"], t["round"]) for t in second["prior_round_turns"]]
        assert sides == [("bull", 0), ("bear", 0)]
        narratives = {
            source: section["text"]
            for section in second["derived_context"]
            for source in section["source_ids"]
        }
        arguments = {narratives[t["argument_context_id"]] for t in second["prior_round_turns"]}
        assert arguments == {"bull r0", "bear r0"}
        assert all(t["structured_claims"] for t in second["prior_round_turns"])


def _claim_reply(argument: str, path: str, value: Any) -> str:
    return json.dumps(
        {
            "argument": argument,
            "claims": [
                {
                    "statement": argument,
                    "evidence": f"{path}={value}",
                    "source_path": path,
                    "source_value": value,
                    "confidence": 0.5,
                }
            ],
        },
        ensure_ascii=False,
    )


async def test_debate_stops_early_when_round_adds_no_new_verified_evidence() -> None:
    fakes = _fakes()
    fakes["bull_debater"] = _SlowDebateModel(
        responses=[
            _claim_reply("bull r0", "snapshot_data.ticker_24h.lastPrice", "100"),
            _claim_reply("bull r1 换措辞", "snapshot_data.ticker_24h.lastPrice", "100"),
            "SENTINEL",
        ]
    )
    fakes["bear_debater"] = _SlowDebateModel(
        responses=[
            _claim_reply("bear r0", "snapshot_data.funding.last_funding_rate", 0.0001),
            _claim_reply("bear r1 反驳", "snapshot_data.ticker_24h.lastPrice", "100"),
            "SENTINEL",
        ]
    )
    graph = build_graph(lambda agent: fakes[agent])
    state = _initial_state(with_event=False)
    state["max_debate_rounds"] = 4

    result = await graph.ainvoke(state)

    assert result["debate_round"] == 2
    assert len(result["debate_turns"]) == 4
    assert result["debate_grounding"]["convergence"] == {
        "rounds_run": 2,
        "max_rounds": 4,
        "stop_reason": "no_new_evidence",
    }
    bull_model = fakes["bull_debater"]
    assert isinstance(bull_model, _SlowDebateModel)
    assert len(bull_model.start_times) == 2
    second = json.loads(bull_model.human_payloads[1])
    assert second["previously_cited_paths"] == [
        "snapshot_data.funding.last_funding_rate",
        "snapshot_data.ticker_24h.lastPrice",
    ]


async def test_debate_runs_to_max_rounds_while_new_evidence_keeps_arriving() -> None:
    fakes = _fakes()
    fakes["bull_debater"] = _SlowDebateModel(
        responses=[
            _claim_reply("bull r0", "snapshot_data.ticker_24h.lastPrice", "100"),
            _claim_reply("bull r1", "snapshot_data.ticker_24h.highPrice", "105"),
            _claim_reply("bull r2", "snapshot_data.ticker_24h.priceChangePercent", "1.2"),
        ]
    )
    fakes["bear_debater"] = _SlowDebateModel(
        responses=[
            _claim_reply("bear r0", "snapshot_data.funding.last_funding_rate", 0.0001),
            _claim_reply("bear r1", "snapshot_data.ticker_24h.lowPrice", "95"),
            _claim_reply("bear r2", "snapshot_data.ticker_24h.volume", "1000"),
        ]
    )
    graph = build_graph(lambda agent: fakes[agent])
    state = _initial_state(with_event=False)
    state["max_debate_rounds"] = 3

    result = await graph.ainvoke(state)

    assert result["debate_round"] == 3
    assert len(result["debate_turns"]) == 6
    assert result["debate_grounding"]["convergence"]["stop_reason"] == "max_rounds"
    assert [r["new_verified_sources"] for r in result["debate_grounding"]["rounds"]] == [2, 2, 2]
