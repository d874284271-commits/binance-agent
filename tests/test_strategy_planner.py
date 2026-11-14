"""Bounded strategy planner tests."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import MemorySaver

from binance_agent.agents.graph import build_graph
from binance_agent.agents.tools import build_default_tool_registry


class _CountingToolRegistry:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._delegate = build_default_tool_registry()

    async def invoke(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append(name)
        return await self._delegate.invoke(name, payload, context=context)


class _FailingToolRegistry:
    async def invoke(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> Any:
        raise RuntimeError(f"{name} tool boom")


def _fakes(*, strategy_json: str | None = None) -> dict[str, BaseChatModel]:
    return {
        "technical": FakeListChatModel(
            responses=['{"stance":"bullish","confidence":0.8,"summary":"technical"}']
        ),
        "funding": FakeListChatModel(
            responses=['{"stance":"bullish","confidence":0.8,"summary":"funding"}']
        ),
        "news_sentiment": FakeListChatModel(
            responses=['{"stance":"bullish","confidence":0.8,"summary":"news"}']
        ),
        "bull_debater": FakeListChatModel(responses=["bull"]),
        "bear_debater": FakeListChatModel(responses=["bear"]),
        "research_manager": FakeListChatModel(responses=["research"]),
        "strategy_synth": FakeListChatModel(
            responses=[
                strategy_json
                or (
                    '{"direction":"long","decision_type":"trade","entry_zone":"100",'
                    '"stop_loss":"95","targets":["115"],"confidence":0.8,'
                    '"rationale":"planner test"}'
                )
            ]
        ),
    }


async def test_strategy_planner_disabled_keeps_default_graph_behavior() -> None:
    graph = build_graph(lambda agent: _fakes()[agent])

    result = await graph.ainvoke(_initial_state(complete_market=True))

    assert result.get("planner_steps") in (None, [])
    assert result.get("planner_decision") in (None, {})
    assert result["final_advice"].decision_type == "trade"


async def test_data_preflight_blocks_critical_market_before_strategy_planner() -> None:
    graph = build_graph(
        lambda agent: _fakes()[agent],
        enable_strategy_planner=True,
        planner_max_steps=3,
    )

    result = await graph.ainvoke(_initial_state(complete_market=False))

    assert result["preflight_report"]["status"] == "blocked"
    assert result["planner_decision"] is None
    assert result["planner_steps"] == []
    assert result["analyst_reports"] == []
    assert result["final_advice"].decision_type == "no_trade"
    assert result["final_advice"].entry_zone is None


async def test_strategy_planner_tool_failure_downgrades_without_graph_failure() -> None:
    graph = build_graph(
        lambda agent: _fakes()[agent],
        enable_strategy_planner=True,
        planner_max_steps=3,
        planner_tool_registry=_FailingToolRegistry(),
    )

    result = await graph.ainvoke(_initial_state(complete_market=True))

    assert result["planner_decision"]["decision_type"] == "no_trade"
    assert "tool boom" in result["planner_decision"]["reason"]
    assert result["final_advice"].decision_type == "no_trade"
    assert result["final_advice"].no_trade_reason


async def test_strategy_planner_max_steps_fallback_caps_planning() -> None:
    graph = build_graph(
        lambda agent: _fakes()[agent],
        enable_strategy_planner=True,
        planner_max_steps=1,
    )

    result = await graph.ainvoke(_initial_state(complete_market=True))

    assert len(result["planner_steps"]) == 1
    assert result["planner_decision"]["decision_type"] == "watchlist"
    assert "max_steps" in result["planner_decision"]["reason"]
    assert result["final_advice"].decision_type in {"watchlist", "no_trade"}


async def test_strategy_planner_resume_does_not_repeat_completed_planner_step() -> None:
    registry = _CountingToolRegistry()
    graph = build_graph(
        lambda agent: _fakes()[agent],
        checkpointer=MemorySaver(),
        interrupt_after=["strategy_planner"],
        enable_strategy_planner=True,
        planner_max_steps=3,
        planner_tool_registry=registry,
    )
    config = {"configurable": {"thread_id": "planner-resume"}}

    updates = []
    async for update in graph.astream(
        _initial_state(complete_market=True),
        config=config,
        stream_mode="updates",
    ):
        updates.append(update)

    assert any("__interrupt__" in update for update in updates)
    calls_after_interrupt = list(registry.calls)
    result = await graph.ainvoke(None, config=config)

    assert result["final_advice"].decision_type == "trade"
    assert registry.calls == calls_after_interrupt


def _initial_state(*, complete_market: bool) -> dict[str, Any]:
    rows = [
        {"open": i, "high": i + 2, "low": i - 2, "close": i + 0.5, "volume": 10}
        for i in range(1, 50)
    ]
    market_snapshot = (
        {
            "klines": {"5m": rows, "1h": rows, "4h": rows, "1d": rows},
            "funding": {"last_funding_rate": 0.0001},
            "ticker_24h": {
                "lastPrice": "100",
                "priceChangePercent": "1.2",
                "highPrice": "105",
                "lowPrice": "95",
                "volume": "1000",
                "quoteVolume": "100000",
            },
            "errors": [],
        }
        if complete_market
        else {
            "klines": {},
            "funding": {"last_funding_rate": 0.0001},
            "ticker_24h": {},
            "errors": ["klines[1h]: boom"],
        }
    )
    return {
        "symbol": "BTCUSDT",
        "market": "fapi",
        "market_snapshot": market_snapshot,
        "intel_snapshot": {
            "sentiment_features": {"long_short_ratio": 1.2},
            "news_items": [{"title": "headline"}],
            "macro_events": [],
            "source_status": {"rss": "ok", "reddit": "ok"},
        },
        "analyst_reports": [],
        "debate_turns": [],
        "runtime_events": [],
        "tool_events": [],
        "planner_steps": [],
        "planner_decision": None,
        "debate_round": 0,
        "max_debate_rounds": 1,
        "research_summary": None,
        "risk_assessment": None,
        "risk_gate_result": None,
        "final_advice": None,
    }
