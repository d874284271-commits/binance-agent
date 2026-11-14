from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field

from binance_agent.agents.graph import build_graph
from binance_agent.agents.preflight import run_data_preflight


def _complete_market(last_price: str = "101") -> dict[str, Any]:
    rows = [
        {"open": i, "high": i + 2, "low": i - 2, "close": i + 0.5, "volume": 10}
        for i in range(1, 50)
    ]
    return {
        "klines": {"5m": rows, "1h": rows, "4h": rows, "1d": rows},
        "ticker_24h": {
            "lastPrice": last_price,
            "highPrice": "105",
            "lowPrice": "95",
            "volume": "1000",
            "quoteVolume": "100000",
        },
        "errors": [],
    }


def _broken_market() -> dict[str, Any]:
    return {"klines": {}, "ticker_24h": {}, "errors": ["market unavailable"]}


def _good_intel() -> dict[str, Any]:
    return {
        "news_items": [{"title": "headline"}],
        "sentiment_features": {"long_short_ratio": 1.1},
        "source_status": {"rss": "ok", "reddit": "ok"},
    }


def _broken_intel() -> dict[str, Any]:
    return {
        "news_items": [],
        "sentiment_features": {},
        "source_status": {"rss": "error: timeout", "reddit": "error: http 403"},
    }


def _state() -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "market": "fapi",
        "market_snapshot": _broken_market(),
        "intel_snapshot": _good_intel(),
    }


def _graph_state() -> dict[str, Any]:
    return {
        **_state(),
        "analyst_reports": [],
        "debate_turns": [],
        "runtime_events": [],
        "tool_events": [],
        "planner_steps": [],
        "planner_decision": None,
        "debate_round": 0,
        "max_debate_rounds": 1,
        "research_summary": None,
        "preflight_report": None,
        "debate_grounding": None,
        "risk_assessment": None,
        "risk_gate_result": None,
        "final_advice": None,
    }


async def test_preflight_refetches_missing_market_once_and_promotes_snapshot() -> None:
    calls: list[str] = []

    async def build_market_snapshot(client: Any, symbol: str) -> dict[str, Any]:
        calls.append(symbol)
        return _complete_market()

    update = await run_data_preflight(
        _state(),
        tool_context={
            "rest_clients": {"fapi": object()},
            "build_market_snapshot": build_market_snapshot,
        },
    )

    assert calls == ["BTCUSDT"]
    assert update["market_snapshot"]["ticker_24h"]["lastPrice"] == "101"
    assert update["preflight_report"]["status"] == "ready"
    assert update["preflight_report"]["requested_refetches"] == ["market"]
    assert update["preflight_report"]["final_quality"]["trade_blocked"] is False
    assert update["analysis_at"] == update["preflight_report"]["analysis_at"]
    assert update.get("final_advice") is None


async def test_preflight_refetches_market_and_intel_at_most_once_each() -> None:
    market_calls: list[str] = []

    async def build_market_snapshot(client: Any, symbol: str) -> dict[str, Any]:
        market_calls.append(symbol)
        return _complete_market()

    class _Aggregator:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def gather(self, symbol: str) -> dict[str, Any]:
            self.calls.append(symbol)
            return _good_intel()

    aggregator = _Aggregator()
    state = _state()
    state["intel_snapshot"] = _broken_intel()

    update = await run_data_preflight(
        state,
        tool_context={
            "rest_clients": {"fapi": object()},
            "build_market_snapshot": build_market_snapshot,
            "aggregator": aggregator,
        },
    )

    assert market_calls == ["BTCUSDT"]
    assert aggregator.calls == ["BTCUSDT"]
    assert update["preflight_report"]["requested_refetches"] == ["market", "intel"]
    assert [attempt["kind"] for attempt in update["preflight_report"]["refetch_attempts"]] == [
        "market",
        "intel",
    ]
    assert update["preflight_report"]["status"] == "ready"


async def test_state_can_disable_refetch_for_snapshot_only_replay() -> None:
    calls: list[str] = []

    async def build_market_snapshot(client: Any, symbol: str) -> dict[str, Any]:
        calls.append(symbol)
        return _complete_market()

    state = _state()
    state["preflight_refetch_enabled"] = False

    update = await run_data_preflight(
        state,
        tool_context={
            "rest_clients": {"fapi": object()},
            "build_market_snapshot": build_market_snapshot,
        },
    )

    assert calls == []
    assert update["preflight_report"]["refetch_enabled"] is False
    assert update["preflight_report"]["status"] == "blocked"


class _CountingModel(FakeListChatModel):
    calls: int = Field(default=0)

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self.calls += 1
        return await super().ainvoke(messages, *args, **kwargs)


async def test_graph_preflight_failure_returns_no_trade_without_calling_llms() -> None:
    model = _CountingModel(responses=["unused"])

    async def fail_market_refetch(client: Any, symbol: str) -> dict[str, Any]:
        raise RuntimeError("market refetch boom")

    graph = build_graph(
        lambda agent: model,
        preflight_tool_context={
            "rest_clients": {"fapi": object()},
            "build_market_snapshot": fail_market_refetch,
        },
    )
    result = await graph.ainvoke(_graph_state())

    assert result["preflight_report"]["status"] == "blocked"
    assert result["final_advice"].decision_type == "no_trade"
    assert "market refetch boom" in result["final_advice"].no_trade_reason
    assert model.calls == 0


async def test_graph_analysts_consume_preflight_promoted_snapshot() -> None:
    from test_graph_offline import _fakes

    captured: dict[str, Any] = {}

    class _CaptureTechnical(FakeListChatModel):
        async def ainvoke(  # type: ignore[override]
            self, messages: Any, *args: Any, **kwargs: Any
        ) -> Any:
            captured["payload"] = str(messages[-1].content)
            return await super().ainvoke(messages, *args, **kwargs)

    async def build_market_snapshot(client: Any, symbol: str) -> dict[str, Any]:
        return _complete_market(last_price="111")

    fakes = _fakes()
    fakes["technical"] = _CaptureTechnical(
        responses=['{"stance":"bullish","confidence":0.7,"summary":"fresh"}']
    )
    graph = build_graph(
        lambda agent: fakes[agent],
        preflight_tool_context={
            "rest_clients": {"fapi": object()},
            "build_market_snapshot": build_market_snapshot,
        },
    )

    result = await graph.ainvoke(_graph_state())

    assert result["preflight_report"]["status"] == "ready"
    assert result["market_snapshot"]["ticker_24h"]["lastPrice"] == "111"
    assert json.loads(captured["payload"])["ticker_24h"]["lastPrice"] == "111"
    assert len(result["analyst_reports"]) == 3


async def test_graph_historical_clock_survives_preflight_without_changing_live_default() -> None:
    from datetime import datetime, timezone

    from test_graph_offline import _fakes

    frozen = datetime(2023, 4, 7, 12, tzinfo=timezone.utc)
    state = {**_graph_state(), "market_snapshot": _complete_market()}
    fakes = _fakes()
    graph = build_graph(
        lambda agent: fakes[agent],
        preflight_refetch_enabled=False,
        analysis_clock=lambda: frozen,
    )
    result = await graph.ainvoke(state)
    assert result["analysis_at"] == frozen.isoformat()
    assert result["preflight_report"]["analysis_at"] == frozen.isoformat()
    # Ordinary callers cannot set historical analysis time through state.
    live = await run_data_preflight({**state, "analysis_at": frozen.isoformat()})
    assert datetime.fromisoformat(live["analysis_at"]) > frozen


async def test_checkpoint_resume_does_not_repeat_completed_preflight_refetch() -> None:
    from test_graph_offline import _fakes

    calls: list[str] = []

    async def build_market_snapshot(client: Any, symbol: str) -> dict[str, Any]:
        calls.append(symbol)
        return _complete_market()

    fakes = _fakes()
    graph = build_graph(
        lambda agent: fakes[agent],
        checkpointer=MemorySaver(),
        interrupt_after=["data_preflight"],
        preflight_tool_context={
            "rest_clients": {"fapi": object()},
            "build_market_snapshot": build_market_snapshot,
        },
    )
    config = {"configurable": {"thread_id": "preflight-resume"}}

    updates = []
    async for update in graph.astream(_graph_state(), config=config, stream_mode="updates"):
        updates.append(update)

    assert any("__interrupt__" in update for update in updates)
    assert calls == ["BTCUSDT"]

    result = await graph.ainvoke(None, config=config)

    assert result["preflight_report"]["status"] == "ready"
    assert calls == ["BTCUSDT"]
