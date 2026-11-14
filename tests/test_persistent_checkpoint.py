"""Persistent checkpoint storage for cross-process resume."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, TypedDict

import httpx
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from binance_agent.agents.graph import build_graph
from binance_agent.app.checkpoints import SQLiteCheckpointSaver
from binance_agent.app.run_store import FileRunStore
from binance_agent.app.server import create_app
from binance_agent.infra.settings import Settings


class _CountingChatModel(FakeListChatModel):
    calls: int = 0
    cancel_first: bool = False

    async def ainvoke(  # type: ignore[override]
        self, messages: Any, *args: Any, **kwargs: Any
    ) -> AIMessage:
        self.calls += 1
        if self.cancel_first and self.calls == 1:
            raise asyncio.CancelledError("interrupted")
        return await super().ainvoke(messages, *args, **kwargs)


class _FailIfCalledChatModel(FakeListChatModel):
    async def ainvoke(  # type: ignore[override]
        self, messages: Any, *args: Any, **kwargs: Any
    ) -> AIMessage:
        raise AssertionError("completed node should not be rerun from persistent checkpoint")


class _ServerResumeState(TypedDict, total=False):
    symbol: str
    market: str
    marker: str
    final_advice: dict[str, Any]


class _Dumpable:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        return self._payload


class _CountingAggregator:
    def __init__(self) -> None:
        self.calls = 0

    async def gather(self, symbol: str) -> _Dumpable:
        self.calls += 1
        return _Dumpable(
            {
                "symbol": symbol,
                "source_status": {"rss": "ok"},
                "news_items": [],
                "macro_events": [],
                "sentiment_features": {},
            }
        )


async def test_sqlite_checkpointer_resumes_graph_after_rebuild_without_rerunning_nodes(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    first_models = {
        "technical": _CountingChatModel(
            responses=['{"stance":"bullish","confidence":0.8,"summary":"technical"}']
        ),
        "funding": _CountingChatModel(
            responses=['{"stance":"bullish","confidence":0.8,"summary":"funding"}']
        ),
        "news_sentiment": _CountingChatModel(
            responses=['{"stance":"bullish","confidence":0.8,"summary":"news"}']
        ),
        "bull_debater": _CountingChatModel(responses=["bull"]),
        "bear_debater": _CountingChatModel(responses=["bear"]),
        "research_manager": _CountingChatModel(responses=["research"]),
        "strategy_synth": _CountingChatModel(
            responses=[
                '{"direction":"long","decision_type":"trade","entry_zone":"100",'
                '"stop_loss":"95","targets":["115"],"confidence":0.8,'
                '"rationale":"resumed"}'
            ],
            cancel_first=True,
        ),
    }
    thread_id = "persistent-resume"
    config = {"configurable": {"thread_id": thread_id}}
    first_saver = SQLiteCheckpointSaver(db_path)
    first_graph = build_graph(lambda agent: first_models[agent], checkpointer=first_saver)

    try:
        await first_graph.ainvoke(_initial_state(), config=config)
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("first strategy_synth call should interrupt")

    assert first_saver.has_thread(thread_id)
    first_saver.close()

    second_models = {
        "technical": _FailIfCalledChatModel(responses=["{}"]),
        "funding": _FailIfCalledChatModel(responses=["{}"]),
        "news_sentiment": _FailIfCalledChatModel(responses=["{}"]),
        "bull_debater": _FailIfCalledChatModel(responses=["{}"]),
        "bear_debater": _FailIfCalledChatModel(responses=["{}"]),
        "research_manager": _FailIfCalledChatModel(responses=["{}"]),
        "strategy_synth": _CountingChatModel(
            responses=[
                '{"direction":"long","decision_type":"trade","entry_zone":"100",'
                '"stop_loss":"95","targets":["115"],"confidence":0.8,'
                '"rationale":"resumed after rebuild"}'
            ],
        ),
    }
    second_saver = SQLiteCheckpointSaver(db_path)
    second_graph = build_graph(lambda agent: second_models[agent], checkpointer=second_saver)

    result = await second_graph.ainvoke(None, config=config)

    assert result["final_advice"].decision_type == "trade"
    strategy_model = second_models["strategy_synth"]
    assert isinstance(strategy_model, _CountingChatModel)
    assert strategy_model.calls == 1
    second_saver.close()


async def test_resume_endpoint_uses_persisted_checkpoint_after_app_rebuild(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    store = FileRunStore(tmp_path / "runs.jsonl")
    market_calls = {"count": 0}

    async def fake_market_snapshot(client: object, symbol: str) -> _Dumpable:
        market_calls["count"] += 1
        return _Dumpable(
            {
                "symbol": symbol,
                "market": "fapi",
                "ticker_24h": {"lastPrice": "100"},
                "klines": {},
                "errors": [],
            }
        )

    monkeypatch.setattr("binance_agent.app.server.build_market_snapshot", fake_market_snapshot)

    first_saver = SQLiteCheckpointSaver(checkpoint_path)
    first_app = create_app(Settings())
    first_aggregator = _CountingAggregator()
    first_app.state.run_store = store
    first_app.state.rest_client = object()
    first_app.state.aggregator = first_aggregator
    first_app.state.graph_checkpointer = first_saver
    first_app.state.graph = _server_resume_graph(first_saver, fail_after_marker=True)

    transport = httpx.ASGITransport(app=first_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        first = await client.get("/analyze", params={"symbol": "BTCUSDT", "market": "fapi"})
        first_events = _parse_sse(first.text)
        run_id = next(event["data"]["run_id"] for event in first_events if event["event"] == "run")
        assert next(event for event in first_events if event["event"] == "failed")

    first_saver.close()
    assert market_calls["count"] == 1
    assert first_aggregator.calls == 1
    failed_record = store.get(run_id)
    assert failed_record is not None
    assert failed_record["resumable"] is True

    second_saver = SQLiteCheckpointSaver(checkpoint_path)
    second_app = create_app(Settings())
    second_aggregator = _CountingAggregator()
    second_app.state.run_store = FileRunStore(tmp_path / "runs.jsonl")
    second_app.state.rest_client = object()
    second_app.state.aggregator = second_aggregator
    second_app.state.graph_checkpointer = second_saver
    second_app.state.graph = _server_resume_graph(second_saver, fail_after_marker=False)

    transport = httpx.ASGITransport(app=second_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        resumed = await client.get(f"/runs/{run_id}/resume")
        events = _parse_sse(resumed.text)

    assert market_calls["count"] == 1
    assert second_aggregator.calls == 0
    final = next(event for event in events if event["event"] == "final")
    assert final["data"]["run_id"] == run_id
    assert final["data"]["advice"]["decision_type"] == "no_trade"
    completed_record = second_app.state.run_store.get(run_id)
    assert completed_record is not None
    assert completed_record["status"] == "completed"
    assert completed_record["resumable"] is False
    second_saver.close()


async def test_resume_endpoint_returns_conflict_when_checkpoint_missing(tmp_path: Path) -> None:
    app = create_app(Settings())
    store = FileRunStore(tmp_path / "runs.jsonl")
    run_id = store.create(symbol="BTCUSDT", market="fapi")
    store.append_event(
        run_id,
        "checkpoint",
        {"thread_id": run_id, "backend": "sqlite", "status": "resumable"},
    )
    store.fail(run_id, "interrupted")
    app.state.run_store = store
    app.state.graph_checkpointer = SQLiteCheckpointSaver(tmp_path / "empty.sqlite")
    app.state.graph = object()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        response = await client.get(f"/runs/{run_id}/resume")

    assert response.status_code == 409
    assert "checkpoint not found" in response.json()["detail"]


def test_sqlite_checkpointer_can_delete_thread_for_retention_cleanup(tmp_path: Path) -> None:
    saver = SQLiteCheckpointSaver(tmp_path / "checkpoints.sqlite")
    graph = build_graph(
        lambda agent: _CountingChatModel(
            responses=[
                '{"stance":"bullish","confidence":0.8,"summary":"ok"}',
                "debate",
                '{"direction":"long","decision_type":"trade","entry_zone":"100",'
                '"stop_loss":"95","targets":["115"],"confidence":0.8,'
                '"rationale":"ok"}',
            ]
        ),
        checkpointer=saver,
    )
    thread_id = "cleanup-thread"

    asyncio.run(graph.ainvoke(_initial_state(), config={"configurable": {"thread_id": thread_id}}))

    assert saver.has_thread(thread_id)
    saver.delete_thread(thread_id)
    assert not saver.has_thread(thread_id)


def _initial_state() -> dict[str, Any]:
    rows = [
        {"open": i, "high": i + 2, "low": i - 2, "close": i + 0.5, "volume": 10}
        for i in range(1, 50)
    ]
    return {
        "symbol": "BTCUSDT",
        "market": "fapi",
        "market_snapshot": {
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
        },
        "intel_snapshot": {
            "sentiment_features": {"long_short_ratio": 1.2},
            "news_items": [{"title": "headline"}],
            "macro_events": [],
            "source_status": {"rss": "ok", "reddit": "ok"},
        },
        "analyst_reports": [],
        "debate_turns": [],
        "runtime_events": [],
        "debate_round": 0,
        "max_debate_rounds": 1,
        "research_summary": None,
        "risk_assessment": None,
        "risk_gate_result": None,
        "final_advice": None,
    }


def _server_resume_graph(checkpointer: SQLiteCheckpointSaver, *, fail_after_marker: bool) -> Any:
    builder: StateGraph = StateGraph(_ServerResumeState)

    def marker(state: _ServerResumeState) -> dict[str, Any]:
        if not fail_after_marker:
            raise AssertionError("completed marker node should not rerun during resume")
        return {"marker": "done"}

    def strategy_synth(state: _ServerResumeState) -> dict[str, Any]:
        if fail_after_marker:
            raise RuntimeError("strategy_synth failed")
        return {
            "final_advice": {
                "symbol": state.get("symbol", "BTCUSDT"),
                "direction": "neutral",
                "decision_type": "no_trade",
                "entry_zone": None,
                "stop_loss": None,
                "targets": [],
                "confidence": 0.2,
                "rationale": "resumed",
                "risk_warnings": [],
                "no_trade_reason": "恢复后观望",
                "reentry_conditions": ["等待信号恢复"],
            }
        }

    builder.add_node("marker", marker)
    builder.add_node("strategy_synth", strategy_synth)
    builder.add_edge(START, "marker")
    builder.add_edge("marker", "strategy_synth")
    builder.add_edge("strategy_synth", END)
    return builder.compile(checkpointer=checkpointer)


def _parse_sse(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, str] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                events.append(
                    {
                        "event": current.get("event"),
                        "data": json.loads(current.get("data", "{}")),
                    }
                )
                current = {}
            continue
        if line.startswith("event:"):
            current["event"] = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            current["data"] = line.removeprefix("data:").strip()
    if current:
        events.append(
            {"event": current.get("event"), "data": json.loads(current.get("data", "{}"))}
        )
    return events
