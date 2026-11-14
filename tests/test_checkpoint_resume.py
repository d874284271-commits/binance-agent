"""LangGraph checkpoint/resume 与 /runs/{run_id}/resume 测试。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from binance_agent.agents.graph import build_graph
from binance_agent.app.run_replay import check_replay_invariants
from binance_agent.app.run_store import FileRunStore
from binance_agent.app.server import create_app
from binance_agent.infra.settings import LlmConfig, LlmEndpoint, Settings
from binance_agent.llm.registry import LlmRegistry


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


async def test_graph_resume_after_strategy_interrupt_does_not_rerun_successful_nodes() -> None:
    fakes = {
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
    graph = build_graph(lambda agent: fakes[agent], checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "resume-strategy"}}

    try:
        await graph.ainvoke(_initial_state(), config=config)
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("first strategy_synth call should interrupt")

    result = await graph.ainvoke(None, config=config)

    assert result["final_advice"].decision_type == "trade"
    assert fakes["technical"].calls == 1
    assert fakes["funding"].calls == 1
    assert fakes["news_sentiment"].calls == 1
    assert fakes["strategy_synth"].calls == 2


async def test_graph_resume_after_analyst_interrupt_skips_completed_analysts() -> None:
    fakes = {
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
        ),
    }
    graph = build_graph(
        lambda agent: fakes[agent],
        checkpointer=MemorySaver(),
        interrupt_after=["technical", "funding", "news_sentiment"],
    )
    config = {"configurable": {"thread_id": "resume-analysts"}}

    updates = []
    async for update in graph.astream(_initial_state(), config=config, stream_mode="updates"):
        updates.append(update)

    assert any("__interrupt__" in update for update in updates)
    result = await graph.ainvoke(None, config=config)

    assert result["final_advice"].decision_type == "trade"
    assert fakes["technical"].calls == 1
    assert fakes["funding"].calls == 1
    assert fakes["news_sentiment"].calls == 1


class _Dumpable:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        return self._payload


class _CountingAggregator:
    calls = 0

    async def gather(self, symbol: str) -> _Dumpable:
        self.calls += 1
        return _Dumpable(
            {
                "symbol": symbol,
                "source_status": {"rss": "ok"},
                "news_items": [{"title": "headline"}],
                "macro_events": [],
                "sentiment_features": {},
            }
        )


class _ResumeGraph:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def astream(
        self,
        state: dict[str, Any] | None,
        config: dict[str, Any] | None = None,
        *,
        stream_mode: str,
    ) -> Any:
        self.calls.append({"state": state, "config": config})
        if state is not None:
            yield {"technical": {"analyst_reports": [{"analyst": "technical"}]}}
            raise RuntimeError("strategy_synth failed")
        yield {
            "strategy_synth": {
                "final_advice": {
                    "symbol": "BTCUSDT",
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
        }


async def test_resume_endpoint_reuses_snapshots_and_records_resume_event(
    tmp_path: Path, monkeypatch: Any
) -> None:
    app = create_app(Settings())
    store = FileRunStore(tmp_path / "runs.jsonl")
    graph = _ResumeGraph()
    aggregator = _CountingAggregator()
    market_calls = {"count": 0}
    app.state.run_store = store
    app.state.rest_client = object()
    app.state.aggregator = aggregator
    app.state.graph = graph

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

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        first = await client.get("/analyze", params={"symbol": "BTCUSDT", "market": "fapi"})
        run_id = next(e["data"]["run_id"] for e in _parse_sse(first.text) if e["event"] == "run")
        assert next(e for e in _parse_sse(first.text) if e["event"] == "failed")

        resumed = await client.get(f"/runs/{run_id}/resume")
        events = _parse_sse(resumed.text)

    assert market_calls["count"] == 1
    assert aggregator.calls == 1
    assert graph.calls[0]["state"] is not None
    assert graph.calls[1]["state"] is None
    assert graph.calls[0]["config"]["configurable"]["thread_id"] == run_id
    assert graph.calls[1]["config"]["configurable"]["thread_id"] == run_id
    assert any(event["event"] == "resume" for event in events)
    assert next(event for event in events if event["event"] == "final")["data"]["run_id"] == run_id

    record = store.get(run_id)
    assert record is not None
    assert any(event["event_type"] == "resume" for event in record["events"])
    assert record["final_advice"]["decision_type"] == "no_trade"
    assert check_replay_invariants(record, record["final_advice"]) == []


async def test_config_switch_preserves_graph_checkpointer(monkeypatch: Any) -> None:
    settings = Settings(
        llm=LlmConfig(
            active_endpoint="a",
            endpoints=[
                LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A", model="ma"),
                LlmEndpoint(name="b", base_url="https://b/v1", api_key_ref="B", model="mb"),
            ],
        )
    )
    app = create_app(settings)
    app.state.llm_registry = LlmRegistry(
        settings.llm, model_factory=lambda endpoint, model: FakeListChatModel(responses=["{}"])
    )
    checkpointer = MemorySaver()
    app.state.graph_checkpointer = checkpointer
    captured: dict[str, Any] = {}

    def fake_build_graph(model_for: Any, *, checkpointer: Any = None, **kwargs: Any) -> object:
        captured["checkpointer"] = checkpointer
        return object()

    monkeypatch.setattr("binance_agent.app.server.build_graph", fake_build_graph)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        response = await client.post("/config/llm-endpoint/switch", json={"endpoint": "b"})

    assert response.status_code == 200
    assert captured["checkpointer"] is checkpointer


async def test_config_switch_persists_active_endpoint(monkeypatch: Any) -> None:
    settings = Settings(
        llm=LlmConfig(
            active_endpoint="a",
            endpoints=[
                LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A", model="ma"),
                LlmEndpoint(name="b", base_url="https://b/v1", api_key_ref="B", model="mb"),
            ],
        )
    )
    app = create_app(settings)
    app.state.llm_registry = LlmRegistry(
        settings.llm, model_factory=lambda endpoint, model: FakeListChatModel(responses=["{}"])
    )
    app.state.graph_checkpointer = MemorySaver()
    saved: dict[str, Any] = {}

    monkeypatch.setattr("binance_agent.app.server.build_graph", lambda *args, **kwargs: object())
    monkeypatch.setattr("binance_agent.app.server.save_overrides", lambda data: saved.update(data))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        response = await client.post("/config/llm-endpoint/switch", json={"endpoint": "b"})

    assert response.status_code == 200
    assert saved["llm"]["active"] == "b"


async def test_fetch_llm_models_uses_endpoint_api_key_ref(monkeypatch: Any) -> None:
    settings = Settings(
        llm=LlmConfig(
            active_endpoint="blaze",
            endpoints=[
                LlmEndpoint(
                    name="blaze",
                    base_url="https://blaze.example/v1",
                    api_key_ref="BLAZE_API_KEY",
                    model="mb",
                )
            ],
        )
    )
    app = create_app(settings)
    captured: dict[str, Any] = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"data": [{"id": "mb"}, {"id": "mb-large"}]}

    class _Http:
        async def get(self, url: str, headers: dict[str, str]) -> _Response:
            captured["url"] = url
            captured["headers"] = headers
            return _Response()

    monkeypatch.setattr(
        "binance_agent.app.server.resolve_secret",
        lambda ref: "real-blaze-key" if ref == "BLAZE_API_KEY" else None,
    )
    app.state.intel_http = _Http()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        response = await client.post(
            "/config/llm/models",
            json={"endpoint": "blaze", "base_url": "https://blaze.example/v1"},
        )

    assert response.status_code == 200
    assert response.json()["models"] == ["mb", "mb-large"]
    assert captured["url"] == "https://blaze.example/v1/models"
    assert captured["headers"] == {"Authorization": "Bearer real-blaze-key"}


async def test_config_params_strategy_planner_mode_rebuilds_graph(monkeypatch: Any) -> None:
    settings = Settings(
        llm=LlmConfig(
            active_endpoint="a",
            endpoints=[LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A", model="ma")],
        )
    )
    app = create_app(settings)
    app.state.llm_registry = LlmRegistry(
        settings.llm, model_factory=lambda endpoint, model: FakeListChatModel(responses=["{}"])
    )
    checkpointer = MemorySaver()
    app.state.graph_checkpointer = checkpointer
    app.state.intel_http = object()
    app.state.rest_client = object()
    captured: dict[str, Any] = {}

    def fake_build_graph(model_for: Any, *, checkpointer: Any = None, **kwargs: Any) -> object:
        captured["checkpointer"] = checkpointer
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("binance_agent.app.server.build_graph", fake_build_graph)
    monkeypatch.setattr("binance_agent.app.server._build_aggregator", lambda *args: object())
    monkeypatch.setattr("binance_agent.app.server.save_overrides", lambda data: None)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        response = await client.post(
            "/config/params",
            json={"strategy_planner_mode": "llm_shadow", "strategy_planner_max_steps": 4},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["strategy_planner_mode"] == "llm_shadow"
    assert body["strategy_planner_enabled"] is True
    assert body["strategy_planner_llm_actions"] is True
    assert captured["checkpointer"] is checkpointer
    assert captured["kwargs"]["strategy_planner_mode"] == "llm_shadow"
    assert captured["kwargs"]["enable_strategy_planner"] is True
    assert captured["kwargs"]["enable_strategy_planner_llm_actions"] is True
    assert captured["kwargs"]["enforce_strategy_planner_decision"] is False
    assert captured["kwargs"]["planner_max_steps"] == 4


async def test_config_params_rejects_invalid_bounds(monkeypatch: Any) -> None:
    settings = Settings(
        llm=LlmConfig(
            active_endpoint="a",
            endpoints=[LlmEndpoint(name="a", base_url="https://a/v1", api_key_ref="A", model="ma")],
        )
    )
    app = create_app(settings)
    app.state.llm_registry = LlmRegistry(
        settings.llm, model_factory=lambda endpoint, model: FakeListChatModel(responses=["{}"])
    )
    app.state.intel_http = object()
    app.state.rest_client = object()
    app.state.graph_checkpointer = MemorySaver()

    monkeypatch.setattr("binance_agent.app.server._build_aggregator", lambda *args: object())
    monkeypatch.setattr("binance_agent.app.server.save_overrides", lambda data: None)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        for payload in (
            {"time_window_hours": 0},
            {"time_window_hours": -24},
            {"max_debate_rounds": -1},
            {"max_debate_rounds": 51},
            {"per_source_timeout": 0},
            {"max_news_items": -5},
            {"strategy_planner_max_steps": 0},
        ):
            response = await client.post("/config/params", json=payload)
            assert response.status_code == 422, payload


async def test_config_delete_endpoint_persists_authoritative_endpoint_list(
    monkeypatch: Any,
) -> None:
    settings = Settings(
        llm=LlmConfig(
            active_endpoint="blaze",
            endpoints=[
                LlmEndpoint(name="default", base_url="https://a/v1", api_key_ref="A", model="ma"),
                LlmEndpoint(
                    name="blaze", base_url="https://blaze/api", api_key_ref="B", model="mb"
                ),
                LlmEndpoint(
                    name="custom", base_url="https://custom/v1", api_key_ref="C", model="mc"
                ),
            ],
            agent_models={"technical": ("blaze", "mb")},
        )
    )
    app = create_app(settings)
    app.state.llm_registry = LlmRegistry(
        settings.llm, model_factory=lambda endpoint, model: FakeListChatModel(responses=["{}"])
    )
    app.state.graph_checkpointer = MemorySaver()
    saved: dict[str, Any] = {}

    monkeypatch.setattr("binance_agent.app.server.build_graph", lambda *args, **kwargs: object())
    monkeypatch.setattr("binance_agent.app.server.save_overrides", lambda data: saved.update(data))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1", trust_env=False
    ) as client:
        response = await client.delete("/config/llm/endpoint/blaze")

    assert response.status_code == 200
    returned_names = {endpoint["name"] for endpoint in response.json()["endpoints"]}
    saved_names = {endpoint["name"] for endpoint in saved["llm"]["endpoints"]}
    assert "blaze" not in returned_names
    assert "blaze" not in saved_names
    assert saved["llm"]["active"] != "blaze"
    assert "technical" not in saved["llm"]["agent_models"]


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
