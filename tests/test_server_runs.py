"""RunStore 与 /analyze SSE 集成测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from binance_agent.app.run_store import FileRunStore
from binance_agent.app.server import create_app
from binance_agent.infra.settings import Settings


class _Dumpable:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        return self._payload


class _Aggregator:
    async def gather(self, symbol: str) -> _Dumpable:
        return _Dumpable(
            {
                "symbol": symbol,
                "source_status": {"rss": "ok", "reddit": "ok"},
                "news_items": [],
                "macro_events": [],
                "sentiment_features": {},
            }
        )


class _TickerClient:
    async def ticker_24h(self, symbol: str) -> dict[str, str]:
        return {"lastPrice": "101"}


class _Graph:
    def __init__(self) -> None:
        self.states: list[dict[str, Any]] = []

    async def astream(self, state: dict[str, Any], *, stream_mode: str) -> Any:
        assert stream_mode == "updates"
        assert state["runtime_events"] == []
        self.states.append(state)
        yield {
            "technical": {
                "analyst_reports": [{"analyst": "technical", "stance": "neutral"}],
                "runtime_events": [
                    {
                        "agent": "technical",
                        "ok": False,
                        "latency_ms": 10.0,
                        "attempts": 1,
                        "errors": [{"category": "timeout", "message": "llm timeout"}],
                    }
                ],
            }
        }
        yield {
            "strategy_synth": {
                "final_advice": {
                    "symbol": state["symbol"],
                    "direction": "neutral",
                    "decision_type": "no_trade",
                    "entry_zone": None,
                    "stop_loss": None,
                    "targets": [],
                    "confidence": 0.2,
                    "rationale": "fallback",
                    "risk_warnings": [],
                    "no_trade_reason": "策略合成失败",
                    "reentry_conditions": ["等待模型服务恢复"],
                }
            }
        }


class _ReplayGraph:
    def __init__(self) -> None:
        self.configs: list[dict[str, Any]] = []

    async def astream(
        self, state: dict[str, Any], config: dict[str, Any], *, stream_mode: str
    ) -> Any:
        assert stream_mode == "updates"
        self.configs.append(config)
        yield {
            "strategy_planner": {
                "planner_decision": {
                    "decision_type": "no_trade",
                    "mode": "llm_shadow",
                    "enforced": False,
                    "affects_final_advice": False,
                    "action_status": "accepted",
                    "reason": "shadow only",
                }
            }
        }
        yield {
            "strategy_synth": {
                "final_advice": {
                    "symbol": state["symbol"],
                    "direction": "neutral",
                    "decision_type": "no_trade",
                    "entry_zone": None,
                    "stop_loss": None,
                    "targets": [],
                    "confidence": 0.2,
                    "rationale": "replay",
                    "risk_warnings": [],
                    "no_trade_reason": "数据不足",
                    "reentry_conditions": ["等待数据恢复"],
                }
            }
        }


async def test_analyze_writes_run_record_and_exposes_run_id(
    tmp_path: Path, monkeypatch: Any
) -> None:
    app = create_app(Settings())
    store = FileRunStore(tmp_path / "runs.jsonl")
    app.state.run_store = store
    app.state.rest_client = _TickerClient()
    app.state.aggregator = _Aggregator()
    app.state.graph = _Graph()

    async def fake_market_snapshot(client: object, symbol: str) -> _Dumpable:
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
        response = await client.get("/analyze", params={"symbol": "btcusdt", "market": "fapi"})
        assert response.status_code == 200
        events = _parse_sse(response.text)
        run_event = next(event for event in events if event["event"] == "run")
        run_id = run_event["data"]["run_id"]

        final_event = next(event for event in events if event["event"] == "final")
        assert final_event["data"]["run_id"] == run_id
        assert final_event["data"]["advice"]["decision_type"] == "no_trade"
        assert final_event["data"]["run_health"]["llm_status"] == "warn"
        assert final_event["data"]["run_health"]["fallback_agents"] == ["technical"]
        assert final_event["data"]["evidence_summary"]["decision_type"] == "no_trade"
        assert final_event["data"]["evidence_factors"] == []

        run_response = await client.get(f"/runs/{run_id}")
        assert run_response.status_code == 200
        record = run_response.json()
        assert record["symbol"] == "BTCUSDT"
        assert record["market_snapshot"]["ticker_24h"]["lastPrice"] == "100"
        assert record["intel_snapshot"]["source_status"]["rss"] == "ok"
        assert [event["tool"] for event in record["tool_events"][:2]] == [
            "market_snapshot",
            "intel_snapshot",
        ]
        assert record["tool_events"][0]["ok"] is True
        assert record["tool_events"][1]["ok"] is True
        assert record["node_outputs"][0]["node"] == "technical"
        assert record["runtime_events"][0]["agent"] == "technical"
        assert record["runtime_events"][0]["errors"][0]["category"] == "timeout"
        assert record["final_advice"]["decision_type"] == "no_trade"

        recent_response = await client.get("/runs", params={"limit": 5})
        assert recent_response.status_code == 200
        assert recent_response.json()[0]["run_id"] == run_id
        recent = recent_response.json()[0]
        assert "market_snapshot" not in recent
        assert "intel_snapshot" not in recent
        assert "node_outputs" not in recent
        assert app.state.graph.states[0]["snapshot_freshness_enabled"] is True
        assert app.state.graph.states[0]["latest_price"] == 101.0

        full_recent_response = await client.get("/runs", params={"limit": 5, "detail": "full"})
        assert full_recent_response.status_code == 200
        assert full_recent_response.json()[0]["market_snapshot"]["ticker_24h"]["lastPrice"] == "100"


async def test_analyze_redacts_secret_in_sse_and_store(tmp_path: Path, monkeypatch: Any) -> None:
    # #5 回归:已注册密钥若夹带进节点错误消息,SSE 推流与 .runs JSONL 落盘均须脱敏。
    from binance_agent.infra.logging import register_secret

    secret = "BLAZEKEY-feedfacecafe-secret-2024"
    register_secret(secret)

    class _LeakyGraph:
        async def astream(self, state: dict[str, Any], *, stream_mode: str) -> Any:
            yield {
                "technical": {
                    "analyst_reports": [{"analyst": "technical", "stance": "neutral"}],
                    "runtime_events": [
                        {
                            "agent": "technical",
                            "ok": False,
                            "latency_ms": 1.0,
                            "attempts": 1,
                            "errors": [
                                {
                                    "category": "llm_error",
                                    "message": f"401 auth failed key={secret}",
                                }
                            ],
                        }
                    ],
                }
            }
            yield {
                "strategy_synth": {
                    "final_advice": {
                        "symbol": state["symbol"],
                        "direction": "neutral",
                        "decision_type": "no_trade",
                        "entry_zone": None,
                        "stop_loss": None,
                        "targets": [],
                        "confidence": 0.2,
                        "rationale": "fallback",
                        "risk_warnings": [],
                        "no_trade_reason": "策略合成失败",
                        "reentry_conditions": [],
                    }
                }
            }

    app = create_app(Settings())
    store = FileRunStore(tmp_path / "runs.jsonl")
    app.state.run_store = store
    app.state.rest_client = object()
    app.state.aggregator = _Aggregator()
    app.state.graph = _LeakyGraph()

    async def fake_market_snapshot(client: object, symbol: str) -> _Dumpable:
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
        response = await client.get("/analyze", params={"symbol": "btcusdt", "market": "fapi"})
        assert response.status_code == 200
        assert secret not in response.text  # SSE 推流脱敏
        assert "***REDACTED***" in response.text

    raw = (tmp_path / "runs.jsonl").read_text(encoding="utf-8")
    assert secret not in raw  # 落盘脱敏


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
