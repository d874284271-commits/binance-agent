from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

import binance_agent.app.server as server_module
from binance_agent.app.server import create_app
from binance_agent.infra.profiles import apply_profile
from binance_agent.infra.settings import Settings

_FULL_NODE_SEQUENCE = [
    "data_preflight",
    "funding",
    "news_sentiment",
    "technical",
    "bear_debater",
    "bull_debater",
    "debate_controller",
    "bear_debater",
    "bull_debater",
    "debate_controller",
    "debate_referee",
    "research_manager",
    "risk_manager",
    "strategy_synth",
]


def _events(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    name = "message"
    for line in raw.splitlines():
        if line.startswith("event: "):
            name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            events.append((name, json.loads(line.removeprefix("data: "))))
    return events


def _fail_network(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("offline demo attempted to construct a network dependency")


async def test_demo_three_scenarios_are_offline_and_deterministic(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server_module, "BinanceRestClient", _fail_network)
    monkeypatch.setattr(server_module, "BinanceWsManager", _fail_network)
    monkeypatch.setattr(server_module, "_build_aggregator", _fail_network)
    monkeypatch.setattr(server_module, "LlmRegistry", _fail_network)

    settings = apply_profile(Settings(), "demo")
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1",
            trust_env=False,
            timeout=30,
        ) as client:
            results: dict[str, dict[str, Any]] = {}
            for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
                response = await client.get("/analyze", params={"symbol": symbol, "market": "fapi"})
                assert response.status_code == 200
                events = _events(response.text)
                final = next(payload for event, payload in events if event == "final")
                run = (await client.get(f"/runs/{final['run_id']}")).json()
                results[symbol] = {"events": events, "final": final, "run": run}

            assert (await client.get("/outcomes/stats")).status_code == 404

    for symbol in ("BTCUSDT", "ETHUSDT"):
        assert [
            payload["node"] for event, payload in results[symbol]["events"] if event == "node"
        ] == _FULL_NODE_SEQUENCE
        assert results[symbol]["final"]["advice"]["decision_type"] == "trade"
        assert results[symbol]["run"]["preflight_report"]["status"] == "ready"
        assert all(
            event["token_usage"]["total_tokens"] == 88
            for event in results[symbol]["run"]["runtime_events"]
        )

    assert results["BTCUSDT"]["run"]["preflight_report"]["refetch_attempts"] == []
    assert results["ETHUSDT"]["run"]["preflight_report"]["refetch_attempts"] == [
        {
            "kind": "market",
            "tool": "market_snapshot",
            "ok": True,
            "reason": "refetch succeeded",
        }
    ]
    assert len(results["ETHUSDT"]["run"]["market_snapshot"]["klines"]["1h"]) == 36

    sol = results["SOLUSDT"]
    assert [payload["node"] for event, payload in sol["events"] if event == "node"] == [
        "data_preflight"
    ]
    assert sol["final"]["advice"]["decision_type"] == "no_trade"
    assert sol["run"]["preflight_report"]["status"] == "blocked"
    assert sol["run"]["runtime_events"] == []

    assert (tmp_path / ".demo" / ".runs" / "runs.jsonl").exists()
    assert not (tmp_path / ".runs" / "runs.jsonl").exists()


async def test_demo_repeats_preflight_scenarios_in_fresh_request_contexts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server_module, "BinanceRestClient", _fail_network)
    monkeypatch.setattr(server_module, "BinanceWsManager", _fail_network)
    monkeypatch.setattr(server_module, "_build_aggregator", _fail_network)
    monkeypatch.setattr(server_module, "LlmRegistry", _fail_network)

    settings = apply_profile(Settings(), "demo")
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1",
            trust_env=False,
            timeout=30,
        ) as client:
            for symbol in ("ETHUSDT", "SOLUSDT"):
                for _ in range(2):
                    response = await client.get(
                        "/analyze", params={"symbol": symbol, "market": "fapi"}
                    )
                    assert response.status_code == 200
                    events = _events(response.text)
                    market = next(payload for event, payload in events if event == "market")
                    final = next(payload for event, payload in events if event == "final")
                    run = (await client.get(f"/runs/{final['run_id']}")).json()

                    assert market["errors"] == ["demo scripted initial snapshot is incomplete"]
                    assert [payload["node"] for event, payload in events if event == "node"] == (
                        _FULL_NODE_SEQUENCE if symbol == "ETHUSDT" else ["data_preflight"]
                    )
                    assert run["preflight_report"]["requested_refetches"] == ["market"]
                    assert len(run["preflight_report"]["refetch_attempts"]) == 1
                    assert run["preflight_report"]["refetch_attempts"][0]["ok"] is (
                        symbol == "ETHUSDT"
                    )
                    assert final["advice"]["decision_type"] == (
                        "trade" if symbol == "ETHUSDT" else "no_trade"
                    )
