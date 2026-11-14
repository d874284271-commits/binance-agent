from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from binance_agent.agents.graph import build_graph
from binance_agent.app.run_store import FileRunStore
from binance_agent.app.server import _research_config_fingerprint, create_app
from binance_agent.infra.settings import ResearchConfig, Settings
from binance_agent.intel.evidence import evidence_document
from test_graph_offline import _complete_market_snapshot, _fakes
from test_news_research_graph import ResearchModel
from test_server_runs import _Dumpable, _parse_sse, _TickerClient


def test_research_config_fingerprint_is_stable_and_changes_with_limits() -> None:
    settings = Settings(research=ResearchConfig(enabled=True, max_requests=30, deadline_s=30))
    same = Settings(research=ResearchConfig(enabled=True, max_requests=30, deadline_s=30))
    changed = Settings(research=ResearchConfig(enabled=True, max_requests=31, deadline_s=30))
    assert _research_config_fingerprint(settings) == _research_config_fingerprint(same)
    assert _research_config_fingerprint(settings) != _research_config_fingerprint(changed)


async def test_research_api_persists_body_separately_and_exposes_citation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = Settings(research=ResearchConfig(enabled=True, max_requests=30, deadline_s=30))
    app = create_app(settings)
    store = FileRunStore(tmp_path / "runs.jsonl")
    app.state.run_store = store
    app.state.rest_client = _TickerClient()

    class Aggregator:
        async def gather(self, symbol: str) -> Any:
            return _Dumpable(
                {
                    "symbol": symbol,
                    "source_status": {"fixture": "ok"},
                    "news_items": [
                        {"title": "Bitcoin ETF approved!", "url": "https://news.example.com/etf"}
                    ],
                }
            )

    async def market(*_: Any) -> Any:
        return _Dumpable(_complete_market_snapshot(funding_rate=0.0001))

    body = "Bitcoin ETF application is pending, not approved."

    async def fetch(url: str) -> Any:
        return evidence_document(url=url, text=body, level="body")

    monkeypatch.setattr("binance_agent.app.server.build_market_snapshot", market)
    app.state.aggregator = Aggregator()
    models = _fakes()
    model = ResearchModel(responses=["unused"])
    models.update(
        dict.fromkeys(["research_supervisor", "news_researcher", "evidence_verifier"], model)
    )
    app.state.graph = build_graph(
        lambda name: models[name],
        research_config=settings.research,
        research_fetch=fetch,
        research_persist=store.save_evidence,
        research_load=store.load_evidence,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get(
            "/analyze", params={"symbol": "BTCUSDT", "research_question": "Is the ETF approved?"}
        )
        assert response.status_code == 200
        events = _parse_sse(response.text)
        assert any(e["event"] == "research_progress" for e in events), [e["event"] for e in events]
        final = next(e["data"] for e in events if e["event"] == "final")
        run_id = final["run_id"]
        assert final["advice"]["evidence_confidence"] == 1.0
        record = store.get(run_id)
        assert record and record["news_research"]["question"] == "Is the ETF approved?"
        assert record["research_budget"]["requests_reserved"] > 0
        assert not any(
            "evidence_documents" in (e.get("update") or {}) for e in record["node_outputs"]
        )
        evidence_id = final["news_research"]["facts"][0]["evidence_id"]
        document = await client.get(f"/runs/{run_id}/evidence/{evidence_id}")
        assert document.json()["text"] == body
        assert (await client.get(f"/runs/{run_id}/evidence/not-real")).status_code == 404
        saved = json.loads((tmp_path / "evidence" / f"{run_id}.json").read_text())
        assert any(d["text"] == body for d in saved["documents"])


async def test_research_requires_explicit_finite_execution_limits() -> None:
    app = create_app(Settings(research=ResearchConfig(enabled=True)))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get("/analyze", params={"symbol": "BTCUSDT"})
        assert response.status_code == 409
