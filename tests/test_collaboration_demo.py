"""The shipped offline demo serves the collaborative desktop contract."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from binance_agent.research_demo import create_research_demo_app


async def test_research_demo_config_and_collaborative_turn(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    app = create_research_demo_app()

    def reject_keychain(ref):
        raise AssertionError("offline demo must not access real credentials")

    monkeypatch.setattr("binance_agent.app.server.resolve_secret", reject_keychain)
    monkeypatch.setattr("binance_agent.llm.registry.resolve_secret", reject_keychain)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client,
    ):
        config = await client.get("/config")
        assert config.status_code == 200
        assert config.json()["collaboration"]["enabled"]
        session = (
            await client.post("/research/sessions", json={"symbol": "ETHUSDT", "request_id": "s"})
        ).json()
        turn = (
            await client.post(
                "/research/sessions/" + session["session_id"] + "/turns",
                json={"question": "区分申请和批准", "request_id": "t"},
            )
        ).json()
        manager = app.state.research_manager
        await asyncio.wait_for(asyncio.shield(manager.tasks[turn["run_id"]]), 10)
        detail = (await client.get("/research/sessions/" + session["session_id"])).json()
        assert detail["runs"][0]["status"] == "completed"
        assert detail["agents"] and detail["tasks"]
        record = manager.store.get(turn["run_id"])
        assert record["news_research"]["tasks"]
        assert record["research_budget"]["requests_reserved"] > 0
        assert (tmp_path / ".demo/.runs/checkpoints.sqlite").is_file()
        assert not (tmp_path / ".demo/.runs/runs.jsonl").read_text()
