"""Exercise bounded history polling through the default collaboration store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from binance_agent.app.checkpoints import SQLiteCheckpointSaver
from binance_agent.app.collaboration_store import CollaborationStore
from binance_agent.app.run_store import FileRunStore, _summary_record
from binance_agent.app.server import _initialize_collaboration, create_app
from binance_agent.infra.settings import load_settings


def add_runs(store: CollaborationStore, start: int, count: int) -> None:
    for number in range(start, start + count):
        session = store.create_session("BTCUSDT", "fapi", f"session-{number}")
        run = store.start_turn(session["session_id"], "question", f"turn-{number}", "fixture")
        store.status(run["run_id"], "failed", f"error-{number}")


def record_decoder(decoded: list[str]) -> Any:
    original_loads = json.loads

    def loads(*args: Any, **kwargs: Any) -> Any:
        value = original_loads(*args, **kwargs)
        if isinstance(value, dict) and "run_id" in value and "node_outputs" in value:
            decoded.append(value["run_id"])
        return value

    return loads


async def test_default_runs_summary_polling_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(Path(__file__).parents[1] / "config" / "config.toml")
    assert settings.collaboration.enabled
    settings.runtime.checkpoint_backend = "sqlite"
    settings.runtime.checkpoint_path = str(tmp_path / "control.sqlite")
    app = create_app(settings)
    app.state.run_store = FileRunStore(tmp_path / "runs.jsonl")
    app.state.graph_checkpointer = SQLiteCheckpointSaver(tmp_path / "control.sqlite")
    manager = _initialize_collaboration(app, settings)
    store = app.state.run_store
    assert isinstance(store, CollaborationStore)
    statements: list[str] = []
    counts = []
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            for start, count in ((0, 30), (30, 30)):
                add_runs(store, start, count)
                expected = [_summary_record(r) for r in store.list_recent(limit=5)]
                for _ in range(2):
                    with monkeypatch.context() as patch:

                        def fail(*args: Any, **kwargs: Any) -> Any:
                            raise AssertionError("summary polling hydrated full run history")

                        decoded: list[str] = []
                        patch.setattr(json, "loads", record_decoder(decoded))
                        patch.setattr(store, "get", fail)
                        statements.clear()
                        store._conn.set_trace_callback(statements.append)
                        response = await client.get("/runs", params={"limit": 5})
                        store._conn.set_trace_callback(None)
                    assert response.status_code == 200
                    assert response.json() == expected
                    assert len(decoded) == 5
                    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
                    counts.append(len(selects))
                    assert not any(
                        "research_messages" in s or "research_tasks" in s for s in selects
                    )
        assert counts == [1, 1, 1, 1]
    finally:
        store._conn.set_trace_callback(None)
        await manager.shutdown()
        app.state.graph_checkpointer.close()
        store.close()


def test_collaboration_summary_merge_matches_detail_history(tmp_path: Path) -> None:
    legacy = FileRunStore(tmp_path / "runs.jsonl")
    store = CollaborationStore(tmp_path / "control.sqlite", legacy=legacy)
    try:
        legacy.create(symbol="ETH", market="api")
        add_runs(store, 0, 4)
        legacy.create(symbol="SOL", market="api")
        add_runs(store, 4, 4)
        for limit in (0, 1, 5, 50):
            expected = [_summary_record(r) for r in store.list_recent(limit=limit)]
            assert store.list_recent_summaries(limit=limit) == expected
        session = store.create_session("BTC", "fapi", "live")
        run = store.start_turn(session["session_id"], "question", "live-turn", "fixture")
        assert store.list_recent_summaries(limit=1)[0]["status"] == "running"
        assert store.status(run["run_id"], "cancelled", "updated")
        assert store.list_recent_summaries(limit=1)[0]["status"] == "cancelled"
    finally:
        store.close()
