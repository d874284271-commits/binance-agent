"""A saved article is reusable; an answer to a different question is not."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import Field

from binance_agent.agents.collaboration import AgentEngine, CollaborationScope, collaboration_scope
from binance_agent.agents.research import build_news_research_graph
from binance_agent.app.checkpoints import SQLiteCheckpointSaver
from binance_agent.app.collaboration_store import CollaborationStore
from binance_agent.infra.settings import ResearchConfig
from binance_agent.intel.evidence import evidence_document
from test_news_research_graph import ResearchModel, initial, model_for

BODY = "Bitcoin ETF application is pending, not approved. The management fee is 0.25%."


class RevisionModel(ResearchModel):
    research_payloads: list[dict[str, Any]] = Field(default_factory=list)
    verifier_payloads: list[dict[str, Any]] = Field(default_factory=list)
    cancel_verifier_once: bool = False

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        prompt = str(messages[0].content)
        payload = json.loads(messages[-1].content)
        if "金融事件研究者" in prompt:
            self.research_payloads.append(payload)
            body = next(doc for doc in payload["documents"] if doc["level"] == "body")
            statement = (
                "The management fee is 0.25%."
                if "费率" in payload["research_question"]
                else "Bitcoin ETF application is pending, not approved."
            )
            return AIMessage(
                content=json.dumps(
                    {
                        "facts": [
                            {
                                "statement": statement,
                                "quote": statement,
                                "evidence_id": body["evidence_id"],
                                "confirmation_status": "confirmed_in_source",
                            }
                        ],
                        "gaps": [],
                        "conflicts": [],
                        "stance": "neutral",
                    }
                )
            )
        if "verdicts" in prompt:
            self.verifier_payloads.append(payload)
            if self.cancel_verifier_once:
                self.cancel_verifier_once = False
                raise asyncio.CancelledError()
        return await super().ainvoke(messages, *args, **kwargs)


@pytest.mark.parametrize("change", ["supplement", "same_revision_question", "legacy"])
async def test_new_question_reextracts_cached_body_without_resetting_allowance(
    tmp_path: Path, change: str
) -> None:
    store = CollaborationStore(tmp_path / "control.sqlite")
    saver = SQLiteCheckpointSaver(store.path)
    engine = AgentEngine(store, saver)
    session = store.create_session("BTCUSDT", "fapi", "s")
    run = store.start_turn(session["session_id"], "ETF是否获批", "t", "fixture")
    model = RevisionModel(responses=["unused"])
    fetched: list[str] = []

    async def fetch(url: str) -> Any:
        fetched.append(url)
        return evidence_document(url=url, text=BODY, level="body")

    graph = build_news_research_graph(
        model_for(model),
        config=ResearchConfig(enabled=True),
        fetch=fetch,
        persist=store.save_evidence,
        load=store.load_evidence,
    )
    state = {**initial(), "research_run_id": run["run_id"]}

    def scope(revision: int) -> CollaborationScope:
        return CollaborationScope(
            engine,
            session["session_id"],
            run["run_id"],
            revision,
            "ETF是否获批",
            "ETF是否获批",
            store.updates(run["run_id"], revision),
            "fixture",
        )

    try:
        with collaboration_scope(scope(0)):
            first = await graph.ainvoke(state)
        assert len(model.research_payloads) == 1
        task_id = first["news_research"]["tasks"][0]["task_id"]
        previous = store.progress(run["run_id"], task_id)
        old_key = previous["scratch"].get("analysis_key")
        if change == "legacy":
            scratch = previous["scratch"]
            scratch.pop("analysis_key", None)
            for record in scratch["coverage"].values():
                record.pop("analysis_key", None)
            store.save_progress(run["run_id"], task_id, 0, scratch)
        revision = 0 if change == "same_revision_question" else 1
        if revision:
            store.inform(session["session_id"], run["run_id"], "提取管理费率", "m")
        with collaboration_scope(scope(revision)):
            second = await graph.ainvoke({**state, "research_question": "补充:提取管理费率"})
        assert len(model.research_payloads) == 2
        assert len(fetched) == 1
        research = second["news_research"]
        assert any("0.25%" in fact["statement"] for fact in research["facts"])
        assert all(fact.get("citations") for fact in research["facts"])
        assert research["reading_complete"]
        assert not any("未读材料" in gap for gap in research["gaps"])
        assert store.progress(run["run_id"], task_id)["rounds"] == 2
        new_scratch = store.progress(run["run_id"], task_id)["scratch"]
        assert new_scratch["analysis_key"] != old_key
        assert all(
            record["analysis_key"] == new_scratch["analysis_key"]
            for record in new_scratch["coverage"].values()
        )
        assert all(
            "support" not in fact for fact in model.research_payloads[-1]["provisional_facts"]
        )
        # A third question cannot reset the per-task investigation allowance.
        store.inform(session["session_id"], run["run_id"], "再核实适用范围", "m2")
        with collaboration_scope(scope(revision + 1)):
            third = await graph.ainvoke({**state, "research_question": "费率适用范围"})
        assert len(model.research_payloads) == 2
        assert third["news_research"]["critical_unresolved"]
        assert third["news_research"]["facts"] == []
        assert len(model.verifier_payloads) == 2
        assert store.progress(run["run_id"], task_id)["rounds"] == 2
    finally:
        saver.close()
        store.close()


@pytest.mark.parametrize("window", ["before_verification", "after_verification"])
async def test_same_analysis_resume_keeps_committed_research_and_round(
    tmp_path: Path, monkeypatch: Any, window: str
) -> None:
    store = CollaborationStore(tmp_path / "control.sqlite")
    saver = SQLiteCheckpointSaver(store.path)
    session = store.create_session("BTCUSDT", "fapi", "s")
    run = store.start_turn(session["session_id"], "ETF是否获批", "t", "fixture")
    model = RevisionModel(
        responses=["unused"], cancel_verifier_once=window == "before_verification"
    )

    async def fetch(url: str) -> Any:
        return evidence_document(url=url, text=BODY, level="body")

    graph = build_news_research_graph(
        model_for(model),
        config=ResearchConfig(enabled=True, max_rounds=1),
        fetch=fetch,
        persist=store.save_evidence,
        load=store.load_evidence,
    )
    graph.checkpointer = saver
    config = {"configurable": {"thread_id": run["run_id"]}}
    original = store.save_progress
    crash = window == "after_verification"

    def save_progress(run_id: str, task_id: str, revision: int, scratch: Any) -> None:
        nonlocal crash
        if crash and scratch.get("stage") == "finish":
            crash = False
            raise OSError("verifier committed before parent checkpoint")
        original(run_id, task_id, revision, scratch)

    monkeypatch.setattr(store, "save_progress", save_progress)

    def scope() -> CollaborationScope:
        return CollaborationScope(
            AgentEngine(store, saver),
            session["session_id"],
            run["run_id"],
            0,
            "ETF是否获批",
            "ETF是否获批",
            [],
            "fixture",
        )

    try:
        with (
            collaboration_scope(scope()),
            pytest.raises(asyncio.CancelledError if window == "before_verification" else OSError),
        ):
            await graph.ainvoke(
                {**initial(), "research_run_id": run["run_id"]}, config, durability="sync"
            )
        assert len(model.research_payloads) == 1
        with collaboration_scope(scope()):
            restored = await graph.ainvoke(None, config, durability="sync")
        assert len(model.research_payloads) == 1
        assert restored["news_research"]["facts"]
        task_id = restored["news_research"]["tasks"][0]["task_id"]
        assert store.progress(run["run_id"], task_id)["rounds"] == 1
    finally:
        saver.close()
        store.close()


async def test_old_fingerprint_rejects_resume_but_history_remains_readable(tmp_path: Path) -> None:
    from test_collaboration import Model, app_fixture

    async with app_fixture(tmp_path, Model()) as (_, manager, client):
        session = manager.store.create_session("BTCUSDT", "fapi", "s")
        run = manager.store.start_turn(session["session_id"], "旧版本问题", "t", "old-version")
        manager.store.status(run["run_id"], "interrupted")
        response = await client.post(f"/runs/{run['run_id']}/resume", json={"request_id": "r"})
        assert response.status_code == 409
        assert not manager.tasks and not manager.store.budget(run["run_id"])
        assert (await client.get(f"/runs/{run['run_id']}")).status_code == 200


def test_implementation_version_changes_checkpoint_fingerprint(monkeypatch: Any) -> None:
    import binance_agent.app.server as server
    from binance_agent.infra.settings import Settings

    current = server._research_config_fingerprint(Settings())
    monkeypatch.setattr(server, "ANALYSIS_VERSION", "previous-question-analysis")
    assert server._research_config_fingerprint(Settings()) != current
