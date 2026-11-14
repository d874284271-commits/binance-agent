"""Fatal accounting errors and production checkpoint recovery across two workers."""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from binance_agent.agents.budget import BudgetPersistenceError, RunBudget, run_budget_scope
from binance_agent.agents.collaboration import AgentEngine, CollaborationScope, collaboration_scope
from binance_agent.agents.research import _call_tool, build_news_research_graph
from binance_agent.agents.tools import ToolDefinition, ToolRegistry
from binance_agent.app.checkpoints import SQLiteCheckpointSaver
from binance_agent.app.collaboration_store import CollaborationStore
from binance_agent.app.run_replay import replay_runs
from binance_agent.app.run_store import FileRunStore
from binance_agent.infra.settings import ResearchConfig
from binance_agent.intel.evidence import evidence_document
from test_news_research_graph import initial, model_for
from test_research_revision import BODY, RevisionModel


async def test_tool_success_settlement_failure_is_fatal_once() -> None:
    def persist(snapshot: dict[str, Any]) -> None:
        if snapshot["requests_settled"]:
            raise OSError("ledger unavailable")

    budget = RunBudget(max_requests=10, on_change=persist)
    operation = AsyncMock(return_value="paid result")
    original_settle = budget.settle
    budget.settle = AsyncMock(wraps=original_settle)  # type: ignore[method-assign]
    with run_budget_scope(budget), pytest.raises(BudgetPersistenceError):
        await _call_tool("search", "task", operation, fixed_cost_usd=0.1)
    assert operation.await_count == 1
    assert budget.settle.await_count == 1
    assert budget.cost_usd == 0.1
    assert budget.unknown_cost_requests == 0
    with pytest.raises(BudgetPersistenceError):
        await budget.reserve("search", "next", kind="tool", fixed_cost_usd=0.1)


async def test_generic_tool_registry_does_not_fallback_on_budget_persistence() -> None:
    async def handler(data: BaseModel, context: Any) -> BaseModel:
        raise BudgetPersistenceError("durability lost")

    class Empty(BaseModel):
        pass

    fallback = Mock()
    registry = ToolRegistry(
        [
            ToolDefinition(
                name="fixture",
                input_model=Empty,
                output_model=Empty,
                handler=handler,
                fallback=fallback,
            )
        ]
    )
    with pytest.raises(BudgetPersistenceError):
        await registry.invoke("fixture", {})
    fallback.assert_not_called()


async def test_replay_batch_reads_history_once_and_propagates_fatal_budget(
    tmp_path: Path, monkeypatch: Any
) -> None:
    store = FileRunStore(tmp_path / "runs.jsonl")
    ids = [store.create(symbol="BTC", market="api") for _ in range(2)]
    calls = 0
    original = store.list_all

    def list_all() -> Any:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(store, "list_all", list_all)
    monkeypatch.setattr(store, "get", lambda _: pytest.fail("per-ID history lookup"))

    class Graph:
        fatal = False

        async def astream(self, state: Any, **kwargs: Any) -> Any:
            if self.fatal:
                raise BudgetPersistenceError("ledger unavailable")
            yield {"node": {}}

    graph = Graph()
    batch = await replay_runs(store, ids, graph)
    assert len(batch.results) == 2 and calls == 1
    graph.fatal = True
    with pytest.raises(BudgetPersistenceError):
        await replay_runs(store, ids, graph)
    assert calls == 2


async def test_two_task_sqlite_resume_keeps_committed_worker_without_reissuing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    store = CollaborationStore(tmp_path / "control.sqlite")
    saver = SQLiteCheckpointSaver(store.path)
    session = store.create_session("BTCUSDT", "fapi", "s")
    run = store.start_turn(session["session_id"], "ETF和漏洞事件", "t", "fixture")
    model = RevisionModel(responses=["unused"])
    verified = asyncio.Event()
    fetched: list[str] = []
    original_progress = store.save_progress
    original_persist = store.save_evidence
    fail = True

    def save_progress(run_id: str, task_id: str, revision: int, scratch: Any) -> None:
        original_progress(run_id, task_id, revision, scratch)
        if scratch.get("stage") == "finish":
            verified.set()

    monkeypatch.setattr(store, "save_progress", save_progress)

    async def fetch(url: str) -> Any:
        fetched.append(url)
        if url.endswith("/hack"):
            await asyncio.wait_for(verified.wait(), timeout=2)
            # Let the verified sibling finish and commit its child checkpoint.
            await asyncio.sleep(0.05)
        return evidence_document(url=url, text=BODY, level="body")

    def persist(run_id: str, documents: Any) -> None:
        nonlocal fail
        if fail and any(
            doc.url.endswith("/hack") and doc.status == "available" for doc in documents
        ):
            fail = False
            raise OSError("second worker evidence write failed")
        original_persist(run_id, documents)

    graph = build_news_research_graph(
        model_for(model),
        config=ResearchConfig(enabled=True, max_rounds=1, max_workers=2),
        fetch=fetch,
        persist=persist,
        load=store.load_evidence,
    )
    graph.checkpointer = saver
    config = {"configurable": {"thread_id": run["run_id"]}}
    state = initial()
    state["research_question"] = "核实 Bitcoin ETF 和 Bitcoin exchange hack"
    state["research_run_id"] = run["run_id"]
    state["intel_snapshot"]["news_items"].append(
        {
            "title": "Bitcoin exchange hack emergency withdrawal suspension",
            "summary": "Bitcoin exchange security vulnerability incident.",
            "url": "https://security.example.com/hack",
            "source": "security",
            "published_at": None,
        }
    )

    def scope() -> CollaborationScope:
        return CollaborationScope(
            AgentEngine(store, saver),
            session["session_id"],
            run["run_id"],
            0,
            state["research_question"],
            state["research_question"],
            [],
            "fixture",
        )

    try:
        with collaboration_scope(scope()), pytest.raises(OSError, match="second worker"):
            await graph.ainvoke(state, config, durability="sync")
        checkpoint = await graph.aget_state(config)
        assert len(checkpoint.values["tasks"]) == 2
        assert verified.is_set()
        first_calls = Counter(payload["task"]["task_id"] for payload in model.research_payloads)
        assert sum(first_calls.values()) == 1
        # Reopen the production saver so recovery cannot rely on in-memory state.
        saver.close()
        saver = SQLiteCheckpointSaver(store.path)
        graph.checkpointer = saver
        with collaboration_scope(scope()):
            restored = await graph.ainvoke(None, config, durability="sync")
        assert len(restored["news_research"]["tasks"]) == 2
        after_calls = Counter(payload["task"]["task_id"] for payload in model.research_payloads)
        assert all(after_calls[task] == count for task, count in first_calls.items())
        assert restored["news_research"]["facts"]
        assert Counter(fetched) == Counter(
            {"https://news.example.com/etf": 1, "https://security.example.com/hack": 1}
        )
        assert any(
            doc.status == "available" and doc.url.endswith("/etf")
            for doc in store.load_evidence(run["run_id"])
        )
    finally:
        saver.close()
        store.close()


@pytest.mark.parametrize("failure", ["fetch_budget", "load"])
async def test_research_does_not_convert_fatal_errors_to_gaps(failure: str) -> None:
    async def fetch(url: str) -> Any:
        raise BudgetPersistenceError("budget persistence failed")

    def load(run_id: str) -> Any:
        if failure == "load":
            raise OSError("evidence load failed")
        return []

    model = RevisionModel(responses=["unused"])
    graph = build_news_research_graph(
        model_for(model), config=ResearchConfig(enabled=True), fetch=fetch, load=load
    )
    with pytest.raises(OSError if failure == "load" else BudgetPersistenceError):
        await graph.ainvoke({**initial(), "research_run_id": "run"})
    assert model.research_payloads == []
