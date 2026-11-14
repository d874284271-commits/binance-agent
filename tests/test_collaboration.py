"""Durable collaboration behavior with controlled offline models and clocks."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from binance_agent.agents.budget import RunBudget
from binance_agent.agents.collaboration import (
    AgentEngine,
    CollaborationScope,
    collaboration_scope,
    current_scope,
)
from binance_agent.agents.runtime import AgentRuntime
from binance_agent.agents.state import AgentState, StrategyAdvice
from binance_agent.agents.tools import build_default_tool_registry
from binance_agent.app.checkpoints import SQLiteCheckpointSaver
from binance_agent.app.collaboration_store import CollaborationStore, SessionConflict
from binance_agent.app.run_store import FileRunStore
from binance_agent.app.server import _initialize_collaboration, create_app
from binance_agent.infra.settings import Settings
from test_graph_offline import _complete_market_snapshot
from test_server_runs import _Dumpable, _TickerClient


class Model:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    def bind(self, **_: Any) -> Model:
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        payload = json.loads(messages[-1].content)
        self.payloads.append(payload)
        self.entered.set()
        await self.release.wait()
        return AIMessage(
            content=json.dumps({"answer": payload.get("handoff", {}).get("user_updates", [])})
        )


def scope_for(
    store: CollaborationStore, engine: AgentEngine, run: dict[str, Any], revision: int = 0
) -> CollaborationScope:
    session = store.session(run["session_id"])
    return CollaborationScope(
        engine,
        run["session_id"],
        run["run_id"],
        revision,
        "本轮问题",
        session["root_question"],
        store.updates(run["run_id"], revision),
        "fixture",
    )


async def test_same_agent_survives_two_turns_and_idle_eviction(tmp_path: Path) -> None:
    store = CollaborationStore(tmp_path / "control.sqlite")
    saver = SQLiteCheckpointSaver(store.path)
    engine = AgentEngine(store, saver, idle_s=0.001)
    session = store.create_session("BTCUSDT", "fapi", "session")
    model = Model()
    runtime = AgentRuntime(model, max_retries=0)  # type: ignore[arg-type]
    first = store.start_turn(session["session_id"], "原始问题与必须保留的限制", "turn1", "fixture")
    with collaboration_scope(scope_for(store, engine, first)):
        result = await runtime.invoke_structured(
            "technical", "technical only", {}, fallback_data={}
        )
    assert result.ok
    identity = store.agent(session["session_id"], "technical")["agent_id"]
    assert store.finalize(
        first["run_id"],
        0,
        StrategyAdvice(symbol="BTCUSDT", rationale="first", decision_type="no_trade", confidence=0),
    )
    engine.evict(force=True)
    assert not engine._resident
    assert not store.agent(session["session_id"], "technical")["resident"]
    second = store.start_turn(session["session_id"], "继续研究", "turn2", "fixture")
    with collaboration_scope(scope_for(store, engine, second)):
        await runtime.invoke_structured("technical", "technical only", {}, fallback_data={})
    assert store.agent(session["session_id"], "technical")["agent_id"] == identity
    assert model.payloads[-1]["handoff"]["root_question"] == "原始问题与必须保留的限制"
    assert model.payloads[-1]["derived_context"][0]["source_ids"] == ["participant_prior_work"]
    assert store.get(first["run_id"])["final_advice"]["rationale"] == "first"
    saver.close()
    store.close()


@pytest.mark.parametrize("after_delivery", [False, True])
async def test_outbox_recovery_does_not_repeat_successful_model(
    tmp_path: Path, monkeypatch: Any, after_delivery: bool
) -> None:
    store = CollaborationStore(tmp_path / "control.sqlite")
    saver = SQLiteCheckpointSaver(store.path)
    engine = AgentEngine(store, saver)
    session = store.create_session("BTCUSDT", "fapi", "s")
    run = store.start_turn(session["session_id"], "question", "t", "fixture")
    model = Model()
    runtime = AgentRuntime(model, max_retries=0)  # type: ignore[arg-type]
    original = store.deliver

    def fail_publish(**kwargs: Any) -> Any:
        if kwargs["kind"] == "task_result":
            if after_delivery:
                original(**kwargs)
            raise OSError("crash before delivery confirmation")
        return original(**kwargs)

    monkeypatch.setattr(store, "deliver", fail_publish)
    with collaboration_scope(scope_for(store, engine, run)), pytest.raises(OSError):
        await runtime.invoke_structured("technical", "technical only", {}, fallback_data={})
    monkeypatch.setattr(store, "deliver", original)
    recovered = AgentEngine(store, saver)
    with collaboration_scope(scope_for(store, recovered, run)):
        result = await runtime.invoke_structured(
            "technical", "technical only", {}, fallback_data={}
        )
    assert result.ok and len(model.payloads) == 1
    participant = store.agent(session["session_id"], "technical")
    assert participant["status"] == "idle" and not participant["current_task"]
    supervisor = store.agent(session["session_id"], "research_supervisor")["agent_id"]
    results = [m for m in store.inbox(supervisor) if m["kind"] == "task_result"]
    assert len(results) == 1
    saver.close()
    store.close()


async def test_mailbox_wakes_waiter_and_rejects_old_publication(tmp_path: Path) -> None:
    store = CollaborationStore(tmp_path / "control.sqlite")
    saver = SQLiteCheckpointSaver(store.path)
    engine = AgentEngine(store, saver)
    session = store.create_session("BTCUSDT", "fapi", "s")
    run = store.start_turn(session["session_id"], "question", "t", "fixture")
    agent = store.agent(session["session_id"], "research_supervisor")["agent_id"]
    waiter = asyncio.create_task(engine.wait_for_message(agent))
    await asyncio.sleep(0)
    message = store.inform(session["session_id"], run["run_id"], "更正条件", "m")
    await asyncio.to_thread(engine.notify, agent)
    assert (await asyncio.wait_for(waiter, 1))[0]["message_id"] == message["message_id"]
    assert store.inform(session["session_id"], run["run_id"], "更正条件", "m") == message
    with pytest.raises(SessionConflict):
        store.inform(session["session_id"], run["run_id"], "不同条件", "m")
    assert not store.finalize(run["run_id"], 0, {})
    await engine.incorporate(session["session_id"], run["run_id"], 1)
    assert store.session(session["session_id"])["messages"][0]["status"] == "incorporated"
    saver.close()
    store.close()


@asynccontextmanager
async def app_fixture(tmp_path: Path, model: Model) -> Any:
    settings = Settings.model_validate(
        {
            "research": {"enabled": True, "max_requests": 30, "deadline_s": 60},
            "runtime": {
                "checkpoint_backend": "sqlite",
                "checkpoint_path": str(tmp_path / "control.sqlite"),
            },
            "collaboration": {"enabled": True},
        }
    )
    app = create_app(settings)
    app.state.run_store = FileRunStore(tmp_path / "runs.jsonl")
    app.state.graph_checkpointer = SQLiteCheckpointSaver(tmp_path / "control.sqlite")
    app.state.tool_registry = build_default_tool_registry()
    app.state.rest_client = _TickerClient()
    app.state.spot_client = _TickerClient()
    app.state.prepares = 0
    app.state.observed_scopes = []

    class Aggregator:
        async def gather(self, symbol: str, **_: Any) -> Any:
            return _Dumpable(
                {"symbol": symbol, "news_items": [], "source_status": {"fixture": "ok"}}
            )

    async def snapshot(*_: Any, **__: Any) -> Any:
        app.state.prepares += 1
        return _Dumpable(_complete_market_snapshot(funding_rate=0.0001))

    app.state.aggregator = Aggregator()
    app.state.market_snapshot_builder = snapshot

    async def work(state: AgentState) -> dict[str, Any]:
        app.state.observed_scopes.append((state.get("input_revision"), current_scope().revision))
        response = await AgentRuntime(model, max_retries=0).invoke_structured(
            "technical", "technical only", {}, fallback_data={}
        )  # type: ignore[arg-type]
        return {
            "final_advice": StrategyAdvice(
                symbol=state["symbol"],
                direction="neutral",
                decision_type="no_trade",
                rationale=json.dumps(response.data, ensure_ascii=False),
                confidence=0,
            )
        }

    builder = StateGraph(AgentState)
    builder.add_node("technical", work)
    builder.add_edge(START, "technical")
    builder.add_edge("technical", END)
    app.state.graph = builder.compile(checkpointer=app.state.graph_checkpointer)
    manager = _initialize_collaboration(app, settings)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            yield app, manager, client
    finally:
        await manager.shutdown()
        manager.store.close()
        app.state.graph_checkpointer.close()


async def wait_finished(manager: Any, run_id: str, *, timeout: float = 5) -> dict[str, Any]:
    task = manager.tasks.get(run_id)
    if task:
        await asyncio.wait_for(asyncio.shield(task), timeout)
    return manager.store.run(run_id)


async def test_api_supplement_disconnect_and_continue(tmp_path: Path) -> None:
    model = Model()
    model.release.clear()
    async with app_fixture(tmp_path, model) as (app, manager, client):
        session = (
            await client.post("/research/sessions", json={"symbol": "BTCUSDT", "request_id": "s"})
        ).json()
        sid = session["session_id"]
        turn = (
            await client.post(
                f"/research/sessions/{sid}/turns", json={"question": "调查", "request_id": "t"}
            )
        ).json()
        rid = turn["run_id"]
        await asyncio.wait_for(model.entered.wait(), 5)
        stream = manager.subscribe(sid)
        first = await anext(stream)
        await stream.aclose()
        assert rid in manager.tasks
        assert (await client.post("/config/reset", json={})).status_code == 409
        supplement = await client.post(
            f"/research/sessions/{sid}/messages",
            json={"run_id": rid, "content": "必须检查否定条件", "request_id": "m"},
        )
        assert supplement.status_code == 202
        model.release.set()
        run = await wait_finished(manager, rid)
        assert run["status"] == "completed" and run["input_revision"] == 1
        assert "必须检查否定条件" in manager.store.get(rid)["final_advice"]["rationale"], (
            app.state.observed_scopes,
            model.payloads,
        )
        notices = [
            e["payload"]
            for e in manager.store.events(sid)
            if e["event"] == "research_progress" and e["payload"].get("status") == "input_updated"
        ]
        assert (
            notices
            and notices[-1]["input_revision"] == 1
            and notices[-1]["superseded_revision"] == 0
        )
        assert manager.store.budget(rid)["requests_reserved"] == 2
        assert manager.store.session(sid)["messages"][0]["status"] == "incorporated"
        replay = manager.store.events(sid, int(first["id"]))
        assert any(event["event"] == "final" for event in replay)
        assert all(event["seq"] > int(first["id"]) for event in replay)
        identities = {a["role"]: a["agent_id"] for a in manager.store.session(sid)["agents"]}
        repeat = await client.post(
            f"/research/sessions/{sid}/turns", json={"question": "调查", "request_id": "t"}
        )
        assert repeat.json()["run_id"] == rid
        later = (
            await client.post(
                f"/research/sessions/{sid}/turns", json={"question": "再调查", "request_id": "t2"}
            )
        ).json()
        await wait_finished(manager, later["run_id"])
        assert app.state.prepares == 2
        assert {
            a["role"]: a["agent_id"] for a in manager.store.session(sid)["agents"]
        } == identities
        assert manager.store.budget(later["run_id"])["requests_reserved"] == 1
        assert (
            await client.post(
                f"/research/sessions/{sid}/messages",
                json={"run_id": rid, "content": "迟到", "request_id": "late"},
            )
        ).status_code == 409


async def test_stop_and_explicit_resume_after_shutdown(tmp_path: Path) -> None:
    model = Model()
    model.release.clear()
    async with app_fixture(tmp_path, model) as (_, manager, client):
        session = manager.store.create_session("BTCUSDT", "fapi", "s")
        turn = await manager.turn(session["session_id"], "调查", "t")
        rid = turn["run_id"]
        await asyncio.wait_for(model.entered.wait(), 5)
        before = manager.store.budget(rid)
        await manager.shutdown()
        assert manager.store.run(rid)["status"] == "interrupted"
        assert manager.store.budget(rid)["requests_reserved"] == before["requests_reserved"]
    model.release.set()
    async with app_fixture(tmp_path, model) as (_, manager, client):
        assert not manager.tasks
        resumed = await client.post(f"/runs/{rid}/resume", json={"request_id": "resume"})
        assert resumed.status_code == 202
        await wait_finished(manager, rid)
        assert manager.store.run(rid)["status"] == "completed"
        assert manager.store.budget(rid)["deadline_at"] == before["deadline_at"]
        assert manager.store.budget(rid)["requests_reserved"] >= before["requests_reserved"]
        fresh = await manager.turn(session["session_id"], "停止测试", "t2")
        stopped = await client.post(f"/runs/{fresh['run_id']}/stop", json={"request_id": "stop"})
        assert stopped.status_code == 202
        await wait_finished(manager, fresh["run_id"])
        assert manager.store.run(fresh["run_id"])["status"] == "cancelled"


async def test_session_mutations_reject_cross_site(tmp_path: Path) -> None:
    async with app_fixture(tmp_path, Model()) as (_, _, client):
        response = await client.post(
            "/research/sessions",
            headers={"origin": "https://example.com"},
            json={"symbol": "BTCUSDT", "request_id": "s"},
        )
        assert response.status_code == 403


async def test_full_research_graph_uses_durable_participants_and_refreshes_evidence(
    tmp_path: Path,
) -> None:
    from binance_agent.agents.graph import build_graph
    from binance_agent.intel.evidence import evidence_document
    from test_graph_offline import _fakes
    from test_news_research_graph import ResearchModel

    async with app_fixture(tmp_path, Model()) as (app, manager, _):
        models = _fakes()
        researcher = ResearchModel(responses=["unused"])
        models.update(
            dict.fromkeys(
                ["research_supervisor", "news_researcher", "evidence_verifier"], researcher
            )
        )
        fetches: list[str] = []

        async def fetch(url: str) -> Any:
            fetches.append(url)
            return evidence_document(
                url=url, text="Bitcoin ETF application is pending, not approved.", level="body"
            )

        class News:
            async def gather(self, symbol: str, **_: Any) -> Any:
                return _Dumpable(
                    {
                        "symbol": symbol,
                        "source_status": {"fixture": "ok"},
                        "news_items": [
                            {
                                "title": "Bitcoin ETF approved!",
                                "url": "https://news.example.com/etf",
                            }
                        ],
                    }
                )

        app.state.aggregator = News()
        prepare = manager.prepare

        async def frozen(*args: Any) -> dict[str, Any]:
            state = await prepare(*args)
            state["snapshot_freshness_enabled"] = False
            return state

        manager.prepare = frozen
        app.state.graph = build_graph(
            lambda name: models[name],
            checkpointer=app.state.graph_checkpointer,
            research_config=manager.settings.research,
            research_fetch=fetch,
            research_persist=manager.store.save_evidence,
            research_load=manager.store.load_evidence,
        )
        session = manager.store.create_session("BTCUSDT", "fapi", "s")
        first = await manager.turn(session["session_id"], "核实ETF状态,保留否定和适用条件", "t")
        # This exercises the full graph's durable SQLite writes, not a 5s SLA.
        # Allow its configured research deadline plus bounded shutdown headroom.
        timeout = manager.settings.research.deadline_s + 5
        run = await wait_finished(manager, first["run_id"], timeout=timeout)
        assert run["status"] == "completed", run["record"].get("error")
        record = manager.store.get(first["run_id"])
        assert record["news_research"]["facts"]
        agents = manager.store.session(session["session_id"])["agents"]
        assert {a["role"] for a in agents} >= {
            "technical",
            "funding",
            "news_researcher",
            "research_supervisor",
            "evidence_verifier",
            "research_manager",
            "strategy_synth",
        }
        assert all(a["status"] == "idle" for a in agents)
        second = await manager.turn(session["session_id"], "继续核实", "t2")
        assert (await wait_finished(manager, second["run_id"], timeout=timeout))[
            "status"
        ] == "completed"
        assert len(fetches) == 2  # A historical body must not suppress the next turn's fetch.
        assert {a["agent_id"] for a in manager.store.session(session["session_id"])["agents"]} == {
            a["agent_id"] for a in agents
        }
        assert any("historical_context" in payload for payload in researcher.seen_payloads)


def test_store_terminal_fence_versions_and_message_identity(tmp_path: Path) -> None:
    legacy = FileRunStore(tmp_path / "legacy.jsonl")
    old = legacy.create(symbol="ETHUSDT", market="api")
    store = CollaborationStore(tmp_path / "control.sqlite", legacy=legacy)
    store.append_event(old, "market", {"symbol": "ETHUSDT"})
    store.complete(old, {"decision_type": "no_trade"})
    assert store.get(old)["status"] == "completed"
    assert store.get(old)["market_snapshot"]["symbol"] == "ETHUSDT"
    assert "session_id" not in store.get(old)
    session = store.create_session("BTCUSDT", "fapi", "s")
    run = store.start_turn(session["session_id"], "question", "t", "fixture")
    rid = run["run_id"]
    message = store.inform(session["session_id"], rid, "new condition", "m")
    with pytest.raises(SessionConflict):
        store.apply_revision(rid, 1)
    assert not store.finalize(rid, 1, {})
    store.acknowledge(message["receiver"], message["seq"])
    store.apply_revision(rid, 1)
    for invalid in (0, 2):
        with pytest.raises(SessionConflict):
            store.apply_revision(rid, invalid)
    other = store.agent(session["session_id"], "technical")["agent_id"]
    with pytest.raises(SessionConflict):
        store.deliver(
            session_id=session["session_id"],
            run_id=rid,
            sender="user",
            receiver=other,
            kind="inform",
            task_id=None,
            revision=1,
            payload={"content": "new condition"},
            key="user:" + session["session_id"] + ":m",
        )
    assert store.finalize(rid, 1, {"decision_type": "no_trade"})
    for status in ("stopping", "interrupted", "cancelled", "running"):
        assert not store.status(rid, status)
    store.recover()
    assert store.run(rid)["status"] == "completed"
    assert store.session(session["session_id"])["active_run_id"] is None
    store.close()


@pytest.mark.parametrize("after_commit", [False, True])
async def test_ingest_checkpoint_fault_recovers_without_losing_message(
    tmp_path: Path, monkeypatch: Any, after_commit: bool
) -> None:
    store = CollaborationStore(tmp_path / "control.sqlite")
    saver = SQLiteCheckpointSaver(store.path)
    engine = AgentEngine(store, saver)
    session = store.create_session("BTCUSDT", "fapi", "s")
    run = store.start_turn(session["session_id"], "question", "t", "fixture")
    message = store.inform(session["session_id"], run["run_id"], "keep me", "m")
    original = engine.graph.aupdate_state

    async def crash(*args: Any, **kwargs: Any) -> Any:
        if after_commit:
            await original(*args, **kwargs)
        raise OSError("crash at ingest commit")

    monkeypatch.setattr(engine.graph, "aupdate_state", crash)
    with pytest.raises(OSError):
        await engine.incorporate(session["session_id"], run["run_id"], 1)
    assert store.session(session["session_id"])["messages"][0]["status"] == "queued"
    recovered = AgentEngine(store, saver)
    await recovered.incorporate(session["session_id"], run["run_id"], 1)
    assert store.session(session["session_id"])["messages"][0]["status"] == "incorporated"
    supervisor = message["receiver"]
    snapshot = await recovered.graph.aget_state({"configurable": {"thread_id": supervisor}})
    assert snapshot.values["inbox_cursor"] == message["seq"]
    assert store.finalize(run["run_id"], 1, {})
    saver.close()
    store.close()


@pytest.mark.parametrize("limit", ["requests", "deadline"])
async def test_exhausted_budget_does_not_publish_obsolete_levels(
    tmp_path: Path, monkeypatch: Any, limit: str
) -> None:
    model = Model()
    model.release.clear()
    clock = [time.time()]
    if limit == "deadline":
        monkeypatch.setattr(RunBudget, "_now", lambda self: clock[0])
    async with app_fixture(tmp_path, model) as (_, manager, _):
        manager.settings.research.max_requests = 1 if limit == "requests" else 30
        session = manager.store.create_session("BTCUSDT", "fapi", "s")
        run = await manager.turn(session["session_id"], "question", "t")
        await asyncio.wait_for(model.entered.wait(), 5)
        original_deadline = manager.store.budget(run["run_id"])["deadline_at"]
        manager.store.inform(session["session_id"], run["run_id"], "new condition", "m")
        if limit == "deadline":
            clock[0] = original_deadline + 1
        model.release.set()
        result = await wait_finished(manager, run["run_id"])
        advice = result["record"]["final_advice"]
        assert result["status"] == "completed" and result["input_revision"] == 1
        assert advice["decision_type"] == "no_trade" and not advice.get("entry_price")
        assert (
            manager.store.session(session["session_id"])["messages"][0]["status"] == "not_processed"
        )
        assert "未完成" in advice["rationale"]
        assert manager.store.budget(run["run_id"])["deadline_at"] == original_deadline
        assert len(model.payloads) == 1
        assert manager.store.budget(run["run_id"])["requests_reserved"] == 1


def test_research_round_and_task_allowance_survives_input_revisions(tmp_path: Path) -> None:
    store = CollaborationStore(tmp_path / "control.sqlite")
    store.save_progress("r", "a", 0, {"facts": []})
    store.save_progress("r", "b", 0, {"facts": []})
    assert store.reserve_round("r", "a", 2, 1)
    store.save_progress("r", "a", 1, {"facts": ["carried"]})
    assert store.reserve_round("r", "a", 2, 1)
    assert not store.reserve_round("r", "a", 2, 1)
    assert not store.reserve_round("r", "b", 2, 1)
    assert store.progress("r", "a")["rounds"] == 2
    store.close()


@pytest.mark.parametrize("tool", [False, True])
async def test_input_update_while_waiting_for_endpoint_never_sends_stale_request(
    tmp_path: Path, tool: bool
) -> None:
    from binance_agent.agents.budget import EndpointPool, RunBudget, run_budget_scope
    from binance_agent.agents.collaboration import InputSuperseded
    from binance_agent.agents.research import _call_tool

    store = CollaborationStore(tmp_path / "control.sqlite")
    saver = SQLiteCheckpointSaver(store.path)
    engine = AgentEngine(store, saver)
    session = store.create_session("BTCUSDT", "fapi", "s")
    run = store.start_turn(session["session_id"], "question", "t", "fixture")
    pool = EndpointPool()
    semaphore = asyncio.Semaphore(0)
    pool.semaphores["public_html" if tool else "primary"] = semaphore
    reserved = asyncio.Event()
    budget = RunBudget(max_requests=10, endpoint_pool=pool, on_change=lambda _: reserved.set())
    model = Model()

    async def operation() -> None:
        pytest.fail("obsolete tool must not be called")

    async def work() -> Any:
        with run_budget_scope(budget), collaboration_scope(scope_for(store, engine, run)):
            if tool:
                return await _call_tool("public_html", "t", operation, fixed_cost_usd=0)
            return await AgentRuntime(model, max_retries=3).invoke_structured(
                "technical", "technical only", {}, fallback_data={}
            )

    task = asyncio.create_task(work())
    await asyncio.wait_for(reserved.wait(), 5)
    store.inform(session["session_id"], run["run_id"], "new requirements", "m")
    semaphore.release()
    if tool:
        with pytest.raises(InputSuperseded):
            await task
    else:
        result = await task
        assert result.error_category == "input_superseded" and result.attempts == 1
    assert not model.payloads and budget.attempts[0]["status"] == "not_sent"
    assert budget.unknown_cost_requests == 0 and budget.requests_reserved == 1
    saver.close()
    store.close()


async def test_http_cancellation_keeps_background_concurrency_reservation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    model = Model()
    model.release.clear()
    async with app_fixture(tmp_path, model) as (app, manager, _):
        session = manager.store.create_session("BTCUSDT", "fapi", "s")
        original_notify = manager.notify

        def cancel_response(session_id: str) -> None:
            original_notify(session_id)
            if asyncio.current_task().get_name() == "cancel-turn-response":
                asyncio.current_task().cancel()

        monkeypatch.setattr(manager, "notify", cancel_response)
        request = asyncio.create_task(
            manager.turn(session["session_id"], "question", "t"), name="cancel-turn-response"
        )
        with pytest.raises(asyncio.CancelledError):
            await request
        await asyncio.wait_for(model.entered.wait(), 5)
        assert len(manager.tasks) == 1 and app.state.analysis_limiter._active == 1
        retry = await manager.turn(session["session_id"], "question", "t")
        assert retry["run_id"] in manager.tasks
        model.release.set()
        await wait_finished(manager, retry["run_id"])
        assert app.state.analysis_limiter._active == 0


async def test_new_turn_uses_current_endpoint_capacity(tmp_path: Path) -> None:
    async with app_fixture(tmp_path, Model()) as (_, manager, _):
        from binance_agent.infra.settings import LlmEndpoint

        endpoint = LlmEndpoint(
            name="primary",
            base_url="https://example.invalid/v1",
            api_key_ref="TEST_UNUSED",
            model="fixture",
            max_concurrency=3,
        )
        manager.settings.llm.endpoints = [endpoint]
        session = manager.store.create_session("BTCUSDT", "fapi", "s")
        run = await manager.turn(session["session_id"], "question", "t")
        await wait_finished(manager, run["run_id"])
        assert manager.pool.concurrency_for(endpoint.name) == 3


async def test_research_result_committed_before_parent_step_recovers_at_round_limit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from binance_agent.agents.research import ResearchState, build_news_research_graph
    from binance_agent.infra.settings import ResearchConfig
    from binance_agent.intel.evidence import evidence_document
    from test_news_research_graph import ResearchModel, initial, model_for

    store = CollaborationStore(tmp_path / "control.sqlite")
    saver = SQLiteCheckpointSaver(store.path)
    engine = AgentEngine(store, saver)
    session = store.create_session("BTCUSDT", "fapi", "s")
    run = store.start_turn(session["session_id"], "ETF是否获批", "t", "fixture")

    async def fetch(url: str) -> Any:
        return evidence_document(
            url=url, text="Bitcoin ETF application is pending, not approved.", level="body"
        )

    subgraph = build_news_research_graph(
        model_for(ResearchModel(responses=["unused"])),
        config=ResearchConfig(enabled=True, max_rounds=1),
        fetch=fetch,
        persist=store.save_evidence,
        load=store.load_evidence,
    )
    builder = StateGraph(ResearchState)
    builder.add_node("research", subgraph)
    builder.add_edge(START, "research")
    builder.add_edge("research", END)
    graph = builder.compile(checkpointer=saver)
    cfg = {"configurable": {"thread_id": run["run_id"]}}
    original = store.save_progress
    crash = True

    def save_progress(run_id: str, task_id: str, revision: int, scratch: Any) -> None:
        nonlocal crash
        if crash and scratch.get("stage") == "verify":
            crash = False
            raise OSError("child result committed, parent node not committed")
        original(run_id, task_id, revision, scratch)

    monkeypatch.setattr(store, "save_progress", save_progress)
    with collaboration_scope(scope_for(store, engine, run)), pytest.raises(OSError):
        await graph.ainvoke({**initial(), "research_run_id": run["run_id"]}, cfg, durability="sync")
    with collaboration_scope(scope_for(store, AgentEngine(store, saver), run)):
        restored = await graph.ainvoke(None, cfg, durability="sync")
    assert restored["news_research"]["facts"]
    assert not restored["news_research"]["gaps"]
    records = store.get(run["run_id"])
    researcher_ids = {
        agent["agent_id"]
        for agent in store.session(session["session_id"])["agents"]
        if agent["role"] == "news_researcher"
    }
    researcher_messages = [
        m
        for m in records["collaboration_messages"]
        if m["kind"] == "task_result" and m["sender"] in researcher_ids
    ]
    assert any(m["payload"]["result"]["data"].get("facts") for m in researcher_messages)
    with store._lock:
        rows = store._conn.execute(
            "SELECT rounds FROM research_progress WHERE run_id=?", (run["run_id"],)
        ).fetchall()
    assert [r[0] for r in rows] == [1]
    saver.close()
    store.close()


def test_incomplete_publication_clears_old_planner_and_keeps_audit(tmp_path: Path) -> None:
    store = CollaborationStore(tmp_path / "control.sqlite")
    session = store.create_session("BTCUSDT", "fapi", "s")
    run = store.start_turn(session["session_id"], "question", "t", "fixture")
    store.append_event(
        run["run_id"],
        "node",
        {
            "node": "strategy_planner",
            "update": {
                "planner_decision": {"approved": True},
                "planner_steps": [{"approved": True}],
            },
        },
    )
    store.inform(session["session_id"], run["run_id"], "retract assumptions", "m")
    assert store.finalize(run["run_id"], 1, {"decision_type": "no_trade"}, incomplete=True)
    record = store.get(run["run_id"])
    assert record["planner_decision"] is None and record["planner_steps"] == []
    assert any(e["event_type"] == "node" for e in record["events"])
    store.close()


async def test_desktop_shutdown_stops_background_work_and_wakes_subscription(
    tmp_path: Path,
) -> None:
    from binance_agent.desktop.shell import _shutdown_backend

    model = Model()
    model.release.clear()
    async with app_fixture(tmp_path, model) as (app, manager, _):
        app.state.backend_loop = asyncio.get_running_loop()
        session = manager.store.create_session("BTCUSDT", "fapi", "s")
        run = await manager.turn(session["session_id"], "question", "t")
        await asyncio.wait_for(model.entered.wait(), 5)
        stream = manager.subscribe(session["session_id"], after=10**9)
        waiter = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)

        class Server:
            @property
            def should_exit(self) -> bool:
                return False

            @should_exit.setter
            def should_exit(self, value: bool) -> None:
                assert value and manager.closing and not manager.tasks
                assert manager.store.run(run["run_id"])["status"] == "interrupted"

        class Thread:
            def join(self, timeout: float) -> None:
                pass

            def is_alive(self) -> bool:
                return False

        await asyncio.wait_for(asyncio.to_thread(_shutdown_backend, app, Server(), Thread()), 5)
        with pytest.raises(StopAsyncIteration):
            await waiter
        assert app.state.analysis_limiter._active == 0
        await stream.aclose()
