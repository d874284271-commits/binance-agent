"""Product contracts: independent research, persisted versions and read-only export."""

import asyncio
import sqlite3
import time

import httpx
import pytest

from binance_agent.agents.budget import RunBudget
from binance_agent.agents.reports import ReportResult, report_markdown
from binance_agent.app.collaboration_store import CollaborationStore, SessionConflict
from binance_agent.intel.aggregator import IntelAggregator
from binance_agent.intel.base import Provider, ProviderResult
from binance_agent.intel.models import NewsItem
from binance_agent.research_demo import create_research_demo_app


async def test_independent_topic_never_calls_market_and_exports_saved_version(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    app = create_research_demo_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client,
    ):

        async def no_market(*args, **kwargs):
            raise AssertionError("independent topic requested Binance")

        monkeypatch.setattr(app.state.rest_client, "ticker_24h", no_market)
        app.state.market_snapshot_builder = no_market
        s = await client.post(
            "/research/sessions",
            json={
                "mode": "topic",
                "target": {"kind": "independent"},
                "question": "Digital asset regulation proposal",
                "request_id": "s",
            },
        )
        assert s.status_code == 201, s.text
        sid = s.json()["session_id"]
        turn = await client.post(
            f"/research/sessions/{sid}/turns", json={"action": "initial", "request_id": "t"}
        )
        rid = turn.json()["run_id"]
        await asyncio.wait_for(asyncio.shield(app.state.research_manager.tasks[rid]), 15)
        record = app.state.run_store.get(rid)
        assert record["status"] == "completed", record.get("error")
        assert record["symbol"] is None and record["market_snapshot"] is None
        assert record["final_advice"] is None
        report = (await client.get(f"/runs/{rid}/report")).json()["report"]
        assert report["kind"] == "topic" and report["target"] == {"kind": "independent"}
        assert report["basis"] and report["advice"] is None
        assert report["completeness"] in {"complete", "partial"}
        before = app.state.run_store.budget(rid)
        markdown = await client.get(f"/runs/{rid}/report.md")
        assert markdown.status_code == 200
        assert "Digital asset regulation" in markdown.text
        assert "BTCUSDT" not in markdown.text
        assert "runtime_events" not in markdown.text
        assert app.state.run_store.budget(rid) == before
        events = app.state.run_store.events(sid)
        final = [e for e in events if e["event"] == "final"]
        assert len(final) == 1 and final[0]["payload"]["report"] == report
        again = await client.post(
            f"/research/sessions/{sid}/turns", json={"action": "initial", "request_id": "t"}
        )
        assert again.json()["run_id"] == rid
        follow = await client.post(
            f"/research/sessions/{sid}/turns",
            json={"action": "followup", "question": "What remains pending?", "request_id": "t2"},
        )
        await asyncio.wait_for(
            asyncio.shield(app.state.research_manager.tasks[follow.json()["run_id"]]), 15
        )
        assert (await client.get(f"/runs/{rid}/report")).json()["report"] == report
        detail = (await client.get(f"/research/sessions/{sid}")).json()
        assert detail["root_question"] == "Digital asset regulation proposal"
        assert len(detail["runs"]) == 2


def test_session_intent_idempotency_and_revision_fence(tmp_path):
    store = CollaborationStore(tmp_path / "research.sqlite")
    target = {"kind": "independent"}
    s = store.create_session(None, None, "s", mode="topic", target=target, question="监管进展")
    with pytest.raises(SessionConflict):
        store.create_session(
            "BTCUSDT",
            "fapi",
            "s",
            mode="report",
            target={"kind": "asset", "symbol": "BTCUSDT", "market": "fapi"},
        )
    r = store.start_turn(s["session_id"], "", "t", "fixture", "initial")
    report = ReportResult(
        kind="topic",
        title="监管进展",
        target=target,
        analysis_at="2026-11-14",
        summary="证据不足",
        completeness="insufficient_evidence",
    )
    store.inform(s["session_id"], r["run_id"], "补充地区限制", "m")
    assert not store.finalize(r["run_id"], 0, report=report)
    assert not store.finalize(r["run_id"], 1, report=report)
    assert not [e for e in store.events(s["session_id"]) if e["event"] == "final"]
    store.close()


def test_migration_preserves_legacy_and_checkpoints(tmp_path):
    path = tmp_path / "shared.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""CREATE TABLE research_sessions (session_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, market TEXT NOT NULL, status TEXT NOT NULL, root_question TEXT NOT NULL DEFAULT '', active_run_id TEXT, created_at TEXT NOT NULL);
        INSERT INTO research_sessions VALUES ('old','ETHUSDT','api','open','old question',NULL,'2026-01-01');
        CREATE TABLE checkpoints (payload TEXT); INSERT INTO checkpoints VALUES ('retain');""")
    store = CollaborationStore(path)
    assert store.session("old")["mode"] is None
    assert store.session("old")["root_question"] == "old question"
    assert store._conn.execute("SELECT payload FROM checkpoints").fetchone()[0] == "retain"
    backup = path.with_name(path.name + ".before-reports-v1.bak")
    assert backup.exists()
    with sqlite3.connect(backup) as db:
        assert db.execute("SELECT symbol FROM research_sessions").fetchone()[0] == "ETHUSDT"
    store.close()
    store = CollaborationStore(path)
    assert store._conn.execute("SELECT COUNT(*) FROM research_schema").fetchone()[0] == 1
    store.close()


def test_export_escapes_external_html():
    report = ReportResult(
        kind="topic",
        title="<script>bad</script>",
        target={"kind": "independent"},
        analysis_at="today",
        summary="<img src=x>",
        completeness="insufficient_evidence",
    )
    assert "<script>" not in report_markdown(report)


async def test_independent_discovery_skips_asset_only_providers():
    calls = []

    class AssetOnly(Provider):
        name = "asset"

        async def fetch(self, ctx):
            raise AssertionError("asset provider must not run")

    class Public(Provider):
        name = "public"
        supports_independent = True

        async def fetch(self, ctx):
            calls.append((ctx.symbol, ctx.question))
            return ProviderResult(
                source=self.name,
                news_items=[
                    NewsItem(title="Regulation proposal", url="https://example.invalid/public")
                ],
            )

    result = await IntelAggregator([AssetOnly(), Public()]).gather(
        None, research_question="regulation"
    )
    assert calls == [(None, "regulation")]
    assert result.symbol is None and len(result.news_items) == 1
    assert result.source_status["asset"].startswith("not_applicable")


async def test_expired_topic_resume_preserves_budget_and_publishes_no_trade_advice(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    app = create_research_demo_app()
    async with app.router.lifespan_context(app):
        manager = app.state.research_manager
        store = manager.store
        session = store.create_session(
            None, None, "s", mode="topic", target={"kind": "independent"}, question="regulation"
        )
        run = store.start_turn(session["session_id"], "", "t", manager.fingerprint(), "initial")
        rid = run["run_id"]
        store.save_initial(
            rid,
            {
                "mode": "topic",
                "target": {"kind": "independent"},
                "title": "regulation",
                "analysis_at": "2026-11-14",
            },
        )
        budget = RunBudget(max_requests=7, deadline_at=time.time() - 1).snapshot()
        budget["requests_reserved"] = 7
        store.append_event(rid, "research_budget", budget)
        store.status(rid, "interrupted", "fixture")
        await manager.control(rid, "resume", "resume")
        task = manager.tasks.get(rid)
        if task:
            await asyncio.wait_for(asyncio.shield(task), 5)
        record = store.get(rid)
        assert record["status"] == "completed", record.get("error")
        assert record["report"]["completeness"] == "insufficient_evidence"
        assert record["report"]["advice"] is None and record["final_advice"] is None
        assert store.budget(rid)["requests_reserved"] == 7
        assert store.budget(rid)["deadline_at"] == budget["deadline_at"]


def test_migration_failure_rolls_back_without_losing_original_tables(tmp_path):
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""CREATE TABLE research_sessions (session_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, market TEXT NOT NULL, status TEXT NOT NULL, root_question TEXT NOT NULL DEFAULT '', active_run_id TEXT, created_at TEXT NOT NULL);
        INSERT INTO research_sessions VALUES ('old','ETHUSDT','api','open','keep',NULL,'2026-01-01');
        CREATE TABLE research_runs (run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, status TEXT NOT NULL, input_revision INTEGER DEFAULT 0, applied_revision INTEGER DEFAULT 0, question TEXT NOT NULL, fingerprint TEXT NOT NULL, record TEXT NOT NULL, initial_state TEXT, created_at TEXT NOT NULL, report TEXT);""")
    with pytest.raises(Exception, match="未自动重建数据库"):
        CollaborationStore(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT root_question FROM research_sessions").fetchone()[0] == "keep"
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='research_sessions_v1'"
        ).fetchone()
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='research_schema'").fetchone()


async def test_search_failure_keeps_public_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    app = create_research_demo_app()

    async def unavailable(query):
        raise httpx.ConnectError("search unavailable")

    app.state.research_search = unavailable
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client,
    ):
        session = (
            await client.post(
                "/research/sessions",
                json={
                    "mode": "topic",
                    "target": {"kind": "independent"},
                    "question": "Digital asset regulation",
                    "request_id": "s",
                },
            )
        ).json()
        run = (
            await client.post(
                f"/research/sessions/{session['session_id']}/turns",
                json={"action": "initial", "request_id": "t"},
            )
        ).json()
        await asyncio.wait_for(asyncio.shield(app.state.research_manager.tasks[run["run_id"]]), 15)
        record = app.state.run_store.get(run["run_id"])
        assert record["status"] == "completed", record.get("error")
        assert record["report"]["basis"]
        assert any("搜索" in gap for gap in record["report"]["gaps"])
        assert record["report"]["completeness"] == "partial"


@pytest.mark.parametrize("phase", ["queries", "candidates"])
async def test_topic_supplement_restarts_revision_in_same_run_and_budget(
    tmp_path, monkeypatch, phase
):
    from binance_agent.research_demo import ResearchDemoContent, ResearchDemoModel

    fetched = []
    original_fetch = ResearchDemoContent.fetch

    async def fetch(self, url):
        fetched.append(url)
        return await original_fetch(self, url)

    monkeypatch.setattr(ResearchDemoContent, "fetch", fetch)
    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    entered, release = asyncio.Event(), asyncio.Event()
    original = ResearchDemoModel.ainvoke
    paused = False

    async def model(self, messages, **kwargs):
        nonlocal paused
        import json

        payload = json.loads(messages[-1].content)
        matches = (
            '"queries"' in str(messages[0].content)
            if phase == "queries"
            else "candidates" in payload
        )
        if matches and not paused:
            paused = True
            entered.set()
            await release.wait()
        return await original(self, messages, **kwargs)

    monkeypatch.setattr(ResearchDemoModel, "ainvoke", model)
    app = create_research_demo_app()
    async with app.router.lifespan_context(app):
        manager = app.state.research_manager
        session = manager.store.create_session(
            None, None, "s", mode="topic", target={"kind": "independent"}, question="regulation"
        )
        sid = session["session_id"]
        turn = await manager.turn(sid, "", "t", "initial")
        rid = turn["run_id"]
        await asyncio.wait_for(entered.wait(), 5)
        before = manager.store.budget(rid)
        manager.store.inform(sid, rid, "补充:请核实生效日期", "m")
        release.set()
        await asyncio.wait_for(asyncio.shield(manager.tasks[rid]), 15)
        record = manager.store.get(rid)
        assert record["status"] == "completed", record.get("error")
        report = record["report"]
        assert report["input_revision"] == 1 and "生效日期" in report["question"]
        assert report["basis"] and report["completeness"] == "complete"
        documents = manager.store.load_evidence(rid)
        assert any(d.level == "body" and d.status == "available" for d in documents)
        assert fetched == ["https://example.invalid/independent"]
        final = [e for e in manager.store.events(sid) if e["event"] == "final"]
        assert len(final) == 1 and final[0]["payload"]["report"]["input_revision"] == 1
        after = manager.store.budget(rid)
        assert after["deadline_at"] == before["deadline_at"]
        assert after["requests_reserved"] > before["requests_reserved"]
        assert not any(e["event"] == "interrupted" for e in manager.store.events(sid))


async def test_topic_preserves_noncritical_task_gaps_in_verified_report(tmp_path, monkeypatch):
    import json

    from langchain_core.messages import AIMessage

    from binance_agent.research_demo import ResearchDemoModel

    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    original = ResearchDemoModel.ainvoke
    verifier_payloads = []

    async def model(self, messages, **kwargs):
        payload = json.loads(messages[-1].content)
        result = await original(self, messages, **kwargs)
        data = json.loads(result.content)
        if "candidates" in payload:
            data["tasks"][0]["critical"] = False
        if "documents" in payload and "task" in payload:
            data["gaps"] = ["生效日期缺失"]
        if "report" in payload:
            verifier_payloads.append(payload)
        return AIMessage(content=json.dumps(data))

    monkeypatch.setattr(ResearchDemoModel, "ainvoke", model)
    app = create_research_demo_app()
    async with app.router.lifespan_context(app):
        manager = app.state.research_manager
        session = manager.store.create_session(
            None, None, "s", mode="topic", target={"kind": "independent"}, question="regulation"
        )
        run = await manager.turn(session["session_id"], "", "t", "initial")
        await asyncio.wait_for(asyncio.shield(manager.tasks[run["run_id"]]), 15)
        record = manager.store.get(run["run_id"])
        assert record["status"] == "completed", record.get("error")
        assert record["report"]["completeness"] == "partial"
        assert any("生效日期缺失" in gap for gap in record["report"]["gaps"])
        assert verifier_payloads and any(
            "生效日期缺失" in gap for gap in verifier_payloads[-1]["task_gaps"]
        )
        assert verifier_payloads[-1]["unresolved_tasks"]


async def test_historical_topic_passes_dates_to_discovery_and_retains_evidence(
    tmp_path, monkeypatch
):
    import json
    from datetime import date, datetime, timezone

    from langchain_core.messages import AIMessage

    from binance_agent.research_demo import ResearchDemoModel

    monkeypatch.setenv("BINANCE_AGENT_DATA_DIR", str(tmp_path))
    original = ResearchDemoModel.ainvoke
    contexts = []

    async def model(self, messages, **kwargs):
        if '"queries"' in str(messages[0].content):
            return AIMessage(
                content=json.dumps(
                    {
                        "queries": ["digital asset regulation November 2025"],
                        "time_scope": "2025年11月",
                        "start_date": "2025-11-01",
                        "end_date": "2025-11-30",
                    }
                )
            )
        return await original(self, messages, **kwargs)

    class Archive(Provider):
        name = "archive"
        supports_independent = True

        async def fetch(self, ctx):
            contexts.append(ctx)
            return ProviderResult(
                source=self.name,
                news_items=[
                    NewsItem(
                        title="Digital asset regulation proposal",
                        url="https://example.invalid/independent",
                        published_at=datetime(2025, 11, 30, 23, 59, tzinfo=timezone.utc),
                    ),
                    NewsItem(
                        title="New regulation outside requested period",
                        url="https://example.invalid/recent",
                        published_at=datetime.now(timezone.utc),
                    ),
                    NewsItem(
                        title="Earlier regulation",
                        url="https://example.invalid/earlier",
                        published_at=datetime(2025, 10, 31, tzinfo=timezone.utc),
                    ),
                ],
            )

    monkeypatch.setattr(ResearchDemoModel, "ainvoke", model)
    app = create_research_demo_app()
    async with app.router.lifespan_context(app):
        app.state.aggregator = IntelAggregator([Archive()], window_hours=168)
        manager = app.state.research_manager
        session = manager.store.create_session(
            None,
            None,
            "s",
            mode="topic",
            target={"kind": "independent"},
            question="2025 年 11 月监管变化",
        )
        run = await manager.turn(session["session_id"], "", "t", "initial")
        await asyncio.wait_for(asyncio.shield(manager.tasks[run["run_id"]]), 15)
        record = manager.store.get(run["run_id"])
        assert record["status"] == "completed", record.get("error")
        assert contexts[0].start_date == date(2025, 11, 1)
        assert contexts[0].end_date == date(2025, 11, 30)
        assert [n["url"] for n in record["intel_snapshot"]["news_items"]] == [
            "https://example.invalid/independent"
        ]
        assert record["report"]["basis"]
        discovery = next(
            n["update"]["discovery"] for n in record["node_outputs"] if n["update"].get("discovery")
        )
        assert discovery["start_date"] == "2025-11-01"


@pytest.mark.parametrize(
    "start,end", [("2025-02-30", None), ("2025-12-01", "2025-11-01"), ("20251101", None)]
)
def test_topic_plan_rejects_invalid_date_ranges(start, end):
    from binance_agent.agents.topic import _plan_output

    with pytest.raises(ValueError):
        _plan_output(
            {
                "queries": ["regulation"],
                "time_scope": "specified",
                "start_date": start,
                "end_date": end,
            }
        )
