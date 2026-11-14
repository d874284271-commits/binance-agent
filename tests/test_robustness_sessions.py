"""Shortcut admission is atomic; shutdown stays bounded and truthful."""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException

from binance_agent.app import research_sessions
from binance_agent.app.collaboration_store import CollaborationStore
from binance_agent.app.research_sessions import RunManager
from binance_agent.app.security import AnalysisLimiter
from binance_agent.infra.settings import Settings


@pytest.fixture
def manager(tmp_path):
    app = FastAPI()
    app.state.analysis_limiter = AnalysisLimiter(1)
    store = CollaborationStore(tmp_path / "collaboration.sqlite")
    engine = SimpleNamespace(evict=lambda **kw: None)
    settings = Settings.model_validate(
        {"research": {"enabled": True, "max_requests": 10, "deadline_s": 30}}
    )
    result = RunManager(
        app,
        settings,
        store,
        engine,
        prepare=None,
        stream_graph=None,
        fingerprint=lambda: "fixture",
    )
    yield result
    store.close()


def assert_empty(manager):
    for table in ("research_sessions", "research_runs", "research_requests", "research_events"):
        assert manager.store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert manager.app.state.analysis_limiter._active == 0
    assert not manager.tasks


async def test_shortcut_full_admission_creates_nothing(manager):
    reservation = await manager.app.state.analysis_limiter.try_acquire()
    with pytest.raises(HTTPException) as error:
        await manager.new_session_turn("BTCUSDT", "fapi", "question")
    assert error.value.status_code == 429
    await reservation.release()
    assert_empty(manager)


async def test_shortcut_task_creation_failure_rolls_back(manager, monkeypatch):
    def fail(coroutine):
        raise RuntimeError("task scheduler failed")

    monkeypatch.setattr(
        research_sessions,
        "asyncio",
        SimpleNamespace(create_task=fail, CancelledError=asyncio.CancelledError),
    )
    with pytest.raises(RuntimeError, match="scheduler"):
        await manager.new_session_turn("BTCUSDT", "fapi", "question")
    assert_empty(manager)


async def test_shortcut_commit_failure_rolls_back_and_cancels_unstarted_task(manager):
    original = manager.store._conn

    class Connection:
        def __getattr__(self, name):
            return getattr(original, name)

        def commit(self):
            raise sqlite3.OperationalError("disk full at commit")

    manager.store._conn = Connection()
    with pytest.raises(sqlite3.OperationalError, match="disk full"):
        await manager.new_session_turn("BTCUSDT", "fapi", "question")
    assert_empty(manager)


async def test_shortcut_transfers_reservation_only_after_atomic_commit(manager, monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    ownership = []

    async def drive(run_id, reservation, *, resume):
        ownership.append(reservation.detached)
        assert manager.store.run(run_id)["status"] == "running"
        started.set()
        try:
            await release.wait()
        finally:
            await reservation.release()

    monkeypatch.setattr(manager, "_drive", drive)
    session, turn = await manager.new_session_turn("BTCUSDT", "fapi", "question")
    assert started.is_set() and ownership == [True]
    assert session["active_run_id"] == turn["run_id"]
    assert manager.app.state.analysis_limiter._active == 1
    release.set()
    await manager.tasks[turn["run_id"]]
    assert manager.app.state.analysis_limiter._active == 0
    await manager.shutdown()


async def test_shutdown_maintenance_exception_does_not_skip_runs(manager, monkeypatch):
    cancelled = asyncio.Event()

    async def maintenance():
        raise RuntimeError("maintenance failed")

    async def worker():
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    manager._maintenance = asyncio.create_task(maintenance())
    manager.tasks["run"] = asyncio.create_task(worker())
    await asyncio.sleep(0)
    await manager.shutdown()
    assert cancelled.is_set() and not manager.pending_shutdown
    await manager.shutdown()


async def test_shutdown_reports_task_that_ignores_cancel_without_unbounded_gather(
    manager, monkeypatch
):
    stop = asyncio.Event()
    evicted = []
    monkeypatch.setattr(manager.engine, "evict", lambda **kw: evicted.append(True))

    async def worker():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await stop.wait()

    task = asyncio.create_task(worker())
    manager.tasks["run"] = task
    await asyncio.sleep(0)
    original_wait = asyncio.wait
    timeouts = []

    async def bounded_wait(tasks, *, timeout):
        timeouts.append(timeout)
        return await original_wait(tasks, timeout=0.001)

    monkeypatch.setattr(research_sessions.asyncio, "wait", bounded_wait)
    await manager.shutdown()
    assert timeouts == [5]
    assert manager.pending_shutdown and not task.done() and not evicted
    stop.set()
    await task
    assert not manager.pending_shutdown


async def test_drive_initialization_failure_releases_reservation(manager, monkeypatch):
    app = manager.app
    app.state.graph = SimpleNamespace()
    app.state.active_research_runs = set()

    def fail_pin(run_id):
        raise OSError("checkpoint unavailable")

    manager.engine.checkpointer = SimpleNamespace(pin_thread=fail_pin)
    _, turn = await manager.new_session_turn("BTCUSDT", "fapi", "question")
    # new_session_turn yields once, so the task may be finishing its shielded release.
    task = manager.tasks.get(turn["run_id"])
    if task is not None:
        await task
    for _ in range(5):
        if not app.state.analysis_limiter._active:
            break
        await asyncio.sleep(0)
    assert app.state.analysis_limiter._active == 0
    assert manager.store.run(turn["run_id"])["status"] == "interrupted"
    assert not manager.tasks and not app.state.active_research_runs
    await manager.shutdown()


async def test_existing_session_task_creation_failure_leaves_no_running_turn(manager, monkeypatch):
    session = manager.store.create_session("BTCUSDT", "fapi", "session")

    def fail(coroutine):
        raise RuntimeError("task scheduler failed")

    monkeypatch.setattr(
        research_sessions,
        "asyncio",
        SimpleNamespace(create_task=fail, CancelledError=asyncio.CancelledError),
    )
    with pytest.raises(RuntimeError, match="scheduler"):
        await manager.turn(session["session_id"], "question", "turn")
    assert manager.store.session(session["session_id"])["active_run_id"] is None
    assert manager.store._conn.execute("SELECT COUNT(*) FROM research_runs").fetchone()[0] == 0
    assert manager.app.state.analysis_limiter._active == 0


def test_collaboration_constructor_failure_closes_and_preserves_cause(tmp_path, monkeypatch):
    from binance_agent.app import collaboration_store
    from binance_agent.infra.startup_errors import StartupError

    class Connection:
        row_factory = None
        closed = False

        def execute(self, query):
            return self

        def executescript(self, schema):
            raise sqlite3.OperationalError("disk full: secret-value-must-not-be-shown")

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(collaboration_store.sqlite3, "connect", lambda *a, **kw: connection)
    path = tmp_path / "state.sqlite"
    with pytest.raises(StartupError) as error:
        CollaborationStore(path)
    assert connection.closed
    assert isinstance(error.value.__cause__, sqlite3.OperationalError)
    assert str(path) in str(error.value) and "OperationalError" in str(error.value)
    assert "secret-value" not in str(error.value)


def test_corrupt_collaboration_database_keeps_original_and_sidecars(tmp_path):
    from binance_agent.infra.startup_errors import StartupError

    path = tmp_path / "broken.sqlite"
    originals = {path: b"not SQLite"}
    for suffix in ("-wal", "-shm"):
        originals[path.with_name(path.name + suffix)] = b"original-sidecar"
    for file, content in originals.items():
        file.write_bytes(content)
    with pytest.raises(StartupError):
        CollaborationStore(path)
    assert all(file.read_bytes() == content for file, content in originals.items())
