"""Focused tests for the row-oriented SQLite checkpoint saver."""

from __future__ import annotations

import asyncio
import pickle
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from binance_agent.app.checkpoints import SQLiteCheckpointSaver


def _checkpoint(checkpoint_id: str, *, value: Any, version: str) -> dict[str, Any]:
    return {
        "v": 1,
        "id": checkpoint_id,
        "ts": "2026-01-01T00:00:00+00:00",
        "channel_values": {"state": value},
        "channel_versions": {"state": version},
        "versions_seen": {},
        "updated_channels": ["state"],
    }


def _legacy_payload() -> dict[str, Any]:
    memory = MemorySaver()
    root_config = {"configurable": {"thread_id": "legacy-thread", "checkpoint_ns": "worker-a"}}
    first = _checkpoint("0001", value={"step": 1}, version="v1")
    first_config = memory.put(root_config, first, {"source": "input", "step": -1}, {"state": "v1"})
    memory.put_writes(first_config, [("task_result", {"ok": True})], "task-1", "path")
    child_config = {**first_config, "configurable": {**first_config["configurable"]}}
    child = _checkpoint("0002", value={"step": 2}, version="v2")
    memory.put(child_config, child, {"source": "loop", "step": 0}, {"state": "v2"})
    second_config = {"configurable": {"thread_id": "legacy-thread", "checkpoint_ns": "worker-b"}}
    memory.put(
        second_config,
        _checkpoint("0003", value="other namespace", version="v3"),
        {"source": "input", "step": -1},
        {"state": "v3"},
    )
    return {
        "storage": {
            thread: {namespace: dict(checkpoints) for namespace, checkpoints in namespaces.items()}
            for thread, namespaces in memory.storage.items()
        },
        "writes": dict(memory.writes),
        "blobs": dict(memory.blobs),
    }


def test_sqlite_rows_preserve_namespaces_parents_blobs_and_pending_writes(tmp_path: Path) -> None:
    saver = SQLiteCheckpointSaver(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "thread", "checkpoint_ns": "agent-a"}}
    first = saver.put(
        config,
        _checkpoint("0001", value={"step": 1}, version="v1"),
        {"source": "input", "step": -1},
        {"state": "v1"},
    )
    saver.put_writes(first, [("task_result", {"ok": True})], "task-1", "root")
    saver.put(
        first,
        _checkpoint("0002", value={"step": 2}, version="v2"),
        {"source": "loop", "step": 0},
        {"state": "v2"},
    )
    saver.put(
        {"configurable": {"thread_id": "thread", "checkpoint_ns": "agent-b"}},
        _checkpoint("0001", value="separate", version="v1"),
        {"source": "input", "step": -1},
        {"state": "v1"},
    )

    latest = saver.get_tuple(config)
    assert latest is not None
    assert latest.checkpoint["channel_values"] == {"state": {"step": 2}}
    assert latest.parent_config == {
        "configurable": {
            "thread_id": "thread",
            "checkpoint_ns": "agent-a",
            "checkpoint_id": "0001",
        }
    }
    first_tuple = saver.get_tuple(
        {"configurable": {**config["configurable"], "checkpoint_id": "0001"}}
    )
    assert first_tuple is not None
    assert first_tuple.pending_writes == [("task-1", "task_result", {"ok": True})]
    assert list(saver.list(config))
    assert len(list(saver.list(None))) == 3
    assert not hasattr(saver, "storage")

    with sqlite3.connect(tmp_path / "checkpoints.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM checkpoint_v2_checkpoints").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM checkpoint_v2_blobs").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM checkpoint_v2_writes").fetchone()[0] == 1
    saver.close()


def test_legacy_snapshot_migrates_once_and_retains_original_table(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite"
    payload = _legacy_payload()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE checkpoint_snapshots (key TEXT PRIMARY KEY, payload BLOB NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE checkpoint_threads (thread_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO checkpoint_snapshots VALUES (?, ?, ?)",
            ("memory_saver", sqlite3.Binary(pickle.dumps(payload)), "2026-01-01"),
        )
        conn.execute(
            "INSERT INTO checkpoint_threads VALUES (?, ?)", ("legacy-thread", "2026-01-01")
        )

    saver = SQLiteCheckpointSaver(db_path)
    config = {"configurable": {"thread_id": "legacy-thread", "checkpoint_ns": "worker-a"}}
    latest = saver.get_tuple(config)
    assert latest is not None
    assert latest.checkpoint["channel_values"] == {"state": {"step": 2}}
    assert latest.parent_config is not None
    assert latest.pending_writes == []
    assert saver.get_tuple(
        {"configurable": {"thread_id": "legacy-thread", "checkpoint_ns": "worker-b"}}
    )
    assert len(list(saver.list(config))) == 2
    saver.close()

    # Reopening must use the migration marker and must not duplicate rows.
    reopened = SQLiteCheckpointSaver(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM checkpoint_v2_checkpoints").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM checkpoint_snapshots").fetchone()[0] == 1
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM checkpoint_v2_meta WHERE key = 'legacy_memory_saver_migrated'"
            ).fetchone()[0]
            == 1
        )
    reopened.close()


def test_failed_legacy_migration_leaves_snapshot_and_allows_retry(tmp_path: Path) -> None:
    db_path = tmp_path / "broken.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE checkpoint_snapshots (key TEXT PRIMARY KEY, payload BLOB NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO checkpoint_snapshots VALUES (?, ?, ?)",
            ("memory_saver", sqlite3.Binary(b"not-a-pickle"), "2026-01-01"),
        )

    with pytest.raises(pickle.UnpicklingError):
        SQLiteCheckpointSaver(db_path)
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT payload FROM checkpoint_snapshots WHERE key = 'memory_saver'"
            ).fetchone()[0]
            == b"not-a-pickle"
        )
        assert conn.execute("SELECT COUNT(*) FROM checkpoint_v2_meta").fetchone()[0] == 0


def test_pins_protect_threads_from_cleanup(tmp_path: Path) -> None:
    saver = SQLiteCheckpointSaver(tmp_path / "cleanup.sqlite", retention=1)
    for thread_id in ("old", "new", "newest"):
        saver.put(
            {"configurable": {"thread_id": thread_id}},
            _checkpoint("0001", value=thread_id, version="v1"),
            {"source": "input", "step": -1},
            {"state": "v1"},
        )
        if thread_id == "old":
            saver.pin_thread(thread_id)
    deleted = saver.cleanup()
    assert deleted == ["new"]
    assert saver.has_thread("old")
    assert saver.has_thread("newest")
    assert not saver.has_thread("new")
    saver.unpin_thread("old")
    saver.close()


def test_async_langgraph_methods_are_real_storage_operations(tmp_path: Path) -> None:
    async def run() -> None:
        saver = SQLiteCheckpointSaver(tmp_path / "async.sqlite")
        config = {"configurable": {"thread_id": "async"}}
        saved = await saver.aput(
            config,
            _checkpoint("0001", value="async", version="v1"),
            {"source": "input", "step": -1},
            {"state": "v1"},
        )
        await saver.aput_writes(saved, [("value", 42)], "task")
        loaded = await saver.aget_tuple(config)
        assert loaded is not None
        assert loaded.checkpoint["channel_values"] == {"state": "async"}
        assert loaded.pending_writes == [("task", "value", 42)]
        assert [item async for item in saver.alist(config)]
        await saver.adelete_thread("async")
        assert not saver.has_thread("async")
        saver.close()

    asyncio.run(run())


def test_get_tuple_uses_one_read_snapshot_during_concurrent_delete(tmp_path: Path) -> None:
    ready = threading.Event()
    release = threading.Event()

    class _PausingSaver(SQLiteCheckpointSaver):
        def _row_to_tuple_locked(self, *args: Any, **kwargs: Any) -> Any:
            ready.set()
            assert release.wait(5)
            return super()._row_to_tuple_locked(*args, **kwargs)

    db_path = tmp_path / "concurrent.sqlite"
    saver = _PausingSaver(db_path)
    config = {"configurable": {"thread_id": "concurrent"}}
    saved = saver.put(
        config,
        _checkpoint("0001", value={"kept": True}, version="v1"),
        {"source": "input", "step": -1},
        {"state": "v1"},
    )
    saver.put_writes(saved, [("task_result", {"kept": True})], "task")
    reader_result: list[Any] = []
    reader_errors: list[BaseException] = []

    def read() -> None:
        try:
            reader_result.append(saver.get_tuple(config))
        except BaseException as exc:  # pragma: no cover - assertion reports the error
            reader_errors.append(exc)

    reader = threading.Thread(target=read)
    reader.start()
    assert ready.wait(5)

    deleter = SQLiteCheckpointSaver(db_path)
    deleter.delete_thread("concurrent")
    deleter.close()
    release.set()
    reader.join(5)
    saver.close()

    assert not reader_errors
    assert len(reader_result) == 1
    result = reader_result[0]
    assert result is not None
    assert result.checkpoint["channel_values"] == {"state": {"kept": True}}
    assert result.pending_writes == [("task", "task_result", {"kept": True})]
