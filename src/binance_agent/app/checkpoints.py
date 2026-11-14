"""LangGraph 检查点存储,按线程和 namespace 读取需要的状态。

SQLiteCheckpointSaver 将 checkpoint、blob 和 pending write 分行存储,避免启动时
把全库装入内存。首次打开旧数据库时以事务迁移 checkpoint_snapshots,并保留旧表。
"""

from __future__ import annotations

import asyncio
import builtins
import pickle
import random
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memory import MemorySaver

from ..infra.startup_errors import startup_error

if TYPE_CHECKING:  # LangGraph keeps these aliases internal to its checkpoint package.
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import (
        ChannelVersions,
        Checkpoint,
        CheckpointMetadata,
        SerializerProtocol,
    )
else:
    RunnableConfig = dict[str, Any]
    Checkpoint = dict[str, Any]
    CheckpointMetadata = dict[str, Any]
    ChannelVersions = dict[str, Any]
    SerializerProtocol = Any


CheckpointBackend = Literal["memory", "sqlite"]
_V = TypeVar("_V", int, float, str)


class SQLiteCheckpointSaver(
    BaseCheckpointSaver[str],
    AbstractContextManager["SQLiteCheckpointSaver"],
    AbstractAsyncContextManager["SQLiteCheckpointSaver"],
):
    """线程安全的 SQLite 检查点存储,与协作表共享数据库但使用独立表前缀。

    serde 只序列化单个带类型的值,不序列化整个数据库;调用者负责关闭连接,
    retention 与线程 pin 配合保留需要恢复的状态。
    """

    _LEGACY_SNAPSHOT_KEY = "memory_saver"
    _META_LEGACY_MIGRATED = "legacy_memory_saver_migrated"

    def __init__(
        self,
        path: Path | str,
        *,
        retention: int | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.path = Path(path)
        self.retention = retention
        self._lock = threading.RLock()
        self._closed = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._validate_existing_database()
            self._conn = sqlite3.connect(
                self.path,
                check_same_thread=False,
                timeout=30.0,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()
            with self._lock:
                self._migrate_legacy_locked()
        except BaseException as exc:
            if hasattr(self, "_conn"):
                self._conn.close()
                self._closed = True
            if isinstance(exc, (sqlite3.Error, OSError)):
                raise startup_error("checkpoint SQLite", self.path, exc, sqlite=True) from exc
            raise

    def _validate_existing_database(self) -> None:
        """只读检查数据库主体,损坏时保留原 WAL/SHM;正常连接随后执行常规 WAL 恢复。"""
        # Opening a corrupt database writable and then closing it can discard its
        # original WAL/SHM. Validate the base image without touching either file.
        # A WAL database's base image is itself a consistent earlier checkpoint;
        # the normal connection below still performs ordinary WAL recovery.
        if not self.path.exists():
            return
        uri = self.path.resolve().as_uri() + "?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, timeout=30.0)
        try:
            rows = connection.execute("PRAGMA quick_check").fetchall()
            if rows != [("ok",)]:
                raise sqlite3.DatabaseError("checkpoint database integrity check failed")
        finally:
            connection.close()

    @property
    def backend(self) -> str:
        return "sqlite"

    def __enter__(self) -> SQLiteCheckpointSaver:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> SQLiteCheckpointSaver:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id, checkpoint_ns = _thread_and_namespace(config)
        requested_id = get_checkpoint_id(config)
        with self._lock, self._read_snapshot_locked():
            if requested_id is None:
                row = self._conn.execute(
                    """
                    SELECT checkpoint_id, checkpoint, checkpoint_type,
                           metadata, metadata_type, parent_checkpoint_id
                    FROM checkpoint_v2_checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY checkpoint_id DESC LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                ).fetchone()
            else:
                row = self._conn.execute(
                    """
                    SELECT checkpoint_id, checkpoint, checkpoint_type,
                           metadata, metadata_type, parent_checkpoint_id
                    FROM checkpoint_v2_checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (thread_id, checkpoint_ns, requested_id),
                ).fetchone()
            if row is None:
                return None
            return self._row_to_tuple_locked(
                thread_id,
                checkpoint_ns,
                row,
                config if requested_id is not None else None,
            )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        with self._lock, self._read_snapshot_locked():
            rows = self._list_rows_locked(config, filter=filter, before=before, limit=limit)
            result = [self._row_to_tuple_locked(row[0], row[1], row[2:], None) for row in rows]
        yield from result

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id, checkpoint_ns = _thread_and_namespace(config)
        checkpoint_id = str(checkpoint["id"])
        parent_checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        checkpoint_copy: dict[str, Any] = dict(checkpoint)
        values = checkpoint_copy.pop("channel_values", {})
        metadata_value = get_checkpoint_metadata(config, metadata)
        checkpoint_type, checkpoint_payload = _split_typed(self.serde.dumps_typed(checkpoint_copy))
        metadata_type, metadata_payload = _split_typed(self.serde.dumps_typed(metadata_value))
        now = _now_iso()

        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO checkpoint_v2_checkpoints(
                    thread_id, checkpoint_ns, checkpoint_id,
                    checkpoint, checkpoint_type, metadata, metadata_type,
                    parent_checkpoint_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id, checkpoint_ns, checkpoint_id) DO UPDATE SET
                    checkpoint = excluded.checkpoint,
                    checkpoint_type = excluded.checkpoint_type,
                    metadata = excluded.metadata,
                    metadata_type = excluded.metadata_type,
                    parent_checkpoint_id = excluded.parent_checkpoint_id,
                    updated_at = excluded.updated_at
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    sqlite3.Binary(checkpoint_payload),
                    checkpoint_type,
                    sqlite3.Binary(metadata_payload),
                    metadata_type,
                    str(parent_checkpoint_id) if parent_checkpoint_id is not None else None,
                    now,
                ),
            )
            for channel, version in new_versions.items():
                if channel in values:
                    value_type, value_payload = _split_typed(
                        self.serde.dumps_typed(values[channel])
                    )
                else:
                    value_type, value_payload = "empty", b""
                self._conn.execute(
                    """
                    INSERT INTO checkpoint_v2_blobs(
                        thread_id, checkpoint_ns, channel, version_key,
                        value, value_type
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id, checkpoint_ns, channel, version_key) DO UPDATE SET
                        value = excluded.value, value_type = excluded.value_type
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        channel,
                        _version_key(version),
                        sqlite3.Binary(value_payload),
                        value_type,
                    ),
                )
            self._touch_thread_locked(thread_id, now)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, checkpoint_ns = _thread_and_namespace(config)
        checkpoint_id = str(config["configurable"]["checkpoint_id"])
        with self._lock, self._conn:
            max_order = self._conn.execute(
                """
                SELECT COALESCE(MAX(write_order), -1)
                FROM checkpoint_v2_writes
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                """,
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchone()[0]
            for index, (channel, value) in enumerate(writes):
                write_index = WRITES_IDX_MAP.get(channel, index)
                existing = self._conn.execute(
                    """
                    SELECT write_order FROM checkpoint_v2_writes
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                      AND task_id = ? AND write_index = ?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, write_index),
                ).fetchone()
                if write_index >= 0 and existing is not None:
                    continue
                value_type, value_payload = _split_typed(self.serde.dumps_typed(value))
                if existing is None:
                    max_order += 1
                    write_order = max_order
                else:
                    write_order = existing[0]
                self._conn.execute(
                    """
                    INSERT INTO checkpoint_v2_writes(
                        thread_id, checkpoint_ns, checkpoint_id, task_id,
                        write_index, channel, value, value_type, task_path, write_order
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id, checkpoint_ns, checkpoint_id, task_id, write_index)
                    DO UPDATE SET channel = excluded.channel, value = excluded.value,
                        value_type = excluded.value_type, task_path = excluded.task_path,
                        write_order = excluded.write_order
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_index,
                        channel,
                        sqlite3.Binary(value_payload),
                        value_type,
                        task_path,
                        write_order,
                    ),
                )
            self._touch_thread_locked(thread_id, _now_iso())

    def delete_thread(self, thread_id: str) -> None:
        with self._lock, self._conn:
            self._delete_thread_locked(thread_id)

    def has_thread(self, thread_id: str) -> bool:
        with self._lock:
            return (
                self._conn.execute(
                    "SELECT 1 FROM checkpoint_v2_checkpoints WHERE thread_id = ? LIMIT 1",
                    (thread_id,),
                ).fetchone()
                is not None
            )

    def pin_thread(self, thread_id: str) -> None:
        """Protect a thread from retention cleanup until ``unpin_thread``."""

        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO checkpoint_v2_pins(thread_id, pinned_at)
                VALUES (?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET pinned_at = excluded.pinned_at
                """,
                (thread_id, _now_iso()),
            )

    def unpin_thread(self, thread_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM checkpoint_v2_pins WHERE thread_id = ?", (thread_id,))

    def is_thread_pinned(self, thread_id: str) -> bool:
        with self._lock:
            return (
                self._conn.execute(
                    "SELECT 1 FROM checkpoint_v2_pins WHERE thread_id = ?", (thread_id,)
                ).fetchone()
                is not None
            )

    protect_thread = pin_thread
    unprotect_thread = unpin_thread

    def cleanup(
        self,
        *,
        retain_latest: int | None = None,
        keep_thread_ids: Iterable[str] = (),
    ) -> builtins.list[str]:
        """Delete old unprotected threads and return their IDs."""

        with self._lock, self._conn:
            limit = retain_latest if retain_latest is not None else self.retention
            if limit is None or limit <= 0:
                return []
            keep = set(keep_thread_ids)
            pinned = {
                row[0] for row in self._conn.execute("SELECT thread_id FROM checkpoint_v2_pins")
            }
            rows = self._conn.execute(
                "SELECT thread_id FROM checkpoint_v2_threads ORDER BY updated_at DESC, thread_id DESC"
            ).fetchall()
            retained = 0
            deleted: list[str] = []
            for (thread_id,) in rows:
                if thread_id in keep or thread_id in pinned:
                    continue
                if retained < limit:
                    retained += 1
                    continue
                self._delete_thread_locked(thread_id)
                deleted.append(thread_id)
            return deleted

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        rows = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for row in rows:
            yield row

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)

    def get_next_version(self, current: _V | None, channel: None = None) -> str:
        """Return LangGraph's sortable version format used by MemorySaver."""

        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        return f"{current_v + 1:032}.{random.random():016}"

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_snapshots (
                    key TEXT PRIMARY KEY,
                    payload BLOB NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoint_threads (
                    thread_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoint_v2_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoint_v2_threads (
                    thread_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoint_v2_pins (
                    thread_id TEXT PRIMARY KEY,
                    pinned_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoint_v2_checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    checkpoint BLOB NOT NULL,
                    checkpoint_type TEXT NOT NULL,
                    metadata BLOB NOT NULL,
                    metadata_type TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(thread_id, checkpoint_ns, checkpoint_id)
                );
                CREATE INDEX IF NOT EXISTS checkpoint_v2_checkpoints_latest
                    ON checkpoint_v2_checkpoints(thread_id, checkpoint_ns, checkpoint_id DESC);
                CREATE TABLE IF NOT EXISTS checkpoint_v2_blobs (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    version_key TEXT NOT NULL,
                    value BLOB NOT NULL,
                    value_type TEXT NOT NULL,
                    PRIMARY KEY(thread_id, checkpoint_ns, channel, version_key)
                );
                CREATE TABLE IF NOT EXISTS checkpoint_v2_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    write_index INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    value BLOB NOT NULL,
                    value_type TEXT NOT NULL,
                    task_path TEXT NOT NULL,
                    write_order INTEGER NOT NULL,
                    PRIMARY KEY(thread_id, checkpoint_ns, checkpoint_id, task_id, write_index)
                );
                CREATE INDEX IF NOT EXISTS checkpoint_v2_writes_order
                    ON checkpoint_v2_writes(thread_id, checkpoint_ns, checkpoint_id, write_order);
                """
            )

    def _migrate_legacy_locked(self) -> None:
        # Any malformed pickle or row raises before the marker is committed,
        # so retrying is safe and the legacy table remains untouched.
        with self._conn:
            marker = self._conn.execute(
                "SELECT value FROM checkpoint_v2_meta WHERE key = ?",
                (self._META_LEGACY_MIGRATED,),
            ).fetchone()
            if marker is not None:
                return
            row = self._conn.execute(
                "SELECT payload FROM checkpoint_snapshots WHERE key = ?",
                (self._LEGACY_SNAPSHOT_KEY,),
            ).fetchone()
            if row is not None:
                payload = pickle.loads(row[0])
                self._import_legacy_payload_locked(payload)
            self._conn.execute(
                "INSERT INTO checkpoint_v2_meta(key, value) VALUES (?, ?)",
                (self._META_LEGACY_MIGRATED, _now_iso()),
            )

    def _import_legacy_payload_locked(self, payload: dict[str, Any]) -> None:
        storage = payload.get("storage") or {}
        writes = payload.get("writes") or {}
        blobs = payload.get("blobs") or {}
        now = _now_iso()
        for thread_id, namespaces in storage.items():
            thread_id = str(thread_id)
            self._touch_thread_locked(thread_id, now)
            for checkpoint_ns, checkpoints in (namespaces or {}).items():
                checkpoint_ns = str(checkpoint_ns)
                for checkpoint_id, value in (checkpoints or {}).items():
                    checkpoint_blob, metadata_blob, parent_id = value
                    checkpoint_type, checkpoint_payload = _split_typed(checkpoint_blob)
                    metadata_type, metadata_payload = _split_typed(metadata_blob)
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO checkpoint_v2_checkpoints(
                            thread_id, checkpoint_ns, checkpoint_id,
                            checkpoint, checkpoint_type, metadata, metadata_type,
                            parent_checkpoint_id, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            thread_id,
                            checkpoint_ns,
                            str(checkpoint_id),
                            sqlite3.Binary(checkpoint_payload),
                            checkpoint_type,
                            sqlite3.Binary(metadata_payload),
                            metadata_type,
                            str(parent_id) if parent_id is not None else None,
                            now,
                        ),
                    )
        for key, entries in writes.items():
            thread_id, checkpoint_ns, checkpoint_id = key
            thread_id, checkpoint_ns, checkpoint_id = (
                str(thread_id),
                str(checkpoint_ns),
                str(checkpoint_id),
            )
            for write_order, ((task_id, write_index), value) in enumerate((entries or {}).items()):
                task_id, channel, serialized_value, task_path = value
                value_type, value_payload = _split_typed(serialized_value)
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO checkpoint_v2_writes(
                        thread_id, checkpoint_ns, checkpoint_id, task_id,
                        write_index, channel, value, value_type, task_path, write_order
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        str(task_id),
                        int(write_index),
                        str(channel),
                        sqlite3.Binary(value_payload),
                        value_type,
                        str(task_path),
                        write_order,
                    ),
                )
        for key, serialized_value in blobs.items():
            thread_id, checkpoint_ns, channel, version = key
            value_type, value_payload = _split_typed(serialized_value)
            self._conn.execute(
                """
                INSERT OR IGNORE INTO checkpoint_v2_blobs(
                    thread_id, checkpoint_ns, channel, version_key, value, value_type
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(thread_id),
                    str(checkpoint_ns),
                    str(channel),
                    _version_key(version),
                    sqlite3.Binary(value_payload),
                    value_type,
                ),
            )
        for thread_id, updated_at in self._conn.execute(
            "SELECT thread_id, updated_at FROM checkpoint_threads"
        ).fetchall():
            self._conn.execute(
                """
                INSERT INTO checkpoint_v2_threads(thread_id, updated_at)
                VALUES (?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (thread_id, updated_at),
            )

    def _list_rows_locked(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None,
        before: RunnableConfig | None,
        limit: int | None,
    ) -> builtins.list[tuple[Any, ...]]:
        if limit is not None and limit <= 0:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        configurable = config.get("configurable", {}) if config else {}
        if config is not None and "thread_id" in configurable:
            clauses.append("thread_id = ?")
            params.append(str(configurable["thread_id"]))
        if config is not None and "checkpoint_ns" in configurable:
            clauses.append("checkpoint_ns = ?")
            params.append(str(configurable["checkpoint_ns"]))
        config_checkpoint_id = get_checkpoint_id(config) if config else None
        if config_checkpoint_id:
            clauses.append("checkpoint_id = ?")
            params.append(str(config_checkpoint_id))
        if before is not None:
            before_id = get_checkpoint_id(before)
            if before_id:
                clauses.append("checkpoint_id < ?")
                params.append(str(before_id))
        query = """
            SELECT thread_id, checkpoint_ns, checkpoint_id,
                   checkpoint, checkpoint_type, metadata, metadata_type,
                   parent_checkpoint_id
            FROM checkpoint_v2_checkpoints
        """
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC"
        if limit is not None and not filter:
            query += " LIMIT ?"
            params.append(max(0, limit))
        rows = self._conn.execute(query, params).fetchall()
        result: list[tuple[Any, ...]] = []
        for row in rows:
            metadata = self._decode_typed(row[5], row[6])
            if filter and not all(value == metadata.get(key) for key, value in filter.items()):
                continue
            result.append(row)
            if limit is not None and len(result) >= limit:
                break
        return result

    def _row_to_tuple_locked(
        self,
        thread_id: str,
        checkpoint_ns: str,
        row: tuple[Any, ...],
        requested_config: RunnableConfig | None,
    ) -> CheckpointTuple:
        (
            checkpoint_id,
            checkpoint_payload,
            checkpoint_type,
            metadata_payload,
            metadata_type,
            parent_id,
        ) = row
        checkpoint = self._decode_typed(checkpoint_payload, checkpoint_type)
        channel_versions = checkpoint.get("channel_versions", {})
        channel_values: dict[str, Any] = {}
        for channel, version in channel_versions.items():
            blob = self._conn.execute(
                """
                SELECT value, value_type FROM checkpoint_v2_blobs
                WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? AND version_key = ?
                """,
                (thread_id, checkpoint_ns, channel, _version_key(version)),
            ).fetchone()
            if blob is None or blob[1] == "empty":
                continue
            channel_values[channel] = self._decode_typed(blob[0], blob[1])
        checkpoint = cast(Checkpoint, {**checkpoint, "channel_values": channel_values})
        writes = self._conn.execute(
            """
            SELECT task_id, channel, value, value_type
            FROM checkpoint_v2_writes
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            ORDER BY write_order
            """,
            (thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()
        pending_writes = [
            (task_id, channel, self._decode_typed(value, value_type))
            for task_id, channel, value, value_type in writes
        ]
        config = requested_config or {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        parent_config = cast(
            RunnableConfig | None,
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }
            if parent_id
            else None,
        )
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=self._decode_typed(metadata_payload, metadata_type),
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def _decode_typed(self, payload: bytes, value_type: str) -> Any:
        return self.serde.loads_typed((value_type, bytes(payload)))

    @contextmanager
    def _read_snapshot_locked(self) -> Iterator[None]:
        """Read all checkpoint parts from one SQLite snapshot.

        ``sqlite3`` does not start a transaction for a SELECT by default, so
        the checkpoint row, its channel blobs, and pending writes would
        otherwise be allowed to observe different commits from another
        connection.  The caller already owns ``_lock``; this transaction also
        makes reads through a second saver connection stable while cleanup is
        running.
        """

        self._conn.execute("BEGIN")
        try:
            yield
        finally:
            self._conn.rollback()

    def _touch_thread_locked(self, thread_id: str, updated_at: str) -> None:
        self._conn.execute(
            """
            INSERT INTO checkpoint_v2_threads(thread_id, updated_at)
            VALUES (?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (thread_id, updated_at),
        )
        # Keep the legacy index for old diagnostic tools.  The legacy payload
        # itself is never rewritten and remains available for migration audit.
        self._conn.execute(
            """
            INSERT INTO checkpoint_threads(thread_id, updated_at)
            VALUES (?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (thread_id, updated_at),
        )

    def _delete_thread_locked(self, thread_id: str) -> None:
        self._conn.execute(
            "DELETE FROM checkpoint_v2_checkpoints WHERE thread_id = ?", (thread_id,)
        )
        self._conn.execute("DELETE FROM checkpoint_v2_blobs WHERE thread_id = ?", (thread_id,))
        self._conn.execute("DELETE FROM checkpoint_v2_writes WHERE thread_id = ?", (thread_id,))
        self._conn.execute("DELETE FROM checkpoint_v2_threads WHERE thread_id = ?", (thread_id,))
        self._conn.execute("DELETE FROM checkpoint_v2_pins WHERE thread_id = ?", (thread_id,))
        self._conn.execute("DELETE FROM checkpoint_threads WHERE thread_id = ?", (thread_id,))


def has_checkpoint_thread(checkpointer: Any, thread_id: str) -> bool:
    if checkpointer is None:
        return False
    if hasattr(checkpointer, "has_thread"):
        return bool(checkpointer.has_thread(thread_id))
    if hasattr(checkpointer, "get_tuple"):
        try:
            return checkpointer.get_tuple({"configurable": {"thread_id": thread_id}}) is not None
        except Exception:
            return False
    return False


def checkpoint_backend_name(checkpointer: Any) -> str:
    backend = getattr(checkpointer, "backend", None)
    if backend:
        return str(backend)
    return "memory" if isinstance(checkpointer, MemorySaver) else type(checkpointer).__name__


def build_checkpointer(
    backend: CheckpointBackend,
    *,
    path: Path | str,
    retention: int | None = None,
) -> BaseCheckpointSaver[Any]:
    if backend == "sqlite":
        return SQLiteCheckpointSaver(path, retention=retention)
    return MemorySaver()


def _split_typed(value: tuple[str, bytes]) -> tuple[str, bytes]:
    value_type, payload = value
    return str(value_type), bytes(payload)


def _thread_and_namespace(config: RunnableConfig) -> tuple[str, str]:
    configurable = config["configurable"]
    return str(configurable["thread_id"]), str(configurable.get("checkpoint_ns", ""))


def _version_key(version: Any) -> str:
    if isinstance(version, bool):
        return f"bool:{version}"
    if isinstance(version, int):
        return f"int:{version}"
    if isinstance(version, float):
        return f"float:{version!r}"
    return f"str:{version}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
