"""SQLite authority for research sessions, mailbox, budgets and resumable events."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agents.reports import ReportResult, market_report
from ..infra.logging import redact_text
from ..infra.startup_errors import startup_error
from ..intel.evidence import load_evidence_documents, save_evidence_documents
from ..intel.models import EvidenceDocument
from .run_store import (
    FileRunStore,
    _apply_event,
    _derive_record_views,
    _empty_record,
    _summary_record,
)
from .streaming import jsonable


class SessionConflict(ValueError):
    """命令与当前持久化会话状态冲突,调用者应返回冲突而非静默重试。"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value: Any) -> str:
    return redact_text(json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True))


class CollaborationStore:
    """SQLite 会话、消息、事件和预算存储;写命令在事务中保证状态更新及幂等回执一致。"""

    def __init__(self, path: Path | str, *, legacy: FileRunStore | None = None) -> None:
        self.path = Path(path)
        self.legacy = legacy
        self._lock = threading.RLock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._validate_existing_database()
            self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS research_sessions (
                    session_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, market TEXT NOT NULL,
                    status TEXT NOT NULL, root_question TEXT NOT NULL DEFAULT '',
                    active_run_id TEXT, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS research_runs (
                    run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, status TEXT NOT NULL,
                    input_revision INTEGER NOT NULL DEFAULT 0, applied_revision INTEGER NOT NULL DEFAULT 0,
                    question TEXT NOT NULL, fingerprint TEXT NOT NULL, record TEXT NOT NULL,
                    initial_state TEXT, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS research_agents (
                    agent_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
                    slot INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'idle',
                    resident INTEGER NOT NULL DEFAULT 0, current_task TEXT,
                    UNIQUE(session_id,role,slot));
                CREATE INDEX IF NOT EXISTS research_run_created ON research_runs(created_at);
                CREATE TABLE IF NOT EXISTS research_messages (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL, run_id TEXT NOT NULL, sender TEXT NOT NULL,
                    receiver TEXT NOT NULL, kind TEXT NOT NULL, task_id TEXT,
                    input_revision INTEGER NOT NULL, payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued', dedupe_key TEXT NOT NULL UNIQUE);
                CREATE INDEX IF NOT EXISTS research_inbox ON research_messages(receiver,seq);
                CREATE TABLE IF NOT EXISTS research_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    run_id TEXT, event TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS research_event_session ON research_events(session_id,seq);
                CREATE TABLE IF NOT EXISTS research_requests (
                    scope TEXT NOT NULL, request_id TEXT NOT NULL, payload TEXT NOT NULL,
                    response TEXT NOT NULL, PRIMARY KEY(scope,request_id));
                CREATE TABLE IF NOT EXISTS research_budgets (run_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS research_results (
                    command_id TEXT PRIMARY KEY,agent_id TEXT NOT NULL,run_id TEXT NOT NULL,
                    input_revision INTEGER NOT NULL,result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS research_tasks (
                    task_id TEXT PRIMARY KEY,agent_id TEXT NOT NULL,run_id TEXT NOT NULL,
                    input_revision INTEGER NOT NULL,status TEXT NOT NULL,
                    handoff TEXT NOT NULL,result TEXT);
                CREATE TABLE IF NOT EXISTS research_round_reservations (
                    run_id TEXT NOT NULL,task_id TEXT NOT NULL,reservation_key TEXT NOT NULL,
                    PRIMARY KEY(run_id,task_id,reservation_key));
                CREATE TABLE IF NOT EXISTS research_progress (
                    run_id TEXT NOT NULL,task_id TEXT NOT NULL,rounds INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 0,scratch TEXT,
                    PRIMARY KEY(run_id,task_id));
            """)
            self._migrate_reports()
        except BaseException as exc:
            if hasattr(self, "_conn"):
                with suppress(Exception):
                    self._conn.close()
            if isinstance(exc, (sqlite3.Error, OSError)):
                raise startup_error("协作 SQLite", self.path, exc, sqlite=True) from exc
            raise

    def _migrate_reports(self) -> None:
        """Versioned, backed-up transaction; checkpoints share this database."""
        table = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_schema'"
        ).fetchone()
        version = (
            self._conn.execute("SELECT version FROM research_schema").fetchone() if table else None
        )
        if version and version[0] == 1:
            return
        if version:
            raise sqlite3.DatabaseError("unsupported research schema version")
        backup = self.path.with_name(self.path.name + ".before-reports-v1.bak")
        # SQLite backup includes committed WAL and the checkpoint tables.
        if not backup.exists():
            with sqlite3.connect(backup) as destination:
                self._conn.backup(destination)
        with self.transaction() as db:
            db.execute("""CREATE TABLE research_sessions_v1 (
                session_id TEXT PRIMARY KEY, symbol TEXT, market TEXT,
                status TEXT NOT NULL, root_question TEXT NOT NULL DEFAULT '',
                active_run_id TEXT, created_at TEXT NOT NULL, mode TEXT, target TEXT)""")
            db.execute("""INSERT INTO research_sessions_v1
                (session_id,symbol,market,status,root_question,active_run_id,created_at)
                SELECT session_id,symbol,market,status,root_question,active_run_id,created_at
                FROM research_sessions""")
            db.execute("DROP TABLE research_sessions")
            db.execute("ALTER TABLE research_sessions_v1 RENAME TO research_sessions")
            db.execute("ALTER TABLE research_runs ADD COLUMN report TEXT")
            db.execute("ALTER TABLE research_runs ADD COLUMN action TEXT")
            db.execute("CREATE TABLE research_schema (version INTEGER NOT NULL)")
            db.execute("INSERT INTO research_schema VALUES (1)")

    def _validate_existing_database(self) -> None:
        # Match checkpoint preflight: a writable open/close of a corrupt base
        # can delete the original WAL/SHM before an actionable error is raised.
        if not self.path.exists():
            return
        uri = self.path.resolve().as_uri() + "?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise sqlite3.DatabaseError("collaboration database integrity check failed")
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """串行开启 BEGIN IMMEDIATE,正常提交,异常或取消回滚;事务内操作共用同一连接。"""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _request(self, db: sqlite3.Connection, scope: str, key: str, payload: Any) -> Any:
        """查询同一作用域的幂等回执;相同 request_id 更换内容时抛出冲突,不能复用旧结果。"""
        row = db.execute(
            "SELECT payload,response FROM research_requests WHERE scope=? AND request_id=?",
            (scope, key),
        ).fetchone()
        if row is None:
            return None
        if row["payload"] != encode(payload):
            raise SessionConflict("request_id was already used with different content")
        return json.loads(row["response"])

    def _remember(
        self, db: sqlite3.Connection, scope: str, key: str, payload: Any, response: Any
    ) -> None:
        db.execute(
            "INSERT INTO research_requests VALUES(?,?,?,?)",
            (scope, key, encode(payload), encode(response)),
        )

    def _event(
        self, db: sqlite3.Connection, session_id: str, run_id: str | None, event: str, payload: Any
    ) -> None:
        value = {**jsonable(payload), "session_id": session_id, "run_id": run_id}
        db.execute(
            "INSERT INTO research_events(session_id,run_id,event,payload) VALUES(?,?,?,?)",
            (session_id, run_id, event, encode(value)),
        )

    def create_session(
        self,
        symbol: str | None,
        market: str | None,
        request_id: str,
        *,
        mode: str | None = None,
        target: dict[str, Any] | None = None,
        question: str = "",
    ) -> dict[str, Any]:
        with self.transaction() as db:
            return self._create_session(
                db, symbol, market, request_id, mode=mode, target=target, question=question
            )

    def _create_session(
        self,
        db: sqlite3.Connection,
        symbol: str | None,
        market: str | None,
        request_id: str,
        *,
        mode: str | None = None,
        target: dict[str, Any] | None = None,
        question: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"symbol": symbol.upper() if symbol else None, "market": market}
        if mode is not None:
            if mode not in {"report", "topic"} or target is None:
                raise SessionConflict("研究类型与范围不完整")
            if mode == "report" and target.get("kind") != "asset":
                raise SessionConflict("市场研报必须选择资产")
            if mode == "topic" and not question.strip():
                raise SessionConflict("请输入专题问题")
            payload.update(mode=mode, target=target, question=question.strip())
        previous = self._request(db, "create_session", request_id, payload)
        if previous:
            return dict(previous)
        identity = uuid.uuid4().hex
        db.execute(
            "INSERT INTO research_sessions(session_id,symbol,market,status,created_at,mode,target,root_question) VALUES(?,?,?,'open',?,?,?,?)",
            (
                identity,
                payload["symbol"],
                market,
                now_iso(),
                mode,
                encode(target) if target else None,
                question.strip(),
            ),
        )
        response = {"session_id": identity, **payload, "status": "open", "active_run_id": None}
        self._remember(db, "create_session", request_id, payload, response)
        return response

    def start_turn(
        self,
        session_id: str,
        question: str,
        request_id: str,
        fingerprint: str,
        action: str | None = None,
    ) -> dict[str, Any]:
        """事务内创建独立一轮并绑定配置指纹;每轮保留自己的 run_id、输入版本和预算。"""
        with self.transaction() as db:
            return self._start_turn(db, session_id, question, request_id, fingerprint, action)

    def _start_turn(
        self,
        db: sqlite3.Connection,
        session_id: str,
        question: str,
        request_id: str,
        fingerprint: str,
        action: str | None = None,
    ) -> dict[str, Any]:
        payload = {"question": question, **({"action": action} if action else {})}
        old = self._request(db, f"turn:{session_id}", request_id, payload)
        if old:
            return dict(old)
        session = db.execute(
            "SELECT * FROM research_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if session is None:
            raise KeyError("session not found")
        if session["status"] != "open" or session["active_run_id"]:
            raise SessionConflict("session is closed or already has an unfinished run")
        has_turn = db.execute(
            "SELECT 1 FROM research_runs WHERE session_id=? LIMIT 1", (session_id,)
        ).fetchone()
        if action == "initial" and has_turn:
            raise SessionConflict("首次研究已存在;请更新或追问")
        if action in {"update", "followup"} and not has_turn:
            raise SessionConflict("请先开始首次研究")
        if action == "followup" and not question.strip():
            raise SessionConflict("请输入追问问题")
        if session["mode"] and action in {"initial", "update"}:
            if question.strip():
                raise SessionConflict("更新保留原始目标;新的问题请使用追问")
            question = session["root_question"] if session["mode"] == "topic" else ""
        elif session["mode"] == "topic" and not question.strip():
            question = session["root_question"]
        run_id = uuid.uuid4().hex
        record = _empty_record(run_id)
        _apply_event(
            record,
            {
                "type": "created",
                "symbol": session["symbol"],
                "market": session["market"],
                "created_at": now_iso(),
            },
        )
        record.update(
            session_id=session_id,
            mode=session["mode"],
            target=json.loads(session["target"]) if session["target"] else None,
            title=session["root_question"]
            if session["mode"] == "topic"
            else f"{session['symbol']} 市场研报"
            if session["mode"] == "report"
            else "历史研究",
            action=action,
            input_revision=0,
            research_input={
                "research_question": question,
                "evidence_mode": True,
                "config_fingerprint": fingerprint,
            },
        )
        db.execute(
            "INSERT INTO research_runs(run_id,session_id,status,question,fingerprint,record,created_at,action) VALUES(?,?,'running',?,?,?,?,?)",
            (
                run_id,
                session_id,
                question,
                fingerprint,
                encode(record),
                record["created_at"],
                action,
            ),
        )
        db.execute(
            "UPDATE research_sessions SET active_run_id=?,root_question=CASE WHEN mode IS NULL AND root_question='' THEN ? ELSE root_question END WHERE session_id=?",
            (run_id, question, session_id),
        )
        response = {"session_id": session_id, "run_id": run_id, "status": "running"}
        self._remember(db, f"turn:{session_id}", request_id, payload, response)
        self._event(
            db,
            session_id,
            run_id,
            "run",
            {**response, "symbol": session["symbol"], "market": session["market"]},
        )
        return response

    def agent(self, session_id: str, role: str, slot: int = 0) -> dict[str, Any]:
        with self.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO research_agents(agent_id,session_id,role,slot) VALUES(?,?,?,?)",
                (uuid.uuid4().hex, session_id, role, slot),
            )
            row = db.execute(
                "SELECT * FROM research_agents WHERE session_id=? AND role=? AND slot=?",
                (session_id, role, slot),
            ).fetchone()
        return dict(row)

    def agent_status(
        self, agent_id: str, status: str, resident: bool, task: str | None = None
    ) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE research_agents SET status=?,resident=?,current_task=? WHERE agent_id=?",
                (status, int(resident), task, agent_id),
            )
            if task:
                db.execute(
                    "UPDATE research_tasks SET status=? WHERE task_id=? AND status NOT IN ('completed','failed','cancelled')",
                    (status, task),
                )
            row = db.execute(
                "SELECT * FROM research_agents WHERE agent_id=?", (agent_id,)
            ).fetchone()
            if row:
                run = db.execute(
                    "SELECT active_run_id FROM research_sessions WHERE session_id=?",
                    (row["session_id"],),
                ).fetchone()
                self._event(db, row["session_id"], run[0], "agent", dict(row))

    def _message(
        self,
        db: sqlite3.Connection,
        *,
        session_id: str,
        run_id: str,
        sender: str,
        receiver: str,
        kind: str,
        task_id: str | None,
        revision: int,
        payload: Any,
        key: str,
    ) -> dict[str, Any]:
        old = db.execute("SELECT * FROM research_messages WHERE dedupe_key=?", (key,)).fetchone()
        if old:
            message_identity = {
                "session_id": session_id,
                "run_id": run_id,
                "sender": sender,
                "receiver": receiver,
                "kind": kind,
                "task_id": task_id,
                "input_revision": revision,
                "payload": encode(payload),
            }
            if any(old[field] != value for field, value in message_identity.items()):
                raise SessionConflict("message identity conflicts with existing content")
            return self._message_view(old)
        run = db.execute(
            "SELECT session_id,input_revision FROM research_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        recipient = db.execute(
            "SELECT session_id FROM research_agents WHERE agent_id=?", (receiver,)
        ).fetchone()
        if (
            run is None
            or run["session_id"] != session_id
            or recipient is None
            or recipient[0] != session_id
        ):
            raise SessionConflict("message must reference a run and recipient in the same session")
        if revision < 0 or revision > run["input_revision"]:
            raise SessionConflict("message input revision does not exist")
        if kind not in {"assign_task", "inform", "task_result"}:
            raise SessionConflict("unsupported message kind")
        if kind == "assign_task":
            db.execute(
                "INSERT OR IGNORE INTO research_tasks(task_id,agent_id,run_id,input_revision,status,handoff) VALUES(?,?,?,?,'queued',?)",
                (task_id, receiver, run_id, revision, encode(payload)),
            )
        elif kind == "task_result":
            result = payload.get("result", {})
            db.execute(
                "UPDATE research_tasks SET status=?,result=? WHERE task_id=?",
                ("completed" if result.get("ok") else "failed", encode(result), task_id),
            )
        identity = uuid.uuid4().hex
        db.execute(
            "INSERT INTO research_messages(message_id,session_id,run_id,sender,receiver,kind,task_id,input_revision,payload,dedupe_key) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                identity,
                session_id,
                run_id,
                sender,
                receiver,
                kind,
                task_id,
                revision,
                encode(payload),
                key,
            ),
        )
        row = db.execute(
            "SELECT * FROM research_messages WHERE message_id=?", (identity,)
        ).fetchone()
        return self._message_view(row)

    @staticmethod
    def _message_view(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["content"] = result["payload"].get("content", "")
        return result

    def deliver(self, **kwargs: Any) -> dict[str, Any]:
        with self.transaction() as db:
            return self._message(db, **kwargs)

    def inform(self, session_id: str, run_id: str, content: str, request_id: str) -> dict[str, Any]:
        """把用户补充、递增的输入版本和请求回执一起落盘;只允许当前运行中的本会话任务。"""
        supervisor = self.agent(session_id, "research_supervisor")["agent_id"]
        with self.transaction() as db:
            payload = {"run_id": run_id, "content": content}
            previous = self._request(db, f"inform:{session_id}", request_id, payload)
            if previous:
                return dict(previous)
            run = db.execute(
                "SELECT * FROM research_runs WHERE run_id=? AND session_id=?", (run_id, session_id)
            ).fetchone()
            if run is None:
                raise KeyError("run not found")
            if run["status"] != "running":
                raise SessionConflict("run is not running; use continue research for a new turn")
            revision = run["input_revision"] + 1
            db.execute(
                "UPDATE research_runs SET input_revision=? WHERE run_id=?", (revision, run_id)
            )
            message = self._message(
                db,
                session_id=session_id,
                run_id=run_id,
                sender="user",
                receiver=supervisor,
                kind="inform",
                task_id=None,
                revision=revision,
                payload={"content": content},
                key=f"user:{session_id}:{request_id}",
            )
            self._remember(db, f"inform:{session_id}", request_id, payload, message)
            self._event(db, session_id, run_id, "message", message)
        return message

    def inbox(
        self,
        agent_id: str,
        after: int = 0,
        *,
        run_id: str | None = None,
        revision: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM research_messages WHERE receiver=? AND seq>?"
        args: list[Any] = [agent_id, after]
        if run_id is not None:
            query += " AND run_id=?"
            args.append(run_id)
        if revision is not None:
            query += " AND input_revision<=?"
            args.append(revision)
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY seq", args).fetchall()
        return [self._message_view(row) for row in rows]

    def acknowledge(self, agent_id: str, cursor: int) -> None:
        with self.transaction() as db:
            rows = db.execute(
                "SELECT * FROM research_messages WHERE receiver=? AND seq<=? AND status='queued'",
                (agent_id, cursor),
            ).fetchall()
            db.execute(
                "UPDATE research_messages SET status='incorporated' WHERE receiver=? AND seq<=? AND status='queued'",
                (agent_id, cursor),
            )
            for row in rows:
                if row["sender"] == "user":
                    self._event(
                        db,
                        row["session_id"],
                        row["run_id"],
                        "message",
                        {**self._message_view(row), "status": "incorporated"},
                    )

    def updates(self, run_id: str, revision: int | None = None) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT input_revision,payload FROM research_messages WHERE run_id=? AND sender='user' ORDER BY seq",
                (run_id,),
            ).fetchall()
        return [
            json.loads(r["payload"])["content"]
            for r in rows
            if revision is None or r["input_revision"] <= revision
        ]

    def run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM research_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError("run not found")
        result = dict(row)
        result["record"] = json.loads(result["record"])
        result["initial_state"] = (
            json.loads(result["initial_state"]) if result["initial_state"] else None
        )
        return result

    def save_initial(self, run_id: str, state: dict[str, Any]) -> None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT record FROM research_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            record = json.loads(row[0])
            record["analysis_at"] = state.get("analysis_at")
            db.execute(
                "UPDATE research_runs SET initial_state=?,record=? WHERE run_id=?",
                (encode(state), encode(record), run_id),
            )

    def apply_revision(self, run_id: str, revision: int) -> None:
        """在补充已进入上下文后激活对应 revision,清空旧版派生视图但保留审计事件。"""
        with self.transaction() as db:
            row = db.execute("SELECT * FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError("run not found")
            pending = db.execute(
                "SELECT 1 FROM research_messages WHERE run_id=? AND sender='user' AND input_revision<=? AND status!='incorporated' LIMIT 1",
                (run_id, revision),
            ).fetchone()
            if (
                row["status"] != "running"
                or revision > row["input_revision"]
                or revision < row["applied_revision"]
                or pending
            ):
                raise SessionConflict(
                    "input revision is invalid or user messages are not incorporated"
                )
            record = json.loads(row["record"])
            if revision != row["applied_revision"]:
                # Audit events retain all versions; derived views only contain the current one.
                for key in ("node_outputs", "runtime_events", "tool_events", "planner_steps"):
                    record[key] = []
                for key in (
                    "news_research",
                    "research_summary",
                    "risk_gate_result",
                    "debate_grounding",
                    "planner_decision",
                    "candidate_advice",
                    "final_advice",
                    "report",
                    "preflight_report",
                    "data_quality_report",
                ):
                    record[key] = None
                record["evidence_factors"] = []
                _derive_record_views(record)
            record["input_revision"] = revision
            db.execute(
                "UPDATE research_runs SET applied_revision=?,record=? WHERE run_id=?",
                (revision, encode(record), run_id),
            )

    def append_event(self, run_id: str, event_type: str, payload: Any) -> None:
        with self.transaction() as db:
            row = db.execute("SELECT * FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                if self.legacy and self.legacy.get(run_id) is not None:
                    self.legacy.append_event(run_id, event_type, payload)
                    return
                raise KeyError("run not found")
            record = json.loads(row["record"])
            event = {
                "type": "event",
                "event_type": event_type,
                "payload": jsonable(payload),
                "created_at": now_iso(),
            }
            if isinstance(payload, dict) and payload.get("superseded"):
                record["events"].append(
                    {
                        "event_type": event_type,
                        "payload": jsonable(payload),
                        "created_at": event["created_at"],
                    }
                )
            else:
                _apply_event(record, event)
            _derive_record_views(record)
            db.execute("UPDATE research_runs SET record=? WHERE run_id=?", (encode(record), run_id))
            if event_type == "research_budget":
                db.execute(
                    "INSERT INTO research_budgets VALUES(?,?) ON CONFLICT(run_id) DO UPDATE SET payload=excluded.payload",
                    (run_id, encode(payload)),
                )
                self._event(db, row["session_id"], run_id, "budget", payload)
            elif event_type not in {"final", "checkpoint", "research_input"}:
                self._event(db, row["session_id"], run_id, event_type, payload)

    def finalize(
        self,
        run_id: str,
        revision: int,
        advice: Any = None,
        *,
        report: Any = None,
        incomplete: bool = False,
    ) -> bool:
        """比较运行状态与 revision 后原子发布最终建议,成功返回 True,版本过期返回 False。

        正常完成要求补充均已处理;incomplete 路径清除过时研究视图并标记未处理消息,
        不能把旧结果当作新输入的结论。
        """
        with self.transaction() as db:
            row = db.execute("SELECT * FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
            if row["status"] != "running" or row["input_revision"] != revision:
                return False
            if not incomplete:
                pending = db.execute(
                    "SELECT 1 FROM research_messages WHERE run_id=? AND sender='user' AND status!='incorporated' LIMIT 1",
                    (run_id,),
                ).fetchone()
                if row["applied_revision"] != revision or pending:
                    return False
            record = json.loads(row["record"])
            if incomplete:
                for key in (
                    "preflight_report",
                    "data_quality_report",
                    "debate_grounding",
                    "risk_gate_result",
                    "news_research",
                ):
                    record[key] = None
                record["evidence_factors"] = []
                record["runtime_events"] = []
                record["node_outputs"] = []
                record["planner_decision"] = None
                record["planner_steps"] = []
            record.update(
                status="completed",
                final_advice=jsonable(advice),
                completed_at=now_iso(),
                input_revision=revision,
                checkpoint_status="completed",
                resumable=False,
            )
            if report is None and advice is not None:
                report = market_report(record, advice)
            if report is None:
                raise SessionConflict("没有可发布的报告")
            published = ReportResult.model_validate(jsonable(report))
            if record.get("mode") and (
                published.input_revision != revision
                or jsonable(published.target) != record.get("target")
            ):
                raise SessionConflict("报告范围或输入版本与本轮不一致")
            published.input_revision = revision
            if record.get("mode") == "topic" and (
                published.kind != "topic" or advice is not None or published.advice is not None
            ):
                raise SessionConflict("专题结果不能包含交易建议")
            record["report"] = jsonable(published)
            _derive_record_views(record)
            db.execute(
                "UPDATE research_runs SET status='completed',record=?,report=? WHERE run_id=?",
                (encode(record), encode(published), run_id),
            )
            db.execute(
                "UPDATE research_sessions SET active_run_id=NULL WHERE session_id=?",
                (row["session_id"],),
            )
            if incomplete:
                db.execute(
                    "UPDATE research_messages SET status='not_processed' WHERE run_id=? AND sender='user' AND input_revision>?",
                    (run_id, row["applied_revision"]),
                )
            self._event(
                db,
                row["session_id"],
                run_id,
                "final",
                {
                    "advice": advice,
                    "report": record["report"],
                    "input_revision": revision,
                    **{
                        k: record.get(k)
                        for k in (
                            "news_research",
                            "research_budget",
                            "run_health",
                            "evidence_summary",
                            "evidence_factors",
                        )
                    },
                },
            )
        return True

    def status(self, run_id: str, status: str, reason: str = "") -> bool:
        with self.transaction() as db:
            row = db.execute("SELECT * FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError("run not found")
            allowed = {
                "running": {"interrupted"},
                "stopping": {"running"},
                "interrupted": {"running", "stopping"},
                "cancelled": {"running", "interrupted", "stopping"},
                "failed": {"running", "interrupted", "stopping"},
            }
            if row["status"] not in allowed.get(status, set()):
                return False
            record = json.loads(row["record"])
            record.update(
                status=status,
                error=reason,
                resumable=status == "interrupted",
                checkpoint_status="resumable" if status == "interrupted" else status,
            )
            db.execute(
                "UPDATE research_runs SET status=?,record=? WHERE run_id=?",
                (status, encode(record), run_id),
            )
            if status in {"cancelled", "failed"}:
                db.execute(
                    "UPDATE research_sessions SET active_run_id=NULL WHERE session_id=?",
                    (row["session_id"],),
                )
                db.execute(
                    "UPDATE research_messages SET status='not_processed' WHERE run_id=? AND status='queued'",
                    (run_id,),
                )
            if status in {"cancelled", "interrupted", "failed"}:
                db.execute(
                    "UPDATE research_tasks SET status=? WHERE run_id=? AND status IN ('queued','running','waiting','interrupted')",
                    (status, run_id),
                )
            self._event(
                db, row["session_id"], run_id, status, {"status": status, "message": reason}
            )
            return True

    def recover(self) -> None:
        """启动时将残留活动轮标为 interrupted 并取消实例驻留标记,等待用户手动恢复。"""
        with self._lock:
            active = [
                r[0]
                for r in self._conn.execute(
                    "SELECT run_id FROM research_runs WHERE status IN ('running','stopping')"
                )
            ]
        for run_id in active:
            self.status(run_id, "interrupted", "后端已退出,等待手动恢复")
        with self.transaction() as db:
            db.execute(
                "UPDATE research_agents SET resident=0,status=CASE WHEN status='running' THEN 'interrupted' ELSE status END"
            )

    def session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM research_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError("session not found")
            agents = [
                dict(r)
                for r in self._conn.execute(
                    "SELECT * FROM research_agents WHERE session_id=? ORDER BY role,slot",
                    (session_id,),
                )
            ]
            rows = self._conn.execute(
                "SELECT * FROM research_runs WHERE session_id=? ORDER BY created_at", (session_id,)
            ).fetchall()
            messages = self._conn.execute(
                "SELECT * FROM research_messages WHERE session_id=? AND sender='user' ORDER BY seq",
                (session_id,),
            ).fetchall()
        runs = []
        for r in rows:
            record = json.loads(r["record"])
            runs.append(
                {
                    "run_id": r["run_id"],
                    "status": r["status"],
                    "input_revision": r["input_revision"],
                    "question": r["question"],
                    "created_at": r["created_at"],
                    "analysis_at": record.get("analysis_at"),
                    "final_advice": record.get("final_advice"),
                    "report": record.get("report"),
                    "action": r["action"],
                    "budget": record.get("research_budget"),
                }
            )
        with self._lock:
            tasks = [
                dict(r)
                for r in self._conn.execute(
                    "SELECT task_id,agent_id,run_id,input_revision,status FROM research_tasks WHERE run_id IN (SELECT run_id FROM research_runs WHERE session_id=?) ORDER BY rowid",
                    (session_id,),
                )
            ]
        return {
            **dict(row),
            "target": json.loads(row["target"]) if row["target"] else None,
            "title": row["root_question"]
            if row["mode"] == "topic"
            else f"{row['symbol']} 市场研报"
            if row["mode"] == "report"
            else "历史研究",
            "agents": agents,
            "runs": runs,
            "tasks": tasks,
            "messages": [self._message_view(r) for r in messages],
        }

    def sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            ids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT session_id FROM research_sessions ORDER BY created_at DESC LIMIT 100"
                )
            ]
        return [self.session(i) for i in ids]

    def close_session(self, session_id: str, request_id: str) -> dict[str, Any]:
        with self.transaction() as db:
            previous = self._request(db, f"close:{session_id}", request_id, {})
            if previous:
                return dict(previous)
            row = db.execute(
                "SELECT active_run_id FROM research_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError("session not found")
            if row[0]:
                raise SessionConflict("stop the unfinished run before closing the session")
            db.execute(
                "UPDATE research_sessions SET status='closed' WHERE session_id=?", (session_id,)
            )
            db.execute(
                "UPDATE research_agents SET status='closed',resident=0 WHERE session_id=?",
                (session_id,),
            )
            response = {"session_id": session_id, "status": "closed"}
            self._remember(db, f"close:{session_id}", request_id, {}, response)
            return response

    def events(self, session_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM research_events WHERE session_id=? AND seq>? ORDER BY seq LIMIT 200",
                (session_id, after),
            ).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def get(self, run_id: str) -> dict[str, Any] | None:
        try:
            run = self.run(run_id)
        except KeyError:
            return self.legacy.get(run_id) if self.legacy else None
        record = run["record"]
        record.update(
            status=run["status"], session_id=run["session_id"], input_revision=run["input_revision"]
        )
        with self._lock:
            messages = self._conn.execute(
                "SELECT * FROM research_messages WHERE run_id=? ORDER BY seq", (run_id,)
            ).fetchall()
            tasks = self._conn.execute(
                "SELECT task_id,agent_id,run_id,input_revision,status FROM research_tasks WHERE run_id=? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        record["collaboration_messages"] = [self._message_view(row) for row in messages]
        record["collaboration_tasks"] = [dict(row) for row in tasks]
        return dict(record)

    def list_all(self) -> list[dict[str, Any]]:
        records = {r["run_id"]: r for r in self.legacy.list_all()} if self.legacy else {}
        with self._lock:
            ids = [
                r[0]
                for r in self._conn.execute("SELECT run_id FROM research_runs ORDER BY created_at")
            ]
        for identity in ids:
            record = self.get(identity)
            if record is not None:
                records[identity] = record
        return sorted(records.values(), key=lambda r: str(r.get("created_at") or ""))

    def list_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return list(reversed(self.list_all()))[: max(0, limit)]

    def list_recent_summaries(self, *, limit: int = 20) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with self._lock:
            # The creation index also orders equal timestamps by rowid. Only
            # decode the requested records; never hydrate mailboxes or tasks.
            rows = self._conn.execute(
                "SELECT record,status FROM research_runs ORDER BY created_at DESC,rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
            summaries = [
                _summary_record({**json.loads(row["record"]), "status": row["status"]})
                for row in rows
            ]
            if self.legacy is not None:
                offset = 0
                legacy_summaries: list[dict[str, Any]] = []
                page_size = min(limit, 500)
                while len(legacy_summaries) < limit:
                    page = self.legacy.list_recent_summaries(
                        limit=page_size, offset=offset, newest_first_on_ties=True
                    )
                    if not page:
                        break
                    # SQLite remains authoritative for IDs also present in old
                    # JSONL, even when that SQLite row is outside the top limit.
                    placeholders = ",".join("?" for _ in page)
                    owned = {
                        r[0]
                        for r in self._conn.execute(
                            f"SELECT run_id FROM research_runs WHERE run_id IN ({placeholders})",
                            [r["run_id"] for r in page],
                        )
                    }
                    legacy_summaries.extend(r for r in page if r["run_id"] not in owned)
                    offset += len(page)
                    if len(page) < page_size:
                        break
                summaries.extend(legacy_summaries[:limit])
        return sorted(summaries, key=lambda r: str(r.get("created_at") or ""), reverse=True)[:limit]

    def complete(self, run_id: str, final_advice: Any) -> None:
        if self.legacy and self.legacy.get(run_id) is not None:
            self.legacy.complete(run_id, final_advice)
            return
        raise SessionConflict(
            "collaborative results require the revision-checked finalize operation"
        )

    def fail(self, run_id: str, message: str) -> None:
        if self.legacy and self.legacy.get(run_id) is not None:
            self.legacy.fail(run_id, message)
            return
        self.status(run_id, "failed", message)

    def save_evidence(self, run_id: str, documents: list[EvidenceDocument]) -> None:
        try:
            self.run(run_id)
        except KeyError:
            if self.legacy and self.legacy.get(run_id) is not None:
                self.legacy.save_evidence(run_id, documents)
                return
            raise
        with self._lock:
            old = {d.evidence_id: d for d in load_evidence_documents(self.path.parent, run_id)}
            old.update({d.evidence_id: d for d in documents})
            save_evidence_documents(self.path.parent, run_id, list(old.values()))

    def load_evidence(self, run_id: str) -> list[EvidenceDocument]:
        try:
            self.run(run_id)
        except KeyError:
            return self.legacy.load_evidence(run_id) if self.legacy else []
        with self._lock:
            return load_evidence_documents(self.path.parent, run_id)

    def budget(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM research_budgets WHERE run_id=?", (run_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def cached_result(self, command_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT result FROM research_results WHERE command_id=?", (command_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def result(
        self, command_id: str, agent_id: str, run_id: str, revision: int, result: Any
    ) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO research_results VALUES(?,?,?,?,?)",
                (command_id, agent_id, run_id, revision, encode(result)),
            )

    def progress(self, run_id: str, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM research_progress WHERE run_id=? AND task_id=?", (run_id, task_id)
            ).fetchone()
        return (
            {**dict(row), "scratch": json.loads(row["scratch"]) if row["scratch"] else None}
            if row
            else None
        )

    def save_progress(self, run_id: str, task_id: str, revision: int, scratch: Any) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO research_progress(run_id,task_id,revision,scratch) VALUES(?,?,?,?) ON CONFLICT(run_id,task_id) DO UPDATE SET revision=excluded.revision,scratch=excluded.scratch",
                (run_id, task_id, revision, encode(scratch)),
            )

    def reserve_round(
        self,
        run_id: str,
        task_id: str,
        limit: int,
        task_limit: int,
        *,
        reservation_key: str | None = None,
    ) -> bool:
        with self.transaction() as db:
            if (
                reservation_key is not None
                and db.execute(
                    "SELECT 1 FROM research_round_reservations WHERE run_id=? AND task_id=? AND reservation_key=?",
                    (run_id, task_id, reservation_key),
                ).fetchone()
            ):
                return True
            row = db.execute(
                "SELECT rounds FROM research_progress WHERE run_id=? AND task_id=?",
                (run_id, task_id),
            ).fetchone()
            count = db.execute(
                "SELECT count(*) FROM research_progress WHERE run_id=? AND rounds>0", (run_id,)
            ).fetchone()[0]
            if ((row is None or row[0] == 0) and count >= task_limit) or (row and row[0] >= limit):
                return False
            db.execute(
                "INSERT INTO research_progress(run_id,task_id,rounds) VALUES(?,?,1) ON CONFLICT(run_id,task_id) DO UPDATE SET rounds=rounds+1",
                (run_id, task_id),
            )
            if reservation_key is not None:
                db.execute(
                    "INSERT INTO research_round_reservations VALUES(?,?,?)",
                    (run_id, task_id, reservation_key),
                )
            return True
