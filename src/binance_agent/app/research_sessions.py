"""Background research execution and HTTP session commands."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal, cast

from fastapi import FastAPI, HTTPException, Query, Request
from langgraph.types import Overwrite
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sse_starlette.sse import EventSourceResponse

from ..agents.budget import EndpointPool, RunBudget, run_budget_scope
from ..agents.collaboration import (
    AgentEngine,
    CollaborationScope,
    InputSuperseded,
    collaboration_scope,
)
from ..agents.reports import AssetTarget, ResearchTarget, incomplete_report
from ..agents.state import StrategyAdvice
from ..infra.settings import Settings
from ..intel.models import EvidenceDocument
from .collaboration_store import CollaborationStore, SessionConflict
from .streaming import jsonable

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _closing_stream(stream: Any):
    try:
        yield stream
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, max_length=120)


class NewSession(Command):
    symbol: str | None = Field(default=None, min_length=2, max_length=40, pattern=r"^[A-Za-z0-9]+$")
    market: Literal["api", "fapi"] | None = None
    mode: Literal["report", "topic"] | None = None
    target: ResearchTarget | None = None
    question: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def scope_valid(self) -> NewSession:
        if self.mode is None:
            if not self.symbol or self.target is not None or self.question:
                raise ValueError("旧版请求需要交易对;新研究请提供 mode 和 target")
            self.market = self.market or "fapi"
        else:
            if self.symbol is not None or self.market is not None or self.target is None:
                raise ValueError("新研究请仅使用 target 指定范围")
            if self.mode == "report" and (
                not isinstance(self.target, AssetTarget) or self.question.strip()
            ):
                raise ValueError("市场研报需要资产范围,无需专题问题")
            if self.mode == "topic" and not self.question.strip():
                raise ValueError("请输入专题问题")
        return self


class NewTurn(Command):
    question: str = Field(default="", max_length=2000)
    action: Literal["initial", "update", "followup"] | None = None


class Supplement(Command):
    run_id: str
    content: str = Field(min_length=1, max_length=2000)


class RunManager:
    """持有后台研究任务和执行槽,将持久化命令、预算、图执行与 SSE 订阅分离。"""

    def __init__(
        self,
        app: FastAPI,
        settings: Settings,
        store: CollaborationStore,
        engine: AgentEngine,
        *,
        prepare: Callable[..., Awaitable[dict[str, Any]]],
        stream_graph: Callable[..., AsyncIterator[dict[str, Any]]],
        fingerprint: Callable[[], str],
    ) -> None:
        self.app = app
        self.settings = settings
        self.store = store
        self.engine = engine
        self.prepare = prepare
        self.stream_graph = stream_graph
        self.fingerprint = fingerprint
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.stopping: set[str] = set()
        self.closing = False
        self._wake: dict[str, asyncio.Event] = {}
        self._command_lock = asyncio.Lock()
        self._maintenance: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._pending_shutdown: set[asyncio.Task[None]] = set()
        self.pool = EndpointPool(
            per_endpoint_concurrency={e.name: e.max_concurrency for e in settings.llm.endpoints}
        )

    def start(self) -> None:
        """把上次退出留下的活动任务标记为中断,固定待恢复 checkpoint 并启动空闲实例回收。"""
        self.store.recover()
        if hasattr(self.engine.checkpointer, "pin_thread"):
            with self.store._lock:
                identities = [
                    r[0]
                    for r in self.store._conn.execute(
                        "SELECT agent_id FROM research_agents WHERE session_id IN (SELECT session_id FROM research_sessions WHERE status='open') UNION SELECT run_id FROM research_runs WHERE status='interrupted'"
                    )
                ]
            for identity in identities:
                self.engine.checkpointer.pin_thread(identity)
        self._maintenance = asyncio.create_task(self._maintain())

    async def _maintain(self) -> None:
        while True:
            await asyncio.sleep(min(10, self.engine.idle_s))
            self.engine.evict()

    def notify(self, session_id: str) -> None:
        signal = self._wake.get(session_id)
        if signal:
            signal.set()

    def _check_enabled(self) -> None:
        if getattr(self.app.state, "graph", False) is None:
            raise SessionConflict("请先在设置中连接模型")
        if self.closing:
            raise SessionConflict("backend is shutting down")
        if not self.settings.research.enabled:
            raise SessionConflict("enable evidence research before starting a collaborative turn")
        if self.settings.research.max_requests is None or self.settings.research.deadline_s is None:
            raise SessionConflict("research requires explicit max_requests and deadline_s")

    async def new_session_turn(
        self, symbol: str, market: str, question: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """取得执行槽后在同一事务创建快捷会话及首轮;创建失败回滚并释放槽位。"""
        async with self._command_lock:
            self._check_enabled()
            reservation = await self.app.state.analysis_limiter.try_acquire()
            if reservation is None:
                raise HTTPException(429, "too many concurrent analyses")
            task: asyncio.Task[None] | None = None
            launched = False
            try:
                with self.store.transaction() as db:
                    session = self.store._create_session(db, symbol, market, uuid.uuid4().hex)
                    response = self.store._start_turn(
                        db, session["session_id"], question, uuid.uuid4().hex, self.fingerprint()
                    )
                    coroutine = self._drive(response["run_id"], reservation, resume=False)
                    try:
                        task = asyncio.create_task(coroutine)
                    except BaseException:
                        coroutine.close()
                        raise
                self.tasks[response["run_id"]] = task
                reservation.detach()
                launched = True
                session["active_run_id"] = response["run_id"]
                self.notify(session["session_id"])
                await asyncio.sleep(0)
                return session, response
            except BaseException:
                if not launched:
                    if task is not None:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                    await reservation.release()
                raise

    async def turn(
        self, session_id: str, question: str, request_id: str, action: str | None = None
    ) -> dict[str, Any]:
        async with self._command_lock:
            self._check_enabled()
            # Check idempotency before admission; retries remain successful after completion.
            with self.store._lock:
                row = self.store._conn.execute(
                    "SELECT payload,response FROM research_requests WHERE scope=? AND request_id=?",
                    (f"turn:{session_id}", request_id),
                ).fetchone()
            if row:
                if json.loads(row["payload"]) != {
                    "question": question,
                    **({"action": action} if action else {}),
                }:
                    raise SessionConflict("request_id was already used with different content")
                return dict(json.loads(row["response"]))
            reservation = await self.app.state.analysis_limiter.try_acquire()
            if reservation is None:
                raise HTTPException(429, "too many concurrent analyses")
            launched = False
            task: asyncio.Task[None] | None = None
            try:
                with self.store.transaction() as db:
                    response = self.store._start_turn(
                        db, session_id, question, request_id, self.fingerprint(), action
                    )
                    run_id = response["run_id"]
                    coroutine = self._drive(run_id, reservation, resume=False)
                    try:
                        task = asyncio.create_task(coroutine)
                    except BaseException:
                        coroutine.close()
                        raise
                self.tasks[run_id] = task
                reservation.detach()
                launched = True  # Only the background task now owns the reservation.
                self.notify(session_id)
                await asyncio.sleep(
                    0
                )  # Establish cancellation cleanup before accepting stop commands.
                return response
            except BaseException:
                if not launched:
                    if task is not None:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                    await reservation.release()
                raise

    async def control(self, run_id: str, action: str, request_id: str) -> dict[str, Any]:
        """按 request_id 幂等处理停止/恢复;恢复须指纹一致并重新准入,停止通过取消后台任务落盘。"""
        async with self._command_lock:
            payload = {"action": action}
            with self.store.transaction() as db:
                old = self.store._request(db, f"control:{run_id}", request_id, payload)
                if old:
                    return dict(old)
            run = self.store.run(run_id)
            if action == "resume":
                self._check_enabled()
                if run["status"] != "interrupted" or run_id in self.tasks:
                    raise SessionConflict("only an interrupted run can be resumed")
                if run["fingerprint"] != self.fingerprint():
                    raise SessionConflict("研究配置或版本已改变,请重新研究;原记录已保留")
                reservation = await self.app.state.analysis_limiter.try_acquire()
                if reservation is None:
                    raise HTTPException(429, "too many concurrent analyses")
                try:
                    self.store.status(run_id, "running")
                    self.tasks[run_id] = asyncio.create_task(
                        self._drive(run_id, reservation, resume=True)
                    )
                    reservation.detach()
                except BaseException:
                    await reservation.release()
                    raise
            else:
                if run["status"] not in {"running", "interrupted", "stopping"}:
                    raise SessionConflict("run has already ended")
                self.stopping.add(run_id)
                task = self.tasks.get(run_id)
                if task:
                    self.store.status(run_id, "stopping")
                    task.cancel()
                else:
                    self.store.status(run_id, "cancelled", "用户停止本轮")
                    self.stopping.discard(run_id)
            response = {
                "run_id": run_id,
                "session_id": run["session_id"],
                "status": "running"
                if action == "resume"
                else "stopping"
                if run_id in self.tasks
                else "cancelled",
            }
            with self.store.transaction() as db:
                self.store._remember(db, f"control:{run_id}", request_id, payload, response)
            self.notify(run["session_id"])
            if action == "resume":
                await asyncio.sleep(0)
            return response

    def _budget(self, run_id: str) -> RunBudget:
        """从已存快照恢复累计请求与期限,或为新轮创建预算;每次变更同步记录,失败阻止继续消费。"""
        concurrency = {e.name: e.max_concurrency for e in self.settings.llm.endpoints}
        if concurrency != self.pool.per_endpoint_concurrency:
            self.pool = EndpointPool(per_endpoint_concurrency=concurrency)

        def callback(snapshot: dict[str, Any]) -> None:
            self.store.append_event(run_id, "research_budget", snapshot)

        previous = self.store.budget(run_id)
        if previous:
            budget = RunBudget.from_snapshot(previous, endpoint_pool=self.pool, on_change=callback)
        else:
            budget = RunBudget(
                max_requests=self.settings.research.max_requests,
                deadline_s=self.settings.research.deadline_s,
                endpoint_pool=self.pool,
                on_change=callback,
            )
        callback(budget.snapshot())
        return budget

    @staticmethod
    def _conservative(symbol: str, reason: str) -> StrategyAdvice:
        return StrategyAdvice(
            symbol=symbol,
            direction="neutral",
            decision_type="no_trade",
            confidence=0,
            rationale=reason,
            no_trade_reason="collaboration_incomplete",
            evidence_gaps=[reason],
        )

    def _history(self, session_id: str, run_id: str) -> list[dict[str, Any]]:
        """只取最近四轮已完成建议作历史上下文,并标明不是本轮证据;不重新执行历史分析。"""
        runs = self.store.session(session_id)["runs"]
        result = []
        for run in runs:
            if run["run_id"] == run_id or not (run.get("final_advice") or run.get("report")):
                continue
            record = self.store.get(run["run_id"]) or {}
            result.append(
                {
                    "origin_run_id": run["run_id"],
                    "analysis_at": (record.get("news_research") or {}).get(
                        "analysis_at", run["created_at"]
                    ),
                    "question": run["question"],
                    "historical_advice": run.get("final_advice"),
                    "historical_report": run.get("report"),
                    "evidence_refs": (record.get("news_research") or {}).get("documents", []),
                    "status": "historical_not_current_evidence",
                }
            )
        return result[-4:]

    async def _drive(self, run_id: str, reservation: Any, *, resume: bool) -> None:
        """在后台运行一轮研究,保留预算并按输入 revision 处理补充和恢复。

        旧版本节点作为审计事件保存,但只有当前版本可提交最终结果;用户停止记 cancelled,
        后端退出/异常记 interrupted。所有退出路径归还执行槽并通知订阅者。
        """
        session_id: str | None = None
        demo_token: Any = None
        try:
            run = self.store.run(run_id)
            session_id = run["session_id"]
            topic = run["record"].get("mode") == "topic"
            graph = self.app.state.graph
            if hasattr(self.engine.checkpointer, "pin_thread"):
                self.engine.checkpointer.pin_thread(run_id)
            self.app.state.active_research_runs.add(run_id)
            from ..demo import DEMO_REQUEST_ID

            if self.settings.profile == "demo":
                demo_token = DEMO_REQUEST_ID.set(run_id)
            budget = self._budget(run_id)
            with run_budget_scope(budget):
                initial = run["initial_state"]
                if initial is None:
                    session = self.store.session(session_id)
                    initial = await self.prepare(
                        run_id, session["symbol"], session["market"], run["question"]
                    )
                    self.store.save_initial(run_id, initial)
                    resume = False
                if resume:
                    snapshot = await graph.aget_state({"configurable": {"thread_id": run_id}})
                    # A message may have committed while the old graph was interrupted.
                    resume = (
                        bool(snapshot.next)
                        and snapshot.values.get("input_revision", 0) == run["input_revision"]
                    )
                while True:
                    run = self.store.run(run_id)
                    revision = run["input_revision"]
                    if budget.remaining_s() == 0 or (
                        budget.max_requests is not None
                        and budget.requests_reserved >= budget.max_requests
                    ):
                        if self.store.finalize(
                            run_id,
                            revision,
                            None
                            if topic
                            else self._conservative(
                                initial["symbol"],
                                "本轮预算或期限已耗尽,最新要求仍有未完成事项;可发起新一轮研究",
                            ),
                            report=incomplete_report(
                                {**run["record"], "input_revision": revision},
                                "本轮预算或期限已耗尽,最新要求仍有未完成事项;可更新话题开始新一轮",
                            )
                            if topic
                            else None,
                            incomplete=True,
                        ):
                            break
                        continue
                    updates = self.store.updates(run_id, revision)
                    await self.engine.incorporate(session_id, run_id, revision)
                    combined_question = run["question"] + (
                        "\n\n用户补充(按时间顺序):\n" + "\n".join(updates) if updates else ""
                    )
                    state = {
                        **initial,
                        "research_question": combined_question,
                        "input_revision": revision,
                        "evidence_documents": [
                            d.model_dump(mode="json") for d in self.store.load_evidence(run_id)
                        ],
                    }
                    if isinstance(state.get("snapshot_freshness_now"), str):
                        state["snapshot_freshness_now"] = datetime.fromisoformat(
                            state["snapshot_freshness_now"]
                        )
                    if revision:
                        canonical = self.store.get(run_id) or {}
                        state.update(
                            market_snapshot=canonical.get("market_snapshot")
                            or initial["market_snapshot"],
                            intel_snapshot=canonical.get("intel_snapshot")
                            or initial["intel_snapshot"],
                            preflight_refetch_enabled=False,
                        )
                    if not resume:
                        for key in (
                            "analyst_reports",
                            "debate_turns",
                            "runtime_events",
                            "tool_events",
                            "planner_steps",
                        ):
                            state[key] = Overwrite([])
                        state.update(
                            news_research={},
                            research_summary=None,
                            candidate_advice=None,
                            final_advice=None,
                            final_report=None,
                            debate_round=0,
                            debate_stop_reason=None,
                            debate_grounding=None,
                            risk_assessment=None,
                            risk_gate_result=None,
                            planner_decision=None,
                        )
                    scope = CollaborationScope(
                        self.engine,
                        session_id,
                        run_id,
                        revision,
                        run["question"],
                        self.store.session(session_id)["root_question"],
                        updates,
                        run["fingerprint"],
                        initial.get("analysis_at"),
                        self._history(session_id, run_id),
                    )
                    candidate: Any = None
                    candidate_report: Any = None
                    try:
                        with collaboration_scope(scope):
                            async with _closing_stream(
                                self.stream_graph(
                                    graph,
                                    None if resume else state,
                                    {
                                        "configurable": {"thread_id": run_id},
                                        "metadata": {
                                            "input_revision": revision,
                                            "session_id": session_id,
                                        },
                                    },
                                )
                            ) as owned_stream_0:
                                async for update in owned_stream_0:
                                    for node, output in update.items():
                                        if node == "__interrupt__":
                                            self.store.status(
                                                run_id, "interrupted", "研究已保存,等待恢复"
                                            )
                                            return
                                        raw = jsonable(output)
                                        if isinstance(raw, dict):
                                            if raw.get("evidence_documents"):
                                                self.store.save_evidence(
                                                    run_id,
                                                    [
                                                        EvidenceDocument.model_validate(d)
                                                        for d in raw.pop("evidence_documents")
                                                    ],
                                                )
                                            if raw.get("final_advice"):
                                                candidate = raw.pop("final_advice")
                                                raw["candidate_advice"] = candidate
                                            if raw.get("final_report"):
                                                candidate_report = raw.pop("final_report")
                                                # Uncommitted report text never enters the public event stream.
                                                raw["report_ready"] = True
                                        stale = not scope.is_current()
                                        if node == "__research_progress__":
                                            self.store.append_event(
                                                run_id,
                                                "research_progress",
                                                {
                                                    **raw,
                                                    "input_revision": revision,
                                                    "superseded": stale,
                                                },
                                            )
                                        else:
                                            self.store.append_event(
                                                run_id,
                                                "node",
                                                {
                                                    "node": node,
                                                    "update": raw,
                                                    "input_revision": revision,
                                                    "superseded": stale,
                                                },
                                            )
                                        self.notify(session_id)
                    except InputSuperseded:
                        if scope.is_current():
                            raise
                        # The old stream is closed before restarting with the same budget.
                        # Fall through to the existing revision loop, never mark interruption.
                    resume = False
                    if not scope.is_current():
                        self.store.append_event(
                            run_id,
                            "research_progress",
                            {
                                "status": "input_updated",
                                "input_revision": self.store.run(run_id)["input_revision"],
                                "superseded_revision": revision,
                                "message": "已保存旧版工作,将按最新补充重新汇总",
                            },
                        )
                        continue
                    await self.engine.incorporate(session_id, run_id, revision)
                    if candidate is None and hasattr(graph, "aget_state"):
                        snapshot = await graph.aget_state({"configurable": {"thread_id": run_id}})
                        candidate = jsonable(snapshot.values.get("final_advice"))
                        candidate_report = candidate_report or jsonable(
                            snapshot.values.get("final_report")
                        )
                    if topic:
                        candidate = None
                        candidate_report = candidate_report or incomplete_report(
                            run["record"], "没有形成经过最终核验的专题结论"
                        )
                    else:
                        candidate = candidate or self._conservative(
                            initial["symbol"], "没有形成经过最终闸门的结论"
                        )
                    if self.store.finalize(run_id, revision, candidate, report=candidate_report):
                        break
        except asyncio.CancelledError:
            self.store.status(
                run_id,
                "cancelled" if run_id in self.stopping else "interrupted",
                "用户停止本轮" if run_id in self.stopping else "后端退出,等待手动恢复",
            )
        except Exception as exc:
            logger.exception("collaborative research interrupted")
            self.store.status(run_id, "interrupted", str(exc))
        finally:
            try:
                if demo_token is not None:
                    from ..demo import DEMO_REQUEST_ID

                    DEMO_REQUEST_ID.reset(demo_token)
            finally:
                self.app.state.active_research_runs.discard(run_id)
                self.tasks.pop(run_id, None)
                self.stopping.discard(run_id)
                await reservation.release()
            if session_id is not None:
                self.notify(session_id)
            if self.store.run(run_id)["status"] != "interrupted" and hasattr(
                self.engine.checkpointer, "unpin_thread"
            ):
                self.engine.checkpointer.unpin_thread(run_id)

    async def subscribe(
        self, session_id: str, after: int = 0, *, until_run: str | None = None
    ) -> AsyncIterator[dict[str, str]]:
        """按持久化事件 seq 从 after 之后补发 SSE;订阅断线只结束等待,不取消后台研究。

        until_run 指定时在该轮终态事件后返回;否则持续订阅至后端关闭。
        """
        self.store.session(session_id)
        signal = self._wake.setdefault(session_id, asyncio.Event())
        try:
            while not self.closing:
                signal.clear()
                events = self.store.events(session_id, after)
                for event in events:
                    after = event["seq"]
                    yield {
                        "id": str(after),
                        "event": event["event"],
                        "data": json.dumps(event["payload"], ensure_ascii=False),
                    }
                    if (
                        until_run
                        and event["run_id"] == until_run
                        and event["event"] in {"final", "failed", "interrupted", "cancelled"}
                    ):
                        return
                if events:
                    continue
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(signal.wait(), timeout=10)
        finally:
            # Signals carry no research state; durable events survive subscriber cancellation.
            pass

    @property
    def pending_shutdown(self) -> set[asyncio.Task[None]]:
        """Tasks still owning resources after the bounded shutdown wait."""
        return {task for task in self._pending_shutdown if not task.done()}

    async def shutdown(self) -> None:
        # The desktop and ASGI lifespan can both request shutdown. Share the
        # operation so a second cancellation cannot interrupt run cleanup.
        """桌面与 ASGI 共享一次关闭任务,避免重复取消破坏落盘;超时仍持有资源的任务由 pending_shutdown 暴露。"""
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self) -> None:
        self.closing = True
        for signal in self._wake.values():
            signal.set()
        tasks = set(self.tasks.values())
        if self._maintenance is not None:
            tasks.add(self._maintenance)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=5)
            self._pending_shutdown = pending
            for task in done:
                if not task.cancelled():
                    error = task.exception()
                    if error is not None:
                        logger.error("后台任务关闭失败: %s", type(error).__name__)
            if pending:
                logger.error("后台关闭等待超过 5 秒,仍有 %d 个任务持有资源", len(pending))
        if not self.pending_shutdown:
            try:
                self.engine.evict(force=True)
            except Exception:
                logger.exception("关闭时清理 agent residency 失败")


def register_session_routes(app: FastAPI) -> None:
    def manager() -> RunManager:
        value = getattr(app.state, "research_manager", None)
        if value is None:
            raise HTTPException(409, "collaboration is not enabled")
        return cast(RunManager, value)

    def translate(exc: Exception) -> HTTPException:
        return HTTPException(404 if isinstance(exc, KeyError) else 409, str(exc))

    @app.post("/research/sessions", status_code=201)
    async def create_session(body: NewSession) -> dict[str, Any]:
        try:
            target = body.target.model_dump(mode="json") if body.target else None
            symbol = body.target.symbol if isinstance(body.target, AssetTarget) else body.symbol
            market = body.target.market if isinstance(body.target, AssetTarget) else body.market
            return manager().store.create_session(
                symbol,
                market,
                body.request_id,
                mode=body.mode,
                target=target,
                question=body.question,
            )
        except SessionConflict as exc:
            raise translate(exc) from exc

    @app.get("/research/sessions")
    async def sessions() -> dict[str, Any]:
        return {"sessions": manager().store.sessions()}

    @app.get("/research/sessions/{session_id}")
    async def session(session_id: str) -> dict[str, Any]:
        try:
            return manager().store.session(session_id)
        except KeyError as exc:
            raise translate(exc) from exc

    @app.post("/research/sessions/{session_id}/turns", status_code=202)
    async def turn(session_id: str, body: NewTurn) -> dict[str, Any]:
        try:
            return await manager().turn(session_id, body.question, body.request_id, body.action)
        except (KeyError, SessionConflict) as exc:
            raise translate(exc) from exc

    @app.post("/research/sessions/{session_id}/messages", status_code=202)
    async def inform(session_id: str, body: Supplement) -> dict[str, Any]:
        try:
            runtime = manager()
            result = runtime.store.inform(session_id, body.run_id, body.content, body.request_id)
            runtime.engine.notify(result["receiver"])
            runtime.notify(session_id)
            return result
        except (KeyError, SessionConflict) as exc:
            raise translate(exc) from exc

    @app.get("/research/sessions/{session_id}/events")
    async def events(
        session_id: str, request: Request, after: int = Query(0, ge=0)
    ) -> EventSourceResponse:
        runtime = manager()
        try:
            runtime.store.session(session_id)
            cursor = max(after, int(request.headers.get("last-event-id", "0")))
            if cursor < 0:
                raise ValueError("negative cursor")
        except KeyError as exc:
            raise translate(exc) from exc
        except ValueError as exc:
            raise HTTPException(422, "invalid event cursor") from exc
        return EventSourceResponse(runtime.subscribe(session_id, cursor))

    @app.post("/runs/{run_id}/resume", status_code=202)
    async def resume_run(run_id: str, body: Command) -> dict[str, Any]:
        try:
            return await manager().control(run_id, "resume", body.request_id)
        except (KeyError, SessionConflict) as exc:
            raise translate(exc) from exc

    @app.post("/runs/{run_id}/stop", status_code=202)
    async def stop_run(run_id: str, body: Command) -> dict[str, Any]:
        try:
            return await manager().control(run_id, "stop", body.request_id)
        except (KeyError, SessionConflict) as exc:
            raise translate(exc) from exc

    @app.post("/research/sessions/{session_id}/close")
    async def close_session(session_id: str, body: Command) -> dict[str, Any]:
        try:
            runtime = manager()
            session = runtime.store.session(session_id)
            result = runtime.store.close_session(session_id, body.request_id)
            for agent in session["agents"]:
                runtime.engine.forget(agent["agent_id"])
                if hasattr(runtime.engine.checkpointer, "unpin_thread"):
                    runtime.engine.checkpointer.unpin_thread(agent["agent_id"])
            return result
        except (KeyError, SessionConflict) as exc:
            raise translate(exc) from exc
