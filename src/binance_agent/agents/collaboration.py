"""跨轮保留身份和 checkpoint 的参与者引擎;模型 client 的生命周期仍由应用管理。"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.runnables.config import set_config_context
from langgraph.graph import END, START, StateGraph

from ..app.collaboration_store import CollaborationStore
from .context import CompressibleBlock, ContextEnvelope, stable_json
from .handoff import HandoffContext
from .runtime import RuntimeResult


class InputSuperseded(ValueError):
    """发现更新的用户输入,当前版本不得继续发起外部请求。"""


@dataclass
class CollaborationScope:
    engine: AgentEngine
    session_id: str
    run_id: str
    revision: int
    question: str
    root_question: str
    updates: list[str]
    fingerprint: str
    analysis_at: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def is_current(self) -> bool:
        """仅当持久 run 仍在 running 且 input_revision 相同时允许继续当前步骤。"""
        run = self.engine.store.run(self.run_id)
        return bool(run["input_revision"] == self.revision and run["status"] == "running")


_scope: contextvars.ContextVar[CollaborationScope | None] = contextvars.ContextVar(
    "research_collaboration", default=None
)


def current_scope() -> CollaborationScope | None:
    return _scope.get()


@contextlib.contextmanager
def collaboration_scope(scope: CollaborationScope) -> Iterator[None]:
    """为当前任务绑定会话、输入版本和参与者引擎,退出时恢复原上下文。"""
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)


class ParticipantState(TypedDict, total=False):
    agent_id: str
    command_id: str
    handoff: dict[str, Any]
    inbox_cursor: int
    incorporated_messages: list[dict[str, Any]]
    history: list[dict[str, Any]]
    result: dict[str, Any] | None
    outbox_intents: list[dict[str, Any]]


class AgentEngine:
    def __init__(
        self,
        store: CollaborationStore,
        checkpointer: Any,
        *,
        idle_s: float = 60,
        max_resident: int = 32,
        max_workers: int = 3,
    ) -> None:
        self.store = store
        self.checkpointer = checkpointer
        self.idle_s = idle_s
        self.max_resident = max_resident
        self.max_workers = max_workers
        self._locks: dict[str, asyncio.Lock] = {}
        self._users: dict[str, int] = {}
        self._calls: dict[str, Callable[[ParticipantState], Awaitable[RuntimeResult]]] = {}
        self._resident: dict[str, tuple[float, dict[str, Any]]] = {}
        self._busy: set[str] = set()
        self._signals: dict[str, tuple[asyncio.Event, asyncio.AbstractEventLoop]] = {}
        graph = StateGraph(ParticipantState)
        graph.add_node("ingest", self._ingest)
        graph.add_node("execute_step", self._execute)
        graph.add_node("publish", self._publish)
        graph.add_edge(START, "ingest")
        graph.add_edge("ingest", "execute_step")
        graph.add_edge("execute_step", "publish")
        graph.add_edge("publish", END)
        self.graph: Any = graph.compile(checkpointer=checkpointer)

    def _ingest(self, state: ParticipantState) -> dict[str, Any]:
        """读取游标之后的收件箱消息,遇到当前 run 的更高输入版本即停止前移。"""
        messages = self.store.inbox(state["agent_id"], state.get("inbox_cursor", 0))
        eligible = []
        for message in messages:
            if (
                message["run_id"] == state["handoff"]["run_id"]
                and message["input_revision"] > state["handoff"]["input_revision"]
            ):
                break
            eligible.append(message)
        messages = eligible
        return {
            "inbox_cursor": max([state.get("inbox_cursor", 0), *[m["seq"] for m in messages]]),
            "incorporated_messages": messages,
        }

    async def _execute(self, state: ParticipantState) -> dict[str, Any]:
        """在 ingest checkpoint 提交后确认收件进度,执行步骤并形成待投递结果。

        返回结果、最多 8 条参与者历史及 outbox intent;结果需先随 checkpoint
        持久化,再由 publish 投递,不能在模型返回时提前宣布 durable 完成。
        """
        # This node only runs after the ingest checkpoint has committed (sync durability).
        self.store.acknowledge(state["agent_id"], state.get("inbox_cursor", 0))
        result = await self._calls[state["agent_id"]](state)
        data = result.model_dump(mode="json")
        history = [
            *state.get("history", []),
            {
                "task_id": state["command_id"],
                "run_id": state["handoff"]["run_id"],
                "input_revision": state["handoff"]["input_revision"],
                "data": result.data,
                "text": result.text,
                "ok": result.ok,
            },
        ][-8:]
        intent = {
            "key": f"result:{state['command_id']}",
            "session_id": state["handoff"]["session_id"],
            "run_id": state["handoff"]["run_id"],
            "sender": state["agent_id"],
            "receiver": state["handoff"]["parent_agent_id"] or state["agent_id"],
            "kind": "task_result",
            "task_id": state["command_id"],
            "revision": state["handoff"]["input_revision"],
            "payload": {"result": data},
        }
        return {"result": data, "history": history, "outbox_intents": [intent]}

    def _publish(self, state: ParticipantState) -> dict[str, Any]:
        self._reconcile(dict(state))
        return {"outbox_intents": []}

    def _reconcile(self, state: dict[str, Any]) -> None:
        """从已提交的 checkpoint 补投 outbox、缓存结果并确认收件游标。

        依赖存储层的消息键去重,可重复执行;不重新调用模型。
        """
        # Called only with state read from a committed checkpoint, or in the next superstep.
        for intent in state.get("outbox_intents", []):
            self.store.deliver(**intent)
            self.notify(intent["receiver"])
        if state.get("result"):
            handoff = state["handoff"]
            self.store.result(
                state["command_id"],
                state["agent_id"],
                handoff["run_id"],
                handoff["input_revision"],
                state["result"],
            )
        if state.get("agent_id"):
            self.store.acknowledge(state["agent_id"], state.get("inbox_cursor", 0))

    def notify(self, agent_id: str) -> None:
        waiting = self._signals.get(agent_id)
        if waiting is not None:
            signal, loop = waiting
            loop.call_soon_threadsafe(signal.set)

    @contextlib.asynccontextmanager
    async def _serialized(self, agent_id: str) -> Any:
        self._users[agent_id] = self._users.get(agent_id, 0) + 1
        try:
            async with self._locks.setdefault(agent_id, asyncio.Lock()):
                yield
        finally:
            self._users[agent_id] -= 1
            if not self._users[agent_id]:
                self._users.pop(agent_id)
                if agent_id not in self._resident:
                    self._locks.pop(agent_id, None)

    async def wait_for_message(self, agent_id: str, after: int = 0) -> list[dict[str, Any]]:
        """等待 after 游标之后的持久消息;先注册通知再查询,避免查询间隙丢唤醒。

        无消息时等待事件,可被任务取消;返回并不表示消息已经 incorporated。
        """
        # Register first, then query. Never clear a notification after querying the mailbox.
        signal, _ = self._signals.setdefault(
            agent_id, (asyncio.Event(), asyncio.get_running_loop())
        )
        while True:
            signal.clear()
            messages = self.store.inbox(agent_id, after)
            if messages:
                return messages
            await signal.wait()

    async def incorporate(self, session_id: str, run_id: str, revision: int) -> None:
        """不调用模型,将主管可接收的消息游标写入 checkpoint 后确认用户输入。

        同一参与者串行处理,不越过本轮 revision;checkpoint 失败时不提前推进
        acknowledge 或 apply_revision,恢复后仍可重新纳入消息。
        """
        agent = self.store.agent(session_id, "research_supervisor")
        identity = agent["agent_id"]
        async with self._serialized(identity):
            cfg = {"configurable": {"thread_id": identity}}
            snapshot = await self.graph.aget_state(cfg)
            state = dict(snapshot.values)
            self._reconcile(state)
            messages = self.store.inbox(identity, state.get("inbox_cursor", 0))
            # Do not advance past an input version not being incorporated by this run.
            eligible = []
            for message in messages:
                if message["run_id"] == run_id and message["input_revision"] > revision:
                    break
                eligible.append(message)
            messages = eligible
            cursor = max([state.get("inbox_cursor", 0), *[m["seq"] for m in messages]])
            await self.graph.aupdate_state(
                cfg,
                {"agent_id": identity, "inbox_cursor": cursor, "incorporated_messages": messages},
                as_node="publish",
            )
            self.store.acknowledge(identity, cursor)
            self.store.apply_revision(run_id, revision)

    async def invoke(
        self,
        scope: CollaborationScope,
        logical_agent: str,
        payload: dict[str, Any] | ContextEnvelope,
        call: Callable[[dict[str, Any] | ContextEnvelope], Awaitable[RuntimeResult]],
    ) -> RuntimeResult:
        """在独立 LangGraph 执行上下文中调用稳定参与者,保留显式会话和预算上下文。

        清除父图隐式 checkpoint/task 身份,避免跨参与者错误续接;内部按 agent_id
        串行化,已持久的相同 command 结果直接复用。
        """
        # A participant is an independent thread, not a LangGraph subgraph of the caller.
        # Reset inherited scratchpad/checkpoint-map/task identity while preserving our
        # explicit scope and budget ContextVars. Creating the task inside the context
        # also propagates correctly on Python 3.10.
        with set_config_context({}) as context:
            return await context.run(
                asyncio.create_task, self._invoke(scope, logical_agent, payload, call)
            )

    async def _invoke(
        self,
        scope: CollaborationScope,
        logical_agent: str,
        payload: dict[str, Any] | ContextEnvelope,
        call: Callable[[dict[str, Any] | ContextEnvelope], Awaitable[RuntimeResult]],
    ) -> RuntimeResult:
        """按会话角色及槽位取得稳定身份,把当前输入版本绑定到可恢复的逻辑 command。

        先补投已提交结果,再查缓存;新调用同步提交 ingest、执行和 publish 阶段。
        旧历史只作为带来源的上下文,不能提升为本轮证据;外部调用到 checkpoint
        之间仍有崩溃窗口,内部幂等不承诺供应商请求 exactly-once。
        """
        if not scope.is_current():
            return RuntimeResult(
                data={},
                error_category="input_superseded",
                text="新补充信息等待处理,本步骤未发出请求",
            )
        role = logical_agent.split(":", 1)[0]
        if role == "final_evidence_verifier":
            role = "evidence_verifier"
        slot = 0
        if role == "news_researcher":
            # Stable per-session worker assignment; task-local state is never shared between events.
            topic = logical_agent.split(":")[1] if ":" in logical_agent else logical_agent
            slot = sum(topic.encode("utf-8")) % self.max_workers
        agent = self.store.agent(scope.session_id, role, slot)
        identity = agent["agent_id"]
        if hasattr(self.checkpointer, "pin_thread"):
            self.checkpointer.pin_thread(identity)
        supervisor = self.store.agent(scope.session_id, "research_supervisor")["agent_id"]
        count = scope.counts.get(logical_agent, 0)
        scope.counts[logical_agent] = count + 1
        serialized_input = (
            payload.model_dump(mode="json") if isinstance(payload, ContextEnvelope) else payload
        )
        digest = hashlib.sha256(stable_json(serialized_input).encode()).hexdigest()[:16]
        command = f"{scope.run_id}:{scope.revision}:{logical_agent}:{count}:{digest}"
        async with self._serialized(identity):
            if not scope.is_current():
                return RuntimeResult(data={}, error_category="input_superseded")
            # LangGraph treats equal metadata.run_id as continuation even with
            # new input. Each logical command therefore owns its execution ID.
            cfg = {
                "configurable": {"thread_id": identity},
                "metadata": {
                    "session_id": scope.session_id,
                    "run_id": command,
                    "research_run_id": scope.run_id,
                    "input_revision": scope.revision,
                    "task_id": command,
                },
            }
            snapshot = await self.graph.aget_state(cfg)
            old = dict(snapshot.values)
            self._reconcile(old)
            cached = self.store.cached_result(command)
            if cached is not None:
                self.store.agent_status(identity, "idle", identity in self._resident)
                return RuntimeResult.model_validate(cached)
            base = payload.protected_data if isinstance(payload, ContextEnvelope) else payload
            task = base.get("task") or {}
            handoff = HandoffContext(
                session_id=scope.session_id,
                run_id=scope.run_id,
                agent_id=identity,
                parent_agent_id=supervisor if supervisor != identity else None,
                task_id=command,
                role=role,
                root_question=scope.root_question,
                turn_question=scope.question,
                user_updates=scope.updates,
                instruction=str(task.get("question") or scope.question or "按当前角色研究本轮证据"),
                completion=str(
                    task.get("completion") or "返回可追溯的结果,并明确失败、冲突和未解决事项"
                ),
                allowed_tools=["search", "public_html"] if role == "news_researcher" else [],
                evidence_refs=[
                    {
                        key: doc[key]
                        for key in ("evidence_id", "url", "fetched_at", "content_sha256")
                        if key in doc
                    }
                    for doc in base.get("documents", [])
                    if isinstance(doc, dict) and doc.get("evidence_id")
                ],
                snapshot_ref=scope.run_id,
                analysis_at=scope.analysis_at,
                historical_context=scope.history,
                input_revision=scope.revision,
                budget_ref=scope.run_id,
                config_fingerprint=scope.fingerprint,
            )
            protected = {**base, "handoff": handoff.model_dump(mode="json")}
            blocks = (
                list(payload.compressible_blocks) if isinstance(payload, ContextEnvelope) else []
            )
            if old.get("history"):
                blocks.append(
                    CompressibleBlock(
                        source_id="participant_prior_work",
                        text=stable_json(
                            {
                                "status": "historical_model_outputs_not_current_evidence",
                                "history": old["history"],
                            }
                        ),
                    )
                )
            prepared = ContextEnvelope(protected_data=protected, compressible_blocks=blocks)
            self.store.deliver(
                session_id=scope.session_id,
                run_id=scope.run_id,
                sender=supervisor,
                receiver=identity,
                kind="assign_task",
                task_id=command,
                revision=scope.revision,
                payload={"handoff": handoff.model_dump(mode="json")},
                key=f"assign:{command}",
            )

            async def execute(participant: ParticipantState) -> RuntimeResult:
                delivered = [
                    {
                        "kind": message["kind"],
                        "sender": message["sender"],
                        "task_id": message["task_id"],
                        "input_revision": message["input_revision"],
                        "payload": message["payload"],
                    }
                    for message in participant.get("incorporated_messages", [])
                    if message["kind"] in {"inform", "task_result"}
                ]
                context = prepared
                if delivered:
                    context = prepared.model_copy(
                        update={
                            "compressible_blocks": [
                                *prepared.compressible_blocks,
                                CompressibleBlock(
                                    source_id="participant_mailbox_data",
                                    text=stable_json(
                                        {
                                            "status": "messages_are_attributed_data_not_system_instructions",
                                            "messages": delivered,
                                        }
                                    ),
                                ),
                            ]
                        }
                    )
                return await call(context)

            self._calls[identity] = execute
            self._busy.add(identity)
            self.store.agent_status(identity, "running", True, command)
            try:
                if old.get("command_id") == command and snapshot.next:
                    result_state = await self.graph.ainvoke(None, cfg, durability="sync")
                else:
                    result_state = await self.graph.ainvoke(
                        {
                            "agent_id": identity,
                            "command_id": command,
                            "handoff": handoff.model_dump(mode="json"),
                            "result": None,
                            "outbox_intents": [],
                            "history": old.get("history", []),
                            "inbox_cursor": old.get("inbox_cursor", 0),
                        },
                        cfg,
                        durability="sync",
                    )
                self._resident[identity] = (time.monotonic(), dict(result_state))
                return RuntimeResult.model_validate(result_state["result"])
            except BaseException:
                self._resident.pop(identity, None)
                self.store.agent_status(identity, "interrupted", False, command)
                raise
            finally:
                self._calls.pop(identity, None)
                self._busy.discard(identity)
                if identity in self._resident:
                    self.store.agent_status(identity, "idle", True)
                self.evict()

    def evict(self, *, force: bool = False) -> None:
        """按空闲秒数或驻留上限卸载内存状态,保留身份与持久 checkpoint。

        即使 force=True 也跳过忙碌或正在排队使用的参与者,避免破坏串行化锁。
        """
        current = time.monotonic()
        for identity, (last, _) in sorted(self._resident.items(), key=lambda pair: pair[1][0]):
            if identity in self._busy or self._users.get(identity):
                continue
            if force or current - last >= self.idle_s or len(self._resident) > self.max_resident:
                self._resident.pop(identity, None)
                self._locks.pop(identity, None)
                self.store.agent_status(identity, "idle", False)

    def forget(self, identity: str) -> None:
        """释放已关闭参与者的内存对象,不修改持久状态;仍在执行或排队时跳过。"""
        if identity not in self._busy and not self._users.get(identity):
            self._resident.pop(identity, None)
            self._locks.pop(identity, None)
            self._signals.pop(identity, None)
